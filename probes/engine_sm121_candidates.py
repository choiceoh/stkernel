"""Kernels the seed image already carries for engine/SM121_INTAKE.md, judged on one GB10 against what the engine serves
(probe, single-GPU lane).

    gdn     U13: Qwen3.8's GDN chunked prefill on FlashInfer's chunk_gated_delta_rule (its SM120 CuTe-DSL path, in the
            image: measurements/sm121_inventory_20260919) against the served kda/chunk_decay.chunk_kda_with_decay, at the
            rank's 4 key / 12 value heads x 128 cell with a carried state (probes/engine_qwen38_kda.prefill_args). Both
            are held to engine/modules/linear_attention.gated_delta_rule (the fp32 recurrence, value heads widened the
            way lanes.value_heads widens them) at the lengths the recurrence can walk, and to each other at every length;
            then timed eager, FlashInfer with and without the copies that make q/k/v contiguous (the served kernel reads
            views of the conv output)
    fp8_l2  U11: the FP8 prefill GEMM's throughput as M grows, at weights past the GB10's 24 MiB L2 and one inside it --
            vllm#55180 found CUTLASS's blockwise FP8 GEMM on SM 12.x falling from 165 to 52 TFLOPS once M spans enough
            tiles that its raster re-streams the weight; here deep_gemm (FP8Linear without a reader) and the cuBLASLt
            reader (FP8Linear.prepare_cublas, the served prefill reader) at M 1,024..16,384

    python3 probes/engine_kernel_check.py --lanes sm121_gdn --output /cache/sm121-gdn.json
    python3 probes/engine_kernel_check.py --lanes sm121_fp8_l2 --output /cache/sm121-fp8-l2.json

Numbers, not a verdict: a kernel that wins here is bound by a pull request that says so, and a speed claim on the
served engine is the fleet's (D17). An arm that fails records its error and the others still run.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GDN_TOKENS = (128, 1024, 8192)
ORACLE_TOKENS = (128, 1024)          # the fp32 recurrence walks one token a step; 8192 compares the kernels
WARMUP, REPEATS = 3, 9
MEMORY_CAP_GIB = 6

# (N, K) of an FP8 weight, and why it is here. 24 MiB is the GB10's L2.
L2_SHAPES = {
    "vllm#55180 16384x2560 (42 MB)": (16384, 2560),
    "vllm#55180 5120x5120 (26 MB)": (5120, 5120),
    "glm53 dense gate_up 24576x4096 (101 MB)": (24576, 4096),
    "glm53 dense down 4096x12288 (50 MB)": (4096, 12288),
    "inside the L2 4096x4096 (17 MB)": (4096, 4096),
}
L2_ROWS = (1024, 4096, 8192, 16384)
L2_ROUNDS = 9                        # arms alternate inside a round: production's steps land on both (0919b was one-sided)
L2_MAX_BYTES = 1 << 30               # activations and output of one call, whichever is larger


def _error(got, want) -> float:
    got, want = got.float(), want.float()
    return round(float((got - want).abs().max() / want.abs().max().clamp_min(1e-30)), 6)


def _time(fn, repeats=REPEATS, warmup=WARMUP) -> dict:
    import torch
    for _ in range(warmup):
        fn()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {"median_us": round(statistics.median(samples), 1), "best_us": round(min(samples), 1)}


def _write(output, report):
    text = json.dumps(report, indent=1, default=str)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
    return text


def _device(report, cap_gib):
    import torch
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gib * 2**30 / props.total_memory))
    report["device"] = dict(name=props.name, capability=list(torch.cuda.get_device_capability()),
                            torch=torch.__version__, cuda=torch.version.cuda, memory_cap_gib=cap_gib)


# -- U13: GDN prefill ------------------------------------------------------------------------------------------------------
def flashinfer_gdn_args(args) -> dict:
    """The served call's inputs as FlashInfer's chunk_gated_delta_rule takes them (sm121-batchA-0919b: the image's
    build refuses int32 cu_seqlens): [T, H, D] contiguous q/k/v, the
    forget gate as alpha = exp(log decay) and beta as probabilities, both fp32 [T, HV]; the carried state is already the
    kernel layout [1, HV, V, K] both kernels share."""
    import torch
    t = args["q"].shape[1]
    initial = args["initial_state"].float().contiguous()
    return dict(q=args["q"][0].contiguous(), k=args["k"][0].contiguous(), v=args["v"][0].contiguous(),
                g=args["decay"][0].exp().contiguous(), beta=args["beta"][0].float().contiguous(),
                scale=args["scale"], initial_state=initial, output_final_state=True,
                cu_seqlens=torch.tensor([0, t], dtype=torch.int64, device=initial.device),   # the image's build: int64
                use_qk_l2norm_in_kernel=True, output_state=torch.empty_like(initial))


def gdn_oracle(args):
    """engine/modules/linear_attention.gated_delta_rule on the served inputs: (o [T, HV, V], state [HV, V, K])."""
    from engine.modules.linear_attention import gated_delta_rule
    q, k, v = args["q"], args["k"], args["v"]
    g = v.shape[2] // q.shape[2]
    q, k = q.repeat_interleave(g, dim=2), k.repeat_interleave(g, dim=2)          # lanes.value_heads
    o, state = gated_delta_rule(q, k, v, args["decay"], args["beta"], args["initial_state"].transpose(-1, -2),
                                scale=args["scale"], qk_l2norm=True, decay_per_channel=False)
    return o[0], state[0].transpose(-1, -2)


def run_gdn(output=None) -> dict:
    import torch
    from engine.base import kernel_shape as ks
    from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
    from engine.profiles.qwen38 import shapes
    from probes.engine_qwen38_kda import QWEN38, prefill_args
    config = ROOT / "probes" / "qwen38_config.json"
    ks.bind(shapes.kernel_shape(json.loads(config.read_text())["text_config"]))      # the GDN entries refuse a KDA cell
    report = {"lane": "sm121_gdn", "cell": [QWEN38.k_heads, QWEN38.v_heads, QWEN38.dim], "unavailable": {}, "rows": {}}
    _device(report, MEMORY_CAP_GIB)
    try:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
    except Exception as exc:                                        # noqa: BLE001 -- the import's failure is the answer
        report["unavailable"]["flashinfer"] = f"{type(exc).__name__}: {exc}"[:300]
        chunk_gated_delta_rule = None
    with torch.inference_mode():
        for t in GDN_TOKENS:
            args = prefill_args(QWEN38, t, torch.device("cuda"))
            row = {}
            o_s, st_s = chunk_kda_with_decay(**args)
            o_s, st_s = o_s[0], st_s[0]
            row["served"] = _time(lambda: chunk_kda_with_decay(**args))
            o_f = st_f = None
            if chunk_gated_delta_rule is not None:
                fa = flashinfer_gdn_args(args)
                try:
                    began = time.perf_counter()
                    o_f, st_f = chunk_gated_delta_rule(**fa)
                    torch.cuda.synchronize()
                    row["flashinfer_first_call_s"] = round(time.perf_counter() - began, 2)
                    row["flashinfer"] = _time(lambda: chunk_gated_delta_rule(**fa))
                    row["flashinfer_with_copies"] = _time(lambda: chunk_gated_delta_rule(**flashinfer_gdn_args(args)))
                    st_f = st_f[0]
                    row["flashinfer_vs_served"] = {"o": _error(o_f, o_s), "state": _error(st_f, st_s)}
                except Exception as exc:                            # noqa: BLE001
                    report["unavailable"][f"flashinfer {t}"] = f"{type(exc).__name__}: {exc}"[:300]
                    o_f = st_f = None
            if t in ORACLE_TOKENS:
                o_r, st_r = gdn_oracle(args)
                row["served_vs_oracle"] = {"o": _error(o_s, o_r), "state": _error(st_s, st_r)}
                if o_f is not None:
                    row["flashinfer_vs_oracle"] = {"o": _error(o_f, o_r), "state": _error(st_f, st_r)}
            if "flashinfer" in row:
                row["speedup"] = round(row["served"]["median_us"] / row["flashinfer"]["median_us"], 3)
            report["rows"][t] = row
            print(json.dumps({f"gdn {t}": row}), flush=True)
            del args
            torch.cuda.empty_cache()
    print(_write(output, report), flush=True)
    return report


def run_gdn_diag(output=None, tokens=1024) -> dict:
    """sm121-batchC-0919c: FlashInfer's GDN prefill ran 1.2-6.9x faster than the served kernel but returned NaN. Which of
    its inputs this probe hands it differently from what it expects -- the value heads grouped over fewer query heads
    (4 / 12) against q and k widened to 12, the forget gate as alpha against log decay, the carried state against a
    zero one, and the carried state's last two axes swapped -- each against the fp32 recurrence, with NaN counts."""
    import itertools
    import torch
    from engine.base import kernel_shape as ks
    from engine.profiles.qwen38 import shapes
    from probes.engine_qwen38_kda import QWEN38, prefill_args
    config = ROOT / "probes" / "qwen38_config.json"
    ks.bind(shapes.kernel_shape(json.loads(config.read_text())["text_config"]))
    report = {"lane": "sm121_gdn_diag", "tokens": tokens, "unavailable": {}, "variants": {}}
    _device(report, MEMORY_CAP_GIB)
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    with torch.inference_mode():
        args = prefill_args(QWEN38, tokens, torch.device("cuda"))
        zero = dict(args, initial_state=torch.zeros_like(args["initial_state"]))
        oracles = {"carried": gdn_oracle(args), "zero": gdn_oracle(zero)}
        for heads, gate, state in itertools.product(("gva 4/12", "widened 12/12"), ("alpha", "log"),
                                                    ("carried", "zero", "carried swapped")):
            name = f"{heads} | gate {gate} | state {state}"
            base = zero if state == "zero" else args
            fa = flashinfer_gdn_args(base)
            if heads.startswith("widened"):
                g = fa["v"].shape[1] // fa["q"].shape[1]
                fa["q"], fa["k"] = (fa[x].repeat_interleave(g, dim=1).contiguous() for x in ("q", "k"))
            if gate == "log":
                fa["g"] = base["decay"][0].float().contiguous()
            if state == "carried swapped":
                fa["initial_state"] = fa["initial_state"].transpose(-1, -2).contiguous()
            fa["output_state"] = torch.empty_like(fa["initial_state"])
            try:
                o, st = chunk_gated_delta_rule(**fa)
                torch.cuda.synchronize()
                o_r, st_r = oracles["zero" if state == "zero" else "carried"]
                st = st[0]
                report["variants"][name] = {
                    "o_nan": int(torch.isnan(o).sum()), "o_inf": int(torch.isinf(o).sum()), "o_numel": o.numel(),
                    "state_nan": int(torch.isnan(st).sum()),
                    "o_vs_oracle": _error(torch.nan_to_num(o), o_r), "state_vs_oracle": _error(torch.nan_to_num(st), st_r),
                    "state_vs_oracle_swapped": _error(torch.nan_to_num(st), st_r.transpose(-1, -2)),
                    "nan_heads": sorted({int(h) for h in torch.isnan(o).any(-1).nonzero()[:, 1].tolist()})[:16]}
            except Exception as exc:                                    # noqa: BLE001
                report["variants"][name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            print(json.dumps({name: report["variants"][name]}), flush=True)
    print(_write(output, report), flush=True)
    return report


# -- U11: the FP8 prefill GEMM past the L2 ------------------------------------------------------------------------------
def run_fp8_l2(output=None) -> dict:
    import torch
    from engine.kernels.dense import FP8Linear
    report = {"lane": "sm121_fp8_l2", "unavailable": {}, "shapes": {}}
    _device(report, MEMORY_CAP_GIB)
    torch.manual_seed(0)
    with torch.inference_mode():
        for label, (n, k) in L2_SHAPES.items():
            w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
            deep = FP8Linear(w)
            served = FP8Linear(w, quantized=deep.weight)
            try:
                served.prepare_cublas(split_decode=False)
            except Exception as exc:                                # noqa: BLE001
                report["unavailable"][f"cublaslt {label}"] = f"{type(exc).__name__}: {exc}"[:300]
                served = None
            rows = {}
            for m in L2_ROWS:
                if m * max(n, k) * 2 > L2_MAX_BYTES:
                    continue
                x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
                flops = 2 * m * n * k
                arms = {arm: layer for arm, layer in (("deep_gemm", deep), ("cublaslt (served)", served)) if layer}
                samples = {arm: [] for arm in arms}
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                try:
                    for layer in arms.values():                    # compile, tune, first touch
                        layer(x)
                        layer(x)
                    for r in range(L2_ROUNDS):
                        for arm in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
                            torch.cuda.synchronize()
                            start.record()
                            arms[arm](x)
                            end.record()
                            end.synchronize()
                            samples[arm].append(start.elapsed_time(end) * 1000)
                except Exception as exc:                            # noqa: BLE001
                    report["unavailable"][f"{label} M{m}"] = f"{type(exc).__name__}: {exc}"[:300]
                row = {}
                for arm, got in samples.items():
                    if got:
                        med = statistics.median(got)
                        row[arm] = {"median_us": round(med, 1), "best_us": round(min(got), 1),
                                    "spread": round(max(got) / min(got), 2), "TFLOPS": round(flops / med / 1e6, 1),
                                    "best_TFLOPS": round(flops / min(got) / 1e6, 1)}
                rows[m] = row
                print(json.dumps({f"{label} M{m}": row}), flush=True)
                del x
            report["shapes"][label] = {"n": n, "k": k, "weight_MB": round(n * k / 1e6, 1), "rows": rows}
            del w, deep, served
            torch.cuda.empty_cache()
    print(_write(output, report), flush=True)
    return report


if __name__ == "__main__":
    {"gdn": run_gdn, "fp8_l2": run_fp8_l2}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else None)
