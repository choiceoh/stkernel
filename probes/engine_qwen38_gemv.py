"""engine/kernels/common/skinny_gemv against cuBLAS at the decode step's own shapes, on one GB10 (probe, single-GPU lane).

The decode step census (probes/engine_qwen38_step.py, 2026-09-19) put 11.2 ms of a C=1 step's 20.2 in BF16 GEMMs of 2 to
8 rows: the hyper-connection mixers' two a site (cuBLAS picks sm80 WMMA kernels, about 201 GB/s -- 6.2 ms), the router
(cuBLAS gemv, about 105 GB/s -- 1.2 ms). Every one of them is a weight read once for a handful of rows, so its floor is
its bytes over the memory's bandwidth (273 GB/s). This times the kernel's configurations (BLOCK_N, BLOCK_K, SPLIT, warps,
stages) against torch.mm at 1, 4, 8 and 16 rows -- a draft step at C=1, a K=3 verify at C=1, 2 and 4 -- both inside CUDA
graphs, interleaved A/B so production's own steps on the same GPU land on every arm, with the weights rotated over more
copies than the L2 holds (a step reads a mixer's weights once and 1,600 other launches between). The verdict a shape
reads is the configuration fastest summed over the rows, and its speedup at each row count.

    python3 probes/engine_kernel_check.py --lanes qwen38_gemv --output /cache/qwen38-gemv.json
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

ROWS = (1, 4, 8, 16)
COPIES_BYTES = 64 << 20        # weights rotated over at least this many bytes: more than GB10's L2
CALLS = 16                     # calls a graph (one per copy in turn)
ROUNDS = 9

# (outputs, K) of W [outputs, K] -> the configurations tried: (BLOCK_N, BLOCK_K, SPLIT, warps, stages)
SHAPES = {
    "router": ((513, 2560), ((32, 256, 1, 4, 3), (16, 256, 1, 4, 3), (16, 512, 1, 4, 2), (32, 256, 2, 4, 3),
                             (16, 256, 2, 4, 3), (16, 256, 5, 4, 2), (32, 256, 5, 4, 2), (16, 256, 10, 4, 1),
                             (32, 128, 4, 4, 3), (64, 256, 5, 4, 2), (16, 256, 5, 2, 2), (32, 256, 5, 8, 2))),
    "hc down": ((324, 10240), ((16, 256, 8, 4, 3), (16, 256, 10, 4, 3), (16, 256, 20, 4, 2), (16, 256, 40, 4, 1),
                               (32, 256, 10, 4, 3), (32, 256, 20, 4, 2), (16, 512, 10, 4, 2), (32, 128, 20, 4, 3),
                               (64, 256, 20, 4, 2), (16, 256, 10, 2, 3), (32, 256, 10, 8, 3), (16, 256, 4, 4, 3))),
    "hc up": ((10240, 320), ((64, 128, 1, 4, 3), (32, 64, 1, 4, 3), (64, 64, 1, 4, 3), (128, 64, 1, 4, 3),
                             (16, 512, 1, 4, 1), (32, 512, 1, 4, 1), (64, 512, 1, 4, 1), (128, 512, 1, 8, 1),
                             (32, 64, 1, 2, 3), (16, 64, 1, 2, 3), (64, 64, 1, 8, 3), (32, 128, 1, 4, 2))),
}


def run(output=None) -> dict:
    import torch
    from engine.kernels.common import skinny_gemv
    torch.manual_seed(0)
    skinny_gemv.prepare("cuda")
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS, "shapes": {}}
    for label, ((n, k), configs) in SHAPES.items():
        nbytes = n * k * 2
        copies = max(CALLS, -(-COPIES_BYTES // nbytes))
        weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        shape = {"weight_MB": round(nbytes / 1e6, 2), "rows": {}}
        for m in ROWS:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = (x.float() @ weights[0].float().t())
            keep = []                                     # a graph writes its outputs' addresses: they live with it

            def graph_of(fn):
                outs = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(CALLS)]
                keep.append(outs)
                for i in range(CALLS):
                    fn(weights[i % copies], outs[i])
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(CALLS):
                        fn(weights[(i * 7) % copies], outs[i])
                return g

            arms = {"cublas": graph_of(lambda w, o: torch.mm(x, w.t(), out=o))}
            errors = {}
            for config in configs:
                got = skinny_gemv.gemv(x, weights[0], config).float()
                again = skinny_gemv.gemv(x, weights[0], config).float()
                errors[str(config)] = (round(float((got - ref).abs().max() / ref.abs().max()), 6),
                                       bool(torch.equal(got, again)))
                arms[str(config)] = graph_of(lambda w, o, config=config: skinny_gemv.gemv(x, w, config, out=o))
            times = {name: [] for name in arms}
            for _ in range(ROUNDS):
                for name, g in arms.items():              # interleaved: contention lands on every arm alike
                    g.replay()
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    times[name].append((time.perf_counter() - began) / CALLS * 1e6)
            del arms, keep
            row = {}
            for name, samples in times.items():
                us = statistics.median(samples)
                row[name] = {"us": round(us, 2), "GBps": round(nbytes / us / 1e3, 1)}
                if name in errors:
                    row[name].update(rel_err=errors[name][0], repeatable=errors[name][1])
            shape["rows"][m] = row
            best = min((c for c in row if c != "cublas"), key=lambda c: row[c]["us"])
            print(json.dumps({f"{label} rows {m}": {"cublas": row["cublas"], "best": best, **row[best],
                                                     "speedup": round(row["cublas"]["us"] / row[best]["us"], 3)}}),
                  flush=True)
        names = [str(c) for c in configs]
        chosen = min(names, key=lambda c: sum(shape["rows"][m][c]["us"] for m in ROWS))
        shape["chosen"] = chosen
        shape["speedup"] = {m: round(shape["rows"][m]["cublas"]["us"] / shape["rows"][m][chosen]["us"], 3) for m in ROWS}
        shape["module_config"] = str(skinny_gemv.CONFIGS.get((n, k)))
        print(json.dumps({label: {"chosen": chosen, "speedup": shape["speedup"],
                                  "module_config": shape["module_config"]}}), flush=True)
        report["shapes"][label] = shape
        del weights
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


HC, HIDDEN, RANK = 4, 2560, 320                     # Qwen3.8's streams, hidden width and mixer rank
DOWN_CFG = (16, 256, 4, 4, 3)                       # skinny_gemv.CONFIGS' down projection
UP_CFGS = ((16, 64, 4, 3), (32, 64, 4, 3), (16, 128, 4, 3), (16, 64, 2, 3), (32, 128, 4, 2))   # BLOCK_D, BLOCK_K, warps, stages


try:                                                # the kernels are module globals: a jit function finds
    import triton                                   # the functions it calls among its module's names
    import triton.language as tl
except ImportError:                                 # the probe's CPU tests read its tables without triton
    triton = None

if triton is not None:
    # Carry H1 + H2 as two Triton launches over the skinny GEMV: the down(+inject) product with the gates in the
    # last program's store, and the up product with the streams' mean in its store -- the lane's `_gates` and
    # `_mix_mean` arithmetic on the product rounded to BF16 exactly where the lane rounds it.

    @triton.jit
    def _gate_store(total, rows, cols, M, MIX, INJ, sm, si, HC_F, R: tl.constexpr, HCN: tl.constexpr,
                   WITH_INJECT: tl.constexpr):
        di = total.to(tl.bfloat16)
        q = (di.to(tl.float32) / HC_F).to(tl.bfloat16).to(tl.float32)
        live = rows[:, None] < M
        tl.store(MIX + rows[:, None] * sm + cols[None, :], (q * tl.sigmoid(q)).to(tl.bfloat16),
                 mask=live & (cols[None, :] < R))
        if WITH_INJECT:
            g = tl.sigmoid(q).to(tl.bfloat16).to(tl.float32)
            tl.store(INJ + rows[:, None] * si + (cols - R)[None, :], (2.0 * g).to(tl.bfloat16),
                     mask=live & (cols[None, :] >= R) & (cols[None, :] < R + HCN))

    @triton.jit
    def _down_gates(X, W, MIX, INJ, PART, LOCKS, M, N, K, sx, sw, sm, si, HC_F, R: tl.constexpr, HCN: tl.constexpr,
                   WITH_INJECT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT: tl.constexpr,
                   FP32_DOT: tl.constexpr):
        pid_n, pid_k = tl.program_id(0), tl.program_id(1)
        rows = tl.arange(0, 16)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        span = tl.cdiv(tl.cdiv(K, SPLIT), BLOCK_K) * BLOCK_K
        k0 = pid_k * span
        k1 = tl.minimum(k0 + span, K)
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for k in range(k0, k1, BLOCK_K):
            ks = k + tl.arange(0, BLOCK_K)
            kmask = ks < k1
            x = tl.load(X + rows[:, None] * sx + ks[None, :], mask=(rows[:, None] < M) & kmask[None, :], other=0.0)
            w = tl.load(W + cols[:, None] * sw + ks[None, :], mask=(cols[:, None] < N) & kmask[None, :], other=0.0)
            if FP32_DOT:                                  # the interpreter reads a BF16 dot's bits as integers
                x, w = x.to(tl.float32), w.to(tl.float32)
            acc += tl.dot(x, tl.trans(w))
        if SPLIT == 1:
            _gate_store(acc, rows, cols, M, MIX, INJ, sm, si, HC_F, R, HCN, WITH_INJECT)
        else:
            keep = (rows[:, None] < M) & (cols[None, :] < N)
            at = rows[:, None] * N + cols[None, :]
            tl.store(PART + pid_k * M * N + at, acc, mask=keep)
            arrived = tl.atomic_add(LOCKS + pid_n, 1, sem="acq_rel")
            if arrived == SPLIT - 1:
                total = tl.zeros((16, BLOCK_N), dtype=tl.float32)
                for s in range(SPLIT):
                    total += tl.load(PART + s * M * N + at, mask=keep, other=0.0, cache_modifier=".cg")
                _gate_store(total, rows, cols, M, MIX, INJ, sm, si, HC_F, R, HCN, WITH_INJECT)
                tl.atomic_xchg(LOCKS + pid_n, 0)

    @triton.jit
    def _up_mean(G, W, NORMED, OUT, M, sg, sw, sn, so, HC_F, HID: tl.constexpr, R: tl.constexpr, HCN: tl.constexpr,
                BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, FP32_DOT: tl.constexpr):
        pid = tl.program_id(0)
        rows = tl.arange(0, 16)
        d = pid * BLOCK_D + tl.arange(0, BLOCK_D)
        live = rows[:, None] < M
        dm = d < HID
        mix = tl.zeros((16, BLOCK_D), dtype=tl.float32)
        for s in tl.static_range(HCN):
            acc = tl.zeros((16, BLOCK_D), dtype=tl.float32)
            for k in range(0, R, BLOCK_K):
                ks = k + tl.arange(0, BLOCK_K)
                km = ks < R
                x = tl.load(G + rows[:, None] * sg + ks[None, :], mask=live & km[None, :], other=0.0)
                w = tl.load(W + (s * HID + d)[:, None] * sw + ks[None, :], mask=dm[:, None] & km[None, :], other=0.0)
                if FP32_DOT:
                    x, w = x.to(tl.float32), w.to(tl.float32)
                acc += tl.dot(x, tl.trans(w))
            g = tl.sigmoid(acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            n = tl.load(NORMED + rows[:, None] * sn + (s * HID + d)[None, :], mask=live & dm[None, :],
                        other=0.0).to(tl.float32)
            mix += (g * n).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + rows[:, None] * so + d[None, :], (mix / HC_F).to(tl.bfloat16), mask=live & dm[None, :])


def fused_site(normed, down_inject, up, *, up_cfg, inject=True, fused_up=True, out=None):
    """(mixed [M, H], injection [M, hc] | None) in two launches (or the lane's up + mix_mean when not `fused_up`)."""
    import torch
    import triton
    from engine.kernels import gated_residual as hcr
    from engine.kernels.common import skinny_gemv
    m, width = normed.shape
    n = down_inject.shape[0]
    block_n, block_k, split, warps, stages = DOWN_CFG
    gates = torch.empty(m, RANK, device=normed.device, dtype=normed.dtype)
    injection = torch.empty(m, HC, device=normed.device, dtype=normed.dtype) if inject else gates
    partial = torch.empty(split, m, n, device=normed.device, dtype=torch.float32)
    _down_gates[(triton.cdiv(n, block_n), split)](
        normed, down_inject, gates, injection, partial, skinny_gemv.prepare(normed.device), m, n, width,
        normed.stride(0), down_inject.stride(0), gates.stride(0), injection.stride(0), float(HC), R=RANK, HCN=HC,
        WITH_INJECT=inject, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT=split, FP32_DOT=not normed.is_cuda,
        num_warps=warps, num_stages=stages)
    mixed = torch.empty(m, HIDDEN, device=normed.device, dtype=normed.dtype) if out is None else out
    if fused_up:
        block_d, bk, w, st = up_cfg
        _up_mean[(triton.cdiv(HIDDEN, block_d),)](gates, up, normed, mixed, m, gates.stride(0), up.stride(0),
                                                 normed.stride(0), mixed.stride(0), float(HC), HID=HIDDEN, R=RANK,
                                                 HCN=HC, BLOCK_D=block_d, BLOCK_K=bk, FP32_DOT=not normed.is_cuda,
                                                 num_warps=w, num_stages=st)
    else:
        weights = torch.mm(gates, up.t())
        hcr._mix_mean[(m,)](weights, normed, mixed, weights.stride(0), normed.stride(0), mixed.stride(0), float(HC),
                            HID=HIDDEN, BD=triton.next_power_of_2(HIDDEN), HC=HC, num_warps=hcr._warps(HIDDEN))
    return mixed, (injection if inject else None)



def run_site(output=None) -> dict:
    """A site's mixer three ways, 16 sites a graph over rotated weights, interleaved: the lane on cuBLAS (before the
    skinny GEMV), the lane with its down projection on it (skinny_gemv.linear_rows), and carry H1 + H2 as two launches
    (`fused_site`), also with only H1. Each fused arm is held byte for byte to the lane with the same products
    (skinny_gemv.gemv at the same tiles) -- the fold changes launches, not arithmetic -- and to the cuBLAS lane within
    two BF16 steps."""
    import torch
    from engine.kernels import gated_residual as hcr
    from engine.kernels.common import skinny_gemv
    torch.manual_seed(0)
    skinny_gemv.prepare("cuda")
    width = HC * HIDDEN
    pair = (RANK + HC) * width * 2 + width * RANK * 2
    copies = max(CALLS, -(-COPIES_BYTES // pair))
    downs = [torch.randn(RANK + HC, width, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    ups = [torch.randn(width, RANK, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "sites_a_graph": CALLS,
              "pair_MB": round(pair / 1e6, 2), "rows": {}}
    for m in ROWS:
        normed = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
        keep = []

        def graph_of(fn):
            outs = [torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(CALLS)]
            keep.append(outs)
            for i in range(CALLS):
                fn(downs[i % copies], ups[i % copies], outs[i])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(CALLS):
                    fn(downs[(i * 7) % copies], ups[(i * 7) % copies], outs[i])
            return g

        def lane(d, u, o):
            o.copy_(hcr.mix(normed, d, u, HC)[0])

        def lane_down(d, u, o):
            o.copy_(hcr.mix(normed, d, u, HC, project_down=lambda t: skinny_gemv.linear_rows(t, d))[0])

        arms = {"lane cublas (+copy)": graph_of(lane), "lane skinny down (+copy)": graph_of(lane_down),
                "H1 only": graph_of(lambda d, u, o: fused_site(normed, d, u, up_cfg=None, fused_up=False, out=o))}
        checks = {}
        ref_cublas = hcr.mix(normed, downs[0], ups[0], HC)
        for cfg in UP_CFGS:
            got = fused_site(normed, downs[0], ups[0], up_cfg=cfg)
            same = hcr.mix(normed, downs[0], ups[0], HC,
                           project_down=lambda t: skinny_gemv.gemv(t, downs[0], DOWN_CFG),
                           project_up=lambda t, c=cfg: skinny_gemv.gemv(t, ups[0], (c[0], c[1], 1, c[2], c[3])))
            checks[str(cfg)] = {"byte_equal_mixed": bool(torch.equal(got[0], same[0])),
                                "byte_equal_inject": bool(torch.equal(got[1], same[1])),
                                "vs_cublas_mixed": round(hcr.drift(got[0], ref_cublas[0])[0], 6),
                                "vs_cublas_inject": round(hcr.drift(got[1], ref_cublas[1])[0], 6)}
            arms[f"H1+H2 {cfg}"] = graph_of(lambda d, u, o, cfg=cfg: fused_site(normed, d, u, up_cfg=cfg, out=o))
        times = {name: [] for name in arms}
        for _ in range(ROUNDS):
            for name, g in arms.items():
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        del arms, keep
        row = {name: round(statistics.median(v), 2) for name, v in times.items()}
        report["rows"][m] = {"us_a_site": row, "checks": checks}
        print(json.dumps({f"site rows {m}": row, "checks": checks}), flush=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
