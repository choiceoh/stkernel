"""Does the 128-block fp8 scale contract earn what it costs on prefill?

`glm53_fp8_dense` builds its own fp8 pack from the bf16 weights and calls
deepgemm's `fp8_gemm_nt`, so the scale granularity is ours to choose -- neither
the checkpoint (NVFP4) nor vLLM pins it.  The block scheme rescales the
accumulator every 128 elements of K; a per-row/per-token scale is a pure
epilogue multiply and lets the K loop run at the part's own issue rate.

Two questions, one probe:

  accuracy  quantize BOTH operands on real checkpoint weights, with and
            without the activation outliers that motivate blockwise fp8,
            and compare relative error against the bf16 reference.
  speed     time the GEMM each scheme would actually issue, at the shapes
            the serving path hits during chunked prefill.

Run inside the serving image, with no engine up (it needs the whole GPU):

  docker run --rm --gpus all --entrypoint python3 \
    -v $PWD/probes:/p:ro -v <model>:/models/glm:ro glm53:v13-b12x \
    /p/fp8_scale_granularity.py
"""
import glob, json, statistics as st, struct, sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

F8 = torch.float8_e4m3fn
E4M3_MAX = 448.0
# The dense projections glm53_fp8_dense actually replaces, at the per-rank
# shapes of a TP4 boot; M is the chunked-prefill token count.
SHAPES = [(8192, 4096, 4096), (8192, 2048, 4096), (8192, 512, 4096),
          (8192, 3072, 4096), (4096, 4096, 4096), (2048, 4096, 4096)]
MODEL_GLOB = "/models/glm/*.safetensors"


def load_weights(limit=6):
    """Real bf16 dense projections, straight out of the safetensors."""
    want = ("self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj",
            "fused_qkv_a_proj")
    out = []
    for f in sorted(glob.glob(MODEL_GLOB)):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__" or not k.endswith(".weight"):
                continue
            if v.get("dtype") != "BF16" or not any(w in k for w in want):
                continue
            shape = v["shape"]
            if len(shape) != 2 or min(shape) < 512 or shape[0] * shape[1] > 4e7:
                continue
            s, e = v["data_offsets"]
            with open(f, "rb") as fh:
                fh.seek(base + s)
                raw = fh.read(e - s)
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
            out.append((k.split("model.language_model.")[-1], t.view(*shape)))
            if len(out) >= limit:
                return out
    return out


def quant_dequant(t, br, bc):
    """Quantize-dequantize in (br, bc) blocks; a negative size means the
    whole dimension, so (1, -1) is per-row and (1, 128) is deepgemm's
    activation layout."""
    n, k = t.shape
    br = n if br < 0 else br
    bc = k if bc < 0 else bc
    pn, pk = -(-n // br) * br, -(-k // bc) * bc
    p = torch.zeros(pn, pk, device=t.device, dtype=torch.float32)
    p[:n, :k] = t.float()
    v = p.view(pn // br, br, pk // bc, bc)
    s = v.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-6) / E4M3_MAX
    return ((v / s).to(F8).float() * s).view(pn, pk)[:n, :k]


def activations(m, k, outliers, device):
    """LLM activations put 10-100x the typical magnitude in a few channels;
    resolving those is the case blockwise scaling is supposed to win."""
    x = torch.randn(m, k, device=device, dtype=torch.float32)
    if outliers:
        ch = torch.randperm(k, device=device)[: max(1, k // 256)]
        x[:, ch] *= 40.0
    return x


def report_accuracy(device):
    print("accuracy -- relative error vs bf16, both operands quantized")
    print(f"  {'tensor':38s} {'outliers':>8s} {'block128':>10s} {'rowwise':>10s}"
          f" {'ratio':>6s}")
    agg = {}
    for name, w in load_weights():
        w = w.to(device)
        for outliers in (False, True):
            x = activations(1024, w.shape[1], outliers, device)
            ref = x @ w.float().T
            rn = ref.norm().item()
            # deepgemm: activation 1x128 along K, weight 128x128.
            blk = ((quant_dequant(x, 1, 128) @ quant_dequant(w, 128, 128).T)
                   - ref).norm().item() / rn
            # rowwise: activation per token over all of K, weight per output
            # channel over all of K -- no K-direction blocking at all.
            row = ((quant_dequant(x, 1, -1) @ quant_dequant(w, 1, -1).T)
                   - ref).norm().item() / rn
            agg.setdefault(outliers, []).append((blk, row))
            print(f"  {name[:38]:38s} {str(outliers):>8s} {blk:10.2e}"
                  f" {row:10.2e} {row / blk:5.1f}x")
    for outliers in (False, True):
        b = [x for x, _ in agg[outliers]]
        r = [x for _, x in agg[outliers]]
        print(f"  MEAN outliers={outliers!s:5s} block {st.mean(b):.2e}"
              f"  rowwise {st.mean(r):.2e}"
              f"  -> rowwise is {st.mean(r) / st.mean(b):.1f}x the error")


def timed(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def report_speed(device):
    # 48 SMs x 4 tensor cores x 512 e4m3 MAC/cycle/core x 2 flop @ 1.592 GHz.
    ceiling = 155.1
    print(f"\nspeed -- TFLOP/s ({ceiling:.0f} is this part's e4m3 issue rate)")
    print(f"  {'M':>6s} {'N':>5s} {'K':>5s} {'deepgemm':>9s} {'rowwise':>9s}"
          f" {'pertensor':>9s} {'bf16':>7s}")
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8_packed_for_deepgemm)
        from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8
    except Exception as exc:                       # probe still useful without
        print(f"  (deepgemm column unavailable: {exc})")
        fp8_gemm_nt = None
    for m, n, k in SHAPES:
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        gflop = 2.0 * m * n * k / 1e9
        cols = []

        if fp8_gemm_nt is not None:
            bq, bs = per_block_cast_to_fp8(b.float(), [128, 128], use_ue8m0=True)
            out = torch.empty(m, n, device=device, dtype=torch.bfloat16)

            def run_deepgemm():
                aq, asc = per_token_group_quant_fp8_packed_for_deepgemm(a, 128)
                fp8_gemm_nt((aq, asc), (bq, bs), out)

            cols.append(gflop / timed(run_deepgemm))
        else:
            cols.append(float("nan"))

        sa = (a.abs().amax(1, keepdim=True).float() / E4M3_MAX).clamp_min(1e-12)
        sb = (b.abs().amax(1, keepdim=True).float() / E4M3_MAX).clamp_min(1e-12)
        aq = (a.float() / sa).to(F8)
        bq = (b.float() / sb).to(F8)
        sbt = sb.view(1, n).contiguous()
        for scale_a, scale_b in ((sa, sbt),
                                 (sa.amax().view(1, 1), sb.amax().view(1, 1))):
            try:
                cols.append(gflop / timed(
                    lambda: torch._scaled_mm(aq, bq.t(), scale_a=scale_a,
                                             scale_b=scale_b,
                                             out_dtype=torch.bfloat16)))
            except Exception:
                cols.append(float("nan"))
        cols.append(gflop / timed(lambda: a @ b.t()))
        print(f"  {m:6d} {n:5d} {k:5d} " + " ".join(f"{c:9.1f}" for c in cols))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    report_accuracy(device)
    if device == "cuda":
        report_speed(device)
    else:
        print("\nspeed -- skipped, no GPU visible")


if __name__ == "__main__":
    main()
