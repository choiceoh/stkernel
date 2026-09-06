#!/usr/bin/env python3
"""33차: the W4 lane's accuracy against the bf16 TRUTH, per pack lever.

The bench's rel_err column is the gap between the MK lane and the stock
fp8 pair -- two quantized arms disagreeing -- which cannot rank levers
that move the lane closer to what the bf16 weights would have produced.
This probe measures every arm against x_bf16 @ W_bf16^T in fp32:

  stock      the deepgemm W8A8 pair (what the lane replaced)
  rtn/ten    the 32차 packer: e2m1 x e4m3-scale, per-tensor shift
  rtn/row    + per-row shift (lever 3)
  gptq/row   + GPTQ error feedback on a synthetic anisotropic Hessian
             (lever 2; the served numbers come from the calibration boot)
  +lorc r    + rank-r activation-aware low-rank correction (lever 4)

each as the KERNEL's output (v2 lane) and, beside it, the pure-torch
reference of the same pack (so a kernel-side defect shows as a gap between
the two columns). The activation scale is the exact one on every arm
(lever 1 is unconditional in the kernel). Also the v2 launch time per arm,
so "speed untouched" is a number, not a claim.

Weights: the checkpoint's own bf16 dense projections when /models/glm53 is
mounted (layer L q/k/v/o, shared-expert gate_up / down), else N(0, 0.02).
Activations: synthetic hidden states with a log-normal per-channel scale
(sigma 1) and a few outlier channels -- the shape real residual streams
have -- so the Hessian the GPTQ/LoRC arms see is anisotropic like a real
one. With --calib <dir> the dumped Hessians of a calibration boot are used
instead for the named linears.

    mkprobe.sh probes/mk_pack_accuracy.py [--layer 1] [--m 32] [--iters 20]
        [--lorc 16,32] [--shapes 6416:4096,1024:4096,4096:512]
"""
import argparse
import json
import os
import sys
import time

import torch

DEV = "cuda"


def _rel(a, b):
    a = a.float(); b = b.float()
    d = (a - b).norm(); n = b.norm()
    return float(d / n) if n > 0 else float(d)


def _time(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        t.append(s.elapsed_time(e) * 1e3)
    t.sort()
    return t[len(t) // 2]


def real_weights(layer):
    """{name: bf16 [n, k]} of the layer's dense projections from the
    checkpoint, TP-unsharded (the probe is single-device: the shard is a
    row/column slice, the error statistics are the same)."""
    root = "/models/glm53"
    idx = os.path.join(root, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return {}
    from safetensors import safe_open
    wm = json.load(open(idx))["weight_map"]
    want = {
        "q_proj": f"model.language_model.layers.{layer}.self_attn.q_proj.weight",
        "k_proj": f"model.language_model.layers.{layer}.self_attn.k_proj.weight",
        "v_proj": f"model.language_model.layers.{layer}.self_attn.v_proj.weight",
        "o_proj": f"model.language_model.layers.{layer}.self_attn.o_proj.weight",
        "se.gate_up": f"model.language_model.layers.{layer}.mlp.shared_experts.gate_up_proj.weight",
        "se.down": f"model.language_model.layers.{layer}.mlp.shared_experts.down_proj.weight",
        # the vocab head: one TP rank's row slice, the shape the head lane packs
        "lm_head": "lm_head.weight",
        "se.gate": f"model.language_model.layers.{layer}.mlp.shared_experts.gate_proj.weight",
        "se.up": f"model.language_model.layers.{layer}.mlp.shared_experts.up_proj.weight",
    }
    out = {}
    files = {}
    for tag, key in want.items():
        if key not in wm:
            continue
        f = wm[key]
        if f not in files:
            files[f] = safe_open(os.path.join(root, f), framework="pt", device="cpu")
        t = files[f].get_tensor(key)
        if tag == "lm_head":
            t = t[:t.shape[0] // 4]        # per-rank rows (TP4); the stats are per row
        if t.dtype not in (torch.bfloat16, torch.float16):
            continue  # fp8/fp4 experts are not the lane's
        out[tag] = t.to(torch.bfloat16)
    if "se.gate" in out and "se.up" in out and "se.gate_up" not in out:
        out["se.gate_up"] = torch.cat([out.pop("se.gate"), out.pop("se.up")], 0)
    return out


class SynthActs:
    """Hidden-state-like activations: N(0,1) plus a mild low-rank
    correlation, times a log-normal per-channel scale with 8 outlier
    channels at 20x. ONE generator model for calibration and evaluation
    (held-out seeds): GPTQ/LoRC are judged on the distribution they were
    fitted on, as they will be in serving (the first probe evaluated on a
    different one and read GPTQ 12% WORSE than RTN -- an artefact)."""

    def __init__(self, k, seed=1):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.k = k
        self.s = torch.exp(torch.randn(k, generator=g) * 1.0)
        self.s[torch.randperm(k, generator=g)[:8]] *= 20.0
        self.mix = torch.randn(8, k, generator=g) * 0.3

    def draw(self, m, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(m, self.k, generator=g)
        u = torch.randn(m, 8, generator=g) @ self.mix
        return ((z + u) * self.s[None, :]).to(DEV, torch.float32)

    def hessian(self, ntok, seed=2):
        x = self.draw(ntok, seed)
        return x.T @ x, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lorc", default="16,32")
    ap.add_argument("--shapes", default="6416:4096,1024:4096,4096:512,4096:4096")
    ap.add_argument("--calib", default=None, help="a calibration dump dir (rank<r>/<name>.pt)")
    ap.add_argument("--ntok", type=int, default=4096, help="synthetic calibration tokens")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--eval-rows", type=int, default=256, help="held-out rows for the error (twin)")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
    os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
    os.environ["VLLM_GLM53_MK_PACK_CACHE"] = "off"
    from vllm.model_executor.layers import glm53_megakernel as mk
    from vllm.model_executor.layers.glm53_fp8_dense import _fp8_dense_gemm
    ext = mk._build()
    ext.set_gemm2(0)   # the rule's split (the production lane)

    cases = []
    if not args.no_real:
        for tag, w in real_weights(args.layer).items():
            cases.append((f"L{args.layer}.{tag}", w.to(DEV)))
    for sh in args.shapes.split(","):
        n, k = (int(v) for v in sh.split(":"))
        torch.manual_seed(0)
        cases.append((f"randn {n}x{k}", (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)))
    ranks = [int(r) for r in args.lorc.split(",") if r.strip()]

    print(f"{'weight':<22}{'arm':<14}{'err/truth':>11}{'build s':>11}{'kernel-twin':>13}{'v2 us':>8}")
    print("err/truth: the arm's pure-torch twin on 256 held-out rows vs x_bf16 @ W_bf16^T; build s: the pack "
          "build (GPTQ solve / SVD included); kernel-twin: the v2 kernel's bf16 output vs the twin's bf16")
    m = args.m
    for name, w in cases:
        n, k = w.shape
        if k % 128 or k > mk.MK_GEMM_KMAX:
            print(f"{name:<22}skipped (k={k})")
            continue
        acts = SynthActs(k)
        x_eval = acts.draw(args.eval_rows, seed=3)           # held out: twin error
        x = x_eval[:m].to(torch.bfloat16)                    # the kernel's rows
        truth_eval = x_eval.to(torch.bfloat16).float() @ w.float().T
        truth = truth_eval[:m]
        rows = []
        # stock fp8 pair
        sq, sws, srows, scols = mk._stock_fp8_pair(w)
        got = _fp8_dense_gemm(x, sq, sws, srows, scols)
        got_eval = _fp8_dense_gemm(x_eval.to(torch.bfloat16), sq, sws, srows, scols)
        t = _time(lambda: _fp8_dense_gemm(x, sq, sws, srows, scols), args.iters)
        rows.append(("stock w8a8", _rel(got_eval, truth_eval), 0.0, float("nan"), t))
        # the pack arms
        H, ntok = acts.hessian(args.ntok, seed=2)
        calib_tag = None
        if args.calib:
            p = os.path.join(args.calib, name.split(".", 1)[-1] + ".pt")
            if os.path.exists(p):
                blob = torch.load(p, map_location="cpu")
                H, ntok = blob["H"].to(DEV), int(blob["ntok"]); calib_tag = "real H"
        arms = [("rtn/ten", "0", None, 0), ("rtn/row", "1", None, 0),
                ("gptq/row", "1", "H", 0)]
        for r in ranks:
            arms.append((f"gptq+lorc{r}", "1", "H", r))
            arms.append((f"rtn+lorc{r}", "1", None, r))
        for arm, rowshift, hess, lr in arms:
            os.environ["VLLM_GLM53_MK_PACK_ROWSHIFT"] = rowshift
            os.environ["VLLM_GLM53_MK_PACK_LORC"] = str(lr)
            tb = time.perf_counter()
            if hess is None:
                os.environ["VLLM_GLM53_MK_PACK_GPTQ"] = "0"
                pack = mk.build_mk_weight_w4(w)
            else:
                os.environ["VLLM_GLM53_MK_PACK_GPTQ"] = "1"
                mk._CALIB_OVERRIDE = (H, ntok)
                pack = mk.build_mk_weight_w4(w, name="__probe__")
                mk._CALIB_OVERRIDE = None
            torch.cuda.synchronize()
            build_s = time.perf_counter() - tb
            if lr and pack[4] is None:
                pack = mk.MKPack(pack[0], pack[1], pack[2], pack[3],
                                 *mk._w4_lorc(w, mk.mk_w4_dequant(pack[0], pack[1], n, pack[2], pack[3]),
                                              H if hess else None, lr)[:2])
            twin_eval = mk.mk_pack_twin(x_eval.to(torch.bfloat16), pack, n)
            twin = twin_eval[:m]
            got = mk._gemm_call(x, pack, n)
            torch.cuda.synchronize()
            t = _time(lambda: mk._gemm_call(x, pack, n), args.iters)
            rows.append((arm + ("" if hess is None or calib_tag is None else "*"),
                         _rel(twin_eval, truth_eval), build_s,
                         _rel(got, twin.to(torch.bfloat16)), t))
        for arm, e, b, gap, t in rows:
            print(f"{name:<22}{arm:<14}{e:>11.3e}{b:>11.2f}{gap:>13.2e}{t:>8.1f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
