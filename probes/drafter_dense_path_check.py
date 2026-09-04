#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53 DFlash2 drafter dense path, end to end and offline.

Everything a boot with VLLM_DFLASH2_FP8_DENSE=1 would exercise on the
drafter's GEMMs, short of acceptance: the fp8-dense build pass on a
drafter-shaped module tree under the drafter knob (the merged qkv_proj, the
K=20480 fc and the conv kernel_projections that the base patterns never
matched), the MK W4 packs it attaches (the fc as five K-chunks), the opaque
MK-or-fp8 op that a torch.compiled caller serves through, and its
fallbacks. The drafter's forward is compiled (@support_torch_compile), so
"works in eager" is not the contract here: the op has to survive dynamo
with fullgraph=True, dynamic shapes, and CUDA-graph capture, and produce
the same bytes in all three as in eager.

Gates:
  build     every drafter linear quantized under the knob, none under the
            target's pattern set; an MK pack on each (fc: a list of 5)
  serve     apply() == mk.gemm_w4a8() bitwise for M <= 32 (the lane served);
            rel err vs bf16 in the e2m1 class (<= 0.15); M > 32 falls back
            to the fp8 pair bitwise (rel <= 5e-2)
  compile   torch.compile(fullgraph=True) and dynamic=True of a function
            calling apply(): no graph break, output == eager bitwise
  capture   CUDA-graph replay == eager bitwise

Run inside the glm53 image with the composed overlays mounted (the
megakernel bench runner does exactly that):

    sed 's#megakernel_glm53_bench.py#drafter_dense_path_check.py#' \\
        probes/run_megakernel_bench.sh > /tmp/run_path.sh && bash /tmp/run_path.sh
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
os.environ.setdefault("VLLM_GLM53_MK_MHC", "0")
os.environ.setdefault("VLLM_GLM53_MK_KDA", "0")
os.environ.setdefault("VLLM_GLM53_MK_MLA", "0")
os.environ.setdefault("VLLM_GLM53_MK_PDL", "1")

DRAFTER_ENV = "VLLM_DFLASH2_FP8_DENSE"
HIDDEN, LAYERS, TP = 4096, 5, 4
QKV_N, O_K, GU_N, DOWN_K, KPROJ_N = 1536, 1024, 6144, 3072, 1024
TOL_W4, TOL_FP8 = 0.15, 5e-2

fails: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def _load():
    for name in ("glm53_fp8_dense", "glm53_megakernel"):
        if importlib.util.find_spec(f"vllm.model_executor.layers.{name}") is None:
            print(f"!! {name} not mounted -- nothing to check")
            sys.exit(2)
    from vllm.model_executor.layers import glm53_fp8_dense as fd
    from vllm.model_executor.layers import glm53_megakernel as mk
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    except Exception:
        UnquantizedLinearMethod = None
    return fd, mk, UnquantizedLinearMethod


class Lin(nn.Module):
    """A Linear the way the build pass sees one: a 2-D bf16 weight and a
    quant_method whose apply(layer, x, bias) is the bf16 GEMM."""

    def __init__(self, n, k, base_cls):
        super().__init__()
        self.weight = nn.Parameter(
            (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16),
            requires_grad=False)
        self.quant_method = base_cls()


def _base_cls(real):
    if real is not None:
        return real

    class UnquantizedLinearMethod:  # name is what the build pass tests
        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    return UnquantizedLinearMethod


class Attn(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.qkv_proj = Lin(QKV_N, HIDDEN, base)
        self.o_proj = Lin(HIDDEN, O_K, base)


class Mlp(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.gate_up_proj = Lin(GU_N, HIDDEN, base)
        self.down_proj = Lin(HIDDEN, DOWN_K, base)


class Conv(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.kernel_projection = Lin(KPROJ_N, HIDDEN, base)


class Layer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.self_attn = Attn(base)
        self.mlp = Mlp(base)
        self.attention_conv = Conv(base)
        self.mlp_conv = Conv(base)


class Drafter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.layers = nn.ModuleList([Layer(base) for _ in range(LAYERS)])
        self.fc = Lin(HIDDEN, LAYERS * HIDDEN, base)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, Lin)]


def main() -> int:
    torch.manual_seed(0)
    fd, mk, real_base = _load()
    base = _base_cls(real_base)
    mk.maybe_arm()
    check(bool(mk._ARMED.get("gemm")), "MK-GEMM armed (self-tests passed)")

    # --- the target's pattern set must not reach the drafter's names
    names = [n for n, _ in _linears(Drafter(base))]
    tgt = fd._include_patterns()
    drf = fd._include_patterns(DRAFTER_ENV)
    hit_t = [n for n in names if any(p.search(n) for p in tgt)]
    hit_d = [n for n in names if any(p.search(n) for p in drf)]
    check(len(hit_d) == len(names),
          f"drafter patterns match every drafter linear ({len(hit_d)}/{len(names)})")
    check(all(n.endswith(("o_proj", "gate_up_proj", "down_proj")) for n in hit_t),
          "the target's pattern set matches o/gate_up/down of the drafter only "
          "(qkv_proj, fc, kernel_projection are the drafter set's)")

    # --- build under the drafter knob
    os.environ[DRAFTER_ENV] = "1"
    model = Drafter(base)
    ok = fd.maybe_build_fp8_dense(model, env=DRAFTER_ENV)
    check(ok, "build pass armed something under the drafter knob")
    lins = _linears(model)
    meth = [(n, m.quant_method) for n, m in lins]
    check(all(isinstance(q, fd.Fp8DenseMethod) for _, q in meth),
          f"every drafter linear carries an Fp8DenseMethod ({len(meth)})")
    check(all(getattr(q, "_opaque", False) for _, q in meth),
          "every drafter method is marked opaque (compiled caller)")
    check(all(getattr(q, "_mk", None) is not None for _, q in meth),
          "every drafter method carries an MK W4 pack")
    fc_q = dict(meth)["fc"]
    check(isinstance(fc_q._mk, list) and len(fc_q._mk) == LAYERS * HIDDEN // mk.MK_GEMM_KMAX,
          f"the fc pack is {LAYERS * HIDDEN // mk.MK_GEMM_KMAX} K-chunks (K=20480)")
    check(all(not isinstance(q._mk, list) for n, q in meth if n != "fc"),
          "every K <= 4096 linear carries a single pack")

    # --- serve: apply == the lane bitwise, e2m1 class vs bf16; M > 32 -> fp8
    for m in (7, 8, 32):
        for n, mod in lins:
            q = mod.quant_method
            x = (torch.randn(m, mod.weight.shape[1], device="cuda") * 1.5).to(torch.bfloat16)
            got = q.apply(mod, x)
            direct = mk.gemm_w4a8(x, q._mk, q._rows)
            ref = torch.nn.functional.linear(x.float(), mod.weight.float())
            r = _rel(got, ref)
            check(direct is not None and torch.equal(got, direct) and r <= TOL_W4,
                  f"M={m} {n}: apply == MK lane bitwise, rel {r:.2e} <= {TOL_W4}")
    for n, mod in (lins[0], lins[-1]):
        q = mod.quant_method
        x = (torch.randn(40, mod.weight.shape[1], device="cuda") * 1.5).to(torch.bfloat16)
        got = q.apply(mod, x)
        fp8 = fd._fp8_dense_gemm(x, q._q, q._ws, q._rows, q._cols)
        r = _rel(got, torch.nn.functional.linear(x.float(), mod.weight.float()))
        check(mk.gemm_w4a8(x, q._mk, q._rows) is None and torch.equal(got, fp8)
              and r <= TOL_FP8,
              f"M=40 {n}: the lane declines, the fp8 pair serves bitwise, rel {r:.2e}")

    # --- compile: the op is opaque to dynamo, fullgraph and dynamic
    qkv = model.layers[0].self_attn.qkv_proj
    fc = model.fc

    def f(x, xfc):
        return (qkv.quant_method.apply(qkv, x), fc.quant_method.apply(fc, xfc))

    x8 = (torch.randn(8, HIDDEN, device="cuda") * 1.5).to(torch.bfloat16)
    xfc8 = (torch.randn(8, LAYERS * HIDDEN, device="cuda") * 1.5).to(torch.bfloat16)
    eager = f(x8, xfc8)
    try:
        cf = torch.compile(f, fullgraph=True, dynamic=False)
        got = cf(x8, xfc8)
        check(all(torch.equal(a, b) for a, b in zip(got, eager)),
              "torch.compile(fullgraph=True): no graph break, == eager bitwise")
    except Exception as e:
        check(False, f"torch.compile(fullgraph=True) raised: {e!r}"[:200])
    try:
        cfd = torch.compile(f, fullgraph=True, dynamic=True)
        got8 = cfd(x8, xfc8)
        x16 = (torch.randn(16, HIDDEN, device="cuda") * 1.5).to(torch.bfloat16)
        xfc16 = (torch.randn(16, LAYERS * HIDDEN, device="cuda") * 1.5).to(torch.bfloat16)
        got16 = cfd(x16, xfc16)
        e16 = f(x16, xfc16)
        check(all(torch.equal(a, b) for a, b in zip(got8, eager))
              and all(torch.equal(a, b) for a, b in zip(got16, e16)),
              "torch.compile(dynamic=True): M=8 and M=16 == eager bitwise")
    except Exception as e:
        check(False, f"torch.compile(dynamic=True) raised: {e!r}"[:200])

    # --- capture: a CUDA graph of the served path replays the same bytes
    try:
        st = torch.cuda.Stream()
        xs = x8.clone()
        xfs = xfc8.clone()
        with torch.cuda.stream(st):
            for _ in range(2):
                f(xs, xfs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(st):
            with torch.cuda.graph(g, stream=st):
                outs = f(xs, xfs)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        check(all(torch.equal(a, b) for a, b in zip(outs, eager)),
              "CUDA-graph replay of apply() == eager bitwise")
    except Exception as e:
        check(False, f"CUDA-graph capture raised: {e!r}"[:200])

    os.environ[DRAFTER_ENV] = "0"

    if fails:
        print("\nFAIL:", *fails, sep="\n  ")
    print("VERDICT:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
