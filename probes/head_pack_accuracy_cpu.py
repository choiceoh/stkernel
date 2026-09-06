#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pack accuracy of ONE weight on the host CPU -- no GPU, no container, no fleet.

probes/mk_pack_accuracy.py is the full tool (kernel columns, timings, LoRC
arms) and needs a GPU in a fresh container. This one answers only the
accuracy question, on the host, so it can run while the fleet is serving or
booting: it loads a real weight out of the checkpoint, builds the shipped
packs with the driver's own builder, and scores each arm against the bf16
truth in fp32 on 256 held-out rows of the same synthetic generator the 33차
table used (log-normal per-channel scale, 8 outlier channels, weak low-rank
correlation; synthetic Hessian, 4096 tokens).

    python3 probes/head_pack_accuracy_cpu.py qproj   # the harness check
    python3 probes/head_pack_accuracy_cpu.py head    # the served vocab head

`qproj` MUST reproduce the published 33차 row (fp8 2.66e-2 / RTN 8.33e-2 /
GPTQ 7.27e-2) before the `head` row means anything -- the fp8 arm here is a
torch twin of the deepgemm pair (ue8m0 128x128 weight blocks AND ue8m0
per-token-group activation scales; an fp32 activation scale reads 1.62e-2
and is the wrong twin).

CAVEAT the numbers carry: the Hessian is SYNTHETIC. 33차 §(real dumps)
measured GPTQ at -74% on real Hessians against -13% here, so this
underestimates GPTQ. It is an upper bound on the W4 error, not the served
one; the served number needs a calibration boot that dumps the weight's own
Hessian (for lm_head, that means MK_HEAD_DRAFT=1 during MK_CALIB=1 -- the
calib hook only sees linears that go through the GEMM lane).
"""
import importlib.util, json, os, sys, time
import torch
torch.set_num_threads(int(os.environ.get("NT", "6")))
REPO = os.environ.get("MK_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("MK_MODEL", "/home/choiceoh/models/glm53-redhat-nvfp4")
sys.path.insert(0, os.path.join(REPO, "probes"))
if not os.path.exists(os.path.join(MODEL, "model.safetensors.index.json")):
    sys.exit(f"no checkpoint at {MODEL} (set MK_MODEL)")

spec = importlib.util.spec_from_file_location(
    "mkdrv", os.path.join(REPO, "overlay/modules/glm53_megakernel/glm53_megakernel.py"))
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
import mk_pack_accuracy as mkpa
mkpa.DEV = "cpu"                                                  # host CPU: the GPU belongs to serving
from mk_pack_accuracy import SynthActs, _rel                      # same generator

def load(key, rows=None):
    from safetensors import safe_open
    wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    f = safe_open(os.path.join(MODEL, wm[key]), framework="pt", device="cpu")
    t = f.get_slice(key)
    return (t[rows[0]:rows[1]] if rows else t[:]).to(torch.bfloat16)

def fp8_twin(w, x):
    """The stock deepgemm pair in torch: ue8m0 128x128 weight blocks, exact
    per-token-group activation scale, both dequantized into an fp32 matmul."""
    n, k = w.shape
    rpad = (-n) % 128                                             # the real path zero-pads to 128s
    wp = torch.cat([w, w.new_zeros(rpad, k)], 0) if rpad else w
    wb = wp.float().reshape((n + rpad) // 128, 128, k // 128, 128)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))          # ue8m0: pow2 only
    wq = (wb / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    wdq = wq.reshape(n + rpad, k)[:n]                             # trim the padded rows
    xg = x.float().reshape(x.shape[0], k // 128, 128)
    xs = (xg.abs().amax(dim=2, keepdim=True) / 448.0).clamp(min=1e-30)
    if os.environ.get("XPOW2", "1") == "1":
        xs = torch.exp2(torch.ceil(torch.log2(xs)))               # ue8m0 activations too
    xdq = ((xg / xs).clamp(-448, 448).to(torch.float8_e4m3fn).float() * xs).reshape(x.shape[0], k)
    return xdq @ wdq.T

def arms(w, H, ntok, x_eval, truth):
    out = []
    out.append(("stock w8a8", _rel(fp8_twin(w, x_eval), truth), 0.0))
    for arm, rowshift, hess in (("rtn/ten", "0", None), ("rtn/row", "1", None), ("gptq/row", "1", "H")):
        os.environ["VLLM_GLM53_MK_PACK_ROWSHIFT"] = rowshift
        os.environ["VLLM_GLM53_MK_PACK_LORC"] = "0"
        os.environ["VLLM_GLM53_MK_PACK_GPTQ"] = "0" if hess is None else "1"
        t0 = time.perf_counter()
        if hess is None:
            pack = mk.build_mk_weight_w4(w)
        else:
            mk._CALIB_OVERRIDE = (H, ntok)
            pack = mk.build_mk_weight_w4(w, name="__probe__")
            mk._CALIB_OVERRIDE = None
        b = time.perf_counter() - t0
        out.append((arm, _rel(mk.mk_pack_twin(x_eval, pack, w.shape[0]), truth), b))
        del pack
    return out

def run(tag, w):
    n, k = w.shape
    acts = SynthActs(k)
    x_eval = acts.draw(256, seed=3).to(torch.bfloat16)
    truth = x_eval.float() @ w.float().T
    H, ntok = acts.hessian(4096, seed=2)
    print(f"\n{tag}  [{n} x {k}]")
    for arm, e, b in arms(w, H, ntok, x_eval, truth):
        print(f"    {arm:<12}{e:>11.3e}   build {b:6.1f}s")
    del truth, x_eval, H

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "qproj"
    if which not in ("qproj", "head"):
        sys.exit("usage: head_pack_accuracy_cpu.py [qproj|head]")
    if which == "qproj":     # harness check vs the published 33차 row
        run("L1 q_proj [real]  (published: fp8 2.66e-2 / rtn 8.33e-2 / gptq 7.27e-2)",
            load("model.language_model.layers.1.self_attn.q_proj.weight"))
    else:                    # the question: the served head, one rank's row slice
        run("lm_head [real, rank-3 slice]", load("lm_head.weight", (3 * 38720, 4 * 38720)))
