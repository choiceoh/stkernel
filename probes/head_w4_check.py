#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What a W4 (e2m1 x e4m3-scale) vocabulary head would do to the logits.

The armed 09-03 trace ends every decode step with two fp8 (W8A16) head GEMMs,
target 836 us and draft 812 us, at ~190 GB/s -- their fp8 floor. The MK W4
lane runs the same [vocab/tp, 4096] shape in 418 us (probes/
drafter_fc_check.py --head, srv2). Whether the halved bytes are worth having
is a numerics question first: a W4 weight carries ~8.6e-2 relative error
(MEASUREMENTS 24차), and the head's argmax is the served token (target) or
the candidate set (draft).

This probe answers it offline and CPU-capable, on the REAL head shard: the
checkpoint's lm_head.weight, rows of TP rank 0 (vocab-parallel), against
hidden states of two kinds -- Gaussian at the final-norm scale, and rows of
the checkpoint's embedding table RMS-normalised the way the final norm does
it (real token geometry, if not real decode states). Arms:

  bf16      the reference, fp32 matmul of the bf16 weight
  w8        block-fp8 W8A16 as fp8_lm_head serves it: 128x128 blocks, ue8m0
            scales, e4m3 weights, bf16 activations (a pure-torch twin)
  w4        the MK pack round-tripped through mk_w4_dequant, times the
            kernel's fp8 activation quant twin (_mk_quant_x_ref) -- what the
            lane would feed the mma

Reported per arm: top-1 agreement with bf16 (the number that matters), top-8
overlap (the DFlash2 selector's candidate set), logit relative error, and the
margin of the flipped rows. Run on the host (no GPU, no vllm needed):

    nice -n 19 python3 probes/head_w4_check.py --threads 2 --rows 256
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import struct
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/choiceoh/models/glm53-redhat-nvfp4"


def _load_defs(path: str, names: set[str], ns: dict) -> dict:
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


def _mk_defs():
    ns = {"logging": logging, "logger": logging.getLogger("head_w4"), "math": __import__("math"),
          "os": os, "torch": torch}
    return _load_defs(os.path.join(REPO, "overlay/modules/glm53_megakernel/glm53_megakernel.py"),
                      {"_E2M1_GRID", "_E2M1_MIDS", "_mk_pad128", "build_mk_weight_w4",
                       "mk_w4_dequant", "_mk_quant_x_ref", "MK_GEMM_KMAX"}, ns)


def _safetensor(path: str, name: str) -> torch.Tensor:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
        meta = h[name]
        a, b = meta["data_offsets"]
        fh.seek(8 + n + a)
        buf = fh.read(b - a)
    dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[meta["dtype"]]
    return torch.frombuffer(bytearray(buf), dtype=dt).view(*meta["shape"])


def _index(model: str) -> dict:
    return json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]


def block_fp8_w8(w: torch.Tensor) -> torch.Tensor:
    """fp8_lm_head's weight numerics: 128x128 blocks, ue8m0 scale = 2^ceil(
    log2(amax/448)), e4m3 round-to-nearest, dequantised back to fp32."""
    n, k = w.shape
    npad = -(-n // 128) * 128  # the vocab shard (38720 rows) is not a 128 multiple
    x = torch.zeros(npad, k, dtype=torch.float32)
    x[:n] = w.float()
    x = x.view(npad // 128, 128, k // 128, 128)
    amax = x.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    sf = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (x / sf).to(torch.float8_e4m3fn).float() * sf
    return q.view(npad, k)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--rows", type=int, default=256, help="hidden states per kind")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="smoke test: only the first N rows of the shard")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    t0 = time.time()
    idx = _index(args.model)
    head = _safetensor(os.path.join(args.model, idx["lm_head.weight"]), "lm_head.weight")
    normw = _safetensor(os.path.join(args.model, idx["model.language_model.norm.weight"]),
                        "model.language_model.norm.weight").float()
    V, H = head.shape
    shard = V // args.tp
    w = head[args.rank * shard:(args.rank + 1) * shard]
    if args.limit_rows:
        w = w[:args.limit_rows]
    w = w.contiguous().to(dev)
    print(f"lm_head {tuple(head.shape)} -> rank {args.rank}/{args.tp} shard {tuple(w.shape)} "
          f"({w.numel() * 2 / 2**20:.0f} MiB bf16), loaded in {time.time() - t0:.0f}s")

    # hidden states: Gaussian at the final-norm scale, and RMS-normed embedding rows
    emb_file = os.path.join(args.model, idx["model.language_model.embed_tokens.weight"])
    emb = _safetensor(emb_file, "model.language_model.embed_tokens.weight")
    g = torch.Generator().manual_seed(args.seed)
    rows = torch.randint(0, emb.shape[0], (args.rows,), generator=g)
    e = emb[rows].float()
    e = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-5) * normw
    gauss = torch.randn(args.rows, H, generator=g) * normw
    kinds = {"gauss": gauss.to(dev), "embed": e.to(dev)}
    del emb, head

    mk = _mk_defs()
    tb = time.time()
    pack = mk["build_mk_weight_w4"](w)
    w4 = mk["mk_w4_dequant"](pack[0], pack[1], w.shape[0], pack[2]).float()
    print(f"W4 pack built in {time.time() - tb:.0f}s (gscale 2^{int(round(__import__('math').log2(pack[2])))}), "
          f"weight rel err {float((w4 - w.float()).norm() / w.float().norm()):.3e}")
    w8 = block_fp8_w8(w)
    print(f"W8 block-fp8 weight rel err {float((w8 - w.float()).norm() / w.float().norm()):.3e}")
    wf = w.float()

    # A flip where bf16's top-1 leads top-2 by a hair is sampling noise at
    # the lane's temperatures (0.8-0.95); a flip on a confident row is a
    # served-token change. Both are reported, and so is the KL of the
    # softmax at T=0.8 -- the quantity rejection sampling actually sees.
    print(f"\n{'hidden':<7}{'arm':<5}{'top1 agree':>11}{'conf(m>=1) agree':>17}{'top8 overlap':>13}"
          f"{'logit rel':>11}{'KL@T0.8':>10}{'flipped margin (median, n)':>28}")
    for kind, x in kinds.items():
        xb = x.to(torch.bfloat16).float()               # the served activation dtype
        ref = xb @ wf.t()
        top1 = ref.argmax(-1)
        top8 = ref.topk(8, dim=-1).indices
        srt = ref.topk(2, dim=-1).values
        margin = srt[:, 0] - srt[:, 1]
        arms = {
            "w8": xb @ w8.t(),
            "w4": mk["_mk_quant_x_ref"](x.to(torch.bfloat16)).to(dev) @ w4.t(),
        }
        conf = margin >= 1.0
        logp_ref = torch.log_softmax(ref / 0.8, dim=-1)
        for name, out in arms.items():
            t1 = out.argmax(-1)
            same = t1 == top1
            agree = float(same.float().mean())
            cagree = float(same[conf].float().mean()) if int(conf.sum()) else float("nan")
            o8 = out.topk(8, dim=-1).indices
            ov = float(torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 8.0
                                     for a, b in zip(o8, top8)]).mean())
            rel = float((out - ref).norm() / ref.norm())
            logp = torch.log_softmax(out / 0.8, dim=-1)
            kl = float((logp_ref.exp() * (logp_ref - logp)).sum(-1).mean())
            flipped = margin[~same]
            fm = f"{float(flipped.median()):.3f} (n={flipped.numel()})" if flipped.numel() else "-"
            print(f"{kind:<7}{name:<5}{agree:>10.1%}{cagree:>16.1%}{ov:>13.1%}{rel:>11.2e}"
                  f"{kl:>10.4f}{fm:>28}")
        print(f"{'':<7}{'':<5}  confident rows (bf16 margin >= 1): {int(conf.sum())}/{conf.numel()}")
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
