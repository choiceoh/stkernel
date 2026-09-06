#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How many distinct values do the NVFP4 block scales actually use?

The gate for "SF 5-bit" (ledger 39차 §4 row 4): in NVFP4 every 16 fp4 values
(8 B) carry one e4m3 scale byte, so scales are 1/9 = 11.1% of the packed
bytes. A 101 MB sample of this checkpoint used only 30 of the 256 e4m3 codes
(entropy 3.464 bit/byte), which fits a 5-bit index into a 32-entry LUT --
lossless, and -37.5% of the scale bytes = -4.2% of all expert bytes. With the
static MoE kernel measured bandwidth-bound at 93-95% of the linear DRAM
ceiling (39차 §3e/§3g), removed bytes convert to time nearly 1:1.

This walks the checkpoint's safetensors headers and counts the alphabet of the
expert weight_scale tensors -- per tensor (what a per-tensor LUT would need)
and globally. A tensor over 32 codes cannot use a 5-bit index and would fall
back to 8 bit, so the number that matters is the fraction over 32.

Host CPU only (no GPU, no container), reads only the scale byte ranges:

    nice -n 19 python3 probes/nvfp4_scale_alphabet.py [--model PATH]
        [--experts-per-layer 4] [--all]
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np

_SCALE = re.compile(r"\.experts\.(\d+)\.(\w+)\.weight_scale$")


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/choiceoh/models/glm53-redhat-nvfp4")
    ap.add_argument("--experts-per-layer", type=int, default=4)
    ap.add_argument("--all", action="store_true", help="every expert (reads ~11% of the checkpoint)")
    ap.add_argument("--pack", action="store_true",
                    help="also pack every scale tensor with the overlay's 6-bit "
                         "block packer and check the roundtrip byte for byte")
    args = ap.parse_args()

    shards = sorted(p for p in os.listdir(args.model) if p.endswith(".safetensors"))
    if not shards:
        print(f"no safetensors under {args.model}", file=sys.stderr)
        return 2
    packer = None
    if args.pack:
        import importlib.util
        import torch
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(here, "..", "overlay", "modules", "glm53_moe", "moe_sf_pack.py")
        spec = importlib.util.spec_from_file_location("moe_sf_pack", src)
        packer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packer)
        pack_ok = {packer.SF_PACK_BLOCK: 0, packer.SF_PACK_BLOCK_FC2: 0}
        pack_bad = dict(pack_ok)
        pack_in = dict(pack_ok)
        pack_out = dict(pack_ok)
    counts = np.zeros(256, dtype=np.int64)
    per_tensor = []        # (name, distinct, bytes, lo, hi)
    tensor_codes = []      # (name, [codes])
    block_span = {}        # chunk bytes -> [per-block (max-min+1)]
    read_bytes = 0
    for shard in shards:
        path = os.path.join(args.model, shard)
        head, base = _header(path)
        picked = []
        for name, meta in head.items():
            if not isinstance(meta, dict) or "data_offsets" not in meta:
                continue
            m = _SCALE.search(name)
            if not m:
                continue
            if not args.all and int(m.group(1)) % max(1, 288 // args.experts_per_layer):
                continue
            picked.append((name, meta))
        picked.sort(key=lambda kv: kv[1]["data_offsets"][0])
        with open(path, "rb") as f:
            for name, meta in picked:
                a, b = meta["data_offsets"]
                f.seek(base + a)
                raw = np.frombuffer(f.read(b - a), dtype=np.uint8)
                c = np.bincount(raw, minlength=256)
                counts += c
                nz = np.nonzero(c)[0]
                per_tensor.append((name, int((c > 0).sum()), int(raw.size),
                                   int(nz[0]), int(nz[-1])))
                tensor_codes.append((name, [int(v) for v in nz]))
                # the kernel's own unit: one (128 rows x 512 k) SF block is
                # 4096 contiguous bytes (39차 §3b). A base per block, not per
                # tensor, is the natural packing unit -- the DMA already reads
                # exactly this much, so the base is one scalar per stage.
                for cs in (4096, 512):
                    n = raw.size // cs
                    if n:
                        blk = raw[: n * cs].reshape(n, cs)
                        lo = blk.min(axis=1).astype(np.int32)
                        hi = blk.max(axis=1).astype(np.int32)
                        block_span.setdefault(cs, []).append(hi - lo + 1)
                read_bytes += raw.size
                if packer is not None:
                    t = torch.from_numpy(raw.copy())
                    for blk in (packer.SF_PACK_BLOCK, packer.SF_PACK_BLOCK_FC2):
                        try:
                            packed, bases = packer.pack_sf(t, blk)
                        except ValueError as exc:
                            pack_bad[blk] += 1
                            if sum(pack_bad.values()) <= 2:
                                print(f"  REFUSED {name} shape {meta['shape']} "
                                      f"{raw.size} B block {blk}: {exc}", flush=True)
                            continue
                        if torch.equal(packer.unpack_sf(packed, bases, blk), t):
                            pack_ok[blk] += 1
                        else:
                            pack_bad[blk] += 1
                            print(f"  ROUNDTRIP MISMATCH {name} block {blk}", flush=True)
                        pack_in[blk] += t.numel()
                        pack_out[blk] += packed.numel() + bases.numel()
        print(f"  {shard}: {len(picked)} scale tensors, {read_bytes / 1e6:.0f} MB read",
              flush=True)

    if not per_tensor:
        print("no expert weight_scale tensors matched", file=sys.stderr)
        return 2
    used = np.nonzero(counts)[0]
    dist = np.array([d for _, d, _, _, _ in per_tensor])
    span = np.array([hi - lo + 1 for _, _, _, lo, hi in per_tensor])
    total = counts.sum()
    p = counts[used] / total
    entropy = float(-(p * np.log2(p)).sum())
    print(f"\ntensors {len(per_tensor)}, scale bytes read {total / 1e6:.0f} MB")
    print(f"global alphabet: {len(used)} of 256 codes, entropy {entropy:.3f} bit/byte")
    print(f"codes: {' '.join('0x%02x' % v for v in used)}")
    print(f"per-tensor distinct codes: min {dist.min()} median {int(np.median(dist))} "
          f"max {dist.max()}; over 32 codes: {int((dist > 32).sum())} of {len(dist)}")
    if dist.max() > 32:
        worst = max(per_tensor, key=lambda t: t[1])
        print(f"  widest: {worst[0]} with {worst[1]} codes")
    # base+offset (no LUT: e4m3 = base + index) needs the codes of a tensor to
    # be one contiguous run; the kernel side is then an add, not a table read
    print(f"per-tensor code SPAN (max-min+1): min {span.min()} median "
          f"{int(np.median(span))} max {span.max()}; fits 5 bit (<=32): "
          f"{int((span <= 32).sum())} of {len(span)}, 6 bit (<=64): "
          f"{int((span <= 64).sum())}, 7 bit: {int((span <= 128).sum())}")
    holes = span - dist
    print(f"  holes inside the span: median {int(np.median(holes))} max {int(holes.max())} "
          f"(0 = the alphabet is one solid run)")
    wide = [t for t in per_tensor if t[4] - t[3] + 1 > 32]
    for t in wide[:3]:
        print(f"  span>32: {t[0]} codes {t[1]} span {t[4] - t[3] + 1} "
              f"[0x{t[3]:02x}..0x{t[4]:02x}]")
    # what breaks the run? A zero scale (0x00 / 0x80 = -0, an all-zero block,
    # e.g. padding rows) is not part of the value band. If the band without
    # the zeros fits 32 codes, the kernel can spend index 0 on "zero" and read
    # every other scale as base + index -- an add and a select, no LUT.
    zeros = {0x00, 0x80}
    fit_plain = fit_zero = 0
    still = []
    for name, d, sz, lo, hi in per_tensor:
        pass
    for name, nzcodes in tensor_codes:
        band = sorted(c for c in nzcodes if c not in zeros)
        spanb = (band[-1] - band[0] + 1) if band else 0
        if spanb <= 32:
            fit_zero += 1
            if not (set(nzcodes) & zeros):
                fit_plain += 1
        else:
            still.append((name, len(band), spanb, band[:3], band[-3:]))
    for cs, chunks in sorted(block_span.items()):
        sp = np.concatenate(chunks)
        print(f"per-{cs} B block span: median {int(np.median(sp))} max {int(sp.max())}; "
              f"fits 5 bit (<=32): {100 * (sp <= 32).mean():.2f}% of {len(sp)} blocks, "
              f"6 bit: {100 * (sp <= 64).mean():.2f}%")
    print(f"band without zeros (0x00/0x80): fits 5 bit in {fit_zero} of "
          f"{len(tensor_codes)} tensors ({100 * fit_zero / len(tensor_codes):.1f}%), "
          f"of which {fit_plain} never use a zero scale at all")
    for t in still[:4]:
        print(f"  still wide: {t[0]} band {t[1]} codes span {t[2]} "
              f"lo {[hex(c) for c in t[3]]} hi {[hex(c) for c in t[4]]}")
    top = np.argsort(counts)[::-1][:8]
    print("most common: " + " ".join(f"0x{v:02x}:{counts[v] / total:.1%}" for v in top))
    bits = 1
    while (1 << bits) < len(used):
        bits += 1
    if packer is not None:
        for blk in sorted(pack_ok):
            if not pack_in[blk]:
                continue
            cut = (pack_in[blk] - pack_out[blk]) / pack_in[blk]
            print(f"\n6-bit pack, {blk} B blocks: {pack_ok[blk]} tensors exact, "
                  f"{pack_bad[blk]} refused; {pack_in[blk] / 1e6:.0f} MB -> "
                  f"{pack_out[blk] / 1e6:.0f} MB (scale bytes -{100 * cut:.2f}%, "
                  f"all packed expert bytes -{100 * cut / 9:.2f}%)")
    print(f"\nfixed-width index: {bits} bit -> scale bytes -{100 * (8 - bits) / 8:.1f}%, "
          f"all packed expert bytes -{100 * (8 - bits) / 8 / 9:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
