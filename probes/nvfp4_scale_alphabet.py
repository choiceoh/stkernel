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
    args = ap.parse_args()

    shards = sorted(p for p in os.listdir(args.model) if p.endswith(".safetensors"))
    if not shards:
        print(f"no safetensors under {args.model}", file=sys.stderr)
        return 2
    counts = np.zeros(256, dtype=np.int64)
    per_tensor = []        # (name, distinct, bytes)
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
                per_tensor.append((name, int((c > 0).sum()), int(raw.size)))
                read_bytes += raw.size
        print(f"  {shard}: {len(picked)} scale tensors, {read_bytes / 1e6:.0f} MB read",
              flush=True)

    if not per_tensor:
        print("no expert weight_scale tensors matched", file=sys.stderr)
        return 2
    used = np.nonzero(counts)[0]
    dist = np.array([d for _, d, _ in per_tensor])
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
    top = np.argsort(counts)[::-1][:8]
    print("most common: " + " ".join(f"0x{v:02x}:{counts[v] / total:.1%}" for v in top))
    bits = 1
    while (1 << bits) < len(used):
        bits += 1
    print(f"\nfixed-width index: {bits} bit -> scale bytes -{100 * (8 - bits) / 8:.1f}%, "
          f"all packed expert bytes -{100 * (8 - bits) / 8 / 9:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
