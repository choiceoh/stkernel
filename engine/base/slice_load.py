#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Put a slice of the real checkpoint on one GPU, instrumented. Seconds, no fleet.

This is the engine's first brick and its cheapest debugging tool at the same
time. 40차 could not ask "what shape is this layer's scale tensor" without a
4-8 minute four-node boot and a fleet turn; here it is a command.

    python3 engine/slice_load.py --layers 0-2
    python3 engine/slice_load.py --layers 3 --device cuda --json /tmp/slice.json

`--layers 3` is the first sparse-attention layer on GLM-5.3 (3, 7, ... 43); the
dense ones before it are small enough to land instantly, which makes them the
right place to check plumbing before paying for an expert-heavy layer.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint import Checkpoint            # noqa: E402
from instruments import Recorder             # noqa: E402

DEFAULT_MODEL = "/home/choiceoh/models/glm53-redhat-nvfp4"


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(piece))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", default="0", help="e.g. 0  |  0-2  |  3,7")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--shared", action="store_true", help="also load the non-layer tensors")
    ap.add_argument("--json", default="")
    ap.add_argument("--show", type=int, default=8, help="how many tensors to print")
    args = ap.parse_args()

    rec = Recorder("slice_load")
    layers = parse_layers(args.layers)
    with rec.phase("index"):
        ckpt = Checkpoint(args.model)
        summary = ckpt.summary()
        rec.gauge("tensors", summary["tensors"])
        rec.gauge("shards", summary["shards"])
        rec.gauge("layers", summary["layers"])
    print(f"checkpoint: {summary['layers']} layers, {summary['tensors']:,} tensors, "
          f"{summary['shards']} shards, prefix {summary['prefix']!r}")

    if args.device.startswith("cuda"):
        # Create the context in its OWN phase. Otherwise it is created inside
        # the load, and the load's memory delta reads empty -- the instruments
        # refuse to sample before is_initialized(), precisely so that sampling
        # never creates the boundary it is measuring.
        with rec.phase("cuda-init") as span:
            import torch

            torch.cuda.init()
            torch.empty(1, device=args.device)
            free, total = torch.cuda.mem_get_info()
            rec.gauge("dev_free_GiB", round(free / (1 << 30), 2))
            rec.gauge("dev_total_GiB", round(total / (1 << 30), 2))

    with rec.phase("select"):
        keys = ckpt.keys_for(layers, include_shared=args.shared)
        rec.gauge("selected", len(keys))
    print(f"slice: layers {layers} -> {len(keys):,} tensors"
          + (" (+ shared)" if args.shared else ""))

    with rec.phase(f"load:{args.device}"):
        tensors = ckpt.load(keys, device=args.device, recorder=rec)

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"loaded: {len(tensors):,} tensors, {total / (1 << 30):.3f} GiB on {args.device}")
    for name in list(tensors)[: args.show]:
        t = tensors[name]
        print(f"  {name:<58} {str(tuple(t.shape)):<20} {str(t.dtype).replace('torch.', '')}")
    if len(tensors) > args.show:
        print(f"  ... {len(tensors) - args.show:,} more")

    print()
    print(rec.table())
    if args.json:
        rec.dump(args.json)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
