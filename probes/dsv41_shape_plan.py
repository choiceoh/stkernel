#!/usr/bin/env python3
"""Does the tensor plan predict all 96,085 tensors? No GPU, no weights loaded.

`dsv41_shapes.tensor_plan` reads config.json and produces a name -> (dtype,
shape) map. This checks it against the checkpoint's safetensors headers: the
name sets must be equal, and every shared name must agree on dtype and shape.

Shapes, not just names. A name-only check would pass a plan that had
`wq_b` as [q_lora, heads*head_dim] instead of the other way round, or that gave
the shared expert a routed expert's scale rank -- both load, and both compute
nonsense.

    python3 probes/dsv41_shape_plan.py [--index ...] [--config ...]

Headers are read from local shards where they exist and range-fetched where
they do not, so this runs before a download finishes.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_model"))

from dsv41_shapes import tensor_plan  # noqa: E402

HF = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/"
REPO = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash")


def load_json(path, name):
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    if (REPO / name).is_file():
        return json.loads((REPO / name).read_text())
    return json.loads(urllib.request.urlopen(HF + "raw/main/" + name,
                                             timeout=180).read())


def header(shard: str):
    local = REPO / shard
    if local.is_file():
        with open(local, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(n))
    url = HF + "resolve/main/" + shard
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req, timeout=60).read())[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    return json.loads(urllib.request.urlopen(req, timeout=240).read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--index")
    args = ap.parse_args()

    full = load_json(args.config, "config.json")
    plan = tensor_plan(full["text_config"], full.get("vision_config"))
    weight_map = load_json(args.index, "model.safetensors.index.json")["weight_map"]

    actual = {}
    for shard in sorted(set(weight_map.values())):
        for name, meta in header(shard).items():
            if name != "__metadata__":
                actual[name] = (meta["dtype"], list(meta["shape"]))

    print(f"  plan {len(plan):,} tensors   checkpoint {len(actual):,}")
    missing = sorted(set(actual) - set(plan))
    extra = sorted(set(plan) - set(actual))

    def summarize(names):
        return collections.Counter(re.sub(r"\.\d+\.", ".N.", n) for n in names)

    fail = 0
    if missing:
        fail = 1
        print(f"  FAIL: {len(missing)} in the checkpoint the plan does not "
              f"produce:")
        for pat, n in summarize(missing).most_common(6):
            print(f"      {n:6d}x {pat}  {actual[missing[0]] if n else ''}")
            ex = next(m for m in missing if re.sub(r'\.\d+\.', '.N.', m) == pat)
            print(f"             e.g. {ex} {actual[ex]}")
    if extra:
        fail = 1
        print(f"  FAIL: {len(extra)} the plan produces that do not exist:")
        for pat, n in summarize(extra).most_common(6):
            ex = next(m for m in extra if re.sub(r'\.\d+\.', '.N.', m) == pat)
            print(f"      {n:6d}x {pat}  e.g. {ex} {plan[ex]}")

    bad = [(n, plan[n], actual[n]) for n in sorted(set(plan) & set(actual))
           if list(plan[n][1]) != list(actual[n][1]) or plan[n][0] != actual[n][0]]
    if bad:
        fail = 1
        print(f"  FAIL: {len(bad)} disagree on dtype or shape:")
        for pat, n in summarize(x[0] for x in bad).most_common(8):
            ex = next(b for b in bad if re.sub(r"\.\d+\.", ".N.", b[0]) == pat)
            print(f"      {n:6d}x {pat}")
            print(f"             plan {ex[1]}  checkpoint {ex[2]}")
    if fail:
        return 1
    kinds = collections.Counter(d for d, _ in plan.values())
    print(f"  MATCH: every name, dtype and shape "
          f"({', '.join(f'{k} {v:,}' for k, v in kinds.most_common())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
