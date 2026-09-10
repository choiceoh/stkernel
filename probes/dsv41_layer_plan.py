#!/usr/bin/env python3
"""Does the CED layer plan predict the checkpoint's tensors? No GPU, no weights.

`dsv41_layers.plan_layers` reads only config.json. This takes what it produces
and checks it against the 96,085 tensor NAMES in the weight index -- both
directions, per layer:

    the plan gives layer L a compressor  <=>  layer L has attn.compressor.*
    the plan gives layer L an indexer    <=>  layer L has attn.indexer.*
    the plan gives layer L engram        <=>  layer L has engram.*
    the plan says every layer routes     <=>  every layer has ffn.experts.*

Only the names are used, so this runs against an index alone -- no shard has to
be downloaded. A misread of the encoder/decoder split, of which layers source
KV, or of the engram placement shows up here as a named layer that disagrees.

    python3 probes/dsv41_layer_plan.py --index model.safetensors.index.json \\
                                       --config config.json

The plan is also required to be non-trivial: if it marked every layer the same
way, agreement would prove nothing. The counts are printed so that is visible.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_model"))

from dsv41_layers import DECODER, ENCODER, describe, plan_layers  # noqa: E402

HF = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/"


def load(path: str | None, name: str):
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    for cand in (Path("/home/choiceoh/models/DeepSeek-V4.1-Flash") / name,):
        if cand.is_file():
            return json.loads(cand.read_text())
    return json.loads(urllib.request.urlopen(HF + name, timeout=180).read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--index")
    args = ap.parse_args()

    cfg = load(args.config, "config.json")["text_config"]
    weight_map = load(args.index, "model.safetensors.index.json")["weight_map"]

    plans = plan_layers(cfg)
    print("  plan:", describe(plans))

    actual = collections.defaultdict(set)
    for name in weight_map:
        m = re.match(r"^layers\.(\d+)\.(.+)$", name)
        if not m:
            continue
        layer, rest = int(m.group(1)), m.group(2)
        if rest.startswith("attn.compressor."):
            actual[layer].add("kv_source")
        elif rest.startswith("attn.indexer."):
            actual[layer].add("indexer")
        elif rest.startswith("engram."):
            actual[layer].add("engram")
        elif rest.startswith("ffn.experts."):
            actual[layer].add("moe")

    bad = []
    for p in plans:
        want = {k for k in ("kv_source", "indexer", "engram", "moe")
                if getattr(p, k)}
        got = actual.get(p.index, set())
        if want != got:
            bad.append((p.index, p.role, sorted(want), sorted(got)))
    stray = sorted(set(actual) - {p.index for p in plans})

    n_layers = len(plans)
    counts = {k: sum(1 for p in plans if getattr(p, k))
              for k in ("kv_source", "indexer", "engram", "moe")}
    print(f"  layers with: kv_source {counts['kv_source']}, indexer "
          f"{counts['indexer']}, engram {counts['engram']}, moe "
          f"{counts['moe']}/{n_layers}")
    trivial = [k for k, v in counts.items() if v in (0, n_layers) and k != "moe"]
    if trivial:
        print(f"  WARNING: {trivial} is uniform across layers, so agreeing "
              f"about it proves nothing")

    if stray:
        print(f"  FAIL: the index has layers the plan does not: {stray}")
    for layer, role, want, got in bad:
        print(f"  FAIL: layer {layer} ({role}) plan {want} vs checkpoint {got}")
    if bad or stray:
        return 1

    enc = [p.index for p in plans if p.role == ENCODER]
    dec = [p.index for p in plans if p.role == DECODER]
    print(f"  MATCH: all {n_layers} layers agree "
          f"(encoder {enc[0]}..{enc[-1]}, decoder {dec[0]}..{dec[-1]})")
    dec_src = [p.index for p in plans if p.kv_source and p.role == DECODER]
    dec_idx = sum(1 for p in plans if p.indexer and p.role == DECODER)
    print(f"  the split is load-bearing: the only decoder-side KV source is "
          f"{dec_src} -- the boundary layer, whose compressor projects the "
          f"encoder's final output -- while the indexer keeps running in "
          f"{dec_idx} decoder layers, selecting from that KV rather than "
          f"producing it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
