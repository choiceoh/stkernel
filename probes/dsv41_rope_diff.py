#!/usr/bin/env python3
"""Do our rotary tables and rotation equal DeepSeek's? No GPU, no weights.

Rope is where a wrong answer looks most like a right one: no error, no NaN,
just positions that are off by a smooth amount. Three things are checked
against the reference, each exact:

  1. the tables. YaRN on (a compressing layer, compress_rope_theta) and YaRN
     off (compress_ratio 0, base rope_theta) -- this model builds both, and
     `LayerPlan.rope` is what picks. A layer handed the other one still runs.
  2. the rotation, on [b, s, d] and [b, s, h, d].
  3. the INVERSE. The reference removes the query's rotation from the output so
     the cache can hold one shared rotated form; skipping it leaves every
     output rotated by its own position. Checked both against the reference and
     as a round trip.

    python3 probes/dsv41_rope_diff.py --model-py .../inference/model.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_model"))

from dsv41_layers import plan_layers  # noqa: E402
from dsv41_rope import apply_rotary_emb, precompute_freqs_cis  # noqa: E402


def load_reference(model_py: Path):
    text = model_py.read_text()
    parts = []
    for start, end in (("@lru_cache(2)\ndef precompute_freqs_cis", "def apply_rotary_emb"),
                       ("def apply_rotary_emb", "class Compressor(nn.Module):")):
        assert text.count(start) == 1, start
        i = text.index(start)
        parts.append(text[i:text.index(end, i)])
    # lru_cache is supplied rather than stripped: the reference caches these
    # tables and removing the decorator would be editing what is being graded.
    from functools import lru_cache

    src = "\n".join(parts)
    ns = {"torch": torch, "math": math, "lru_cache": lru_cache}
    exec(compile(src, str(model_py), "exec"), ns)
    return ns["precompute_freqs_cis"], ns["apply_rotary_emb"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-py", required=True)
    ap.add_argument("--config")
    ap.add_argument("--rope-dim", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=4096)
    args = ap.parse_args()

    ref_freqs, ref_rot = load_reference(Path(args.model_py))
    if args.config and Path(args.config).is_file():
        cfg = json.loads(Path(args.config).read_text())["text_config"]
    else:
        cfg = json.loads(urllib.request.urlopen(
            "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/"
            "config.json", timeout=180).read())["text_config"]
    scaling = cfg["rope_scaling"]
    fail = 0

    # -- 0. the SELECTION itself. Everything below feeds the reference the
    #    parameters `rope()` returns, so if those are wrong both sides are
    #    wrong together and agree -- the same blind spot a verify that re-reads
    #    the source through its own offsets has. So check the choice first,
    #    against the rule in the reference's Attention.__init__:
    #        if self.compress_ratio: original_seq_len, compress_rope_theta
    #        else:                   0,                 rope_theta
    want_orig = int(scaling["original_max_position_embeddings"])
    branches = set()
    for p in plan_layers(cfg):
        got = p.rope(cfg)
        want = ((0, float(cfg["rope_theta"])) if p.compress_ratio == 0
                else (want_orig, float(cfg["compress_rope_theta"])))
        branches.add(p.compress_ratio == 0)
        if got != want:
            print(f"  FAIL: layer {p.index} (compress_ratio "
                  f"{p.compress_ratio}) chose {got}, the reference's rule "
                  f"gives {want}")
            fail = 1
    if len(branches) != 2:
        print("  FAIL: only one branch of the rope choice occurs, so checking "
              "the choice proves nothing")
        fail = 1
    if float(cfg["rope_theta"]) == float(cfg["compress_rope_theta"]):
        print("  FAIL: the two thetas are equal in this config; the choice is "
              "unobservable")
        fail = 1
    if not fail:
        print(f"  choose  every layer takes the reference's branch "
              f"(theta {cfg['rope_theta']:g} / {cfg['compress_rope_theta']:g}, "
              f"both branches occur)")

    # -- 1. both tables, chosen the way the layer plan chooses -------------
    seen = {}
    for p in plan_layers(cfg):
        seen.setdefault(p.rope(cfg), []).append(p.index)
    for (orig, theta), layers in seen.items():
        ours = precompute_freqs_cis(args.rope_dim, args.seqlen, orig, theta,
                                    scaling["factor"], scaling["beta_fast"],
                                    scaling["beta_slow"])
        ref = ref_freqs(args.rope_dim, args.seqlen, orig, theta,
                        scaling["factor"], scaling["beta_fast"],
                        scaling["beta_slow"])
        ok = torch.equal(ours, ref)
        fail |= not ok
        kind = "YaRN off (pure SWA)" if orig == 0 else "YaRN on"
        print(f"  table   {kind:20s} theta {theta:>7g}  layers "
              f"{layers[0]}..{layers[-1]:<3d} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            d = (ours - ref).abs()
            print(f"      max {d.max().item():.3g} at dim "
                  f"{d.max(0).values.argmax().item()}")

    # the two tables must actually differ, or agreeing about them is empty
    keys = list(seen)
    if len(keys) == 2:
        a = precompute_freqs_cis(args.rope_dim, args.seqlen, keys[0][0],
                                 keys[0][1], scaling["factor"],
                                 scaling["beta_fast"], scaling["beta_slow"])
        b = precompute_freqs_cis(args.rope_dim, args.seqlen, keys[1][0],
                                 keys[1][1], scaling["factor"],
                                 scaling["beta_fast"], scaling["beta_slow"])
        differ = not torch.equal(a, b)
        fail |= not differ
        print(f"  table   the two are different: {'yes' if differ else 'NO -- '
              'then picking between them proves nothing'}")

    # -- 2 and 3. the rotation, both ranks, forward and inverse ------------
    freqs = precompute_freqs_cis(args.rope_dim, 128,
                                 scaling["original_max_position_embeddings"],
                                 cfg["compress_rope_theta"], scaling["factor"],
                                 scaling["beta_fast"], scaling["beta_slow"])
    gen = torch.Generator().manual_seed(20260910)
    for shape, label in (((2, 128, args.rope_dim), "[b, s, d]"),
                         ((2, 128, 4, args.rope_dim), "[b, s, h, d]")):
        for inverse in (False, True):
            x = torch.randn(*shape, generator=gen, dtype=torch.float32)
            a, b = x.clone(), x.clone()
            ref_rot(a, freqs[:128], inverse)
            apply_rotary_emb(b, freqs[:128], inverse)
            ok = torch.equal(a, b)
            fail |= not ok
            print(f"  rotate  {label:14s} inverse={str(inverse):5s} "
                  f"{'OK' if ok else 'MISMATCH'}")

        # round trip: rotating and un-rotating must return the input
        x = torch.randn(*shape, generator=gen, dtype=torch.float32)
        y = x.clone()
        apply_rotary_emb(y, freqs[:128], False)
        moved = not torch.allclose(y, x, atol=1e-6)
        apply_rotary_emb(y, freqs[:128], True)
        back = torch.allclose(y, x, atol=1e-5)
        fail |= not (moved and back)
        print(f"  rotate  {label:14s} round trip    "
              f"{'OK' if moved and back else 'MISMATCH'}"
              f"   (moved {moved}, returned {back})")

    print("\n" + ("ROPE FAIL" if fail else "ROPE PASS"))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
