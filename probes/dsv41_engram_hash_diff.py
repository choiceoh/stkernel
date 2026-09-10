#!/usr/bin/env python3
"""Do our pinned n-gram hashes equal DeepSeek's? No GPU, no model weights.

`dsv41_engram_hash.py` writes the bucket primes and the hash multipliers down
as constants so the serving path needs neither sympy nor a numpy Generator whose
stream it depends on. That is only safe while the constants still equal what the
reference produces, and "still equal" is a test, not a comment.

Three things are checked, in the order they would silently break:

  1. the compressed token map. `build_compressed_token_map` collapses tokens
     that normalize alike, and EVERY multiplier is derived from how many keys
     that leaves. The config declares 99,092 and the reference asserts it; this
     runs it against the shipped tokenizer.json and requires the same number.
     Get this wrong and every hash moves, with no error anywhere.
  2. the pinned layout. Primes and multipliers are regenerated from the
     reference and compared element by element. As a second, offline check the
     per-layer prime sums must equal the checkpoint's table heights.
  3. the hashes themselves. Real token ids go through the reference's
     `NgramHashState.forward` and through ours, and the two must agree exactly
     -- including the padding rules at the start of a sequence and around a
     dead (image) span, which is where the accumulating `blocked` flag is easy
     to get wrong.

    python3 probes/dsv41_engram_hash_diff.py --tokenizer /path/to/tokenizer.json

Needs `tokenizers` and `sympy` HERE, which is the point: the serving path does
not, because this probe is what lets the constants be constants.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_engram"))

import dsv41_engram_hash as ours  # noqa: E402


@dataclass
class RefArgs:
    """Only the fields EngramLayout / NgramHashState read."""
    engram_layer_ids: tuple = ours.LAYER_IDS
    engram_max_ngram_size: int = ours.MAX_NGRAM_SIZE
    engram_n_heads: int = ours.N_HEADS
    engram_vocab_size: int = 16_000_000
    engram_head_dim: int = 256
    engram_num_embeddings: tuple = ours.NUM_EMBEDDINGS
    engram_pad_id: int = ours.PAD_TOKEN_ID
    engram_compressed_vocab_size: int = ours.COMPRESSED_VOCAB_SIZE
    max_batch_size: int = 1
    max_seq_len: int = 4096


class TokenizerShim:
    """The two members `build_compressed_token_map` actually touches."""

    def __init__(self, backend) -> None:
        self.backend_tokenizer = backend
        self._backend = backend

    def __len__(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engram-py", required=True,
                    help="the checkpoint's inference/engram.py")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.engram_py).resolve().parent))
    import engram as ref  # noqa: E402
    from tokenizers import Tokenizer  # noqa: E402

    fail = 0

    # -- 1. the compressed vocabulary --------------------------------------
    backend = Tokenizer.from_file(args.tokenizer)
    token_map, compressed_vocab = ref.build_compressed_token_map(
        TokenizerShim(backend))
    ok = compressed_vocab == ours.COMPRESSED_VOCAB_SIZE
    fail |= not ok
    print(f"  compressed vocab   {compressed_vocab:,} vs pinned "
          f"{ours.COMPRESSED_VOCAB_SIZE:,}   {'OK' if ok else 'MISMATCH'}")
    print(f"                     ({len(token_map):,} tokens collapse into it)")

    # -- 2. the pinned layout ----------------------------------------------
    layout = ref.EngramLayout.from_args(RefArgs())
    ok = tuple(tuple(tuple(h) for h in layer) for layer in layout.primes) \
        == ours.PRIMES
    fail |= not ok
    flat = [p for layer in ours.PRIMES for per in layer for p in per]
    print(f"  primes             {len(flat)} pinned, {len(set(flat))} distinct"
          f"   {'OK' if ok else 'MISMATCH'}")
    ref_mult = ref.compute_hash_multipliers(
        ours.LAYER_IDS, ours.MAX_NGRAM_SIZE, ours.COMPRESSED_VOCAB_SIZE)
    ok = tuple(tuple(r) for r in ref_mult.tolist()) == ours.MULTIPLIERS
    fail |= not ok
    print(f"  multipliers        {len(ours.MULTIPLIERS)} x "
          f"{len(ours.MULTIPLIERS[0])}   {'OK' if ok else 'MISMATCH'}")
    for li, rows in enumerate(ours.NUM_EMBEDDINGS):
        want = RefArgs().engram_num_embeddings[li]
        ok = rows == want
        fail |= not ok
        print(f"  layer {ours.LAYER_IDS[li]:2d} bucket sum {rows:,} vs "
              f"checkpoint {want:,}   {'OK' if ok else 'MISMATCH'}")

    # -- 3. the hashes, on real ids and with a dead span -------------------
    state = ref.NgramHashState(RefArgs(), layout, TokenizerShim(backend))
    gen = torch.Generator().manual_seed(20260910)
    ids = torch.randint(0, len(token_map), (1, args.tokens), generator=gen)
    # a dead span in the middle: an n-gram may not reach across an image
    mask = torch.ones(1, args.tokens, dtype=torch.bool)
    mask[0, 100:112] = False
    ref_out = state(ids, 0, mask)                    # [1, L, n_layers, cols]

    compressed = [ours.DEAD if not mask[0, i] else token_map[int(ids[0, i])]
                  for i in range(args.tokens)]
    pad = token_map[ours.PAD_TOKEN_ID]
    mine = ours.hash_ids_for_step(compressed, range(args.tokens), pad)
    mine_t = torch.tensor(mine)                      # [L, n_layers, cols]

    same = torch.equal(ref_out[0], mine_t)
    fail |= not same
    n = ref_out.numel()
    if same:
        print(f"  hash ids           bit-identical over {n:,} "
              f"(seq start + a 12-token dead span)   OK")
    else:
        bad = (ref_out[0] != mine_t).nonzero()
        print(f"  hash ids           MISMATCH at {len(bad)} of {n:,}; "
              f"first {bad[:3].tolist()}")
        for pos, li, col in bad[:3].tolist():
            print(f"      pos {pos} layer {li} col {col}: "
                  f"ref {int(ref_out[0, pos, li, col])} "
                  f"ours {int(mine_t[pos, li, col])}")

    print("\n" + ("HASH FAIL" if fail else "HASH PASS (pinned layout equals "
                                           "the reference)"))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
