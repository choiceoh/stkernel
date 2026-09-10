"""The n-gram hash the engram tables are indexed by, with the layout pinned.

`EngramLayout` derives its bucket ranges by walking primes upward from
`engram_vocab_size - 1` with sympy, and `compute_hash_multipliers` draws its
multipliers from `np.random.default_rng(10007 * layer_id)`. Both are
deterministic, both are cheap to get wrong, and neither belongs in a serving
path -- sympy is not in the image and a silent change in either produces hashes
that index a different table entirely, with no error anywhere.

So they are computed once and written down here. Two things make that safe
rather than a copy waiting to rot:

  1. `probes/dsv41_engram_hash_diff.py` regenerates both from the reference and
     requires them to match these constants exactly, so drift is a test
     failure rather than a quality regression nobody can source.
  2. The primes are checkable against the checkpoint without running anything:
     each layer's 24 bucket sizes must SUM to that layer's table height, and
     they do --

         layer  1   sum 384,006,168 == engram_num_embeddings[0]
         layer 14   sum 384,016,682 == engram_num_embeddings[1]

     with no prime reused across the 48. A table built on different primes
     would not have those heights.

The multipliers depend on the COMPRESSED vocabulary size, not the tokenizer's:
`multiplier_bound = (int64_max // 99092) // 2`. That 99,092 is
`engram_compressed_vocab_size` in the config, and the reference asserts it
against what `build_compressed_token_map` derives from the shipped tokenizer --
"a mismatch there would silently rehash the whole table". Checked on the real
tokenizer.json: 129,280 tokens collapse to exactly 99,092 keys.
"""

from __future__ import annotations

# engram_layer_ids (1, 14); (engram_max_ngram_size 4 - 1) n-gram sizes;
# engram_n_heads 8. [layer][n-gram size][head] -> bucket modulus.
PRIMES = (
    (
        (16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183),
        (16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321),
        (16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463),
    ),
    (
        (16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627),
        (16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769),
        (16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889),
    ),
)

# [layer][lookback], odd by construction and bounded so that
# `token_id * multiplier` cannot overflow int64.
MULTIPLIERS = (
    (76632096046245, 4839876093313, 35959672319349, 73987337458391),
    (67716810739261, 51510806800915, 30921347202721, 82619226485591),
)

LAYER_IDS = (1, 14)
MAX_NGRAM_SIZE = 4
N_HEADS = 8
COMPRESSED_VOCAB_SIZE = 99_092
PAD_TOKEN_ID = 2
# The row each (n-gram size, head) bucket range starts at, per layer: the
# running sum of the ranges before it. The reference computes this as
# `cumsum([0, *sizes[:-1]])` over the flattened per-layer primes.
OFFSETS = tuple(
    tuple(__import__("itertools").accumulate(
        [0] + [p for per in layer for p in per][:-1]))
    for layer in PRIMES
)
NUM_EMBEDDINGS = tuple(sum(p for per in layer for p in per) for layer in PRIMES)


def hash_ids(compressed, layer_index: int):
    """Hash ids for one position's lookback window, one table.

    `compressed` is the window [id_at_pos, id_at_pos-1, ... ] already mapped
    through the compressed token map and already padded, MAX_NGRAM_SIZE long,
    newest first -- the same order the reference stacks it in.

    Returns MAX_NGRAM_SIZE - 1 groups of N_HEADS ids, flattened: the running
    XOR after step i is the hash of the (i+1)-gram, and each lands in its own
    prime-sized range.
    """
    if len(compressed) != MAX_NGRAM_SIZE:
        raise ValueError(
            f"need a {MAX_NGRAM_SIZE}-token window, got {len(compressed)}")
    mult = MULTIPLIERS[layer_index]
    offs = OFFSETS[layer_index]
    products = [compressed[i] * mult[i] for i in range(MAX_NGRAM_SIZE)]
    rolling = products[0]
    out = []
    for i in range(1, MAX_NGRAM_SIZE):
        rolling ^= products[i]
        for head in range(N_HEADS):
            col = (i - 1) * N_HEADS + head
            out.append(rolling % PRIMES[layer_index][i - 1][head] + offs[col])
    return out


DEAD = -1   # a position that takes no part in an n-gram (an image span)


def window(compressed, pos: int, pad_compressed: int):
    """The MAX_NGRAM_SIZE lookback at `pos`, newest first, already padded.

    `compressed` is the whole sequence mapped through the compressed token map,
    with DEAD at positions that are not text. Two things stop the walk, and once
    either stops it the rest of the window is padding too -- an n-gram may not
    span the start of the sequence and may not span a dead token:

        blocked |= (pos < shift) | (source is DEAD)

    which is the reference's rule, including that `blocked` accumulates rather
    than being tested per shift. Getting that wrong produces n-grams that reach
    across an image and hash to rows trained on text that never followed.
    """
    out, blocked = [], False
    for shift in range(MAX_NGRAM_SIZE):
        source = compressed[max(pos - shift, 0)]
        blocked = blocked or pos < shift or source == DEAD
        out.append(pad_compressed if blocked else source)
    return out


def hash_ids_for_step(compressed, positions, pad_compressed):
    """[position][layer][col] for a whole step, before any layer runs.

    This is the prefetch premise made concrete: nothing here reads a hidden
    state, an activation or a KV entry -- only token ids and their positions --
    so every row the step will want from the two tables is addressable at layer
    0. What layer 1 cannot cover with one layer of compute (4.19 ms p95, see
    dsv41_engram.py) it covers by being handed the DSpark draft's ids a step
    early, which this same function accepts without knowing the difference.
    """
    return [[hash_ids(window(compressed, pos, pad_compressed), li)
             for li in range(len(LAYER_IDS))]
            for pos in positions]
