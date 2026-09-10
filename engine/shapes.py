"""The batch shapes DSv4.1's kernels support, as data.

D2: the scheduler builds batches only out of shapes the kernels already accept.
On GLM-5.3 that rule was learned the expensive way -- it took several holds to
find out why a prefill chunk was 6,912, and the answer turned out to be three
side effects deep: `floor((MAX_BATCHED - draft_slots) / 2304) * 2304`, where
2304 was a Mamba block size nobody had written down as a constraint. Raising
`MAX_BATCHED` to 9,216 did not move it, because it missed by five tokens.

So this file exists before the scheduler does. Every constraint carries where
it came from, and nothing here is a preference -- a shape that violates one of
these does not run slower, it computes the wrong thing or reads out of bounds.

Sources, in descending order of how badly a violation bites:

  config.json                      facts about the architecture
  dsv41_sparse_contract.py         what the TileLang kernel requires of indices
  dsv41_window.py                  the ring invariant `p % window_size`
  overlay dsv41_* module defaults  tile bounds the existing code already holds
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path


@dataclass(frozen=True)
class Constraint:
    name: str
    value: object
    source: str
    bites: str          # what goes wrong when a shape violates it


def constraints(repo: "str | Path") -> "list[Constraint]":
    cfg = json.loads((Path(repo) / "config.json").read_text())["text_config"]
    ratios = [r for r in cfg["compress_ratios"] if r > 1]
    group = 1
    for r in ratios:
        group = group * r // gcd(group, r)
    block = cfg["candidate_block_size"]
    return [
        Constraint(
            "compressor group", group, "config.compress_ratios",
            "a chunk that ends mid-group leaves the tail in Compressor.kv_state "
            "and the next chunk pools across a boundary the reference never does"),
        Constraint(
            "candidate block", block, "config.candidate_block_size",
            "candidate selection scores whole blocks of compressed positions; "
            "a partial block at the end is scored against padding"),
        Constraint(
            "chunk alignment", group * block, "the two above",
            "the raw-token multiple that leaves neither a partial compressor "
            "group nor a partial candidate block"),
        Constraint(
            "sliding window", cfg["sliding_window"], "config.sliding_window",
            "prefill seeds the ring from the LAST `window` tokens with a "
            "rotated two-part copy; a chunk shorter than the window seeds "
            "fewer slots than a decode step will read (dsv41_window.py)"),
        Constraint(
            "index topk", cfg["index_topk"], "config.index_topk",
            "the indexer keeps this many compressed positions per query"),
        Constraint(
            "candidate domain", block * cfg["candidate_topk_blocks"],
            "candidate_block_size x candidate_topk_blocks",
            "compressed positions the candidate stage narrows to before top-k"),
        Constraint(
            "topk dtype", "int32", "dsv41_sparse_contract.py",
            "the TileLang kernel declares T.Tensor[(b, m, topk), INT32]"),
        Constraint(
            "topk sentinel", -1, "dsv41_sparse_contract.py",
            "-1 and ONLY -1; any other negative value is an index to the "
            "kernel, and its gather has no upper bound at all"),
        Constraint(
            "packed index tile", 262144, "dsv41_packed_index.py::_MAX_TILE_VALUES",
            "the packed E2M1 score path restores BF16 in tiles of this many "
            "values; a larger tile is an unbounded temporary"),
    ]


def chunk_for(repo: "str | Path", token_budget: int, draft_slots: int = 0) -> int:
    """The largest legal prefill chunk inside a token budget.

    This is DSv4.1's version of GLM's floor((MAX_BATCHED - K) / 2304) * 2304.
    Written as one function so that the answer is never three side effects deep
    again -- if a chunk is not what you expected, print this.
    """
    align = next(c.value for c in constraints(repo) if c.name == "chunk alignment")
    usable = token_budget - draft_slots
    if usable < align:
        return 0
    return (usable // align) * align


def report(repo: "str | Path", budgets=(2048, 4096, 8192, 9216, 16384)) -> str:
    cs = constraints(repo)
    width = max(len(c.name) for c in cs)
    out = ["  constraints -- a shape that breaks one of these computes the wrong",
           "  thing or reads out of bounds. None of them is a preference.", ""]
    for c in cs:
        out.append(f"  {c.name:<{width}}  {str(c.value):>8}   [{c.source}]")
        out.append(f"  {'':<{width}}            {c.bites}")
    align = next(c.value for c in cs if c.name == "chunk alignment")
    out.append("")
    out.append(f"  legal prefill chunks (alignment {align}):")
    out.append(f"    {'token budget':>13}  {'no drafter':>11}  {'with 3 MTP slots':>18}")
    for b in budgets:
        out.append(f"    {b:>13}  {chunk_for(repo, b):>11,}  "
                   f"{chunk_for(repo, b, 3):>18,}")
    return "\n".join(out)


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="/home/choiceoh/models/DeepSeek-V4.1-Flash")
    args = parser.parse_args(argv)
    print(report(args.repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
