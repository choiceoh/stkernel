"""What DSv4.1 actually caches, and therefore what a KV budget buys.

`budget.py` ends at "KV = 8.76 GiB". That is not an answer an operator can use.
The answer is a context length at a concurrency, and getting there needs to
know that DSv4.1 does not have "a KV cache" -- it has five buffers with four
different length rules, and only FOUR of its forty layers produce KV at all
(`kv_source_layer_ids = [2, 8, 14, 20]`; that is the CED split).

The rules are read off the reference's `__init__`s, not guessed:

  window_kv_cache     model.py:664  [B, window_size, head_dim]        every block
  compress_kv_cache   model.py:670  [B, S // ratio, head_dim]         kv_source only
  Indexer.k_cache     model.py:521  [B, S // ratio, index_head_dim]   kv_source AND index_source
  Compressor states   model.py:453  [B, ratio, head_dim] fp32, x2     kv_source, ratio > 1
  freqs_cis           model.py:697  [S, rope_head_dim // 2] complex64 every block, NO batch term

Two of these surprise:

  freqs_cis is per BLOCK and full length. 43 blocks x S x 256 B is 1.34 GiB at
  128K and 10.75 GiB at the config's 1,048,576 -- with no batch term at all, so
  it is pure overhead that a shared table would erase.

  none of it divides by world_size. MLA's compressed KV is shared across heads,
  so every rank holds the WHOLE cache. `head_dim` here is 512 (the latent), not
  n_heads * per_head, which is why that is affordable.

The reference is pinned; a vendor change to these buffers fails rather than
drifts, the same contract tp_plan.py holds convert.py to.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

GIB = 1 << 30
REFERENCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"


@dataclass(frozen=True)
class Cache:
    name: str
    blocks: int          # how many layers carry one
    per_batch_per_token: float   # bytes per sequence per token of context
    per_batch: float             # bytes per sequence, independent of context
    per_token: float             # bytes per token of context, independent of batch
    note: str


def _cfg(repo: Path) -> dict:
    return json.loads((repo / "config.json").read_text())["text_config"]


def caches(repo: "str | Path") -> "list[Cache]":
    repo = Path(repo)
    reference = repo / "inference" / "model.py"
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    if REFERENCE_SHA256 != "REPLACED_AT_WRITE_TIME" and digest != REFERENCE_SHA256:
        raise ValueError(f"model.py is not the pinned reference (sha256 {digest})")

    cfg = _cfg(repo)
    ratios = cfg["compress_ratios"]
    blocks = len(ratios)                       # backbone + MTP, one Attention each
    head_dim = cfg["head_dim"]
    index_head_dim = cfg["index_head_dim"]
    window = cfg["sliding_window"]
    rope_half = cfg["qk_rope_head_dim"] // 2
    kv_src = cfg["kv_source_layer_ids"]
    idx_src = set(cfg["index_source_layer_ids"])

    bf16 = 2
    # compress_kv and indexer_k both walk S // ratio, so the per-token cost is
    # the sum of 1/ratio over the layers that carry them.
    kv_inv = sum(1.0 / ratios[i] for i in kv_src if ratios[i])
    idx_inv = sum(1.0 / ratios[i] for i in kv_src if ratios[i] and i in idx_src)
    state_layers = [i for i in kv_src if ratios[i] > 1]

    return [
        Cache("window_kv", blocks, 0.0, blocks * window * head_dim * bf16, 0.0,
              f"[B, {window}, {head_dim}] bf16 on every block"),
        Cache("compress_kv", len(kv_src), kv_inv * head_dim * bf16, 0.0, 0.0,
              f"[B, S/ratio, {head_dim}] bf16 on layers {kv_src}"),
        Cache("indexer_k", len(kv_src & idx_src) if isinstance(kv_src, set)
              else len([i for i in kv_src if i in idx_src]),
              idx_inv * index_head_dim * bf16, 0.0, 0.0,
              f"[B, S/ratio, {index_head_dim}] bf16 where a layer owns its keys"),
        Cache("compressor_state", len(state_layers), 0.0,
              sum(ratios[i] for i in state_layers) * head_dim * 4 * 2, 0.0,
              "[B, ratio, head_dim] fp32 x2 where ratio > 1"),
        Cache("freqs_cis", blocks, 0.0, 0.0, blocks * rope_half * 8,
              f"[S, {rope_half}] complex64 on every block -- NO batch term"),
    ]


def freqs_variants(repo: "str | Path") -> "tuple[int, int]":
    """(distinct tables, blocks holding one).

    Attention.__init__ picks the rope arguments with a single branch --
    `if self.compress_ratio:` -> (original_seq_len, compress_rope_theta), else
    (0, rope_theta). Everything else fed to precompute_freqs_cis is a constant.
    So the 43 tables take exactly TWO distinct values and the rest is copies.
    """
    ratios = _cfg(Path(repo))["compress_ratios"]
    return len({bool(r) for r in ratios}), len(ratios)


def shared_freqs(cs: "list[Cache]", repo: "str | Path") -> "list[Cache]":
    """The same caches, with one freqs_cis table per distinct rope config."""
    distinct, blocks = freqs_variants(repo)
    out = []
    for c in cs:
        if c.name == "freqs_cis":
            c = Cache(c.name, distinct, c.per_batch_per_token, c.per_batch,
                      c.per_token * distinct / blocks,
                      f"{distinct} distinct tables instead of {blocks} copies")
        out.append(c)
    return out


def total_bytes(cs: "list[Cache]", batch: int, seq: int) -> float:
    return sum(c.per_batch_per_token * batch * seq + c.per_batch * batch
               + c.per_token * seq for c in cs)


def max_seq(cs: "list[Cache]", budget_gib: float, batch: int) -> int:
    """Longest context that fits, at this concurrency."""
    per_token = sum(c.per_batch_per_token * batch + c.per_token for c in cs)
    fixed = sum(c.per_batch * batch for c in cs)
    if per_token <= 0:
        return 0
    return int((budget_gib * GIB - fixed) / per_token)


def report(repo: "str | Path", budget_gib: float,
           batches=(1, 2, 4, 8, 16, 32), at_seq: int = 131072) -> str:
    cs = caches(repo)
    ceiling = _cfg(Path(repo))["max_position_embeddings"]
    out = ["  per-cache cost, one rank (nothing divides by world_size: MLA's",
           "  compressed KV is shared across heads, so every rank holds it all)",
           ""]
    width = max(len(c.name) for c in cs)
    for c in cs:
        at = (c.per_batch_per_token * at_seq + c.per_batch + c.per_token * at_seq / 1)
        out.append(f"  {c.name:<{width}}  {c.blocks:>3} blocks  "
                   f"{at / GIB:>7.3f} GiB @ B=1 S={at_seq // 1024}K   {c.note}")
    out.append("")
    out.append(f"  total @ B=1 S={at_seq // 1024}K: "
               f"{total_bytes(cs, 1, at_seq) / GIB:.3f} GiB")
    out.append("")
    shared = shared_freqs(cs, repo)
    distinct, blocks = freqs_variants(repo)
    out.append(f"  lever: freqs_cis takes exactly {distinct} distinct values across "
               f"{blocks} blocks (Attention.__init__ branches once, on "
               "`if self.compress_ratio:`), so 41 of them are copies.")
    out.append(f"  sharing them: {cs[-1].per_token * at_seq / GIB:.3f} -> "
               f"{shared[-1].per_token * at_seq / GIB:.3f} GiB at S={at_seq // 1024}K")
    out.append("")
    out.append(f"  what {budget_gib:.2f} GiB of KV buys "
               f"(model ceiling {ceiling:,} tok):")
    out.append(f"    {'concurrency':>12}  {'as written':>16}  {'freqs shared':>16}")
    for b in batches:
        a = min(max_seq(cs, budget_gib, b), ceiling)
        c = min(max_seq(shared, budget_gib, b), ceiling)
        mark = "  (ceiling)" if c >= ceiling else ""
        out.append(f"    {b:>12}  {a:>11,} tok  {c:>11,} tok{mark}")
    return "\n".join(out)


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="/home/choiceoh/models/DeepSeek-V4.1-Flash")
    parser.add_argument("--kv-gib", type=float, default=8.76)
    args = parser.parse_args(argv)
    print(report(args.repo, args.kv_gib))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
