"""What `sparse_attn` requires of the indices it is handed, and why.

The kernel is TileLang and cannot be run here -- tilelang is not installed on
this fleet -- so this is not a port of it. It is the contract its source states,
written down on the side that CAN be checked: whatever produces `topk_idxs`.

Three clauses, read off `inference/kernel.py`'s `sparse_attn_kernel`:

  DTYPE     `topk_idxs: T.Tensor[(b, m, topk), INT32]`. Not int64.

  SENTINEL  -1, and only -1. The kernel tests `idxs[i] != -1` in two places --
            it zeroes the gathered KV row and sets that score to -inf -- and
            pads its own tail with -1 when `t * block + i >= topk`. Any other
            negative value is not a sentinel to it; it is an index.

  IN RANGE  and this is the one nothing else checks. The gather is

                kv_shared[i, j] = if(idxs[i] != -1, kv[by, idxs[i], j], 0)

            with NO upper bound. An id at or past `kv`'s length is an
            out-of-bounds device read, not a masked one. The reference's
            `torch.where(idxs < compress_lens, idxs + offset, -1)` is what
            keeps that from happening, and note that the bound is checked
            BEFORE `offset` is added -- so a caller that offsets into a cache
            shorter than compress_lens + offset defeats it.

One consequence is worth stating because it looks like a bug and is not: a row
whose ids are ALL -1 is legal and must produce a zero output rather than NaN.
The kernel gets that by seeding its running max with -1e30 instead of -inf,
which its own comment calls out: "a row with no valid index (all -1) would
otherwise produce exp(-inf - (-inf)) = NaN". So an indexer that emits an
all-(-1) row for an unreachable query is correct, and one that emits some other
filler to avoid it is not.

probes/dsv41_sparse_contract.py holds the producers -- the reference's
`Indexer.forward` tail and this repo's `select_candidate_ids` -- to all three
clauses, and pins the kernel source these clauses were read from so that a
vendor change to it fails rather than drifts.
"""

from __future__ import annotations

SENTINEL = -1


class ContractViolation(ValueError):
    """Raised with the clause, because "bad indices" is not actionable."""


def check_topk_idxs(idxs, kv_len: int, *, where: str = "topk_idxs") -> None:
    """Every clause, on one produced tensor. Cheap enough to leave armed.

    `kv_len` is the number of rows `sparse_attn` will be given, which for the
    compressed path is positions AFTER compression -- not tokens. Passing a
    token count here is the mistake the range clause exists to catch, and it
    would pass a check written against tokens.
    """
    import torch

    if idxs.dtype is not torch.int32:
        raise ContractViolation(
            f"{where}: dtype {idxs.dtype}, the kernel declares INT32")
    bad = idxs[(idxs < 0) & (idxs != SENTINEL)]
    if bad.numel():
        raise ContractViolation(
            f"{where}: {bad.numel()} negative values that are not {SENTINEL}, "
            f"e.g. {int(bad.flatten()[0])}. The kernel treats only -1 as a "
            f"sentinel; anything else it uses as an index.")
    over = idxs[idxs >= kv_len]
    if over.numel():
        raise ContractViolation(
            f"{where}: {over.numel()} ids at or past the {kv_len} rows of kv, "
            f"e.g. {int(over.flatten()[0])}. The gather has no upper bound -- "
            f"this is an out-of-bounds device read, not a masked one.")


def rows_all_sentinel(idxs):
    """Rows the kernel will answer with zeros. Legal; report, do not reject."""
    return (idxs == SENTINEL).all(dim=-1)
