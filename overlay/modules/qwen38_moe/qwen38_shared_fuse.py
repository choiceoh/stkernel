"""Fuse Qwen3.8-Flash-Next's shared expert into the routed grouped GEMM.

## Why

The shared expert has the SAME shape as a routed one -- `moe_intermediate_size`
and `shared_expert_intermediate_size` are both 640 -- so a grouped GEMM can host
it as one more group instead of a separate dense GEMM.

That is not only a launch saved. It is what keeps the shared expert ALIGNED:

    shared expert, TP-split      640 / 4 = 160  -> 320 gate+up rows, 320 % 128 = 64
    shared expert, fused slot    640            -> 1280 rows,       1280 % 128 = 0

Expert parallelism already gives the routed experts the second row; fusing hands
the shared expert the same one. The first row is not a performance question on
this fleet -- padding a misaligned FP4 intermediate is recorded as booting and
then DESTROYING the model ('안녕하세요' -> '1'), so avoiding the misalignment is
the only safe move.

## The one hard constraint

**The fused slot is RANK-LOCAL and must never enter the all-to-all.**

Every token uses the shared expert. Routing it as a real expert id would send
the entire batch to whichever rank owns that id, which is the opposite of what
expert parallelism is for. So the shared expert is REPLICATED on every rank
(about 150 MiB per rank across 48 layers at NVFP4) and addressed by a sentinel
id that dispatch resolves locally.

`-1` is already "no expert" in this stack's sparse contracts, so the shared slot
is `-2`: distinguishable from both a real id and from emptiness, and impossible
to confuse with a rank's local index.

## The one ordering trap

`norm_topk_prob` is true: the routed weights are normalised ACROSS THE TOP-K.
The shared gate is not part of that distribution -- it is an independent sigmoid.
Appending the slot before normalisation would renormalise the routed weights
against a value that does not belong to them, changing every routed weight in
the batch while leaving every shape correct. `fuse_routing` therefore takes
weights that are already normalised, and says so.
"""

from __future__ import annotations

import torch

# The sentinel id for the fused shared slot in the GLOBAL expert id space.
# -1 is "no expert" in this stack's contracts (see dsv41_sparse_contract), so
# the shared slot takes -2 rather than an id a rank could mistake for its own.
SHARED_SLOT_ID = -2
NO_EXPERT_ID = -1


def local_shared_index(num_local_experts: int) -> int:
    """Where the replicated shared expert sits in a rank's expert table."""
    return int(num_local_experts)


def fuse_routing(topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                 shared_gate: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Append the shared slot to an ALREADY-NORMALISED routing decision.

    topk_ids     [T, K] int32, global expert ids (-1 = none)
    topk_weights [T, K] float, normalised across K when norm_topk_prob is set
    shared_gate  [T] or [T, 1] float, the shared expert's sigmoid gate

    Returns [T, K+1] pairs with the shared slot last. The caller must not
    renormalise afterwards: the gate is an independent scale, not a share of
    the routed distribution.
    """
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"ids {tuple(topk_ids.shape)} and weights "
            f"{tuple(topk_weights.shape)} must line up; one row of each per "
            f"token per slot.")
    gate = shared_gate.reshape(topk_weights.shape[0], 1).to(topk_weights.dtype)
    ids = torch.cat(
        [topk_ids,
         torch.full((topk_ids.shape[0], 1), SHARED_SLOT_ID,
                    dtype=topk_ids.dtype, device=topk_ids.device)], dim=1)
    weights = torch.cat([topk_weights, gate], dim=1)
    return ids, weights


def dispatch_ids(fused_ids: torch.Tensor) -> torch.Tensor:
    """The ids the all-to-all may see: the shared slot removed.

    Returned as a COPY with the shared slot replaced by `NO_EXPERT_ID`, so a
    dispatch that forgets to call this sends the whole batch to one rank and a
    dispatch that calls it twice is unchanged.
    """
    out = fused_ids.clone()
    out[out == SHARED_SLOT_ID] = NO_EXPERT_ID
    return out


def local_ids(fused_ids: torch.Tensor, *, rank: int, num_local_experts: int,
              num_experts: int) -> torch.Tensor:
    """Global ids -> this rank's expert-table indices, shared slot included.

    Ids owned by another rank become `NO_EXPERT_ID`. The shared slot becomes
    the replicated local index on EVERY rank, which is the whole point.
    """
    base = rank * num_local_experts
    out = torch.full_like(fused_ids, NO_EXPERT_ID)
    mine = (fused_ids >= base) & (fused_ids < base + num_local_experts)
    out[mine] = fused_ids[mine] - base
    out[fused_ids == SHARED_SLOT_ID] = local_shared_index(num_local_experts)
    if int((fused_ids >= num_experts).sum()):
        raise ValueError(
            f"an expert id is >= num_experts ({num_experts}); the fused slot "
            f"must be {SHARED_SLOT_ID}, not an id past the table.")
    return out


def fuse_expert_weights(w13: torch.Tensor, w2: torch.Tensor,
                        shared_w13: torch.Tensor,
                        shared_w2: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Append the replicated shared expert to a rank's stacked expert weights.

    w13 [E, 2*I, H], w2 [E, H, I] -> [E+1, ...] with the shared expert last.
    The shared tensors keep their FULL intermediate: not splitting them is
    what makes 1280 gate+up rows instead of 320.
    """
    if shared_w13.shape != w13.shape[1:]:
        raise ValueError(
            f"shared w13 {tuple(shared_w13.shape)} must match a routed "
            f"expert's {tuple(w13.shape[1:])}. Fusing is only possible because "
            f"moe_intermediate_size and shared_expert_intermediate_size are "
            f"both 640; a different shared size needs its own GEMM.")
    if shared_w2.shape != w2.shape[1:]:
        raise ValueError(
            f"shared w2 {tuple(shared_w2.shape)} must match "
            f"{tuple(w2.shape[1:])}")
    return (torch.cat([w13, shared_w13.unsqueeze(0)], dim=0),
            torch.cat([w2, shared_w2.unsqueeze(0)], dim=0))


def check_fused_routing(fused_ids: torch.Tensor, *, num_experts: int,
                        where: str = "") -> None:
    """Refuse a routing tensor the fused lane cannot mean.

    Exactly one shared slot per token, no id outside the table, and the shared
    slot in the LAST column -- a grouped GEMM reads the columns positionally,
    and a shared slot that has drifted is a routed expert given the shared
    gate's weight.
    """
    tag = f" [{where}]" if where else ""
    if fused_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"fused ids must be integral, got {fused_ids.dtype}{tag}")
    per_token = (fused_ids == SHARED_SLOT_ID).sum(dim=-1)
    if int((per_token != 1).sum()):
        bad = int((per_token != 1).nonzero()[0][0])
        raise ValueError(
            f"token {bad} has {int(per_token[bad])} shared slots, expected "
            f"exactly 1{tag}")
    if not bool((fused_ids[:, -1] == SHARED_SLOT_ID).all()):
        raise ValueError(f"the shared slot must be the last column{tag}")
    body = fused_ids[:, :-1]
    if int(((body >= num_experts) | (body < NO_EXPERT_ID)).sum()):
        raise ValueError(
            f"a routed id is outside [0, {num_experts}) and is not "
            f"{NO_EXPERT_ID}{tag}")
