"""Qwen3.8's prefill before the door (profile): what GLM-5.3's boot runs in `_warmup_prefill_memory` and
`_warmup_serving_kernels` (engine/profiles/glm53/adapter.py), over this net.

Qwen3.8's fleet boot opened its door with no prefill ever run (2026-09-18). The first request paid every prefill kernel's
first-use compile -- the first fleet boot's MoE JIT storm sat exactly there, 75 artifacts on rank 0 in twelve minutes
before #1180 -- and the largest prefill chunk (one forward since #1183) had never been held to the allocator ceiling the
boot enforces (fleet.WORKSPACE_GIB, GLM-5.3's value, unmeasured for this model). So, after the runner and before the
capture, every rank runs the same passes:

    memory   the largest prefill chunk at context 0 with its prefix marks, through the target and the MTP head as a
             prompt runs them: a ledger row (base/runtime_memory -- every rank votes, the byte ceiling is checked) and a
             vote that the outputs are finite. The far end of the context is not run at full length: the one buffer
             there that grows with context, the QSA scores, is cut to 128 MiB of rows whatever the context
             (kernels/qsa._LOGITS_WORKSPACE_BYTES), so a far full chunk adds seconds, not bytes.
    kernels  a prefill of each of WIDTHS at context 0, and EDGE tokens ending on each context bucket's last position
             (net.bucket_blocks: the paged QSA kernels compile their table width in, one compile a rung), each through
             the target and the MTP head.

The inputs are zeros. This compiles kernels and qualifies memory; it does not judge output. The slot, its blocks and
every cache are released and reset afterwards, on failure too.
"""
from __future__ import annotations

import time

WIDTHS = (64, 512, 4095)
"""GLM-5.3's prompt widths above a decode step (its 1 and 8 are decode-sized, and an eager step of that size keys its
static MoE kernel by the routes the tokens take, which zeros cannot stand in for)."""
EDGE = 64


def rungs(block: int, capacity: int, top: int) -> "list[int]":
    """The last position (exclusive) of each context bucket a prefill can reach, smallest first: 4,096 * 2^i tokens in
    whole blocks (net.bucket_blocks), up to `capacity` tokens."""
    from engine.profiles.qwen38.net import FIRST_BUCKET
    edges, reach = [], FIRST_BUCKET
    while True:
        blocks = min(top, -(-min(reach, capacity) // block))
        end = min(blocks * block, capacity)
        if not edges or end > edges[-1]:
            edges.append(end)
        if end >= capacity or blocks >= top:
            return edges
        reach *= 2


def plan(F, *, chunk: int, capacity: int, top: int) -> "list[tuple[str, int, int]]":
    """(kind, tokens, context) in run order: the memory pass first (it sizes the allocator the rest reuse), then the
    widths, then the bucket edges above the first rung."""
    largest = min(chunk, capacity)
    passes = [("memory", largest, 0)]
    passes += [("kernels", w, 0) for w in WIDTHS if w < largest]
    passes += [("kernels", min(EDGE, end), end - min(EDGE, end)) for end in rungs(F.block, capacity, top)[1:]]
    return passes


def warmup(net, caches, *, memory, chunk: int, max_context: int, mtp: bool, seq: int = 0) -> dict:
    """Run `plan`'s passes on sequence `seq` -> {"<kind>/<tokens>/<context>": seconds}."""
    import torch
    from engine.profiles.qwen38.net import Segment, Step
    F = net.F
    if caches.pool.rows_in_use or any(owner >= 0 for owner in caches.slots.owner[1:]):
        raise ValueError("prefill warmup requires empty request and state slots")
    capacity = min(caches.pool.num_blocks * F.block, max_context)
    passes = plan(F, chunk=chunk, capacity=capacity, top=caches.block_table.shape[1])
    paid = {}
    slot = caches.slots.take(seq)
    try:
        caches.pool.reserve(seq, capacity)
        for kind, length, context in passes:
            name = f"prefill/{length}/{context}"
            caches.reset_slot(slot)
            began = time.perf_counter()
            error = None
            try:
                ids = torch.zeros(length, device=caches.device, dtype=torch.int64)
                # the memory pass carries a prompt's prefix boundaries (the snapshots a served chunk writes on its way)
                marks = tuple((p, i) for i, p in enumerate(range(F.block, length, F.block))
                              if i < caches.snapshots and net.takes_mark(p)) if kind == "memory" else ()
                step = Step(ids, (Segment(seq, slot, context, 0, length),), marks)
                caches.prepare(step)
                out, streams = net.forward(step, caches, streams=True)
                valid = torch.isfinite(net.head(out[-1:])).all() & torch.isfinite(streams).all()
                if mtp:
                    # a prompt's observation: the head over the next tokens with the target's streams (adapter.ServedMTP)
                    head = Step(ids, (Segment(seq, slot, context, 0, length),))
                    hidden, _ = net.mtp_forward(head, streams, caches, last_hidden_only=True)
                    valid &= torch.isfinite(net.head(hidden)).all()
                bad = (~valid).to(torch.int32).reshape(1)
                del out, streams, valid, step, ids
            except Exception as exc:                       # noqa: BLE001 -- cast into the vote, re-raised below
                error, bad = exc, torch.ones(1, dtype=torch.int32, device=caches.device)
            # every rank reaches this collective whatever happened on its way, so a rank that raised stops its peers
            # here instead of leaving them in the next pass's collectives until NCCL's deadline
            if int(net.comm.all_reduce_max(bad).item()):
                if error is not None:
                    raise error
                raise FloatingPointError(f"{name}: non-finite output in the prefill warmup (this rank or a peer)")
            if memory is not None:
                memory.checkpoint(f"{name}/{kind}", release_cache=True)
            elif str(caches.device).startswith("cuda"):
                torch.cuda.synchronize()
            paid[f"{kind}/{length}/{context}"] = round(time.perf_counter() - began, 3)
    finally:
        caches.pool.release(seq)
        caches.slots.give(slot)
        caches.reset()
    return paid


__all__ = ["WIDTHS", "EDGE", "rungs", "plan", "warmup"]
