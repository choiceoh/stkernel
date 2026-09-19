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
    head     the MTP head alone over 1..K+1 positions (`head` = K+1), at context 0 and ending on each bucket's last
             position: what adapter.ServedMTP runs eagerly for a row whose reservation does not hold the draft replay's
             positions -- its observation (up to K+1), then each chain position (1). Its QSA layer compiles the bucket's
             table width and the step's size in, and no prefill width is decode-sized, so on 2026-09-19 those kernels
             were first built inside requests. Zeros in, the drafter's own pick out (`draft_tokens`).
    eager    the eager MoE's decode-sized launches (`eager_moe`): an uncaptured step -- the MTP head observing a parked
             row's positions, a step no graph admits -- dispatches only this rank's (token, route) pairs, one route a
             pair, and up to EAGER_PAIRS of them run the micro kernel, keyed by the pair count AND the capacity of the
             workspace the dispatcher has grown so far. Nothing above reaches that path, so the first requests of the
             2026-09-19 K=3 window compiled six of them mid-request (the first step 6.8 s), their capacities (r2, r4)
             set by the order the pair counts arrived in. `eager_moe` grows the workspace to its ceiling first and then
             runs every count: eight kernels, one capacity, whatever order the requests bring.

The inputs are zeros. This compiles kernels and qualifies memory; it does not judge output. The slot, its blocks and
every cache are released and reset afterwards, on failure too.
"""
from __future__ import annotations

import time

WIDTHS = (1, 8, 64, 512, 4095)
"""GLM-5.3's prompt widths. 1 and 8 were left out while an eager step of that size keyed its static MoE kernel by the
routes its tokens take -- token 0's routes, here, which are not a prompt's. `eager_moe` now builds every such kernel at
the one capacity the workspace keeps, and the boot runs it before these passes, so whatever pair count token 0 gives a
rank reads one of those kernels (up to 8 pairs) or the dynamic kernel, free of the count (more)."""
EDGE = 64
EAGER_PAIRS = 8
"""= engine/kernels/b12x/moe_dispatch._MICRO_MAX_TOKENS: an eager launch of more one-route pairs than this over every
local expert runs the dynamic kernel, whose artifact is free of the count (select_sm120_moe_backend)."""


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


def plan(F, *, chunk: int, capacity: int, top: int, head: int = 0) -> "list[tuple[str, int, int]]":
    """(kind, tokens, context) in run order: the memory pass first (it sizes the allocator the rest reuse), then the
    widths, then the bucket edges above the first rung, then -- with `head` (K+1) -- the MTP head alone over 1..head
    positions at context 0 and ending on every rung's last position."""
    largest = min(chunk, capacity)
    edges = rungs(F.block, capacity, top)
    passes = [("memory", largest, 0)]
    passes += [("kernels", w, 0) for w in WIDTHS if w < largest]
    passes += [("kernels", min(EDGE, end), end - min(EDGE, end)) for end in edges[1:]]
    for t in range(1, head + 1):
        passes += [("head", t, 0)] + [("head", t, end - t) for end in edges if end - t > 0]
    return passes


def warmup(net, caches, *, memory, chunk: int, max_context: int, mtp: bool, head: int = 0, seq: int = 0) -> dict:
    """Run `plan`'s passes on sequence `seq` -> {"<kind>/<tokens>/<context>": seconds}. `head`: K+1, the drafter's
    widest eager observation (0, or no `mtp`: no head passes)."""
    import torch
    from engine.profiles.qwen38.net import Segment, Step
    F = net.F
    if caches.pool.rows_in_use or any(owner >= 0 for owner in caches.slots.owner[1:]):
        raise ValueError("prefill warmup requires empty request and state slots")
    capacity = min(caches.pool.num_blocks * F.block, max_context)
    passes = plan(F, chunk=chunk, capacity=capacity, top=caches.block_table.shape[1], head=head if mtp else 0)
    paid = {}
    slot = caches.slots.take(seq)
    try:
        caches.pool.reserve(seq, capacity)
        for kind, length, context in passes:
            name = f"{'head' if kind == 'head' else 'prefill'}/{length}/{context}"
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
                if kind == "head":
                    # a parked row's eager head (adapter.ServedMTP._head): the target's streams it would be given are
                    # zeros here, and its pick is the drafter's own
                    given = torch.zeros(length, F.hc * F.hidden, device=caches.device, dtype=torch.bfloat16)
                    hidden, _ = net.mtp_forward(step, given, caches, last_hidden_only=True)
                    net.draft_tokens(hidden)
                    valid = torch.isfinite(hidden).all()
                    del hidden, given
                else:
                    out, streams = net.forward(step, caches, streams=True)
                    valid = torch.isfinite(net.head(out[-1:])).all() & torch.isfinite(streams).all()
                    if mtp:
                        # a prompt's observation: the head over the next tokens with the target's streams (ServedMTP)
                        head = Step(ids, (Segment(seq, slot, context, 0, length),))
                        hidden, _ = net.mtp_forward(head, streams, caches, last_hidden_only=True)
                        valid &= torch.isfinite(net.head(hidden)).all()
                    del out, streams
                bad = (~valid).to(torch.int32).reshape(1)
                del valid, step, ids
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


def eager_counts(pairs: int = EAGER_PAIRS) -> "list[int]":
    """The pair counts `eager_moe` launches, in order: the ceiling first -- the workspace grows to it and never again,
    so every count after it (and every request's) is keyed by that one capacity -- then each count below it."""
    return [pairs] + list(range(1, pairs))


def eager_routes(pairs: int, *, first_expert: int, local: int, experts: int, topk: int):
    """ids [pairs, topk] int32 and weights [pairs, topk] fp32, the router's types, where each row's first route is one
    of this rank's experts (a different one a row) and the others are another rank's: the compact path keeps exactly
    `pairs` pairs."""
    import torch
    rows = torch.arange(pairs, dtype=torch.int64)
    ids = torch.empty(pairs, topk, dtype=torch.int64)
    ids[:, 0] = first_expert + rows % local
    for r in range(1, topk):                       # outside [first_expert, first_expert + local), distinct in a row
        ids[:, r] = (first_expert + local + (rows + r - 1) % (experts - local)) % experts
    return ids.to(torch.int32), torch.full((pairs, topk), 1.0 / topk, dtype=torch.float32)


def eager_moe(net, *, pairs: int = EAGER_PAIRS) -> dict:
    """Every decode-sized eager MoE launch this rank can make, once, before the door (the module docstring's `eager`):
    through one target layer's experts -- they share the MTP head's workspace and kernels (same experts, hidden and
    width) -- or the MTP head's when the net has no target layer. Zeros for x: the kernel is keyed by shapes. The MTP
    head on FP8 experts takes another path and is not this one's. -> {"eager/<pairs>": seconds}. A rank that raised
    stops its peers at the vote, as the prefill passes do."""
    import torch
    F = net.F
    prefix = next((p for p in net._experts if not p.startswith("mtp.")), None)
    if prefix is None and getattr(net, "mtp_experts", "nvfp4") == "nvfp4" and "mtp.L0." in net._experts:
        prefix = "mtp.L0."
    if prefix is None:
        return {}
    w13 = net.p[prefix + "moe.w13"]
    local, device = w13.shape[0], w13.device               # this rank's experts, as lanes.moe counts them
    paid, error = {}, None
    try:
        for m in eager_counts(pairs):
            began = time.perf_counter()
            ids, weights = eager_routes(m, first_expert=net.first_expert, local=local, experts=F.experts,
                                        topk=F.topk_experts)
            x = torch.zeros(m, F.hidden, dtype=torch.bfloat16, device=device)
            net._experts[prefix](x, ids.to(device), weights.to(device), compact=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            paid[f"eager/{m}"] = round(time.perf_counter() - began, 3)
        bad = torch.zeros(1, dtype=torch.int32, device=device)
    except Exception as exc:                           # noqa: BLE001 -- cast into the vote, re-raised below
        error, bad = exc, torch.ones(1, dtype=torch.int32, device=device)
    if int(net.comm.all_reduce_max(bad).item()):
        if error is not None:
            raise error
        raise RuntimeError("the eager MoE warmup failed on a peer")
    return paid


__all__ = ["WIDTHS", "EDGE", "EAGER_PAIRS", "rungs", "plan", "warmup", "eager_counts", "eager_routes", "eager_moe"]
