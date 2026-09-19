"""The kernel lanes Qwen3.8-Flash-Next runs on (profile), bound two ways -- the shape wizard's work plan, wired.

    reference()   engine/modules' torch forms, the oracles each lane is judged against
    served()      engine/kernels, all or nothing: a lane that will not import raises and the boot dies (D3)

The wizard (python3 -m engine.base.kernel_shape wizard --profile qwen38) judged this shape's lanes and ordered the work;
this table is where its wire items land (engine/kernels/cells.py names the serving kernels):

    dense         the projections are engine/kernels/dense lanes bound in net.bind (DenseLinear, PaddedDenseLinear for
                  the shared expert's 160-column down projection); not a table entry, as in GLM-5.3's profile
    kda_chunk     gdn_chunk: chunk_kda_with_decay over the decay `gdn_gates` computes (engine/kernels/gdn)
    kda_ring      gdn_ring / gdn_ring_rows: recurrent_gdn_ring(_rows), GDN's gate computed in the ring kernel
    mhc_decode    hc_*: engine/kernels/gated_residual -- the gated residual in five launches a site (three for a
    mhc_prefill     decode step's rows: mix_rows), not the dozen of the composed form the wizard's recipe named (its
                    "fused kernel when launches matter"); a leave is its TP sum's programmatic dependent (LEAVES)
    mla           qsa_attend: the BF16-KV sparse paged GQA ported with the QSA ops (engine/kernels/qsa), the kernel
                  that served this model in the vLLM stack, instead of glue.gqa's one-scale e4m3 latent
    indexer       qsa_compress / qsa_store / qsa_select: engine/kernels/qsa
    moe           b12x's NVFP4 dispatcher over this rank's 128 experts (b12x refuses EP as such): global expert ids are
                  remapped to local ones; an eager step dispatches only this rank's pairs, a captured step keeps every
                  route and runs another rank's on local expert 0 with weight 0

The wizard's four measurements (the MoE tile at 128 local experts, the recurrent tile at 4/12 heads, one-shot and the
prefill collectives at hidden 2560) are GPU tickets; nothing here claims them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Lanes:
    name: str
    # hyper-connections (engine/kernels/gated_residual)
    hc_norm: object         # (h [N, hc*H], w [hc*H], eps, hc) -> normed [N, hc*H]; hc 1: one unit-offset norm over the row
    hc_leave: object        # (h, out [N, H], inject [N, hc], hc) -> h, in place
    hc_leave_norm: object   # (h, out, inject, w, eps, hc, *, prefetch=None) -> (h in place, normed); `prefetch`: the
                            #  weight the site's mixer reads next, which a served leave may pull into L2 while the sum
                            #  before it waits (served(leave=...), carry H4); a lane that does not prefetch ignores it
    hc_mix: object          # (normed, down_inject [r(+hc), hc*H], up [hc*H, r], hc, *, inject, project_down=None,
                            #  project_up=None) -> (mixed [N, H], inject [N, hc] | None); the projections replace the
                            #  BF16 matmuls when a quantised lane serves the mixer (net.Qwen38Net hc_fp8); without
                            #  them a decode step's rows take the served lane's two-launch fold (gated_residual.mix_rows)
    # GatedDeltaNet (engine/kernels/gdn, engine/kernels/kda, the causal conv kernels)
    gdn_gates: object       # (a [N, HV], b [N, HV], A_log f32, dt_bias f32, *, sigmoid_beta) -> (decay f32 [N, HV], beta [N, HV])
    gdn_chunk: object       # (q, k [1, T, Hk, D], v [1, T, HV, D], decay f32 [1, T, HV], beta [1, T, HV] sigmoided,
                            #  state0 [1, HV, K, V] f32 | None, states_at=None) -> (o [1, T, HV, D], state [1, HV, K, V] f32
                            #  [, states [n, HV, K, V] at the starts of the named 64-token kernel chunks])
    gdn_ring: object        # (q, k, v, a [1, T, HV], b_raw [1, T, HV], A_log f32, dt_bias f32, ring [slots, R, HV, K, V] f32, slot,
                            #  context) -> o; GDN's decay and beta computed with the recurrence; writes every token's state
    gdn_ring_rows: object   # the same over every row of a captured decode step: inputs [1, rows*T, ...], slots and contexts [rows]
    gdn_norm: object        # (core [N, HV, D], z [N, HV, D], w [D], eps) -> [N, HV*D]: GDN's rounding, sigmoid gate
    conv_prefill: object    # (x [T, C] bf16, w [C, K] f32, state [C, K-1] | None) -> (y [T, C], state' [C, K-1])
    conv_ring: object       # (x [T, C], w, ring [slots, C, R], slot, context) -> y; writes raw inputs, T <= min(8, R)
    conv_ring_rows: object  # (x [rows*T, C], w, ring, slots [rows], contexts [rows]) -> y
    # gated GQA attention with QSA selection (engine/kernels/qsa)
    norm_rope: object       # (x [N, h, D], w [D], eps, positions [N] i64, theta, rotary_dim) -> [N, h, D]
    qsa_store: object       # (cache [pages, page, 1, D], flat slots [N] (-1 skipped), rows [N, D]) -> None
    qsa_compress: object    # qsa_compress_groups_with_ratio(...) -> (pooled [N, 1, D], first positions [N, 3] i64)
    qsa_select: object      # qsa_select_paged_blocks(iq, key cache, page table, token_to_req, positions, lengths, topk, ratio,
                            #  *, group) -> the chosen blocks int32 [N, topk / ratio]; `group`: the rows come in runs
                            #  of that many of one request, scored from one read of each key tile (carry Q8)
    qsa_attend: object      # qsa_sparse_paged_attention_blocks(q [N, Hq, D], k, v caches [pages, page, Hkv, D], blocks,
                            #  positions, lengths, ratio, topk, table, token_to_req, *, gate): the blocks expanded inside
                            #  its tiles, the output gate applied in its final store
    # MoE
    route: object           # (logits [N, E], k) -> (ids int32 [N, k] global, weights f32 [N, k]): softmax fp32, top-k, renormalised
    moe: object             # (x [N, H] bf16, ids [N, k] global, weights [N, k] f32, w13, w13_sf, w2, w2_sf, *, scales,
                            #  first_expert, compact, local=False) -> [N, H] bf16: this rank's routed partial; `compact`
                            #  (an eager step) runs only this rank's pairs, reading their count on the host; `local`:
                            #  the routes are route_local's, already this rank's
    route_local: object = None      # (scores [N, >= E] BF16/FP32, k, *, experts, first_expert, w13, hidden) -> (ids int32
                                    #  [N, k] local, weights f32 [N, k]): a captured step's router and its EP remap
                                    #  (local_routes, with the launch shape's sentinel) in one launch; None: the layer
                                    #  composes route and moe
    moe_prepare: object = None      # (w13, w13_sf, w2, w2_sf, top_k, *, scales) -> views, once per bound layer before capture
    graph_resources: object = None  # () -> workspace owners to retain until the captured graphs close
    swiglu: object = None           # (fused [N, 2I], pad_to=None) -> silu(gate) * up: the shared expert's activation,
                                    #  with zero columns to `pad_to` for its padded down projection
    moe_finish: object = None       # (routed [N, H] bf16, shared [N, H] bf16, gate [N, 1] f32) -> BF16(f32 routed +
                                    #  f32 shared * gate): the MoE output before its all-reduce
    qsa_index_keys: object = None   # qsa.qsa_index_keys(ik [N, Di], ring, slot_table, token_to_req, starts, positions,
                                    #  key_slots, ratio, idx_k_norm, eps, theta, rotary_dim, index key cache): compress,
                                    #  norm, rope and store the keys the step's rows close, in one launch
    qsa_inputs: object = None       # qsa.qsa_inputs(q, k, v, iq, ik, positions, q_norm, k_norm, iq_norm, eps, theta,
                                    #  rotary_dim, K, V, kv_slots, ring, ring_slots) -> (q, iq): the layer's norms and
                                    #  rotations with the K/V and ring stores, in one launch
    qsa_attend_covered: object = None  # qsa.qsa_covered_paged_attention(q, k, v caches, positions, lengths, ratio, topk,
                                    #  table, token_to_req, *, gate, group): qsa_attend for a step the budget covers,
                                    #  without blocks -- a dense causal launch, a run of `group` rows of one request
                                    #  sharing each K/V tile; the same bytes (carry Q10). None: such a step attends its
                                    #  unscored ids through qsa_attend
    qsa_select_alike: object = None  # qsa.shards_select_alike(rows, shards, columns, topk / ratio, group) -> bool: whether
                                    #  qsa_select over disjoint row ranges of a step (their row counts) chooses row for
                                    #  row what one call does. The selection lane's own statement -- it picks its
                                    #  selector by the rows it is handed; None: no lane has said, and no net splits
    moe_rows: object = None         # (x [N <= 16, H] bf16, local ids [N, k] int32, weights [N, k] f32, w13, s13, w2, s2)
                                    #  -> routed [N, H] bf16: experts on BF16 (s13, s2 None) or block-scaled FP8
                                    #  weights -- the MTP head's, kernels/moe_rows; None: no side-file experts
    rows_linear: object = None      # (x [N, K] bf16, w [M, K] bf16) -> x @ w.T: a decode step's handful of rows by
                                    #  a weight it reads once -- the router (engine/kernels/common/skinny_gemv,
                                    #  torch.mm past its shapes); None: torch.mm
    router_logits: object = None   # (x BF16, w FP32) -> IEEE FP32 logits, including the top-k boundary's low bits
    leave: object = None            # how the served leaves meet the TP sum before them (served(leave=...), LEAVES);
                                    #  None: a table whose leaves are not the served kernel's
    ple_gate: object = None         # (h [N, hc*H], key [N, hc*H], value [N, H], q_norm, k_norm, conv_norm, eps, hc)
                                    #  -> (gated, normed) [N, hc*H]: the PLE injection's gate and conv norm in one launch
                                    #  (engine/kernels/ngram_gate); None: the module's torch form
    hc_site: object = None          # (h, out | None, inject | None, w, eps, hc, down_inject, up, *, inject, prefetch=None)
                                    #  -> (mixed [N, H], inject [N, hc] | None), h left into in place: hc_leave_norm (or
                                    #  hc_norm) and hc_mix in one call, so a prefill step's rows need not write the
                                    #  normalised streams (gated_residual.site); None: the two calls


# How a served leave meets the TP sum before it (carry H4, engine/kernels/gated_residual.leave_norm): "off", launched
# after the sum as any launch is; "pdl", the sum's programmatic dependent, resident through the other ranks' wait;
# "prefetch", that and the site's down projection pulled into L2 during the wait.
LEAVES = ("off", "pdl", "prefetch")
LEAVE = "prefetch"


def route_softmax_topk(logits: torch.Tensor, k: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """Qwen3.8's router (engine/modules/moe.route_softmax_topk with norm_topk_prob): softmax in fp32, top-k, weights
    renormalised to sum one. Ids int32 for the dispatcher."""
    from engine.modules.moe import route_softmax_topk as route
    ids, w = route(logits, k, True)
    return ids.to(torch.int32), w.float()


STATIC_TILE_ROWS = 16                 # the static MoE kernel's smallest judged row count (static_pad)


def static_pad(rows: int, micro_cap: int = 8) -> int:
    """Rows a captured MoE launch adds when it is above the micro kernel's cap and below one 16-row tile of the static
    kernel. On 2026-09-18 (srv4, Qwen3.8's cell E128/I640/top-10) the static kernel faulted -- an illegal address -- at
    10 and 12 rows, in isolation and in the four-row K=3 boot's capture, and passed at 14, 16, 20, 24, 28 and 32 rows
    (oracle within 0.7%, replay byte-equal). The served lane pads such a launch to 16 rows -- zero rows routed to this
    rank's expert 0 at weight 0 -- and 10, 12 and 14 then pass through the 16-row kernel the four-row shape runs."""
    return STATIC_TILE_ROWS - rows if micro_cap < rows < STATIC_TILE_ROWS else 0


def pad_static_launch(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, fill: int,
                      micro_cap: int = 8) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]":
    """(x, ids, weights, rows): a captured MoE launch widened by `static_pad` rows of zeros routed to `fill` at weight
    0 -- this rank's expert 0, by its global id on the global routes' path and 0 on route_local's -- and the rows the
    caller keeps of the output. Both captured paths take it: route_local's is the one a served step runs, and a K=3
    step of three rows (12) faulted there at capture on 2026-09-19 while the pad sat on the other path only."""
    rows = x.shape[0]
    pad = static_pad(rows, micro_cap)
    if pad:
        x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
        ids = torch.cat([ids, ids.new_full((pad, ids.shape[1]), fill)])
        weights = torch.cat([weights, weights.new_zeros(pad, weights.shape[1])])
    return x, ids, weights, rows


def local_routes(ids: torch.Tensor, weights: torch.Tensor, first: int, local: int,
                 sentinel: "int | None" = None) -> "tuple[torch.Tensor, torch.Tensor]":
    """Global expert ids to this rank's [0, local): a route to another rank's expert gets weight 0 and names local
    expert 0 -- the product is an exact zero, and a kernel that indexes with every route needs one of its experts -- or
    `sentinel` (E) where the launch admits the micro kernel's zero-weight skip (moe_dispatch.ep_zero_weight_sentinel),
    which drops the pair before it claims a row."""
    shifted = ids.to(torch.int32) - first
    foreign = (shifted < 0) | (shifted >= local)
    other = torch.zeros_like(shifted) if sentinel is None else torch.full_like(shifted, sentinel)
    return torch.where(foreign, other, shifted), torch.where(foreign, torch.zeros_like(weights), weights)


def reference() -> Lanes:
    from engine.modules.causal_conv import causal_conv1d
    from engine.modules.linear_attention import gated_delta_rule, gated_delta_rule_marked, gdn_decay
    from engine.modules.norm import rmsnorm_gated, rmsnorm_unit_offset
    from engine.modules.rotary import apply_rope, rope_tables

    def hc_norm(h, w, eps, hc):
        return rmsnorm_unit_offset(h, w, eps, group=None if hc == 1 else h.shape[1] // hc)

    def hc_leave(h, out, inject, hc):
        return h.add_((out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2))

    def hc_leave_norm(h, out, inject, w, eps, hc, *, prefetch=None):
        hc_leave(h, out, inject, hc)
        return h, hc_norm(h, w, eps, hc)

    def hc_mix(normed, down_inject, up, hc, *, inject=True, project_down=None, project_up=None):
        # the oracle normalises inside gated_residual; here the input is already normalised, so its mixer is replayed
        rank, hid = up.shape[1], normed.shape[1] // hc
        di = torch.nn.functional.linear(normed, down_inject) if project_down is None else project_down(normed)
        gates = torch.nn.functional.silu(di[:, :rank] / hc)
        up_rows = torch.nn.functional.linear(gates, up) if project_up is None else project_up(gates)
        weights = torch.sigmoid(up_rows).unflatten(-1, (hc, hid))
        mixed = (weights * normed.unflatten(-1, (hc, hid))).mean(dim=-2)
        return mixed, (2 * torch.sigmoid(di[:, rank:] / hc) if inject else None)

    def gdn_gates(a, b, A_log, dt_bias, *, sigmoid_beta):
        return gdn_decay(a, A_log, dt_bias), (torch.sigmoid(b) if sigmoid_beta else b)

    def value_heads(q, k, v):
        # key head i serves value heads i*g .. i*g + g - 1 (transformers qwen4_exp repeat_interleave), as the kernels'
        # grouped loads read them; the oracle's recurrence takes one head count
        g = v.shape[2] // q.shape[2]
        return (q, k) if g == 1 else (q.repeat_interleave(g, dim=2), k.repeat_interleave(g, dim=2))

    def gdn_chunk(q, k, v, decay, beta, state0, states_at=None):
        q, k = value_heads(q, k, v)
        return gated_delta_rule_marked(q, k, v, decay, beta, state0, scale=q.shape[-1] ** -0.5, qk_l2norm=True,
                                       decay_per_channel=False, marks=states_at)

    def gdn_ring(q, k, v, a, b_raw, A_log, dt_bias, ring, slot, context):
        q, k = value_heads(q, k, v)
        decay = gdn_decay(a, A_log, dt_bias)
        slot, context = int(slot), int(context)
        r = ring.shape[1]
        state0 = ring[slot, (context - 1) % r].unsqueeze(0).float() if context else None
        outs = []
        state = state0
        for i in range(q.shape[1]):
            o, state = gated_delta_rule(q[:, i:i + 1], k[:, i:i + 1], v[:, i:i + 1], decay[:, i:i + 1],
                                        torch.sigmoid(b_raw[:, i:i + 1]), state, scale=q.shape[-1] ** -0.5,
                                        qk_l2norm=True, decay_per_channel=False)
            ring[slot, (context + i) % r].copy_(state[0])
            outs.append(o)
        return torch.cat(outs, dim=1)

    def gdn_ring_rows(q, k, v, a, b_raw, A_log, dt_bias, ring, slots, contexts):
        rows = slots.numel()
        t = q.shape[1] // rows
        return torch.cat([gdn_ring(q[:, i * t:(i + 1) * t], k[:, i * t:(i + 1) * t], v[:, i * t:(i + 1) * t],
                                   a[:, i * t:(i + 1) * t], b_raw[:, i * t:(i + 1) * t], A_log, dt_bias, ring, slots[i],
                                   contexts[i])
                          for i in range(rows)], dim=1)

    def gdn_norm(core, z, w, eps):
        return rmsnorm_gated(core, z, w, eps, "sigmoid").reshape(core.shape[0], -1)

    def conv_prefill(x, w, state):
        return causal_conv1d(x, w, None, state, "silu")

    def conv_ring(x, w, ring, slot, context):
        slot, context = int(slot), int(context)
        width = w.shape[1] - 1
        r = ring.shape[2]
        history = torch.stack([ring[slot, :, (context - width + j) % r] if context - width + j >= 0 else
                               torch.zeros_like(ring[slot, :, 0]) for j in range(width)], dim=1)
        y, _ = causal_conv1d(x, w, None, history.to(x.dtype), "silu")
        for i in range(x.shape[0]):
            ring[slot, :, (context + i) % r] = x[i].to(ring.dtype)
        return y

    def conv_ring_rows(x, w, ring, slots, contexts):
        rows = slots.numel()
        t = x.shape[0] // rows
        return torch.cat([conv_ring(x[i * t:(i + 1) * t], w, ring, slots[i], contexts[i]) for i in range(rows)])

    def norm_rope(x, w, eps, positions, theta, rotary_dim):
        cos, sin = rope_tables(positions, rotary_dim, theta, dtype=x.dtype)
        return apply_rope(rmsnorm_unit_offset(x, w, eps), cos, sin)

    def unported(name):
        def refuse(*args, **kwargs):
            raise NotImplementedError(f"{name} addresses paged caches; its oracle is engine/modules/attention.QSA over "
                                      "the composition's State (tests/test_engine_composition.py), not a table lane")
        return refuse

    def moe(x, ids, weights, w13, w13_sf, w2, w2_sf, *, scales, first_expert, compact=False):
        """W4A4 as the served kernel sees it: activations quantised per 16 under the ModelOpt input scale, weights
        dequantised from the rank's packed layout (the fidelity the kernel is held to; the model's own reference is
        weight-only, engine/modules/moe, and the quality gate weighs the difference). It visits only local pairs
        either way, so `compact` changes nothing here."""
        from engine.modules.moe import expert_gemm
        from engine.modules.nvfp4_sf import unswizzle_sf
        E, two_i, half_h = w13.shape
        inter, hidden = two_i // 2, half_h * 2
        local, w = local_routes(ids, weights, first_expert, E)
        out = torch.zeros(x.shape[0], hidden, dtype=torch.float32, device=x.device)
        for e in local.unique().tolist():
            rows, slot = (local == e).nonzero(as_tuple=True)
            gain = w[rows, slot]
            if not bool((gain != 0).any()):
                continue
            s13 = unswizzle_sf(w13_sf[e].view(torch.uint8), two_i, hidden // 16).view(torch.float8_e4m3fn)
            s2 = unswizzle_sf(w2_sf[e].view(torch.uint8), hidden, inter // 16).view(torch.float8_e4m3fn)
            xe = x[rows].float()
            up = expert_gemm(xe, w13[e, :inter], s13[:inter], scales.weight13[e], scales.input13[e], quantize_act=True)
            gate = expert_gemm(xe, w13[e, inter:], s13[inter:], scales.weight13[e], scales.input13[e], quantize_act=True)
            act = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
            y = expert_gemm(act, w2[e], s2, scales.weight2[e], scales.input2[e], quantize_act=True)
            out.index_add_(0, rows, y.float() * gain[:, None])
        return out.to(x.dtype)

    def swiglu(fused, pad_to=None):
        gate, up = fused.chunk(2, -1)
        out = torch.nn.functional.silu(gate) * up
        return torch.nn.functional.pad(out, (0, pad_to - out.shape[-1])) if pad_to else out

    def moe_finish(routed, shared, gate):
        return (routed.float() + shared.float() * gate).to(routed.dtype)

    from engine.kernels.moe_rows import reference as moe_rows
    return Lanes("reference", hc_norm, hc_leave, hc_leave_norm, hc_mix, gdn_gates, gdn_chunk, gdn_ring, gdn_ring_rows,
                 gdn_norm, conv_prefill, conv_ring, conv_ring_rows, norm_rope, unported("qsa_store"),
                 unported("qsa_compress"), unported("qsa_select"), unported("qsa_attend"), route_softmax_topk, moe,
                 swiglu=swiglu, moe_finish=moe_finish, qsa_index_keys=unported("qsa_index_keys"),
                 qsa_inputs=unported("qsa_inputs"), moe_rows=moe_rows)


KERNEL_MODULES = ("engine.kernels.gated_residual", "engine.kernels.gdn", "engine.kernels.moe_output",
                  "engine.kernels.moe_route", "engine.kernels.moe_rows", "engine.kernels.ngram_gate",
                  "engine.kernels.qsa", "engine.kernels.router_fp32",
                  "engine.kernels.causal_conv_ring", "engine.kernels.causal_conv_single", "engine.kernels.kda.chunk_decay",
                  "engine.kernels.kda.index", "engine.kernels.kda.ring", "engine.kernels.b12x", "engine.kernels.moe_route",
                  "engine.modules.nvfp4_sf", "engine.kernels.common.decode_commit", "engine.kernels.common.norm_rope",
                  "engine.kernels.common.skinny_gemv", "engine.kernels.common.swiglu")
"""What `served` binds over, with the common lanes it starts from (engine/base/lanes). `import_kernels` exists so the
fleet boot can pay for them where it is already waiting; a test holds this list to the `from` lines in both."""


def import_kernels() -> None:
    """Import the kernel packages, nothing else -- GLM-5.3's `import_kernels`, over this table's packages.

    A Qwen3.8 fleet boot spent 3.96 s of rank 3 between `collectives` and `lanes qualified` (2026-09-18 17:35); these
    imports alone take 1.76 s in the ST image on the CPU -- triton, flashinfer, the CuTe DSL under b12x. None of it
    holds CUDA or reads anything the engine has produced, so the boot runs it on a thread under its rendezvous.
    `served` still does its own `from` imports; after this they are dictionary lookups."""
    import importlib
    for name in KERNEL_MODULES:
        importlib.import_module(name)


def served(*, tp=None, leave: str = LEAVE) -> Lanes:
    """Bind the ST kernel package for this shape. `tp` (a base/comm.LocalTP) hands each call to the main thread, where
    Triton's autotuner and the b12x JIT can run; on the fleet (one rank a process) the calls are direct. `leave`: one of
    LEAVES, the leaves' launch (carry H4); every choice computes the same bytes."""
    if leave not in LEAVES:
        raise ValueError(f"leave {leave!r}: one of {LEAVES}")
    from engine.base.lanes import served as common_lanes
    from engine.kernels import gated_residual as hcr
    from engine.kernels import gdn, moe_output, moe_route, qsa
    from engine.kernels.causal_conv_ring import causal_conv1d_ring, causal_conv1d_ring_rows
    from engine.kernels.causal_conv_single import causal_conv1d_single
    from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
    from engine.kernels.kda.index import single_sequence_bounds
    from engine.kernels.kda.ring import recurrent_gdn_ring, recurrent_gdn_ring_rows
    from engine.kernels.b12x import b12x_fused_moe
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels.common.skinny_gemv import linear_rows
    from engine.kernels import moe_rows, ngram_gate, router_fp32
    from engine.modules.nvfp4_sf import mma_sf_view

    def gdn_chunk(q, k, v, decay, beta, state0, states_at=None):
        t = q.shape[1]
        out = torch.empty_like(v)
        result = chunk_kda_with_decay(
            q, k, v, decay, beta, scale=q.shape[-1] ** -0.5,
            initial_state=state0.transpose(-1, -2).contiguous() if state0 is not None else None,
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=single_sequence_bounds(t, q.device),
            out=out, states_at=list(states_at) if states_at else None)
        # the kernel keeps [HV, V, K]; the engine's contract (and the ring's) is [HV, K, V]
        if states_at:
            o, state, states = result
            return o, state.transpose(-1, -2).contiguous(), states.transpose(-1, -2).contiguous()
        o, state = result
        return o, state.transpose(-1, -2).contiguous()

    prepared = {}

    def moe_prepare(w13, w13_sf, w2, w2_sf, top_k, *, scales):
        """The dispatcher's weight views for one layer, built once before capture (stock layout: the tiled and SF6
        cells are GLM-5.3's measured cells and are not admitted for 128 local experts at n 640)."""
        key = (w13.data_ptr(), w13_sf.data_ptr(), w2.data_ptr(), w2_sf.data_ptr())
        got = prepared.get(key)
        if got is not None:
            return got
        E, n, k = w13.shape[0], w13.shape[1] // 2, w13.shape[2] * 2
        sf13 = mma_sf_view(w13_sf, w13.shape[1], k)
        sf2 = mma_sf_view(w2_sf, w2.shape[1], w2.shape[2] * 2)
        views = md._get_weight_views(w1_fp4=w13, w1_blockscale=sf13, w2_fp4=w2, w2_blockscale=sf2,
                                     w1_alphas=scales.alpha13, w2_alphas=scales.alpha2, n=n, k=k,
                                     activation_precision="fp4", quant_mode="nvfp4",
                                     tiled=False, sf_pack=False, reform_sf_pack=False, packed_only=False)
        prepared[key] = (views, sf13, sf2)
        return prepared[key]

    def dispatch(x, ids, weights, w13, sf13, w2, sf2, views, scales, E):
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
        return b12x_fused_moe(x=x.contiguous(), output=output, w1_weight=w13, w1_weight_sf=sf13, w2_weight=w2,
                              w2_weight_sf=sf2, token_selected_experts=ids.contiguous(),
                              token_final_scales=weights.contiguous(), num_experts=E, num_local_experts=E, top_k=ids.shape[1],
                              w1_alpha=scales.alpha13, w2_alpha=scales.alpha2, fc2_input_scale=scales.input2,
                              input_global_scale=scales.input13, activation="silu", swiglu_alpha=1.0, swiglu_beta=0.0,
                              swiglu_limit=None, activation_precision="fp4", quant_mode="nvfp4", _weight_views=views)

    def sentinel_of(rows, k, w13, hidden):
        # A captured step's shapes are fixed, so every route stays in the launch. Another rank's routes carry
        # sentinel E where the shape admits the micro kernel's zero-weight skip (its pairs claim no rows and read no
        # expert), else local expert 0 at weight 0. The decision is the dispatcher's, per launch shape.
        return md.ep_zero_weight_sentinel(num_tokens=rows, num_topk=k, experts=w13.shape[0], hidden_size=hidden,
                                          intermediate_size=w13.shape[1] // 2, activation="silu", swiglu_limit=None)

    def route_local(scores, k, *, experts, first_expert, w13, hidden):
        rows = scores.shape[0]                  # the launch `moe` makes of them: padded to the static kernel's tile
        sentinel = sentinel_of(rows + static_pad(rows, md._MICRO_MAX_TOKENS), k, w13, hidden)
        return moe_route.softmax_topk(scores, k, experts=experts, first=first_expert, local=w13.shape[0],
                                      foreign=0 if sentinel is None else sentinel)

    def moe(x, ids, weights, w13, w13_sf, w2, w2_sf, *, scales, first_expert, compact=False, local=False):
        E = w13.shape[0]
        views, sf13, sf2 = moe_prepare(w13, w13_sf, w2, w2_sf, ids.shape[1], scales=scales)
        if local:
            if compact:
                raise ValueError("an eager step counts its own pairs from the global routes")
            # the static kernel's rows below one 16-row tile: zero rows on local expert 0 at weight 0
            x, ids, weights, rows = pad_static_launch(x, ids, weights, 0, md._MICRO_MAX_TOKENS)
            out = dispatch(x, ids, weights, w13, sf13, w2, sf2, views, scales, E)
            return out[:rows] if out.shape[0] != rows else out
        if not compact:
            x, ids, weights, rows = pad_static_launch(x, ids, weights, first_expert, md._MICRO_MAX_TOKENS)
            sentinel = sentinel_of(x.shape[0], ids.shape[1], w13, x.shape[1])
            local_ids, w = local_routes(ids, weights, first_expert, E, sentinel)
            out = dispatch(x, local_ids, w, w13, sf13, w2, sf2, views, scales, E)
            return out[:rows] if out.shape[0] != rows else out
        local_ids, w = local_routes(ids, weights, first_expert, E)
        # An eager step runs only this rank's (token, route) pairs, one route a row: at EP=4 the other ranks' routes are
        # ~3/4 of a prefill chunk's pairs, and on expert 0 they are rows of compute for a product of zero. Each pair's
        # weighted output (bf16) is summed per token in fp32, in the pairs' order, and rounded once (moe_output.pair_sum)
        shifted = ids.to(torch.int32) - first_expert
        token, route = ((shifted >= 0) & (shifted < E)).nonzero(as_tuple=True)
        if not token.numel():
            return torch.zeros_like(x, memory_format=torch.contiguous_format)
        pairs = dispatch(x.index_select(0, token), local_ids[token, route][:, None], w[token, route][:, None],
                         w13, sf13, w2, sf2, views, scales, E)
        return moe_output.pair_sum(pairs, token, x.shape[0])

    def on_main(fn):
        if tp is None:
            return fn
        def run(*a, **k):
            return tp.on_main(fn, *a, **k)
        return run

    pdl = leave != "off"

    def hc_leave(h, out, inject, hc):
        return hcr.leave(h, out, inject, hc, pdl=pdl)

    def hc_leave_norm(h, out, inject, w, eps, hc, *, prefetch=None):
        return hcr.leave_norm(h, out, inject, w, eps, hc, pdl=pdl, prefetch=prefetch if leave == "prefetch" else None)

    def hc_site(h, out, injection, w, eps, hc, down_inject, up, *, inject=True, prefetch=None):
        return hcr.site(h, out, injection, w, eps, hc, down_inject, up, inject=inject, pdl=pdl,
                        prefetch=prefetch if leave == "prefetch" else None)

    # the bound EP cell's decode routes to other ranks skip in the micro kernel (engine/base/kernel_shape bound first)
    md.configure_ep_zero_weight_micro(True)
    common = common_lanes()
    bound = [hcr.norm_streams, hc_leave, hc_leave_norm, hcr.mix, gdn.gates, gdn_chunk, recurrent_gdn_ring,
             recurrent_gdn_ring_rows, gdn.gated_norm, causal_conv1d_single, causal_conv1d_ring, causal_conv1d_ring_rows,
             qsa.norm_rope_partial, qsa.qsa_store_cache_rows, qsa.qsa_compress_groups_with_ratio,
             # Eager and captured selection share arithmetic and the lowest-id tie rule.
             qsa.qsa_select_paged_blocks, qsa.qsa_sparse_paged_attention_blocks, moe_route.softmax_topk, moe]
    return Lanes("served", *(on_main(f) for f in bound), moe_prepare=on_main(moe_prepare),
                 graph_resources=md.cached_workspace_owners, swiglu=on_main(common.swiglu),
                 moe_finish=on_main(moe_output.gated_sum), qsa_index_keys=on_main(qsa.qsa_index_keys),
                 qsa_inputs=on_main(qsa.qsa_inputs), qsa_select_alike=qsa.shards_select_alike,
                 qsa_attend_covered=on_main(qsa.qsa_covered_paged_attention), route_local=on_main(route_local),
                 rows_linear=on_main(linear_rows), router_logits=on_main(router_fp32.router_logits),
                 moe_rows=on_main(moe_rows.moe), ple_gate=on_main(ngram_gate.gate), hc_site=on_main(hc_site),
                 leave=leave)


def qualify(device, F) -> dict:
    """The served lanes that own arithmetic the wizard's glue does not cover, held to their oracles on `device` before
    a boot serves (D3): the gated residual at the model's widths, GDN's gates and output norm, QSA's head norm with
    its partial rotation (the query heads and the indexer's), the skinny GEMV at the shapes it takes, and the head's
    FP8 decode-row kernel against its recipe, and the PLE injection's gate."""
    from engine.kernels import gated_residual, gdn, ngram_gate, qsa
    from engine.kernels.common import skinny_gemv
    from engine.kernels.dense import fp8_rows
    return {"gated_residual": gated_residual.qualify(device, hc=F.hc, hidden=F.hidden, rank=F.hc_rank, eps=F.rms_eps),
            "gdn": gdn.qualify(device, heads=F.v_heads_local, dim=F.v_dim, eps=F.rms_eps),
            "qsa_norm_rope": qsa.qualify(device, heads=((F.heads_local, F.head_dim), (F.idx_heads, F.idx_dim)),
                                         rotary_dim=F.rotary_dim, theta=F.rope_theta, eps=F.rms_eps,
                                         max_position=F.max_position),
            "skinny_gemv": skinny_gemv.qualify(device), "fp8_rows": fp8_rows.qualify(device),
            "ngram_gate": ngram_gate.qualify(device, hc=F.hc, hidden=F.hidden, eps=F.rms_eps)}


__all__ = ["Lanes", "KERNEL_MODULES", "LEAVES", "LEAVE", "import_kernels", "reference", "served", "qualify",
           "route_softmax_topk", "local_routes", "static_pad", "pad_static_launch"]
