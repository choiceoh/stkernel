# SPDX-License-Identifier: Apache-2.0
"""deneb fork: fused decode input preparation for GLM-5.3 (VLLM_GLM53_PREP_FUSED).

Between the drafter graph and the target graph the V2 runner prepares the
step on the host: prepare_inputs, prepare_attn and the metadata builders of
the seven KV-cache groups issue ~1,000 aten calls, ~45 memcpys and ~100 tiny
kernels per step, and with DFlash the scheduler runs synchronously, so the
GPU idles through all of it (2026-09-01 trace, rank 3: 8.9 ms of a 72 ms
profiled step idle, 5.7 ms of it in this region; nvidia-smi without the
profiler says ~7 pct, i.e. 4-5 ms of a 66 ms step).

For the steady-state decode step -- every request in spec-verify with the
full draft, FULL cudagraph dispatched, no request padding -- this module
replaces the whole region with ONE staged H2D copy (idx_mapping), ONE Triton
launch that writes every persistent buffer the captured graph reads, and the
deep_gemm scheduler-metadata call the indexer needs. Everything else the
stock path produced was Python objects nothing consumes under FULL replay
(the attention metadata dicts are only handed to the speculator, and DFlash
ignores them -- the plan refuses any other speculator), so those are served
from a per-shape cache.

What the kernel writes, per request r with state slot rs and query start
qs = r*Q (Q = decode_query_len = num_spec + 1):
  input_ids[qs] = last_sampled[rs], input_ids[qs+1+k] = draft[rs, k]
  positions[qs+j] = num_computed[rs] + j, seq_lens[r], query_start_loc[r]
  expanded_idx_mapping / expanded_local_pos, is_padding
  per group g: input_block_tables[g][r, :num_blocks] (the gather) and
    slot_mappings[g][qs+j] (generic pos -> bt[pos // bs] * bs + pos % bs)
  per GDN builder: spec_state_indices, spec_sequence_masks, spec_token_indx,
    spec_query_start_loc, num_accepted_tokens (the FULL-graph buffers)
  sparse MLA builder: req_id_per_token
  indexer builder: expanded_block_table rows, decode_seq_lens (compressed),
    decode_lens, per_req_decode_lens, indexer_decode_block_table,
    compressed_slot_mapping
  tails: query_start_loc / seq_lens padding, slot-mapping PAD tails, the
    indexer/MLA buffer tails the stock path re-fills every step.

Live handles, not snapshots. The runner's request-state mirrors are
UvaBackedTensors whose `.gpu` is REBOUND to the next round-robin buffer on
every copy_to_uva() (block_tables.apply_staged_writes rotates num_blocks
every step, admissions rotate prefill_len), and the block-table pointer
tensors are re-made after a KV-cache wake-up. The plan therefore keeps the
OWNING objects (BlockTables, the UvaBackedTensor) and dereferences them at
every launch, exactly as the stock kernels do.

Bit-exactness with the stock path is the contract, and it is checked where
it matters. shadow mode (VLLM_GLM53_PREP_FUSED=shadow) runs the fused path,
then the WHOLE stock chain (prepare_inputs, prepare_attn, the builders) over
the same buffers, diffs every buffer above and the InputBatch index fields,
and -- when clean -- hands the FUSED batch to the rest of the step, so the
armed control flow (view-based InputBatch through sampler / rejection sampler
/ drafter, the metadata cache) is exercised under shadow too. Armed mode
repeats that verification every VLLM_GLM53_PREP_FUSED_SELFCHECK_EVERY fused
steps (default 64) and DISARMs on the first drift.

Guards: the install pins the sha256 of every runner/builder file whose
control flow is bypassed (as shipped in glm53:v13-b12x, plus the mounted
glm53_tail_slot_persistent indexer). Any drift -> the module stays inert and
says so in the boot log. The plan is (re)built after every cudagraph capture
from the live runner and refuses (stock for the boot, logged) any geometry it
was not read against. The kpool tail group keeps the generic slot mapping
because the runner never hands the tail builder `positions` (its circular
mapping is dormant in this image); the plan asserts that dormancy (the
builder's circular buffer must stay unallocated) and every verification pass
re-checks it.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool, _load_ptr

logger = init_logger(__name__)

ENV = "VLLM_GLM53_PREP_FUSED"
ENV_SHADOW_EVERY = "VLLM_GLM53_PREP_FUSED_SHADOW_EVERY"
ENV_SELFCHECK_EVERY = "VLLM_GLM53_PREP_FUSED_SELFCHECK_EVERY"
_BLOCK = 1024
_SPECULATORS = ("DFlashSpeculator", "DSparkSpeculator")

# sha256 of the files whose control flow this module bypasses, as shipped in
# glm53:v13-b12x. mla/indexer.py is the glm53_tail_slot_persistent copy that
# is mounted over the image's. Drift anywhere -> stay stock (the fast path
# was read against exactly these files).
PREIMAGES: dict[str, str] = {
    "v1/worker/gpu/model_runner.py":
        "f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415",
    "v1/worker/gpu/input_batch.py":
        "3929c92e42ae90e4410bb4537dcbdcd171a662e44afc1189a9a3e19037a84410",
    "v1/worker/gpu/block_table.py":
        "da4e6011514c883dbb6bf0f59675294e6a6e76cfcd0e4b534bc575cb4d7adfc5",
    "v1/worker/gpu/attn_utils.py":
        "1dd3dd2826a2cc73005e7baecb71c26de8d56285b35d780716ab11ffe0f8495b",
    "v1/worker/gpu/buffer_utils.py":
        "51d37bde4f2f17d5aa9354faf35269ca1adde0eb16e73ae0d8b9860353ce57b7",
    "v1/worker/gpu/cudagraph_utils.py":
        "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f",
    "v1/worker/gpu/dp_utils.py":
        "3c882f85109ba47e473953351d166c4377ceb995205e91cbf128ca5075775a5d",
    "v1/worker/gpu/states.py":
        "99418f5df43ca612ded72609fee011620b065b2cab2f249ccb387096bf4ae71a",
    "v1/worker/gpu/model_states/mamba_hybrid.py":
        "3d1d3edc157d87f10aa6fb4862fbd25b435ede51499d7c64e68eb713de85e648",
    "v1/worker/gpu/spec_decode/dflash/speculator.py":
        "bd7f4c63d1196cb53bee0a81339aa5651e36938fc38b73c4ce89d978e0176a87",
    "v1/worker/utils.py":
        "3dcd6ad34ee1d1db2875f7f7dd51d90ee0e64041ab282180687770a38b26acb1",
    "v1/attention/backend.py":
        "301c76c90d5f26cdecfedfb385f8f17453154106d2decf763a31cb978f1a5d99",
    "v1/attention/backends/gdn_attn.py":
        "f27f887e71a7d092b79f7de55044a303456cfbd1e5eaf3461b394da9b3f21eba",
    "v1/attention/backends/utils.py":
        "cb9a34eb45a94847c8c9862952fe9ca5df4e7c7510425a5280afcbb48d941890",
    "v1/attention/backends/mla/compressor_utils.py":
        "b51d8905d0373611eb38504d9daa61a972cf59749aec54452157258db6eeea1d",
    "v1/attention/backends/mla/indexer.py":
        "c79f77d4f6326abb6718030f38d796b25bfff15471f1e61a1b1ea03e49720319",
    "model_executor/layers/attention/sparse_mla_attention.py":
        "0e784684bfe73bfc57ecc75464ba98c0163572e3961ca9a40c31df7480a1d0a9",
}


def prep_fused_mode() -> str:
    """off | on | shadow. Unknown values DISARM (stock path) and are logged.

    This knob selects a runner bypass, so a typo must land on the safe side
    (the repo's read_b12x_ep_exact_bool rule), never on "armed"."""
    raw = os.environ.get(ENV, "0")
    v = raw.strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return "off"
    if v == "shadow":
        return "shadow"
    if v == "1":
        return "on"
    logger.warning("[prep-fused] %s=%r is not one of 0/off/false/no, shadow, 1 -> DISARM "
                   "(stock path)", ENV, raw)
    return "off"


def _every(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def shadow_every() -> int:
    return max(1, _every(ENV_SHADOW_EVERY, 1))


def selfcheck_every() -> int:
    """Armed-mode verification cadence in fused steps; 0 disables."""
    return _every(ENV_SELFCHECK_EVERY, 64)


def check_preimages(root: str) -> list[str]:
    """Return the relative paths whose sha256 differs from PREIMAGES."""
    bad = []
    for rel, want in PREIMAGES.items():
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            got = "absent"
        if got != want:
            bad.append(f"{rel}: {got[:12]} != {want[:12]}")
    return bad


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fill(ptr, start, end, value, BLOCK: tl.constexpr):
    for i in range(start, end, BLOCK):
        off = i + tl.arange(0, BLOCK)
        tl.store(ptr + off, value, mask=off < end)


_NO_SPECIALIZE = [
    "num_reqs", "num_tokens", "max_num_reqs", "max_num_tokens", "draft_stride",
    "num_blocks_stride", "slot_stride", "req_id_cap", "exp_bt_stride", "dec_seq_cap",
    "idx_bt_stride", "idx_bt_cols", "comp_slot_cap",
]


@triton.jit(do_not_specialize=_NO_SPECIALIZE)
def _glm53_prep_fused_kernel(
    num_reqs,
    num_tokens,
    max_num_reqs,
    max_num_tokens,
    # request state (read)
    idx_mapping_ptr,
    num_computed_ptr,
    prefill_len_ptr,
    last_sampled_ptr,
    draft_tokens_ptr,
    draft_stride,
    num_accepted_ptr,
    # input buffers (write)
    input_ids_ptr,
    positions_ptr,
    qsl_ptr,
    seq_lens_ptr,
    is_padding_ptr,
    expanded_idx_ptr,
    expanded_pos_ptr,
    # block tables / slot mappings
    src_bt_ptrs,
    dst_bt_ptrs,
    bt_strides,
    block_sizes,
    num_blocks_ptr,
    num_blocks_stride,
    slot_ptr,
    slot_stride,
    # GDN builders
    gdn_group_idx_ptr,
    gdn_state_ptrs,
    gdn_state_strides,
    gdn_mask_ptrs,
    gdn_tok_ptrs,
    gdn_qsl_ptrs,
    gdn_nacc_ptrs,
    # attention group: sparse MLA + indexer builders
    req_id_ptr,
    req_id_cap,
    exp_bt_ptr,
    exp_bt_stride,
    dec_seq_lens_ptr,
    dec_seq_cap,
    dec_lens_ptr,
    per_req_dec_lens_ptr,
    idx_bt_ptr,
    idx_bt_stride,
    idx_bt_cols,
    comp_slot_ptr,
    comp_slot_cap,
    Q: tl.constexpr,
    NUM_SPEC: tl.constexpr,
    NS: tl.constexpr,
    NS_P2: tl.constexpr,
    G: tl.constexpr,
    N_GDN: tl.constexpr,
    ATTN_G: tl.constexpr,
    FACTOR: tl.constexpr,
    RATIO: tl.constexpr,
    SBS: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        role = pid - num_reqs
        if role == 0:
            # prepare_inputs pads query_start_loc[num_reqs:] with num_tokens
            # and the pos/seq_lens kernel zeroes seq_lens[num_reqs:].
            _fill(qsl_ptr, num_reqs, max_num_reqs + 1, num_tokens, BLOCK)
            _fill(seq_lens_ptr, num_reqs, max_num_reqs, 0, BLOCK)
            # VLLM_MOE_SKIP_PADDING: real rows are not padding. No request
            # padding on this path, so there is no True range to write.
            _fill(is_padding_ptr, 0, num_tokens, 0, BLOCK)
            for k in tl.static_range(N_GDN):
                qp = _load_ptr(gdn_qsl_ptrs + k, tl.int32)
                tl.store(qp + num_reqs, num_tokens)
            # sparse MLA: req_id_per_token_buffer.fill_(0) before the copy;
            # indexer: decode_seq_lens_buffer[num_decode_tokens:] = 0 and
            # compressed_slot_mapping_buffer.fill_(-1) before its kernel.
            _fill(req_id_ptr, num_tokens, req_id_cap, 0, BLOCK)
            _fill(dec_seq_lens_ptr, num_tokens, dec_seq_cap, 0, BLOCK)
            _fill(comp_slot_ptr, num_tokens, comp_slot_cap, PAD_ID, BLOCK)
        else:
            # _compute_slot_mappings_kernel's last program: PAD from the
            # actual token count to the end of the buffer, every group.
            g = role - 1
            _fill(slot_ptr + g * slot_stride, num_tokens, max_num_tokens, PAD_ID, BLOCK)
        return

    r = pid
    offs = tl.arange(0, Q)
    rs = tl.load(idx_mapping_ptr + r)
    ncomp = tl.load(num_computed_ptr + rs)
    qs = r * Q
    seq_len = ncomp + Q
    tl.store(seq_lens_ptr + r, seq_len)
    tl.store(qsl_ptr + r, qs)
    pos = ncomp.to(tl.int64) + offs.to(tl.int64)
    tl.store(positions_ptr + qs + offs, pos)

    # combine_sampled_and_draft_tokens, NUM_NEW_SAMPLED_TOKENS=1
    prefill_len = tl.load(prefill_len_ptr + rs)
    if seq_len > prefill_len:
        if seq_len - Q >= prefill_len:
            tok = tl.load(last_sampled_ptr + rs)
            tl.store(input_ids_ptr + qs, tok.to(tl.int32))
        dmask = offs < NUM_SPEC
        dr = tl.load(draft_tokens_ptr + rs * draft_stride + offs, mask=dmask, other=0)
        tl.store(input_ids_ptr + qs + 1 + offs, dr.to(tl.int32), mask=dmask)

    # expand_idx_mapping
    tl.store(expanded_idx_ptr + qs + offs, tl.full([Q], 0, tl.int64) + rs)
    tl.store(expanded_pos_ptr + qs + offs, offs.to(tl.int32))

    # gather_block_tables + compute_slot_mappings, every group
    for g in tl.static_range(G):
        src = _load_ptr(src_bt_ptrs + g, tl.int32)
        dst = _load_ptr(dst_bt_ptrs + g, tl.int32)
        stride = tl.load(bt_strides + g)
        bs = tl.load(block_sizes + g)
        nb = tl.load(num_blocks_ptr + g * num_blocks_stride + rs)
        src_row = src + rs * stride
        dst_row = dst + r * stride
        for i in range(0, nb, BLOCK):
            off = i + tl.arange(0, BLOCK)
            msk = off < nb
            v = tl.load(src_row + off, mask=msk, other=0)
            tl.store(dst_row + off, v, mask=msk)
        bsz = bs.to(tl.int64)
        bidx = pos // bsz
        boff = pos % bsz
        bn = tl.load(src_row + bidx)
        slot = (bn * bs).to(tl.int64) + boff
        tl.store(slot_ptr + g * slot_stride + qs + offs, slot)

    # the gathered rows are read back below by other threads of this program
    tl.debug_barrier()

    # GDN builders (FULL-graph branch): spec rows only, no padding
    soffs = tl.arange(0, NS_P2)
    smask = soffs < NS
    nacc = tl.load(num_accepted_ptr + rs)
    for k in tl.static_range(N_GDN):
        gm = tl.load(gdn_group_idx_ptr + k)
        dst = _load_ptr(dst_bt_ptrs + gm, tl.int32)
        stride = tl.load(bt_strides + gm)
        st = tl.load(dst + r * stride + soffs, mask=smask, other=0)
        sp = _load_ptr(gdn_state_ptrs + k, tl.int32)
        ss = tl.load(gdn_state_strides + k)
        tl.store(sp + r * ss + soffs, st, mask=smask)
        mp = _load_ptr(gdn_mask_ptrs + k, tl.int8)
        tl.store(mp + r + tl.arange(0, 1), tl.full([1], 1, tl.int8))
        tp = _load_ptr(gdn_tok_ptrs + k, tl.int32)
        tl.store(tp + qs + offs, (qs + offs).to(tl.int32))
        qp = _load_ptr(gdn_qsl_ptrs + k, tl.int32)
        tl.store(qp + r, qs)
        ap = _load_ptr(gdn_nacc_ptrs + k, tl.int32)
        tl.store(ap + r, nacc)

    # attention group: sparse MLA req ids + indexer decode buffers
    dsta = _load_ptr(dst_bt_ptrs + ATTN_G, tl.int32)
    wa = tl.load(bt_strides + ATTN_G)
    row = dsta + r * wa
    tl.store(req_id_ptr + qs + offs, tl.full([Q], 0, tl.int32) + r)
    tl.store(dec_lens_ptr + qs + offs, tl.full([Q], 1, tl.int32))
    tl.store(per_req_dec_lens_ptr + r, Q)
    # _prepare_uniform_decode_kernel: per-token context length, then
    # `seq_lens //= compress_ratio` on the same buffer
    tok_seq = seq_len - Q + offs + 1
    tl.store(dec_seq_lens_ptr + qs + offs, tok_seq // RATIO)
    # get_compressed_slot_mapping over indexer_block_table = bt[:, ::F] // F
    pos32 = seq_len - Q + offs
    valid = (pos32 + 1) % RATIO == 0
    pc = pos32 // RATIO
    bid = pc // SBS
    bn = tl.load(row + bid.to(tl.int64) * FACTOR, mask=valid, other=0) // FACTOR
    cslot = bn * SBS + pc % SBS
    cslot = tl.where(valid, cslot, PAD_ID)
    tl.store(comp_slot_ptr + qs + offs, cslot.to(tl.int64))
    # expanded_block_table_buffer[t] = the gathered row, full width, for the
    # Q tokens of this request: read each chunk once, store it Q times.
    trow = (qs + offs).to(tl.int64)
    exp_rows = exp_bt_ptr + trow[:, None] * exp_bt_stride
    idx_rows = idx_bt_ptr + trow[:, None] * idx_bt_stride
    for i in range(0, wa, BLOCK):
        off = i + tl.arange(0, BLOCK)
        msk = off < wa
        v = tl.load(row + off, mask=msk, other=0)
        tl.store(exp_rows + off[None, :], v[None, :], mask=msk[None, :])
    # indexer_decode_block_table_buffer[t, c] = row[c*F] // F
    for i in range(0, idx_bt_cols, BLOCK):
        off = i + tl.arange(0, BLOCK)
        msk = off < idx_bt_cols
        v = tl.load(row + off * FACTOR, mask=msk, other=0) // FACTOR
        tl.store(idx_rows + off[None, :], v[None, :], mask=msk[None, :])


# ---------------------------------------------------------------------------
# plan: every buffer the kernel writes and every live handle it reads
# ---------------------------------------------------------------------------
def _ptrs(tensors: list[torch.Tensor], device) -> torch.Tensor:
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.uint64, device=device)


@dataclass
class PrepPlan:
    q: int
    num_spec: int
    max_num_reqs: int
    max_num_tokens: int
    device: torch.device
    # request state; prefill_len_src is the UvaBackedTensor (its .gpu rotates)
    num_computed: torch.Tensor
    prefill_len_src: Any
    last_sampled: torch.Tensor
    draft_tokens: torch.Tensor
    num_accepted: torch.Tensor
    # input buffers
    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    is_padding: torch.Tensor
    # the runner's BlockTables: pointer tensors, num_blocks (.gpu rotates),
    # slot mappings and the gathered tables are read from it at launch
    bt: Any
    # GDN builders (group index, buffers)
    gdn_groups: list[int]
    gdn_state: list[torch.Tensor]
    gdn_mask: list[torch.Tensor]
    gdn_tok: list[torch.Tensor]
    gdn_qsl: list[torch.Tensor]
    gdn_nacc: list[torch.Tensor]
    # attention group
    attn_g: int
    req_id_buf: torch.Tensor
    exp_bt: torch.Tensor
    dec_seq_lens: torch.Tensor
    dec_lens: torch.Tensor
    per_req_dec_lens: torch.Tensor
    idx_bt: torch.Tensor
    comp_slot: torch.Tensor
    factor: int
    ratio: int
    sbs: int
    num_sms: int
    sched_buf: torch.Tensor
    # the kpool tail builder whose circular slot buffer must stay dormant
    tail_builder: Any = None
    owned: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dev = self.device
        if self.q & (self.q - 1):
            raise RuntimeError(f"decode_query_len {self.q} is not a power of two (tl.arange)")
        # idx_mapping staging: the image's round-robin UVA pool, sized by the
        # runner to max_concurrent_batches, so a step's host write can never
        # land under the previous step's in-flight copy (async scheduling).
        self.owned["idx_pool"] = UvaBufferPool(self.max_num_reqs, torch.int64)
        self.owned["idx_gpu"] = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=dev)
        self.owned["expanded_idx"] = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=dev)
        self.owned["expanded_pos"] = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=dev)
        self.owned["cu_num_logits"] = (
            torch.arange(self.max_num_reqs + 1, dtype=torch.int32, device=dev) * self.q)
        self.owned["logits_arange"] = torch.arange(self.max_num_tokens, dtype=torch.int64, device=dev)
        self.owned["gdn_group_idx"] = torch.tensor(self.gdn_groups, dtype=torch.int32, device=dev)
        self.owned["gdn_state_ptrs"] = _ptrs(self.gdn_state, dev)
        self.owned["gdn_state_strides"] = torch.tensor(
            [t.stride(0) for t in self.gdn_state], dtype=torch.int64, device=dev)
        self.owned["gdn_mask_ptrs"] = _ptrs(self.gdn_mask, dev)
        self.owned["gdn_tok_ptrs"] = _ptrs(self.gdn_tok, dev)
        self.owned["gdn_qsl_ptrs"] = _ptrs(self.gdn_qsl, dev)
        self.owned["gdn_nacc_ptrs"] = _ptrs(self.gdn_nacc, dev)
        wa = int(self.bt.block_table_strides[self.attn_g].item())
        if self.exp_bt.stride(0) != wa:
            raise RuntimeError(f"expanded block table stride {self.exp_bt.stride(0)} != "
                               f"attention block table width {wa}")
        self.idx_bt_cols = -(-wa // self.factor)
        if self.idx_bt.shape[1] < self.idx_bt_cols:
            raise RuntimeError("indexer decode table narrower than bt[:, ::factor]")
        ns = self.num_spec + 1
        self.ns_p2 = 1 << (ns - 1).bit_length()
        self.tail_ok()

    @property
    def G(self) -> int:
        return self.bt.num_kv_cache_groups

    def tail_ok(self) -> bool:
        """The kpool tail builder must never have allocated its circular
        slot buffer: that only happens when `positions` reach it, and then
        the graph reads a buffer this kernel does not write."""
        b = self.tail_builder
        if b is not None and getattr(b, "_tail_slot_buf", None) is not None:
            raise RuntimeError("kpool tail builder holds a live circular slot buffer: "
                               "positions reached it, the generic mapping is no longer "
                               "what the graph reads")
        return True

    def _consts(self) -> dict[str, int]:
        return dict(
            Q=self.q, NUM_SPEC=self.num_spec, NS=self.num_spec + 1, NS_P2=self.ns_p2,
            G=self.G, N_GDN=len(self.gdn_groups), ATTN_G=self.attn_g,
            FACTOR=self.factor, RATIO=self.ratio, SBS=self.sbs,
            PAD_ID=PAD_SLOT_ID, BLOCK=_BLOCK,
        )

    def _args(self, idx: torch.Tensor, num_reqs: int, num_tokens: int) -> tuple:
        o = self.owned
        bt = self.bt
        num_blocks = bt.num_blocks.gpu  # live: rebound on every apply_staged_writes
        return (
            num_reqs, num_tokens, self.max_num_reqs, self.max_num_tokens,
            idx, self.num_computed, self.prefill_len_src.gpu, self.last_sampled,
            self.draft_tokens, self.draft_tokens.stride(0), self.num_accepted,
            self.input_ids, self.positions, self.query_start_loc, self.seq_lens,
            self.is_padding.view(torch.int8), o["expanded_idx"], o["expanded_pos"],
            bt.block_table_ptrs, bt.input_block_table_ptrs, bt.block_table_strides,
            bt.block_sizes_tensor, num_blocks, num_blocks.stride(0),
            bt.slot_mappings, bt.slot_mappings.stride(0),
            o["gdn_group_idx"], o["gdn_state_ptrs"], o["gdn_state_strides"],
            o["gdn_mask_ptrs"], o["gdn_tok_ptrs"], o["gdn_qsl_ptrs"], o["gdn_nacc_ptrs"],
            self.req_id_buf, self.req_id_buf.numel(),
            self.exp_bt, self.exp_bt.stride(0),
            self.dec_seq_lens, self.dec_seq_lens.numel(),
            self.dec_lens, self.per_req_dec_lens,
            self.idx_bt, self.idx_bt.stride(0), self.idx_bt_cols,
            self.comp_slot, self.comp_slot.numel(),
        )

    def launch(self, idx_mapping_np: np.ndarray, num_reqs: int) -> torch.Tensor:
        """Stage idx_mapping, run the kernel, build the deep_gemm schedule.

        Returns the GPU idx_mapping view ([num_reqs], int64)."""
        o = self.owned
        idx = o["idx_pool"].copy_to_gpu(
            idx_mapping_np.astype(np.int64, copy=False), out=o["idx_gpu"][:num_reqs])
        num_tokens = num_reqs * self.q
        args = self._args(idx, num_reqs, num_tokens)
        compiled = o.get("compiled")
        if compiled is not None:
            # the binary warmup() built: same launcher the JIT path ends in,
            # minus its per-call binder/specialization/cache-key work (~12 us
            # of host time per step on this CPU). Valid for the boot because
            # every scalar is do_not_specialize and every pointer is a fixed
            # allocation; constexprs are passed positionally and ignored.
            compiled[(num_reqs + 1 + self.G, 1, 1)](*args, *self._consts().values())
        else:
            _glm53_prep_fused_kernel[(num_reqs + 1 + self.G,)](*args, **self._consts())
        self.schedule(num_tokens)
        return idx

    def warmup(self) -> None:
        """Compile the kernel without launching it (no buffer is touched).
        Raises on failure so the caller can DISARM instead of JIT-ing (or
        failing) on the first real request. Keeps the compiled binary for
        direct launches when this Triton hands one back."""
        idx = self.owned["idx_gpu"][:1]
        compiled = _glm53_prep_fused_kernel.warmup(
            *self._args(idx, 1, self.q), **self._consts(), grid=(1,))
        self.owned["compiled"] = compiled if hasattr(compiled, "__getitem__") else None

    def schedule(self, num_tokens: int) -> None:
        from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

        seq_lens = self.dec_seq_lens[:num_tokens].unsqueeze(-1)
        self.sched_buf[:] = get_paged_mqa_logits_metadata(seq_lens, self.sbs, self.num_sms)

    # -- verification support ---------------------------------------------
    def snapshot(self, num_reqs: int) -> dict[str, torch.Tensor]:
        t = num_reqs * self.q
        o = self.owned
        snap = {
            "input_ids": self.input_ids[:t], "positions": self.positions[:t],
            "query_start_loc": self.query_start_loc, "seq_lens": self.seq_lens,
            "is_padding": self.is_padding[:t],
            "expanded_idx": o["expanded_idx"][:t], "expanded_pos": o["expanded_pos"][:t],
            "slot_mappings": self.bt.slot_mappings,
            "req_id": self.req_id_buf, "exp_bt": self.exp_bt[:t],
            "dec_seq_lens": self.dec_seq_lens, "dec_lens": self.dec_lens[:t],
            "per_req_dec_lens": self.per_req_dec_lens[:num_reqs],
            "idx_bt": self.idx_bt[:t], "comp_slot": self.comp_slot, "sched": self.sched_buf,
        }
        for g, bt in enumerate(self.bt.input_block_tables):
            snap[f"bt{g}"] = bt[:num_reqs]
        for m in range(len(self.gdn_groups)):
            snap[f"gdn{m}_state"] = self.gdn_state[m][:num_reqs]
            snap[f"gdn{m}_mask"] = self.gdn_mask[m][:num_reqs]
            snap[f"gdn{m}_tok"] = self.gdn_tok[m][:t]
            snap[f"gdn{m}_qsl"] = self.gdn_qsl[m][:num_reqs + 1]
            snap[f"gdn{m}_nacc"] = self.gdn_nacc[m][:num_reqs]
        return {k: v.clone() for k, v in snap.items()}

    def diff(self, snap: dict[str, torch.Tensor], num_reqs: int) -> list[str]:
        live = self.snapshot(num_reqs)
        bad = []
        for k, v in snap.items():
            w = live[k]
            if v.shape != w.shape or not torch.equal(v, w):
                n = int((v != w).sum().item()) if v.shape == w.shape else -1
                bad.append(f"{k}({n})")
        return bad


# ---------------------------------------------------------------------------
# runner integration
# ---------------------------------------------------------------------------
@dataclass
class _State:
    mode: str
    shadow_every: int
    selfcheck_every: int
    plan: PrepPlan | None = None
    plan_failed: bool = False
    metadata_cache: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    steps_fused: int = 0
    steps_stock: int = 0
    checks_ok: int = 0
    checks_drift: int = 0

    def disarm(self, why: str) -> None:
        self.plan = None
        self.plan_failed = True
        self.metadata_cache.clear()
        logger.warning("[prep-fused] DISARM -> stock path for the rest of this boot: %s", why)


def _builder_name(b) -> str:
    return type(b).__name__


def build_plan(runner) -> PrepPlan:
    """Read every buffer the fast path writes off the live runner.

    Raises with a reason when the runner is not the shape this module was
    read against; the caller logs it once and stays stock."""
    from vllm.v1.kv_cache_interface import MambaSpec

    q = int(runner.decode_query_len)
    num_spec = int(runner.num_speculative_steps)
    if q != num_spec + 1 or num_spec <= 0:
        raise RuntimeError(f"decode_query_len {q} != num_spec {num_spec} + 1")
    if q & (q - 1):
        raise RuntimeError(f"decode_query_len {q} is not a power of two")
    if runner.model_state.num_new_sampled_tokens_per_step != 1:
        raise RuntimeError("num_new_sampled_tokens_per_step != 1")
    spec = runner.speculator
    if _builder_name(spec) not in _SPECULATORS:
        # the cached attention-metadata dict is only safe with a speculator
        # that never reads the target's attn_metadata (DFlash and its subclass)
        raise RuntimeError(f"speculator {_builder_name(spec)} may consume the target "
                           "attention metadata; only DFlash ignores it")
    for attr, want in (("use_dcp", False), ("use_pp", False)):
        if bool(getattr(runner, attr)) != want:
            raise RuntimeError(f"runner.{attr} is {getattr(runner, attr)}")
    if runner.pcp_manager is not None or runner.lora_config is not None:
        raise RuntimeError("PCP or LoRA active")
    if runner.adaptive_verification is not None:
        raise RuntimeError("adaptive verification active")
    if runner.model_config.rswa_window is not None:
        raise RuntimeError("R-SWA active")
    ms = runner.model_state
    if not hasattr(ms, "num_accepted_tokens_gpu"):
        raise RuntimeError("model_state has no num_accepted_tokens_gpu (not MambaHybrid)")
    if getattr(ms, "_align_mode", False):
        raise RuntimeError("mamba align mode: preprocess_state is not a no-op")
    bt = runner.block_tables
    if bt.cp_size != 1:
        raise RuntimeError("context parallel block tables")
    G = bt.num_kv_cache_groups
    groups = runner.kv_cache_config.kv_cache_groups
    if len(groups) != G:
        raise RuntimeError("kv_cache_groups / block table group count mismatch")
    draft_names = set(getattr(spec, "draft_attn_layer_names", ()) or ())
    draft_gids = set(getattr(spec, "draft_kv_cache_group_ids", ()) or ())

    gdn_groups, gdn_state, gdn_mask, gdn_tok, gdn_qsl, gdn_nacc = [], [], [], [], [], []
    attn_g = None
    mla_b = idx_b = tail_b = None
    for g in range(G):
        names_in_group = set(groups[g].layer_names)
        if not names_in_group or g in draft_gids or (draft_names and names_in_group <= draft_names):
            # the drafter's group (or a PP-empty one): the target runner
            # gathers its block table and slot mapping -- the drafter's prep
            # reads them -- but the drafter builds its own attention metadata
            # from its own builders, so the target-side builder this group
            # carries (FlashInfer for DFlash2) writes buffers nothing reads
            continue
        builders = [ag.get_metadata_builder(0) for ag in runner.attn_groups[g]]
        names = [_builder_name(b) for b in builders]
        spec_g = groups[g].kv_cache_spec
        if names == ["GDNAttentionMetadataBuilder"]:
            b = builders[0]
            if not isinstance(spec_g, MambaSpec):
                raise RuntimeError(f"group {g}: GDN builder on non-mamba spec")
            if runner.cache_config.mamba_cache_mode != "none":
                raise RuntimeError("mamba_cache_mode != none (block table select differs)")
            if not b.use_full_cuda_graph or b.num_spec != num_spec:
                raise RuntimeError(f"group {g}: GDN builder cudagraph/num_spec contract")
            if b.spec_state_indices_tensor.shape[1] != num_spec + 1:
                raise RuntimeError(f"group {g}: spec_state_indices width")
            if int(bt.block_table_strides[g].item()) < num_spec + 1:
                raise RuntimeError(f"mamba group {g} block table narrower than num_spec+1")
            gdn_groups.append(g)
            gdn_state.append(b.spec_state_indices_tensor)
            gdn_mask.append(b.spec_sequence_masks)
            gdn_tok.append(b.spec_token_indx)
            gdn_qsl.append(b.spec_query_start_loc)
            gdn_nacc.append(b.num_accepted_tokens)
        elif names == ["KpoolTailMetadataBuilder"]:
            tail_b = builders[0]  # dormant circular mapping: asserted by the plan
        elif sorted(names) == ["DeepseekV32IndexerMetadataBuilder", "FlashInferMLASparseMetadataBuilder"]:
            attn_g = g
            for b in builders:
                if _builder_name(b) == "FlashInferMLASparseMetadataBuilder":
                    mla_b = b
                else:
                    idx_b = b
        else:
            raise RuntimeError(f"group {g} {sorted(names_in_group)[:3]}: unexpected builders {names}")
    if attn_g is None or mla_b is None or idx_b is None:
        raise RuntimeError("no MLA+indexer attention group")
    if not gdn_groups:
        raise RuntimeError("no GDN groups")
    if tail_b is None:
        raise RuntimeError("no kpool tail group")
    if idx_b.compress_ratio <= 1 or not idx_b.use_flattening or idx_b.supports_varlen:
        raise RuntimeError("indexer builder is not on the flattened uniform-decode path")
    if idx_b.dcp_world_size != 1 or idx_b.pcp_world_size != 1:
        raise RuntimeError("indexer DCP/PCP")
    kbs = getattr(idx_b, "kernel_block_size", None)
    bs_idx = idx_b.kv_cache_spec.block_size
    if kbs is None or bs_idx == kbs or bs_idx % kbs != 0:
        raise RuntimeError("indexer block table is not compress-translated")
    if idx_b.indexer_decode_block_table_buffer is None:
        raise RuntimeError("indexer decode block table buffer not allocated yet")
    if bt.kernel_block_sizes[attn_g] != kbs:
        raise RuntimeError("attn group kernel block size != indexer kernel block size")

    rs = runner.req_states
    ib = runner.input_buffers
    plan = PrepPlan(
        q=q, num_spec=num_spec,
        max_num_reqs=int(runner.max_num_reqs), max_num_tokens=int(ib.max_num_tokens),
        device=runner.device,
        num_computed=rs.num_computed_tokens.gpu, prefill_len_src=rs.prefill_len,
        last_sampled=rs.last_sampled_tokens, draft_tokens=rs.draft_tokens,
        num_accepted=ms.num_accepted_tokens_gpu,
        input_ids=ib.input_ids, positions=ib.positions, query_start_loc=ib.query_start_loc,
        seq_lens=ib.seq_lens, is_padding=ib.is_padding,
        bt=bt,
        gdn_groups=gdn_groups, gdn_state=gdn_state, gdn_mask=gdn_mask, gdn_tok=gdn_tok,
        gdn_qsl=gdn_qsl, gdn_nacc=gdn_nacc,
        attn_g=attn_g, req_id_buf=mla_b.req_id_per_token_buffer,
        exp_bt=idx_b.expanded_block_table_buffer, dec_seq_lens=idx_b.decode_seq_lens_buffer,
        dec_lens=idx_b.decode_lens_buffer, per_req_dec_lens=idx_b.per_req_decode_lens_buffer,
        idx_bt=idx_b.indexer_decode_block_table_buffer, comp_slot=idx_b.compressed_slot_mapping_buffer,
        factor=bs_idx // kbs, ratio=int(idx_b.compress_ratio),
        sbs=int(idx_b.kv_cache_spec.storage_block_size), num_sms=int(idx_b.num_sms),
        sched_buf=idx_b.scheduler_metadata_buffer,
        tail_builder=tail_b,
    )
    for name, t, dt in (
        ("num_computed", plan.num_computed, torch.int32), ("prefill_len", rs.prefill_len.gpu, torch.int32),
        ("last_sampled", plan.last_sampled, torch.int64), ("draft_tokens", plan.draft_tokens, torch.int64),
        ("num_accepted", plan.num_accepted, torch.int32), ("input_ids", plan.input_ids, torch.int32),
        ("positions", plan.positions, torch.int64), ("query_start_loc", plan.query_start_loc, torch.int32),
        ("seq_lens", plan.seq_lens, torch.int32), ("is_padding", plan.is_padding, torch.bool),
        ("slot_mappings", bt.slot_mappings, torch.int64), ("num_blocks", bt.num_blocks.gpu, torch.int32),
        ("req_id", plan.req_id_buf, torch.int32), ("exp_bt", plan.exp_bt, torch.int32),
        ("dec_seq_lens", plan.dec_seq_lens, torch.int32), ("dec_lens", plan.dec_lens, torch.int32),
        ("idx_bt", plan.idx_bt, torch.int32), ("comp_slot", plan.comp_slot, torch.int64),
    ):
        if t.dtype != dt or not t.is_contiguous():
            raise RuntimeError(f"{name}: dtype {t.dtype} contiguous {t.is_contiguous()}")
    if plan.draft_tokens.shape[1] != num_spec:
        raise RuntimeError("draft_tokens width != num_spec")
    return plan


def _ensure_plan(runner, st: _State) -> bool:
    """Build (and warm) the plan once per capture; False keeps stock."""
    if st.plan is not None:
        return True
    if st.plan_failed:
        return False
    try:
        plan = build_plan(runner)
        plan.warmup()
    except Exception as e:  # loud, never fatal: the boot serves stock
        st.disarm(f"plan build/warmup failed: {e!r}")
        logger.exception("[prep-fused] plan build failed")
        return False
    st.plan = plan
    runner.model_state._glm53_prep = st
    logger.warning("[prep-fused] plan built: mode=%s groups=%d gdn=%s attn_g=%d factor=%d "
                   "ratio=%d sbs=%d q=%d shadow_every=%d selfcheck_every=%d", st.mode,
                   plan.G, plan.gdn_groups, plan.attn_g, plan.factor, plan.ratio, plan.sbs,
                   plan.q, st.shadow_every, st.selfcheck_every)
    return True


def _eligible(runner, st: _State, scheduler_output, batch_req_state, batch_desc) -> bool:
    from vllm.config import CUDAGraphMode

    if batch_req_state is None or batch_req_state.has_prefill:
        return False
    if batch_desc.cg_mode != CUDAGraphMode.FULL:
        return False
    num_reqs = len(batch_req_state.req_ids)
    q = int(runner.decode_query_len)
    if (batch_desc.uniform_token_count != q or batch_desc.num_reqs != num_reqs
            or batch_desc.num_tokens != num_reqs * q or batch_req_state.num_tokens != num_reqs * q):
        return False
    drafts = scheduler_output.scheduled_spec_decode_tokens
    if not drafts:
        return False
    nd = q - 1
    for rid in batch_req_state.req_ids:
        if len(drafts.get(rid, ())) != nd:
            return False
    if runner.adaptive_verification is not None:
        return False
    return _ensure_plan(runner, st)


def _fused_prepare_inputs(runner, st: _State, scheduler_output, batch_req_state, batch_desc):
    from vllm.v1.worker.gpu.input_batch import InputBatch

    plan = st.plan
    assert plan is not None
    q = plan.q
    req_ids = batch_req_state.req_ids
    num_reqs = len(req_ids)
    num_tokens = num_reqs * q
    idx_mapping_np = batch_req_state.idx_mapping_np
    idx_mapping = plan.launch(idx_mapping_np, num_reqs)
    o = plan.owned
    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32) * q
    num_computed_np = runner.req_states.num_computed_tokens_np[idx_mapping_np]
    num_scheduled = batch_req_state.num_scheduled_tokens
    seq_lens_ub_np = np.zeros(num_reqs, dtype=np.int32)
    np.add(num_computed_np, num_scheduled, out=seq_lens_ub_np)
    num_draft_per_req = np.full(num_reqs, q - 1, dtype=np.int32)
    ib = runner.input_buffers
    batch = InputBatch(
        req_ids=req_ids,
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        expanded_idx_mapping=o["expanded_idx"][:num_tokens],
        expanded_local_pos=o["expanded_pos"][:num_tokens],
        num_scheduled_tokens=num_scheduled,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=num_reqs * (q - 1),
        num_draft_tokens_per_req=num_draft_per_req,
        query_start_loc=ib.query_start_loc[:num_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=ib.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_ub_np),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=num_computed_np,
        prefill_len_np=batch_req_state.prefill_len_np,
        num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
        is_prefilling_np=batch_req_state.is_prefilling_np,
        has_prefill=False,
        max_seq_len_np=None,
        input_ids=ib.input_ids[:num_tokens],
        positions=ib.positions[:num_tokens],
        is_padding=ib.is_padding[:num_tokens],
        logits_indices=o["logits_arange"][:num_tokens],
        cu_num_logits=o["cu_num_logits"][:num_reqs + 1],
        cu_num_logits_np=query_start_loc_np.copy(),
        has_structured_output_reqs=scheduler_output.has_structured_output_requests,
        prompt_lens=None,
        max_query_len=None,
    )
    batch._glm53_fused = True
    return batch


def _compare_input_batches(fused, stock) -> list[str]:
    bad = []
    pairs = [
        ("logits_indices", fused.logits_indices, stock.logits_indices),
        ("expanded_idx_mapping", fused.expanded_idx_mapping, stock.expanded_idx_mapping),
        ("expanded_local_pos", fused.expanded_local_pos, stock.expanded_local_pos),
        ("cu_num_logits", fused.cu_num_logits, stock.cu_num_logits),
        ("idx_mapping", fused.idx_mapping, stock.idx_mapping),
    ]
    for name, a, b in pairs:
        if a.shape != b.shape or not torch.equal(a, b):
            bad.append(name)
    if not np.array_equal(fused.query_start_loc_np, stock.query_start_loc_np):
        bad.append("query_start_loc_np")
    if not np.array_equal(fused.cu_num_logits_np, stock.cu_num_logits_np):
        bad.append("cu_num_logits_np")
    if not torch.equal(fused.seq_lens_cpu_upper_bound, stock.seq_lens_cpu_upper_bound):
        bad.append("seq_lens_cpu_upper_bound")
    for name in ("num_tokens", "num_tokens_after_padding", "num_draft_tokens",
                 "num_reqs_after_padding", "has_prefill"):
        if getattr(fused, name) != getattr(stock, name):
            bad.append(name)
    return bad


_ORIG: dict[str, Any] = {}


def _state_of(runner) -> _State | None:
    st = getattr(runner, "_glm53_prep", None)
    if st is None:
        mode = prep_fused_mode()
        st = _State(mode=mode, shadow_every=shadow_every(),
                    selfcheck_every=selfcheck_every()) if mode != "off" else None
        runner._glm53_prep = st
    return st


def _verify(runner, st: _State, fused, scheduler_output, batch_req_state, batch_desc):
    """Run the whole stock chain over the fused buffers and diff.

    Returns the stock InputBatch (buffers now hold stock bytes) and the list
    of drifted names; an empty list means the fused batch is exact."""
    plan = st.plan
    assert plan is not None
    num_reqs = fused.num_reqs
    snap = plan.snapshot(num_reqs)
    stock = _ORIG["prepare_inputs"](runner, scheduler_output, batch_req_state, batch_desc)
    bad = _compare_input_batches(fused, stock)
    block_tables, slot_mappings = _ORIG["prepare_attn"](runner, stock)
    _ORIG["ms_prepare_attn"](runner.model_state, stock, batch_desc.cg_mode, block_tables,
                             slot_mappings, runner.attn_groups, runner.kv_cache_config, False)
    bad += plan.diff(snap, num_reqs)
    try:
        plan.tail_ok()
    except RuntimeError as e:
        bad.append(f"tail:{e}")
    return stock, bad


def _patched_prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
    st = _state_of(self)
    if st is None or not _eligible(self, st, scheduler_output, batch_req_state, batch_desc):
        if st is not None:
            st.steps_stock += 1
        return _ORIG["prepare_inputs"](self, scheduler_output, batch_req_state, batch_desc)
    try:
        fused = _fused_prepare_inputs(self, st, scheduler_output, batch_req_state, batch_desc)
    except Exception as e:
        st.disarm(f"fused launch failed: {e!r}")
        logger.exception("[prep-fused] fused prepare failed")
        return _ORIG["prepare_inputs"](self, scheduler_output, batch_req_state, batch_desc)
    st.steps_fused += 1
    if st.mode == "shadow":
        check = st.steps_fused % st.shadow_every == 0
    else:
        check = st.selfcheck_every > 0 and st.steps_fused % st.selfcheck_every == 0
    if not check:
        return fused
    stock, bad = _verify(self, st, fused, scheduler_output, batch_req_state, batch_desc)
    if bad:
        st.checks_drift += 1
        logger.warning("[prep-fused] %s DRIFT at fused step %d: %s", st.mode, st.steps_fused, bad[:12])
        if st.mode != "shadow":
            st.disarm("self-check drift")
        return stock
    st.checks_ok += 1
    if st.checks_ok % 64 == 0 or st.mode == "shadow" and st.checks_ok % 16 == 0:
        logger.warning("[prep-fused] %s: fused_steps=%d stock_steps=%d checks ok=%d drift=%d",
                       st.mode, st.steps_fused, st.steps_stock, st.checks_ok, st.checks_drift)
    # the buffers hold bytes identical to the fused ones: let the fused batch
    # (persistent views) drive the rest of the step, as arming would
    return fused


def _patched_prepare_attn(self, input_batch):
    if getattr(input_batch, "_glm53_fused", False):
        bt = self.block_tables
        n = input_batch.num_reqs_after_padding
        t = input_batch.num_tokens_after_padding
        return (tuple(x[:n] for x in bt.input_block_tables), bt.slot_mappings[:, :t])
    return _ORIG["prepare_attn"](self, input_batch)


def _patched_ms_prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                             attn_groups, kv_cache_config, for_capture=False):
    st = getattr(self, "_glm53_prep", None)
    orig = _ORIG["ms_prepare_attn"]
    if getattr(input_batch, "_glm53_fused", False) and not for_capture and st is not None \
            and st.plan is not None:
        key = (input_batch.num_reqs_after_padding, input_batch.num_tokens_after_padding)
        md = st.metadata_cache.get(key)
        if md is None:
            # first fused step of this shape: the stock builders over the fused
            # buffers (idempotent by contract) give the dict the speculator is
            # handed; every later step reuses it. The dict is only safe with a
            # speculator that ignores it, which build_plan enforced.
            md = orig(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                      attn_groups, kv_cache_config, for_capture)
            st.plan.tail_ok()
            st.metadata_cache[key] = md
        return md
    return orig(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                attn_groups, kv_cache_config, for_capture)


def _patched_capture_model(self):
    out = _ORIG["capture_model"](self)
    st = _state_of(self)
    if st is not None:
        # every capture re-creates the geometry the plan was read from
        st.plan = None
        st.metadata_cache.clear()
        st.plan_failed = False
        _ensure_plan(self, st)
    return out


def _patched_post_kv_cache_wake_up(self):
    out = _ORIG["post_kv_cache_wake_up"](self)
    st = getattr(self, "_glm53_prep", None)
    if st is not None:
        # the block-table pointer tensors were just re-made; the plan reads
        # them live, but the cached metadata dicts hold views -> rebuild
        st.plan = None
        st.metadata_cache.clear()
        st.plan_failed = False
    return out


def _memo_slot_mappings_by_layer():
    """build_slot_mappings_by_layer's dict for a persistent slot-mapping view
    is a pure function of (address, shape, config); memoize it."""
    orig = _ORIG["build_slot_mappings_by_layer"]
    cache: dict[tuple[int, int, tuple[int, ...]], dict[str, torch.Tensor]] = {}

    def build(slot_mappings, kv_cache_config):
        key = (id(kv_cache_config), slot_mappings.data_ptr(), tuple(slot_mappings.shape))
        d = cache.get(key)
        if d is None:
            if len(cache) > 64:
                cache.clear()
            d = orig(slot_mappings, kv_cache_config)
            cache[key] = d
        return d

    return build


_INSTALLED = False


def install_glm53_prep_fused() -> bool:
    """Patch the runner classes once. Safe to call from model __init__."""
    global _INSTALLED
    mode = prep_fused_mode()
    if mode == "off" or _INSTALLED:
        return _INSTALLED
    import sys

    import vllm

    root = os.path.dirname(os.path.abspath(vllm.__file__))
    bad = check_preimages(root)
    if bad:
        logger.warning("[prep-fused] preimage drift -> DISARM (stock path): %s", bad)
        return False
    mr = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    mh = sys.modules.get("vllm.v1.worker.gpu.model_states.mamba_hybrid")
    if mr is None or mh is None:
        try:
            import vllm.v1.worker.gpu.model_runner as mr  # noqa: F811
            import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh  # noqa: F811
        except Exception:
            logger.exception("[prep-fused] runner modules not importable -> DISARM")
            return False
    Runner = mr.GPUModelRunner
    MS = mh.MambaHybridModelState
    _ORIG["prepare_inputs"] = Runner.prepare_inputs
    _ORIG["prepare_attn"] = Runner.prepare_attn
    _ORIG["capture_model"] = Runner.capture_model
    _ORIG["post_kv_cache_wake_up"] = Runner.post_kv_cache_wake_up
    _ORIG["ms_prepare_attn"] = MS.prepare_attn
    _ORIG["build_slot_mappings_by_layer"] = mr.build_slot_mappings_by_layer
    Runner.prepare_inputs = _patched_prepare_inputs
    Runner.prepare_attn = _patched_prepare_attn
    Runner.capture_model = _patched_capture_model
    Runner.post_kv_cache_wake_up = _patched_post_kv_cache_wake_up
    MS.prepare_attn = _patched_ms_prepare_attn
    mr.build_slot_mappings_by_layer = _memo_slot_mappings_by_layer()
    _INSTALLED = True
    logger.warning("[prep-fused] installed mode=%s (preimages %d ok); the plan is built after "
                   "cudagraph capture", mode, len(PREIMAGES))
    return True
