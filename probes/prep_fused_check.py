#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53_prep_fused numerics + launch-count probe (fresh container, never serving).

Gate ladder step 2 of overlay/modules/glm53_runtime/README.md: proves the
fused prep kernel reproduces, bit for bit, what the stock building blocks
write -- input_batch's Triton kernels, BlockTables.gather/compute_slot,
the GDN FULL-graph buffer copies, the sparse-MLA req ids, and the indexer's
uniform-decode kernel + compressed slot mapping -- on randomized batches, and
reports the host time of both sequences. The plan is built ONCE and the
request state is driven through the image's own UvaBackedTensor / BlockTables
objects, so the rotating `.gpu` handles (num_blocks rotates on every
apply_staged_writes, prefill_len on every admission) are exercised: the fused
arm launches after a further rotation and must still read the live buffer.
GPU time is taken from a CUDA-graph replay of the launch (events around a
Python launch loop only measure host issue time for kernels this short).

    /repo/probes/prep_fused_check.py [--trials 60] [--width 2052]

or, on srv4, via the wrapper that bind-mounts the composed overlay first:

    bash /repo/probes/run_prep_fused_check.sh [--trials 60]

The synthetic geometry mirrors GLM-5.3 TP=4 serving: 7 KV-cache groups
(MLA+indexer, kpool tail, 4 mamba groups, the drafter), Q=8 (7 drafts + 1),
kernel block 64 for the attention groups, indexer pages of 256 tokens
(storage block 64 pools, translation factor 4), kpool 4 for the tail, one
block per mamba group. Widths are arguments so the multi-chunk row copies get
exercised at the production width (57 x 36 = 2052 for a 128K context).
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEV = "cuda"


def _stock(bt, ib, rs, gdn, mla, idx, plan_like, idx_np, num_reqs, q):
    """The stock building blocks, in the runner's order, over buffer set A."""
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata
    from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
    from vllm.v1.attention.backends.mla.indexer import _prepare_uniform_decode_kernel
    from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
    from vllm.v1.worker.gpu.input_batch import (
        combine_sampled_and_draft_tokens,
        expand_idx_mapping,
        prepare_pos_seq_lens,
    )

    num_tokens = num_reqs * q
    idx_mapping = async_copy_to_gpu(idx_np, device=DEV)
    cu_np = np.arange(num_reqs + 1, dtype=np.int32) * q
    cu_num_logits = async_copy_to_gpu(cu_np, device=DEV)
    qsl_np = np.empty(bt.max_num_reqs + 1, dtype=np.int32)
    qsl_np[: num_reqs + 1] = cu_np
    qsl_np[num_reqs + 1:] = num_tokens
    async_copy_to_gpu(qsl_np, out=ib.query_start_loc)
    ib.is_padding[:num_tokens].fill_(False)
    expanded_idx, expanded_pos = expand_idx_mapping(idx_mapping, num_tokens, cu_num_logits, q)
    qsl = ib.query_start_loc[: num_reqs + 1]
    prepare_pos_seq_lens(idx_mapping, qsl, rs["num_computed"], ib.positions, ib.seq_lens)
    seq_lens = ib.seq_lens[:num_reqs]
    logits_indices = combine_sampled_and_draft_tokens(
        ib.input_ids, idx_mapping, rs["last_sampled"], qsl, seq_lens, rs["prefill_src"].gpu,
        rs["draft_tokens"], cu_num_logits, num_tokens, 1)
    block_tables = bt.gather_block_tables(idx_mapping, num_reqs_padded=num_reqs)
    slot_mappings = bt.compute_slot_mappings(idx_mapping, qsl, ib.positions, num_tokens_padded=num_tokens)
    # MambaHybridModelState.prepare_attn: num_accepted gather
    nacc = rs["num_accepted"].new_ones(num_reqs)
    nacc[:num_reqs] = rs["num_accepted"][idx_mapping]
    # GDN builders, FULL branch, all rows spec decodes (gdn_attn.py build())
    mask_cpu = torch.ones(num_reqs, dtype=torch.bool)
    for m, g in enumerate(plan_like["gdn_groups"]):
        b = gdn[m]
        table = block_tables[g]
        b["state"][:num_reqs].copy_(table[mask_cpu, : q], non_blocking=True)
        b["mask"][:num_reqs].copy_(mask_cpu.to(DEV)[:num_reqs], non_blocking=True)
        tok = torch.arange(num_tokens, dtype=torch.int32, device=DEV)
        b["tok"][:num_tokens].copy_(tok, non_blocking=True)
        b["qsl"][: num_reqs + 1].copy_(qsl[: num_reqs + 1], non_blocking=True)
        b["nacc"][:num_reqs].copy_(nacc[mask_cpu], non_blocking=True)
    # sparse MLA builder: req_id_per_token
    starts = np.asarray(cu_np, dtype=np.int32)
    seg = np.diff(starts)
    req_id = np.repeat(np.arange(seg.shape[0], dtype=np.int32), seg)
    mla["req_id"].fill_(0)
    mla["req_id"][: req_id.shape[0]].copy_(torch.from_numpy(req_id).pin_memory(), non_blocking=True)
    # indexer builder (mla/indexer.py build(), decode branch, uniform path)
    ga = plan_like["attn_g"]
    factor, ratio, sbs = plan_like["factor"], plan_like["ratio"], plan_like["sbs"]
    table = block_tables[ga]
    indexer_block_table = (table[:, ::factor] // factor).contiguous()
    get_compressed_slot_mapping(num_tokens, qsl, seq_lens, indexer_block_table, sbs, ratio,
                                out=idx["comp_slot"])
    torch.diff(qsl[: num_reqs + 1], out=idx["dec_lens"][:num_reqs])
    idx["per_req_dec_lens"][:num_reqs].copy_(idx["dec_lens"][:num_reqs])
    _prepare_uniform_decode_kernel[(num_tokens,)](
        seq_lens, idx["dec_seq_lens"], table, table.stride(0), idx["exp_bt"],
        idx["exp_bt"].stride(0), idx["dec_lens"], q, BLOCK_SIZE=1024)
    idx["dec_seq_lens"][num_tokens:] = 0
    exp = idx["exp_bt"][:num_tokens]
    compressed = exp[:, ::factor] // factor
    rows, cols = compressed.shape
    idx["idx_bt"][:rows, :cols].copy_(compressed)
    dsl = idx["dec_seq_lens"][:num_tokens]
    dsl //= ratio
    idx["sched"][:] = get_paged_mqa_logits_metadata(dsl.unsqueeze(-1), sbs, plan_like["num_sms"])
    return {
        "idx_mapping": idx_mapping, "expanded_idx": expanded_idx, "expanded_pos": expanded_pos,
        "logits_indices": logits_indices, "cu_num_logits": cu_num_logits,
    }


def _make_state(max_num_reqs, max_tokens, widths, kbs):
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import InputBuffers

    bt = BlockTables(
        block_sizes=list(kbs),  # kv block == kernel block here (blocks_per_kv_block 1)
        max_num_reqs=max_num_reqs, max_num_batched_tokens=max_tokens,
        max_num_blocks_per_group=list(widths), device=torch.device(DEV), kernel_block_sizes=list(kbs))
    ib = InputBuffers(max_num_reqs, max_tokens, torch.device(DEV))
    return bt, ib


def _rand_requests(bt, widths, kbs, max_num_reqs, gen, max_seq, num_blocks_total=4096):
    """Stage random block ids for every request slot, like add_request does.

    Every group holds at least cdiv(max_seq, block) blocks, as the allocator
    guarantees in production, so no gathered position falls beyond num_blocks:
    both the stock and the fused kernel read block ids unmasked, and a column
    read past the row would compare undefined memory instead of the layout."""
    for r in range(max_num_reqs):
        ids = []
        for g, w in enumerate(widths):
            need = -(-max_seq // kbs[g])
            assert need <= w, (g, need, w)
            if w <= 16:
                n = w  # mamba: every spec block allocated at admission
            else:
                n = int(gen.integers(need, max(need, min(w, 300)) + 1))
            ids.append([int(x) for x in gen.integers(1, num_blocks_total, size=n)])
        bt.append_block_ids(r, tuple(ids), overwrite=True)
    bt.apply_staged_writes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--width", type=int, default=2052, help="attention-group block table width")
    ap.add_argument("--reps", type=int, default=200, help="host-timing repetitions")
    args = ap.parse_args()
    torch.manual_seed(0)
    gen = np.random.default_rng(0)

    from vllm.models.glm5next.nvidia.glm53_prep_fused import PrepPlan

    q, num_spec = 8, 7
    max_num_reqs, max_tokens = 4, 2048
    # groups: [MLA+indexer, kpool tail, mamba x4, drafter]
    # prefill_len < 3000, num_computed < prefill_len + 2000, plus the Q tokens
    max_seq = 3000 + 2000 + q
    # the kpool tail (block 4) is sized for the whole context in production
    # (262144 columns); here cdiv(max_seq, 4) so every position is inside a row
    widths = [args.width, -(-max_seq // 4), 8, 8, 8, 8, args.width]
    kbs = [64, 4, 2304, 2304, 2304, 2304, 64]
    gdn_groups = [2, 3, 4, 5]
    # indexer spec: 256-token blocks (kpool 4 -> storage block 64 pools, the
    # deep_gemm block_kv), kernel block 64 tokens shared with the MLA ->
    # translation factor 256 // 64 = 4 (mounted mla/indexer.py build())
    attn_g, factor, ratio, sbs, num_sms = 0, 4, 4, 64, 48
    bt, ib = _make_state(max_num_reqs, max_tokens, widths, kbs)
    G = len(widths)

    def fresh_buffers():
        gdn = [{
            "state": torch.zeros(16, num_spec + 1, dtype=torch.int32, device=DEV),
            "mask": torch.zeros(16, dtype=torch.bool, device=DEV),
            "tok": torch.zeros(16 * (num_spec + 1), dtype=torch.int32, device=DEV),
            "qsl": torch.zeros(17, dtype=torch.int32, device=DEV),
            "nacc": torch.zeros(16, dtype=torch.int32, device=DEV),
        } for _ in gdn_groups]
        mla = {"req_id": torch.zeros(max_tokens, dtype=torch.int32, device=DEV)}
        cols = -(-args.width // factor)
        idx = {
            "exp_bt": torch.zeros(max_tokens, args.width, dtype=torch.int32, device=DEV),
            "dec_seq_lens": torch.zeros(max_tokens, dtype=torch.int32, device=DEV),
            "dec_lens": torch.zeros(max_tokens, dtype=torch.int32, device=DEV),
            "per_req_dec_lens": torch.zeros(max_tokens, dtype=torch.int32, device=DEV),
            "idx_bt": torch.zeros(max_tokens, cols, dtype=torch.int32, device=DEV),
            "comp_slot": torch.zeros(max_tokens, dtype=torch.int64, device=DEV),
            "sched": torch.empty(num_sms + 1, 2, dtype=torch.int32, device=DEV),
        }
        return gdn, mla, idx

    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

    rs = {
        "num_computed": torch.zeros(max_num_reqs, dtype=torch.int32, device=DEV),
        # like RequestState.prefill_len: a UVA-backed mirror whose .gpu rotates
        "prefill_src": UvaBackedTensor(max_num_reqs, torch.int32),
        "last_sampled": torch.zeros(max_num_reqs, 1, dtype=torch.int64, device=DEV),
        "draft_tokens": torch.zeros(max_num_reqs, num_spec, dtype=torch.int64, device=DEV),
        "num_accepted": torch.ones(max_num_reqs, dtype=torch.int32, device=DEV),
    }
    plan_like = {"gdn_groups": gdn_groups, "attn_g": attn_g, "factor": factor, "ratio": ratio,
                 "sbs": sbs, "num_sms": num_sms}

    def snapshot(gdn, mla, idx, num_reqs):
        t = num_reqs * q
        s = {
            "input_ids": ib.input_ids[:t], "positions": ib.positions[:t],
            "qsl": ib.query_start_loc, "seq_lens": ib.seq_lens, "is_padding": ib.is_padding[:t],
            "slots": bt.slot_mappings, "req_id": mla["req_id"], "exp_bt": idx["exp_bt"][:t],
            "dec_seq_lens": idx["dec_seq_lens"], "dec_lens": idx["dec_lens"][:t],
            "per_req": idx["per_req_dec_lens"][:num_reqs], "idx_bt": idx["idx_bt"][:t],
            "comp_slot": idx["comp_slot"], "sched": idx["sched"],
        }
        for g in range(G):
            s[f"bt{g}"] = bt.input_block_tables[g][:num_reqs]
        for m in range(len(gdn_groups)):
            for k in ("state", "mask", "nacc"):
                s[f"gdn{m}_{k}"] = gdn[m][k][:num_reqs]
            s[f"gdn{m}_tok"] = gdn[m]["tok"][:t]
            s[f"gdn{m}_qsl"] = gdn[m]["qsl"][: num_reqs + 1]
        return {k: v.clone() for k, v in s.items()}

    def reset_stale():
        for g in range(G):
            bt.input_block_tables[g].fill_(7)
        bt.slot_mappings.fill_(3)
        ib.input_ids.fill_(-5)
        ib.positions.fill_(-5)
        ib.query_start_loc.fill_(-5)
        ib.seq_lens.fill_(-5)
        ib.is_padding.fill_(True)
        for b in gdn:
            for t in b.values():
                t.zero_()
        mla["req_id"].zero_()
        for k, t in idx.items():
            if k == "sched":
                t.fill_(-9)
            else:
                t.zero_()

    class _Tail:  # stands in for the kpool tail builder: dormant circular buffer
        _tail_slot_buf = None

    gdn, mla, idx = fresh_buffers()
    plan = PrepPlan(
        q=q, num_spec=num_spec, max_num_reqs=max_num_reqs, max_num_tokens=max_tokens,
        device=torch.device(DEV),
        num_computed=rs["num_computed"], prefill_len_src=rs["prefill_src"],
        last_sampled=rs["last_sampled"], draft_tokens=rs["draft_tokens"],
        num_accepted=rs["num_accepted"],
        input_ids=ib.input_ids, positions=ib.positions, query_start_loc=ib.query_start_loc,
        seq_lens=ib.seq_lens, is_padding=ib.is_padding,
        bt=bt,
        gdn_groups=gdn_groups, gdn_state=[b["state"] for b in gdn], gdn_mask=[b["mask"] for b in gdn],
        gdn_tok=[b["tok"] for b in gdn], gdn_qsl=[b["qsl"] for b in gdn], gdn_nacc=[b["nacc"] for b in gdn],
        attn_g=attn_g, req_id_buf=mla["req_id"], exp_bt=idx["exp_bt"], dec_seq_lens=idx["dec_seq_lens"],
        dec_lens=idx["dec_lens"], per_req_dec_lens=idx["per_req_dec_lens"], idx_bt=idx["idx_bt"],
        comp_slot=idx["comp_slot"], factor=factor, ratio=ratio, sbs=sbs, num_sms=num_sms,
        sched_buf=idx["sched"], tail_builder=_Tail(),
    )
    plan.warmup()

    fails = 0
    for trial in range(args.trials):
        num_reqs = int(gen.integers(1, max_num_reqs + 1))
        _rand_requests(bt, widths, kbs, max_num_reqs, gen, max_seq)  # stages + apply_staged_writes (rotates num_blocks)
        idx_np = gen.permutation(max_num_reqs)[:num_reqs].astype(np.intp)
        pl = gen.integers(1, 3000, size=max_num_reqs)
        rs["prefill_src"].np[:] = pl
        rs["prefill_src"].copy_to_uva()  # admission: rotates prefill_len.gpu
        rs["num_computed"].copy_(torch.tensor(pl + gen.integers(0, 2000, size=max_num_reqs), dtype=torch.int32))
        rs["last_sampled"].copy_(torch.tensor(gen.integers(0, 150000, size=(max_num_reqs, 1)), dtype=torch.int64))
        rs["draft_tokens"].copy_(torch.tensor(gen.integers(0, 150000, size=(max_num_reqs, num_spec)), dtype=torch.int64))
        rs["num_accepted"].copy_(torch.tensor(gen.integers(1, 9, size=max_num_reqs), dtype=torch.int32))

        reset_stale()
        stock_extra = _stock(bt, ib, rs, gdn, mla, idx, plan_like, idx_np, num_reqs, q)
        torch.cuda.synchronize()
        A = snapshot(gdn, mla, idx, num_reqs)
        A_extra = {k: v.clone() for k, v in stock_extra.items()}

        reset_stale()
        # a further rotation of both UVA handles between the arms: the live
        # buffers hold the same values, the previous ones are now stale
        bt.apply_staged_writes()
        rs["prefill_src"].copy_to_uva()
        idx_mapping = plan.launch(idx_np, num_reqs)
        torch.cuda.synchronize()
        B = snapshot(gdn, mla, idx, num_reqs)
        t = num_reqs * q
        B_extra = {
            "idx_mapping": idx_mapping, "expanded_idx": plan.owned["expanded_idx"][:t],
            "expanded_pos": plan.owned["expanded_pos"][:t],
            "logits_indices": plan.owned["logits_arange"][:t],
            "cu_num_logits": plan.owned["cu_num_logits"][: num_reqs + 1],
        }
        bad = [k for k in A if A[k].shape != B[k].shape or not torch.equal(A[k], B[k])]
        bad += [f"ib:{k}" for k in A_extra
                if A_extra[k].shape != B_extra[k].shape or not torch.equal(A_extra[k], B_extra[k])]
        if bad:
            fails += 1
            print(f"trial {trial} num_reqs={num_reqs}: MISMATCH {bad}")
            for k in bad[:4]:
                if k.startswith("ib:"):
                    a, b = A_extra[k[3:]], B_extra[k[3:]]
                else:
                    a, b = A[k], B[k]
                if a.shape == b.shape:
                    d = (a != b).nonzero()[:6].tolist()
                    print(f"   {k}: first diffs at {d}; a={a.flatten()[:12].tolist()} b={b.flatten()[:12].tolist()}")
                else:
                    print(f"   {k}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        elif trial % 10 == 0:
            print(f"trial {trial} num_reqs={num_reqs}: OK ({len(A) + len(A_extra)} tensors bit-exact)")
    print(f"numerics: {args.trials - fails}/{args.trials} trials bit-exact")

    # host timing of the two sequences (GPU idle; this is the launch cost only,
    # the runner's ~1,000 aten calls of Python around them are not modelled)
    num_reqs = 1
    idx_np = np.arange(num_reqs, dtype=np.intp)
    for _ in range(5):
        _stock(bt, ib, rs, gdn, mla, idx, plan_like, idx_np, num_reqs, q)
        plan.launch(idx_np, num_reqs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.reps):
        _stock(bt, ib, rs, gdn, mla, idx, plan_like, idx_np, num_reqs, q)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(args.reps):
        plan.launch(idx_np, num_reqs)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"host+gpu per step, C=1: stock building blocks {(t1 - t0) / args.reps * 1e6:.0f} us, "
          f"fused {(t2 - t1) / args.reps * 1e6:.0f} us "
          f"(launches: stock ~30 kernels + ~8 memcpy, fused 3 kernels + 1 memcpy)")

    # GPU time: capture N launches in one CUDA graph and time the replay, so
    # the host issue cost (Triton dispatch ~45 us, deep_gemm wrapper ~75 us
    # on this CPU) cannot masquerade as kernel time. The pool staging copy is
    # excluded (host-side numpy write + a UVA copy that captures fine).
    n_cap = 20
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            plan.launch(idx_np, num_reqs)
        stream.synchronize()
        with torch.cuda.graph(g, stream=stream):
            for _ in range(n_cap):
                plan.launch(idx_np, num_reqs)
    torch.cuda.synchronize()
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    s_ev.record()
    for _ in range(10):
        g.replay()
    e_ev.record()
    torch.cuda.synchronize()
    print(f"fused GPU time per step (CUDA-graph replay of the launch incl. deep_gemm "
          f"schedule + copies): {s_ev.elapsed_time(e_ev) / (10 * n_cap) * 1e3:.1f} us")
    t3 = time.perf_counter()
    for _ in range(args.reps):
        plan.schedule(num_reqs * q)
    torch.cuda.synchronize()
    print(f"deep_gemm schedule call, host+gpu: {(time.perf_counter() - t3) / args.reps * 1e6:.0f} us")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
