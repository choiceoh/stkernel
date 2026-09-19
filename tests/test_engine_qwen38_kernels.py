"""Qwen3.8's served kernels held to their engine/modules oracles, on a GPU or on the CPU under TRITON_INTERPRET=1.

The lanes Qwen3.8 serves on (engine/profiles/qwen38/lanes.served) run the model's addressing and arithmetic on Triton
kernels: the QSA ops (engine/kernels/qsa.py), the gated residual in five launches a site (engine/kernels/gated_residual.py)
and GDN's gates and output norm (engine/kernels/gdn.py). A boot's `qualify` holds three of them to a band on random rows;
nothing held the paged QSA ops to the reference, nor a captured step's addressing (`Qwen38Net.step_meta`) to the host
step's, so no fold of them could be judged without a GPU. Each is held here to the oracle its docstring names:

    qsa_store_cache_rows            bytes equal to an index_put, -1 skipped, int32 slots on pages past int32 offsets
    qsa_compress_groups_with_ratio  bytes equal to the fp32 mean of a group's raw keys, ring members and this step's
    qsa_mqa_paged                   scores within fp32 of sum_h relu(q . key) / sqrt(D), visible blocks exact
    qsa_select_paged_tokens         the positions sparse_indexer.qsa_select attends, as sets
    qsa_sparse_paged_attention      within two BF16 steps of sparse_attention.gqa_sparse, in one split and in several
    norm_rope_partial               within qsa.qualify's band of rmsnorm_unit_offset then apply_rope
    gated_residual                  within its qualify's band of hyper_connection.gated_residual, the close included
    gdn.gates, gdn.gated_norm       the decay within fp32 of gdn_decay, the rest within gdn.qualify's band
    Qwen38Net.step_meta             a captured step addresses what the host step with the same rows does

The interpreter is slow: widths are a few tens of channels there, the model's per-rank widths on a GPU, the same tests.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_kernels
"""
import contextlib
import importlib.util
import inspect
import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
RUNS = torch is not None and TRITON and (INTERPRET or torch.cuda.is_available())
RUNS_REASON = "requires CUDA and Triton, or TRITON_INTERPRET=1 with Triton"
BF16_STEP = 2 ** -7                                   # one BF16 step of a value: 7 mantissa bits
THETA, EPS, MAX_POSITION = 1e7, 1e-6, 262144          # the checkpoint's rope theta, rms eps and context

# Qwen3.8 per rank at TP=4 (engine/profiles/qwen38/facts.py) on a GPU. Under the interpreter the head counts, the GQA
# group, the compression ratio and the rotary share of a head (a quarter of the query head, half the index head) stay;
# the widths shrink, and the budget is three blocks so a short sequence already selects among its blocks.
if INTERPRET:
    W = SimpleNamespace(heads=6, kv_heads=1, head_dim=32, rotary=8, idx_heads=4, idx_dim=16, ratio=4, budget=12,
                        block=16, hc=4, hidden=16, hc_rank=8, v_heads=12, v_dim=16, rows=(1, 7, 33))
else:
    W = SimpleNamespace(heads=6, kv_heads=1, head_dim=256, rotary=64, idx_heads=4, idx_dim=128, ratio=4, budget=2048,
                        block=768, hc=4, hidden=2560, hc_rank=320, v_heads=12, v_dim=128, rows=(1, 7, 300))


@contextlib.contextmanager
def served_kernels():
    """The served kernels as they run here. On a GPU: untouched. Under TRITON_INTERPRET=1, five test-only
    accommodations for what the interpreter does differently from a compiled kernel -- none changes a kernel's
    arithmetic; each makes the interpreter compute what the compiled kernel does:

    - a cast between fp32 and BF16 truncates the mantissa in the interpreter; the compiled cast rounds to nearest even,
      so the conversion is torch's;
    - `tl.dot` over BF16 operands multiplies the interpreter's uint16 storage of them; the compiled dot reads their
      values, so they are converted first (QSA's scorer and its sparse attention);
    - a comparison's result keeps its operand's dtype in the interpreter, so a scalar comparison broadcast against a
      block mask (the split-K merge's `split_mask & has_values`) becomes a float array `&` refuses; it is int1, as the
      compiled comparison's is;
    - the interpreter has no extern libdevice call: engine/kernels/gdn's `libdevice.log1p` is numpy's log1p;
    - the wrappers' CUDA-only checks see these CPU tensors as the interpreter's device."""
    if not INTERPRET:
        yield
        return
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as ti
    from engine.kernels import gdn

    builder = ti.InterpreterBuilder
    cast, dot, binary = builder.cast_impl, builder.create_dot, builder.binary_op

    def bf16_values(handle):
        stored = torch.from_numpy(np.array(handle.data, copy=True).view(np.int16))
        return ti.TensorHandle(stored.view(torch.bfloat16).float().numpy(), tl.float32)

    def cast_impl(self, src, dst_type):
        if src.dtype.scalar == tl.bfloat16 and dst_type.scalar == tl.float32:
            return bf16_values(src)
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            rounded = torch.from_numpy(np.array(src.data, copy=True)).to(torch.bfloat16)
            return ti.TensorHandle(rounded.view(torch.int16).numpy().view(np.uint16), tl.bfloat16)
        return cast(self, src, dst_type)

    def create_dot(self, a, b, acc, input_precision, max_num_imprecise_acc):
        if a.dtype.scalar == tl.bfloat16:
            a, b = bf16_values(a), bf16_values(b)
        return dot(self, a, b, acc, input_precision, max_num_imprecise_acc)

    def binary_op(self, lhs, rhs, op):
        out = binary(self, lhs, rhs, op)
        return ti.TensorHandle(out.data, tl.int1) if out.data.dtype == np.bool_ else out

    def log1p(x):
        return tl.core.tensor(ti.TensorHandle(np.log1p(x.handle.data), x.handle.dtype), x.type)

    with contextlib.ExitStack() as stack:
        for name, fn in (("cast_impl", cast_impl), ("create_dot", create_dot), ("binary_op", binary_op)):
            stack.enter_context(mock.patch.object(builder, name, fn))
        stack.enter_context(mock.patch.object(gdn, "libdevice", SimpleNamespace(log1p=log1p)))
        stack.enter_context(mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)))
        stack.enter_context(np.errstate(invalid="ignore", divide="ignore"))   # numpy computes the branch tl.where drops
        yield


def generator(seed: int):
    return torch.Generator(device="cpu").manual_seed(seed)


def randn(gen, *shape, scale=1.0, dtype=None):
    """Drawn on the CPU -- the same values on either device -- and moved to the test's device, BF16 unless named."""
    return (torch.randn(*shape, generator=gen) * scale).to(device=DEVICE, dtype=dtype or torch.bfloat16)


def bits(t):
    return t.contiguous().view(torch.uint8)


def paged(gen, pages: int, page: int, heads: int, dim: int, others: int = 2):
    """A paged region [pages, page, heads, dim] as the served caches present one (caches.Qwen38Caches): a strided view
    over block-major storage, each page one block that holds `others` regions of this size before this one. Returns
    (the view, its storage); the storage starts random, so a write that misses its row shows."""
    row = heads * dim
    stride = page * row * (others + 1)
    storage = randn(gen, pages * stride)
    return storage.as_strided((pages, page, heads, dim), (stride, row, dim, 1), others * page * row), storage


def block_table(gen, requests, blocks: int, block: int, pages: int):
    """[sequences, blocks] int32: each request's (seq, slot, ctx, length) reserved blocks on distinct random pages, -1
    past its reservation."""
    table = torch.full((max(r[0] for r in requests) + 1, blocks), -1, dtype=torch.int32)
    free, taken = torch.randperm(pages, generator=gen), 0
    for seq, _slot, ctx, length in requests:
        need = -(-(ctx + length) // block)
        table[seq, :need] = free[taken:taken + need].to(torch.int32)
        taken += need
    return table.to(DEVICE)


def host_meta(requests, table, *, block: int, ratio: int):
    """Qwen38Net.step_meta over a host step of `requests` (seq, slot, ctx, length), as the served net builds it."""
    from engine.profiles.qwen38.net import Qwen38Net, Segment, Step
    segments, start = [], 0
    for seq, slot, ctx, length in requests:
        segments.append(Segment(seq, slot, ctx, start, length))
        start += length
    step = Step(torch.zeros(start, dtype=torch.int64, device=table.device), tuple(segments))
    net = SimpleNamespace(F=SimpleNamespace(block=block, idx_ratio=ratio))
    return Qwen38Net.step_meta(net, step, SimpleNamespace(block_table=table))


def bands(qualify) -> "tuple[float, float]":
    """(largest error / largest magnitude, error rms / rms): the band a lane's `qualify` holds a boot to by default."""
    parameters = inspect.signature(qualify).parameters
    return parameters["band_max"].default, parameters["band_rms"].default


class Held(unittest.TestCase):
    """A served output held to its reference within a band."""

    def assertWithin(self, got, want, band, what):
        """`got` has `want`'s shape and dtype and drifts from it by no more than `band` (gated_residual.drift)."""
        from engine.kernels.gated_residual import drift
        self.assertEqual((tuple(got.shape), got.dtype), (tuple(want.shape), want.dtype), what)
        largest, rms = drift(got, want)
        self.assertLessEqual(largest, band[0], f"{what}: the largest error is {largest:.3g} of the largest value")
        self.assertLessEqual(rms, band[1], f"{what}: the error rms is {rms:.3g} of the values' rms")


@unittest.skipUnless(RUNS, RUNS_REASON)
class StoreTests(unittest.TestCase):
    """qsa_store_cache_rows: fixed-width rows at flat slots page * page_size + offset of a paged view, -1 skipped."""

    def test_rows_land_at_their_slots_and_nothing_else_moves(self):
        from engine.kernels import qsa
        from engine.profiles.qwen38.caches import QSA_KEY_RING
        gen = generator(11)
        # the three regions the attention layer writes: K or V rows by block, index keys by group, the raw-key ring
        regions = (("kv rows", W.block, W.head_dim), ("index keys", W.block // W.ratio, W.idx_dim),
                   ("key ring", QSA_KEY_RING, W.idx_dim))
        for (name, page, width), rank3 in zip(regions, (False, True, False)):
            with self.subTest(region=name, rows="[N, 1, D]" if rank3 else "[N, D]"):
                pages, n = 7, 9
                cache, storage = paged(gen, pages, page, 1, width)
                slots = torch.randperm(pages * page, generator=gen)[:n].to(torch.int32)
                slots[[0, 5]] = -1
                slots = slots.to(DEVICE)
                rows = randn(gen, n, width)
                expected = storage.clone()
                kept = slots >= 0
                region = expected.as_strided(cache.shape, cache.stride(), cache.storage_offset())[:, :, 0]
                region[(slots[kept] // page).long(), (slots[kept] % page).long()] = rows[kept]      # index_put
                with served_kernels():
                    qsa.qsa_store_cache_rows(cache, slots, rows[:, None, :] if rank3 else rows)
                self.assertTrue(torch.equal(bits(storage), bits(expected)))

    def test_an_int32_slot_past_the_block_major_page_stride(self):
        """A block holds every attention layer's rows: 13 layers x (2 * 768 * 256 + 192 * 128) = 5,431,296 BF16 rows a
        page. The slot mapping is int32; page * stride passes int32 from page 396 on, and at page 791 an int32 product
        wraps back inside page 0's block, where a sentinel sits. The storage is 8.7 GiB of address space and a few
        touched pages on the CPU."""
        from engine.kernels import qsa
        stride, page, width, pages = 13 * (2 * 768 * 256 + 192 * 128), 192, 128, 792
        if DEVICE == "cuda" and torch.cuda.mem_get_info()[0] < 2 * pages * stride * 2:
            self.skipTest("the device has no room for a block-major storage past int32")
        gen = generator(12)
        storage = torch.empty(pages * stride, dtype=torch.bfloat16, device=DEVICE)
        cache = storage.as_strided((pages, page, 1, width), (stride, width, width, 1), 0)
        targets = ((3, 17), (395, 5), (791, 3))
        wrapped = (791 * stride + 3 * width) % 2 ** 32
        self.assertEqual((wrapped // stride, wrapped % stride >= page * width), (0, True))   # page 0, outside its rows
        storage[wrapped:wrapped + width] = 7.0
        rows = randn(gen, len(targets), width)
        for p, t in targets:
            cache[p, t, 0] = -3.0
        slots = torch.tensor([p * page + t for p, t in targets], dtype=torch.int32, device=DEVICE)
        with served_kernels():
            qsa.qsa_store_cache_rows(cache, slots, rows)
        for i, (p, t) in enumerate(targets):
            self.assertTrue(torch.equal(bits(cache[p, t, 0]), bits(rows[i])), (p, t))
        self.assertTrue(bool((storage[wrapped:wrapped + width] == 7.0).all()))


@unittest.skipUnless(RUNS, RUNS_REASON)
class CompressTests(Held):
    """qsa_compress_groups_with_ratio as net._qsa calls it: every group a row closes pooled from the per-sequence ring of
    raw index keys (the members before the step) and this step's rows (the rest)."""

    # (seq, slot, ctx, length): a first group inside the step; groups that take 1, 2 and 3 members from the ring; a
    # decode row that closes nothing; a prefill-like segment longer than the ring that closes three groups
    REQUESTS = ((2, 3, 0, 4), (0, 1, 2, 2), (5, 6, 5, 1), (1, 2, 6, 2), (3, 4, 11, 1), (4, 5, 9, 11))

    def test_a_closed_group_is_the_mean_of_its_raw_keys(self):
        from engine.kernels import qsa
        from engine.modules.norm import rmsnorm_unit_offset
        from engine.modules.rotary import apply_rope, rope_tables
        from engine.profiles.qwen38.caches import QSA_KEY_RING
        gen = generator(21)
        ratio, D, pages = W.ratio, W.idx_dim, 16
        table = block_table(gen, self.REQUESTS, 4, W.block, pages)
        meta = host_meta(self.REQUESTS, table, block=W.block, ratio=ratio)
        histories = [randn(gen, ctx + length, D) for _, _, ctx, length in self.REQUESTS]
        ring = randn(gen, max(r[1] for r in self.REQUESTS) + 1, QSA_KEY_RING, 1, D)       # cells never written: noise
        for (_, slot, ctx, _), history in zip(self.REQUESTS, histories):
            for p in range(max(0, ctx - QSA_KEY_RING), ctx):
                ring[slot, p % QSA_KEY_RING, 0] = history[p]
        raw = torch.cat([history[ctx:] for (_, _, ctx, _), history in zip(self.REQUESTS, histories)])
        n = raw.shape[0]
        with served_kernels():
            pooled, first = qsa.qsa_compress_groups_with_ratio(
                raw[:, None, :], meta.positions[:, None, None].expand(n, 1, 3).contiguous(), ring, meta.slot_table,
                meta.rows_req, meta.starts, meta.positions, meta.key_slots, ratio)
        self.assertEqual((tuple(pooled.shape), pooled.dtype, tuple(first.shape), first.dtype),
                         ((n, 1, D), torch.bfloat16, (n, 3), torch.int64))
        closing, from_ring, want = [], set(), []
        row = 0
        for (_, _, ctx, length), history in zip(self.REQUESTS, histories):
            for p in range(ctx, ctx + length):
                if (p + 1) % ratio == 0:
                    mean = history[p + 1 - ratio:p + 1].float().mean(0).to(torch.bfloat16)
                    with self.subTest(row=row, position=p):
                        self.assertTrue(torch.equal(bits(pooled[row, 0]), bits(mean)))
                        self.assertEqual(first[row].tolist(), [p + 1 - ratio] * 3)
                    closing.append((row, p))
                    from_ring.add(max(0, ctx - (p + 1 - ratio)))
                    want.append(mean)
                row += 1
        self.assertEqual(from_ring, {0, 1, 2, 3})                   # the fixture reaches every split of a group

        # the write that follows in net._qsa: the pooled keys normalised and rotated at their groups' first positions
        # and stored at the step's key slots; a row that closes no group has slot -1 and moves nothing
        k_norm = randn(gen, D, scale=0.1)
        index_keys, storage = paged(gen, pages, W.block // ratio, 1, D)
        expected = storage.clone()
        with served_kernels():
            keys = qsa.norm_rope_partial(pooled, k_norm, EPS, first[:, 0].contiguous(), THETA, W.rotary)
            qsa.qsa_store_cache_rows(index_keys, meta.key_slots, keys[:, 0])
        region = expected.as_strided(index_keys.shape, index_keys.stride(), index_keys.storage_offset())
        per = W.block // ratio
        for r, p in closing:
            slot = int(meta.key_slots[r])
            region[slot // per, slot % per, 0] = keys[r, 0]
        self.assertEqual(int((meta.key_slots >= 0).sum()), len(closing))
        self.assertTrue(torch.equal(bits(storage), bits(expected)))
        rows = torch.tensor([r for r, _ in closing], device=DEVICE)
        starts = torch.tensor([p + 1 - ratio for _, p in closing], device=DEVICE)
        cos, sin = rope_tables(starts, W.rotary, THETA, dtype=torch.bfloat16)
        reference = apply_rope(rmsnorm_unit_offset(torch.stack(want)[:, None, :], k_norm, EPS), cos, sin)
        self.assertWithin(keys.index_select(0, rows), reference, bands(qsa.qualify), "the stored index keys")


@unittest.skipUnless(RUNS, RUNS_REASON)
class SelectionTests(unittest.TestCase):
    """qsa_mqa_paged, select_blocks and expand_qsa_block_indices_cuda, launched by qsa_select_paged_tokens over the
    paged index keys, against modules/sparse_indexer.qsa_select for every query row.

    The tie rule: a block that scores within the scorer's fp32 difference of the budget's edge may fall on either side of
    it. Both sides take the best blocks with torch.topk (the served lane's radix select, ties to the lower block, admits
    only prefill row counts above 64), which orders equal scores as its candidates happen to fall, and the kernel ranks
    its own fp32 sums, not torch's matmul. So a row attends exactly qsa_select's positions wherever its last block taken
    outscores its first block left by more than twice the largest difference between the row's logits and the
    reference's scores. Where the edge is closer than that -- thousands of blocks against a 512-block budget at the
    model's widths leave it no wider than a few hundredths of a percent -- the blocks taken are a top-k of the
    reference's scores within that slack, whole, and the tail of the open group is the reference's."""

    @staticmethod
    def requests():
        r, blocks = W.ratio, W.budget // W.ratio
        # (seq, slot, ctx, length): position 0, whose only position is its tail; half the budget's blocks visible; the
        # budget exactly, then its next position; three budgets' worth of blocks at a row whose next row closes a group
        # it must not see (a verify step's draft), then that row; five budgets' worth and a two-position tail; a segment
        # past two 64-block score tiles whose rows take the step past 32 (the scorer's 8 tiles a program)
        return ((0, 1, 0, 1), (3, 2, blocks // 2 * r + 2, 1), (1, 3, blocks * r - 1, 2), (4, 4, 3 * blocks * r + 2, 2),
                (2, 5, 5 * blocks * r - 3, 1), (5, 6, (128 + blocks) * r + 1, 36))

    def fixture(self, seed):
        from engine.modules.norm import rmsnorm_unit_offset
        from engine.modules.rotary import apply_rope, rope_tables
        gen = generator(seed)
        requests, ratio, D, per = self.requests(), W.ratio, W.idx_dim, W.block // W.ratio
        pages = 40 * len(requests)
        table = block_table(gen, requests, 40, W.block, pages)
        meta = host_meta(requests, table, block=W.block, ratio=ratio)
        k_norm = randn(gen, D, scale=0.1)
        key_cache, _ = paged(gen, pages, per, 1, D)
        histories, cached = [], []
        for seq, _slot, ctx, length in requests:
            history = randn(gen, ctx + length, D)
            groups = history.shape[0] // ratio
            tokens = torch.arange(groups * ratio, device=DEVICE).view(groups, ratio)
            # the pooled, normalised and rotated keys exactly as qsa_select builds them, stored where the step's slots
            # put them
            pooled = history.index_select(0, tokens.flatten()).view(groups, ratio, D).float().mean(dim=1).to(history.dtype)
            cos, sin = rope_tables(torch.arange(history.shape[0], device=DEVICE), W.rotary, THETA, dtype=history.dtype)
            keys = apply_rope(rmsnorm_unit_offset(pooled, k_norm, EPS).unsqueeze(1), cos.index_select(0, tokens[:, 0]),
                              sin.index_select(0, tokens[:, 0])).squeeze(1)
            g = torch.arange(groups, device=DEVICE)
            key_cache[:, :, 0][table[seq, g // per].long(), g % per] = keys
            histories.append((history, cos, sin))
            cached.append(keys)
        queries = randn(gen, meta.positions.numel(), W.idx_heads, D)
        return SimpleNamespace(requests=requests, meta=meta, k_norm=k_norm, key_cache=key_cache, histories=histories,
                               keys=cached, queries=queries)

    def test_each_row_attends_what_qsa_select_chooses(self):
        from engine.kernels import qsa
        from engine.modules.sparse_indexer import qsa_select
        f = self.fixture(31)
        meta, ratio, taken = f.meta, W.ratio, W.budget // W.ratio
        with served_kernels():
            logits, visible = qsa.qsa_mqa_paged(f.queries, f.key_cache, meta.page_table, meta.rows_req,
                                                meta.positions32, meta.lengths, ratio)
            selected = qsa.qsa_select_paged_tokens(f.queries, f.key_cache, meta.page_table, meta.rows_req,
                                                   meta.positions32, meta.lengths, W.budget, ratio)
        self.assertEqual((tuple(selected.shape), selected.dtype), ((meta.positions.numel(), W.budget + ratio - 1),
                                                                   torch.int32))
        row, reached = 0, set()
        for i, (_seq, _slot, ctx, length) in enumerate(f.requests):
            history, cos, sin = f.histories[i]
            for p in range(ctx, ctx + length):
                blocks = (p + 1) // ratio
                scores = torch.relu(f.queries[row].float() @ f.keys[i][:blocks].float().T).sum(0) / math.sqrt(W.idx_dim)
                with self.subTest(row=row, position=p, blocks=blocks):
                    self.assertEqual(int(visible[row]), blocks)                        # the horizon: whole groups <= p
                    torch.testing.assert_close(logits[row, :blocks], scores, rtol=1e-5, atol=1e-5)
                    want = set(qsa_select(f.queries[row], history, p, ratio, taken, cos, sin, f.k_norm, EPS).tolist())
                    got = selected[row][selected[row] >= 0].tolist()
                    self.assertEqual(len(got), len(set(got)))
                    slack = 2 * float((logits[row, :blocks] - scores).abs().max()) if blocks else 0.0
                    ordered = scores.sort(descending=True).values
                    if blocks <= taken or float(ordered[taken - 1] - ordered[taken]) > slack:
                        self.assertEqual(set(got), want)
                    else:
                        whole = blocks * ratio
                        chosen = sorted({q // ratio for q in got if q < whole})
                        self.assertEqual(sorted(q for q in got if q < whole),
                                         [b * ratio + o for b in chosen for o in range(ratio)])
                        self.assertEqual({q for q in got if q >= whole}, {q for q in want if q >= whole})
                        taken_mask = torch.zeros(blocks, dtype=torch.bool, device=scores.device)
                        taken_mask[chosen] = True
                        self.assertEqual(len(chosen), taken)
                        self.assertGreaterEqual(float(scores[taken_mask].min()), float(scores[~taken_mask].max()) - slack)
                reached.add("fewer blocks" if blocks < taken else "the budget" if blocks == taken else "more blocks")
                reached.add("a tail" if (p + 1) % ratio else "no tail")
                row += 1
        self.assertEqual(reached, {"fewer blocks", "the budget", "more blocks", "a tail", "no tail"})

    def test_chunked_rows_select_what_one_launch_does(self):
        """qsa_select_paged_tokens scores in chunks of rows under a logits workspace; two rows a chunk here."""
        from engine.kernels import qsa
        f = self.fixture(32)
        meta = f.meta
        columns = meta.page_table.shape[1] * f.key_cache.shape[1]
        args = (f.queries, f.key_cache, meta.page_table, meta.rows_req, meta.positions32, meta.lengths, W.budget, W.ratio)
        with served_kernels():
            whole = qsa.qsa_select_paged_tokens(*args)
            with mock.patch.object(qsa, "_LOGITS_WORKSPACE_BYTES", 2 * columns * 4):
                chunked = qsa.qsa_select_paged_tokens(*args)
        self.assertTrue(torch.equal(chunked, whole))


class Launches:
    """A kernel's launches, recorded by grid; the kernel itself runs."""

    def __init__(self, kernel):
        self.kernel, self.grids = kernel, []

    def __getitem__(self, grid):
        self.grids.append(tuple(grid))
        return self.kernel[grid]


@unittest.skipUnless(RUNS, RUNS_REASON)
class SparseAttentionTests(Held):
    """qsa_sparse_paged_attention against modules/sparse_attention.gqa_sparse over the same positions: the K/V rows read
    through each request's pages, -1 columns skipped wherever they sit. The kernel rounds each tile's probabilities to
    BF16 before their product with the values, and the output once; the band is two BF16 steps at the largest element
    and one in rms."""

    def test_paged_sparse_gqa_is_the_attention_over_the_selected_positions(self):
        from engine.kernels import qsa
        from engine.modules.sparse_attention import gqa_sparse
        gen = generator(41)
        D, page, pages = W.head_dim, W.block, 16
        lengths = (1, page + 3, 3 * page - 1, 4 * page + 5)       # the positions each request holds
        owners = (0, 1, 1, 2, 3, 3)                                 # the request of each query row
        nothing = 3                                                 # a row with every column -1
        requests = tuple((seq, seq + 1, 0, length) for seq, length in enumerate(lengths))
        rows_req = torch.tensor(owners, dtype=torch.int32, device=DEVICE)
        # a selection narrower than two 16-wide tiles runs in one split; wider ones split the tiles (the served width
        # 2051 on a GPU into 64 or 32 splits)
        widths = (15, 35, 67) if INTERPRET else (15, 67, W.budget + W.ratio - 1)
        split = set()
        for kv_heads in (W.kv_heads, 2):
            for width in widths:
                table = block_table(gen, requests, 8, page, pages)
                k_cache, _ = paged(gen, pages, page, kv_heads, D)
                v_cache, _ = paged(gen, pages, page, kv_heads, D)
                q = randn(gen, len(owners), W.heads, D)
                indices = torch.full((len(owners), width), -1, dtype=torch.int32)
                slots = torch.zeros(len(owners), width, dtype=torch.int32)         # the valid prefix, physical rows
                valid = torch.zeros(len(owners), dtype=torch.int32)
                pages_of = table.cpu()
                for r, owner in enumerate(owners):
                    count = 0 if r == nothing else min(width, lengths[owner])
                    columns = torch.randperm(width, generator=gen)[:count].sort().values
                    indices[r, columns] = torch.randperm(lengths[owner], generator=gen)[:count].to(torch.int32)
                    held = indices[r][indices[r] >= 0].long()
                    slots[r, :count] = pages_of[owner, held // page] * page + held % page
                    valid[r] = count
                attend = Launches(qsa._qsa_sparse_paged_gqa_splitk_kernel)
                merge = Launches(qsa._qsa_merge_splitk_kernel)
                with served_kernels(), mock.patch.object(qsa, "_qsa_sparse_paged_gqa_splitk_kernel", attend), \
                        mock.patch.object(qsa, "_qsa_merge_splitk_kernel", merge):
                    out = qsa.qsa_sparse_paged_attention(q, k_cache, v_cache, indices.to(DEVICE), table, rows_req)
                (grid,) = attend.grids
                splits = grid[2]
                split.add(splits > 1)
                reference = gqa_sparse(q, k_cache.reshape(pages * page, kv_heads, D),
                                       v_cache.reshape(pages * page, kv_heads, D), slots.to(DEVICE), valid.to(DEVICE),
                                       D ** -0.5)
                with self.subTest(kv_heads=kv_heads, width=width, splits=splits):
                    self.assertEqual(len(merge.grids), int(splits > 1))
                    self.assertWithin(out, reference, (2 * BF16_STEP, BF16_STEP), "the attended rows")
                    self.assertFalse(bool(out[nothing].any()))
        self.assertEqual(split, {False, True})                     # one split and several both ran


@unittest.skipUnless(RUNS, RUNS_REASON)
class NormRopeTests(Held):
    """qsa.norm_rope_partial at the four places net._qsa calls it, with the heads laid out as the step hands them over,
    against rmsnorm_unit_offset then apply_rope over the first rotary_dim channels, within qsa.qualify's band."""

    def test_long_context_rotation_in_fp32_without_quantisation(self):
        """FP32 inputs expose phase errors that the BF16 qualification band can hide. Check both the standalone
        rotation and the fused helper that writes the served Q/K/index heads, up to the checkpoint's last position."""
        from engine.kernels import qsa
        from engine.modules.norm import rmsnorm_unit_offset
        from engine.modules.rotary import apply_rope, rope_tables
        gen = generator(52)
        positions = torch.tensor([0, 1, 31, 4095, 32767, 32768, 65535, 131071, 131072, MAX_POSITION - 1],
                                 device=DEVICE, dtype=torch.int64)
        n, D, Di = len(positions), W.head_dim, W.idx_dim
        q = randn(gen, n, W.heads, 2 * D, dtype=torch.float32)[..., :D]
        k = randn(gen, n, 1, D, dtype=torch.float32)
        iq = randn(gen, n, W.idx_heads, Di, dtype=torch.float32)
        weights = [randn(gen, x.shape[-1], scale=.1, dtype=torch.float32) for x in (q, k, iq)]
        kc, vc, rc = (torch.zeros(1, n, 1, dim, device=DEVICE) for dim in (D, D, Di))
        slots = torch.arange(n, device=DEVICE, dtype=torch.int64)
        with served_kernels():
            qout, iqout = qsa.qsa_inputs(q, k, torch.zeros_like(k), iq, rc[0, :, 0].clone(), positions, *weights,
                                         EPS, THETA, W.rotary, kc, vc, slots, rc, slots)
            standalone = [qsa.norm_rope_partial(x, w, EPS, positions, THETA, W.rotary)
                          for x, w in zip((q, k, iq), weights)]
        cos, sin = rope_tables(positions, W.rotary, THETA, dtype=torch.float32)
        for name, x, w, got, fused in zip(("query", "key", "index query"), (q, k, iq), weights,
                                         standalone, (qout, kc[0], iqout)):
            want = apply_rope(rmsnorm_unit_offset(x, w, EPS), cos, sin)
            for entry, actual in (("standalone", got), ("fused", fused)):
                with self.subTest(head=name, entry=entry):
                    self.assertWithin(actual, want, (2e-5, 2e-5), f"{entry} {name} at long positions")

    def test_the_heads_as_the_attention_layer_hands_them_over(self):
        from engine.kernels import qsa
        from engine.modules.norm import rmsnorm_unit_offset
        from engine.modules.rotary import apply_rope, rope_tables
        gen = generator(51)
        D, Di = W.head_dim, W.idx_dim
        for n in W.rows:
            positions = torch.randint(0, MAX_POSITION, (n,), generator=gen).to(DEVICE)
            fused = randn(gen, n, W.heads, 2 * D)                          # each query head beside its gate
            projected = randn(gen, n, W.kv_heads * D + W.idx_heads * Di)   # a key head beside the index queries
            sites = {"query heads": fused[..., :D],                        # strided: a head's row is 2D wide
                     "key head": projected[:, :W.kv_heads * D].view(n, W.kv_heads, D),
                     "index queries": projected[:, W.kv_heads * D:].reshape(n, W.idx_heads, Di),
                     "pooled index key": randn(gen, n, 1, Di)}
            for site, x in sites.items():
                weight = randn(gen, x.shape[-1], scale=0.1)
                with served_kernels():
                    got = qsa.norm_rope_partial(x, weight, EPS, positions, THETA, W.rotary)
                cos, sin = rope_tables(positions, W.rotary, THETA, dtype=torch.bfloat16)
                with self.subTest(rows=n, site=site):
                    self.assertWithin(got, apply_rope(rmsnorm_unit_offset(x, weight, EPS), cos, sin), bands(qsa.qualify),
                                      site)


@unittest.skipUnless(RUNS, RUNS_REASON)
class GatedResidualTests(Held):
    """engine/kernels/gated_residual's launches against modules/hyper_connection.gated_residual (and the residual form's
    leave), within its qualify's band: a site entered, the leave with and without the next site's norm, the closing
    mixer, and the joint norm over a whole row (hc 1, the MTP fuse's)."""

    def test_a_site_its_leaves_and_the_close(self):
        from engine.kernels import gated_residual as hcr
        from engine.modules.hyper_connection import gated_residual
        from engine.modules.norm import rmsnorm_unit_offset
        gen = generator(61)
        hc, hidden, rank = W.hc, W.hidden, W.hc_rank
        width = hc * hidden
        band = bands(hcr.qualify)
        norm, down, up, inject = (randn(gen, width, scale=0.1), randn(gen, rank, width, scale=0.02),
                                  randn(gen, width, rank, scale=0.02), randn(gen, hc, width, scale=0.02))
        close_norm, close_down, close_up = (randn(gen, width, scale=0.1), randn(gen, rank, width, scale=0.02),
                                            randn(gen, width, rank, scale=0.02))
        down_inject, close_packed = hcr.pack_down_inject(down, inject), hcr.pack_down_inject(close_down, None)
        for n in W.rows:
            h, out = randn(gen, n, width), randn(gen, n, hidden)
            mixed_ref, inject_ref = gated_residual(h, norm, down, up, inject, hc, EPS)
            left_ref = h + (out.unsqueeze(-2) * inject_ref.unsqueeze(-1)).flatten(-2)
            with served_kernels():
                normed = hcr.norm_streams(h, norm, EPS, hc)
                mixed, injection = hcr.mix(normed, down_inject, up, hc)
                left = hcr.leave(h.clone(), out, inject_ref, hc)
                left_joined, left_normed = hcr.leave_norm(h.clone(), out, inject_ref, close_norm, EPS, hc)
                closed, closed_injection = hcr.mix(left_normed, close_packed, close_up, hc, inject=False)
                joint = hcr.norm_streams(h, norm, EPS, 1)
            with self.subTest(rows=n):
                self.assertWithin(normed, rmsnorm_unit_offset(h, norm, EPS, group=hidden), band, "the stream norm")
                self.assertWithin(mixed, mixed_ref, band, "the mixed input")
                self.assertWithin(injection, inject_ref, band, "the injection")
                self.assertWithin(left, left_ref, band, "the leave")
                self.assertWithin(left_joined, left_ref, band, "the leave joined to the norm")
                self.assertWithin(left_normed, rmsnorm_unit_offset(left_ref, close_norm, EPS, group=hidden), band,
                                  "the norm joined to the leave")
                self.assertWithin(closed, gated_residual(left_ref, close_norm, close_down, close_up, None, hc, EPS), band,
                                  "the closing mixer")
                self.assertIsNone(closed_injection)
                self.assertWithin(joint, rmsnorm_unit_offset(h, norm, EPS), band, "the joint norm")


@unittest.skipUnless(RUNS, RUNS_REASON)
class GdnTests(Held):
    """engine/kernels/gdn against modules/linear_attention.gdn_decay and modules/norm.rmsnorm_gated: the decay in fp32
    on both sides, so within fp32; the BF16 outputs within gdn.qualify's band."""

    def test_the_gates(self):
        from engine.kernels import gdn
        from engine.modules.linear_attention import gdn_decay
        gen = generator(71)
        heads = W.v_heads
        A_log, dt_bias = randn(gen, heads, dtype=torch.float32), randn(gen, heads, dtype=torch.float32)
        for n in W.rows:
            projected = randn(gen, n, 3 * heads, scale=4.0)                # z's tail, then b and a, as in_proj splits
            b, a = projected[:, heads:2 * heads], projected[:, 2 * heads:]
            # softplus past torch's threshold (a + dt_bias > 20), and deep enough below zero that 1 + exp(g) is 1 in fp32
            a[0, 0] = 30.0 - float(dt_bias[0])
            a[-1, -1] = -60.0
            with served_kernels():
                decay, beta = gdn.gates(a, b, A_log, dt_bias, sigmoid_beta=True)
                decay_raw, raw = gdn.gates(a, b, A_log, dt_bias, sigmoid_beta=False)
            with self.subTest(rows=n):
                reference = gdn_decay(a, A_log, dt_bias)
                self.assertEqual(decay.dtype, torch.float32)
                torch.testing.assert_close(decay, reference, rtol=1e-5, atol=0)
                self.assertTrue(torch.equal(decay_raw, decay))
                self.assertWithin(beta, torch.sigmoid(b), bands(gdn.qualify), "beta")
                self.assertTrue(torch.equal(raw, b) and raw.is_contiguous() and raw.data_ptr() != b.data_ptr())

    def test_the_output_norm(self):
        from engine.kernels import gdn
        from engine.modules.norm import rmsnorm_gated
        gen = generator(72)
        heads, dim = W.v_heads, W.v_dim
        weight = randn(gen, dim, scale=0.1) + 1
        for n in W.rows:
            core = randn(gen, n, heads, dim)
            z = randn(gen, n, 2 * heads * dim)[:, heads * dim:].view(n, heads, dim)   # a slice of the projection
            with served_kernels():
                got = gdn.gated_norm(core, z, weight, EPS)
            with self.subTest(rows=n):
                self.assertWithin(got, rmsnorm_gated(core, z, weight, EPS, "sigmoid").reshape(n, heads * dim),
                                  bands(gdn.qualify), "the gated output norm")


@unittest.skipUnless(torch is not None, "requires torch")
class CapturedStepMetaTests(unittest.TestCase):
    """Qwen38Net.step_meta's captured branch (a DeviceStep: the rows' contexts, slots and sequences on the device, the
    page table gathered at the bucket's width) against its host branch (a Step's segments) for the same decode rows,
    and both against the addresses the caches define (caches.py): kv row page * block + pos % block, the index key
    record of a closed group, the key ring cell slot * ring + pos % ring."""

    def test_the_captured_rows_address_what_the_host_step_does(self):
        from engine.profiles.qwen38.caches import QSA_KEY_RING
        from engine.profiles.qwen38.facts import BLOCK
        from engine.profiles.qwen38.net import DeviceStep, Qwen38Net, Segment, Step
        device = "cuda" if torch.cuda.is_available() and not INTERPRET else "cpu"
        ratio = 4                                                   # the checkpoint's indexer_compress_ratio
        per = BLOCK // ratio
        net = SimpleNamespace(F=SimpleNamespace(block=BLOCK, idx_ratio=ratio))
        gen = generator(81)
        # (seq, slot, ctx): a first row, a context at a block's last position, one closing a group, one past a block
        # boundary, a deep one whose bucket is wider than the shallow rows'
        rows = ((4, 2, 0), (1, 5, BLOCK - 1), (0, 1, 2 * BLOCK - 2), (5, 3, 3 * BLOCK + 17), (2, 6, 12 * BLOCK + 5))
        blocks = 24
        for tokens in (1, 2):
            for n in (1, 2, len(rows)):
                chosen = rows[:n]
                table = torch.full((6, blocks), -1, dtype=torch.int32)
                free, taken = torch.randperm(4 * blocks, generator=gen), 0
                for seq, _slot, ctx in chosen:
                    need = -(-(ctx + tokens) // BLOCK)
                    table[seq, :need] = free[taken:taken + need].to(torch.int32)
                    taken += need
                caches = SimpleNamespace(block_table=table.to(device))
                ids = torch.zeros(n * tokens, dtype=torch.int64, device=device)
                host = Qwen38Net.step_meta(net, Step(ids, tuple(Segment(seq, slot, ctx, i * tokens, tokens)
                                                                 for i, (seq, slot, ctx) in enumerate(chosen))), caches)
                as_device = lambda k: torch.tensor([row[k] for row in chosen], dtype=torch.int64, device=device)
                # the captured bucket is the table's width: wider than the host step's rung for every row set here
                captured = Qwen38Net.step_meta(net, DeviceStep(ids, as_device(2), as_device(1), as_device(0), tokens,
                                                               blocks), caches)
                with self.subTest(tokens=tokens, rows=n):
                    for name in ("positions", "positions32", "rows_req", "starts", "lengths", "slot_table", "kv_slots",
                                 "key_slots", "ring_slots"):
                        a, b = getattr(captured, name), getattr(host, name)
                        self.assertEqual((a.dtype, tuple(a.shape)), (b.dtype, tuple(b.shape)), name)
                        self.assertTrue(torch.equal(a, b), name)
                    self.assertEqual(captured.page_table.shape[1], blocks)
                    self.assertLess(host.page_table.shape[1], blocks)
                    self.assertTrue(bool((captured.page_table >= 0).all()))        # unreserved entries read as page 0
                    self.assertTrue(torch.equal(caches.block_table.cpu(), table))  # ... in the step's copy only
                    for i, (seq, slot, ctx) in enumerate(chosen):
                        reach = -(-(ctx + tokens) // BLOCK)
                        self.assertTrue(torch.equal(captured.page_table[i, :reach], host.page_table[i, :reach]), i)
                        for j in range(tokens):
                            r, p = i * tokens + j, ctx + j
                            group = p // ratio
                            key = int(table[seq, group // per]) * per + group % per if (p + 1) % ratio == 0 else -1
                            self.assertEqual((int(host.positions[r]), int(host.rows_req[r])), (p, i))
                            self.assertEqual(int(host.kv_slots[r]), int(table[seq, p // BLOCK]) * BLOCK + p % BLOCK)
                            self.assertEqual(int(host.key_slots[r]), key)
                            self.assertEqual(int(host.ring_slots[r]), slot * QSA_KEY_RING + p % QSA_KEY_RING)


if __name__ == "__main__":
    unittest.main()
