"""Qwen3.8's QSA launches at its per-rank cell on one GB10: which launch geometry each one wants (carry Q9), and
whether the attention reads K/V faster as one region of records (carry Q13).

engine/kernels/qsa.py launches its kernels with upstream's geometry: the sparse attention's split-K profile was tuned on
GB300 for the Qwen-Air attention shapes -- a decode step's 2,051 columns go out as 64 splits of 16-wide tiles at 4 warps
and come back through a merge launch -- the scorer tiles 64 columns a program at 2 warps, the two input launches take 4
warps. A GB10 has 48 SMs; the vLLM stack carried an unmeasured cap for exactly this (DENEB_QSA_MAX_SPLITS, removed with
D11: the kernel package reads no knobs, and "whether GB10 wants fewer splits is the wizard's measurement"). This is that
measurement. What a QSA layer launches today, 12 layers and the MTP head's a step:

    qsa_index_keys, qsa_inputs      one launch each                         warps
    qsa_mqa_paged                   the block scores, a run of rows a program  tile width, tiles a program, warps
    qsa_select.select               a captured step's block selection       warps; the columns it stops paying at (WIDEST)
    qsa_sparse_paged_attention_blocks   split launch + merge                tile width, splits, warps
    qsa_covered_paged_attention     a covered prefill step, split + merge   the sparse launch's profile, by construction

norm_rope_partial, qsa_store_cache_rows, qsa_compress_groups_with_ratio and expand_qsa_block_indices_cuda are no longer
launched by the served layer (carry Q1-Q5 folded them into the launches above), so they have no arm.

Arms -- each geometry forced through the launchers' probe hooks (qsa._SPLIT_PROFILE_OVERRIDE, _SCORE_PROFILE_OVERRIDE,
_INPUT_WARPS_OVERRIDE, qsa_select._WARPS_OVERRIDE), which keep the rule when None:
  attend          captured, N = 1..32 rows (a draft step's one row; C requests x K+1 tokens), every request past the
                  budget: 512 chosen blocks scattered over its context, the output gate in the final store as served
  attend_prefill  eager, one request's 4,096 rows deep in a 32K context: the sparse launch
  covered         eager, a fresh prompt's 2,048 rows in runs of four: the covered launch
  attend_mid / covered_mid   eager, 64 / 256 / 512 / 1,024 rows -- a short turn, a chunk's tail -- over the decode grid: where
                  upstream's rule changes tiers (8, 32, 256 and 512 programs), and the two runs before this arm existed
                  measured nothing between 32 rows and 2,048
  score           captured, N = 1, 2 and 8 rows in runs of K+1 at the 4K / 32K / 256K context buckets
  score_prefill   eager, a scoring call's rows in runs of four: 2,048 at the 4K and 32K buckets, 508 at the 256K one
                  (what the 128 MiB logits workspace holds of its 65,664 columns, in whole runs)
  select          captured, 2 rows, k 512, every context bucket's columns: the one launch against the torch form it
                  replaced, and its warps -- where the two cross is qsa_select.WIDEST
  inputs          captured, N = 2, 8, 32: qsa_index_keys then qsa_inputs
  records         carry Q13, no hook: the attention over the same K/V bytes laid out two ways -- today's block (a layer's
                  K rows, then its V rows: a position's key and value 768 rows apart) against one region of records (a
                  position's key then its value, adjacent). The kernels read both through the strides they are handed,
                  so the launch and its bytes are the same and only the addresses differ: captured at N = 2, 8, 32 and
                  the eager prefill chunk, at today's rule

Tiles stop at 64 columns for the attention and 256 for the scorer: `tl.dot` stages both operands in a block's shared
memory, and a wider tile asks for more than the device has (triton's OutOfResources; an RTX 5050 reports 101,376 bytes
against the attention's 139,264 at 128 columns of a 256-wide head, the scorer's 135,168 at 512 of a 128-wide one). One
arm past the limit stays in the attention grid so the record shows the refusal on the GB10 itself.

Gates, before any timing; a geometry that fails one is reported and not timed, and today's rule failing one raises:
  oracle   attention outputs within the served band -- two BF16 steps at the largest element, one in rms
           (tests/test_engine_qwen38_kernels.SparseAttentionTests') -- of engine/modules/sparse_attention.gqa_sparse over
           the same positions in fp32. A split profile changes where the online softmax rounds, so bytes are not asked
           of it; `from_rule` reports its drift from today's rule's output in the same two measures.
  gated    the gated store is BF16(attention * sigmoid(gate)) of the same geometry's ungated output: no element further
           than the adjacent BF16 value from torch's form, and how many differ at all is reported. Bytes hold at a decode
           step's rows; over a prefill chunk's 6.3M elements a few land on the other side of a rounding boundary, because
           the store's sigmoid is Triton's exp and the reference's is torch's (the first GB10 run: today's rule itself).
  alike    the covered launch holds the sparse launch's bytes over the covered ids at the same forced profile (carry Q10's
           claim, which the rule's own profiles are tested for).
  layout   the records layout's output equals the block layout's, byte for byte.
  exact    scores, input rows and every cache byte the input launches write equal today's rule's, byte for byte: a
           scorer that rounds differently chooses other blocks at the budget's edge, so an inexact geometry is not a
           candidate however fast. It is still timed and reported (`fastest_launched`): what a numerics-changing
           follow-up would buy. The selection's sets equal the torch form's on scores without ties.

Timings (probes/engine_qwen38_kda.py's method): captured graph replays between CUDA events, `cold` after a 64 MiB write
elsewhere and a page table pointing at pages the previous replay did not read, `warm` the replay right after; arms
alternate their order every iteration; medians. Eager arms: medians of synchronised repeats after two warm-ups. The lane
runs beside production, so read medians and ratios, not samples. `verdict` rows name the fastest passing geometry per
shape against today's rule; `summary` gathers them. None of it is an engine speed claim (CHARTER D17): the record feeds a
GB10 branch of `_split_profile` / `_score_profile`, the WIDEST cut and caches.layout's K/V placement, each judged again
by the served tests.

    bash bench/fleet.sh run --gpu qwen38-qsa-geometry 50 'Qwen3.8 QSA launch geometry: split profile, scorer, select' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_qsa_geometry

Synthetic tensors at the served strides (the lane host has no Qwen3.8 checkpoint): the paged regions are strided views of
block-major storage as caches.Qwen38Caches presents them, the step's addressing is Qwen38Net.step_meta's own.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# Qwen3.8-Flash-Next per rank at TP=4 (engine/profiles/qwen38/facts.py); tests/test_probe_qwen38_qsa_geometry.py holds
# these to the facts of probes/qwen38_config.json
HEADS, KV_HEADS, HEAD_DIM, ROTARY = 6, 1, 256, 64
IDX_HEADS, IDX_DIM, RATIO, BUDGET = 4, 128, 4, 2048
BLOCK, KEY_RING = 768, 8
THETA, EPS = 1e7, 1e-6
FIRST_BUCKET, MAX_POSITION = 4096, 262144          # net.FIRST_BUCKET; the checkpoint's context

BF16_STEP = 2 ** -7
ORACLE_BAND = (2 * BF16_STEP, BF16_STEP)           # SparseAttentionTests' band: (largest / largest, rms / rms)

# (requests, tokens a request): N = 1 -- a draft step at C=1, three of them a step at K=3 -- then 2, 4, 8, 16, 32
DECODE_STEPS = ((1, 1), (1, 2), (2, 2), (4, 2), (8, 2), (8, 4))
ATTEND_CONTEXT = 12000                             # past the budget: every row chooses 512 of ~3,000 blocks
# every power of two of splits: the first full-grid run (1, 4, 16, 64) put each step's best at a different one -- 64
# at N = 2, 16 at 4 and 8, 4 at 16, 1 at 32 -- so the steps between them are where a rule's tiers fall
ATTEND_TILES, ATTEND_SPLITS, ATTEND_WARPS = (16, 32, 64), (1, 2, 4, 8, 16, 32, 64), (2, 4, 8)
WIDE_TILE = (128, 1, 4)                            # one arm past the widest tile tl.dot compiles (see the docstring)
PREFILL_ROWS, PREFILL_CONTEXT = 4096, 28000        # a chunk's rows deep in the 32K bucket
PREFILL_TILES, PREFILL_SPLITS, PREFILL_WARPS = (16, 32, 64), (1, 4), (1, 2, 4, 8)
COVERED_ROWS = 2048                                # a fresh prompt the budget covers (facts.index_blocks groups)
MID_ROWS = (64, 256, 512, 1024)                    # eager steps between the ladder and a chunk: the rule's tiers
ORACLE_ROWS = 48                                   # the rows of an eager arm held to the fp32 oracle
SCORE_STEPS = ((1, 1), (1, 2), (4, 2))             # a draft step's one row takes the row kernel, a run the run kernel
SCORE_BUCKETS = (4096, 32768, 262144)              # context buckets (tokens) of the captured ladder
SCORE_TILES, SCORE_PROGRAM_TILES, SCORE_WARPS = (64, 128, 256), (1, 4), (1, 2, 4)
SCORE_PREFILL_SHAPES = ((2048, 4096), (2048, 32768), (508, 262144))   # (rows of a scoring call, context bucket)
SCORE_PREFILL_TILES, SCORE_PREFILL_PROGRAM_TILES, SCORE_PREFILL_WARPS = (64, 128, 256), (2, 8, 32), (1, 2, 4)
SELECT_ROWS, SELECT_WARPS = 2, (4, 8, 16)
INPUT_STEPS, INPUT_WARPS, INPUT_RULE = ((1, 2), (4, 2), (8, 4)), ((1,), (2,), (4,), (8,)), (4,)
RECORD_STEPS = ((1, 2), (4, 2), (8, 4))            # the layout arm's captured steps: N = 2, 8, 32
LAYOUTS = ("block", "records")
HOOKS = ("_SPLIT_PROFILE_OVERRIDE", "_SCORE_PROFILE_OVERRIDE", "_INPUT_WARPS_OVERRIDE")     # engine/kernels/qsa's
SELECT_HOOKS = ("_WIDEST_OVERRIDE", "_WARPS_OVERRIDE")                                       # engine/kernels/qsa_select's
PAGE_SETS = 3                                      # disjoint page sets a captured arm's replays rotate through
ITERATIONS = 24
EAGER_WARMUP, EAGER_REPEATS = 2, 7
TRASH_MIB = 64
MEMORY_CAP_GIB = 6                                 # inside the single-GPU lane's budget beside production
SEED = 20260919


@dataclass(frozen=True)
class Cell:
    """A rank's QSA cell: query and KV heads, the index heads, the compression, a paged block."""
    heads: int = HEADS
    kv_heads: int = KV_HEADS
    head_dim: int = HEAD_DIM
    rotary: int = ROTARY
    idx_heads: int = IDX_HEADS
    idx_dim: int = IDX_DIM
    ratio: int = RATIO
    budget: int = BUDGET
    block: int = BLOCK
    ring: int = KEY_RING

    @property
    def width(self) -> int:
        """Columns of an attention row: the budget's positions and the open group's tail."""
        return self.budget + self.ratio - 1

    @property
    def index_blocks(self) -> int:
        return self.budget // self.ratio

    @property
    def key_page(self) -> int:
        """Index keys a block holds."""
        return self.block // self.ratio


QWEN38 = Cell()


# -- the hooks ---------------------------------------------------------------------------------------------------------
def _qsa():
    from engine.kernels import qsa
    return qsa


@contextmanager
def forced(module=None, **hooks):
    """The named probe hooks of `module` (engine/kernels/qsa) set inside the block; what they held comes back on any
    exit. A launcher reads its hook when it runs, so a captured graph keeps the geometry it was captured with."""
    module = module or _qsa()
    unknown = [name for name in hooks if not name.endswith("_OVERRIDE") or not hasattr(module, name)]
    if unknown:
        raise ValueError(f"not a probe hook of {module.__name__}: {unknown}")
    saved = {name: getattr(module, name) for name in hooks}
    try:
        for name, value in hooks.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def split_grid(cell: Cell, tiles, splits, warps, rules=()) -> list:
    """(tile width, splits, warps) arms: every listed split a width's tiles can use -- the launcher clips a target to
    the largest power of two within its tiles -- once each, then the rule's own profiles not among them."""
    arms = []
    for block_n in tiles:
        most = 1 << ((-(-cell.width // block_n)).bit_length() - 1)
        for target in sorted({min(s, most) for s in splits}):
            arms += [(block_n, target, w) for w in warps]
    return arms + [rule for rule in rules if rule not in arms]


def rule_profile(cell: Cell, rows: int) -> tuple:
    """Today's (tile width, splits, warps) for `rows` rows: qsa._split_profile's answer with no hook set."""
    import triton
    qsa = _qsa()
    block_m = triton.next_power_of_2(cell.heads // cell.kv_heads)
    block_n, _, splits, warps = qsa._split_profile(rows, cell.kv_heads, block_m, cell.width)
    return block_n, splits, warps


# -- a step at the served strides ----------------------------------------------------------------------------------------
def paged(generator, pages: int, page: int, heads: int, dim: int, device, others: int = 2):
    """[pages, page, heads, dim] BF16 as the served caches present a region: a strided view over block-major storage,
    each page a block that holds `others` regions of this size before this one. Returns (view, storage)."""
    row = heads * dim
    stride = page * row * (others + 1)
    storage = (torch.randn(pages * stride, generator=generator) * 0.5).to(device=device, dtype=torch.bfloat16)
    return storage.as_strided((pages, page, heads, dim), (stride, row, dim, 1), others * page * row), storage


def kv_pair(generator, pages: int, cell: Cell, device, layout: str = "block"):
    """(K, V) [pages, block, kv heads, head_dim] BF16 holding the same values under either layout of a block: "block",
    caches.Qwen38Caches' today -- the K rows, then the V rows -- or "records", a position's key then its value. Both are
    strided views of one storage a page, unit stride along the head; the values depend on the generator alone."""
    if layout not in LAYOUTS:
        raise ValueError(f"a K/V layout is one of {LAYOUTS}")
    shape = (pages, cell.block, cell.kv_heads, cell.head_dim)
    k, v = (torch.randn(*shape, generator=generator) * 0.5 for _ in range(2))
    storage = torch.empty(pages, 2 * cell.block, cell.kv_heads, cell.head_dim, dtype=torch.bfloat16, device=device)
    if layout == "block":
        K, V = storage[:, :cell.block], storage[:, cell.block:]
    else:
        K, V = storage[:, 0::2], storage[:, 1::2]
    K.copy_(k)
    V.copy_(v)
    return K, V


def bucket_pages(cell: Cell, bucket: int) -> int:
    """Blocks of the page table a context bucket of `bucket` tokens captures (decode_graphs.bucket_ladder's rung)."""
    return -(-bucket // cell.block)


@dataclass
class Step:
    """A step's device addressing (Qwen38Net.step_meta's) over `sets` disjoint page sets."""
    meta: object
    tables: list            # a page table per page set; `meta.page_table` is the static one a replay reads
    requests: int
    tokens: int

    @property
    def rows(self) -> int:
        return self.requests * self.tokens

    def turn(self, index: int) -> None:
        self.meta.page_table.copy_(self.tables[index % len(self.tables)])


def step_of(cell: Cell, requests: int, tokens: int, context: int, device, *, sets: int = PAGE_SETS,
            table_blocks: "int | None" = None) -> "tuple[Step, int]":
    """`requests` segments of `tokens` rows, each `context` positions in (staggered by whole groups so the rows do not
    all close a group together), addressed as the served net does; returns (step, pages the pool must hold). The page
    sets are disjoint, in a scattered order."""
    from engine.profiles.qwen38.net import Qwen38Net, Segment, Step as NetStep
    contexts = [context + cell.ratio * r + r for r in range(requests)]
    need = max(-(-(c + tokens) // cell.block) for c in contexts)
    blocks = max(table_blocks or 0, need)
    pages = sets * requests * need
    order = torch.randperm(pages, generator=torch.Generator().manual_seed(SEED + requests * 131 + tokens)).to(torch.int32)
    tables = []
    for s in range(sets):
        table = torch.full((requests, blocks), -1, dtype=torch.int32)
        for r in range(requests):
            start = (s * requests + r) * need
            table[r, :need] = order[start:start + need]
        tables.append(table.to(device))
    segments = tuple(Segment(r, r + 1, contexts[r], r * tokens, tokens) for r in range(requests))   # slot 0 is no request's
    net = SimpleNamespace(F=SimpleNamespace(block=cell.block, idx_ratio=cell.ratio))
    ids = torch.zeros(requests * tokens, dtype=torch.int64, device=device)
    meta = Qwen38Net.step_meta(net, NetStep(ids, segments), SimpleNamespace(block_table=tables[0]))
    # a host step cuts its table to the rung its longest segment reaches; a captured one reads its bucket's whole
    # width, which is what `table_blocks` asks for -- the row addresses do not depend on it
    meta.page_table = tables[0].clone()
    return Step(meta, tables, requests, tokens), pages


def chosen_blocks(cell: Cell, step: Step, generator, device) -> torch.Tensor:
    """int32 [rows, index_blocks]: each row's blocks -- a random subset of the complete groups it sees, in a random
    order (the attention sorts them), -1 past what it sees."""
    out = torch.full((step.rows, cell.index_blocks), -1, dtype=torch.int32)
    positions, lengths, owners = (t.cpu() for t in (step.meta.positions32, step.meta.lengths, step.meta.rows_req))
    for row in range(step.rows):
        seen = int(min(positions[row] + 1, lengths[owners[row]]) // cell.ratio)
        count = min(seen, cell.index_blocks)
        out[row, :count] = torch.randperm(seen, generator=generator)[:count].to(torch.int32)
    return out.to(device)


def covered_ids(cell: Cell, step: Step, device) -> torch.Tensor:
    """The blocks of a covered step: every complete group a row sees (prefill_indexer.covered_pool_ids)."""
    from engine.modules.prefill_indexer import covered_pool_ids
    return covered_pool_ids((step.meta.positions32 + 1) // cell.ratio, cell.index_blocks)


# -- gates ---------------------------------------------------------------------------------------------------------------
def drift(got, want) -> tuple:
    from engine.kernels.gated_residual import drift as measure
    return tuple(float(x) for x in measure(got, want))


def oracle(cell: Cell, q, K, V, blocks, step: Step, rows=None) -> torch.Tensor:
    """engine/modules/sparse_attention.gqa_sparse over the positions the blocks expand to (ascending, the open group's
    tail after them), fp32 inside: the reference of the sparse launch, for `rows` (all of them when None)."""
    from engine.modules.sparse_attention import gqa_sparse
    qsa = _qsa()
    meta = step.meta
    ordered = torch.where(blocks < 0, torch.full_like(blocks, 2 ** 31 - 1), blocks).sort(dim=1).values
    ordered = torch.where(ordered == 2 ** 31 - 1, torch.full_like(ordered, -1), ordered).to(torch.int32)
    positions = qsa.expand_qsa_block_indices_cuda(ordered, meta.positions32, meta.lengths, meta.rows_req, cell.ratio,
                                                  cell.budget)
    pick = torch.arange(step.rows, device=q.device) if rows is None else rows
    positions = positions.index_select(0, pick).long()
    live = positions >= 0
    # the valid prefix first, physical rows: a stable sort keeps the ascending order of the live columns
    order = (~live).to(torch.int8).sort(dim=1, stable=True).indices
    positions, live = positions.gather(1, order), live.gather(1, order)
    owners = meta.rows_req.index_select(0, pick).long()
    pages = meta.page_table[owners[:, None], positions.clamp_min(0) // cell.block].long()
    slots = torch.where(live, pages * cell.block + positions.clamp_min(0) % cell.block, torch.zeros_like(pages))
    flat = lambda cache: cache.reshape(cache.shape[0] * cache.shape[1], cache.shape[2], cache.shape[3])
    return gqa_sparse(q.index_select(0, pick), flat(K), flat(V), slots.to(torch.int32), live.sum(1).to(torch.int32),
                      cell.head_dim ** -0.5)


def gated_reference(plain, gate):
    return (plain.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)


def bf16_steps(a, b) -> "tuple[int, int]":
    """(the largest distance in adjacent BF16 values, the elements that differ at all): BF16 bits in value order, +0 and
    -0 both 0 (probes/engine_qwen38_moe._bf16_order)."""
    def order(t):
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)
    distance = (order(a) - order(b)).abs()
    return (int(distance.max()), int((distance != 0).sum())) if a.numel() else (0, 0)


# -- timing --------------------------------------------------------------------------------------------------------------
def summary(cold_us, warm_us) -> dict:
    return dict(cold_us=round(statistics.median(cold_us), 2), warm_us=round(statistics.median(warm_us), 2),
                cold_min_us=round(min(cold_us), 2), warm_min_us=round(min(warm_us), 2), samples=len(cold_us))


def capture(launch, stream):
    """Warm `launch` on a side stream, then capture it: (graph, its output)."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = launch()
    return graph, output


def replay_timings(graphs: dict, turn, trash, iterations: int = ITERATIONS) -> dict:
    """{key: summary} of the captured arms: a `cold` replay after a write elsewhere and `turn(i)` (a page table the
    previous replay did not read), a `warm` one right after; the order alternates every iteration."""
    order = list(graphs)
    samples = {key: ([], []) for key in order}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for graph in graphs.values():
        graph.replay()
    turns = 0
    for iteration in range(iterations):
        for key in (order if iteration % 2 == 0 else order[::-1]):
            turns += 1
            turn(turns)
            trash.zero_()
            torch.cuda.synchronize()
            for bucket in samples[key]:
                start.record(); graphs[key].replay(); end.record(); end.synchronize()
                bucket.append(start.elapsed_time(end) * 1000)
    return {key: summary(*samples[key]) for key in order}


def eager_timings(launches: dict, repeats: int = EAGER_REPEATS, warmup: int = EAGER_WARMUP) -> dict:
    """{key: median and minimum us} of eager launches, synchronised, the order alternating every repeat."""
    order = list(launches)
    for key in order:
        for _ in range(warmup):
            launches[key]()
    torch.cuda.synchronize()
    samples = {key: [] for key in order}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for repeat in range(repeats):
        for key in (order if repeat % 2 == 0 else order[::-1]):
            start.record(); launches[key](); end.record(); end.synchronize()
            samples[key].append(start.elapsed_time(end) * 1000)
    return {key: dict(median_us=round(statistics.median(s), 1), min_us=round(min(s), 1), samples=len(s))
            for key, s in samples.items()}


def verdict(rule, passing: dict, timings: dict, metric: str) -> dict:
    """The fastest passing geometry by `metric` against today's rule (ties to the rule, then the smaller tuple), and
    the fastest of everything that was timed when that is another (an inexact scorer or input warps)."""
    timed = {arm: timings[arm][metric] for arm in passing if passing[arm] and arm in timings}
    result = dict(rule=list(rule), passing=len([a for a in passing if passing[a]]), failing=[list(a) for a in passing
                                                                                             if not passing[a]])
    if timed:
        best = min(timed, key=lambda arm: (timed[arm], arm != rule, arm))
        result.update(fastest=list(best), fastest_us=timed[best])
        if rule in timed:
            result.update(rule_us=timed[rule], fastest_over_rule=round(timed[best] / timed[rule], 4))
        launched = min(timings, key=lambda arm: (timings[arm][metric], arm != rule, arm))
        if launched != best:                                                # a faster geometry that failed its gate
            result.update(fastest_launched=list(launched), fastest_launched_us=timings[launched][metric])
    return result


# -- attention arms --------------------------------------------------------------------------------------------------------
@dataclass
class Attention:
    """One attention case: the step, its caches and rows, and the launch under whatever hook is set."""
    cell: Cell
    step: Step
    q: torch.Tensor
    gate: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    blocks: "torch.Tensor | None"       # None: the covered launch
    group: int = 1

    def launch(self, gated: bool = True, out=None):
        qsa, m, c = _qsa(), self.step.meta, self.cell
        gate = self.gate if gated else None
        if self.blocks is None:
            return qsa.qsa_covered_paged_attention(self.q, self.K, self.V, m.positions32, m.lengths, c.ratio, c.budget,
                                                   m.page_table, m.rows_req, out, gate=gate, group=self.group)
        return qsa.qsa_sparse_paged_attention_blocks(self.q, self.K, self.V, self.blocks, m.positions32, m.lengths,
                                                     c.ratio, c.budget, m.page_table, m.rows_req, out, gate=gate)


def attention_case(cell: Cell, requests: int, tokens: int, context: int, device, generator, *, covered: bool = False,
                   sets: int = PAGE_SETS, layout: "str | None" = None) -> Attention:
    """`layout` None: K and V each a region of its own block-major storage (the geometry arms); "block" / "records":
    kv_pair's two placements of one storage (the layout arm)."""
    step, pages = step_of(cell, requests, tokens, context, device, sets=sets)
    if layout is None:
        K, _ = paged(generator, pages, cell.block, cell.kv_heads, cell.head_dim, device)
        V, _ = paged(generator, pages, cell.block, cell.kv_heads, cell.head_dim, device)
    else:
        K, V = kv_pair(generator, pages, cell, device, layout)
    rows = step.rows
    q = torch.randn(rows, cell.heads, cell.head_dim, generator=generator).to(device=device, dtype=torch.bfloat16)
    gate = torch.randn(rows, cell.heads, cell.head_dim, generator=generator).to(device=device, dtype=torch.bfloat16)
    blocks = None if covered else chosen_blocks(cell, step, generator, device)
    return Attention(cell, step, q, gate, K, V, blocks, group=min(4, tokens) if covered else 1)


def attention_gate(case: Attention, arms, rule, sample=None) -> dict:
    """{arm: row} -- the oracle band on the ungated output (over `sample` rows when given), the gated store's bytes,
    the BF16 distance from the rule's output; for a covered case, its bytes against the sparse launch over the covered
    ids at the same profile. Raises when the rule itself fails."""
    cell, step = case.cell, case.step
    blocks = case.blocks if case.blocks is not None else covered_ids(cell, step, case.q.device)
    want = oracle(cell, case.q, case.K, case.V, blocks, step, sample)
    sparse = Attention(cell, step, case.q, case.gate, case.K, case.V, blocks)
    rows, rule_plain = {}, None
    for arm in [rule] + [a for a in arms if a != rule]:
        row = dict(geometry=list(arm))
        try:
            with forced(_SPLIT_PROFILE_OVERRIDE=arm):
                plain, gated = case.launch(gated=False), case.launch(gated=True)
                alike = None if case.blocks is not None else bool(torch.equal(plain, sparse.launch(gated=False)))
        except Exception as error:                                          # a geometry the compiler refuses
            rows[arm] = dict(row, passed=False, refused=f"{type(error).__name__}: {str(error)[:200]}")
            if arm == rule:
                raise
            continue
        if arm == rule:
            rule_plain = plain
        held = plain if sample is None else plain.index_select(0, sample)
        largest, rms = drift(held, want)
        gated_steps, gated_differ = bf16_steps(gated, gated_reference(plain, case.gate))
        passed = largest <= ORACLE_BAND[0] and rms <= ORACLE_BAND[1] and gated_steps <= 1 and alike is not False
        rows[arm] = dict(row, passed=passed, largest=round(largest, 6), rms=round(rms, 6), gated_steps=gated_steps,
                         gated_differ=gated_differ,
                         from_rule=[round(x, 6) for x in drift(plain, rule_plain)],
                         **({} if alike is None else dict(alike=alike)))
        if arm == rule and not passed:
            raise RuntimeError(f"today's rule {rule} fails its own gate: {rows[arm]}")
    return rows


def attend_arm(report, cell: Cell = QWEN38, steps=DECODE_STEPS, grid=None, iterations: int = ITERATIONS,
               context: int = ATTEND_CONTEXT) -> dict:
    """The captured sparse attention over the decode ladder, the whole grid at every step: the first GB10 run carried
    only the N = 8 step's five fastest `cold` geometries up to N = 16 and 32, and they were not the `warm` ones."""
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(SEED)
    stream = torch.cuda.Stream()
    trash = torch.empty(TRASH_MIB * 2 ** 20, dtype=torch.uint8, device=device)
    verdicts = {}
    for requests, tokens in steps:
        rows = requests * tokens
        rule = rule_profile(cell, rows)
        arms = (split_grid(cell, ATTEND_TILES, ATTEND_SPLITS, ATTEND_WARPS, [rule, WIDE_TILE]) if grid is None
                else list(grid))
        arms = arms + [rule] if rule not in arms else arms
        case = attention_case(cell, requests, tokens, context, device, generator)
        gates = attention_gate(case, arms, rule)
        graphs = {}
        for arm in arms:
            if gates[arm]["passed"]:
                with forced(_SPLIT_PROFILE_OVERRIDE=arm):
                    graphs[arm] = capture(case.launch, stream)[0]
        timings = replay_timings(graphs, case.step.turn, trash, iterations)
        for arm in arms:
            report("attend", rows=rows, requests=requests, tokens=tokens, **gates[arm], **timings.get(arm, {}))
        passing = {arm: gates[arm]["passed"] for arm in arms}
        verdicts[rows] = dict(cold=verdict(rule, passing, timings, "cold_us"),
                              warm=verdict(rule, passing, timings, "warm_us"))
        report("attend_verdict", rows=rows, **verdicts[rows])
        del graphs, case
        torch.cuda.empty_cache()
    return verdicts


def eager_attention_arm(report, event: str, cell: Cell, rows: int, context: int, grid, *, covered: bool) -> dict:
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(SEED + rows)
    case = attention_case(cell, 1, rows, context, device, generator, covered=covered, sets=1)
    rule = rule_profile(cell, rows)
    arms = list(grid) + ([rule] if rule not in grid else [])
    sample = torch.linspace(0, rows - 1, min(rows, ORACLE_ROWS), device=device).long().unique()
    gates = attention_gate(case, arms, rule, sample)
    out = torch.empty_like(case.q)

    def launch_at(arm):
        def launch():
            with forced(_SPLIT_PROFILE_OVERRIDE=arm):
                case.launch(out=out)
        return launch

    timings = eager_timings({arm: launch_at(arm) for arm in arms if gates[arm]["passed"]})
    for arm in arms:
        report(event, rows=rows, context=context, **gates[arm], **timings.get(arm, {}))
    result = verdict(rule, {arm: gates[arm]["passed"] for arm in arms}, timings, "median_us")
    report(event + "_verdict", rows=rows, **result)
    del case
    torch.cuda.empty_cache()
    return result


def layout_cases(cell: Cell, requests: int, tokens: int, context: int, device, sets: int) -> dict:
    """The same step, rows and K/V values under both layouts (one seed each), and the gate: the same bytes out."""
    cases = {layout: attention_case(cell, requests, tokens, context, device,
                                    torch.Generator().manual_seed(SEED + requests * 7 + tokens), sets=sets, layout=layout)
             for layout in LAYOUTS}
    block, records = (cases[layout] for layout in LAYOUTS)
    if not (torch.equal(block.K, records.K) and torch.equal(block.V, records.V) and torch.equal(block.q, records.q)
            and torch.equal(block.blocks, records.blocks)):
        raise RuntimeError("the two layouts must hold the same step and the same K/V values")
    if records.K.stride(1) != 2 * block.K.stride(1) or block.V.data_ptr() - block.K.data_ptr() != \
            cell.block * block.K.stride(1) * 2 or records.V.data_ptr() - records.K.data_ptr() != block.K.stride(1) * 2:
        raise RuntimeError("the layouts are not the block's and the records' strides")
    return cases


def records_arm(report, cell: Cell = QWEN38, steps=RECORD_STEPS, prefill_rows: int = PREFILL_ROWS,
                prefill_context: int = PREFILL_CONTEXT, context: int = ATTEND_CONTEXT,
                iterations: int = ITERATIONS) -> dict:
    """Carry Q13: the sparse attention at today's rule over K/V as today's block and as one region of records."""
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    trash = torch.empty(TRASH_MIB * 2 ** 20, dtype=torch.uint8, device=device)
    verdicts = {}
    for requests, tokens in steps:
        cases = layout_cases(cell, requests, tokens, context, device, PAGE_SETS)
        outputs = {layout: case.launch() for layout, case in cases.items()}
        same = bool(torch.equal(outputs["block"], outputs["records"]))
        if not same:
            raise RuntimeError("the records layout changed the attention's bytes: the kernels read strides, not layouts")
        graphs = {layout: capture(case.launch, stream)[0] for layout, case in cases.items()}

        def turn(index, cases=cases):
            for case in cases.values():
                case.step.turn(index)

        timings = replay_timings(graphs, turn, trash, iterations)
        rows = requests * tokens
        for layout in LAYOUTS:
            report("records", rows=rows, layout=layout, same_bytes=same, **timings[layout])
        verdicts[rows] = dict(cold_records_over_block=round(timings["records"]["cold_us"] / timings["block"]["cold_us"], 4),
                              warm_records_over_block=round(timings["records"]["warm_us"] / timings["block"]["warm_us"], 4))
        report("records_verdict", rows=rows, **verdicts[rows])
        del graphs, cases, outputs
        torch.cuda.empty_cache()
    cases = layout_cases(cell, 1, prefill_rows, prefill_context, device, 1)
    outs = {layout: torch.empty_like(case.q) for layout, case in cases.items()}
    for layout, case in cases.items():
        case.launch(out=outs[layout])
    if not torch.equal(outs["block"], outs["records"]):
        raise RuntimeError("the records layout changed the prefill attention's bytes")
    timings = eager_timings({layout: (lambda case=case, out=outs[layout]: case.launch(out=out))
                             for layout, case in cases.items()})
    for layout in LAYOUTS:
        report("records_prefill", rows=prefill_rows, layout=layout, same_bytes=True, **timings[layout])
    verdicts["prefill"] = dict(records_over_block=round(timings["records"]["median_us"] / timings["block"]["median_us"], 4))
    report("records_prefill_verdict", rows=prefill_rows, **verdicts["prefill"])
    return verdicts


# -- scoring arms --------------------------------------------------------------------------------------------------------------
@dataclass
class Scoring:
    cell: Cell
    step: Step
    iq: torch.Tensor
    keys: torch.Tensor
    group: int

    def launch(self):
        qsa, m = _qsa(), self.step.meta
        return qsa.qsa_mqa_paged(self.iq, self.keys, m.page_table, m.rows_req, m.positions32, m.lengths,
                                 self.cell.ratio, group=self.group)


def scoring_case(cell: Cell, requests: int, tokens: int, bucket: int, device, generator, sets: int = PAGE_SETS):
    blocks = bucket_pages(cell, bucket)
    context = max(0, blocks * cell.block * 9 // 10 - tokens - requests * (cell.ratio + 1))   # a bucket nine tenths full
    step, pages = step_of(cell, requests, tokens, context, device, sets=sets, table_blocks=blocks)
    keys, _ = paged(generator, pages, cell.key_page, 1, cell.idx_dim, device)
    iq = torch.randn(step.rows, cell.idx_heads, cell.idx_dim, generator=generator).to(device=device,
                                                                                      dtype=torch.bfloat16)
    return Scoring(cell, step, iq, keys, min(4, tokens))


def written(logits, visible) -> torch.Tensor:
    """The logits with the columns the scorer never writes (at or past a row's visible blocks) zeroed."""
    live = torch.arange(logits.shape[1], device=logits.device)[None, :] < visible[:, None]
    return torch.where(live, logits, torch.zeros_like(logits))


def scoring_gate(case: Scoring, arms, rule) -> dict:
    """{arm: row}: the written scores' bytes and the visible counts against today's rule's."""
    with forced(_SCORE_PROFILE_OVERRIDE=None):
        logits, visible = case.launch()
    want = written(logits, visible)
    rows = {}
    for arm in arms:
        try:
            with forced(_SCORE_PROFILE_OVERRIDE=arm):
                got, seen = case.launch()
        except Exception as error:
            rows[arm] = dict(geometry=list(arm), passed=False, refused=f"{type(error).__name__}: {str(error)[:200]}")
            continue
        got = written(got, seen)
        exact = bool(torch.equal(got.view(torch.int32), want.view(torch.int32)) and torch.equal(seen, visible))
        rows[arm] = dict(geometry=list(arm), passed=exact, launched=True,
                         largest_difference=float((got - want).abs().max()))
    if not rows[rule]["passed"]:
        raise RuntimeError(f"today's scoring rule {rule} forced does not hold its own bytes: {rows[rule]}")
    return rows


def score_grid(tiles, program_tiles, warps, rule) -> list:
    arms = [(n, t, w) for n in tiles for t in program_tiles for w in warps]
    return arms + ([rule] if rule not in arms else [])


def score_arm(report, cell: Cell = QWEN38, steps=SCORE_STEPS, buckets=SCORE_BUCKETS, grid=None,
              iterations: int = ITERATIONS) -> dict:
    device = torch.device("cuda")
    qsa = _qsa()
    stream = torch.cuda.Stream()
    trash = torch.empty(TRASH_MIB * 2 ** 20, dtype=torch.uint8, device=device)
    verdicts = {}
    for bucket in buckets:
        for requests, tokens in steps:
            generator = torch.Generator().manual_seed(SEED + bucket + requests)
            case = scoring_case(cell, requests, tokens, bucket, device, generator)
            rule = qsa._score_profile(case.step.rows)
            arms = score_grid(SCORE_TILES, SCORE_PROGRAM_TILES, SCORE_WARPS, rule) if grid is None else list(grid)
            arms = arms + [rule] if rule not in arms else arms
            gates = scoring_gate(case, arms, rule)
            graphs = {}
            for arm in arms:
                if gates[arm].get("launched"):                              # an inexact geometry is timed, never chosen
                    with forced(_SCORE_PROFILE_OVERRIDE=arm):
                        graphs[arm] = capture(case.launch, stream)[0]
            timings = replay_timings(graphs, case.step.turn, trash, iterations)
            key = f"{bucket}x{case.step.rows}"
            for arm in arms:
                report("score", bucket=bucket, rows=case.step.rows, columns=case.step.meta.page_table.shape[1]
                       * cell.key_page, group=case.group, **gates[arm], **timings.get(arm, {}))
            passing = {arm: gates[arm]["passed"] for arm in arms}
            verdicts[key] = dict(cold=verdict(rule, passing, timings, "cold_us"),
                                 warm=verdict(rule, passing, timings, "warm_us"))
            report("score_verdict", bucket=bucket, rows=case.step.rows, **verdicts[key])
            del graphs, case
            torch.cuda.empty_cache()
    return verdicts


def score_prefill_arm(report, cell: Cell = QWEN38, shapes=SCORE_PREFILL_SHAPES, grid=None) -> dict:
    return {f"{bucket}x{rows}": score_prefill_case(report, cell, rows, bucket, grid) for rows, bucket in shapes}


def score_prefill_case(report, cell: Cell, rows: int, bucket: int, grid=None) -> dict:
    device = torch.device("cuda")
    qsa = _qsa()
    generator = torch.Generator().manual_seed(SEED + rows + bucket)
    blocks = bucket_pages(cell, bucket)
    step, pages = step_of(cell, 1, rows, blocks * cell.block * 9 // 10 - rows, device, sets=1, table_blocks=blocks)
    keys, _ = paged(generator, pages, cell.key_page, 1, cell.idx_dim, device)
    iq = torch.randn(rows, cell.idx_heads, cell.idx_dim, generator=generator).to(device=device, dtype=torch.bfloat16)
    case = Scoring(cell, step, iq, keys, 4)
    rule = qsa._score_profile(rows)
    arms = (score_grid(SCORE_PREFILL_TILES, SCORE_PREFILL_PROGRAM_TILES, SCORE_PREFILL_WARPS, rule) if grid is None
            else list(grid) + ([rule] if rule not in grid else []))
    gates = scoring_gate(case, arms, rule)

    def launch_at(arm):
        def launch():
            with forced(_SCORE_PROFILE_OVERRIDE=arm):
                case.launch()
        return launch

    timings = eager_timings({arm: launch_at(arm) for arm in arms if gates[arm].get("launched")})
    for arm in arms:
        report("score_prefill", rows=rows, bucket=bucket, **gates[arm], **timings.get(arm, {}))
    result = verdict(rule, {arm: gates[arm]["passed"] for arm in arms}, timings, "median_us")
    report("score_prefill_verdict", rows=rows, bucket=bucket, **result)
    return result


# -- the selection ---------------------------------------------------------------------------------------------------------------
def select_arm(report, cell: Cell = QWEN38, rows: int = SELECT_ROWS, warps=SELECT_WARPS, buckets=None,
               iterations: int = ITERATIONS) -> dict:
    """The decode selection's one launch against the torch form it replaced, at every context bucket's columns: the
    crossing is where qsa_select.WIDEST belongs on this device."""
    from engine.kernels import qsa_select
    qsa = _qsa()
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    trash = torch.empty(TRASH_MIB * 2 ** 20, dtype=torch.uint8, device=device)
    if buckets is None:
        buckets, reach = [], FIRST_BUCKET
        while not buckets or buckets[-1] < MAX_POSITION:
            buckets.append(min(reach, MAX_POSITION))
            reach *= 2
    k = cell.index_blocks
    crossing = {}
    for bucket in buckets:
        columns = bucket_pages(cell, bucket) * cell.key_page
        generator = torch.Generator().manual_seed(SEED + columns)
        logits = torch.randn(rows, columns, generator=generator).to(device)          # no ties: the two forms' sets agree
        visible = torch.full((rows,), columns * 9 // 10, dtype=torch.int32, device=device)
        outs = {arm: torch.empty(rows, k, dtype=torch.int32, device=device) for arm in ("torch", *warps)}

        def torch_form(out=outs["torch"]):
            with forced(qsa_select, _WIDEST_OVERRIDE=0):
                return qsa.select_blocks(logits, visible, k, out)

        def one_launch(w):
            def launch(out=outs[w]):
                with forced(qsa_select, _WIDEST_OVERRIDE=1 << 30, _WARPS_OVERRIDE=(w,)):
                    return qsa_select.select(logits, visible, k, out)
            return launch

        want = torch_form().clone().sort(dim=1).values
        graphs, gates = {"torch": capture(torch_form, stream)[0]}, {"torch": True}
        for w in warps:
            try:
                gates[w] = bool(torch.equal(one_launch(w)().sort(dim=1).values, want))
                if gates[w]:
                    graphs[w] = capture(one_launch(w), stream)[0]
            except Exception as error:
                gates[w] = False
                report("select_refused", bucket=bucket, columns=columns, warps=w,
                       error=f"{type(error).__name__}: {str(error)[:200]}")
        timings = replay_timings(graphs, lambda index: None, trash, iterations)
        for arm, timing in timings.items():
            report("select", bucket=bucket, columns=columns, rows=rows, form=str(arm), set_equal=gates[arm], **timing)
        launches = {w: timings[w]["warm_us"] for w in warps if w in timings}
        best = min(launches, key=launches.get) if launches else None
        crossing[bucket] = dict(columns=columns, torch_us=timings["torch"]["warm_us"],
                                one_launch_us=launches.get(best), warps=best,
                                one_launch_wins=bool(launches) and launches[best] < timings["torch"]["warm_us"],
                                admitted_today=columns <= qsa_select.WIDEST)
        report("select_verdict", bucket=bucket, **crossing[bucket])
        del graphs
        torch.cuda.empty_cache()
    return crossing


# -- the input launches --------------------------------------------------------------------------------------------------------------
def inputs_arm(report, cell: Cell = QWEN38, steps=INPUT_STEPS, warps=INPUT_WARPS, iterations: int = ITERATIONS) -> dict:
    """qsa_index_keys then qsa_inputs, as a layer launches them: every warps' outputs and every cache byte against the
    rule's, then the pair's captured replay."""
    qsa = _qsa()
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    trash = torch.empty(TRASH_MIB * 2 ** 20, dtype=torch.uint8, device=device)
    verdicts = {}
    for requests, tokens in steps:
        generator = torch.Generator().manual_seed(SEED + requests * 17 + tokens)
        step, pages = step_of(cell, requests, tokens, 3 * cell.block + 1, device)
        m, rows = step.meta, step.rows
        (K, k_store), (V, v_store) = (paged(generator, pages, cell.block, 1, cell.head_dim, device) for _ in range(2))
        keys, key_store = paged(generator, pages, cell.key_page, 1, cell.idx_dim, device)
        ring, ring_store = paged(generator, requests + 1, cell.ring, 1, cell.idx_dim, device, others=0)
        stores = (k_store, v_store, key_store, ring_store)
        initial = [s.clone() for s in stores]
        width = cell.heads * 2 * cell.head_dim + 2 * cell.head_dim + cell.idx_heads * cell.idx_dim + cell.idx_dim
        proj = torch.randn(rows, width, generator=generator).to(device=device, dtype=torch.bfloat16)
        qg, k, v, idx = proj.split([cell.heads * 2 * cell.head_dim, cell.head_dim, cell.head_dim,
                                    cell.idx_heads * cell.idx_dim + cell.idx_dim], dim=-1)   # net._qsa's views
        qg = qg.view(rows, cell.heads, 2 * cell.head_dim)
        idx_q = cell.idx_heads * cell.idx_dim
        ik = idx[:, idx_q:]
        norms = [torch.randn(d, generator=generator).to(device=device, dtype=torch.bfloat16) * 0.1
                 for d in (cell.head_dim, cell.head_dim, cell.idx_dim, cell.idx_dim)]

        def launch():
            qsa.qsa_index_keys(ik, ring, m.slot_table, m.rows_req, m.starts, m.positions, m.key_slots, cell.ratio,
                               norms[3], EPS, THETA, cell.rotary, keys)
            return qsa.qsa_inputs(qg[..., :cell.head_dim], k.view(rows, 1, cell.head_dim),
                                  v.view(rows, 1, cell.head_dim), idx[:, :idx_q].view(rows, cell.idx_heads, cell.idx_dim),
                                  ik, m.positions, norms[0], norms[1], norms[2], EPS, THETA, cell.rotary, K, V,
                                  m.kv_slots, ring, m.ring_slots)

        def from_the_start(arm):
            for store, first in zip(stores, initial):
                store.copy_(first)
            with forced(_INPUT_WARPS_OVERRIDE=arm):
                q_out, iq_out = launch()
            return [q_out.clone(), iq_out.clone()] + [s.clone() for s in stores]

        want = from_the_start(None)
        if all(torch.equal(a, b) for a, b in zip(want[2:], initial)):
            raise RuntimeError("the input launches wrote nothing: the step addresses no cache row")
        arms = list(warps) + ([INPUT_RULE] if INPUT_RULE not in warps else [])
        gates = {arm: all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                          for a, b in zip(from_the_start(arm), want)) for arm in arms}
        if not gates[INPUT_RULE]:
            raise RuntimeError(f"{INPUT_RULE} forced is the rule and must hold its bytes")
        graphs = {}
        for arm in arms:                                                    # inexact warps are timed too, never chosen
            with forced(_INPUT_WARPS_OVERRIDE=arm):
                graphs[arm] = capture(launch, stream)[0]
        timings = replay_timings(graphs, lambda index: None, trash, iterations)     # the launches read no page table
        for arm in arms:
            report("inputs", rows=rows, warps=arm[0], exact=gates[arm], **timings.get(arm, {}))
        verdicts[rows] = verdict(INPUT_RULE, gates, timings, "warm_us")
        report("inputs_verdict", rows=rows, **verdicts[rows])
        del graphs
        torch.cuda.empty_cache()
    return verdicts


def run(output=None):
    events = []

    def report(event, **values):
        row = dict(event=event, **values)
        events.append(row)
        print(json.dumps(row), flush=True)
        if output:
            Path(output).write_text("".join(json.dumps(e) + "\n" for e in events))

    import triton
    from engine.kernels import qsa_select
    qsa = _qsa()
    assert torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    if any(getattr(qsa, hook) is not None for hook in HOOKS) or any(getattr(qsa_select, hook) is not None
                                                                   for hook in SELECT_HOOKS):
        raise RuntimeError("the geometry hooks must start unset: today's rule is the reference")
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_CAP_GIB * 2 ** 30 / props.total_memory))
    report("device", name=props.name, torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
           memory_cap_gib=MEMORY_CAP_GIB)
    cell = QWEN38
    report("cell", heads=cell.heads, kv_heads=cell.kv_heads, head_dim=cell.head_dim, idx_heads=cell.idx_heads,
           idx_dim=cell.idx_dim, ratio=cell.ratio, budget=cell.budget, block=cell.block, width=cell.width,
           widest=qsa_select.WIDEST, rule={str(r * t): list(rule_profile(cell, r * t)) for r, t in DECODE_STEPS})
    with torch.inference_mode():
        attend = attend_arm(report)
        prefill_grid = split_grid(cell, PREFILL_TILES, PREFILL_SPLITS, PREFILL_WARPS)
        prefill = eager_attention_arm(report, "attend_prefill", cell, PREFILL_ROWS, PREFILL_CONTEXT, prefill_grid,
                                      covered=False)
        covered = eager_attention_arm(report, "covered", cell, COVERED_ROWS, 0, prefill_grid, covered=True)
        mid_grid = split_grid(cell, ATTEND_TILES, ATTEND_SPLITS, ATTEND_WARPS)
        mid = {rows: dict(sparse=eager_attention_arm(report, "attend_mid", cell, rows, ATTEND_CONTEXT, mid_grid,
                                                     covered=False),
                          covered=eager_attention_arm(report, "covered_mid", cell, rows, 0, mid_grid, covered=True))
               for rows in MID_ROWS}
        score = score_arm(report)
        score_prefill = score_prefill_arm(report)
        select = select_arm(report)
        inputs = inputs_arm(report)
        records = records_arm(report)
    report("summary", attend={str(rows): v for rows, v in attend.items()}, attend_prefill=prefill, covered=covered,
           mid={str(rows): v for rows, v in mid.items()},
           score=score, score_prefill=score_prefill, select={str(b): v for b, v in select.items()},
           inputs={str(rows): v for rows, v in inputs.items()}, records={str(rows): v for rows, v in records.items()})
    return events


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
