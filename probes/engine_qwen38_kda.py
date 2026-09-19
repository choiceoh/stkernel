"""Qwen3.8's GDN recurrence on the KDA kernels at its per-rank cell on one GB10: which value tile BV is exact, which
fastest (carry C3).

Qwen3.8 serves GatedDeltaNet -- one log-decay per head -- on the KDA kernels: a captured decode step runs
kda/ring.recurrent_gdn_ring_rows, the ring kernel computing GDN's decay from the in_proj columns in its own launch
(engine/profiles/qwen38/net._gdn_rows; HEAD_GATE, carry K1), the functional recurrent lane is
fused_recurrent_kda(compute_gate=False) over linear_decay.per_channel (kda/kda.py, glue), and prefill is
kda/chunk_decay.chunk_kda_with_decay (glue). A rank's cell is 4 key / 12 value heads x 128 x 128 with a state ring of
SPEC_K + 1 cells (the profile's K, facts.SPEC_K = 3): a C=1 step is one row of K + 1 tokens, C=2 two rows. Both recurrent launchers tile the value axis by BV and
pick 16 only at GLM-5.3's cell (16/16 x 128, T <= 6; kda/ring.py records that BV=16 at seven tokens changed rollback
results in the GPU exact gate), min(next_power_of_2(V), 8) = 8 everywhere else. This cell has served BV=8 unmeasured:
cells.KDA_MEASURED_CELLS names GLM-5.3's cell only.

Arms -- each tile forced through the launchers' probe hooks (kda/ring._BV_OVERRIDE, kda/kda._BV_OVERRIDE):
  ring        recurrent_gdn_ring_rows, rows 1/2/4 x tokens 1..K+1    BV 8 | 16 | 32
  ring_eager  recurrent_gdn_ring, one row x tokens 1..K+1            BV 8 | 16 | 32, gated and not timed: net._gdn's
              uncaptured decode (host slot and context, its own specialization) under the same rule in `_recurrent`
  recurrent   fused_recurrent_kda(compute_gate=False), tokens 1/2/4   BV 8 | 16 | 32
  prefill     chunk_kda_with_decay at 128/1024/8192 tokens            no tile: the record's prefill baseline

Gates, all before any timing:
  launcher  a stand-in kernel object records the tile each launcher passes and runs nothing: today's rule passes 8 at
            this cell, the hook the forced tile
  eager     on identical inputs and identical initial ring copies, every arm's output and every ring cell or state it
            writes must hold today's rule's bytes -- the ring over a chain of steps (row 0 opening a sequence at context
            0, every draft accepted, a rejected draft's rollback, accepted again) beside a slot no row addresses
  replay    the captured graphs that are timed replay the same chain and must hold the same bytes
A failure of the rule itself raises: BV 8 forced is the rule's own tile, so it must match eagerly and replayed, and the
rule must write the addressed slots, leave the unaddressed one alone and stay finite. A wider tile that differs, or that
the compiler refuses, is reported (`inexact`, `refused`) and not timed.

Timings (probes/engine_kda_ring_bench.py's method): captured graph replays. The ring field holds a step's 36 GDN layers x
4 slots and every replay reads the next layer's slots through the device slot vector, so no timed launch meets the ring
it wrote last in cache; `cold` follows a 64 MiB write elsewhere, `warm` replays right after; cases alternate their order
every iteration; medians over iterations x layers. The prefill arm runs eager after its autotuners.

The decision it feeds (engine/QWEN38_CARRY.md C3): whether (4, 12, 128, 128) joins cells.KDA_MEASURED_CELLS and whether
the BV rule (kda/ring.py `_recurrent`, kda/kda.py fused_recurrent_kda_fwd) gets a Qwen3.8 branch. The `verdict` rows name
the exact tiles and the fastest of them per (rows, tokens); `summary` holds every verdict in one line.

    bash bench/fleet.sh run --gpu qwen38-kda 40 'Qwen3.8 GDN on the KDA kernels: BV 8/16/32 exact gate and timings' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_kda

Synthetic tensors at the served strides (the lane host has no Qwen3.8 checkpoint): q/k/v are views of one conv output
row (net._heads); the ring arms read a and b as the in_proj row's last columns (net._gdn_rows' split, so the stride the
kernel compiles is the served one) with fp32 A_log and dt_bias; the functional and prefill arms take the fp32 per-head
decay those give (read through a stride-0 channel axis) and raw bf16 beta logits (sigmoided for prefill); the state ring
is fp32 (facts.GDN_STATE_DTYPE). The field's slot stride is dense where the served arena's is a whole slot's bytes: a
constexpr of the ring address arithmetic, not of the recurrence. The GDN entries refuse a KDA cell, so `run` binds
Qwen3.8's kernel shape from probes/qwen38_config.json.
"""
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# Qwen3.8-Flash-Next per rank at TP=4 (engine/profiles/qwen38/shapes.kernel_shape: 16 key and 48 value heads over four
# ranks, 128 wide; facts.SPEC_K); tests/test_probe_qwen38_kda.py holds these to the derived kernel shape
from engine.profiles.qwen38.facts import SPEC_K   # noqa: E402 -- the profile owns K, never this probe
K_HEADS, V_HEADS, DIM = 4, 12, 128
RING_CELLS = SPEC_K + 1                 # GDN states a slot keeps: one per verify position (net.rec_ring)
GDN_LAYERS = 36                         # a step's GDN layers: 48 layers, a QSA layer closing every four
MAX_ROWS = 4
RULE_BV = 8                             # min(next_power_of_2(128), 8): not GLM-5.3's 16/16 cell, so no BV=16 branch
BVS = (8, 16, 32)
RING_CASES = tuple((rows, tokens) for rows in (1, 2, 4) for tokens in range(1, RING_CELLS + 1))
RING_EAGER_TOKENS = tuple(range(1, RING_CELLS + 1))
RECURRENT_TOKENS = (1, 2, 4)
PREFILL_TOKENS = (128, 1024, 8192)
RECURRENT_SETS = 3                      # input sets a recurrent gate runs: a carried state, a fresh zero one, a carried one
ITERATIONS = 12
TRASH_MIB = 64
PREFILL_WARMUP, PREFILL_REPEATS = 3, 9
MEMORY_CAP_GIB = 4                      # far inside the single-GPU lane's budget beside production
SEED = 20260917


@dataclass(frozen=True)
class Cell:
    """A rank's linear-attention cell: key heads, value heads, the square state width, the state ring's cells."""
    k_heads: int = K_HEADS
    v_heads: int = V_HEADS
    dim: int = DIM
    ring_cells: int = RING_CELLS

    @property
    def qkv(self) -> int:
        """The conv output row q|k|v (facts.qkv_local)."""
        return (2 * self.k_heads + self.v_heads) * self.dim

    @property
    def in_proj(self) -> int:
        """The in_proj row q|k|v|z|b|a (net._gdn_rows' split): the row stride the ring kernel reads a and b through."""
        return self.qkv + self.v_heads * self.dim + 2 * self.v_heads

    @property
    def state_bytes(self) -> int:
        return self.v_heads * self.dim * self.dim * 4                # one fp32 [HV, K, V] state


QWEN38 = Cell()


# -- the hooks and the launch they reach ---------------------------------------------------------------------------------
def _kda_modules():
    from engine.kernels.kda import kda, ring
    return ring, kda


@contextmanager
def forced_bv(bv):
    """Both launchers' value tile forced to `bv` (None: today's rule) inside the block; what they held comes back on any
    exit. The hooks are read when a launcher runs, so a captured graph keeps the tile it was captured with."""
    ring, kda = _kda_modules()
    saved = ring._BV_OVERRIDE, kda._BV_OVERRIDE
    ring._BV_OVERRIDE = kda._BV_OVERRIDE = bv
    try:
        yield
    finally:
        ring._BV_OVERRIDE, kda._BV_OVERRIDE = saved


def at_bv(bv, launch):
    with forced_bv(bv):
        return launch()


class LaunchRecorder:
    """Stands in for the recurrent kernel object: `kernel[grid](**meta)` records the grid and the tiles, runs nothing."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **meta):
            self.launches.append(dict(grid=tuple(int(g) for g in grid), BK=meta["BK"], BV=meta["BV"]))
        return launch


@contextmanager
def recorded_launches():
    """Both launchers' kernel object swapped for a LaunchRecorder inside the block, restored on any exit."""
    ring, kda = _kda_modules()
    saved = ring.fused_recurrent_gated_delta_rule_fwd_kernel, kda.fused_recurrent_gated_delta_rule_fwd_kernel
    recorder = LaunchRecorder()
    ring.fused_recurrent_gated_delta_rule_fwd_kernel = kda.fused_recurrent_gated_delta_rule_fwd_kernel = recorder
    try:
        yield recorder
    finally:
        ring.fused_recurrent_gated_delta_rule_fwd_kernel, kda.fused_recurrent_gated_delta_rule_fwd_kernel = saved


def launcher_tile(launch, bv=None) -> dict:
    """The launch `launch()` makes with the hook at `bv` -- {grid, BK, BV} -- recorded without running a kernel."""
    with forced_bv(bv), recorded_launches() as recorder:
        launch()
    if len(recorder.launches) != 1:
        raise RuntimeError(f"expected one recurrent launch, recorded {len(recorder.launches)}")
    return recorder.launches[0]


# -- inputs at the served strides ------------------------------------------------------------------------------------------
@dataclass
class Inputs:
    """One launch's inputs as net._gdn_rows hands them to the lanes: q/k/v views of the conv output y [N, qkv]
    (net._heads); a and b [1, N, HV] views of the in_proj row [N, in_proj] with fp32 A_log and dt_bias [HV] (the ring
    arms); the per-head fp32 decay [1, N, HV] they give (gdn.gates) and raw beta logits [1, N, HV] in y's dtype (the
    functional and prefill arms)."""
    y: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    decay: torch.Tensor
    beta: torch.Tensor
    proj: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor


def served_inputs(cell, n, device, dtype=torch.bfloat16) -> Inputs:
    y = torch.zeros(n, cell.qkv, device=device, dtype=dtype)
    qk = cell.k_heads * cell.dim
    q, k, v = y.split([qk, qk, cell.v_heads * cell.dim], dim=-1)
    decay = torch.zeros(n, cell.v_heads, device=device, dtype=torch.float32)
    beta = torch.zeros(n, cell.v_heads, device=device, dtype=dtype)
    proj = torch.zeros(n, cell.in_proj, device=device, dtype=dtype)
    _, _, b, a = proj.split([cell.qkv, cell.v_heads * cell.dim, cell.v_heads, cell.v_heads], dim=-1)
    # exp(A_log) 1 and dt_bias -1: the decay -softplus(a - 1) of a ~ N(0, 2) spans exp(decay) about (0.007, 0.999)
    A_log = torch.zeros(cell.v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.full((cell.v_heads,), -1.0, device=device, dtype=torch.float32)
    return Inputs(y, q.reshape(1, n, cell.k_heads, cell.dim), k.reshape(1, n, cell.k_heads, cell.dim),
                  v.reshape(1, n, cell.v_heads, cell.dim), decay[None], beta[None], proj, a[None], b[None], A_log, dt_bias)


def fill(inputs, seed) -> None:
    """One step's values written in place (graphs read these tensors): generic rows, decay logits spread so exp(decay)
    spans about (0.007, 0.999), beta logits around zero; the functional arms' decay and beta are the ring arms' own.
    A seed writes the same bytes every time on the same device."""
    device = inputs.y.device
    gen = torch.Generator(device=device).manual_seed(seed)

    def normal(shape):
        return torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    inputs.y.copy_(normal(inputs.y.shape))
    inputs.proj.copy_(normal(inputs.proj.shape))
    inputs.a.copy_(normal(inputs.a.shape) * 2)
    inputs.b.copy_(normal(inputs.b.shape) * 2)
    inputs.decay.copy_(-inputs.A_log.exp() * torch.nn.functional.softplus(inputs.a.float() + inputs.dt_bias))
    inputs.beta.copy_(inputs.b)


# -- the exact gate's arithmetic ------------------------------------------------------------------------------------------
def first_difference(got, want):
    """None when `got` holds `want`'s shape, dtype and bytes (NaN-safe; -0.0 is not 0.0). Otherwise how many elements
    differ, the first one's flat index and the largest finite |difference| among them."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return dict(shape=list(got.shape), want_shape=list(want.shape), dtype=str(got.dtype), want_dtype=str(want.dtype))
    size = got.element_size()
    a = got.detach().contiguous().view(-1).view(torch.uint8).view(-1, size)
    b = want.detach().contiguous().view(-1).view(torch.uint8).view(-1, size)
    differs = (a != b).any(dim=1)
    count = int(differs.sum())
    if not count:
        return None
    delta = (got.detach().float().reshape(-1) - want.detach().float().reshape(-1)).abs()[differs]
    finite = delta[torch.isfinite(delta)]
    return dict(elements=count, of=got.numel(), first=int(differs.nonzero()[0, 0]),
                max_abs=float(finite.max()) if finite.numel() else None, nonfinite=int(delta.numel() - finite.numel()))


def compare_records(got, want):
    """The first step and tensor where two runs' records -- per step, ((name, tensor), ...) -- differ; None when none does."""
    if len(got) != len(want):
        return dict(steps=len(got), want_steps=len(want))
    for step, (a, b) in enumerate(zip(got, want)):
        for (name, x), (_, y) in zip(a, b):
            difference = first_difference(x, y)
            if difference is not None:
                return dict(step=step, tensor=name, **difference)
    return None


def ring_chain(rows, tokens):
    """(slots, contexts) per step of the ring gate. Rows address their slots in reverse order; row 0 opens a sequence at
    context 0 (the masked initial load) while the others continue at mixed parities; then every draft is accepted
    (+tokens), a draft is rejected (+1: the step re-reads the state after the previous step's first token), and every
    draft is accepted again."""
    slots = tuple(range(rows - 1, -1, -1))
    contexts = (0,) + tuple(4095 + row for row in range(1, rows))
    steps = []
    for advance in (0, tokens, 1, tokens):
        contexts = tuple(c + advance for c in contexts)
        steps.append((slots, contexts))
    return tuple(steps)


# -- cases -------------------------------------------------------------------------------------------------------------------
@dataclass
class RingCase:
    cell: Cell
    rows: int
    tokens: int
    inputs: Inputs
    field: torch.Tensor         # [slots, ring cells, HV, K, V] fp32
    slots: torch.Tensor         # [rows] int64: the step's device vectors
    contexts: torch.Tensor
    region: torch.Tensor        # field[:rows + 1]: the gated slots and slot `rows`, which no row addresses
    initial: torch.Tensor       # the region's bytes every gated run starts from
    seed: int
    eager: bool = False         # the one-row entry net._gdn calls: the slot's own ring, host slot 0 and context

    def launch(self):
        from engine.kernels.kda.ring import recurrent_gdn_ring, recurrent_gdn_ring_rows
        i = self.inputs
        if self.eager:
            ring = self.field[int(self.slots[0])][None]                 # net._gdn: rec[None], slot 0, s.ctx
            return recurrent_gdn_ring(i.q, i.k, i.v, i.a, i.b, i.A_log, i.dt_bias, ring, 0, int(self.contexts[0]))
        return recurrent_gdn_ring_rows(i.q, i.k, i.v, i.a, i.b, i.A_log, i.dt_bias, self.field, self.slots,
                                       self.contexts)


def ring_case(cell, rows, tokens, field, dtype=torch.bfloat16, eager=False) -> RingCase:
    if not 1 <= tokens <= cell.ring_cells or not 0 < rows < field.shape[0] or (eager and rows != 1):
        raise ValueError("a ring case needs 1 <= tokens <= the ring's cells, a slot past its rows, one row when eager")
    inputs = served_inputs(cell, rows * tokens, field.device, dtype)
    region = field[:rows + 1]
    return RingCase(cell, rows, tokens, inputs, field, torch.zeros(rows, dtype=torch.int64, device=field.device),
                    torch.zeros(rows, dtype=torch.int64, device=field.device), region, region.clone(),
                    SEED + 100 * rows + tokens + (50 if eager else 0), eager)


def ring_steps(run, case):
    """The ring chain through `run()` (an eager launch or a graph replay) from the case's initial bytes: per step the
    output and the whole region."""
    case.region.copy_(case.initial)
    records = []
    for step, (slots, contexts) in enumerate(ring_chain(case.rows, case.tokens)):
        fill(case.inputs, case.seed + step)
        case.slots.copy_(torch.tensor(slots, dtype=torch.int64))
        case.contexts.copy_(torch.tensor(contexts, dtype=torch.int64))
        out = run()
        records.append((("output", out.clone()), ("ring", case.region.clone())))
    return records


def ring_rule_check(case, rule) -> None:
    """Today's rule writes every addressed slot each step, leaves slot `rows` alone and stays finite -- or the byte gate
    would compare nothing."""
    before = case.initial
    for step, ((_, out), (_, region)) in enumerate(rule):
        if not (bool(torch.isfinite(out).all()) and bool(torch.isfinite(region).all())):
            raise RuntimeError(f"ring rows={case.rows} tokens={case.tokens} step {step}: today's rule is not finite")
        if first_difference(region[case.rows], case.initial[case.rows]) is not None:
            raise RuntimeError(f"ring rows={case.rows} tokens={case.tokens} step {step}: today's rule wrote slot "
                               f"{case.rows}, which no row addresses")
        for slot in range(case.rows):
            if first_difference(region[slot], before[slot]) is None:
                raise RuntimeError(f"ring rows={case.rows} tokens={case.tokens} step {step}: slot {slot} was not written")
        before = region


@dataclass
class RecurrentCase:
    cell: Cell
    tokens: int
    inputs: Inputs
    state: torch.Tensor         # [1, HV, K, V] fp32 contiguous: the lane's initial state
    seed: int

    def launch(self):
        from engine.kernels.kda import fused_recurrent_kda
        from engine.kernels.linear_decay import per_channel
        i, dim = self.inputs, self.cell.dim
        return fused_recurrent_kda(i.q, i.k, i.v, per_channel(i.decay, dim), i.beta, scale=dim ** -0.5,
                                   initial_state=self.state, inplace_final_state=False, use_qk_l2norm_in_kernel=True,
                                   sigmoid_beta=True, compute_gate=False, state_layout="kv")


def recurrent_case(cell, tokens, device, dtype=torch.bfloat16) -> RecurrentCase:
    return RecurrentCase(cell, tokens, served_inputs(cell, tokens, device, dtype),
                         torch.zeros(1, cell.v_heads, cell.dim, cell.dim, device=device), SEED + 7000 + tokens)


def recurrent_steps(run, case):
    """RECURRENT_SETS input sets through `run()`: per set the output and every token's fp32 state."""
    records = []
    for index in range(RECURRENT_SETS):
        fill(case.inputs, case.seed + index)
        if index == 1:
            case.state.zero_()                                       # a fresh sequence: the zero state buffer
        else:
            gen = torch.Generator(device=case.state.device).manual_seed(case.seed + 100 + index)
            case.state.copy_(torch.randn(case.state.shape, generator=gen, device=case.state.device) * .5)
        out, states = run()
        records.append((("output", out.clone()), ("states", states.clone())))
    return records


def recurrent_rule_check(case, rule) -> None:
    for index, ((_, out), (_, states)) in enumerate(rule):
        if not (bool(torch.isfinite(out).all()) and bool(torch.isfinite(states).all())):
            raise RuntimeError(f"recurrent tokens={case.tokens} set {index}: today's rule is not finite")


# -- gates -------------------------------------------------------------------------------------------------------------------
def check_launcher(report, entry, launch, **key) -> None:
    """The tile each launcher passes under today's rule and under every forced tile; raises on a mismatch."""
    rule = launcher_tile(launch)
    forced = {bv: launcher_tile(launch, bv)["BV"] for bv in BVS}
    report("launcher", entry=entry, **key, rule_bv=rule["BV"], rule_grid=list(rule["grid"]),
           forced={str(bv): got for bv, got in forced.items()})
    if rule["BV"] != RULE_BV or any(got != bv for bv, got in forced.items()):
        raise RuntimeError(f"{entry} {key}: the launcher passes BV {rule['BV']} under the rule and {forced} forced")


def eager_gate(steps, launch, bvs=BVS):
    """Today's rule, then each forced tile, eagerly through `steps` (identical inputs and initial bytes every run).
    Returns the rule's records and {bv: None when exact | the first difference | refused}. Raises when the rule's own
    tile forced does not hold the rule's bytes."""
    rule = steps(lambda: at_bv(None, launch))
    arms = {}
    for bv in bvs:
        try:
            got = steps(lambda bv=bv: at_bv(bv, launch))
        except Exception as error:                # a wider tile the compiler or a launcher refuses is a result
            if bv == RULE_BV:
                raise
            arms[bv] = dict(form="eager", refused=f"{type(error).__name__}: {error}"[:500])
            continue
        difference = compare_records(got, rule)
        arms[bv] = None if difference is None else dict(form="eager", **difference)
    if arms.get(RULE_BV) is not None:
        raise RuntimeError(f"BV {RULE_BV} forced does not hold today's rule's bytes: {arms[RULE_BV]}")
    return rule, arms


def capture(launch, bv, stream):
    """Warm the launch at tile `bv` on a side stream, then capture it."""
    with forced_bv(bv):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = launch()
    return graph, output


def replay_gate(steps, graph, output, rule):
    def replay():
        graph.replay()
        return output
    return compare_records(steps(replay), rule)


def gate_case(report, entry, case, steps, stream=None, rule_check=None, **key):
    """launcher -> eager -> capture -> replay for one case. Returns ({bv: exact}, {bv: (graph, output)} for the exact
    tiles). Without a stream (the CPU dry run) nothing is captured and the eager verdict stands."""
    check_launcher(report, entry, case.launch, **key)
    rule, arms = eager_gate(lambda run: steps(run, case), case.launch)
    if rule_check is not None:
        rule_check(case, rule)
    exact, graphs = {}, {}
    for bv in BVS:
        arm = arms[bv]
        if arm is not None:
            report("refused" if "refused" in arm else "inexact", entry=entry, **key, bv=bv, **arm)
            exact[bv] = False
            continue
        if stream is None:
            report("exact", entry=entry, **key, bv=bv, eager=True, replay=None)
            exact[bv] = True
            continue
        graph, output = capture(case.launch, bv, stream)
        difference = replay_gate(lambda run: steps(run, case), graph, output, rule)
        if difference is not None:
            if bv == RULE_BV:
                raise RuntimeError(f"{entry} {key}: the captured BV {bv} graph does not replay today's rule: {difference}")
            report("inexact", entry=entry, **key, bv=bv, form="replay", **difference)
            graph.reset()
            exact[bv] = False
            continue
        report("exact", entry=entry, **key, bv=bv, eager=True, replay=True)
        exact[bv], graphs[bv] = True, (graph, output)
    return exact, graphs


# -- timings -------------------------------------------------------------------------------------------------------------
def ring_bytes(cell, rows, tokens) -> int:
    """What one ring launch moves: each row's initial state read and every token's state written."""
    return rows * (tokens + 1) * cell.state_bytes


def recurrent_bytes(cell, tokens) -> int:
    return (tokens + 1) * cell.state_bytes


def summary(cold_us, warm_us, nbytes) -> dict:
    cold, warm = statistics.median(cold_us), statistics.median(warm_us)
    return dict(cold_us=round(cold, 3), warm_us=round(warm, 3), cold_min_us=round(min(cold_us), 3),
                warm_min_us=round(min(warm_us), 3), samples=len(cold_us), mib=round(nbytes / 2**20, 3),
                cold_gbps=round(nbytes / cold / 1e3, 2), warm_gbps=round(nbytes / warm / 1e3, 2))


def verdict(exact, timings) -> dict:
    """exact {bv: bool} and the timed tiles' summaries -> the exact tiles, the fastest exact one cold and warm (the
    narrower tile on a tie), and each timed exact tile's medians over today's rule's."""
    good = sorted(bv for bv, ok in exact.items() if ok)
    timed = {bv: timings[bv] for bv in good if bv in timings}
    result = dict(exact=good, inexact=sorted(bv for bv, ok in exact.items() if not ok))
    if timed:
        result.update(fastest_cold=min(timed, key=lambda bv: (timed[bv]["cold_us"], bv)),
                      fastest_warm=min(timed, key=lambda bv: (timed[bv]["warm_us"], bv)))
    rule = timed.get(RULE_BV)
    if rule:
        result.update(cold_over_rule={str(bv): round(t["cold_us"] / rule["cold_us"], 4) for bv, t in timed.items()},
                      warm_over_rule={str(bv): round(t["warm_us"] / rule["warm_us"], 4) for bv, t in timed.items()})
    return result


def _replay_pair(graph, start, end, cold, warm):
    start.record(); graph.replay(); end.record(); end.synchronize()
    cold.append(start.elapsed_time(end) * 1000)
    start.record(); graph.replay(); end.record(); end.synchronize()
    warm.append(start.elapsed_time(end) * 1000)


def ring_timings(cases, trash):
    """{(rows, tokens, bv): (cold us, warm us)}: every replay reads the next layer's slots, after a write elsewhere."""
    order = [(rows, tokens, bv) for (rows, tokens), (_, graphs) in cases.items() for bv in graphs]
    layer_slots = {rows: [torch.arange(rows, dtype=torch.int64) + layer * MAX_ROWS for layer in range(GDN_LAYERS)]
                   for rows, _ in cases}
    for (rows, _), (case, graphs) in cases.items():
        case.contexts.copy_(torch.arange(rows, dtype=torch.int64) + 4096)
        for graph, _ in graphs.values():
            graph.replay()
    samples = {key: ([], []) for key in order}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for iteration in range(ITERATIONS):
        for key in (order if iteration % 2 == 0 else order[::-1]):
            rows, tokens, bv = key
            case, graphs = cases[rows, tokens]
            for layer in range(GDN_LAYERS):
                case.slots.copy_(layer_slots[rows][layer])
                trash.zero_()
                torch.cuda.synchronize()
                _replay_pair(graphs[bv][0], start, end, *samples[key])
    return samples


def recurrent_timings(cases, trash):
    """{(tokens, bv): (cold us, warm us)} over the static initial state, after a write elsewhere."""
    order = [(tokens, bv) for tokens, (_, graphs) in cases.items() for bv in graphs]
    for _, graphs in cases.values():
        for graph, _ in graphs.values():
            graph.replay()
    samples = {key: ([], []) for key in order}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for iteration in range(ITERATIONS):
        for key in (order if iteration % 2 == 0 else order[::-1]):
            tokens, bv = key
            graph = cases[tokens][1][bv][0]
            for _ in range(GDN_LAYERS):
                trash.zero_()
                torch.cuda.synchronize()
                _replay_pair(graph, start, end, *samples[key])
    return samples


# -- arms ----------------------------------------------------------------------------------------------------------------------
def ring_arm(report, cell=QWEN38):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(SEED)
    field = torch.empty(GDN_LAYERS * MAX_ROWS, cell.ring_cells, cell.v_heads, cell.dim, cell.dim, device=device)
    field.normal_(0, .5, generator=gen)
    trash = torch.empty(TRASH_MIB * 2**20 // 4, device=device)
    stream = torch.cuda.Stream()
    cases, exact = {}, {}
    for rows, tokens in RING_CASES:
        case = ring_case(cell, rows, tokens, field)
        exact[rows, tokens], graphs = gate_case(report, "ring", case, ring_steps, stream, ring_rule_check,
                                                rows=rows, tokens=tokens)
        cases[rows, tokens] = (case, graphs)
    eager = {}
    for tokens in RING_EAGER_TOKENS:                                  # gated only: an uncaptured step is not timed
        case = ring_case(cell, 1, tokens, field, eager=True)
        eager[tokens] = verdict(gate_case(report, "ring_eager", case, ring_steps, None, ring_rule_check,
                                          rows=1, tokens=tokens)[0], {})
        report("verdict", entry="ring_eager", rows=1, tokens=tokens, **eager[tokens])
    samples = ring_timings(cases, trash)
    verdicts = {}
    for (rows, tokens), (case, graphs) in cases.items():
        timings = {}
        for bv in graphs:
            timings[bv] = summary(*samples[rows, tokens, bv], ring_bytes(cell, rows, tokens))
            report("timing", entry="ring", rows=rows, tokens=tokens, bv=bv,
                   per_row_cold_us=round(timings[bv]["cold_us"] / rows, 3), **timings[bv])
        verdicts[rows, tokens] = verdict(exact[rows, tokens], timings)
        report("verdict", entry="ring", rows=rows, tokens=tokens, **verdicts[rows, tokens])
        for graph, _ in graphs.values():
            graph.reset()
    del cases, field, trash
    torch.cuda.empty_cache()
    return verdicts, eager


def recurrent_arm(report, cell=QWEN38):
    device = torch.device("cuda")
    trash = torch.empty(TRASH_MIB * 2**20 // 4, device=device)
    stream = torch.cuda.Stream()
    cases, exact = {}, {}
    for tokens in RECURRENT_TOKENS:
        case = recurrent_case(cell, tokens, device)
        exact[tokens], graphs = gate_case(report, "recurrent", case, recurrent_steps, stream, recurrent_rule_check,
                                          tokens=tokens)
        cases[tokens] = (case, graphs)
    samples = recurrent_timings(cases, trash)
    verdicts = {}
    for tokens, (case, graphs) in cases.items():
        timings = {}
        for bv in graphs:
            timings[bv] = summary(*samples[tokens, bv], recurrent_bytes(cell, tokens))
            report("timing", entry="recurrent", tokens=tokens, bv=bv, **timings[bv])
        verdicts[tokens] = verdict(exact[tokens], timings)
        report("verdict", entry="recurrent", tokens=tokens, **verdicts[tokens])
        for graph, _ in graphs.values():
            graph.reset()
    del cases, trash
    torch.cuda.empty_cache()
    return verdicts


def prefill_args(cell, tokens, device, dtype=torch.bfloat16, out=True) -> dict:
    """lanes.served().gdn_chunk's call at `tokens` tokens with a carried state: conv output views, the fp32 decay, beta
    after its sigmoid (gdn.gates(sigmoid_beta=True) at prefill), the state transposed to the kernel's [1, HV, V, K], a
    dense output (`out=False`: the pipeline writes over a copy of v, as the CPU interpreter needs)."""
    from engine.kernels.kda.index import single_sequence_bounds
    inputs = served_inputs(cell, tokens, device, dtype)
    fill(inputs, SEED + 9000 + tokens)
    gen = torch.Generator(device=device).manual_seed(SEED + 9500 + tokens)
    state0 = torch.randn(1, cell.v_heads, cell.dim, cell.dim, generator=gen, device=device) * .5
    return dict(q=inputs.q, k=inputs.k, v=inputs.v, decay=inputs.decay,
                beta=torch.sigmoid(inputs.beta.float()).to(dtype), scale=cell.dim ** -0.5,
                initial_state=state0.transpose(-1, -2).contiguous(), output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=single_sequence_bounds(tokens, inputs.q.device),
                out=torch.empty(inputs.v.shape, dtype=dtype, device=device) if out else None)


def prefill_arm(report, cell=QWEN38):
    from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    rows = {}
    for tokens in PREFILL_TOKENS:
        args = prefill_args(cell, tokens, torch.device("cuda"))
        o, state = chunk_kda_with_decay(**args)                     # compiles, autotunes
        if not (bool(torch.isfinite(o).all()) and bool(torch.isfinite(state).all())):
            raise RuntimeError(f"prefill {tokens}: the chunk pipeline is not finite")
        for _ in range(PREFILL_WARMUP - 1):
            chunk_kda_with_decay(**args)
        samples = []
        for _ in range(PREFILL_REPEATS):
            torch.cuda.synchronize()
            start.record()
            chunk_kda_with_decay(**args)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        median = statistics.median(samples)
        rows[tokens] = dict(tokens=tokens, median_us=round(median, 1), best_us=round(min(samples), 1),
                            samples=len(samples), tokens_per_s=round(tokens / median * 1e6, 1), carried_state=True)
        report("prefill", **rows[tokens])
        del args, o, state
        torch.cuda.empty_cache()
    return rows


def run(output=None):
    events = []

    def report(event, **values):
        row = dict(event=event, **values)
        events.append(row)
        print(json.dumps(row), flush=True)
        if output:
            Path(output).write_text("".join(json.dumps(e) + "\n" for e in events))

    import triton
    from engine.base import kernel_shape as ks
    from engine.kernels import cells
    from engine.profiles.qwen38 import shapes
    config = Path(__file__).with_name("qwen38_config.json")
    ks.bind(shapes.kernel_shape(json.loads(config.read_text())["text_config"]))     # the GDN entries refuse a KDA cell
    ring, kda = _kda_modules()
    assert torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    if ring._BV_OVERRIDE is not None or kda._BV_OVERRIDE is not None:
        raise RuntimeError("the BV hooks must start unset: today's rule is the reference")
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_CAP_GIB * 2**30 / props.total_memory))
    report("device", name=props.name, torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
           memory_cap_gib=MEMORY_CAP_GIB)
    report("cell", k_heads=K_HEADS, v_heads=V_HEADS, dim=DIM, ring_cells=RING_CELLS, gdn_layers=GDN_LAYERS,
           rule_bv=RULE_BV, bvs=list(BVS), measured_cells=[list(c) for c in cells.KDA_MEASURED_CELLS])
    with torch.inference_mode():
        ring_verdicts, eager_verdicts = ring_arm(report)
        recurrent_verdicts = recurrent_arm(report)
        prefill_rows = prefill_arm(report)
    report("summary", cell=[K_HEADS, V_HEADS, DIM, DIM], rule_bv=RULE_BV,
           ring={f"{rows}x{tokens}": v for (rows, tokens), v in ring_verdicts.items()},
           ring_eager={str(tokens): v for tokens, v in eager_verdicts.items()},
           recurrent={str(tokens): v for tokens, v in recurrent_verdicts.items()},
           prefill_median_us={str(tokens): row["median_us"] for tokens, row in prefill_rows.items()})
    return events


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)


def run_flashinfer_prefill(output=None, tokens=(1024, 2048, 4096, 8192), rounds=9):
    """The `qwen38_gdn_flashinfer` lane (engine/SM121_INTAKE.md U13): the whole engine lane
    (kernels/gdn_prefill_sm120.chunk -- q/k normalised, v made contiguous, the gate exponentiated, FlashInfer's SM120
    kernel, the states transposed) against the served chunk_kda_with_decay on the same carried-state inputs, the two
    arms alternating inside each round so production's steps land on both; the output and final state against the
    served kernel's first. Medians and minimums in us."""
    import torch
    from engine.base import kernel_shape as ks
    from engine.kernels import gdn_prefill_sm120
    from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
    from engine.profiles.qwen38 import shapes
    config = Path(__file__).with_name("qwen38_config.json")
    ks.bind(shapes.kernel_shape(json.loads(config.read_text())["text_config"]))
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_CAP_GIB * 2**30 / props.total_memory))
    rows = {}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    with torch.inference_mode():
        for t in tokens:
            args = prefill_args(QWEN38, t, torch.device("cuda"))
            state0 = args["initial_state"].transpose(-1, -2).contiguous()           # the engine's [HV, K, V]
            q, k, v, decay, beta = args["q"], args["k"], args["v"], args["decay"], args["beta"]
            arms = {"served": lambda: chunk_kda_with_decay(**args),
                    "flashinfer lane": lambda: gdn_prefill_sm120.chunk(q, k, v, decay, beta, state0)}
            o_s, st_s = arms["served"]()
            o_f, st_f = arms["flashinfer lane"]()
            err = {"o": float((o_f.float() - o_s.float()).abs().max() / o_s.float().abs().max()),
                   "state": float((st_f.float() - st_s.transpose(-1, -2).float()).abs().max()
                                  / st_s.float().abs().max())}
            samples = {name: [] for name in arms}
            for r in range(rounds):
                for name in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
                    torch.cuda.synchronize()
                    start.record()
                    arms[name]()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000)
            rows[t] = {"errors": {n: round(e, 6) for n, e in err.items()},
                       **{name: {"median_us": round(statistics.median(s), 1), "min_us": round(min(s), 1)}
                          for name, s in samples.items()}}
            rows[t]["speedup_median"] = round(rows[t]["served"]["median_us"] / rows[t]["flashinfer lane"]["median_us"], 3)
            print(json.dumps({f"gdn prefill {t}": rows[t]}), flush=True)
            del args
            torch.cuda.empty_cache()
    report = {"lane": "qwen38_gdn_flashinfer", "device": props.name, "rounds": rounds, "rows": rows}
    if output:
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report
