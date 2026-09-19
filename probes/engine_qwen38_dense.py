"""Qwen3.8's dense projections on the W4A8 decode kernel and the FP8 lane: from how many rows is FP8 the faster one? (C2)

engine/kernels/dense.DenseLinear sends <= 32 rows to the W4A8 decode kernel and more to the FP8 lane ("Immutable
dispatch"), a switch measured on GLM-5.3's shapes alone: cells.DENSE_MEASURED_HIDDEN is (4096,), so the wizard marks the
dense lane `unmeasured` at Qwen3.8's hidden 2560 and the padded lane (PaddedDenseLinear: the shared expert's 160-column
down projection zero-extended to 256, DeepSeek-V4.1's 576 to 640) glue unjudged on a GPU (engine/QWEN38_CARRY.md C2).

Cases: every projection Qwen38Net.prepare_dense builds for the served facts (probes/qwen38_config.json through
engine_qwen38_cells.facts), one case per weight shape and lane (the MTP head's projections share the target's shapes),
constructed as prepare_dense constructs it over synthetic BF16 weights with no pack store -- the round-to-nearest packs a
boot without calibration serves. Plus V4.1's padded width, the wizard's dense intermediate: its shared expert's down
projection per rank, hidden x 576, from the checkpoint config tests/test_engine_kernel_shape.py pins (V4.1 has no net).

Arms, the two branches of DenseLinear.__call__ on the case's layer (a padded lane widens the input inside both):
  w4a8   engine/kernels/dense.w4_gemm on the layer's W4 pack: the branch at <= 32 rows, the most the kernel admits
  fp8    the layer's FP8Linear (Triton quantize + deep_gemm): the branch above 32 rows; prepare_dense adds no cuBLAS reader

Gates, all before the first timing. At every row count: the dispatch (the layer called the way the net calls it) equals
its branch's arm byte for byte; each arm is inside tests/test_engine_dense.py's band of F.linear over the BF16 weight (W4A8
.16; FP8 .05 under 1024 rows, .16 from there) and W4A8 within .006 of its fp32 twin (packing.mk_w4_dequant x
_mk_quant_x_ref); a padded lane equals DenseLinear over the padded weight byte for byte (GlueOnTheGpuTests' check). At the
captured row counts every arm's graph replays its eager bytes on changed inputs. A failure of what serves the row count
raises; the other arm off its band is recorded (`broken_arms`) and that cell is neither captured nor timed -- the first
run stopped on FP8 at 4 rows of the shared expert's 320x2560, 22% off while W4A8 served them
(measurements/qwen38_lane_20260917).

Timings at rows 1, 2, 4, 8, 16, 32, 48, 64, 128, 512 and 4096, W4A8 at <= 32 only. The row counts a Qwen3.8 decode graph
serves (up to max_seqs x (spec_k + 1) = 8) replay captured graphs; larger ones (prefill chunks) are eager calls, whose
samples include the host's launch time as a served prefill call pays it. `cold` follows a 128 MiB write elsewhere,
`warm` repeats the launch right after; each iteration reverses the cell order; medians. Per case and cache the summary
gives the fp8/w4a8 ratio of medians at each row count both arms ran, and the crossover: the fewest rows from which FP8
is no slower at every larger such row count (None: W4A8 is faster at 32).

    bash bench/fleet.sh run --gpu qwen38-dense 20 'Qwen3.8 dense W4A8/FP8 switch at hidden 2560' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_dense

What a run decides (engine/kernels/cells.py): whether 2560 -- and the padded widths -- join DENSE_MEASURED_HIDDEN, and
whether the 32-row switch needs a per-shape value. Component timings over synthetic weights, never a serving speed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

ROWS = (1, 2, 4, 8, 16, 32, 48, 64, 128, 512, 4096)
W4A8_ROWS = 32                  # DenseLinear.__call__'s switch, and the most rows w4_gemm admits
WEIGHT_STD = .02                # tests/test_engine_dense.py's synthetic weights
TWIN_BAND = .006                # tests/test_engine_dense.py: W4A8 against its fp32 twin
ITERATIONS = 32
FLUSH_MIB = 128                 # probes/engine_dense_cells.py's eviction write
REPLAY_TRIALS = 3
SEED = 20260917
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('engine/kernels/dense/__init__.py', 'engine/kernels/dense/kernels.cu', 'engine/kernels/dense/fp8.py',
           'engine/kernels/dense/packing.py', 'probes/engine_qwen38_dense.py')


@dataclass(frozen=True)
class Case:
    model: str                  # qwen38 | dsv41
    lane: str                   # the class the net builds: DenseLinear | PaddedDenseLinear
    rows: int                   # the weight's output rows
    cols: int                   # the weight's input width, before any padding
    projections: tuple          # what the weight is (the served keys' projection names)
    keys: tuple = ()            # the served weights of this shape, target layers then the MTP head (none for V4.1)
    name: "str | None" = None   # the calibration name prepare_dense passes for the first of them

    @property
    def label(self) -> str:
        return f"{self.model}:{'+'.join(self.projections)}:{self.rows}x{self.cols}"


def lane_for(cols: int) -> str:
    """The class Qwen38Net.prepare_dense builds for an input width: DenseLinear when it is DENSE_ALIGN-aligned, else
    PaddedDenseLinear (tests/test_probe_qwen38_dense.py holds this to prepare_dense itself)."""
    from engine.kernels.cells import DENSE_ALIGN
    return 'DenseLinear' if cols % DENSE_ALIGN == 0 else 'PaddedDenseLinear'


def projection(key: str) -> str:
    """'L3.attn.in_proj' and 'mtp.L0.attn.in_proj' -> 'attn.in_proj'."""
    parts = key.split('.')
    return '.'.join(parts[2:] if parts[0] == 'mtp' else parts[1:])


def qwen38_cases(F) -> "list[Case]":
    """One case per (lane, weight shape) among the projections Qwen38Net.prepare_dense builds for these facts, in model
    order: the served specs' shapes under Qwen38Net.dense_names -- the target layers' (the MTP head's are BF16 at the
    fleet's default, mtp_precision "bf16", and get no dense lane)."""
    from engine.profiles.qwen38 import specs
    from engine.profiles.qwen38.net import Qwen38Net
    shapes = {s.name: tuple(s.shape) for s in specs.all_specs(F, mtp=True)}
    grouped = {}
    for key, name in Qwen38Net.dense_names(shapes).items():
        if key.startswith("mtp."):
            continue
        rows, cols = shapes[key]
        grouped.setdefault((lane_for(cols), rows, cols), []).append((key, name))
    return [Case('qwen38', lane, rows, cols, tuple(dict.fromkeys(projection(key) for key, _ in members)),
                 tuple(key for key, _ in members), members[0][1])
            for (lane, rows, cols), members in grouped.items()]


def dsv41_cases() -> "list[Case]":
    """DeepSeek-V4.1's padded width: the dense intermediate the wizard asks (cells._dense_widths), the input of its shared
    expert's down projection per rank, whose output is hidden."""
    from engine.profiles.dsv41 import shapes
    from tests.test_engine_kernel_shape import DSV41_TEXT_CONFIG
    shape = shapes.kernel_shape(DSV41_TEXT_CONFIG)
    cols = shape.moe.dense_inter_local
    return [Case('dsv41', lane_for(cols), shape.hidden, cols, ('shared_expert.down_proj',))]


def captured_rows(F) -> int:
    """The most rows one Qwen3.8 decode graph runs: max_seqs rows of spec_k + 1 tokens (decode_graphs, fleet.MAX_SEQS).
    The fleet captures every row count up to it; a larger step is eager."""
    from engine.profiles.qwen38.fleet import MAX_SEQS
    return MAX_SEQS * (F.spec_k + 1)


def plan(cases, rows, captured) -> "list[tuple[int, int, str, str]]":
    """(case index, rows, arm, mode) for every timed cell: W4A8 where its kernel admits the rows, FP8 at every row count;
    captured up to `captured` rows (`captured_rows`), eager above."""
    return [(i, m, arm, 'captured' if m <= captured else 'eager')
            for i in range(len(cases)) for m in rows for arm in (('w4a8', 'fp8') if m <= W4A8_ROWS else ('fp8',))]


def band(arm: str, rows: int) -> float:
    """tests/test_engine_dense.py's relative-error band against F.linear on the BF16 weight, by the lane it judges there:
    .16 at <= 32 rows (W4A8), .05 at 33..1023 (FP8), .16 from 1024. FP8 forced under 33 rows keeps FP8's .05."""
    if arm == 'w4a8':
        return .16
    return .05 if rows < 1024 else .16


def relative_error(got, want) -> float:
    a, b = got.float(), want.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def gate_failures(row: dict) -> "list[str]":
    """Why one row count's served dispatch fails, empty when it passes. A NaN error fails (it is not below any band).
    The other arm is not what serves these rows: its error is `broken_arms`' business."""
    rows, failures = row['rows'], []
    if not row['dispatch_exact']:
        failures.append(f"the dispatch at {rows} rows is not its {row['dispatch']} branch byte for byte")
    error = row['relative_error'][row['dispatch']]
    if not error < row['band'][row['dispatch']]:
        failures.append(f"{row['dispatch']} at {rows} rows: relative error {error} is not below "
                        f"{row['band'][row['dispatch']]}")
    if 'twin_error' in row and not row['twin_error'] < TWIN_BAND:
        failures.append(f"w4a8 at {rows} rows: {row['twin_error']} from its fp32 twin, not below {TWIN_BAND}")
    if row.get('padded_exact') is False:
        failures.append(f"the padded lane at {rows} rows differs from DenseLinear over the padded weight")
    return failures


def broken_arms(row: dict) -> "list[str]":
    """The arms that do not serve this row count and miss their band here (NaN included): a finding the run records,
    and a cell it does not time -- a switch moved onto such an arm would serve a wrong projection."""
    return sorted(arm for arm, error in row['relative_error'].items()
                  if arm != row['dispatch'] and not error < row['band'][arm])


def crossover(w4a8: dict, fp8: dict) -> "int | None":
    """{rows: median us} of both arms -> the fewest rows from which FP8 is no slower at every larger row count both
    ran; None when W4A8 is faster at the largest one."""
    at = None
    for rows in sorted(set(w4a8) & set(fp8), reverse=True):
        if fp8[rows] > w4a8[rows]:
            break
        at = rows
    return at


def summary(cold, warm, rows) -> dict:
    c, w = statistics.median(cold), statistics.median(warm)
    return dict(cold_us=c, warm_us=w, cold_min_us=min(cold), warm_min_us=min(warm), cold_us_per_row=c / rows,
                warm_us_per_row=w / rows, samples=len(cold),
                cold_samples_us=[round(s, 1) for s in cold], warm_samples_us=[round(s, 1) for s in warm])


def build_lane(case: Case, weight):
    """The lane Qwen38Net.prepare_dense builds for this weight, with no pack store."""
    from engine.kernels import dense
    options = dict(prefill=True, smooth=None) if case.lane == 'PaddedDenseLinear' else {}
    return getattr(dense, case.lane)(weight, store=None, name=case.name, **options)


def served_form(layer) -> "list[str]":
    """What keeps `arms` from being the layer's two branches: prepare_dense leaves one round-to-nearest W4 pack, no
    bound input rows, no private workspace, no observer, and an FP8 lane with neither a cuBLAS nor a decode reader."""
    problems = []
    if len(layer.packs) != 1:
        problems.append(f'{len(layer.packs)} W4 packs')
    if layer.calibrated:
        problems.append('calibrated packs')
    if layer.decode_input_rows or layer.workspace is not None or layer.observer is not None:
        problems.append('bound input rows, a private workspace or an observer')
    if layer.fp8 is None or layer.fp8.cublas is not None or layer.decode_fp8 is not None:
        problems.append('no plain FP8 prefill lane')
    return problems


def arms(layer) -> dict:
    """The two branches of DenseLinear.__call__ on `layer`, each behind PaddedDenseLinear's zero extension when it pads."""
    from engine.kernels.dense import w4_gemm
    pad = getattr(layer, 'pad', 0)
    pack = layer.packs[0]

    def widen(x):
        return torch.nn.functional.pad(x, (0, pad)) if pad else x
    return {'w4a8': lambda x: w4_gemm(widen(x), pack, layer.workspace), 'fp8': lambda x: layer.fp8(widen(x))}


def elapsed_us(fn) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.


def gate(report, case, weight, layer, generator) -> "set[tuple[int, str]]":
    """The eager gates at every row count (module docstring); raises on the first row count whose served dispatch
    fails, and returns the (rows, arm) cells an unserved arm missed its band at."""
    from engine.kernels.dense import DenseLinear, extension
    from engine.kernels.dense.packing import _mk_quant_x_ref, mk_w4_dequant
    problems = served_form(layer)
    if problems:
        raise RuntimeError(f'{case.label}: not the lane prepare_dense serves: {problems}')
    arm = arms(layer)
    pad = getattr(layer, 'pad', 0)
    pack = layer.packs[0]
    twin_weight = mk_w4_dequant(pack.data, pack.scale, pack.rows, 1., pack.rowscale)
    manual = DenseLinear(torch.nn.functional.pad(weight, (0, pad))) if pad else None
    ext = extension()
    broken = set()
    for m in ROWS:
        x = torch.randn(m, case.cols, device='cuda', generator=generator).bfloat16()
        want = torch.nn.functional.linear(x, weight)
        served = layer(x)
        outs = {name: fn(x) for name, fn in arm.items() if name == 'fp8' or m <= W4A8_ROWS}
        chosen = 'w4a8' if m <= W4A8_ROWS else 'fp8'
        row = dict(case=case.label, rows=m, dispatch=chosen, dispatch_exact=bool(torch.equal(served, outs[chosen])),
                   relative_error={name: relative_error(out, want) for name, out in outs.items()},
                   band={name: band(name, m) for name in outs})
        if 'w4a8' in outs:
            widened = torch.nn.functional.pad(x, (0, pad)) if pad else x
            row['twin_error'] = relative_error(outs['w4a8'], (_mk_quant_x_ref(widened) @ twin_weight.T).bfloat16())
            row['plan'] = dict(zip(('ksr', 'units', 'blocks_per_sm'), map(int, ext.gemm2_plan(m, pack.rows, pack.cols))))
        if manual is not None:
            row['padded_exact'] = bool(torch.equal(served, manual(torch.nn.functional.pad(x, (0, pad)))))
        failures = gate_failures(row)
        missed = broken_arms(row)
        report('gate', passed=not failures, failures=failures, broken_arms=missed, **row)
        if failures:
            raise RuntimeError(f'{case.label}: {failures}')
        broken.update((m, arm) for arm in missed)
    return broken


def capture(report, case, layer, captured, generator, broken=frozenset()) -> dict:
    """{(rows, arm): (graph, static input, output)} at the captured row counts, each replaying its eager bytes; a
    broken (rows, arm) cell is not captured."""
    from probes.engine_decode_fusions import _capture
    graphs = {}
    for m in ROWS:
        if m > captured:
            continue
        x = torch.randn(m, case.cols, device='cuda', generator=generator).bfloat16()
        for name, fn in arms(layer).items():
            if (name == 'w4a8' and m > W4A8_ROWS) or (m, name) in broken:
                continue
            graph, out = _capture(lambda fn=fn, x=x: fn(x))
            graphs[m, name] = (graph, x, out)
            exact = True
            for _ in range(REPLAY_TRIALS):
                x.normal_(generator=generator)
                want = fn(x)
                graph.replay()
                exact = exact and bool(torch.equal(out, want))
            report('replay', case=case.label, rows=m, arm=name, exact=exact, trials=REPLAY_TRIALS)
            if not exact:
                raise RuntimeError(f'{case.label}: the {name} graph at {m} rows does not replay its eager bytes')
    return graphs


def run(output=None):
    sink = open(output, 'w') if output else None
    graphs = []

    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink is not None:
            sink.write(line + '\n')
            sink.flush()

    try:
        with torch.inference_mode():
            _run(report, graphs)
    finally:
        for case_graphs in graphs:
            for graph, _, _ in case_graphs.values():
                graph.reset()
        if sink is not None:
            sink.close()


def _run(report, graphs):
    from engine.kernels import cells
    from engine.kernels.dense import padded_columns
    from probes.engine_qwen38_cells import CONFIG_SHA256, facts
    assert torch.cuda.get_device_capability() == (12, 1), 'requires GB10'
    F = facts()
    captured = captured_rows(F)
    cases = qwen38_cases(F) + dsv41_cases()
    report('identity', gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
           config_sha256=CONFIG_SHA256, rows=list(ROWS), w4a8_rows=W4A8_ROWS, captured_rows=captured,
           iterations=ITERATIONS, flush_mib=FLUSH_MIB, dense_measured_hidden=list(cells.DENSE_MEASURED_HIDDEN),
           source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in SOURCES},
           scope='component timings over synthetic BF16 weights with round-to-nearest packs; no serving speed')
    for case in cases:
        report('case', case=case.label, **asdict(case), packed_cols=padded_columns(case.cols),
               target=sum(not k.startswith('mtp.') for k in case.keys), mtp=sum(k.startswith('mtp.') for k in case.keys))

    generator = torch.Generator(device='cuda').manual_seed(SEED)
    layers, inputs, broken = [], [], []
    for case in cases:
        weight = (torch.randn(case.rows, case.cols, device='cuda', generator=generator) * WEIGHT_STD).bfloat16()
        layer = build_lane(case, weight)
        broken.append(gate(report, case, weight, layer, generator))
        graphs.append(capture(report, case, layer, captured, generator, broken[-1]))
        layers.append(layer)
        inputs.append({m: torch.randn(m, case.cols, device='cuda', generator=generator).bfloat16()
                       for m in ROWS if m > captured})
        report('gates_passed', case=case.label, pack=list(layer.packs[0].data.shape),
               fp8_weight=list(layer.fp8.weight[0].shape))
        del weight

    cells_ = []
    for i, m, name, mode in plan(cases, ROWS, captured):
        if (m, name) in broken[i]:
            continue
        if mode == 'captured':
            launch = graphs[i][m, name][0].replay
        else:
            launch = (lambda fn, x: lambda: fn(x))(arms(layers[i])[name], inputs[i][m])
        cells_.append((i, m, name, mode, launch))
    for *_, launch in cells_:          # the allocator's blocks and every JIT kernel, before the first sample
        launch()
        launch()
    torch.cuda.synchronize()

    flush = torch.empty(FLUSH_MIB << 20, dtype=torch.uint8, device='cuda')
    samples = [([], []) for _ in cells_]
    order = list(range(len(cells_)))
    for iteration in range(ITERATIONS):
        for c in (order if iteration % 2 == 0 else order[::-1]):
            launch = cells_[c][4]
            flush.zero_()
            torch.cuda.synchronize()
            samples[c][0].append(elapsed_us(launch))
            samples[c][1].append(elapsed_us(launch))

    medians = {}
    for (i, m, name, mode, _), (cold, warm) in zip(cells_, samples):
        row = summary(cold, warm, m)
        medians[i, name, m] = row
        case = cases[i]
        report('timing', case=case.label, model=case.model, lane=case.lane, shape=[case.rows, case.cols], rows=m,
               arm=name, mode=mode, **row)

    crossovers = {}
    for i, case in enumerate(cases):
        crossovers[case.label] = {}
        for cache in ('cold', 'warm'):
            timed = {arm: {m: medians[i, arm, m][cache + '_us'] for m in ROWS if (i, arm, m) in medians}
                     for arm in ('w4a8', 'fp8')}
            both = sorted(set(timed['w4a8']) & set(timed['fp8']))
            at = crossover(timed['w4a8'], timed['fp8'])
            crossovers[case.label][cache] = at
            report('crossover', case=case.label, model=case.model, lane=case.lane, shape=[case.rows, case.cols],
                   cache=cache, crossover_rows=at, switch_rows=W4A8_ROWS,
                   modes={m: 'captured' if m <= captured else 'eager' for m in both},
                   fp8_over_w4a8={m: timed['fp8'][m] / timed['w4a8'][m] for m in both},
                   fp8_faster_rows=[m for m in both if timed['fp8'][m] < timed['w4a8'][m]])
    report('complete', status='PASS', crossovers=crossovers, switch_rows=W4A8_ROWS,
           dense_measured_hidden=list(cells.DENSE_MEASURED_HIDDEN),
           broken_arms={case.label: sorted([m, arm] for m, arm in broken[i]) for i, case in enumerate(cases)})


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
