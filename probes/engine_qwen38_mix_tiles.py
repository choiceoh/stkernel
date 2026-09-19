"""Qwen3.8's mixer mean on one GB10: does tiling the hidden axis pay (carry H3)?

GLM-5.3's mHC post went from a program a token to (tokens, hidden / 512) programs on the GB10 (#634). Qwen3.8's gated
residual has three hidden-wide launches. `leave_norm` and `norm_streams` reduce along the channels (the norm's sum of
squares), so their program is a whole stream and a tile would be another rounding order. `mix_mean` reduces nothing along
them -- sigmoid(up) * normed, averaged over the hc streams, a channel at a time -- so gated_residual._mix_mean takes a
tile axis and any tile width is the one-block launch's bytes. Today that launch is one 4,096-wide block a row at 8 warps
for the model's 2,560 channels: 1,536 masked lanes.

Since #1207 a decode step of 1..16 rows folds the mean into the up product (`_up_mean`), so `mix_mean` is what an eager
step runs -- a prompt, a chunk, its tail -- and a captured step of more than 16 rows, or any step under --hc-fp8 (whose
projections are the dense lane's).

Arms, forced through gated_residual._MIX_TILE_OVERRIDE (the rule when None): tile width 256 / 512 / 1,024 / 2,048 /
4,096 x warps 1 / 2 / 4 / 8, at rows 4, 17, 32 (captured) and 64, 256, 1,024, 4,096 (eager). The inputs are what the
site's two launches before it just wrote, so they are warm in the step too: no cold arm. A chain of SITES launches over
different buffers, as a step's 97 sites are, is one timed unit.

Gate, before any timing: every arm's output equals today's launch byte for byte (and today's the torch form of
engine/modules/hyper_connection within gated_residual.qualify's band, once). An arm that differs or that the compiler
refuses is reported and not timed; today's rule failing raises.

    bash bench/fleet.sh run --gpu qwen38-mix-tiles 15 'Qwen3.8 mix_mean hidden tiles (carry H3)' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_mix_tiles

A kernel component's time on one device; no engine speed is claimed from it (CHARTER D17).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

HIDDEN, HC = 2560, 4                               # facts.hidden, facts.hc (tests hold them to probes/qwen38_config.json)
CAPTURED_ROWS, EAGER_ROWS = (4, 17, 32), (64, 256, 1024, 4096)
TILES, WARPS = (256, 512, 1024, 2048, 4096), (1, 2, 4, 8)
SITES = 8                                          # launches a timed unit chains, each over its own buffers
SITE_BYTES = 1 << 29                               # ... while their buffers fit this (a 4,096-row site is 189 MiB)
ITERATIONS, EAGER_REPEATS = 40, 9
MEMORY_CAP_GIB = 4
SEED = 20260919


def _hcr():
    from engine.kernels import gated_residual
    return gated_residual


@contextmanager
def forced(tile):
    """gated_residual._MIX_TILE_OVERRIDE set to `tile` ((width, warps) or None) inside the block, restored on any exit."""
    hcr = _hcr()
    saved = hcr._MIX_TILE_OVERRIDE
    hcr._MIX_TILE_OVERRIDE = tile
    try:
        yield
    finally:
        hcr._MIX_TILE_OVERRIDE = saved


def arms(hidden: int = HIDDEN) -> list:
    rule = rule_tile(hidden)
    grid = [(tile, warps) for tile in TILES for warps in WARPS]
    return grid + ([rule] if rule not in grid else [])


def rule_tile(hidden: int = HIDDEN) -> tuple:
    hcr = _hcr()
    with forced(None):
        return hcr._mix_tile(hidden)


def site_count(rows: int, hidden: int = HIDDEN, hc: int = HC) -> int:
    return max(1, min(SITES, SITE_BYTES // (rows * (2 * hc + 1) * hidden * 2)))


def site_inputs(rows: int, hidden: int, hc: int, device, generator, sites: "int | None" = None):
    """`sites` (up rows, normalised streams, output) triples, BF16: a step's sites read different activations."""
    sites = site_count(rows, hidden, hc) if sites is None else sites
    make = lambda *shape: torch.randn(*shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    return [(make(rows, hc * hidden), make(rows, hc * hidden), torch.empty(rows, hidden, device=device,
                                                                           dtype=torch.bfloat16)) for _ in range(sites)]


def launch(inputs, hidden: int, hc: int):
    """mix_mean over every site's buffers, as gated_residual.mix launches it (the hook read here)."""
    import triton
    hcr = _hcr()
    tile, warps = hcr._mix_tile(hidden)
    for up_rows, normed, out in inputs:
        rows = up_rows.shape[0]
        hcr._mix_mean[(rows, triton.cdiv(hidden, tile))](up_rows, normed, out, up_rows.stride(0), normed.stride(0),
                                                         out.stride(0), float(hc), HID=hidden, BD=tile, HC=hc,
                                                         num_warps=warps)
    return [out for _, _, out in inputs]


def reference(up_rows, normed, hidden: int, hc: int):
    """engine/modules/hyper_connection's mean of the gated streams, in the torch form."""
    weights = torch.sigmoid(up_rows.float()).to(up_rows.dtype).unflatten(-1, (hc, hidden))
    return (weights * normed.unflatten(-1, (hc, hidden))).float().mean(dim=-2).to(up_rows.dtype)


def gate(inputs, hidden: int, hc: int, grid) -> dict:
    """{arm: {exact, refused?}} against today's launch; today's against the torch form within qualify's band."""
    hcr = _hcr()
    with forced(None):
        want = [out.clone() for out in launch(inputs, hidden, hc)]
    largest, rms = hcr.drift(want[0], reference(inputs[0][0], inputs[0][1], hidden, hc))
    if largest > 5e-2 or rms > 1e-2:
        raise RuntimeError(f"today's mix_mean leaves the torch form: largest {largest:.3g}, rms {rms:.3g}")
    rows = {}
    for arm in grid:
        try:
            with forced(arm):
                got = launch(inputs, hidden, hc)
            rows[arm] = dict(exact=all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(got, want)))
        except Exception as error:                                          # a tile the compiler refuses
            rows[arm] = dict(exact=False, refused=f"{type(error).__name__}: {str(error)[:160]}")
    rule = rule_tile(hidden)
    if not rows[rule]["exact"]:
        raise RuntimeError(f"today's tile {rule} forced does not hold its own bytes")
    return rows


def capture(call, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    return graph


def timings(units: dict, repeats: int, sites: int) -> dict:
    """{arm: median and minimum us a site}: each unit is `sites` launches; the order alternates every repeat."""
    order = list(units)
    for run in units.values():
        run()
    torch.cuda.synchronize()
    samples = {arm: [] for arm in order}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for repeat in range(repeats):
        for arm in (order if repeat % 2 == 0 else order[::-1]):
            start.record(); units[arm](); end.record(); end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000 / sites)
    return {arm: dict(median_us=round(statistics.median(s), 2), min_us=round(min(s), 2), samples=len(s))
            for arm, s in samples.items()}


def verdict(rule, gates: dict, timed: dict) -> dict:
    exact = {arm: timed[arm]["median_us"] for arm in timed if gates[arm]["exact"]}
    best = min(exact, key=lambda arm: (exact[arm], arm != rule, arm))
    return dict(rule=list(rule), rule_us=exact[rule], fastest=list(best), fastest_us=exact[best],
                fastest_over_rule=round(exact[best] / exact[rule], 4),
                inexact=[list(arm) for arm, row in gates.items() if not row["exact"]])


def case(report, rows: int, captured: bool, hidden: int = HIDDEN, hc: int = HC, grid=None, repeats=None) -> dict:
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(SEED + rows)
    inputs = site_inputs(rows, hidden, hc, device, generator)
    grid = arms(hidden) if grid is None else list(grid)
    rule = rule_tile(hidden)
    gates = gate(inputs, hidden, hc, grid)
    stream = torch.cuda.Stream()
    units = {}
    for arm in grid:
        if not gates[arm]["exact"]:
            continue
        if captured:
            with forced(arm):
                graph = capture(lambda: launch(inputs, hidden, hc), stream)
            units[arm] = graph.replay
        else:
            units[arm] = (lambda arm=arm: _eager(arm, inputs, hidden, hc))
    timed = timings(units, repeats or (ITERATIONS if captured else EAGER_REPEATS), len(inputs))
    for arm in grid:
        report("mix_tile", rows=rows, captured=captured, sites=len(inputs), tile=arm[0], warps=arm[1], **gates[arm],
               **timed.get(arm, {}))
    result = verdict(rule, gates, timed)
    report("mix_tile_verdict", rows=rows, captured=captured, **result)
    return result


def _eager(arm, inputs, hidden, hc):
    with forced(arm):
        launch(inputs, hidden, hc)


def run(output=None):
    events = []

    def report(event, **values):
        row = dict(event=event, **values)
        events.append(row)
        print(json.dumps(row), flush=True)
        if output:
            Path(output).write_text("".join(json.dumps(e) + "\n" for e in events))

    import triton
    hcr = _hcr()
    assert torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    if hcr._MIX_TILE_OVERRIDE is not None:
        raise RuntimeError("the tile hook must start unset: today's rule is the reference")
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_CAP_GIB * 2 ** 30 / props.total_memory))
    report("device", name=props.name, torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
           hidden=HIDDEN, hc=HC, rule=list(rule_tile()), sites=SITES)
    verdicts = {}
    with torch.inference_mode():
        for rows in CAPTURED_ROWS:
            verdicts[f"captured {rows}"] = case(report, rows, True)
        for rows in EAGER_ROWS:
            verdicts[f"eager {rows}"] = case(report, rows, False)
    report("summary", **verdicts)
    return events


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
