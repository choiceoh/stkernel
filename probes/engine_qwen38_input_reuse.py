"""Qwen3.8's decode projections with the W4 GEMM's input reuse on one GB10: the same bytes, and what it saves (carry S2).

Input reuse quantizes a launch's input once (mk_input_pack_kernel) and has every tile read the pack
(mk_gemm_input_kernel) instead of every tile quantizing the same rows again; the tiles, k-slices, MMA and fixed-order split
fold are the ordinary launch's. GLM-5.3 takes it at K=4096 (six to eight rows); engine/kernels/dense/kernels.cu
`mk_qwen38_input_shape` now admits Qwen3.8's projections at its verify rows:

    gdn_in    4120 x 2560        qsa_in    4224 x 2560        o_proj    2560 x 1536        rows 2..8

Arms: the input mode the extension is set to (ext.set_gemm_input): 0 = the ordinary launch, 2 = reuse (the served mode).
Each arm is a captured graph of LAYERS different weights of one shape -- a step reads a different weight every
projection, so the timed launches read weights the previous one did not (48 MiB of them, past the L2) -- over one
input. `cold` follows a 64 MiB write elsewhere, `warm` replays right after; arms alternate every iteration; medians.

Gates, before any timing: tests/test_engine_qwen38_input_reuse.InputReuseTests on this device (the admission's plan,
and reuse against the ordinary launch over changed and strided inputs, eager and replayed; a skip fails the run), then
every timed layer's output under reuse against the ordinary launch's, byte for byte, eagerly and replayed. A kernel
component's time on one device; no engine speed is claimed from it (CHARTER D17).

    bash bench/fleet.sh run --gpu qwen38-input-reuse 15 'Qwen3.8 W4 input reuse at its decode shapes (carry S2)' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_input_reuse
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

SHAPES = {"gdn_in": (4120, 2560), "qsa_in": (4224, 2560), "o_proj": (2560, 1536)}   # (n, k) at TP=4
ROWS = (2, 4, 6, 8)
MODES = (0, 2)                                        # ordinary, reuse
WEIGHT_MIB = 48                                       # weights a graph reads per replay, past the L2
ITERATIONS = 30
TRASH_MIB = 64
MEMORY_CAP_GIB = 3
SEED = 20260919


def layers_for(n: int, k: int) -> int:
    """Distinct weights a graph cycles through: W4 bytes plus scales, WEIGHT_MIB of them, at least four."""
    per = n * k // 2 + n * (k // 16)
    return max(4, math.ceil(WEIGHT_MIB * 2 ** 20 / per))


def build(n: int, k: int, count: int, generator):
    from engine.kernels.dense import DenseLinear
    return [DenseLinear((torch.randn(n, k, generator=generator) * .02).to(device="cuda", dtype=torch.bfloat16),
                        prefill=False) for _ in range(count)]


def forward(layers, x):
    return [layer(x) for layer in layers]


def capture(layers, x, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward(layers, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = forward(layers, x)
    return graph, out


def gate(ext, layers, x, graphs) -> bool:
    """Reuse against the ordinary launch: eager, then every arm's graph replayed over changed inputs."""
    for magnitude in (1., .001, 50.):
        x.normal_().mul_(magnitude)
        ext.set_gemm_input(0)
        want = [y.clone() for y in forward(layers, x)]
        ext.set_gemm_input(2)
        eager = forward(layers, x)
        if not all(torch.equal(a, b) for a, b in zip(eager, want)):
            return False
        for graph, out in graphs.values():
            graph.replay()
            torch.cuda.synchronize()
            if not all(torch.equal(a, b) for a, b in zip(out, want)):
                return False
    return True


def timings(graphs: dict, count: int, trash) -> dict:
    """{mode: cold/warm median us a projection}."""
    order = list(graphs)
    samples = {mode: ([], []) for mode in order}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for graph, _ in graphs.values():
        graph.replay()
    for iteration in range(ITERATIONS):
        for mode in (order if iteration % 2 == 0 else order[::-1]):
            trash.zero_()
            torch.cuda.synchronize()
            for bucket in samples[mode]:
                start.record(); graphs[mode][0].replay(); end.record(); end.synchronize()
                bucket.append(start.elapsed_time(end) * 1000 / count)
    return {mode: dict(cold_us=round(statistics.median(c), 2), warm_us=round(statistics.median(w), 2),
                       cold_min_us=round(min(c), 2), warm_min_us=round(min(w), 2), samples=len(c))
            for mode, (c, w) in samples.items()}


def run(output=None):
    events = []

    def report(event, **values):
        row = dict(event=event, **values)
        events.append(row)
        print(json.dumps(row), flush=True)
        if output:
            Path(output).write_text("".join(json.dumps(e) + "\n" for e in events))

    import triton
    from engine.kernels.dense import extension
    assert torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_CAP_GIB * 2 ** 30 / props.total_memory))
    ext = extension()
    saved = (ext.gemm_input_mode(), ext.gemm_input_cta_mode(), ext.probe_state())
    if saved[0] != 2:
        raise RuntimeError(f"the served input mode is 2; this process holds {saved[0]}")
    report("device", name=props.name, torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__)
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_qwen38_input_reuse.InputReuseTests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"input reuse tests: {len(result.failures)} failed, {len(result.errors)} errors, "
                           f"{len(result.skipped)} skipped")
    report("input_reuse_tests", passed=True, tests=result.testsRun)
    generator = torch.Generator().manual_seed(SEED)
    trash = torch.empty(TRASH_MIB * 2 ** 20, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    verdicts = {}
    try:
        with torch.inference_mode():
            for name, (n, k) in SHAPES.items():
                count = layers_for(n, k)
                layers = build(n, k, count, generator)
                for rows in ROWS:
                    plan = ext.gemm_input_plan(rows, n, k, False, False)
                    x = torch.randn(rows, k, generator=generator).to(device="cuda", dtype=torch.bfloat16)
                    graphs = {}
                    for mode in MODES:
                        ext.set_gemm_input(mode)
                        graphs[mode] = capture(layers, x, stream)
                    exact = gate(ext, layers, x, graphs)
                    if not exact:
                        raise RuntimeError(f"{name} x {rows} rows: input reuse is not the ordinary launch's bytes")
                    timed = timings(graphs, count, trash)
                    for mode in MODES:
                        report("input_reuse", shape=name, n=n, k=k, rows=rows, mode=mode, layers=count,
                               admitted=bool(plan[0]), ksr=int(plan[1]), exact=exact, **timed[mode])
                    verdicts[f"{name} x {rows}"] = dict(
                        cold_reuse_over_ordinary=round(timed[2]["cold_us"] / timed[0]["cold_us"], 4),
                        warm_reuse_over_ordinary=round(timed[2]["warm_us"] / timed[0]["warm_us"], 4))
                    report("input_reuse_verdict", shape=name, rows=rows, **verdicts[f"{name} x {rows}"])
                    for graph, _ in graphs.values():
                        graph.reset()
                    del graphs
                del layers
                torch.cuda.empty_cache()
    finally:
        ext.set_gemm_input(saved[0])
        ext.set_input_cta(saved[1])
        ext.restore_probe_state(saved[2])
    report("summary", **verdicts)
    return events


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
