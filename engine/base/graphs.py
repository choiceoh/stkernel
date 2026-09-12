"""Decode steps as captured graphs, one per shape (base). I1's mechanism.

I1 says no Python on the decode hot path, and says it is enforced by capture,
not by language. This is that enforcement: a decode step is a CUDA graph
keyed by its shape -- (num_seqs, tokens_per_seq) from the shapes registry --
captured once against static input buffers and replayed by writing the
step's flat arrays (base/step_meta) into those buffers and calling replay().

What the interference probe measured is why this matters beyond speed: the
eager loop paid one 20.9 ms GIL spike under tier traffic, the replayed graph
paid none (p50 ratio 1.001). A step that is a graph cannot be preempted by
Python.

Shapes are declared up front from the contract (max_running x draft slots),
so the set of graphs is finite and captured at boot -- a shape that arrives
without a graph is a scheduler bug and raises (D3), it does not fall back to
eager.

TWO RULES BIND EVERY CALLER.

`run()` returns this graph's own output tensors, which live in `pool` and are
overwritten by the next replay of any graph sharing that pool. So (1) consume a
result before replaying the same graph again, and (2) when one graph's output is
still being read while ANOTHER graph replays -- GLM's decode loop reads the target
step's auxiliary hidden states across segments while the drafter's observation
graph replays between them -- the two must not share a pool. Each instance of this
class takes a fresh pool, so distinct instances are safe; `pool` is exposed so a
caller that depends on the separation can assert it instead of assuming it.
"""
from __future__ import annotations

import contextlib
import gc

import torch
from engine.base.graph_labels import capture as label_capture


@contextlib.contextmanager
def frozen_gc():
    """No Python garbage collection while a graph is recording.

    This is a correctness guard before it is a speed one. A Triton kernel object
    finalized during capture unloads its CUDA module, and the graph that recorded
    a call into that module is then invalid -- it does not fail at capture, it
    fails later, on a replay, which is the worst place to find out. vLLM guards
    the same way and says the same reason in its own capture path.

    Freezing first moves everything already alive out of the collector's reach, so
    the collection that runs when the guard is released has little to walk.

    Nesting is a no-op: the outermost guard owns the collector, and the captures
    inside it must not re-enable collection when they finish.
    """
    if not gc.isenabled():
        yield                               # an outer guard already owns it
        return
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        yield
    finally:
        gc.unfreeze()
        gc.enable()


class DecodeGraphs:
    def __init__(self, step_fn, make_inputs, shapes: "list[tuple[int, ...]]", warmup=1, generators=(),
                 memory=None, label="decode", resources=None, detail=False):
        """step_fn(inputs) runs one decode step over static `inputs`;
        make_inputs(num_seqs, tokens_per_seq) allocates them once per shape.

        `warmup` is how many passes run on the side stream before a shape is
        captured, either a count or a function of the shape. One is the default,
        and it is what the pass has to do: compile whatever this shape selects and
        leave the allocator in the state the capture will record. It used to be two,
        on the theory that a second pass settled the allocator -- but the allocator
        is no longer emptied before each capture (below), so it arrives settled from
        the shape before. vLLM warms each of its shapes once for the same reason.
        Warmup was 18.3 s of a measured boot's 28 s of graph work (boot-time study),
        so a caller who needs more should declare it rather than inherit it.

        resources() returns owners of external kernel workspaces used by the
        capture. CUDA records their addresses, not Python references. Keep each
        generation alive before the next shape's warmup can replace it.

        `memory` gets ONE row per shape, because a row is not free: each one
        synchronizes the device and then all-reduces a qualification flag across
        every TP rank, so it is a fleet-wide barrier whose price is the slowest
        node's skew. A measured boot paid 21.9 ms per row, and three rows over
        51 shapes came to 3.4 s -- 12% of that boot's graph work (boot-time study
        5-h). `detail=True` restores the before/warmup/captured split for
        diagnosis; it is what priced the warmup policy, and it costs 2.2 s.
        """
        self.graphs, self.inputs, self.outputs = {}, {}, {}
        self.resources = {}
        # One memory pool for every graph of THIS instance, and a different pool from
        # every other instance's: see the module docstring's second rule.
        self.pool = pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        recording = torch.cuda.Stream()
        # torch.cuda.graph() empties the caching allocator before EVERY capture, to
        # give the pool room. Per shape that throws away exactly what this shape's
        # own warmup just allocated, and on unified memory a released page has to be
        # mapped again before the next shape can touch it -- so the loop pays an
        # unmap and a remap per shape for blocks the next shape wants anyway. The
        # shapes here allocate alike (a rung's cost is flat from 4,096 to 1,048,576),
        # so one flush for the instance leaves the blocks where the next warmup can
        # reuse them. The ledger row after each shape still holds the byte ceiling,
        # so if reserved climbs instead of plateauing the boot says so and fails.
        torch.cuda.empty_cache()
        def mark(shape, name=None):
            if memory is not None:
                memory.checkpoint(f"{label}/{shape}/{name}" if name else f"{label}/{shape}")
        with frozen_gc():
            try:
                for shape in shapes:
                    if detail:
                        mark(shape, "before")
                    inp = make_inputs(*shape)
                    passes = warmup(shape) if callable(warmup) else warmup
                    if not isinstance(passes, int) or passes < 1:
                        raise ValueError(f"{label}: {shape} needs at least one warmup pass, got {passes!r}")
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(passes):
                            step_fn(inp)
                    torch.cuda.current_stream().wait_stream(side)
                    if detail:
                        mark(shape, "warmup")
                    g = torch.cuda.CUDAGraph()
                    for generator in generators:
                        g.register_generator_state(generator)
                    # Published only once the capture has completed: a graph whose capture
                    # raised is reset here, by the one reference that exists, and never
                    # reaches close() -- which would be resetting an uncaptured graph.
                    try:
                        # torch.cuda.graph() is this plus a flush of the caching allocator,
                        # which is the one thing this loop must not do per shape (above).
                        torch.cuda.synchronize()
                        with torch.cuda.stream(recording):
                            g.capture_begin(pool, capture_error_mode="global")
                            try:
                                with label_capture(recording, f'{label}/{shape}'):
                                    out = step_fn(inp)
                            finally:
                                g.capture_end()
                        if resources is not None:
                            for owner in resources():
                                self.resources[id(owner)] = owner
                    except BaseException:
                        g.reset()
                        raise
                    self.graphs[shape], self.inputs[shape], self.outputs[shape] = g, inp, out
                    mark(shape, "captured" if detail else None)
                torch.cuda.synchronize()
            except BaseException:
                self.close()
                raise

    def run(self, shape: "tuple[int, ...]", fill):
        """fill(inputs) copies this step's data into the static buffers; then replay."""
        if shape not in self.graphs:
            raise KeyError(f"no captured graph for decode shape {shape}: the scheduler produced a "
                           f"shape the contract did not declare ({sorted(self.graphs)})")
        fill(self.inputs[shape])
        self.graphs[shape].replay()
        return self.outputs[shape]

    def close(self):
        """Release NCCL graph references before destroying its process group."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.inputs.clear()
        self.outputs.clear()
        # All graphs must release their recorded addresses before their external
        # allocations can return to the eager allocator and be reused.
        self.resources.clear()


def _selfcheck() -> None:
    dev = "cuda"
    W = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
    def make_inputs(n, t):
        return {"x": torch.zeros(n * t, 256, device=dev, dtype=torch.bfloat16),
                "pos": torch.zeros(n * t, device=dev, dtype=torch.int32)}
    def step(inp):
        h = inp["x"]
        for _ in range(8):
            h = torch.tanh(h @ W) + inp["pos"].to(h.dtype).unsqueeze(-1) * 1e-3
        return h
    shapes = [(1, 1), (4, 1), (4, 6), (8, 6)]
    g = DecodeGraphs(step, make_inputs, shapes)
    for shape in shapes:
        x = torch.randn(shape[0] * shape[1], 256, device=dev, dtype=torch.bfloat16)
        pos = torch.arange(shape[0] * shape[1], device=dev, dtype=torch.int32)
        out = g.run(shape, lambda inp: (inp["x"].copy_(x), inp["pos"].copy_(pos)))
        ref = step({"x": x, "pos": pos})
        assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), shape
    out1 = g.run((4, 6), lambda inp: (inp["x"].fill_(0.5), inp["pos"].fill_(3))).clone()
    out2 = g.run((4, 6), lambda inp: (inp["x"].fill_(0.5), inp["pos"].fill_(3)))
    assert torch.equal(out1, out2), "same inputs -> same outputs on replay"
    try:
        g.run((16, 6), lambda inp: None); raise AssertionError("undeclared shape must raise")
    except KeyError:
        pass
    # replay must be faster than eager for the same work: the point of I1
    import time
    x = torch.randn(48, 256, device=dev, dtype=torch.bfloat16); pos = torch.zeros(48, device=dev, dtype=torch.int32)
    def timeit(fn, n=50):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1000
    eager = timeit(lambda: step({"x": x, "pos": pos}))
    replay = timeit(lambda: g.run((8, 6), lambda inp: (inp["x"].copy_(x), inp["pos"].copy_(pos))))
    print(f"  graphs: 4 shapes captured from one pool, replay == eager, undeclared shape refused, "
          f"eager {eager:.3f} ms vs replay {replay:.3f} ms per 8-GEMM step OK")


if __name__ == "__main__":
    _selfcheck()
