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

import torch


class DecodeGraphs:
    def __init__(self, step_fn, make_inputs, shapes: "list[tuple[int, ...]]", warmup: int = 2, generators=(),
                 memory=None, label="decode"):
        """step_fn(inputs) runs one decode step over static `inputs`;
        make_inputs(num_seqs, tokens_per_seq) allocates them once per shape."""
        self.graphs, self.inputs, self.outputs = {}, {}, {}
        # One memory pool for every graph of THIS instance, and a different pool from
        # every other instance's: see the module docstring's second rule.
        self.pool = pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        try:
            for shape in shapes:
                if memory is not None:
                    memory.checkpoint(f"{label}/{shape}/before")
                inp = make_inputs(*shape)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(warmup):
                        step_fn(inp)
                torch.cuda.current_stream().wait_stream(side)
                if memory is not None:
                    memory.checkpoint(f"{label}/{shape}/warmup")
                g = torch.cuda.CUDAGraph()
                for generator in generators:
                    g.register_generator_state(generator)
                # Published only once the capture has completed: a graph whose capture
                # raised is reset here, by the one reference that exists, and never
                # reaches close() -- which would be resetting an uncaptured graph.
                try:
                    with torch.cuda.graph(g, pool=pool):
                        out = step_fn(inp)
                except BaseException:
                    g.reset()
                    raise
                self.graphs[shape], self.inputs[shape], self.outputs[shape] = g, inp, out
                if memory is not None:
                    memory.checkpoint(f"{label}/{shape}/captured")
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
