"""Does NVMe tier traffic disturb a running decoder? D16's condition, at the
mechanism level: a decode-shaped GPU loop is timed alone, then while the
tier demotes and promotes ~1 GiB of KV in the background. srv4 is shared with
other tenants, so the comparison is same-node, same-minute, interleaved.
"""
from __future__ import annotations

import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.base.kv_tier import NvmeTier, SECTOR

GIB = 1 << 30


class DecodeLoop:
    """A step-shaped GPU workload, runnable eagerly (Python launches every
    kernel -- what the first probe measured) or as a captured CUDA graph
    (I1: no Python on the hot path -- what a real decode step is)."""

    def __init__(self, seqs: int = 8):
        self.x = torch.randn(seqs, 2560, device="cuda", dtype=torch.bfloat16)
        self.w1 = torch.randn(2560, 2560, device="cuda", dtype=torch.bfloat16)
        self.graph = None

    def _step(self):
        for _ in range(48):
            self.x = torch.tanh(self.x @ self.w1)

    def capture(self):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3): self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    def run(self, iters: int, graph: bool) -> "list[float]":
        times = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            self.graph.replay() if graph else self._step()
            torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
        return times


def main() -> int:
    block_bytes = 51 * SECTOR
    n_blocks = GIB // block_bytes
    storage = torch.randint(0, 256, (n_blocks * block_bytes,), dtype=torch.uint8, device="cuda")
    ids = list(range(n_blocks))
    loop = DecodeLoop(); loop.capture(); loop.run(5, True); loop.run(5, False)
    with tempfile.TemporaryDirectory(dir="/home/choiceoh") as d:
        tier = NvmeTier(d, block_bytes)
        verdicts = []
        for graph in (False, True):
            label = "graph replay (I1)" if graph else "eager launches"
            print(f"### {label}")
            results = {}
            for phase in ("alone", "with tier traffic", "alone again"):
                stop = threading.Event(); cycles = [0]
                def traffic():
                    while not stop.is_set():
                        tier.demote(1, storage, ids, tokens=n_blocks * 16); tier.promote(1, storage, ids); cycles[0] += 1
                th = None
                if "tier" in phase:
                    th = threading.Thread(target=traffic, daemon=True); th.start(); time.sleep(0.3)
                times = loop.run(80, graph)
                if th: stop.set(); th.join()
                results[phase] = times
                srt = sorted(times)
                print(f"  {phase:18s} step p50 {statistics.median(times):6.2f} ms  p95 {srt[int(0.95 * len(srt)) - 1]:6.2f}  "
                      f"max {max(times):6.2f}" + (f"   ({cycles[0]} cycles x {n_blocks * block_bytes / GIB:.2f} GiB each way)" if th else ""))
            ratio = statistics.median(results["with tier traffic"]) / statistics.median(results["alone"])
            verdicts.append((label, ratio))
            print(f"  p50 ratio with/without: {ratio:.3f}  ({'holds' if ratio < 1.05 else 'DISTURBED'})")
    print("\n  " + "; ".join(f"{l}: {r:.3f}" for l, r in verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
