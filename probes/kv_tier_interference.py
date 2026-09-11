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


def decode_loop(iters: int, seqs: int = 8) -> "list[float]":
    """A step-shaped GPU workload: a few GEMMs of Qwen3.8's hidden sizes."""
    x = torch.randn(seqs, 2560, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(2560, 10240, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(10240, 2560, device="cuda", dtype=torch.bfloat16)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(48):                       # one "layer" per pass, 48 layers
            x = (x @ w1)[:, :2560] @ w2[:2560] if False else torch.tanh(x @ w1[:, :2560])
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
    return times


def main() -> int:
    block_bytes = 51 * SECTOR
    n_blocks = GIB // block_bytes
    src = torch.randint(0, 256, (n_blocks * block_bytes,), dtype=torch.uint8, device="cuda")
    blocks = [src[i * block_bytes:(i + 1) * block_bytes] for i in range(n_blocks)]
    dst = torch.zeros_like(src)
    into = [dst[i * block_bytes:(i + 1) * block_bytes] for i in range(n_blocks)]
    decode_loop(5)                                  # warm
    with tempfile.TemporaryDirectory(dir="/home/choiceoh") as d:
        tier = NvmeTier(d, block_bytes)
        results = {}
        for phase in ("alone", "with tier traffic", "alone again"):
            stop = threading.Event(); cycles = [0]
            def traffic():
                while not stop.is_set():
                    tier.demote(1, blocks, tokens=n_blocks * 16); tier.promote(1, into); cycles[0] += 1
            th = None
            if "tier" in phase:
                th = threading.Thread(target=traffic, daemon=True); th.start(); time.sleep(0.3)
            times = decode_loop(60)
            if th: stop.set(); th.join()
            results[phase] = (times, cycles[0])
            print(f"  {phase:18s} step p50 {statistics.median(times):6.2f} ms  p95 {sorted(times)[int(0.95 * len(times)) - 1]:6.2f}  "
                  f"max {max(times):6.2f}" + (f"   ({cycles[0]} demote+promote cycles of {n_blocks * block_bytes / GIB:.2f} GiB, "
                  f"{tier.bytes_written / GIB:.1f} GiB written, {tier.bytes_read / GIB:.1f} read)" if th else ""))
    a, b = results["alone"][0], results["with tier traffic"][0]
    ratio = statistics.median(b) / statistics.median(a)
    print(f"\n  p50 ratio with/without tier traffic: {ratio:.3f}  "
          f"({'holds' if ratio < 1.05 else 'DISTURBED'}: D10 wants the running decoder unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
