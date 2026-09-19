"""The up fold's tile on one GB10 (probe, single-GPU lane `qwen38_up_tiles`).

A prefill site's up fold (engine/kernels/gated_residual.up_mean_block, `_up_mean_rows`) is the site's largest launch
after #1314: 876 us at 4,096 rows beside the leave's 792 and the down fold's 776 (measurements/qwen38_leave_down_20260920).
Its table tile UP_BLOCK_TILE, (64, 64, 32, 4, 4), is the only up tile a GB10 has timed (q38sitecmp-0919a swept the down
fold's tiles and one up tile). A program is a block of rows by a block of channels in each stream: the up weight
(6.5 MB) is read from L2 once a row block -- 64 times at 4,096 rows in 64-row blocks -- and the streams once from DRAM,
so the launch sits at about 120 GB/s of its streams against a 273 GB/s memory: neither the MMA's nor the memory's.
Wider row blocks read the weight fewer times; wider channel blocks read the gates fewer times.

Every tile computes each element in the same order (the dot over the 320-deep K in BLOCK_K steps, the mean over the
streams in stream order), so the outputs are one launch's bytes whatever the tile -- checked first, every tile against
the table's. Then `up_mean_block` timed at every tile as `site` runs it (the streams normalised as read, from
`stream_scales`' scales), eight sites a graph, the arms rotated every round; the minimum is the judge (the lane shares
its GPU). A tile the card cannot hold (shared memory, registers) is recorded, not run.

    python3 probes/engine_kernel_check.py --lanes qwen38_up_tiles --output /cache/qwen38-up-tiles.json

Not a speed claim (D17): one launch's time on one GPU, for the site's record.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HC, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6       # Qwen3.8's site at TP=4
ROWS = (512, 1024, 2048, 4096)
CALLS = 8                                        # sites a graph
ROUNDS = 21
TILES = ((64, 64, 32, 4, 4),                     # the table's (UP_BLOCK_TILE)
         (128, 64, 32, 4, 3), (128, 64, 32, 8, 3), (128, 64, 64, 8, 2), (128, 128, 32, 8, 2), (128, 32, 32, 4, 3),
         (256, 64, 32, 8, 2), (256, 32, 32, 8, 3), (64, 128, 32, 4, 3), (64, 64, 64, 4, 3), (32, 64, 32, 4, 3))
                                                 # (BLOCK_M, BLOCK_D, BLOCK_K, warps, stages) of up_mean_block


def operands(rows: int, device, gen, hidden: int = HIDDEN, rank: int = RANK, hc: int = HC):
    """(streams h, norm weight, gates [rows, rank], up [hc*hidden, rank]) in BF16, and the streams' scales FP32."""
    import torch
    from engine.kernels import gated_residual as hcr
    width = hc * hidden
    h = torch.randn(rows, width, generator=gen).bfloat16().to(device)
    w = (torch.randn(width, generator=gen) * 0.1).bfloat16().to(device)
    gates = (torch.randn(rows, rank, generator=gen) * 0.5).bfloat16().to(device)
    up = (torch.randn(width, rank, generator=gen) * 0.02).bfloat16().to(device)
    scale = hcr.stream_scales(h, None, None, EPS, hc)
    return h, w, gates, up, scale


def fold(h, w, gates, up, scale, tile, hc: int = HC):
    """`up_mean_block` at `tile` as `site` runs it (normalised as read) -> mixed [rows, hidden]."""
    import torch
    from engine.kernels import gated_residual as hcr
    mixed = torch.empty(h.shape[0], h.shape[1] // hc, dtype=h.dtype, device=h.device)
    hcr.up_mean_block(gates, up, h, mixed, hc, tile=tile, norm=(scale, w))
    return mixed


def run(output=None) -> dict:
    import torch
    from engine.kernels import gated_residual as hcr
    from probes.engine_qwen38_leave import gpu_busy
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS, "tiles": TILES,
              "table": hcr.UP_BLOCK_TILE, "gpu_busy_percent": {"start": gpu_busy()}, "rows": {}}
    for t in ROWS:
        h, w, gates, up, scale = operands(t, device, gen)
        want = fold(h, w, gates, up, scale, hcr.UP_BLOCK_TILE)
        checks, fits = {}, []
        for tile in TILES:
            try:
                got = fold(h, w, gates, up, scale, tile)
                torch.cuda.synchronize()
                checks[str(tile)] = record = {"bytes_alike": bool(torch.equal(got, want)),
                                              "max_rel_err": float((got.float() - want.float()).abs().max()
                                                                   / want.float().abs().max())}
                fits.append(tile)
            except Exception as e:                       # a tile the card cannot hold is recorded, not run
                checks[str(tile)] = record = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
                torch.cuda.synchronize()
            print(json.dumps({f"check rows {t} tile {tile}": record}), flush=True)
        graphs = {}
        for tile in fits:
            fold(h, w, gates, up, scale, tile)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(CALLS):
                    fold(h, w, gates, up, scale, tile)
            graphs[str(tile)] = g
        names = list(graphs)
        times = {name: [] for name in names}
        for r in range(ROUNDS):
            for name in names[r % len(names):] + names[:r % len(names)]:
                g = graphs[name]
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        timed = {name: {"min": round(min(v), 1), "median": round(statistics.median(v), 1)} for name, v in times.items()}
        report["rows"][t] = {"checks": checks, "us_a_launch": timed}
        print(json.dumps({f"rows {t}": {name: v["min"] for name, v in timed.items()}}), flush=True)
        del graphs, h, w, gates, up, scale, want
        torch.cuda.empty_cache()
    report["gpu_busy_percent"]["end"] = gpu_busy()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
