"""A prefill site's leave and down fold as one launch on one GB10 (probe, single-GPU lane `qwen38_leave_down`).

engine/kernels/gated_residual serves a prefill step's site (from PREFILL_ROWS) as three launches: `stream_scales` (the
previous sublayer's output left into the streams, each stream's norm scale), the down fold (`down_gates_block`: the
streams read again and normalised as each tile is read, times down(+inject), the gates) and the up fold. The first two
read the streams (84 MB at 4,096 rows) from DRAM once each, and the down fold transforms its A tile once a column block.
`leave_down_block` is the two as one launch: a program leaves into a block of rows of one stream, sums their squares,
reads them back out of L2, normalises them once and multiplies them into every column block at the same time; the
stream partials are summed by the last program of the row block (the module docstring).

Held first, at every row count: the streams after the fused leave byte for byte `stream_scales`' leave, its scales
within a few FP32 ulps of `stream_scales`' (the sum of squares is tiled), and the site's outputs -- the up fold run on
each -- within the oracle band of the two-launch site's and of the torch form. Then the leave + down part of the site
timed: main's two launches against the fused launch at several tiles, eight sites a graph, the arms rotated every
round; the minimum is the judge (the lane shares its GPU). `up_mean_block` is timed once for the site's sum.

    python3 probes/engine_kernel_check.py --lanes qwen38_leave_down --output /cache/qwen38-leave-down.json

Not a speed claim (D17): one site's launches on one GPU, for the site's record.
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

HC, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6       # Qwen3.8's site at TP=4: 324 down(+inject) columns
ROWS = (512, 1024, 2048, 4096)
CALLS = 8                                        # sites a graph
ROUNDS = 21
TILES = ((64, 128, 64, 16, 2), (64, 128, 64, 8, 2), (64, 64, 64, 16, 2), (64, 64, 64, 8, 2), (64, 128, 32, 16, 3),
         (64, 128, 32, 8, 3), (64, 64, 32, 16, 3), (128, 128, 32, 16, 2), (32, 128, 64, 8, 2), (32, 128, 64, 4, 2))
                                                 # (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) of leave_down_block: a
                                                 # stage in flight holds the A tile and every W tile (64 x 64 x 64: 56 KB),
                                                 # and the GB10 has 99 KB, so 64-deep K pipelines two stages, 32-deep three
BAND = 2.0 ** -6                                 # a few BF16 steps: rounding order, not a wrong formula


def operands(rows: int, device, gen, hidden: int = HIDDEN, rank: int = RANK, hc: int = HC):
    """(h, out, injection, norm weight, down(+inject) packed, up, down, inject weight) at the site's widths, BF16."""
    import torch
    from engine.kernels.gated_residual import pack_down_inject
    width = hc * hidden
    down = (torch.randn(rank, width, generator=gen) * 0.02).bfloat16().to(device)
    inj_w = (torch.randn(hc, width, generator=gen) * 0.02).bfloat16().to(device)
    return (torch.randn(rows, width, generator=gen).bfloat16().to(device),
            torch.randn(rows, hidden, generator=gen).bfloat16().to(device),
            (torch.rand(rows, hc, generator=gen) * 2).bfloat16().to(device),
            (torch.randn(width, generator=gen) * 0.1).bfloat16().to(device),
            pack_down_inject(down, inj_w), (torch.randn(width, rank, generator=gen) * 0.02).bfloat16().to(device),
            down, inj_w)


def rel_err(got, want) -> float:
    return float((got.float() - want.float()).abs().max() / want.float().abs().max())


def check(rows: int, device, gen, tile, *, hidden: int = HIDDEN, rank: int = RANK, hc: int = HC, tiles=None) -> dict:
    """The fused site against the two-launch site and the torch form from the same operands -> the record's checks."""
    import torch
    from engine.kernels import gated_residual as hcr
    from engine.modules.hyper_connection import gated_residual
    h, out, injection, w, di, up, down, inj_w = operands(rows, device, gen, hidden, rank, hc)
    tiles = dict(hcr.block_tiles(rows)) if tiles is None else dict(tiles)
    tiles["leave_down"] = tile
    h_two = h.clone()
    scale_two = hcr.stream_scales(h_two, out, injection, EPS, hc)
    mixed_two, inj_two = hcr.mix_block(h_two, di, up, hc, inject=True, tiles=tiles, norm=(scale_two, w))
    h_one = h.clone()
    gates = torch.empty(rows, rank, dtype=h.dtype, device=h.device)
    ij = torch.empty(rows, hc, dtype=h.dtype, device=h.device)
    scale_one = hcr.leave_down_block(h_one, out, injection, w, EPS, hc, di, gates, ij, inject=True, tile=tile)
    mixed_one, inj_one = hcr.leave_mix_block(h.clone(), out, injection, w, EPS, hc, di, up, inject=True, tiles=tiles)
    ref_mixed, ref_inj = gated_residual(h_two, w, down, up, inj_w, hc, EPS)
    return {"h_bytes_alike": bool(torch.equal(h_one, h_two)),
            "scale_max_rel": float(((scale_one - scale_two).abs() / scale_two.abs()).max()),
            "mixed_vs_two_launches": rel_err(mixed_one, mixed_two), "inject_vs_two_launches": rel_err(inj_one, inj_two),
            "mixed_differ": float((mixed_one != mixed_two).float().mean()),
            "mixed_vs_torch": rel_err(mixed_one, ref_mixed), "inject_vs_torch": rel_err(inj_one, ref_inj),
            "two_launches_vs_torch": rel_err(mixed_two, ref_mixed)}


def compiled(fn) -> dict:
    """The kernels Triton has compiled for `fn` on this process's devices, by cache key (Triton 3.x's device_caches,
    or the older `cache`) -> {key: CompiledKernel}; empty where the layout is unknown."""
    out = {}
    caches = getattr(fn, "device_caches", None) or getattr(fn, "cache", None) or {}
    for entry in caches.values():
        cache = entry[0] if isinstance(entry, tuple) else entry
        for key, kern in (cache.items() if hasattr(cache, "items") else ()):
            out[str(key)] = kern
    return out


def resources(kern) -> dict:
    """What the compiled kernel asks of an SM: registers a thread, spills, shared memory bytes -- None where unknown."""
    meta = getattr(kern, "metadata", None)
    return {"n_regs": getattr(kern, "n_regs", None), "n_spills": getattr(kern, "n_spills", None),
            "shared": getattr(meta, "shared", None)}


def passes(record: dict) -> bool:
    return (record["h_bytes_alike"] and record["scale_max_rel"] < 1e-5
            and max(record["mixed_vs_two_launches"], record["inject_vs_two_launches"], record["mixed_vs_torch"],
                    record["inject_vs_torch"]) < BAND)


def run(output=None) -> dict:
    import torch
    from engine.kernels import gated_residual as hcr
    from probes.engine_qwen38_leave import gpu_busy
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS, "tiles": TILES,
              "table": hcr.LEAVE_DOWN_TILES, "gpu_busy_percent": {"start": gpu_busy()}, "rows": {}}
    for t in ROWS:
        tiles = hcr.block_tiles(t)
        checks, fits = {}, []
        for tile in TILES:
            before = set(compiled(hcr._leave_down_rows))
            try:
                checks[str(tile)] = record = check(t, device, gen, tile)
                fresh = [k for key, k in compiled(hcr._leave_down_rows).items() if key not in before]
                record["resources"] = [resources(k) for k in fresh]   # the tile's kernel, compiled by this check
            except Exception as e:                       # a tile the card cannot hold (shared memory) is recorded, not run
                checks[str(tile)] = record = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
                torch.cuda.synchronize()
            if "error" not in record:
                fits.append(tile)
            print(json.dumps({f"check rows {t} tile {tile}": record,
                              "passes": "error" not in record and passes(record)}), flush=True)
        h, out, injection, w, di, up, _, _ = operands(t, device, gen)
        gates = torch.empty(t, RANK, dtype=h.dtype, device=device)
        ij = torch.empty(t, HC, dtype=h.dtype, device=device)
        mixed = torch.empty(t, HIDDEN, dtype=h.dtype, device=device)
        scale_two = hcr.stream_scales(h, out, injection, EPS, HC)

        def two_launches():
            scale = hcr.stream_scales(h, out, injection, EPS, HC)
            hcr.down_gates_block(h, di, gates, ij, HC, inject=True, tile=tiles["down"], norm=(scale, w))

        def fused(tile):
            return lambda: hcr.leave_down_block(h, out, injection, w, EPS, HC, di, gates, ij, inject=True, tile=tile)

        def up_fold():
            hcr.up_mean_block(gates, up, h, mixed, HC, tile=tiles["up"], norm=(scale_two, w))

        arms = {"two launches (main)": two_launches, **{f"fused {tile}": fused(tile) for tile in fits},
                "up fold (both)": up_fold}
        graphs = {}
        for name, fn in arms.items():
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(CALLS):
                    fn()
            graphs[name] = g
        names = list(graphs)
        times = {name: [] for name in names}
        for r in range(ROUNDS):
            order = names[r % len(names):] + names[:r % len(names)]
            for name in order:
                g = graphs[name]
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        timed = {name: {"min": round(min(v), 1), "median": round(statistics.median(v), 1)} for name, v in times.items()}
        report["rows"][t] = {"checks": checks, "us_a_site": timed, "down_tile": tiles["down"], "up_tile": tiles["up"]}
        print(json.dumps({f"rows {t}": {name: v["min"] for name, v in timed.items()}}), flush=True)
        del graphs, h, out, injection, w, di, up, gates, ij, mixed, scale_two
        torch.cuda.empty_cache()
    report["gpu_busy_percent"]["end"] = gpu_busy()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
