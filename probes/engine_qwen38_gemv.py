"""engine/kernels/common/skinny_gemv against cuBLAS at the decode step's own shapes, on one GB10 (probe, single-GPU lane).

The decode step census (probes/engine_qwen38_step.py, 2026-09-19) put 11.2 ms of a C=1 step's 20.2 in BF16 GEMMs of 2 to
8 rows: the hyper-connection mixers' two a site (cuBLAS picks sm80 WMMA kernels, about 201 GB/s -- 6.2 ms), the router
(cuBLAS gemv, about 105 GB/s -- 1.2 ms). Every one of them is a weight read once for a handful of rows, so its floor is
its bytes over the memory's bandwidth (273 GB/s). This times the kernel's configurations (BLOCK_N, BLOCK_K, SPLIT, warps,
stages) against torch.mm at 1, 4, 8 and 16 rows -- a draft step at C=1, a K=3 verify at C=1, 2 and 4 -- both inside CUDA
graphs, interleaved A/B so production's own steps on the same GPU land on every arm, with the weights rotated over more
copies than the L2 holds (a step reads a mixer's weights once and 1,600 other launches between). The verdict a shape
reads is the configuration fastest summed over the rows, and its speedup at each row count.

    python3 probes/engine_kernel_check.py --lanes qwen38_gemv --output /cache/qwen38-gemv.json
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

ROWS = (1, 4, 8, 16)
COPIES_BYTES = 64 << 20        # weights rotated over at least this many bytes: more than GB10's L2
CALLS = 16                     # calls a graph (one per copy in turn)
ROUNDS = 9

# (outputs, K) of W [outputs, K] -> the configurations tried: (BLOCK_N, BLOCK_K, SPLIT, warps, stages)
SHAPES = {
    # the MTP head's BF16 projections (mtp_precision "bf16", the fleet default since 2026-09-19): attention in and out,
    # shared expert gate_up and down (160 columns: one masked K tile), and the head's two fc
    "mtp in_proj": ((4224, 2560), ((16, 256, 1, 4, 3), (32, 256, 1, 4, 3), (32, 128, 1, 4, 3), (64, 128, 1, 4, 3),
                                   (64, 256, 1, 8, 3), (32, 256, 2, 4, 3), (16, 256, 2, 4, 3), (128, 128, 1, 8, 3))),
    "mtp o_proj": ((2560, 1536), ((16, 256, 1, 4, 3), (32, 256, 1, 4, 3), (32, 128, 1, 4, 3), (64, 128, 1, 4, 3),
                                  (16, 256, 2, 4, 3), (32, 256, 2, 4, 3), (16, 128, 2, 4, 3), (64, 256, 1, 8, 3))),
    "mtp sh_gate_up": ((320, 2560), ((16, 256, 1, 4, 3), (16, 256, 2, 4, 3), (16, 256, 4, 4, 3), (32, 256, 2, 4, 3),
                                     (16, 128, 4, 4, 3), (16, 512, 2, 4, 2))),
    "mtp sh_down": ((2560, 160), ((16, 256, 1, 4, 1), (32, 256, 1, 4, 1), (64, 256, 1, 4, 1), (16, 256, 1, 2, 1),
                                  (32, 256, 1, 2, 1), (128, 256, 1, 8, 1))),
    "mtp fc": ((2560, 2560), ((16, 256, 1, 4, 3), (32, 256, 1, 4, 3), (32, 128, 1, 4, 3), (16, 256, 2, 4, 3),
                              (32, 256, 2, 4, 3), (64, 128, 1, 4, 3), (16, 128, 2, 4, 3), (64, 256, 1, 8, 3))),
}


def run(output=None, *, shapes=None, rows=None) -> dict:
    """`shapes` ({label: ((outputs, K), configurations)}) and `rows` default to this model's; another profile's lane
    passes its own (probes/engine_glm53_decode_rows)."""
    import torch
    from engine.kernels.common import skinny_gemv
    shapes = SHAPES if shapes is None else shapes
    rows = ROWS if rows is None else tuple(rows)
    torch.manual_seed(0)
    skinny_gemv.prepare("cuda")
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS, "rows": list(rows),
              "shapes": {}}
    for label, ((n, k), configs) in shapes.items():
        nbytes = n * k * 2
        copies = max(CALLS, -(-COPIES_BYTES // nbytes))
        weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        shape = {"weight_MB": round(nbytes / 1e6, 2), "rows": {}}
        for m in rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = (x.float() @ weights[0].float().t())
            keep = []                                     # a graph writes its outputs' addresses: they live with it

            def graph_of(fn):
                outs = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(CALLS)]
                keep.append(outs)
                for i in range(CALLS):
                    fn(weights[i % copies], outs[i])
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(CALLS):
                        fn(weights[(i * 7) % copies], outs[i])
                return g

            arms = {"cublas": graph_of(lambda w, o: torch.mm(x, w.t(), out=o))}
            errors = {}
            for config in configs:
                got = skinny_gemv.gemv(x, weights[0], config).float()
                again = skinny_gemv.gemv(x, weights[0], config).float()
                errors[str(config)] = (round(float((got - ref).abs().max() / ref.abs().max()), 6),
                                       bool(torch.equal(got, again)))
                arms[str(config)] = graph_of(lambda w, o, config=config: skinny_gemv.gemv(x, w, config, out=o))
            times = {name: [] for name in arms}
            for _ in range(ROUNDS):
                for name, g in arms.items():              # interleaved: contention lands on every arm alike
                    g.replay()
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    times[name].append((time.perf_counter() - began) / CALLS * 1e6)
            del arms, keep
            row = {}
            for name, samples in times.items():
                us = statistics.median(samples)
                row[name] = {"us": round(us, 2), "GBps": round(nbytes / us / 1e3, 1)}
                if name in errors:
                    row[name].update(rel_err=errors[name][0], repeatable=errors[name][1])
            shape["rows"][m] = row
            best = min((c for c in row if c != "cublas"), key=lambda c: row[c]["us"])
            print(json.dumps({f"{label} rows {m}": {"cublas": row["cublas"], "best": best, **row[best],
                                                     "speedup": round(row["cublas"]["us"] / row[best]["us"], 3)}}),
                  flush=True)
        names = [str(c) for c in configs]
        chosen = min(names, key=lambda c: sum(shape["rows"][m][c]["us"] for m in rows))
        shape["chosen"] = chosen
        shape["speedup"] = {m: round(shape["rows"][m]["cublas"]["us"] / shape["rows"][m][chosen]["us"], 3) for m in rows}
        shape["module_config"] = str(skinny_gemv.CONFIGS.get((n, k)))
        print(json.dumps({label: {"chosen": chosen, "speedup": shape["speedup"],
                                  "module_config": shape["module_config"]}}), flush=True)
        report["shapes"][label] = shape
        del weights
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


HC, HIDDEN, RANK = 4, 2560, 320                     # Qwen3.8's streams, hidden width and mixer rank


def run_site(output=None) -> dict:
    """A site's mixer two ways, 16 sites a graph over rotated weights, interleaved: its four launches on cuBLAS (the
    lane before the skinny GEMV: both products through torch.mm) and engine/kernels/gated_residual.mix as it serves a
    decode step's rows (`mix_rows`, two launches: carry H2). The fold is held byte for byte to the unfolded mixer on its
    own products (tests/test_engine_gated_residual_rows' composition) and to the cuBLAS mixer within two BF16 steps. The
    first run of this lane (q38site-0919a) timed the prototypes the engine's kernels came from."""
    import torch
    from engine.kernels import gated_residual as hcr
    from engine.kernels.common import skinny_gemv
    from tests.test_engine_gated_residual_rows import unfolded
    torch.manual_seed(0)
    skinny_gemv.prepare("cuda")
    width = HC * HIDDEN
    pair = (RANK + HC) * width * 2 + width * RANK * 2
    copies = max(CALLS, -(-COPIES_BYTES // pair))
    downs = [torch.randn(RANK + HC, width, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    ups = [torch.randn(width, RANK, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "sites_a_graph": CALLS,
              "pair_MB": round(pair / 1e6, 2), "up_tile": list(hcr.UP_TILE), "rows": {},
              # the boot's own D3 checks at Qwen3.8's widths: 1 and 5 rows fold, 64 do not (fleet eps)
              "qualify": {"gated_residual": {k: [round(x, 6) for x in v] for k, v in hcr.qualify(
                  torch.device("cuda"), hc=HC, hidden=HIDDEN, rank=RANK, eps=1e-6).items()},
                          "skinny_gemv": skinny_gemv.qualify(torch.device("cuda"))}}
    print(json.dumps({"qualify": report["qualify"]}), flush=True)
    for m in ROWS:
        normed = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
        keep = []                                         # a graph's outputs live with it

        def graph_of(fn):
            for i in range(CALLS):
                keep.append(fn(downs[i % copies], ups[i % copies]))
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(CALLS):
                    keep.append(fn(downs[(i * 7) % copies], ups[(i * 7) % copies]))
            return g

        def cublas(d, u):
            return hcr.mix(normed, d, u, HC, project_down=lambda t: torch.mm(t, d.t()),
                           project_up=lambda t: torch.mm(t, u.t()))

        def served(d, u):
            return hcr.mix(normed, d, u, HC)

        got, ref = served(downs[0], ups[0]), cublas(downs[0], ups[0])
        want = unfolded(normed, downs[0], ups[0], True)
        checks = {"folds": hcr.folds(normed, downs[0], ups[0]),
                  "byte_equal_mixed": bool(torch.equal(got[0], want[0])),
                  "byte_equal_inject": bool(torch.equal(got[1], want[1])),
                  "vs_cublas_mixed": round(hcr.drift(got[0], ref[0])[0], 6),
                  "vs_cublas_inject": round(hcr.drift(got[1], ref[1])[0], 6)}
        arms = {"cublas (four launches)": graph_of(cublas), "served (two launches)": graph_of(served)}
        times = {name: [] for name in arms}
        for _ in range(ROUNDS):
            for name, g in arms.items():
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        del arms, keep
        row = {name: round(statistics.median(v), 2) for name, v in times.items()}
        report["rows"][m] = {"us_a_site": row, "checks": checks}
        print(json.dumps({f"site rows {m}": row, "checks": checks}), flush=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


PREFILL_ROWS = (128, 512, 1024, 4096)
PREFILL_SITES = 8                                  # sites a graph at prefill rows: the activations, not the weights, are the bytes
# gated_residual.mix_block's two launches at other tiles than the table's, one launch at a time: down (BLOCK_M, BLOCK_N,
# BLOCK_K, warps, stages, split), up (BLOCK_M, BLOCK_D, BLOCK_K, warps, stages)
BLOCK_TRIES = {"down": ((128, 128, 32, 8, 3, 1), (128, 128, 32, 8, 3, 4), (128, 64, 64, 4, 3, 1)),
               "up": ((64, 64, 64, 4, 3), (128, 64, 64, 4, 3))}


def run_site_prefill(output=None) -> dict:
    """A site's mixer at a prefill step's rows, `PREFILL_SITES` sites a graph, interleaved: the five launches it served
    before (the stream-norm's output through cuBLAS down, `_gates`, cuBLAS up, `_mix_mean`) against
    gated_residual.mix_block's two (down and the gates in one, up and the mean in one -- the [N, hc*H] up product never
    written), at the table's tiles and at the others in BLOCK_TRIES one launch at a time. Held to the five-launch site
    within two BF16 steps (the products are different GEMMs', so not byte for byte)."""
    import torch
    from engine.kernels import gated_residual as hcr
    torch.manual_seed(0)
    width = HC * HIDDEN
    down = torch.randn(RANK + HC, width, device="cuda", dtype=torch.bfloat16) * 0.02
    up = torch.randn(width, RANK, device="cuda", dtype=torch.bfloat16) * 0.02
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "sites_a_graph": PREFILL_SITES,
              "tiles": {"down": [[r, list(t)] for r, t in hcr.DOWN_TILES], "up": list(hcr.UP_BLOCK_TILE)}, "rows": {}}
    for m in PREFILL_ROWS:
        normed = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
        keep = []

        def graph_of(fn):
            keep.append(fn())
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(PREFILL_SITES):
                    keep.append(fn())
            return g

        def five():
            return hcr.mix(normed, down, up, HC, project_down=lambda t: torch.mm(t, down.t()),
                           project_up=lambda t: torch.mm(t, up.t()))

        def tiled(which, tile):
            tiles = dict(hcr.block_tiles(m), **{which: tile})
            return lambda: hcr.mix_block(normed, down, up, HC, tiles=tiles)

        got, ref = hcr.mix_block(normed, down, up, HC), five()
        checks = {"vs_five_mixed": round(hcr.drift(got[0], ref[0])[0], 6),
                  "vs_five_inject": round(hcr.drift(got[1], ref[1])[0], 6),
                  "mix_routes_here": bool(torch.equal(hcr.mix(normed, down, up, HC)[0], got[0]))}
        arms = {"five launches (served before)": graph_of(five), "mix_block (table)": graph_of(lambda: hcr.mix_block(normed, down, up, HC))}
        for which, tries in BLOCK_TRIES.items():
            for tile in tries:
                arms[f"{which} {tile}"] = graph_of(tiled(which, tile))
        times = {name: [] for name in arms}
        for _ in range(ROUNDS):
            for name, g in arms.items():
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / PREFILL_SITES * 1e6)
        del arms, keep
        row = {name: round(statistics.median(v), 1) for name, v in times.items()}
        report["rows"][m] = {"us_a_site": row, "checks": checks}
        print(json.dumps({f"prefill site rows {m}": row, "checks": checks}), flush=True)
        del normed
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


COMPONENT_ROWS = (512, 1024, 2048, 4096)
# down (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, split): whole-K tiles for many rows (64-wide ones end in the 16-wide
# injection block, 256-wide ones in a 128-wide block), split-K ones for few; up (BLOCK_M, BLOCK_D, BLOCK_K, warps,
# stages). q38sitecmp-0919a measured the down tiles before the column blocks were adjacent programs (5-tuples then)
COMPONENT_TRIES = {"down": ((128, 128, 32, 8, 3, 1), (128, 128, 64, 8, 3, 1), (128, 128, 32, 8, 4, 1),
                            (256, 128, 32, 8, 3, 1), (128, 256, 32, 8, 3, 1), (64, 256, 32, 8, 4, 1),
                            (128, 64, 64, 4, 3, 1), (256, 64, 64, 8, 3, 1), (128, 64, 32, 4, 4, 1),
                            (64, 128, 64, 4, 3, 2), (64, 128, 64, 4, 3, 4), (128, 128, 32, 8, 3, 2),
                            (128, 128, 32, 8, 3, 4), (128, 128, 32, 8, 3, 8), (64, 64, 64, 4, 3, 4),
                            (128, 64, 64, 4, 3, 2), (128, 64, 64, 4, 3, 4)),
                   "up": ((64, 64, 32, 4, 4),)}


def run_site_components(output=None) -> dict:
    """mix_block's two launches one at a time against what each replaces, `PREFILL_SITES` a graph, interleaved: the
    down projection and the gates (cuBLAS down + `_gates` / `_down_gates_rows` at every tile in COMPONENT_TRIES) and
    the up projection and the mean (cuBLAS up + `_mix_mean` / `_up_mean_rows` at every tile). Each tile's output held to
    the first tile's (the same products up to the dot's order) within two BF16 steps."""
    import torch
    import triton
    from engine.kernels import gated_residual as hcr
    torch.manual_seed(0)
    width = HC * HIDDEN
    down = torch.randn(RANK + HC, width, device="cuda", dtype=torch.bfloat16) * 0.02
    up = torch.randn(width, RANK, device="cuda", dtype=torch.bfloat16) * 0.02
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "sites_a_graph": PREFILL_SITES, "rows": {}}
    for m in COMPONENT_ROWS:
        normed = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
        gates = torch.nn.functional.silu(torch.randn(m, RANK, device="cuda", dtype=torch.bfloat16))
        keep = []

        def graph_of(fn):
            keep.append(fn())
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(PREFILL_SITES):
                    keep.append(fn())
            return g

        def cublas_down():
            di = torch.mm(normed, down.t())
            g = torch.empty(m, RANK, device="cuda", dtype=torch.bfloat16)
            inj = torch.empty(m, HC, device="cuda", dtype=torch.bfloat16)
            hcr._gates[(m,)](di, g, inj, di.stride(0), g.stride(0), inj.stride(0), float(HC), R=RANK,
                             BR=triton.next_power_of_2(RANK), HC=HC, BH=triton.next_power_of_2(HC), WITH_INJECT=True,
                             num_warps=4)
            return g, inj

        def block_down(tile):
            def fn():
                g = torch.empty(m, RANK, device="cuda", dtype=torch.bfloat16)
                inj = torch.empty(m, HC, device="cuda", dtype=torch.bfloat16)
                hcr.down_gates_block(normed, down, g, inj, HC, inject=True, tile=tile)
                return g, inj
            return fn

        def cublas_up():
            weights = torch.mm(gates, up.t())
            mixed = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
            tile, warps = hcr._mix_tile(HIDDEN, m)
            hcr._mix_mean[(m, triton.cdiv(HIDDEN, tile))](weights, normed, mixed, weights.stride(0), normed.stride(0),
                                                          mixed.stride(0), float(HC), HID=HIDDEN, BD=tile, HC=HC,
                                                          num_warps=warps)
            return mixed

        def block_up(tile):
            def fn():
                mixed = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
                hcr.up_mean_block(gates, up, normed, mixed, HC, tile=tile)
                return mixed
            return fn

        checks = {}
        ref_g, ref_up = cublas_down()[0], cublas_up()
        failed = {}
        for tile in COMPONENT_TRIES["down"]:
            try:                                          # a tile past the device's shared memory fails to compile
                checks[f"down {tile}"] = round(hcr.drift(block_down(tile)()[0], ref_g)[0], 6)
            except Exception as err:                      # noqa: BLE001 -- recorded, and the tile left out
                failed[f"down {tile}"] = f"{type(err).__name__}: {str(err)[:200]}"
        for tile in COMPONENT_TRIES["up"]:
            checks[f"up {tile}"] = round(hcr.drift(block_up(tile)(), ref_up)[0], 6)
        arms = {"down: cublas + gates": graph_of(cublas_down), "up: cublas + mix_mean": graph_of(cublas_up)}
        for tile in COMPONENT_TRIES["down"]:
            if f"down {tile}" not in failed:
                arms[f"down {tile}"] = graph_of(block_down(tile))
        for tile in COMPONENT_TRIES["up"]:
            arms[f"up {tile}"] = graph_of(block_up(tile))
        times = {name: [] for name in arms}
        for _ in range(ROUNDS):
            for name, g in arms.items():
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / PREFILL_SITES * 1e6)
        del arms, keep
        row = {name: {"median": round(statistics.median(v), 1), "min": round(min(v), 1)} for name, v in times.items()}
        report["rows"][m] = {"us_a_site": row, "drift_vs_cublas": checks, "failed": failed}
        print(json.dumps({f"components rows {m}": {k: v["median"] for k, v in row.items()}}), flush=True)
        del normed, gates
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


# down tiles a whole site tries with the streams normalised inside the fold, besides the table's
WHOLE_TRIES = ((128, 64, 64, 4, 3, 1), (128, 128, 64, 8, 3, 1), (128, 128, 32, 8, 3, 1), (64, 128, 64, 4, 3, 2))


def run_site_whole(output=None) -> dict:
    """A whole site at a prefill step's rows -- the leave into the streams, the stream norm, the mixer -- `PREFILL_SITES`
    a graph, interleaved: as main served it (leave_norm, then five launches on cuBLAS); leave_norm then mix_block (the
    normalised streams written and read back); and gated_residual.site's way (the leave keeps each stream's scale,
    mix_block normalises what it reads, the [N, hc*H] write gone) at the table's down tile and WHOLE_TRIES'. Also the
    leave alone both ways. Each arm held to main's within two BF16 steps, and the scale way byte for byte the written
    way at the same tile (the streams reset before each check)."""
    import torch
    from engine.kernels import gated_residual as hcr
    torch.manual_seed(0)
    width, eps = HC * HIDDEN, 1e-6
    down = torch.randn(RANK + HC, width, device="cuda", dtype=torch.bfloat16) * 0.02
    up = torch.randn(width, RANK, device="cuda", dtype=torch.bfloat16) * 0.02
    w = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "sites_a_graph": PREFILL_SITES,
              "table": {"down": [[r, list(t)] for r, t in hcr.DOWN_TILES], "up": list(hcr.UP_BLOCK_TILE)}, "rows": {}}
    for m in COMPONENT_ROWS:
        h0 = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
        h = h0.clone()
        out = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        inj = torch.rand(m, HC, device="cuda", dtype=torch.bfloat16) * 0.01   # small: the streams stay put over replays
        keep = []

        def graph_of(fn):
            keep.append(fn())
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(PREFILL_SITES):
                    keep.append(fn())
            return g

        def main_served():
            _, normed = hcr.leave_norm(h, out, inj, w, eps, HC)
            return hcr.mix(normed, down, up, HC, project_down=lambda t: torch.mm(t, down.t()),
                           project_up=lambda t: torch.mm(t, up.t()))

        def written(tile):
            tiles = dict(hcr.block_tiles(m), down=tile)

            def fn():
                _, normed = hcr.leave_norm(h, out, inj, w, eps, HC)
                return hcr.mix_block(normed, down, up, HC, tiles=tiles)
            return fn

        def scaled(tile):
            tiles = dict(hcr.block_tiles(m), down=tile)

            def fn():
                scale = hcr.stream_scales(h, out, inj, eps, HC)
                return hcr.mix_block(h, down, up, HC, tiles=tiles, norm=(scale, w))
            return fn

        tries = ([hcr.block_tiles(m)["down"]] if hcr.block_tiles(m)["down"] is not None else []) + [
            t for t in WHOLE_TRIES if t != hcr.block_tiles(m)["down"]]
        h.copy_(h0)
        ref = main_served()
        checks = {}
        for tile in tries:
            h.copy_(h0)
            a = written(tile)()
            h.copy_(h0)
            b = scaled(tile)()
            checks[f"{tile}"] = {"scaled_is_written_bytes": bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])),
                                 "vs_main_mixed": round(hcr.drift(b[0], ref[0])[0], 6),
                                 "vs_main_inject": round(hcr.drift(b[1], ref[1])[0], 6)}
        h.copy_(h0)
        arms = {"main: leave_norm + five launches": graph_of(main_served),
                "leave_norm alone": graph_of(lambda: hcr.leave_norm(h, out, inj, w, eps, HC)[1]),
                "stream_scales alone": graph_of(lambda: hcr.stream_scales(h, out, inj, eps, HC))}
        if hcr.block_tiles(m)["down"] is not None:
            arms["site (table)"] = graph_of(lambda: hcr.site(h, out, inj, w, eps, HC, down, up))
        for tile in tries:
            arms[f"written {tile}"] = graph_of(written(tile))
            arms[f"scaled {tile}"] = graph_of(scaled(tile))
        times = {name: [] for name in arms}
        for _ in range(ROUNDS):
            for name, g in arms.items():
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / PREFILL_SITES * 1e6)
        del arms, keep
        row = {name: {"median": round(statistics.median(v), 1), "min": round(min(v), 1)} for name, v in times.items()}
        report["rows"][m] = {"us_a_site": row, "checks": checks}
        print(json.dumps({f"whole site rows {m}": {k: v["median"] for k, v in row.items()}, "checks": checks}),
              flush=True)
        del h, h0, out, inj
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


# down tiles measured both ways -- over the normalised streams, and over the streams normalised as each tile is read
# (site's way) -- and the rounds: beside a serving production, only the minimum of many interleaved rounds is clean
NORM_TRIES = ((64, 128, 64, 4, 3, 2), (128, 64, 64, 4, 3, 1), (128, 128, 64, 8, 3, 1), (128, 128, 32, 8, 3, 1),
              (256, 64, 64, 8, 3, 1), (128, 64, 32, 4, 4, 1), (64, 128, 32, 4, 4, 2))
NORM_ROUNDS = 21


def run_site_norm_in(output=None) -> dict:
    """mix_block's two launches with and without `norm` (the streams normalised inside the launch, gated_residual.site's
    way, against the normalised streams read as written), `PREFILL_SITES` a graph, NORM_ROUNDS interleaved rounds: the
    down fold at every tile of NORM_TRIES and the up fold at the table's, beside cuBLAS down + `_gates`. The two ways'
    outputs held byte for byte at every tile."""
    import torch
    from engine.kernels import gated_residual as hcr
    torch.manual_seed(0)
    width, eps = HC * HIDDEN, 1e-6
    down = torch.randn(RANK + HC, width, device="cuda", dtype=torch.bfloat16) * 0.02
    up = torch.randn(width, RANK, device="cuda", dtype=torch.bfloat16) * 0.02
    w = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1
    report = {"device": torch.cuda.get_device_name(), "rounds": NORM_ROUNDS, "sites_a_graph": PREFILL_SITES, "rows": {}}
    for m in COMPONENT_ROWS:
        h = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
        normed = hcr.norm_streams(h, w, eps, HC)
        scale = hcr.stream_scales(h, None, None, eps, HC)
        gates = torch.nn.functional.silu(torch.randn(m, RANK, device="cuda", dtype=torch.bfloat16))
        keep = []

        def graph_of(fn):
            keep.append(fn())
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(PREFILL_SITES):
                    keep.append(fn())
            return g

        def cublas_down():
            return hcr.mix_block(normed, down, up, HC, tiles={"down": None, "up": hcr.UP_BLOCK_TILE})

        def fold_down(tile, norm):
            def fn():
                g = torch.empty(m, RANK, device="cuda", dtype=torch.bfloat16)
                inj = torch.empty(m, HC, device="cuda", dtype=torch.bfloat16)
                hcr.down_gates_block(h if norm else normed, down, g, inj, HC, inject=True, tile=tile,
                                     norm=(scale, w) if norm else None)
                return g, inj
            return fn

        def fold_up(norm):
            def fn():
                mixed = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
                hcr.up_mean_block(gates, up, h if norm else normed, mixed, HC, tile=hcr.UP_BLOCK_TILE,
                                  norm=(scale, w) if norm else None)
                return mixed
            return fn

        checks = {f"down {t}": all(torch.equal(a, b) for a, b in zip(fold_down(t, False)(), fold_down(t, True)()))
                  for t in NORM_TRIES}
        checks["up"] = bool(torch.equal(fold_up(False)(), fold_up(True)()))
        arms = {"mixer: cublas down + gates, up fold": graph_of(cublas_down)}
        for tile in NORM_TRIES:
            arms[f"down {tile}"] = graph_of(fold_down(tile, False))
            arms[f"down norm {tile}"] = graph_of(fold_down(tile, True))
        arms["up"] = graph_of(fold_up(False))
        arms["up norm"] = graph_of(fold_up(True))
        times = {name: [] for name in arms}
        for r in range(NORM_ROUNDS):
            for name, g in (arms.items() if r % 2 == 0 else list(arms.items())[::-1]):
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / PREFILL_SITES * 1e6)
        del arms, keep
        row = {name: {"median": round(statistics.median(v), 1), "min": round(min(v), 1)} for name, v in times.items()}
        report["rows"][m] = {"us_a_site": row, "bytes_alike": checks}
        print(json.dumps({f"norm_in rows {m}": {k: v["min"] for k, v in row.items()}, "bytes_alike": checks}),
              flush=True)
        del h, normed, scale, gates
        torch.cuda.empty_cache()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
