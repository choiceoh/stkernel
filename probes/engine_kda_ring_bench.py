"""The KDA recurrent ring launch by rows and state storage: bound by the bytes it writes, or by how it writes them?

45차 kernel efficiency (measurements/c4_scaling_20260913 §11): the decode timeline shows the ring kernel writing
8 MiB of state in 26.4 us at one row (318 GB/s) and 32 MiB in 174.7 us at four rows (192 GB/s, below the DRAM
rate), and the per-row launches before the row fold were no better. This times the served shapes by themselves --
16 heads of 128 x 128 states, seven tokens and seven ring cells a row -- at 1..4 rows with FP32 and FP16 storage.
FP16 halves the bytes and keeps the programs: if the four-row launch halves with it, the launch is bound by bytes
and only fewer written states help; if it does not, it is bound by its programs and a layout or schedule is worth
trying.

Each case is one captured graph over a ring field of LAYERS x SLOTS_PER_LAYER slots; a replay reads the slots of
the next "layer" through the device slot vector, so no timed launch meets the ring it wrote last in cache.
`cold` follows a 64 MiB write elsewhere, `warm` replays the same slots right after. Cases alternate their order
every iteration so drift lands on all of them alike. Timings only: the arithmetic is the served kernel's, pinned
by tests/test_engine_kda_ring.py.

    bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes kda_ring_bench
"""
import statistics

import torch

HEADS, DIM, TOKENS, CELLS = 16, 128, 7, 7          # a rank's KDA heads, key = value width, spec_k + 1, the recurrent ring
LAYERS, SLOTS_PER_LAYER = 34, 4                     # the KDA layers of a step, one slot per decode row
TRASH_MIB = 64


def state_bytes(storage) -> int:
    return HEADS * DIM * DIM * torch.empty((), dtype=storage).element_size()


def launch_bytes(rows: int, storage) -> int:
    """What one launch moves: every token's state written to its cell, and each row's initial state read."""
    return rows * (TOKENS + 1) * state_bytes(storage)


def summary(rows, storage, cold_us, warm_us):
    nbytes = launch_bytes(rows, storage)
    cold, warm = statistics.median(cold_us), statistics.median(warm_us)
    return dict(rows=rows, storage=str(storage).replace("torch.", ""), mib=nbytes / 2**20,
                cold_us=cold, warm_us=warm, cold_min_us=min(cold_us), per_row_cold_us=cold / rows,
                cold_gbps=nbytes / cold / 1e3, warm_gbps=nbytes / warm / 1e3, samples=len(cold_us))


def _inputs(rows, generator):
    n = rows * TOKENS
    def strided(width):   # the conv output's split: a column slice of a wider row, as net._kda hands it over
        return torch.randn(1, n, HEADS, width * 3, device="cuda", generator=generator).to(torch.bfloat16)[..., :width]
    q, k, v = strided(DIM), strided(DIM), strided(DIM)
    g = torch.randn(1, n, HEADS, DIM, device="cuda", generator=generator).to(torch.bfloat16)
    beta = torch.randn(1, n, HEADS * 3, device="cuda", generator=generator).to(torch.bfloat16)[..., :HEADS]
    a_log = torch.randn(HEADS, device="cuda", generator=generator) * .2
    bias = torch.randn(HEADS * DIM, device="cuda", generator=generator) * .1
    return q, k, v, g, beta, a_log, bias


@torch.inference_mode()
def bench(report, rows_list=(1, 2, 3, 4), storages=(torch.float32, torch.float16), iterations=8):
    from engine.kernels.kda.ring import recurrent_kda_ring_rows
    generator = torch.Generator(device="cuda").manual_seed(4545)
    trash = torch.empty(TRASH_MIB * 2**20 // 4, device="cuda")
    cases = {}
    for storage in storages:
        field = (torch.randn(LAYERS * SLOTS_PER_LAYER, CELLS, HEADS, DIM, DIM, device="cuda", generator=generator) * .1).to(storage)
        for rows in rows_list:
            args = _inputs(rows, generator)
            slots = torch.arange(rows, device="cuda", dtype=torch.int64)
            contexts = torch.full((rows,), 4096, device="cuda", dtype=torch.int64)
            run = lambda args=args, field=field, slots=slots, contexts=contexts: recurrent_kda_ring_rows(
                *args, field, slots, contexts, -5.)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            cases[storage, rows] = dict(graph=graph, slots=slots, field=field, cold=[], warm=[])
    order = list(cases)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for iteration in range(iterations):
        for key in (order if iteration % 2 == 0 else order[::-1]):
            case = cases[key]
            rows = key[1]
            for layer in range(LAYERS):
                case["slots"].copy_(torch.arange(rows, dtype=torch.int64) + layer * SLOTS_PER_LAYER)
                trash.zero_()
                torch.cuda.synchronize()
                start.record(); case["graph"].replay(); end.record(); end.synchronize()
                case["cold"].append(start.elapsed_time(end) * 1000)
                start.record(); case["graph"].replay(); end.record(); end.synchronize()
                case["warm"].append(start.elapsed_time(end) * 1000)
    results = []
    for (storage, rows), case in cases.items():
        row = summary(rows, storage, case["cold"], case["warm"])
        results.append(row)
        report("kda_ring_bench", **row)
    for case in cases.values():
        case["graph"].reset()
    # the question, answered in one line per width: how much of the FP32 launch the halved bytes took away
    by = {(r["storage"], r["rows"]): r for r in results}
    for rows in rows_list:
        if ("float32", rows) in by and ("float16", rows) in by:
            fp32, fp16 = by["float32", rows], by["float16", rows]
            report("kda_ring_bench_bytes", rows=rows, fp16_over_fp32_cold=fp16["cold_us"] / fp32["cold_us"],
                   fp16_over_fp32_warm=fp16["warm_us"] / fp32["warm_us"],
                   per_row_over_one_row_cold=fp32["per_row_cold_us"] / by["float32", rows_list[0]]["per_row_cold_us"])
    return results
