"""A stream launch's grid order on one GB10: a row's streams adjacent against a stream's rows (probe, single-GPU lane
`qwen38_stream_order`).

engine/kernels/gated_residual's stream launches -- `_leave_norm` (the previous sublayer's output added into the streams,
then each stream's norm scale or the normalised stream) and `_norm_streams` -- run one program a (row, stream). Over the
grid (rows, hc), the grid until 2026-09-20, the device runs a stream's rows back to back and the hc programs of one row
are `rows` programs apart: at 4,096 rows about 60 MB of streams pass between them, past the L2, so the output row [H]
they all add (21 MB at 4,096 rows) is read from DRAM once a stream, four times over -- a quarter of the leave's traffic
(252 MB against 189). Over (hc, rows) a row's programs are adjacent: the output row read from DRAM once, and the row's
streams (20 KB, contiguous) walked in one go. The programs are the same either way, so the bytes are -- checked first,
for every form, before anything is timed.

Three forms at a prefill chunk's rows, eight launches a graph, the two orders interleaved and reversed every round; the
minimum is the judge (the lane shares its GPU with production and whatever else runs on the box: `gpu_busy`):

    stream_scales   the prefill site's leave: the output left into the streams, each stream's scale [N, hc] FP32
    leave_norm      a decode step's and the MTP head's: the leave, and the normalised streams written
    norm_streams    the first site's, and the site after the PLE: the scales of the streams as they are (no output)

    python3 probes/engine_kernel_check.py --lanes qwen38_stream_order --output /cache/qwen38-stream-order.json

Not a speed claim (D17): one launch's time on one GPU, for the site's record. The 2026-09-20 record
(q38streamorder-0920a, measurements/qwen38_stream_order_20260920): stream_scales 1,138 -> 792 us at 4,096 rows, 545 -> 396
at 2,048; leave_norm 1,678 -> 1,159 at 4,096; norm_streams unchanged; every form the same bytes.
"""
from __future__ import annotations

import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HC, HIDDEN, EPS = 4, 2560, 1e-6                  # Qwen3.8's streams at TP=4
ROWS = (512, 1024, 2048, 4096)                   # a prefill chunk's rows (mix_block's, from PREFILL_ROWS)
CALLS = 8                                        # launches a graph
ROUNDS = 21
ORDERS = (("rows first", "rows"), ("streams first", None))     # (arm, gated_residual._STREAM_GRID_OVERRIDE)
FORMS = ("stream_scales", "leave_norm", "norm_streams")


@contextlib.contextmanager
def forced(order):
    """gated_residual's stream grid forced to `order` for the block ("rows", or None for the rule), restored after."""
    from engine.kernels import gated_residual as hcr
    before = hcr._STREAM_GRID_OVERRIDE
    hcr._STREAM_GRID_OVERRIDE = order
    try:
        yield
    finally:
        hcr._STREAM_GRID_OVERRIDE = before


def operands(rows: int, device, gen, hidden: int = HIDDEN, hc: int = HC):
    """(streams [rows, hc*hidden], output [rows, hidden], injection [rows, hc], norm weight [hc*hidden]) in BF16."""
    import torch
    width = hc * hidden
    return (torch.randn(rows, width, generator=gen).bfloat16().to(device),
            torch.randn(rows, hidden, generator=gen).bfloat16().to(device),
            (torch.rand(rows, hc, generator=gen) * 2).bfloat16().to(device),
            (torch.randn(width, generator=gen) * 0.1).bfloat16().to(device))


def form(name: str, h, out, inject, w, *, eps: float = EPS, hc: int = HC):
    """One launch of `name` over the operands, h left into in place where the form leaves -> what it wrote besides h."""
    from engine.kernels import gated_residual as hcr
    if name == "stream_scales":
        return hcr.stream_scales(h, out, inject, eps, hc)
    if name == "leave_norm":
        return hcr.leave_norm(h, out, inject, w, eps, hc)[1]
    if name == "norm_streams":
        return hcr.stream_scales(h, None, None, eps, hc)
    raise ValueError(f"form {name!r}: one of {FORMS}")


def alike(name: str, rows: int, device, gen, hidden: int = HIDDEN, hc: int = HC) -> bool:
    """Whether the two orders write the same bytes from the same operands: the streams after the form and its output."""
    import torch
    h, out, inject, w = operands(rows, device, gen, hidden, hc)
    written = []
    for _, order in ORDERS:
        mine = h.clone()
        with forced(order):
            written.append((mine, form(name, mine, out, inject, w, hc=hc)))
    (h_a, out_a), (h_b, out_b) = written
    return bool(torch.equal(h_a, h_b) and torch.equal(out_a, out_b))


def run(output=None) -> dict:
    import torch
    from engine.kernels import gated_residual as hcr
    from probes.engine_qwen38_leave import gpu_busy
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "calls_a_graph": CALLS,
              "rule": hcr._stream_grid(ROWS[-1], HC), "gpu_busy_percent": {"start": gpu_busy()}, "rows": {}}
    for t in ROWS:
        row = {}
        for name in FORMS:
            same = alike(name, t, device, gen)
            h, out, inject, w = operands(t, device, gen)
            graphs = {}
            for arm, order in ORDERS:
                with forced(order):
                    form(name, h, out, inject, w)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):                # the order is read at capture: the graph keeps it
                        for _ in range(CALLS):
                            form(name, h, out, inject, w)
                graphs[arm] = g
            times = {arm: [] for arm in graphs}
            for r in range(ROUNDS):
                for arm, g in (graphs.items() if r % 2 == 0 else list(graphs.items())[::-1]):
                    g.replay()
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    times[arm].append((time.perf_counter() - began) / CALLS * 1e6)
            timed = {arm: {"min": round(min(v), 1), "median": round(statistics.median(v), 1)} for arm, v in times.items()}
            row[name] = {"bytes_alike": same, "us_a_launch": timed}
            print(json.dumps({f"{name} rows {t}": {arm: v["min"] for arm, v in timed.items()}, "bytes_alike": same}),
                  flush=True)
            del graphs, h, out, inject, w
            torch.cuda.empty_cache()
        report["rows"][t] = row
    report["gpu_busy_percent"]["end"] = gpu_busy()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
