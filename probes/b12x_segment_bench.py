#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the DRAM subsystem care about the ROW SEGMENT WIDTH the b12x MoE
kernel's TMA boxes read (64 B out of every 2 KB row), independent of the
kernel? A torch reduction over a strided [rows, seg] view of a 1 GB buffer,
DRAM-cold (8 rotating 128 MB slices), seg = 32 .. 2048 B; plus the same
read as a copy. torch's reduction/copy kernels put every SM and thousands
of loads in flight, so a drop at narrow segments is the memory system, not
occupancy or pipelining.

The static v2 probe (2026-09-05) found the v2 kernel -- gate+up fused, FC2 on
its own 4 stages, no item barriers, 32-row A box -- within +-1% of the stock
kernel at every U: pipelining is not the lever. This decides whether the
access granularity is.

    bash probes/run_mk_probe.sh probes/b12x_segment_bench.py
"""
from __future__ import annotations

import statistics

import torch

DEV = "cuda"
ROW_BYTES = 2048          # a w13 row (K = 4096 fp4)
SLICE_BYTES = 128 << 20   # one DRAM-cold rotation slice
ROT = 8


def _time(fn, reps=8):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1e3)
    return statistics.median(out)


def main() -> int:
    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()} rot={ROT} x {SLICE_BYTES >> 20} MB")
    buf = torch.empty(ROT * SLICE_BYTES, dtype=torch.uint8, device=DEV)
    buf.random_(0, 255)
    rows = SLICE_BYTES // ROW_BYTES
    print(f"{'seg B':>6}{'rows':>9}{'read MB':>9}{'sum us':>9}{'GB/s':>7}{'copy us':>9}{'GB/s':>7}")
    state = {"r": 0}

    def view(seg, r):
        base = buf[r * SLICE_BYTES:(r + 1) * SLICE_BYTES]
        return base.view(torch.int32).view(rows, ROW_BYTES // 4)[:, : seg // 4]

    for seg in (32, 64, 128, 256, 512, 1024, 2048):
        n_read = rows * seg

        def do_sum():
            state["r"] = (state["r"] + 1) % ROT
            view(seg, state["r"]).sum()

        def do_copy():
            state["r"] = (state["r"] + 1) % ROT
            view(seg, state["r"]).clone()

        us_sum = _time(do_sum)
        us_copy = _time(do_copy)
        print(f"{seg:>6}{rows:>9}{n_read / 1e6:>9.1f}{us_sum:>9.1f}{n_read / us_sum / 1e3:>7.0f}"
              f"{us_copy:>9.1f}{n_read / us_copy / 1e3:>7.0f}")
    # the full-row read at row-major order = the linear reference
    full = buf[:SLICE_BYTES].view(torch.int32)

    def do_full():
        state["r"] = (state["r"] + 1) % ROT
        buf[state["r"] * SLICE_BYTES:(state["r"] + 1) * SLICE_BYTES].view(torch.int32).sum()

    us = _time(do_full)
    print(f"{'full':>6}{'':>9}{SLICE_BYTES / 1e6:>9.1f}{us:>9.1f}{SLICE_BYTES / us / 1e3:>7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
