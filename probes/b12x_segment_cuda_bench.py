#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DRAM bandwidth vs ROW SEGMENT WIDTH, with plain vectorized loads and
thousands of loads in flight (no cp.async, no TMA, no torch kernels): the
instrument the two earlier benches were not.

Why: the v2 static MoE kernel's stamps (33차 §7) show FC1 streaming w13 at
4.5 GB/s per CTA (64 B out of every 2,048 B row per TMA box) while FC2
streams w2 at 5.05 GB/s per CTA (the 4 slice-CTAs read the 4 adjacent 64 B
chunks of each 256 B row, so DRAM sees whole rows). If DRAM itself pays for
narrow segments, widening the FC1 box to 128 B rows (tile_k 256) is the
lever; if not, the loss is in the TMA/L2 path and the fix is elsewhere.

Patterns (each 16 B chunk read once by one thread, XOR-reduced):
  w13/seg=S   rows 2,048 B apart, S contiguous bytes per row (S = 32..2048)
  w2/single   rows 256 B apart, 64 B per row (one slice alone)
  w2/shared   rows 256 B apart, the 4 adjacent 64 B chunks read by 4
              consecutive blocks (co-scheduled) -- the served FC2 pattern
  linear      the whole buffer, contiguous
1 GB buffer, 8 DRAM-cold rotations (column offsets 256 B apart), 5 timed
launches each, median.

    bash probes/run_mk_probe.sh probes/b12x_segment_cuda_bench.py
"""
from __future__ import annotations

import os
import statistics

import torch

DEV = "cuda"
BUF = 1 << 30
ROT = 8

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

// pattern: 0 = strided rows (stride, seg), 1 = w2 shared (stride 256, 4 blocks share a row)
template <int SEG>
__global__ void __launch_bounds__(256) seg_read(const uint8_t* __restrict__ base,
                                                long long rows, int stride, int shared4,
                                                unsigned* sink) {
  constexpr int CPR = SEG / 16;                 // 16 B chunks per row segment
  const long long total = rows * CPR;
  const long long nthreads = (long long)gridDim.x * blockDim.x;
  long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int slice = shared4 ? (blockIdx.x & 3) : 0;
  unsigned acc = 0;
  // 4 independent 16 B loads in flight per thread per iteration
  for (; g < total; g += 4 * nthreads) {
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      long long gg = g + (long long)u * nthreads;
      if (gg < total) {
        long long row = gg / CPR;
        int c = (int)(gg - row * CPR);
        const uint4* p = reinterpret_cast<const uint4*>(base + row * (long long)stride
                                                       + (long long)slice * SEG + c * 16);
        v[u] = __ldg(p);
      } else {
        v[u] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

template <int SEG>
static float run_one(const uint8_t* base, long long rows, int stride, int shared4,
                     unsigned* sink, int grid) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, 0);
  seg_read<SEG><<<grid, 256, 0, 0>>>(base, rows, stride, shared4, sink);
  cudaEventRecord(e1, 0);
  cudaEventSynchronize(e1);
  float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms * 1000.f;
}

float bench(torch::Tensor buf, long long offset, int seg, long long rows, int stride,
            int shared4, int grid, torch::Tensor sink) {
  const uint8_t* base = buf.data_ptr<uint8_t>() + offset;
  unsigned* s = reinterpret_cast<unsigned*>(sink.data_ptr<int32_t>());
  switch (seg) {
    case 32: return run_one<32>(base, rows, stride, shared4, s, grid);
    case 64: return run_one<64>(base, rows, stride, shared4, s, grid);
    case 128: return run_one<128>(base, rows, stride, shared4, s, grid);
    case 256: return run_one<256>(base, rows, stride, shared4, s, grid);
    case 512: return run_one<512>(base, rows, stride, shared4, s, grid);
    case 2048: return run_one<2048>(base, rows, stride, shared4, s, grid);
  }
  return -1.f;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
float bench(torch::Tensor buf, long long offset, int seg, long long rows, int stride,
            int shared4, int grid, torch::Tensor sink);
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    os.makedirs("/tmp/b12x_segcuda_build", exist_ok=True)
    return load_inline(
        name="b12x_segcuda",
        cpp_sources=[CPP_SRC],
        cuda_sources=["#include <torch/extension.h>\n" + CUDA_SRC],
        functions=["bench"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_121a,code=sm_121a"],
        build_directory="/tmp/b12x_segcuda_build",
        verbose=False,
    )


def main() -> int:
    torch.cuda.init()
    ext = _build()
    buf = torch.empty(BUF + 4096, dtype=torch.uint8, device=DEV)
    buf.random_(0, 255)
    sink = torch.zeros(4, dtype=torch.int32, device=DEV)
    grid = 48 * 16
    print(f"device {torch.cuda.get_device_name()} buffer {BUF >> 20} MB rot={ROT} grid={grid}x256")
    print(f"{'pattern':<14}{'seg':>6}{'stride':>8}{'read MB':>9}{'us(med)':>9}{'GB/s':>7}")

    def measure(label, seg, stride, shared4, rows):
        n_read = rows * seg
        times = []
        for rep in range(3 + ROT):
            # column offsets 256 B apart keep rotations on distinct L2 lines
            off = ((rep % ROT) * 256) % stride if stride > 256 else (rep % 4) * 0
            us = ext.bench(buf, off, seg, rows, stride, shared4, grid, sink)
            if rep >= 3:
                times.append(us)
        med = statistics.median(times)
        print(f"{label:<14}{seg:>6}{stride:>8}{n_read / 1e6:>9.1f}{med:>9.1f}{n_read / med / 1e3:>7.0f}")

    rows_w13 = BUF // 2048
    for seg in (32, 64, 128, 256, 512, 2048):
        measure("w13", seg, 2048, 0, rows_w13)
    rows_w2 = BUF // 256
    measure("w2/single", 64, 256, 0, rows_w2)
    measure("w2/shared", 64, 256, 1, rows_w2)
    measure("linear", 2048, 2048, 0, rows_w13)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
