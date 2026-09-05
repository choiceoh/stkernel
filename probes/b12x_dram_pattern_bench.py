#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What the b12x static MoE kernel's DRAM access PATTERN costs, measured
without the kernel: a plain CUDA streamer that reads the same bytes in the
same order the kernel's TMA does, against the same bytes read linearly.

The served decode MoE (C=1: 8 tokens x top-8 = 64 routed pairs, ~40 unique
experts, 4 intermediate slices each) streams 3.5 MB per expert at ~197 GB/s
(27차). A read-only torch.sum over the same class of bytes streams 245, and
the part's sustained ceiling is ~230-245. The kernel's loads are 2-D TMA
boxes: per k-tile, 128 weight rows x 64 B (tile_k = 128 fp4), the rows
2,048 B apart for w13 and 256 B apart for w2. That is 64 B out of every
2 KB row per step -- half an L2 line per DRAM row activation.

Rows:
  box64   the kernel's order: per 8 KB group, 128 rows x 64 B (stride 2048 /
          256 B)
  box128  a tile_k = 256 variant: 64 rows x 128 B per group
  linear  the same regions read as contiguous 8 KB chunks (what a pre-tiled
          weight layout + 1-D bulk copy would do)
  x DEPTH groups (8 KB each) kept in flight per CTA via cp.async groups
  x 48 CTAs (1/SM, the kernel's residency) and 96 (2/SM)

Verdict rule: if linear >> box64 at the kernel's depth (2 groups of 8 KB
per pipeline stage... the kernel keeps 2 stages of 18 KB in flight), the
weight layout is the lever; if box64 at depth 4-6 already reaches the
linear rate, pipeline depth is the lever; if neither moves, the ceiling is
elsewhere (frontend, tail, atomics) and the stamps in b12x_static_probe.py
decide.

    bash probes/run_mk_probe.sh probes/b12x_dram_pattern_bench.py
"""
from __future__ import annotations

import os
import statistics
import sys

import torch

DEV = "cuda"
ROWS_W13 = 128            # one FC1 N-tile (gate or up) per (expert, slice)
STRIDE_W13 = 2048         # K=4096 fp4 -> 2048 B per weight row
STRIDE_W2 = 256           # I_tp=512 fp4 -> 256 B per w2 row
GROUP = 8192              # bytes per in-flight group (one TMA stage's B tile)
GROUPS_W13 = 2 * 32       # gate 32 k-tiles + up 32 k-tiles
GROUPS_W2 = 32            # 32 output tiles
GROUPS_PER_ITEM = GROUPS_W13 + GROUPS_W2
ITEM_BYTES = GROUPS_PER_ITEM * GROUP          # 768 KB
ITEMS = 160                                    # U=40 experts x 4 slices
ROT = 8                                        # DRAM-cold rotation

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

#define GROUPS_PER_ITEM 96
#define ITEM_BYTES (96 * 8192)

__device__ __forceinline__ void cp_async_16(uint32_t smem, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Region layout of one item (768 KB, contiguous): [gate 256 KB][up 256 KB][w2 256 KB].
// gate/up: 128 rows x 2048 B.  w2: 1024 rows x 256 B (the 128-row x 64 B tiles walk
// down the rows; the 4 slices of an expert are the 4 adjacent 64 B chunks of a row).
template <int PATTERN>
__device__ __forceinline__ const uint8_t* group_src(const uint8_t* item, int g, int tid, int j,
                                                    int slice) {
  // returns the global address of this thread's j-th 16 B chunk of group g
  if (PATTERN == 2) {  // linear: 8 KB contiguous chunks, thread t owns 64 B
    return item + (long long)g * 8192 + tid * 64 + j * 16;
  }
  if (g < 64) {  // w13: region gate (g<32) or up
    const uint8_t* region = item + (g < 32 ? 0 : 262144);
    int k = g & 31;
    if (PATTERN == 0) {  // box64: 128 rows x 64 B at column k*64
      return region + (long long)tid * 2048 + k * 64 + j * 16;
    } else {             // box128: 64 rows x 128 B at column (k>>1)*128, two groups per k-pair
      int row = (g & 1) * 64 + (tid >> 1);
      int col = (k >> 1) * 128 + (tid & 1) * 64 + j * 16;
      return region + (long long)row * 2048 + col;
    }
  } else {       // w2
    const uint8_t* region = item + 524288;
    int t = g - 64;      // output tile 0..31
    if (PATTERN == 0) {  // 128 rows x 64 B chunk at column slice*64, row stride 256
      int row = t * 128 + tid;
      return region + (long long)row * 256 + slice * 64 + j * 16;
    } else {             // 64 rows x 128 B (two slices' chunks) per group
      int row = t * 128 + (g & 1) * 64 + (tid >> 1);
      int col = (slice >> 1) * 128 + (tid & 1) * 64 + j * 16;
      return region + (long long)row * 256 + col;
    }
  }
}

template <int PATTERN, int DEPTH>
__global__ void __launch_bounds__(128) stream_k(const uint8_t* __restrict__ base,
                                                int total_items, unsigned* sink) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t smem_base = (uint32_t)__cvta_generic_to_shared(smem);
  // my items: blockIdx.x, +gridDim.x, ...
  int n_items = 0;
  for (int it = blockIdx.x; it < total_items; it += gridDim.x) ++n_items;
  const int G = n_items * GROUPS_PER_ITEM;
  unsigned acc = 0;
  int g_issue = 0;
  auto issue = [&](int g) {
    int local_item = g / GROUPS_PER_ITEM;
    int gi = g - local_item * GROUPS_PER_ITEM;
    int it = blockIdx.x + local_item * gridDim.x;
    const uint8_t* item = base + (long long)it * ITEM_BYTES;
    uint32_t slot = smem_base + (uint32_t)((g % DEPTH) * 8192);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint8_t* src = group_src<PATTERN>(item, gi, tid, j, it & 3);
      cp_async_16(slot + tid * 64 + j * 16, src);
    }
  };
  for (; g_issue < DEPTH && g_issue < G; ++g_issue) { issue(g_issue); cp_commit(); }
  for (int g = 0; g < G; ++g) {
    cp_wait<DEPTH - 1>();
    __syncthreads();
    const unsigned* s = reinterpret_cast<const unsigned*>(smem + (g % DEPTH) * 8192);
    acc ^= s[tid * 16];
    __syncthreads();
    if (g_issue < G) { issue(g_issue); ++g_issue; }
    cp_commit();
  }
  cp_wait<0>();
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

template <int P, int D>
static float run_one(const uint8_t* base, int total_items, int grid, unsigned* sink,
                     cudaStream_t stream) {
  size_t smem = (size_t)D * 8192;
  cudaFuncSetAttribute(stream_k<P, D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, stream);
  stream_k<P, D><<<grid, 128, smem, stream>>>(base, total_items, sink);
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms * 1000.f;  // us
}

float bench(torch::Tensor buf, long long offset, int pattern, int depth, int grid,
            int total_items, torch::Tensor sink) {
  const uint8_t* base = buf.data_ptr<uint8_t>() + offset;
  unsigned* s = reinterpret_cast<unsigned*>(sink.data_ptr<int32_t>());
  cudaStream_t st = 0;   // the probe runs on torch's default stream
#define CASE(P, D) if (pattern == P && depth == D) return run_one<P, D>(base, total_items, grid, s, st);
  CASE(0, 1) CASE(0, 2) CASE(0, 3) CASE(0, 4) CASE(0, 6) CASE(0, 8) CASE(0, 11)
  CASE(1, 1) CASE(1, 2) CASE(1, 3) CASE(1, 4) CASE(1, 6) CASE(1, 8) CASE(1, 11)
  CASE(2, 1) CASE(2, 2) CASE(2, 3) CASE(2, 4) CASE(2, 6) CASE(2, 8) CASE(2, 11)
#undef CASE
  return -1.f;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
float bench(torch::Tensor buf, long long offset, int pattern, int depth, int grid,
            int total_items, torch::Tensor sink);
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    os.makedirs("/tmp/b12x_pattern_build", exist_ok=True)
    return load_inline(
        name="b12x_dram_pattern",
        cpp_sources=[CPP_SRC],
        cuda_sources=["#include <torch/extension.h>\n" + CUDA_SRC],
        functions=["bench"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_121a,code=sm_121a"],
        build_directory="/tmp/b12x_pattern_build",
        verbose=False,
    )


def main() -> int:
    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()} items={ITEMS} "
          f"item={ITEM_BYTES >> 10} KB rot={ROT}")
    ext = _build()
    total = ITEMS * ITEM_BYTES
    buf = torch.empty(ROT * total + 4096, dtype=torch.uint8, device=DEV)
    buf.random_(0, 255)
    sink = torch.zeros(4, dtype=torch.int32, device=DEV)
    mb = total / 1e6
    names = {0: "box64", 1: "box128", 2: "linear"}
    print(f"{'pattern':<8}{'depth':>6}{'grid':>6}{'us(med)':>10}{'GB/s':>8}{'min':>8}{'max':>8}")
    results = {}
    for grid in (48, 96):
        for pattern in (0, 1, 2):
            for depth in (1, 2, 3, 4, 6, 8, 11):
                if grid == 96 and depth > 6:
                    continue      # 2 CTAs/SM need <= 48 KB smem each
                times = []
                for rep in range(3 + ROT):
                    off = (rep % ROT) * total
                    us = ext.bench(buf, off, pattern, depth, grid, ITEMS, sink)
                    if us < 0:
                        raise SystemExit(f"no instantiation for {pattern}/{depth}")
                    if rep >= 3:
                        times.append(us)
                med = statistics.median(times)
                results[(names[pattern], depth, grid)] = mb * 1e6 / med / 1e9
                print(f"{names[pattern]:<8}{depth:>6}{grid:>6}{med:>10.1f}"
                      f"{mb * 1e6 / med / 1e9:>8.0f}{min(times):>8.1f}{max(times):>8.1f}")
    # the two numbers the verdict rule needs
    k2 = results.get(("box64", 2, 48))
    lin = max(v for (n, d, g), v in results.items() if n == "linear")
    box = max(v for (n, d, g), v in results.items() if n == "box64")
    print(f"kernel-like (box64, depth 2, 48 CTAs): {k2:.0f} GB/s; best box64 {box:.0f}; "
          f"best linear {lin:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
