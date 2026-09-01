// SPDX-License-Identifier: Apache-2.0
// Radix core adapted from SGLang kpool_topk_transform (Apache-2.0).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

#ifndef C10_LIKELY
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#endif

constexpr int kGroupTopK = 512;
constexpr int kPoolSize = 4;
constexpr int kTokenTopK = 2048;
constexpr int kOutCols = 2051;
constexpr int kThreads = 1024;
constexpr std::size_t kDynamicSmem = 8 * 1024 * sizeof(uint32_t);

__device__ __forceinline__ uint8_t coarse_key(float x) {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key =
      (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                      : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t ordered_float_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint64_t topk_key(float score, int index) {
  // Larger scores win; exact ties choose the lower pool index.
  return (static_cast<uint64_t>(ordered_float_key(score)) << 32) |
         static_cast<uint32_t>(~static_cast<uint32_t>(index));
}

template <int K>
__device__ void sort_selected_indices(int* index) {
  const int tx = threadIdx.x;
#pragma unroll
  for (int size = 2; size <= K; size <<= 1) {
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      if (tx < K / 2) {
        const int lower = (tx / stride) * (2 * stride) + tx % stride;
        const int upper = lower + stride;
        const int lv = index[lower];
        const int uv = index[upper];
        const int lo = lv < uv ? lv : uv;
        const int hi = lv < uv ? uv : lv;
        const bool ascending = (lower & size) == 0;
        index[lower] = ascending ? lo : hi;
        index[upper] = ascending ? hi : lo;
      }
      __syncthreads();
    }
  }
}

template <int K>
__device__ void radix_topk(const float* __restrict__ input,
                           int* __restrict__ index, int row_start,
                           int length) {
  int topk = K;
  constexpr int kBlock = 1024;
  constexpr int kRadix = 256;
  constexpr int kSmemInputSize = kDynamicSmem / (2 * sizeof(int));

  alignas(128) __shared__ int hist_buf[2][kRadix + 128];
  alignas(128) __shared__ int counter;
  alignas(128) __shared__ int threshold_bin_id;
  alignas(128) __shared__ int num_input[2];
  alignas(128) __shared__ uint64_t key_prefix;
  auto& hist = hist_buf[0];
  extern __shared__ int input_idx[][kSmemInputSize];
  const int tx = threadIdx.x;

  if (tx < kRadix + 1) hist[tx] = 0;
  __syncthreads();
  for (int idx = tx; idx < length; idx += kBlock) {
    atomicAdd(&hist[coarse_key(input[idx + row_start])], 1);
  }
  __syncthreads();

  const auto reverse_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (C10_LIKELY(tx < kRadix)) {
        const int jump = 1 << i;
        const int bank = i & 1;
        int value = hist_buf[bank][tx];
        if (tx < kRadix - jump) value += hist_buf[bank][tx + jump];
        hist_buf[bank ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  reverse_cumsum();
  if (tx < kRadix && hist[tx] > topk && hist[tx + 1] <= topk) {
    threshold_bin_id = tx;
    num_input[0] = 0;
    counter = 0;
  }
  __syncthreads();

  const int coarse_threshold = threshold_bin_id;
  topk -= hist[coarse_threshold + 1];
  const int threshold_candidates =
      hist[coarse_threshold] - hist[coarse_threshold + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += kBlock) {
      if (static_cast<int>(coarse_key(input[idx + row_start])) >
          coarse_threshold) {
        index[atomicAdd(&counter, 1)] = idx;
      }
    }
    __syncthreads();
    return;
  }

  if (threshold_candidates > kSmemInputSize) {
    // Global deterministic fallback for a coarse bin that cannot fit in the
    // shared candidate ring. Refine the score/index key one byte at a time.
    if (tx == 0) key_prefix = 0;
    __syncthreads();
#pragma unroll 8
    for (int round = 0; round < 8; ++round) {
      if (tx < kRadix + 1) hist[tx] = 0;
      __syncthreads();
      const uint64_t prefix = key_prefix;
      const int offset = 56 - round * 8;
      for (int idx = tx; idx < length; idx += kBlock) {
        const float value = input[idx + row_start];
        if (coarse_key(value) != coarse_threshold) continue;
        const uint64_t key = topk_key(value, idx);
        const bool match =
            round == 0 || (key >> (64 - round * 8)) == prefix;
        if (match) atomicAdd(&hist[(key >> offset) & 0xff], 1);
      }
      __syncthreads();
      reverse_cumsum();
      if (tx < kRadix && hist[tx] > topk && hist[tx + 1] <= topk) {
        threshold_bin_id = tx;
      }
      __syncthreads();
      const int key_bin = threshold_bin_id;
      topk -= hist[key_bin + 1];
      if (tx == 0) key_prefix = (prefix << 8) | key_bin;
      __syncthreads();
      if (topk == 0) {
        const uint64_t selected_prefix = key_prefix;
        const int prefix_bits = (round + 1) * 8;
        if (tx == 0) counter = 0;
        __syncthreads();
        for (int idx = tx; idx < length; idx += kBlock) {
          const float value = input[idx + row_start];
          const uint64_t key = topk_key(value, idx);
          const uint64_t candidate_prefix = key >> (64 - prefix_bits);
          if (coarse_key(value) > coarse_threshold ||
              (coarse_key(value) == coarse_threshold &&
               candidate_prefix > selected_prefix)) {
            index[atomicAdd(&counter, 1)] = idx;
          }
        }
        __syncthreads();
        return;
      }
    }
    const uint64_t threshold_key = key_prefix;
    if (tx == 0) counter = 0;
    __syncthreads();
    for (int idx = tx; idx < length; idx += kBlock) {
      if (topk_key(input[idx + row_start], idx) >= threshold_key) {
        index[atomicAdd(&counter, 1)] = idx;
      }
    }
    __syncthreads();
    return;
  }

  if (tx < kRadix + 1) hist[tx] = 0;
  __syncthreads();
  for (int idx = tx; idx < length; idx += kBlock) {
    const float value = input[idx + row_start];
    const int bin = coarse_key(value);
    if (bin > coarse_threshold) {
      index[atomicAdd(&counter, 1)] = idx;
    } else if (bin == coarse_threshold) {
      const int pos = atomicAdd(&num_input[0], 1);
      if (C10_LIKELY(pos < kSmemInputSize)) {
        input_idx[0][pos] = idx;
        atomicAdd(&hist[(ordered_float_key(value) >> 24) & 0xff], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll 8
  for (int round = 0; round < 8; ++round) {
    __shared__ int last_remain;
    const int ring = round & 1;
    const int raw_count = num_input[ring];
    const int count = raw_count < kSmemInputSize ? raw_count : kSmemInputSize;

    if (raw_count == topk && raw_count <= kSmemInputSize) {
      for (int i = tx; i < count; i += kBlock) {
        index[atomicAdd(&counter, 1)] = input_idx[ring][i];
      }
      __syncthreads();
      break;
    }

    reverse_cumsum();
    if (tx < kRadix && hist[tx] > topk && hist[tx + 1] <= topk) {
      threshold_bin_id = tx;
      num_input[ring ^ 1] = 0;
      last_remain = topk - hist[tx + 1];
    }
    __syncthreads();
    const int threshold = threshold_bin_id;
    topk -= hist[threshold + 1];
    if (topk == 0) {
      for (int i = tx; i < count; i += kBlock) {
        const int idx = input_idx[ring][i];
        const int offset = 56 - round * 8;
        const int bin =
            (topk_key(input[idx + row_start], idx) >> offset) & 0xff;
        if (bin > threshold) index[atomicAdd(&counter, 1)] = idx;
      }
      __syncthreads();
      break;
    }

    if (tx < kRadix + 1) hist[tx] = 0;
    __syncthreads();
    for (int i = tx; i < count; i += kBlock) {
      const int idx = input_idx[ring][i];
      const uint64_t key = topk_key(input[idx + row_start], idx);
      const int offset = 56 - round * 8;
      const int bin = (key >> offset) & 0xff;
      if (bin > threshold) {
        index[atomicAdd(&counter, 1)] = idx;
      } else if (bin == threshold) {
        if (round == 7) {
          const int pos = atomicAdd(&last_remain, -1);
          if (pos > 0) index[K - pos] = idx;
        } else {
          const int pos = atomicAdd(&num_input[ring ^ 1], 1);
          if (C10_LIKELY(pos < kSmemInputSize)) {
            input_idx[ring ^ 1][pos] = idx;
            atomicAdd(&hist[(key >> (offset - 8)) & 0xff], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreads) void kpool_topk_kernel(
    const float* __restrict__ scores, int64_t score_stride,
    const int32_t* __restrict__ row_starts,
    const int32_t* __restrict__ row_ends,
    const int32_t* __restrict__ seq_lens, int32_t* __restrict__ output,
    int64_t output_stride) {
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const int row_start = row_starts[row];
  const int length = row_ends[row] - row_start;
  int32_t* dst = output + static_cast<int64_t>(row) * output_stride;
  const int history_len = min(length * kPoolSize, kTokenTopK);
  const int tail_count = seq_lens[row] % kPoolSize;

  if (length <= kGroupTopK) {
    for (int col = tx; col < kOutCols; col += kThreads) {
      if (col < history_len) {
        dst[col] = col;
      } else if (col < history_len + tail_count) {
        dst[col] = length * kPoolSize + col - history_len;
      } else {
        dst[col] = -1;
      }
    }
    return;
  }

  __shared__ int selected[kGroupTopK];
  radix_topk<kGroupTopK>(
      scores + static_cast<int64_t>(row) * score_stride, selected,
      row_start, length);
  sort_selected_indices<kGroupTopK>(selected);
  for (int col = tx; col < kOutCols; col += kThreads) {
    if (col < history_len) {
      const int group = selected[col / kPoolSize];
      dst[col] = group * kPoolSize + col % kPoolSize;
    } else if (col < history_len + tail_count) {
      dst[col] = length * kPoolSize + col - history_len;
    } else {
      dst[col] = -1;
    }
  }
}

void run(torch::Tensor scores, torch::Tensor row_starts,
         torch::Tensor row_ends, torch::Tensor seq_lens,
         torch::Tensor output) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat,
              "scores must be CUDA float32");
  TORCH_CHECK(row_starts.is_cuda() && row_starts.scalar_type() == at::kInt,
              "row_starts must be CUDA int32");
  TORCH_CHECK(row_ends.is_cuda() && row_ends.scalar_type() == at::kInt,
              "row_ends must be CUDA int32");
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == at::kInt,
              "seq_lens must be CUDA int32");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kInt,
              "output must be CUDA int32");
  TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1,
              "scores must be inner-contiguous 2D");
  TORCH_CHECK(output.dim() == 2 && output.size(1) == kOutCols &&
                  output.stride(1) == 1,
              "output must be [rows,2051] inner-contiguous");
  TORCH_CHECK(row_starts.numel() == scores.size(0) &&
                  row_ends.numel() == scores.size(0) &&
                  seq_lens.numel() == scores.size(0) &&
                  output.size(0) == scores.size(0),
              "row metadata shape mismatch");

  c10::cuda::CUDAGuard guard(scores.device());
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kpool_topk_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynamicSmem)));
    configured = true;
  }
  const auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
  kpool_topk_kernel<<<scores.size(0), kThreads, kDynamicSmem, stream>>>(
      scores.data_ptr<float>(), scores.stride(0),
      row_starts.data_ptr<int32_t>(), row_ends.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
      output.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &run, "GLM53 fused KPool top-k/expand/tail");
}
