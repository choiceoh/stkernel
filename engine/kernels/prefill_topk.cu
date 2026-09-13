// SPDX-License-Identifier: Apache-2.0
// Radix core adapted from SGLang kpool_topk_transform (Apache-2.0).
// Repository provenance: 5450b8cf^:overlay/modules/glm53_kernels/glm53_kpool_topk.cu.
// ST prefill-only pool selection: no framework dependency or token expansion.
// Tie order is deterministic lower index; full consumer quality must be checked.
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
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
constexpr int kThreads = 1024;
constexpr std::size_t kDynamicSmem = 8 * 1024 * sizeof(uint32_t);

__device__ __forceinline__ uint8_t coarse_key(float x) {
  if (isnan(x)) return 255;
  if (x == 0.0f) x = 0.0f;
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key =
      (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                      : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t ordered_float_key(float x) {
  if (isnan(x)) return 0xffffffffu;
  if (x == 0.0f) x = 0.0f;
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint64_t topk_key(float score, int index) {
  // Larger scores win; exact ties choose the lower pool index.
  return (static_cast<uint64_t>(ordered_float_key(score)) << 32) |
         static_cast<uint32_t>(~static_cast<uint32_t>(index));
}

template <int K>
__device__ __forceinline__ void append_index(int* index, int* counter, int value) {
  const int position = atomicAdd(counter, 1);
  if (position < K) index[position] = value;
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
        append_index<K>(index, &counter, idx);
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
            append_index<K>(index, &counter, idx);
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
        append_index<K>(index, &counter, idx);
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
      append_index<K>(index, &counter, idx);
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
        append_index<K>(index, &counter, input_idx[ring][i]);
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
        if (bin > threshold) append_index<K>(index, &counter, idx);
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
        append_index<K>(index, &counter, idx);
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

__global__ __launch_bounds__(kThreads) void select_pools(
    const float* __restrict__ scores, int64_t stride, int cols,
    const int32_t* __restrict__ lengths, int32_t* __restrict__ output) {
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const int length = max(0, min(cols, lengths[row]));
  if (length <= kGroupTopK) {
    if (tx < kGroupTopK)
      output[static_cast<int64_t>(row) * kGroupTopK + tx] = tx < length ? tx : -1;
    return;
  }
  __shared__ int selected[kGroupTopK];
  if (tx < kGroupTopK) selected[tx] = -2;
  __syncthreads();
  radix_topk<kGroupTopK>(scores + static_cast<int64_t>(row) * stride, selected, 0, length);
  if (tx < kGroupTopK)
    output[static_cast<int64_t>(row) * kGroupTopK + tx] = selected[tx];
}
}  // namespace

void run(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output) {
  TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device(), "device mismatch");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.dim() == 2 && scores.stride(1) == 1,
              "strided-row FP32 scores required");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt32 && lengths.dim() == 1 && lengths.is_contiguous()
              && lengths.size(0) == scores.size(0), "int32 valid lengths required");
  TORCH_CHECK(output.scalar_type() == torch::kInt32 && output.is_contiguous() && output.dim() == 2
              && output.size(0) == scores.size(0) && output.size(1) == kGroupTopK, "output must be [rows,512]");
  TORCH_CHECK(scores.size(0) > 0 && scores.size(0) <= 32768 && scores.size(1) > 0
              && scores.size(1) <= 262144, "unsupported prefill extent");
  const c10::cuda::CUDAGuard guard(scores.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  select_pools<<<scores.size(0), kThreads, kDynamicSmem, stream>>>(
      scores.data_ptr<float>(), scores.stride(0), scores.size(1),
      lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "ST prefill radix top-512 with valid-prefix masking");
}
