// SPDX-License-Identifier: Apache-2.0
// The GLM-5.3 decode router in ONE launch: IEEE FP32 logits, noaux_tc selection, weights.
//
// The served chain is seven launches per MoE layer (engine/kernels/glm_pointwise.py and
// router_fp32.cpp): x.to(float), cuBLAS sgemm and its split-K reduce, `_scores` (sigmoid + bias),
// torch.topk (gatherTopK + bitonicSortKVInPlace), `_weights`. Forty-two layers a step, so about
// 300 launches whose bytes are one 4.7 MB FP32 gate read per layer.
//
// Here 96 CTAs, two to an SM, stream the resident FP32 gate once -- three experts a CTA, one 16 KB
// row per expert, evict-first, the warp's whole 6 KB of gate requested before its first product --
// and take the IEEE FP32 products against every row (BF16 promoted exactly) in a FIXED order: a
// lane's sequential fmaf chain over its k, a shuffle tree across the warp (lane 0's association
// kept), the eight warps in order. Two CTAs an SM is what the register budget allows at three
// experts (six experts a CTA needed 255 registers, one CTA an SM, and the loads then streamed at
// ~100 GB/s however they were issued: c2rt2/c2rt3-0917; staging the six rows by cp.async.bulk
// into 96 KB of shared memory was slower still, c2rt4-0917, the copies landing before any product;
// requesting every row's x for a chunk before the chunk's products won 1.7% at 8 rows and lost 5%
// at 16, c2rt6-0917, so the ROWS template picks that order at 8). At ~43 us a layer for one 4.7 MB
// gate the 16-row cut is still not DRAM-bound; the next question is a profile, not another blind
// cut (measurements/st_c2_levers_survey_20260917).
// The last CTA to arrive (a ticket, like the mHC tails) selects for every row: score =
// div_rn(1, 1 + exp(-logit)) + bias. Boundary ties select the lower expert id; the selected
// columns then follow the pinned PyTorch gather/bitonic order. Weight = s / (sum s + 1e-20) * scale
// from the RAW sigmoid with Triton's 4/2/1 reduction. The ticket resets itself for graph replay.
//
// Same products, same formulas, another projection add order: a logit moves by a few ulps and a near-tied
// top-8 boundary can flip. That is a serving-numerics change, so adoption is a bracket; this cell
// only sizes the launch-count prize and the flip rate (probes/engine_router_cells.py).
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

#define ST_RT_EXPERTS 288
#define ST_RT_HIDDEN 4096
#define ST_RT_CTAS 96
#define ST_RT_PER_CTA 3      // 288 experts / 96 CTAs
#define ST_RT_THREADS 256
#define ST_RT_WARPS 8
#define ST_RT_KSLICE 512     // 4096 / 8 warps: one warp owns one k slice of every row
#define ST_RT_LANE_CHUNKS 4  // 512 / (32 lanes x 4 floats)
#define ST_RT_SLOTS 9        // 288 / 32: the experts one lane owns at selection
#define ST_RT_TOPK 8
#define ST_RT_MAX_ROWS 16

template <typename X>
__device__ __forceinline__ void st_rt_load4(const X* p, float& a, float& b, float& c, float& d);

template <>
__device__ __forceinline__ void st_rt_load4<float>(const float* p, float& a, float& b, float& c, float& d) {
  const float4 v = *reinterpret_cast<const float4*>(p);
  a = v.x; b = v.y; c = v.z; d = v.w;
}

template <>
__device__ __forceinline__ void st_rt_load4<__nv_bfloat16>(const __nv_bfloat16* p, float& a, float& b, float& c,
                                                           float& d) {
  const uint2 v = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
  const float2 f0 = __bfloat1622float2(lo), f1 = __bfloat1622float2(hi);
  a = f0.x; b = f0.y; c = f1.x; d = f1.y;
}

// tl.div_rn(1., 1. + libdevice.exp(-x)): the same libdevice exp, an IEEE division
__device__ __forceinline__ float st_rt_sigmoid(float v) { return __fdiv_rn(1.0f, 1.0f + expf(-v)); }

// Match the pinned PyTorch CUDA topk tail without another launch. Its single-block gather
// emits scores above the kth score in expert-id order, then boundary ties in id order;
// SmallBitonicSort sorts those eight entries in a 32-entry network, including invalid padding.
// Running the same comparisons preserves even its otherwise unspecified tie permutation.
// Only tied rows need this work: every strictly ordered row already has the same output order.
__device__ __forceinline__ void st_rt_topk_order(int lane, int& id, float& lg, float& score) {
  const float previous = __shfl_up_sync(0xffffffffu, score, 1);
  if (!__ballot_sync(0xffffffffu, lane > 0 && lane < ST_RT_TOPK && score == previous)) return;
  const float boundary = __shfl_sync(0xffffffffu, score, ST_RT_TOPK - 1);
  int gather_rank = 0;
#pragma unroll
  for (int r = 0; r < ST_RT_TOPK; ++r) {
    const int other_id = __shfl_sync(0xffffffffu, id, r);
    const float other_score = __shfl_sync(0xffffffffu, score, r);
    gather_rank += ((score == boundary && other_score > boundary) ||
                    ((score == boundary) == (other_score == boundary) && other_id < id));
  }
  int source = 0;
#pragma unroll
  for (int r = 0; r < ST_RT_TOPK; ++r)
    if (__shfl_sync(0xffffffffu, gather_rank, r) == lane) source = r;
  id = __shfl_sync(0xffffffffu, id, source);
  lg = __shfl_sync(0xffffffffu, lg, source);
  score = __shfl_sync(0xffffffffu, score, source);
  int valid = lane < ST_RT_TOPK;
  if (!valid) { id = 0; lg = 0.f; score = 0.f; }
#pragma unroll
  for (int size = 2; size <= 32; size *= 2) {
    const bool direction = (lane & size) != 0;
#pragma unroll
    for (int stride = size / 2; stride > 0; stride /= 2) {
      const float other_score = __shfl_xor_sync(0xffffffffu, score, stride);
      const float other_lg = __shfl_xor_sync(0xffffffffu, lg, stride);
      const int other_id = __shfl_xor_sync(0xffffffffu, id, stride);
      const int other_valid = __shfl_xor_sync(0xffffffffu, valid, stride);
      const bool lower = (lane & stride) == 0;
      const bool compare = lower ? score > other_score : other_score > score;
      const bool valid_a = lower ? valid : other_valid;
      const bool valid_b = lower ? other_valid : valid;
      if (((compare && valid_a) || !valid_b) == direction) {
        score = other_score; lg = other_lg; id = other_id; valid = other_valid;
      }
    }
  }
}

template <typename X, int ROWS, bool ALIGN>
__global__ void __launch_bounds__(ST_RT_THREADS, 2)
st_router_fused(const X* __restrict__ x, const float* __restrict__ gate, const float* __restrict__ bias,
                unsigned int* ticket, float* __restrict__ logits, int* __restrict__ ids,
                float* __restrict__ weights, int rows, float scale) {
  __shared__ float partial[ST_RT_WARPS][ST_RT_PER_CTA][ROWS];
  __shared__ int last;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int e0 = blockIdx.x * ST_RT_PER_CTA;

  // phase 1: this CTA's three experts against every row, over the warp's k slice. The warp's whole
  // 6 KB of gate first (12 float4 a lane, evict-first), then the products.
  float4 g[ST_RT_LANE_CHUNKS][ST_RT_PER_CTA];
  const int kbase = warp * ST_RT_KSLICE + lane * 4;
#pragma unroll
  for (int j = 0; j < ST_RT_LANE_CHUNKS; ++j)
#pragma unroll
    for (int e = 0; e < ST_RT_PER_CTA; ++e)
      g[j][e] = __ldcs(reinterpret_cast<const float4*>(gate + (size_t)(e0 + e) * ST_RT_HIDDEN + kbase + j * 128));
  float acc[ST_RT_PER_CTA][ROWS];
#pragma unroll
  for (int e = 0; e < ST_RT_PER_CTA; ++e)
#pragma unroll
    for (int t = 0; t < ROWS; ++t) acc[e][t] = 0.f;
#pragma unroll
  for (int j = 0; j < ST_RT_LANE_CHUNKS; ++j) {
    const int k = kbase + j * 128;
    if constexpr (ROWS <= 8) {
      // eight rows: the chunk's x for every row first, one L2 round trip a chunk (c2rt6-0917: -1.7% over
      // the per-row order at 8 rows, +5% at 16, where the extra registers cost more than the round trips)
      float xa[ROWS], xb[ROWS], xc[ROWS], xd[ROWS];
#pragma unroll
      for (int t = 0; t < ROWS; ++t) {
        xa[t] = xb[t] = xc[t] = xd[t] = 0.f;
        if (t < rows) st_rt_load4<X>(x + (size_t)t * ST_RT_HIDDEN + k, xa[t], xb[t], xc[t], xd[t]);
      }
#pragma unroll
      for (int t = 0; t < ROWS; ++t) {
#pragma unroll
        for (int e = 0; e < ST_RT_PER_CTA; ++e) {
          float s = acc[e][t];
          s = fmaf(g[j][e].x, xa[t], s);
          s = fmaf(g[j][e].y, xb[t], s);
          s = fmaf(g[j][e].z, xc[t], s);
          s = fmaf(g[j][e].w, xd[t], s);
          acc[e][t] = s;
        }
      }
    } else {
      // sixteen rows: each row's x beside its products
#pragma unroll
      for (int t = 0; t < ROWS; ++t) {
        float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
        if (t < rows) st_rt_load4<X>(x + (size_t)t * ST_RT_HIDDEN + k, a, b, c, d);
#pragma unroll
        for (int e = 0; e < ST_RT_PER_CTA; ++e) {
          float s = acc[e][t];
          s = fmaf(g[j][e].x, a, s);
          s = fmaf(g[j][e].y, b, s);
          s = fmaf(g[j][e].z, c, s);
          s = fmaf(g[j][e].w, d, s);
          acc[e][t] = s;
        }
      }
    }
  }
  // the warp's tree (lane 0's association is the one kept), then the eight warps in order
#pragma unroll
  for (int e = 0; e < ST_RT_PER_CTA; ++e)
#pragma unroll
    for (int t = 0; t < ROWS; ++t) {
      float v = acc[e][t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      if (lane == 0) partial[warp][e][t] = v;
    }
  __syncthreads();
  if (tid < ST_RT_PER_CTA * ROWS) {
    const int e = tid / ROWS, t = tid % ROWS;
    if (t < rows) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < ST_RT_WARPS; ++w) v += partial[w][e][t];
      logits[t * ST_RT_EXPERTS + e0 + e] = v;
    }
  }

  // hand-off: every CTA publishes its logits, the last one to arrive selects for every row
  __threadfence();
  __syncthreads();
  if (tid == 0) last = (atomicAdd(ticket, 1u) == gridDim.x - 1) ? 1 : 0;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int t = warp; t < rows; t += ST_RT_WARPS) {
    float lg[ST_RT_SLOTS], sc[ST_RT_SLOTS];
    bool taken[ST_RT_SLOTS];
#pragma unroll
    for (int i = 0; i < ST_RT_SLOTS; ++i) {
      const int e = lane + 32 * i;
      lg[i] = __ldcg(logits + t * ST_RT_EXPERTS + e);
      sc[i] = st_rt_sigmoid(lg[i]) + bias[e];
      taken[i] = false;
    }
    int my_id = -1;
    float my_lg = 0.f, my_score = -INFINITY;
    for (int r = 0; r < ST_RT_TOPK; ++r) {
      // the lane's best untaken score; a strict compare keeps the lower slot, i.e. the lower expert id
      float best = -INFINITY;
      int slot = -1;
#pragma unroll
      for (int i = 0; i < ST_RT_SLOTS; ++i)
        if (!taken[i] && sc[i] > best) { best = sc[i]; slot = i; }
      int be = slot >= 0 ? lane + 32 * slot : 0x7fffffff;
      // the warp's best: higher score, then lower expert id
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, off);
        const int oe = __shfl_xor_sync(0xffffffffu, be, off);
        if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; }
      }
      const int owner = be & 31, won = be >> 5;
      float owner_lg = 0.f;
      if (lane == owner) {
#pragma unroll
        for (int i = 0; i < ST_RT_SLOTS; ++i)
          if (i == won) { taken[i] = true; owner_lg = lg[i]; }
      }
      const float v = __shfl_sync(0xffffffffu, owner_lg, owner);
      if (lane == r) { my_id = be; my_lg = v; my_score = best; }
    }
    if constexpr (ALIGN) st_rt_topk_order(lane, my_id, my_lg, my_score);
    // Same 4/2/1 butterfly as glm_pointwise._weights (Triton 3.7.1, eight columns).
    // The old sequential sum is retained only for the same-build component comparison.
    const float s = lane < ST_RT_TOPK ? st_rt_sigmoid(my_lg) : 0.f;
    float sum = 0.f;
    if constexpr (ALIGN) {
      sum = s;
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    } else {
#pragma unroll
      for (int r = 0; r < ST_RT_TOPK; ++r) sum += __shfl_sync(0xffffffffu, s, r);
    }
    if (lane < ST_RT_TOPK) {
      ids[t * ST_RT_TOPK + lane] = my_id;
      weights[t * ST_RT_TOPK + lane] = __fdiv_rn(s, sum + 1e-20f) * scale;
    }
  }
  __syncthreads();
  if (tid == 0) *ticket = 0u;
}

template <typename X, bool ALIGN>
static void launch(const at::Tensor& x, const at::Tensor& gate, const at::Tensor& bias, const at::Tensor& ticket,
                   at::Tensor& logits, at::Tensor& ids, at::Tensor& weights, float scale, cudaStream_t stream) {
  const int rows = (int)x.size(0);
  const X* xp = reinterpret_cast<const X*>(x.data_ptr());
  unsigned int* tk = reinterpret_cast<unsigned int*>(ticket.data_ptr<int>());
  if (rows <= 8)
    st_router_fused<X, 8, ALIGN><<<ST_RT_CTAS, ST_RT_THREADS, 0, stream>>>(
        xp, gate.data_ptr<float>(), bias.data_ptr<float>(), tk, logits.data_ptr<float>(), ids.data_ptr<int>(),
        weights.data_ptr<float>(), rows, scale);
  else
    st_router_fused<X, ST_RT_MAX_ROWS, ALIGN><<<ST_RT_CTAS, ST_RT_THREADS, 0, stream>>>(
        xp, gate.data_ptr<float>(), bias.data_ptr<float>(), tk, logits.data_ptr<float>(), ids.data_ptr<int>(),
        weights.data_ptr<float>(), rows, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(const at::Tensor& x, const at::Tensor& gate, const at::Tensor& bias, const at::Tensor& ticket,
         at::Tensor logits, at::Tensor ids, at::Tensor weights, double scale, bool align) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == ST_RT_HIDDEN && x.is_contiguous() &&
                  x.size(0) >= 1 && x.size(0) <= ST_RT_MAX_ROWS &&
                  (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat) &&
                  (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15u) == 0,
              "fused router: x must be a 16 B aligned contiguous CUDA BF16/FP32 [1..16, 4096]");
  TORCH_CHECK(gate.is_cuda() && gate.device() == x.device() && gate.scalar_type() == at::kFloat && gate.dim() == 2 &&
                  gate.size(0) == ST_RT_EXPERTS && gate.size(1) == ST_RT_HIDDEN && gate.is_contiguous() &&
                  (reinterpret_cast<uintptr_t>(gate.data_ptr()) & 15u) == 0,
              "fused router: gate must be the resident contiguous FP32 [288, 4096]");
  TORCH_CHECK(bias.is_cuda() && bias.device() == x.device() && bias.scalar_type() == at::kFloat &&
                  bias.dim() == 1 && bias.size(0) == ST_RT_EXPERTS && bias.is_contiguous(),
              "fused router: bias must be contiguous FP32 [288]");
  TORCH_CHECK(ticket.is_cuda() && ticket.device() == x.device() && ticket.scalar_type() == at::kInt &&
                  ticket.numel() == 1,
              "fused router: the ticket is one CUDA int32");
  const int64_t rows = x.size(0);
  TORCH_CHECK(logits.is_cuda() && logits.device() == x.device() && logits.scalar_type() == at::kFloat &&
                  logits.dim() == 2 && logits.size(0) == rows && logits.size(1) == ST_RT_EXPERTS &&
                  logits.is_contiguous(),
              "fused router: logits must be contiguous FP32 [rows, 288]");
  TORCH_CHECK(ids.is_cuda() && ids.device() == x.device() && ids.scalar_type() == at::kInt && ids.dim() == 2 &&
                  ids.size(0) == rows && ids.size(1) == ST_RT_TOPK && ids.is_contiguous(),
              "fused router: ids must be contiguous int32 [rows, 8]");
  TORCH_CHECK(weights.is_cuda() && weights.device() == x.device() && weights.scalar_type() == at::kFloat &&
                  weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == ST_RT_TOPK &&
                  weights.is_contiguous(),
              "fused router: weights must be contiguous FP32 [rows, 8]");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == at::kBFloat16) {
    if (align) launch<__nv_bfloat16, true>(x, gate, bias, ticket, logits, ids, weights, (float)scale, stream.stream());
    else launch<__nv_bfloat16, false>(x, gate, bias, ticket, logits, ids, weights, (float)scale, stream.stream());
  } else {
    if (align) launch<float, true>(x, gate, bias, ticket, logits, ids, weights, (float)scale, stream.stream());
    else launch<float, false>(x, gate, bias, ticket, logits, ids, weights, (float)scale, stream.stream());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "ST decode router: FP32 logits + noaux_tc top-8 + weights in one launch");
}
