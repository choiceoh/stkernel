// SPDX-License-Identifier: Apache-2.0
// Radix core adapted from ST prefill_topk.cu (SGLang kpool_topk_transform, Apache-2.0).
// Repository provenance: engine/kernels/prefill_topk.cu.
//
// Decode-path DSA indexer selection: the horizon mask and the top-k in one launch.
//
// What the served path did: write -inf over the scorer's output (a full read-modify-write of
// [rows, n_cand] fp32), then hand it to torch.topk, which past ~20k columns is a 21-kernel
// multi-block selection with a global workspace, an fp32 values buffer nobody reads, and
// int64 indices that pool_slots narrows again.
//
// What this does instead, in one kernel, with no workspace and no allocation:
//   1. read the row ONCE, four floats to a thread; the horizon is a compare in registers
//   2. keep each element's 8-bit bin in shared memory and histogram it in the same sweep
//   3. pick the bin the cut falls in with ONE warp (shuffles, not an eight-deep block sweep)
//   4. sift and refine against the shared bins -- DRAM is touched again only to re-read the
//      few values that land in the threshold bin
// so the DRAM cost is one pass over the logits and the fixed cost is a handful of barriers.
// A row too wide for the bin cache falls back to reading the row a second time; a bin too
// wide for the stash falls back to refining against the row. Both are exact.
//
// The key is (ordered_float_key(score) << 32) | ~index, so an exact tie chooses the LOWER
// pool id. That is what torch.topk(sorted=False) does here: measured over single- and
// multi-block shapes with forced boundary plateaus, 0 mismatched rows (see the record).
// pool_slots consumes the SET (it rewrites the winners in descending token position), so
// matching the set matches the served slots and counts byte for byte.
#ifndef ST_DECODE_TOPK_NVRTC
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#endif

#define ST_K 512
#define ST_THREADS 1024
#define ST_RADIX 256
#define ST_LANES 32
#define ST_BINS_PER_LANE (ST_RADIX / ST_LANES)
// Replicated histograms for the sweep that reads the row were tried against the plateau of
// exact zeros a relu leaves (1, 2, 4, 8 replicas, four shapes): every arm landed inside
// 3%, so the sweep is not contending on the counter and the replicas only ate stash. One.
#define ST_REPLICAS 1

typedef unsigned int st_u32;
typedef unsigned long long st_u64;
typedef unsigned short st_u16;
typedef unsigned char st_u8;

extern "C" {

__device__ __forceinline__ st_u32 st_ordered_key(float x) {
  if (isnan(x)) return 0xffffffffu;
  if (x == 0.0f) x = 0.0f;  // -0.0 and +0.0 are one value
  const st_u32 b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// fp32 -> fp16 bits, round to nearest even, in integer ops. No cuda_fp16.h: the same
// arithmetic runs in the CPU oracle, so the binning is testable without a GPU.
__device__ __forceinline__ st_u16 st_f32_to_f16(float x) {
  const st_u32 b = __float_as_uint(x);
  const st_u32 sign = (b >> 16) & 0x8000u;
  const st_u32 raw_exp = (b >> 23) & 0xffu;
  st_u32 man = b & 0x7fffffu;
  if (raw_exp == 0xffu) return (st_u16)(sign | 0x7c00u | (man ? 0x200u : 0u));
  int exp = (int)raw_exp - 112;  // 127 - 15
  if (exp >= 0x1f) return (st_u16)(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return (st_u16)sign;
    man |= 0x800000u;
    const st_u32 shift = (st_u32)(14 - exp);  // 14 .. 24
    const st_u32 r = man >> shift;
    const st_u32 rem = man & ((1u << shift) - 1u);
    const st_u32 half = 1u << (shift - 1);
    return (st_u16)(sign | (r + ((rem > half || (rem == half && (r & 1u))) ? 1u : 0u)));
  }
  st_u32 r = man >> 13;
  const st_u32 rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
    if (++r == 0x400u) {
      r = 0;
      if (++exp >= 0x1f) return (st_u16)(sign | 0x7c00u);
    }
  }
  return (st_u16)(sign | ((st_u32)exp << 10) | r);
}

// Monotone 8-bit bin. fp16 rounding is monotone, so bin(a) > bin(b) implies a > b and the
// coarse pass can discard whole bins. Five exponent + two mantissa bits put four bins in an
// octave, which keeps the threshold bin inside the stash; the fp32 top byte would hold two
// octaves and spill on every realistic score distribution.
__device__ __forceinline__ int st_coarse_bin(float x) {
  if (isnan(x)) return 255;
  const st_u16 h = st_f32_to_f16(x);
  const st_u16 key = (h & 0x8000u) ? (st_u16)(~h) : (st_u16)(h | 0x8000u);
  return (int)(key >> 8);
}

__device__ __forceinline__ st_u64 st_stash_key(st_u32 ordered, int index) {
  return ((st_u64)ordered << 32) | (st_u32)(~(st_u32)index);
}

__device__ __forceinline__ st_u64 st_topk_key(float score, int index) {
  return st_stash_key(st_ordered_key(score), index);
}

// Plain shared atomics, one a value. Warp aggregation (__match_any_sync on the active mask,
// leader adds the population count) was tried on both users of this -- the sweep, whose bins
// are spread, and the refinement, whose candidates all share a coarse bin and so pile onto
// one counter. It lost on both (sweep: 12 shapes 154 -> 185 us; refinement: the sift stage
// 8.1 -> 8.9 us at 16 x 50688). Blackwell's shared atomics already coalesce same-address
// adds; the match/ballot pair costs more than it saves.
__device__ __forceinline__ void st_hist_add(int* bins, int bin) { atomicAdd(&bins[bin], 1); }

// Four elements' bins in one 32-bit word, in the layout the sweep wrote. Reading the cache
// a byte at a time puts four lanes on one shared-memory bank, and that four-way conflict
// cost as much as reading the row from DRAM (16 x 50688: a 10.2 us sift against a 12.3 us
// sweep). Without a cache the same word is rebuilt from a four-wide DRAM load.
__device__ __forceinline__ st_u32 st_quad_bins(const st_u8* bins, const float* in, int i) {
  if (bins) return ((const st_u32*)bins)[i];
  const float4 q = __ldg(reinterpret_cast<const float4*>(in) + i);
  return (st_u32)st_coarse_bin(q.x) | ((st_u32)st_coarse_bin(q.y) << 8)
         | ((st_u32)st_coarse_bin(q.z) << 16) | ((st_u32)st_coarse_bin(q.w) << 24);
}

// One warp turns per-bin counts into suffix counts (hist[b] = elements in bins >= b) and
// picks the bin the cut falls in. The block-wide eight-step sweep this replaces cost eight
// __syncthreads, and the refinement runs it up to eight times a row -- on the short rows
// that decode actually serves, that sweep was most of the kernel.
__device__ __forceinline__ void st_suffix_and_pick(int* hist, int need, int* picked) {
  const int lane = (int)threadIdx.x;
  if (lane >= ST_LANES) return;
  const int base = lane * ST_BINS_PER_LANE;
  int suffix[ST_BINS_PER_LANE];
  int total = 0;
#pragma unroll
  for (int i = ST_BINS_PER_LANE - 1; i >= 0; --i) {
    total += hist[base + i];
    suffix[i] = total;
  }
  int inclusive = total;  // inclusive suffix scan of lane totals
#pragma unroll
  for (int off = 1; off < ST_LANES; off <<= 1) {
    const int other = __shfl_down_sync(0xffffffffu, inclusive, off);
    if (lane + off < ST_LANES) inclusive += other;
  }
  const int above = inclusive - total;  // elements owned by higher lanes
  int found = -1;
#pragma unroll
  for (int i = 0; i < ST_BINS_PER_LANE; ++i) {
    const int ge = suffix[i] + above;
    const int gt = (i == ST_BINS_PER_LANE - 1) ? above : (suffix[i + 1] + above);
    hist[base + i] = ge;
    if (ge > need && gt <= need) found = base + i;
  }
  if (lane == 0) hist[ST_RADIX] = 0;
  if (found >= 0) *picked = found;
}

// `bin_bytes` > 0 turns on the shared bin cache (the one-DRAM-pass path). `stash_slots` is
// how many threshold-bin candidates the two refinement rings hold. Both come from the host,
// which knows the device's shared-memory budget; the kernel takes whatever it is handed.
__global__ __launch_bounds__(ST_THREADS) void st_dsa_select(
    const float* __restrict__ scores, long long row_stride, int n_cand,
    const int* __restrict__ ke, int* __restrict__ out, int stash_slots, int bin_bytes) {
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  // The block width is chosen per shape by the host, so a short row does not pay for 32
  // warps of barriers to move a few hundred elements. ST_THREADS is only the cap.
  const int nt = (int)blockDim.x;
  const float* __restrict__ in = scores + (long long)row * row_stride;
  int length = ke[row];
  length = length < 0 ? 0 : (length > n_cand ? n_cand : length);

  __shared__ int hist[ST_RADIX + 1];
  __shared__ int hist_rep[ST_REPLICAS][ST_RADIX];
  __shared__ int counter;
  __shared__ int threshold_bin;
  __shared__ int num_input[2];
  __shared__ int last_remain;
  __shared__ st_u64 key_prefix;
  __shared__ int selected[ST_K];
  // [bin cache][keys a][ids a][keys b][ids b]. The stash carries each candidate's ordered
  // score key next to its id, so the refinement rounds never go back to DRAM: the threshold
  // bin holds a few hundred elements (measured 346..587 over the bench shapes) and re-reading
  // them once a round was the whole fixed cost on a short row.
  extern __shared__ st_u32 pool[];
  st_u8* const bins = bin_bytes ? (st_u8*)pool : (st_u8*)0;
  st_u32* const ring = pool + (bin_bytes >> 2);
  st_u32* const key_of[2] = {ring, ring + 2 * stash_slots};
  int* const id_of[2] = {(int*)(ring + stash_slots), (int*)(ring + 3 * stash_slots)};

  if (length <= ST_K) {  // every visible pool wins; -1 pads, as topk_positions pads
    for (int i = tx; i < ST_K; i += nt)
      out[(long long)row * ST_K + i] = i < length ? i : -1;
    return;
  }

  for (int i = tx; i < ST_K; i += nt) selected[i] = -1;
  for (int i = tx; i < ST_REPLICAS * ST_RADIX; i += nt) hist_rep[0][i] = 0;
  if (tx == 0) {
    counter = 0;
    num_input[0] = 0;
    num_input[1] = 0;
  }
  __syncthreads();

#define ST_EMIT(idx)                       \
  {                                        \
    const int _p = atomicAdd(&counter, 1); \
    if (_p < ST_K) selected[_p] = (idx);   \
  }
#define ST_PUBLISH()                                  \
  {                                                   \
    __syncthreads();                                  \
    for (int _i = tx; _i < ST_K; _i += nt)    \
      out[(long long)row * ST_K + _i] = selected[_i]; \
  }

  // The only pass over DRAM. Four floats to a thread, bins straight to shared as one
  // 32-bit store, counts into the replica this thread owns.
  {
    int* const mine = hist_rep[tx & (ST_REPLICAS - 1)];
    const int vec = (((st_u64)(size_t)in & 15ull) == 0ull) ? (length >> 2) : 0;
    for (int i = tx; i < vec; i += nt) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(in) + i);
      const int bx = st_coarse_bin(v.x), by = st_coarse_bin(v.y);
      const int bz = st_coarse_bin(v.z), bw = st_coarse_bin(v.w);
      if (bins)
        ((st_u32*)bins)[i] = (st_u32)bx | ((st_u32)by << 8) | ((st_u32)bz << 16)
                             | ((st_u32)bw << 24);
      st_hist_add(mine, bx);
      st_hist_add(mine, by);
      st_hist_add(mine, bz);
      st_hist_add(mine, bw);
    }
    for (int i = (vec << 2) + tx; i < length; i += nt) {
      const int b = st_coarse_bin(in[i]);
      if (bins) bins[i] = (st_u8)b;
      st_hist_add(mine, b);
    }
  }
  __syncthreads();
  // Strided, not `tx < ST_RADIX`: the block can be narrower than the radix, and a merge that
  // assumed 256 threads left the high bins at zero and picked the wrong cut.
  for (int b = tx; b < ST_RADIX; b += nt) {
    int s = 0;
#pragma unroll
    for (int r = 0; r < ST_REPLICAS; ++r) s += hist_rep[r][b];
    hist[b] = s;
  }
  __syncthreads();

  int need = ST_K;
  st_suffix_and_pick(hist, need, &threshold_bin);
  __syncthreads();
  const int coarse = threshold_bin;
  need -= hist[coarse + 1];
  const int coarse_pop = hist[coarse] - hist[coarse + 1];

// Reading an element's bin: from the cache if we kept one, else recompute from DRAM.
#define ST_BIN_AT(i) (bins ? (int)bins[(i)] : st_coarse_bin(in[(i)]))
#define ST_QUAD_AT(i) st_quad_bins(bins, in, (i))
// How many of the row's elements this block walks four at a time. The bin cache is dense,
// so it is always quad-readable; without it the row has to be 16-byte aligned.
  const int quads = (bins || ((st_u64)(size_t)in & 15ull) == 0ull) ? (length >> 2) : 0;

  if (need == 0) {  // the cut falls on a bin edge: nothing to refine
    for (int i = tx; i < quads; i += nt) {
      const st_u32 q = ST_QUAD_AT(i);
      const int base = i << 2;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if ((int)((q >> (j * 8)) & 0xffu) > coarse) ST_EMIT(base + j);
    }
    for (int i = (quads << 2) + tx; i < length; i += nt)
      if (ST_BIN_AT(i) > coarse) ST_EMIT(i);
    ST_PUBLISH();
    return;
  }

  if (coarse_pop > stash_slots) {
    // A bin wider than the stash. Refine the 64-bit key a byte at a time against the row
    // itself. Bounded at eight rounds because the key is unique per element.
    if (tx == 0) key_prefix = 0;
    __syncthreads();
#pragma unroll 8
    for (int round = 0; round < 8; ++round) {
      for (int i = tx; i <= ST_RADIX; i += nt) hist[i] = 0;
      __syncthreads();
      const st_u64 prefix = key_prefix;
      const int offset = 56 - round * 8;
      for (int i = tx; i < length; i += nt) {
        if (ST_BIN_AT(i) != coarse) continue;
        const st_u64 key = st_topk_key(in[i], i);
        if (round == 0 || (key >> (64 - round * 8)) == prefix)
          st_hist_add(hist, (int)((key >> offset) & 0xffu));
      }
      __syncthreads();
      st_suffix_and_pick(hist, need, &threshold_bin);
      __syncthreads();
      const int bin = threshold_bin;
      need -= hist[bin + 1];
      if (tx == 0) key_prefix = (prefix << 8) | (st_u64)bin;
      __syncthreads();
      if (need == 0) {
        const st_u64 chosen = key_prefix;
        const int bits = (round + 1) * 8;
        for (int i = tx; i < length; i += nt) {
          const int b = ST_BIN_AT(i);
          if (b > coarse || (b == coarse && (st_topk_key(in[i], i) >> (64 - bits)) > chosen))
            ST_EMIT(i);
        }
        ST_PUBLISH();
        return;
      }
    }
    const st_u64 cut = key_prefix;  // all 64 bits resolved
    for (int i = tx; i < length; i += nt)
      if (st_topk_key(in[i], i) >= cut) ST_EMIT(i);
    ST_PUBLISH();
    return;
  }

  // Sift against the shared bins: winners above the threshold bin go straight out, the
  // bin's own candidates go to the stash. Only these few touch DRAM again.
  for (int i = tx; i <= ST_RADIX; i += nt) hist[i] = 0;
  __syncthreads();
  // One atomic a thread a class instead of one an element: a thread classifies its four
  // bins into two masks and claims that many slots at once.
#define ST_SIFT_QUAD(q, base)                                            \
  {                                                                      \
    int _hi = 0, _eq = 0;                                                \
    _Pragma("unroll") for (int _j = 0; _j < 4; ++_j) {                    \
      const int _b = (int)(((q) >> (_j * 8)) & 0xffu);                   \
      if (_b > coarse) _hi |= 1 << _j;                                   \
      else if (_b == coarse) _eq |= 1 << _j;                             \
    }                                                                    \
    if (_hi) {                                                           \
      int _p = atomicAdd(&counter, __popc(_hi));                         \
      _Pragma("unroll") for (int _j = 0; _j < 4; ++_j)                    \
        if (_hi & (1 << _j)) {                                           \
          if (_p < ST_K) selected[_p] = (base) + _j;                     \
          ++_p;                                                          \
        }                                                                \
    }                                                                    \
    if (_eq) {                                                           \
      int _p = atomicAdd(&num_input[0], __popc(_eq));                    \
      _Pragma("unroll") for (int _j = 0; _j < 4; ++_j)                    \
        if (_eq & (1 << _j)) {                                           \
          if (_p < stash_slots) {                                        \
            const int _idx = (base) + _j;                                \
            const st_u32 _key = st_ordered_key(in[_idx]);                \
            key_of[0][_p] = _key;                                        \
            id_of[0][_p] = _idx;                                         \
            st_hist_add(hist, (int)((_key >> 24) & 0xffu));              \
          }                                                              \
          ++_p;                                                          \
        }                                                                \
    }                                                                    \
  }
  for (int i = tx; i < quads; i += nt) ST_SIFT_QUAD(ST_QUAD_AT(i), i << 2)
  for (int i = (quads << 2) + tx; i < length; i += nt) {
    const int b = ST_BIN_AT(i);
    if (b > coarse) {
      ST_EMIT(i);
    } else if (b == coarse) {
      const int p = atomicAdd(&num_input[0], 1);
      if (p < stash_slots) {
        const st_u32 key = st_ordered_key(in[i]);
        key_of[0][p] = key;
        id_of[0][p] = i;
        st_hist_add(hist, (int)((key >> 24) & 0xffu));
      }
    }
  }
#undef ST_SIFT_QUAD
  __syncthreads();

#pragma unroll 8
  for (int round = 0; round < 8; ++round) {
    const int ring = round & 1;
    const st_u32* const src_key = key_of[ring];
    const int* const src_id = id_of[ring];
    st_u32* const dst_key = key_of[ring ^ 1];
    int* const dst_id = id_of[ring ^ 1];
    const int raw = num_input[ring];
    const int count = raw < stash_slots ? raw : stash_slots;

    if (raw == need) {  // the stash is exactly the remainder
      for (int i = tx; i < count; i += nt) ST_EMIT(src_id[i]);
      break;
    }

    st_suffix_and_pick(hist, need, &threshold_bin);
    __syncthreads();
    const int bin = threshold_bin;
    const int offset = 56 - round * 8;
    if (tx == 0) {
      num_input[ring ^ 1] = 0;
      last_remain = need - hist[bin + 1];
    }
    need -= hist[bin + 1];
    __syncthreads();
    if (need == 0) {
      for (int i = tx; i < count; i += nt) {
        const int idx = src_id[i];
        if ((int)((st_stash_key(src_key[i], idx) >> offset) & 0xffu) > bin) ST_EMIT(idx);
      }
      break;
    }

    for (int i = tx; i <= ST_RADIX; i += nt) hist[i] = 0;
    __syncthreads();
    for (int i = tx; i < count; i += nt) {
      const int idx = src_id[i];
      const st_u32 ordered = src_key[i];
      const st_u64 key = st_stash_key(ordered, idx);
      const int b = (int)((key >> offset) & 0xffu);
      if (b > bin) {
        ST_EMIT(idx);
      } else if (b == bin) {
        if (round == 7) {  // 64 bits resolved: the keys are unique, take any `need` of them
          const int p = atomicAdd(&last_remain, -1);
          if (p > 0) ST_EMIT(idx);
        } else {
          const int p = atomicAdd(&num_input[ring ^ 1], 1);
          if (p < stash_slots) {
            dst_key[p] = ordered;
            dst_id[p] = idx;
            st_hist_add(hist, (int)((key >> (offset - 8)) & 0xffu));
          }
        }
      }
    }
    __syncthreads();
  }
  ST_PUBLISH();

#undef ST_BIN_AT
#undef ST_EMIT
#undef ST_PUBLISH
}

}  // extern "C"

#ifndef ST_DECODE_TOPK_NVRTC
void run(torch::Tensor scores, torch::Tensor ke, torch::Tensor out, int64_t stash_slots,
         int64_t bin_bytes, int64_t threads) {
  TORCH_CHECK(scores.is_cuda() && ke.is_cuda() && out.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(scores.device() == ke.device() && scores.device() == out.device(), "device mismatch");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.dim() == 2 && scores.stride(1) == 1,
              "strided-row FP32 scores required");
  TORCH_CHECK(ke.scalar_type() == torch::kInt32 && ke.dim() == 1 && ke.is_contiguous()
              && ke.size(0) == scores.size(0), "int32 horizon lengths required");
  TORCH_CHECK(out.scalar_type() == torch::kInt32 && out.is_contiguous() && out.dim() == 2
              && out.size(0) == scores.size(0) && out.size(1) == ST_K, "output must be [rows, 512]");
  TORCH_CHECK(scores.size(0) > 0 && scores.size(0) <= 4096 && scores.size(1) > 0
              && scores.size(1) <= 1 << 22, "unsupported decode extent");
  TORCH_CHECK(stash_slots >= ST_K && stash_slots <= 1 << 20, "stash must hold at least k");
  TORCH_CHECK(bin_bytes == 0 || (bin_bytes % 4 == 0 && bin_bytes >= scores.size(1)),
              "the bin cache must cover the row and stay 4-byte aligned");
  TORCH_CHECK(threads >= 32 && threads <= ST_THREADS && (threads & (threads - 1)) == 0,
              "block width must be a power of two in [32, 1024]");
  const c10::cuda::CUDAGuard guard(scores.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  // Two ping-pong rings, each with a score-key array and an id array.
  // The final id ring begins at ring + 3 * stash_slots in st_dsa_select.
  const size_t smem = (size_t)bin_bytes + 4 * (size_t)stash_slots * sizeof(int);
  static int configured = 0;
  if (configured < (int)smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(st_dsa_select,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured = (int)smem;
  }
  st_dsa_select<<<scores.size(0), (int)threads, smem, stream>>>(
      scores.data_ptr<float>(), scores.stride(0), (int)scores.size(1),
      ke.data_ptr<int32_t>(), out.data_ptr<int32_t>(), (int)stash_slots, (int)bin_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "ST decode DSA horizon mask + radix top-512");
}
#endif
