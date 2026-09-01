// SPDX-License-Identifier: Apache-2.0
// deneb fork: GLM-5.3-Flash decode megakernel for GB10 (sm_121a, 48 SM, TP=4).
//
// One persistent kernel launch per inter-collective segment of the decode
// step. Three segments (README.md carries the ledger numbers that scoped
// them; the python driver's boot self-test gates every one):
//
//   MK_SEG_MHC  -- hyper-connection (mHC) fused hc_post + hc_pre: post
//                  combine, skinny pre-GEMM, sqrsum, sigmoid split,
//                  sinkhorn, pre-apply, fused RMSNorm. Port of the TileLang
//                  pair this repo already owns (mhc_fused_tilelang +
//                  mhc_pre_big_fuse_with_norm) -- 2 launches -> 1.
//   MK_SEG_GEMM -- W8A8 skinny GEMM, M <= 32: per-token-group fp8 quant
//                  fused into a mma.sync m16n8k32 e4m3 GEMM with pow2 block
//                  scales, replacing the per_token_group_quant kernel +
//                  deepgemm launch pair. 2 launches -> 1.
//   MK_SEG_KDA  -- the whole KDA (linear-attention) block in ONE launch:
//                  in_proj GEMM, f_b/g_b low-rank gates, short conv (k=4,
//                  silu, accepted-window rollback), fine-grained gated
//                  delta-rule recurrence (16 heads, K=V=128, fp32 state),
//                  gated RMSNorm, o_proj GEMM. ~15 launches -> 1.
//
// sm_121a contract (STEP_KERNEL_MAP.md + the 2026-09-01 ledger):
//   * Ampere lineage + FP4 extension. NO WGMMA / tcgen05 / TMEM / clusters /
//     DSMEM. mma.sync (fp8 kinds) and cp.async ARE available. The W stream
//     (the only bandwidth-heavy operand) stages through a 3-buffer cp.async
//     pipeline -- 2 tiles in flight keeps DRAM saturated; a synchronous
//     load->sync->mma chain leaves ~20% of the stream idle, ~1.3 ms/step at
//     the 2 GB/step W8A8 dense footprint. TMA remains a drop-in later.
//   * 48 SMs -> fixed 48-block grid everywhere. A bigger grid deadlocked
//     the osar kernel on this part (#150); 48 is also the barrier contract.
//   * 128 KB smem/SM; this kernel's dynamic budget stays <= 27 KB.
//
// CUDA-graph safety: no host-mutated device state. The grid barrier spins
// on a never-reset monotonic ticket counter in the caller-held workspace
// (the osar done_ctr trick), so graph replay with baked pointers stays
// exact. No PDL is emitted, so same-stream launches never overlap phases.

#include <torch/extension.h>
// c10, not ATen/cuda/CUDAContext.h: that header pulls CUDAContextLight.h
// -> <cusparse.h>, which this image does not ship under /usr/local/cuda
// (it lives only under the pip nvidia/cu13 tree). The kernel needs the
// current stream and nothing else from it.
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// Three tuning constants come in through -D so the probe can sweep them
// without editing this file; the defaults here are the shipped values.
#ifndef MK_GRID_DEF
#define MK_GRID_DEF 48
#endif
#ifndef MK_MHC_GRID_DEF
#define MK_MHC_GRID_DEF 144
#endif
#ifndef MK_W_NBUF_DEF
#define MK_W_NBUF_DEF 3
#endif
constexpr int MK_GRID = MK_GRID_DEF;  // blocks per persistent gemm/kda grid
// mhc only: it takes no dynamic smem, so its occupancy is bounded by
// registers (72 x 256 = 18,432 of 65,536 per SM -> 3 blocks). At MK_GRID
// it ran 8 of a possible 48 warps per SM.
//
// This is a CEILING, not the grid. The launch takes the smaller of it and
// what the device actually has room for, and hands the result down in
// a.grid. A persistent grid that does not fit deadlocks on the grid
// barrier, so a hard constant plus an assert would turn any future
// register-count drift into a refusal to boot; clamping degrades instead.
constexpr int MK_MHC_GRID_CAP = MK_MHC_GRID_DEF;
constexpr int MK_THREADS = 256;   // 8 warps
constexpr int MK_WARPS = MK_THREADS / 32;

// GLM-5.3-Flash per-rank geometry, TP=4 (the python driver gates on every
// one of these before arming a segment).
constexpr int HC = 4;                    // mhc_num_residual_streams
constexpr int HIDDEN = 4096;
constexpr int NOUT = HC * (2 + HC);      // 24 hc pre outputs
constexpr int MAX_TOK = 32;              // C=4 x (SPEC_K+1) verify buckets
constexpr int HCHUNK = 256;
constexpr int NCHUNK = HIDDEN / HCHUNK;  // 16

constexpr int KDA_H = 16;                // local linear heads (64 / 4)
constexpr int KDA_D = 128;               // linear_head_dim
constexpr int KDA_QKV = 3 * KDA_H * KDA_D;         // 6144 merged conv channels
constexpr int KDA_INPROJ_N = KDA_QKV + KDA_H + 2 * KDA_D;  // 6416
constexpr int KDA_INPROJ_N_PAD = ((KDA_INPROJ_N + 127) / 128) * 128;  // 6528
constexpr int CONV_W = 4;                // short_conv_kernel_size
constexpr int KDA_OUT = KDA_H * KDA_D;             // 2048
constexpr int KDA_OUT_PAD = KDA_OUT;               // 2048 = 16 x 128

constexpr int KSTEP = 128;               // one scale block of K
constexpr int KBLK_MAX = 32;             // max k blocks (K <= 4096)
constexpr int SMEM_A_PITCH = KSTEP + 4;  // fp8 A tile row pitch (4B aligned)
constexpr int SMEM_W_PITCH = KSTEP + 16; // fp8 W tile row pitch (16B aligned)
constexpr int SMEM_W_ROWS = 128;         // W rows staged per k-block

// dynamic smem layout used by the GEMM routine:
//   saq[2][16][132]  fp8 A tiles       2*16*132       = 4224
//   swb[3][128][144] fp8 W pipeline    3*128*144      = 55296
//   sxs[32][32]      fp32 group scales 32*32*4        = 4096
// W pipeline depth. 3 buffers keep ~2 k-blocks in flight = 32 KB, which by
// Little's law caps this kernel far below the part's bandwidth; 4 buffers
// with a distance-3 prefetch keep ~3. smem: 4*128*144 = 73728, and
// 4224 + 73728 + 4096 = 82048 <= the 101376 opt-in this part reports.
// 3 is the occupancy cliff, and the reason to write it down: the SM has
// 128 KB of shared memory, so 3 buffers (63,616 B) leaves two blocks
// resident while 4 (82,048 B) leaves one. Deepening to 4 measured EXACTLY
// zero on this kernel and would have halved the warps per SM -- when a
// block's 8 warps all sit on the cp.async wait, a second resident block is
// the only thing that can hide the latency. Do not raise this without
// re-checking 2 * smem <= 131072.
constexpr int MK_W_NBUF = MK_W_NBUF_DEF;
constexpr int GEMM_SMEM = 2 * 16 * SMEM_A_PITCH +
                          MK_W_NBUF * SMEM_W_ROWS * SMEM_W_PITCH +
                          KBLK_MAX * KBLK_MAX * 4;

#define MK_CHECK_CUDA(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    TORCH_CHECK(_e == cudaSuccess, "megakernel cuda error: ",                \
                cudaGetErrorString(_e));                                     \
  } while (0)

// ---------------------------------------------------------------------------
// graph-safe grid barrier. Never-reset monotonic ticket counter: every
// (launch, phase) adds exactly MK_GRID arrivals; a block waits for the next
// multiple of MK_GRID at or after its own ticket, so CUDA-graph replay with
// baked workspace pointers is exact. Device-scope fences make the phases'
// global writes visible across blocks (the osar barrier lesson).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mk_grid_barrier(unsigned long long* ctr,
                                               int grid = MK_GRID) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    // 64-bit monotonic ticket: 450+ arrivals/step wrap a 32-bit counter in
    // about a week of continuous serving, and the wrapped `<` compare then
    // releases blocks early. 2^64 does not wrap.
    unsigned long long t = atomicAdd(ctr, 1ULL);
    unsigned long long target =
        (t / (unsigned long long)grid + 1ULL) * grid;
    volatile unsigned long long* v =
        (volatile unsigned long long*)ctr;
    while (*v < target) __nanosleep(64);
    __threadfence();
  }
  __syncthreads();
}

__device__ __forceinline__ float mk_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// pow2 group scale: 2^frexp_exp(amax/448) >= amax/448 always (m in [0.5,1)),
// so |scaled| <= 448 and SATFINITE never engages. The python build-side
// weight quant (glm53_megakernel.py) uses the identical rule.
__device__ __forceinline__ float mk_pow2_scale(float amax) {
  if (amax <= 0.0f || !isfinite(amax)) return 1.0f;
  int e;
  frexpf(amax * (1.0f / 448.0f), &e);
  return exp2f((float)e);
}

__device__ __forceinline__ uint8_t mk_f32_to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// cp.async (sm_80 lineage, legal on sm_121a) -- 16B global->shared copies
// that do not occupy a register while in flight. wait_group<N> stalls until
// at most N of THIS thread's committed groups are still pending; the
// __syncthreads that follows each wait publishes other threads' copies.
__device__ __forceinline__ void mk_cp_async16(void* smem_dst,
                                              const void* gmem_src) {
  uint32_t d = (uint32_t)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d),
               "l"(gmem_src));
}
template <int N>
__device__ __forceinline__ void mk_cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void mk_cp_commit() {
  asm volatile("cp.async.commit_group;\n");
}

// ===========================================================================
// shared GEMM routine: out[m, n_orig] = x[m, k] @ W[n, k]^T (W row-major).
// W: e4m3 [n, k], n and k multiples of 128 (pad rows zeroed at build);
// ws: fp32 [n/128, k/128] pow2 scales. A is quantized in place per k-block
// (this is the fusion that removes the per_token_group_quant launch).
//
//   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//   A frag (16x32): a0=(r=g,   k=t4..)     a1=(r=g+8, k=t4..)
//                    a2=(r=g,   k=16+t4..) a3=(r=g+8, k=16+t4..)
//   B frag (32x8):  b0=W[n0+g][k=t4..]     b1=W[n0+g][k=16+t4..]
//   C frag (16x8):  c0/c1=(r=g, c=t2/+1)   c2/c3=(r=g+8, c=t2/+1)
//   g = lane>>2, t4 = (lane&3)*4, t2 = (lane&3)*2.
//   W's row-major k-contiguous layout maps 1:1 onto the B fragment.
// ===========================================================================
struct MKGemmCtx {
  const __nv_bfloat16* x;  // [m, k]
  const uint8_t* wq;       // [n, k] e4m3, n % 128 == 0 (W8 path)
  const float* ws;         // [n/128, k/128] (all-ones tensor on the W4 path)
  __nv_bfloat16* out;      // [m, n_orig]
  int m, n, k, n_orig;
  // W4 path: e2m1 nibbles [n, k/2] + per-16-group scale exponents
  // [n, k/16] (int8, clamped to [-5, 6] at build). The kernel expands each
  // nibble to an EXACT e4m3 byte (1-bit mantissas always fit; the 2^s
  // product is an exponent-field add) and then runs the same e4m3 mma
  // pipeline -- the DRAM bytes halve and the arithmetic is unchanged.
  const uint8_t* wq4 = nullptr;  // non-null selects the W4 path
  const int8_t* ws4 = nullptr;
};

// e4m3 encodings of the e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
// Expansion: byte = LUT[mag] + (s << 3) | sign<<7 -- exact while the build
// keeps s in [-5, 6] (exp field stays inside [1, 15), never a denormal,
// never the NaN encoding: LUT mantissas are never 111).
__device__ __constant__ uint8_t mk_e2m1_to_e4m3[8] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C};

// Remainder split-K state. Every block owns one 128-column tile across the
// whole k range, so a tile count that is not a multiple of MK_GRID pays a
// WHOLE extra round for its leftovers: phase 0's 51 tiles take 2 rounds and
// the second carries 3 tiles while 45 blocks idle -- 2 tile-times for 1.06
// tiles of work, and phase 0 is ~43% of the KDA segment.
//
// The leftovers get their k split instead. rem < MK_GRID by construction, so
// the accumulator is at most 32 x (47*128) floats; it lives here rather than
// in the ctx so neither host entry point nor the pybind signature changes.
// ksr * rem <= MK_GRID by construction, so the per-split accumulator is
// bounded by m * 128 * MK_GRID floats regardless of n.
constexpr int MK_SPLIT_MAXCOL = (MK_GRID - 1) * 128;  // 6016
constexpr int MK_SPLIT_ELEMS = 32 * 128 * MK_GRID;    // 196608 = 768 KB
__device__ unsigned long long g_mk_gemm_bar = 0ULL;
__device__ float g_mk_gemm_partial[MK_SPLIT_ELEMS];

__device__ void mk_gemm_phase(const MKGemmCtx& c, uint8_t* smem) {
  const int nblk = c.n / 128;
  const int kblk = c.k / KSTEP;
  const int mtiles = (c.m + 15) / 16;
  // Leftover tiles of the last (partial) round, and how many ways to split
  // their k so those blocks are not idle. ksr == 1 leaves the original
  // single-pass path byte for byte -- which is what n/128 == 32 (o_proj)
  // and any exact multiple of MK_GRID get.
  const int rem = nblk % MK_GRID;
  const int full = nblk - rem;
  const int ksr = (rem > 0) ? (MK_GRID / rem) : 1;
  const bool split = (ksr > 1) && (c.m <= 32) &&
                     (rem * 128 <= MK_SPLIT_MAXCOL);
  const int pcols = rem * 128;
  // One slice per split, summed in a FIXED order below. An atomicAdd
  // accumulator is order-nondeterministic, so back-to-back launches of the
  // same call return bitwise-different results -- the probe's replay-
  // stability check catches exactly that, and this lane cannot afford a
  // kernel whose output depends on scheduling.
  const size_t pslice = (size_t)c.m * pcols;
  const size_t pelem = pslice * ksr;
  if (split) {
    for (size_t i = (size_t)blockIdx.x * MK_THREADS + threadIdx.x;
         i < pelem; i += (size_t)MK_GRID * MK_THREADS)
      g_mk_gemm_partial[i] = 0.0f;
    mk_grid_barrier(&g_mk_gemm_bar);
  }

  uint8_t* saq = smem;  // [2][16][132] fp8 A tiles (single per kb)
  uint8_t* swb = saq + 2 * 16 * SMEM_A_PITCH;  // [MK_W_NBUF][128][144]
  float* sxs =
      (float*)(swb + MK_W_NBUF * SMEM_W_ROWS * SMEM_W_PITCH);  // [32][32]

  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t4 = (lane & 3) * 4;
  const int warp = threadIdx.x >> 5;

  const int units = split ? (full + rem * ksr) : nblk;
  for (int u = blockIdx.x; u < units; u += MK_GRID) {
    int nt, kb0, kbn;
    if (!split || u < full) {          // a whole tile, one block, all of k
      nt = u; kb0 = 0; kbn = kblk;
    } else {                           // a leftover tile's k slice
      const int t = (u - full) / ksr, sp = (u - full) % ksr;
      nt = full + t;
      kb0 = (kblk * sp) / ksr;
      kbn = (kblk * (sp + 1)) / ksr;
    }
    const bool to_partial = split && (u >= full);
    if (kb0 >= kbn) continue;
    float acc[2][2][4];  // [m-tile][n8-half][c-frag]
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.0f;

    // stage one k-block of W rows [nt*128, nt*128+128) into a pipeline
    // buffer (async, 16B copies; both addresses are 16B aligned by
    // construction: pitch 144 and k in {2048, 4096}).
    auto stage_w = [&](int kb, int buf) {
      const uint8_t* wsrc =
          c.wq + (size_t)nt * SMEM_W_ROWS * c.k + kb * KSTEP;
      uint8_t* d0 = swb + buf * (SMEM_W_ROWS * SMEM_W_PITCH);
      // Flatten (row, 16B chunk) so ALL MK_THREADS issue copies. The row-
      // strided form left threads >= SMEM_W_ROWS (128 of 256) idle, halving
      // the bytes in flight -- and this stage is latency-bound, so in-flight
      // bytes ARE the bandwidth (Little's law).
      constexpr int MK_W_CHUNKS = KSTEP / 16;
      for (int t = threadIdx.x; t < SMEM_W_ROWS * MK_W_CHUNKS;
           t += MK_THREADS) {
        const int r = t / MK_W_CHUNKS;
        const int e = (t % MK_W_CHUNKS) * 16;
        mk_cp_async16(d0 + r * SMEM_W_PITCH + e,
                      wsrc + (size_t)r * c.k + e);
      }
      mk_cp_commit();
    };
    // quantize A rows [0, m) for one k-block into the fp8 m-tiles
    // One WARP per row, not one thread. The row-per-thread form used only
    // c.m of MK_THREADS (8 to 32 of 256) and walked KSTEP global elements
    // TWICE per row -- once for the max, once for the pack -- as dependent
    // scalar loads. That is 256 latency-bound loads on a handful of threads,
    // repeated for every k-block, and it dominated this kernel: mk_us stayed
    // ~700 us whether n was 1024 or 4096, i.e. independent of the actual
    // GEMM work. Each lane now owns 4 consecutive elements (32 x 4 = KSTEP),
    // reads them once into registers, and the row max comes from a butterfly
    // shuffle -- same value, since mk_pow2_scale is a pure function of it.
    auto quant_a = [&](int kb) {
      const int qw = threadIdx.x >> 5, ql = threadIdx.x & 31;
      for (int r = qw; r < c.m; r += MK_WARPS) {
        const __nv_bfloat16* src =
            c.x + (size_t)r * c.k + kb * KSTEP + ql * 4;
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = __bfloat162float(src[q]);
        float mx = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])),
                         fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
        for (int off = 16; off; off >>= 1)
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        const float sc = mk_pow2_scale(mx);
        if (ql == 0) sxs[r * KBLK_MAX + kb] = sc;
        uint8_t* dst = saq + (r >> 4) * 16 * SMEM_A_PITCH +
                       (r & 15) * SMEM_A_PITCH;
        uint32_t pack = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          pack |= (uint32_t)mk_f32_to_e4m3(v[q] / sc) << (8 * q);
        *(uint32_t*)(dst + ql * 4) = pack;
      }
      // rows >= m keep stale bytes: their output rows are never written and
      // finite e4m3 cannot poison other rows of the same mma.
    };

    // mma + per-k-block scale fold, shared by the W8 and W4 pipelines
    // (W4 tiles arrive already scale-multiplied, so their wsc is 1).
    auto mma_fold = [&](const uint8_t* sw, int kb, float wsc) {
      // ---- mma: warp covers n-cols [warp*16, warp*16+16) of the tile,
      // two m16n8 mmas per k-slice; each n8 half keeps its own kacc
      // (a shared accumulator would sum the two halves' products).
      float kacc[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) kacc[i][j][c] = 0.0f;

#pragma unroll 2
      for (int ks = 0; ks < KSTEP / 32; ++ks) {
        const int koff = ks * 32;
        uint32_t a[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const uint8_t* base = saq + i * 16 * SMEM_A_PITCH;
          a[i][0] = *(const uint32_t*)(base + g * SMEM_A_PITCH + koff + t4);
          a[i][1] =
              *(const uint32_t*)(base + (g + 8) * SMEM_A_PITCH + koff + t4);
          a[i][2] =
              *(const uint32_t*)(base + g * SMEM_A_PITCH + koff + 16 + t4);
          a[i][3] = *(const uint32_t*)(base + (g + 8) * SMEM_A_PITCH + koff +
                                       16 + t4);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int nrow = warp * 16 + j * 8 + g;
          uint32_t b0 =
              *(const uint32_t*)(sw + nrow * SMEM_W_PITCH + koff + t4);
          uint32_t b1 =
              *(const uint32_t*)(sw + nrow * SMEM_W_PITCH + koff + 16 + t4);
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(kacc[i][j][0]), "+f"(kacc[i][j][1]),
                  "+f"(kacc[i][j][2]), "+f"(kacc[i][j][3])
                : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]),
                  "r"(b0), "r"(b1));
          }
        }
      }

      // ---- fold this k-block into the accumulator with its pow2 scales.
      // ws is per (n-block, k-block); the whole staged tile shares nb == nt
      // and the warp's 16 columns sit inside it. xs is per (row, k-block).
#pragma unroll
      for (int i = 0; i < mtiles; ++i) {
        const int r0 = i * 16 + g;
        const int r1 = r0 + 8;
        const float s0 = (r0 < c.m) ? wsc * sxs[r0 * KBLK_MAX + kb] : 0.0f;
        const float s1 = (r1 < c.m) ? wsc * sxs[r1 * KBLK_MAX + kb] : 0.0f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          acc[i][j][0] += kacc[i][j][0] * s0;
          acc[i][j][1] += kacc[i][j][1] * s0;
          acc[i][j][2] += kacc[i][j][2] * s1;
          acc[i][j][3] += kacc[i][j][3] * s1;
        }
      }

    };

    if (c.wq4 != nullptr) {
      // ---- W4 path: register-staged exact expansion. The next
      // k-block's nibbles load at the top of the iteration and fly
      // through the current mma; the e2m1 -> e4m3 expansion (exact:
      // 1-bit mantissas always fit, 2^s is an exponent-field add)
      // then fills the next tile buffer. Two buffers suffice because
      // expansion and mma never touch the same one.
      const int row4 = threadIdx.x & (SMEM_W_ROWS - 1);
      const int half4 = threadIdx.x >> 7;  // 64 elems per thread
      const int nib_row = c.k / 2, sc_row = c.k / 16;
      // alignas(16): both arrays are accessed as uint4 lanes below, and a
      // plain uint8_t[] carries no alignment guarantee in local memory.
      alignas(16) uint8_t nb[32];
      int8_t sexp[4];

      auto load_w4 = [&](int kb) {
        const uint8_t* r4 = c.wq4 +
            (size_t)nt * SMEM_W_ROWS * nib_row +
            (size_t)row4 * nib_row + kb * (KSTEP / 2) + half4 * 32;
        const uint4* v4 = (const uint4*)r4;
        ((uint4*)nb)[0] = v4[0];
        ((uint4*)nb)[1] = v4[1];
        const int8_t* rs = c.ws4 +
            (size_t)nt * SMEM_W_ROWS * sc_row +
            (size_t)row4 * sc_row + kb * (KSTEP / 16) + half4 * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) sexp[i] = rs[i];
      };
      auto expand_w4 = [&](int buf) {
        alignas(16) uint8_t ob[64];
#pragma unroll
        for (int g4 = 0; g4 < 4; ++g4) {
          const int e = (sexp[g4] << 3);
#pragma unroll
          for (int b = 0; b < 8; ++b) {
            const uint8_t raw = nb[g4 * 8 + b];
            const uint8_t lo = raw & 0xF, hi = raw >> 4;
            ob[g4 * 16 + 2 * b] =
                (lo & 7) ? (uint8_t)(mk_e2m1_to_e4m3[lo & 7] + e +
                                     ((lo & 0x8) << 4))
                         : (uint8_t)((lo & 0x8) << 4);
            ob[g4 * 16 + 2 * b + 1] =
                (hi & 7) ? (uint8_t)(mk_e2m1_to_e4m3[hi & 7] + e +
                                     ((hi & 0x8) << 4))
                         : (uint8_t)((hi & 0x8) << 4);
          }
        }
        uint8_t* d0 = swb + buf * (SMEM_W_ROWS * SMEM_W_PITCH) +
                      row4 * SMEM_W_PITCH + half4 * 64;
        uint4* dv = (uint4*)d0;
#pragma unroll
        for (int i = 0; i < 4; ++i) dv[i] = ((uint4*)ob)[i];
      };

      load_w4(kb0);
      quant_a(kb0);
      expand_w4(kb0 % 2);   // the loop reads (kb % 2); kb starts at kb0
      __syncthreads();
      for (int kb = kb0;; ++kb) {
        if (kb + 1 < kbn) load_w4(kb + 1);  // flies during the mma
        const uint8_t* sw4t =
            swb + (kb % 2) * (SMEM_W_ROWS * SMEM_W_PITCH);
        mma_fold(sw4t, kb, 1.0f);  // scales already inside the bytes
        if (kb + 1 >= kbn) break;
        __syncthreads();  // mma readers of this tile buffer are done
        expand_w4((kb + 1) % 2);
        quant_a(kb + 1);
        __syncthreads();
      }
    } else {
      // ---- W8 path: MK_W_NBUF-buffer cp.async pipeline.
      // Each stage_w commits exactly one cp.async group, so "wait until at
      // most N groups outstanding" is the same as "buffer kb has landed"
      // when N is the number of stages issued after it.
      //
      // The distance is MK_W_NBUF - 1, not a literal: the tile staged while
      // kb is being read must land in a buffer nobody is reading, and with
      // the old hard-coded 2 a depth of 2 would have staged kb+2 into
      // (kb+2) % 2 == kb % 2 -- straight over the tile the mma was reading.
      // That made MK_W_NBUF a knob that silently corrupted below 3.
      static_assert(MK_W_NBUF >= 2, "the pipeline needs a spare buffer");
      constexpr int DIST = MK_W_NBUF - 1;
      stage_w(kb0, kb0 % MK_W_NBUF);
      quant_a(kb0);
#pragma unroll
      for (int d = 1; d < DIST; ++d)
        if (kb0 + d < kbn) stage_w(kb0 + d, (kb0 + d) % MK_W_NBUF);
      // Conservative above depth 3 (it waits for more than the one tile it
      // needs), exact at 2 and 3, and never unsafe.
      if (DIST > 1 && kbn - kb0 > 1) mk_cp_wait<1>(); else mk_cp_wait<0>();
      __syncthreads();

      for (int kb = kb0;; ++kb) {
        if (kb + DIST < kbn) stage_w(kb + DIST, (kb + DIST) % MK_W_NBUF);
        const uint8_t* sw =
            swb + (kb % MK_W_NBUF) * (SMEM_W_ROWS * SMEM_W_PITCH);
        mma_fold(sw, kb, c.ws[(size_t)nt * kblk + kb]);

        if (kb + 1 >= kbn) break;
        __syncthreads();  // every mma reader of saq is done first
        quant_a(kb + 1);  // ALU work while W(kb+1) finishes its flight
        // outstanding after W(kb+1): the deeper stages, when they exist
        if (DIST > 1 && kb + DIST < kbn) mk_cp_wait<1>(); else mk_cp_wait<0>();
        __syncthreads();  // publish W(kb+1) and saq(kb+1) block-wide
      }
    }
    // ---- epilogue: real rows/cols only
#pragma unroll
    for (int i = 0; i < mtiles; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {  // both n8 halves of the warp tile
        const int r0 = i * 16 + g, r1 = r0 + 8;
        const int cbase = nt * 128 + warp * 16 + j * 8 + (lane & 3) * 2;
        if (to_partial) {
          // acc already carries the per-k-block scales, so the slices sum.
          const int pc = cbase - full * 128;
          const int spx = (u - full) % ksr;
          float* pb = g_mk_gemm_partial + (size_t)spx * pslice;
          if (r0 < c.m) {
            pb[(size_t)r0 * pcols + pc] = acc[i][j][0];
            pb[(size_t)r0 * pcols + pc + 1] = acc[i][j][1];
          }
          if (r1 < c.m) {
            pb[(size_t)r1 * pcols + pc] = acc[i][j][2];
            pb[(size_t)r1 * pcols + pc + 1] = acc[i][j][3];
          }
          continue;
        }
        if (r0 < c.m) {
          if (cbase < c.n_orig)
            c.out[(size_t)r0 * c.n_orig + cbase] =
                __float2bfloat16(acc[i][j][0]);
          if (cbase + 1 < c.n_orig)
            c.out[(size_t)r0 * c.n_orig + cbase + 1] =
                __float2bfloat16(acc[i][j][1]);
        }
        if (r1 < c.m) {
          if (cbase < c.n_orig)
            c.out[(size_t)r1 * c.n_orig + cbase] =
                __float2bfloat16(acc[i][j][2]);
          if (cbase + 1 < c.n_orig)
            c.out[(size_t)r1 * c.n_orig + cbase + 1] =
                __float2bfloat16(acc[i][j][3]);
        }
      }
    }
    __syncthreads();
  }
  if (split) {  // fold the leftover tiles' slices into the bf16 output
    mk_grid_barrier(&g_mk_gemm_bar);
    for (size_t i2 = (size_t)blockIdx.x * MK_THREADS + threadIdx.x;
         i2 < pslice; i2 += (size_t)MK_GRID * MK_THREADS) {
      const int r = (int)(i2 / pcols);
      const int col = full * 128 + (int)(i2 % pcols);
      if (col >= c.n_orig) continue;
      float v = 0.0f;
      for (int spx = 0; spx < ksr; ++spx)   // fixed order -> reproducible
        v += g_mk_gemm_partial[(size_t)spx * pslice + i2];
      c.out[(size_t)r * c.n_orig + col] = __float2bfloat16(v);
    }
  }
}

__global__ void mk_gemm_kernel(const MKGemmCtx c) {
  extern __shared__ uint8_t smem[];
  mk_gemm_phase(c, smem);
}

// ===========================================================================
// MK_SEG_MHC -- fused hc_post + hc_pre (+ RMSNorm), T <= MAX_TOK.
// Port of mhc_fused_tilelang + mhc_pre_big_fuse_with_norm_tilelang
// (overlay/modules/glm53_mhc_tilelang/tilelang_kernels.py). Every rounding
// point is kept: residual_out rounds the fp32 rnew to bf16 while the sqrsum
// uses the fp32 value, and the pre-norm activation is stashed as bf16 before
// the norm consumes it.
// ===========================================================================
struct MKMhcArgs {
  const __nv_bfloat16* x_in;
  const __nv_bfloat16* residual_in;  // [T, HC, HIDDEN]
  const float* post_mix_in;          // [T, HC]
  const float* comb_mix_in;          // [T, HC*HC]
  const float* fn;                   // [NOUT, HC*HIDDEN]
  const float* hc_scale;             // [3]
  const float* hc_base;              // [NOUT]
  const __nv_bfloat16* norm_weight;  // [HIDDEN]
  __nv_bfloat16* residual_out;       // [T, HC, HIDDEN]
  float* post_mix_out;               // [T, HC]
  float* comb_mix_out;               // [T, HC*HC]
  __nv_bfloat16* layer_input;        // [T, HIDDEN]
  float* yp;                         // ws [NCHUNK, MAX_TOK, NOUT]
  float* rp;                         // ws [NCHUNK, MAX_TOK]
  float* sq;                         // ws [MAX_TOK]
  float* rsq;                        // ws [MAX_TOK]
  float* pmix;                       // ws [MAX_TOK, HC]
  __nv_bfloat16* ol_stash;           // ws [MAX_TOK, HIDDEN]
  unsigned long long* barrier_ctr;
  int grid;          // resident blocks; see MK_MHC_GRID_CAP
  int num_tokens;
  float rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps;
  int sinkhorn_repeat;
};

__device__ void mk_mhc_p1(const MKMhcArgs& a, int bid) {
  // fn is LOGICALLY re-read once per token (T x 1.5 MB) but the (t, chunk)
  // pair -> block assignment co-schedules all tokens of a chunk, so L2
  // serves T-1 of them and DRAM sees ~1.5 MB per op -- the same traffic
  // the stock grid pattern produces. Do not "fix" the re-read without
  // measuring DRAM first.
  __shared__ float red[MK_WARPS][NOUT + 1];
  const int npairs = a.num_tokens * NCHUNK;
  for (int p = bid; p < npairs; p += a.grid) {
    const int t = p / NCHUNK, c = p % NCHUNK, h0 = c * HCHUNK;
    float cm[HC][HC], pm[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      pm[j] = a.post_mix_in[t * HC + j];
#pragma unroll
      for (int k = 0; k < HC; ++k)
        cm[k][j] = a.comb_mix_in[t * HC * HC + k * HC + j];
    }
    float dot[NOUT], sqr = 0.0f;
#pragma unroll
    for (int m = 0; m < NOUT; ++m) dot[m] = 0.0f;

    for (int h = h0 + threadIdx.x; h < h0 + HCHUNK; h += MK_THREADS) {
      const float xv = __bfloat162float(a.x_in[t * HIDDEN + h]);
      float r[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float v = pm[j] * xv;
#pragma unroll
        for (int k = 0; k < HC; ++k)
          v += cm[k][j] *
               __bfloat162float(a.residual_in[(size_t)t * HC * HIDDEN +
                                              k * HIDDEN + h]);
        r[j] = v;
      }
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        a.residual_out[(size_t)t * HC * HIDDEN + j * HIDDEN + h] =
            __float2bfloat16(r[j]);
        sqr += r[j] * r[j];
      }
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int m = 0; m < NOUT; ++m)
          dot[m] += a.fn[(size_t)m * HC * HIDDEN + j * HIDDEN + h] * r[j];
    }

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 0; m < NOUT; ++m)
      for (int off = 16; off; off >>= 1)
        dot[m] += __shfl_xor_sync(~0u, dot[m], off);
    for (int off = 16; off; off >>= 1) sqr += __shfl_xor_sync(~0u, sqr, off);
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < NOUT; ++m) red[warp][m] = dot[m];
      red[warp][NOUT] = sqr;
    }
    __syncthreads();
    if (threadIdx.x < NOUT + 1) {
      float v = 0.0f;
#pragma unroll
      for (int w = 0; w < MK_WARPS; ++w) v += red[w][threadIdx.x];
      if (threadIdx.x < NOUT)
        a.yp[((size_t)c * MAX_TOK + t) * NOUT + threadIdx.x] = v;
      else
        a.rp[c * MAX_TOK + t] = v;
    }
    __syncthreads();
  }
}

__device__ void mk_mhc_p2(const MKMhcArgs& a, int bid) {
  // One warp per token, lanes 0/1/2 carry the three splits. This used to be
  // `if (blockIdx.x != 0) return;` -- 8 warps of ONE block for every token,
  // with 47 blocks idle, so the phase cost scaled with num_tokens while p1,
  // p3 and p4 all spread over `bid`. Measured: T=8 66.3 us but T=32 145.1
  // (stock 64.2 -> 79.6), i.e. 4 serial passes instead of 1. The body is
  // per-token register work with no block-wide sync, so it distributes.
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gw = bid * MK_WARPS + warp;   // 96 x 8 = 768 warps >= MAX_TOK
  for (int t = gw; t < a.num_tokens; t += a.grid * MK_WARPS) {
    float mixes[NOUT], sqr = 0.0f;
#pragma unroll
    for (int m = 0; m < NOUT; ++m) {
      float v = 0.0f;
#pragma unroll
      for (int c = 0; c < NCHUNK; ++c)
        v += a.yp[((size_t)c * MAX_TOK + t) * NOUT + m];
      mixes[m] = v;
    }
#pragma unroll
    for (int c = 0; c < NCHUNK; ++c) sqr += a.rp[c * MAX_TOK + t];
    const float rms = rsqrtf(sqr / (float)(HC * HIDDEN) + a.rms_eps);
#pragma unroll
    for (int m = 0; m < NOUT; ++m) mixes[m] *= rms;

    if (lane == 0) {  // post mixes: hc_scale[1]
#pragma unroll
      for (int j = 0; j < HC; ++j)
        a.post_mix_out[t * HC + j] =
            mk_sigmoid(mixes[j + HC] * a.hc_scale[1] + a.hc_base[j + HC]) *
            a.post_mult;
    }
    if (lane == 1) {  // comb mixes: hc_scale[2] + sinkhorn
      float cm[HC][HC];
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int k = 0; k < HC; ++k)
          cm[j][k] = mixes[j * HC + k + 2 * HC] * a.hc_scale[2] +
                     a.hc_base[j * HC + k + 2 * HC];

      float row_max[HC], row_sum[HC], col_sum[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        row_max[j] = -INFINITY;
#pragma unroll
        for (int k = 0; k < HC; ++k) row_max[j] = fmaxf(row_max[j], cm[j][k]);
      }
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int k = 0; k < HC; ++k) cm[j][k] = expf(cm[j][k] - row_max[j]);
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        row_sum[j] = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) row_sum[j] += cm[j][k];
      }
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int k = 0; k < HC; ++k)
          cm[j][k] = cm[j][k] / row_sum[j] + a.sinkhorn_eps;
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        col_sum[k] = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) col_sum[k] += cm[j][k];
      }
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int k = 0; k < HC; ++k)
          cm[j][k] = cm[j][k] / (col_sum[k] + a.sinkhorn_eps);
      for (int it = 0; it < a.sinkhorn_repeat - 1; ++it) {
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          row_sum[j] = 0.0f;
#pragma unroll
          for (int k = 0; k < HC; ++k) row_sum[j] += cm[j][k];
        }
#pragma unroll
        for (int j = 0; j < HC; ++j)
#pragma unroll
            for (int k = 0; k < HC; ++k)
              cm[j][k] = cm[j][k] / (row_sum[j] + a.sinkhorn_eps);
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          col_sum[k] = 0.0f;
#pragma unroll
          for (int j = 0; j < HC; ++j) col_sum[k] += cm[j][k];
        }
#pragma unroll
        for (int j = 0; j < HC; ++j)
#pragma unroll
            for (int k = 0; k < HC; ++k)
              cm[j][k] = cm[j][k] / (col_sum[k] + a.sinkhorn_eps);
      }
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int k = 0; k < HC; ++k)
          a.comb_mix_out[t * HC * HC + j * HC + k] = cm[j][k];
    }
    if (lane == 2) {  // pre mixes: hc_scale[0]
#pragma unroll
      for (int j = 0; j < HC; ++j)
        a.pmix[t * HC + j] =
            mk_sigmoid(mixes[j] * a.hc_scale[0] + a.hc_base[j]) + a.pre_eps;
    }
  }
}

__device__ void mk_mhc_p3(const MKMhcArgs& a, int bid) {
  // ol = sum_j pre_mix[j] * residual_out[t, j, :]; bf16 stash + sumsq
  __shared__ float sqred[MK_WARPS];
  const int npairs = a.num_tokens * NCHUNK;
  for (int p = bid; p < npairs; p += a.grid) {
    const int t = p / NCHUNK, c = p % NCHUNK, h0 = c * HCHUNK;
    float pre[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) pre[j] = a.pmix[t * HC + j];
    float sq = 0.0f;
    for (int h = h0 + threadIdx.x; h < h0 + HCHUNK; h += MK_THREADS) {
      float v = 0.0f;
#pragma unroll
      for (int j = 0; j < HC; ++j)
        v += pre[j] *
             __bfloat162float(a.residual_out[(size_t)t * HC * HIDDEN +
                                            j * HIDDEN + h]);
      a.ol_stash[t * HIDDEN + h] = __float2bfloat16(v);
      sq += v * v;
    }
    for (int off = 16; off; off >>= 1) sq += __shfl_xor_sync(~0u, sq, off);
    if ((threadIdx.x & 31) == 0) sqred[threadIdx.x >> 5] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
      float v = 0.0f;
#pragma unroll
      for (int w = 0; w < MK_WARPS; ++w) v += sqred[w];
      // One slot per (chunk, token), not atomicAdd(&sq[t]): each pair is
      // owned by exactly one block, so the store is unique, and p4 sums the
      // NCHUNK slots in a fixed order. atomicAdd made layer_input the only
      // non-deterministic output of this kernel -- 4.1e-05 to 3.3e-04 of
      // drift between identical calls at T=32, where 512 pairs over 48
      // blocks arrive in a different order each launch. (T=8 has 128 pairs
      // and happened to be stable, which is why this hid.) p1/p2 already
      // reduce their sumsq this way through rp[]; p3 was the odd one out.
      a.sq[(size_t)c * MAX_TOK + t] = v;
    }
    __syncthreads();
  }
}

__device__ void mk_mhc_p4(const MKMhcArgs& a, int bid) {
  // Every block reduces the same NCHUNK slots in the same order and so gets
  // bit-identical rsq. This replaced both the old a.rsq[] phase (one thread
  // looping over tokens, plus a grid barrier) and the atomicAdd feeding it.
  __shared__ float rsq_s[MAX_TOK];
  if (threadIdx.x < a.num_tokens) {
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < NCHUNK; ++c) sum += a.sq[c * MAX_TOK + threadIdx.x];
    rsq_s[threadIdx.x] = rsqrtf(sum / (float)HIDDEN + a.norm_eps);
  }
  __syncthreads();
  const int total = a.num_tokens * HIDDEN;
  for (int i = bid * MK_THREADS + threadIdx.x; i < total;
       i += a.grid * MK_THREADS) {
    const int t = i / HIDDEN, h = i % HIDDEN;
    a.layer_input[t * HIDDEN + h] = __float2bfloat16(
        __bfloat162float(a.ol_stash[t * HIDDEN + h]) * rsq_s[t] *
        __bfloat162float(a.norm_weight[h]));
  }
}

__global__ void mk_mhc_kernel(const MKMhcArgs a) {
  // 3 grid barriers, down from 5, and no prologue. Each surviving one is a
  // real data dependency (partials -> mixes -> pre-mixes -> sumsq). What
  // went: p3 now STORES its sumsq per (chunk, token) instead of accumulating
  // into sq[t], so there is nothing to zero and no barrier to order the
  // zeroing against; and p4 reduces those slots itself, which retired the
  // separate rsqrt phase and its barrier.
  mk_mhc_p1(a, blockIdx.x);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  mk_mhc_p2(a, blockIdx.x);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  mk_mhc_p3(a, blockIdx.x);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  // p4 does its own rsqrt, so the fourth phase boundary (sumsq -> rsqrt)
  // and the single-thread loop that used to sit on it are both gone.
  mk_mhc_p4(a, blockIdx.x);
}

// ===========================================================================
// MK_SEG_KDA -- the whole linear-attention block in one launch.
// Phases: 0 in_proj GEMM -> 1 f_b/g_b -> 2 short conv (k=4, silu, accepted
// rollback) -> 3 gated delta-rule recurrence -> 4 gated RMSNorm -> 5 o_proj
// GEMM (partial sums; the segment-boundary osar AR completes it).
// Math mirrors image glm5next/nvidia/kda.py (preimage pinned in manifest):
//   gate y[t,h,d] = lower_bound * sigmoid(exp(A_log[h]) * (g1 + dt_bias))
//   beta' = sigmoid(beta); q, k l2-normalized in fp32
//   S <- diag(y) S;  err = v - q^T S;  S <- S + beta' * k err^T;  o = q^T S
// The driver's boot self-test diffs this against the stock ops (conv update,
// fused_recurrent_kda, FusedRMSNormGated) and refuses to arm on drift.
// ===========================================================================
struct MKKdaArgs {
  const __nv_bfloat16* x;  // [T, HIDDEN] normed layer input
  const uint8_t* in_wq;    // [KDA_INPROJ_N_PAD, HIDDEN] e4m3
  const float* in_ws;      // [KDA_INPROJ_N_PAD/128, HIDDEN/128]
  const __nv_bfloat16* f_b_w;  // [KDA_OUT, KDA_D] bf16
  const __nv_bfloat16* g_b_w;  // [KDA_OUT, KDA_D] bf16
  const float* conv_w;         // [KDA_QKV, CONV_W] fp32
  // conv_state is [slots, KDA_QKV, conv_width] with conv_width >= 3: the
  // engine allocates the spec-decode sliding window (k-1 + num_spec), so a
  // hard (KDA_QKV, 3) layout gate never matches production and the takeover
  // would stay dead. The kernel addresses the active [0, 3) window with the
  // runtime width as stride (review finding).
  float* conv_state;
  int conv_width;
  float* rec_state;            // [slots, KDA_H, KDA_D, KDA_D] fp32
  const float* a_log;          // [KDA_H] fp32
  const float* dt_bias;        // [KDA_H*KDA_D] fp32
  const int* cu_seqlens;       // [n_spec+1]
  const int* state_idx;        // [n_spec, mql] state slot per query position
  const int* n_accepted;       // [n_spec]
  int n_spec, mql;
  // Delta-rule operand order. The image's fused_recurrent_kda source is
  // not in this repo, so the exact order is a BOOT-SETTLED question, not
  // an assumed one: the self-test sweeps these variants against the stock
  // op and arms with the first that matches (none matches -> DISARM).
  int delta_variant;  // 0: err = v - q^T S, write by k, readout q (post)
                      // 1: err = v - k^T S, write by k, readout q (post)
                      // 2: variant 0 with PRE-update readout
  const __nv_bfloat16* onorm_w;  // [KDA_D] affine weight (ones if unused)
  const uint8_t* o_wq;         // [HIDDEN, KDA_OUT_PAD] e4m3
  const float* o_ws;
  __nv_bfloat16* out;          // [T, HIDDEN]
  // workspace
  __nv_bfloat16* qkv;   // [MAX_TOK, KDA_INPROJ_N]
  __nv_bfloat16* g1;    // [MAX_TOK, KDA_OUT]
  __nv_bfloat16* g2;    // [MAX_TOK, KDA_OUT]
  __nv_bfloat16* convq; // [MAX_TOK, KDA_QKV]
  __nv_bfloat16* attn;  // [MAX_TOK, KDA_OUT]
  unsigned long long* barrier_ctr;
  int num_tokens;
  float lower_bound, onorm_eps;
};

__global__ void mk_kda_kernel(const MKKdaArgs a) {
  extern __shared__ uint8_t smem[];

  {  // phase 0: in_proj GEMM into workspace
    MKGemmCtx c;
    c.x = a.x;
    c.wq = a.in_wq;
    c.ws = a.in_ws;
    c.out = a.qkv;
    c.m = a.num_tokens;
    c.n = KDA_INPROJ_N_PAD;
    c.k = HIDDEN;
    c.n_orig = KDA_INPROJ_N;
    mk_gemm_phase(c, smem);
  }
  mk_grid_barrier(a.barrier_ctr);

  {  // phase 1: f_b / g_b low-rank gates (K = 128, SIMT dot)
    const int total = a.num_tokens * KDA_OUT;
    for (int i = blockIdx.x * MK_THREADS + threadIdx.x; i < 2 * total;
         i += MK_GRID * MK_THREADS) {
      const int which = i / total, rem = i - which * total;
      const int t = rem / KDA_OUT, n = rem % KDA_OUT;
      const __nv_bfloat16* src =
          a.qkv + (size_t)t * KDA_INPROJ_N + (KDA_QKV + KDA_H + which * KDA_D);
      const __nv_bfloat16* w = (which ? a.g_b_w : a.f_b_w) + (size_t)n * KDA_D;
      float v = 0.0f;
#pragma unroll 16
      for (int r = 0; r < KDA_D; ++r)
        v += __bfloat162float(src[r]) * __bfloat162float(w[r]);
      (which ? a.g2 : a.g1)[(size_t)t * KDA_OUT + n] = __float2bfloat16(v);
    }
  }
  mk_grid_barrier(a.barrier_ctr);

  {  // phase 2: merged short conv (k=4, silu) with accepted-window rollback.
    // hist(pos) is the channel's input at in-request position pos (pos < 0
    // reads the slot's conv state). Outputs are produced for ALL query
    // tokens; the state window is rolled to the accepted boundary.
    for (int r = 0; r < a.n_spec; ++r) {
      const int t0 = a.cu_seqlens[r], t1 = a.cu_seqlens[r + 1];
      const int slot = a.state_idx[r * a.mql + 0];
      const int acc = a.n_accepted[r];
      for (int ch = blockIdx.x * MK_THREADS + threadIdx.x; ch < KDA_QKV;
           ch += MK_GRID * MK_THREADS) {
        const float* w = a.conv_w + (size_t)ch * CONV_W;
        const size_t sbase = (size_t)slot * KDA_QKV * a.conv_width +
                             (size_t)ch * a.conv_width;
        // The state is conv_width wide (conv_kernel_size - 1 + num_spec, so
        // 10 here), NOT conv_width == CONV_W - 1. Its NEWEST entry is the
        // last one, so the convolution's history for pos < 0 comes off the
        // END of the buffer, not the front. Reading st[pos + CONV_W - 1]
        // took the three OLDEST entries instead.
        float st[CONV_W - 1];
#pragma unroll
        for (int i = 0; i < CONV_W - 1; ++i)
          st[i] = a.conv_state[sbase + a.conv_width - (CONV_W - 1) + i];
        // The two entries the update keeps, read before anything is written:
        // stock keeps conv_width - nq old values starting at `acc`.
        const int nq_tok = t1 - t0;
        const int keep = a.conv_width - nq_tok;
        float kept[CONV_W - 1];
#pragma unroll
        for (int i = 0; i < CONV_W - 1; ++i)
          kept[i] = (i < keep && acc + i < a.conv_width)
                        ? a.conv_state[sbase + acc + i]
                        : 0.0f;
        auto hist = [&](int pos) -> float {
          return (pos >= 0)
                     ? __bfloat162float(
                           a.qkv[(size_t)(t0 + pos) * KDA_INPROJ_N + ch])
                     : st[pos + (CONV_W - 1)];
        };
        for (int j = 0; j < t1 - t0; ++j) {
          float v = 0.0f;
#pragma unroll
          for (int i = 0; i < CONV_W; ++i)
            v += w[i] * hist(j - (CONV_W - 1) + i);
          a.convq[(size_t)(t0 + j) * KDA_QKV + ch] =
              __float2bfloat16(v / (1.0f + expf(-v)));  // silu
        }
        // Write the WHOLE window, matching causal_conv1d_update:
        //   [init[acc] .. init[acc + keep - 1], x[0] .. x[nq - 1]]
        // The old code wrote only CONV_W - 1 slots at the front, which is
        // both the wrong count and the wrong place once conv_width > 3.
        for (int i = 0; i < a.conv_width; ++i)
          a.conv_state[sbase + i] =
              (i < keep) ? kept[i] : hist(i - keep);
      }
    }
  }
  mk_grid_barrier(a.barrier_ctr);

  {  // phase 3: fine-grained gated delta rule. One block per head, two
    // threads per v-column (k split in halves).
    //
    // Deliberately 16 of 48 blocks: the phase is DRAM-bound on the state
    // (2 MB read+write per layer, the census's ~68 MB/step), and 16 blocks
    // x 256 threads with the S rows register-resident issue enough
    // loads to saturate BW; the FLOP tail is ~0.4 us. Splitting heads
    // across 32 blocks adds no bandwidth -- measured, not assumed, before
    // this is "fixed".
    __shared__ float sh[4 * KDA_D];  // y, q, k, err
    __shared__ float sred[2][KDA_D];
    float* y_s = sh;
    float* q_s = sh + KDA_D;
    float* k_s = sh + 2 * KDA_D;
    float* err_s = sh + 3 * KDA_D;

    const int head = blockIdx.x;
    if (head < KDA_H) {
      const int v = threadIdx.x & (KDA_D - 1);  // 0..127
      const int khalf = threadIdx.x >> 7;       // 0..1
      const int k0 = khalf * (KDA_D / 2);

      for (int r = 0; r < a.n_spec; ++r) {
        const int t0 = a.cu_seqlens[r], t1 = a.cu_seqlens[r + 1];
        const int slot = a.state_idx[r * a.mql + 0];
        const int acc = a.n_accepted[r];
        float* Sbase =
            a.rec_state + (((size_t)slot * KDA_H + head) * KDA_D * KDA_D);
        // Element (v, k) lives at v * KDA_D + k, matching the stock
        // writer: fused_recurrent.py stores b_h as
        //   p_ht + o_v[:, None] * K + o_k[None, :]
        // This kernel had k * KDA_D + v, i.e. the TRANSPOSE. Internally the
        // recurrence is transpose-equivariant so it stayed self-consistent,
        // but the buffer it hands back to (and takes from) the stock path is
        // shared -- measured: as-is 1.005, transposed 6.6e-06 against the
        // stock final state at acc == T.
        float S[KDA_D / 2];
#pragma unroll 8
        for (int kk = 0; kk < KDA_D / 2; ++kk)
          S[kk] = Sbase[(size_t)v * KDA_D + (k0 + kk)];

        for (int j = 0; j < t1 - t0; ++j) {
          const int t = t0 + j;
          if (threadIdx.x < KDA_D) {
            const int d = threadIdx.x;
            // The state multiplier is exp(gate), not the gate. Stock
            // fused_recurrent.py:129-130 computes
            //   b_gk = LOWER_BOUND / (1 + exp(-(a_log_amp * (g + bias))))
            //   b_h *= exp(b_gk)
            // The gate itself is in (lower_bound, 0) = (-5, 0), so using it
            // raw flips the state's sign every token and grows it by up to
            // 5x -- which is exactly the observed failure: rel_err 16 at
            // acc=3 and 1570 at acc=8, i.e. ~2.5x per extra token, with
            // rec_state the worst component.
            y_s[d] = expf(
                a.lower_bound *
                mk_sigmoid(
                    expf(a.a_log[head]) *
                    (__bfloat162float(a.g1[(size_t)t * KDA_OUT +
                                           head * KDA_D + d]) +
                     a.dt_bias[head * KDA_D + d])));
            q_s[d] = __bfloat162float(
                a.convq[(size_t)t * KDA_QKV + head * KDA_D + d]);
            k_s[d] = __bfloat162float(a.convq[(size_t)t * KDA_QKV +
                                              KDA_H * KDA_D + head * KDA_D +
                                              d]);
          }
          __syncthreads();
          if (threadIdx.x == 0) {  // l2 normalize q, k (fp32)
            float nq = 0.0f, nk = 0.0f;
#pragma unroll 8
            for (int d = 0; d < KDA_D; ++d) {
              nq += q_s[d] * q_s[d];
              nk += k_s[d] * k_s[d];
            }
            // Match the stock kernel exactly (fused_recurrent.py:137-140):
            //   b_q = b_q / sqrt(sum(b_q*b_q) + 1e-6)
            //   b_k = b_k / sqrt(sum(b_k*b_k) + 1e-6)
            //   b_q = b_q * scale
            // The trailing scale is applied to q ONLY, and kda.py:165
            // defaults it to k.shape[-1] ** -0.5 = KDA_D^-0.5. This kernel
            // had neither the 1e-6 epsilon nor the scale, so its readout ran
            // sqrt(KDA_D) = 11.3x hot. Because only q carries the scale, the
            // error term (which retrieves with k) was unaffected -- which is
            // why rec_state matched while attn/core/out did not.
            constexpr float kda_qk_scale =
                0.088388347648318447f;  // KDA_D ** -0.5, KDA_D = 128
            nq = rsqrtf(nq + 1e-6f) * kda_qk_scale;
            nk = rsqrtf(nk + 1e-6f);
#pragma unroll 8
            for (int d = 0; d < KDA_D; ++d) {
              q_s[d] *= nq;
              k_s[d] *= nk;
            }
          }
          __syncthreads();

          const float beta = mk_sigmoid(__bfloat162float(
              a.qkv[(size_t)t * KDA_INPROJ_N + KDA_QKV + head]));
          // retrieval operand: q (variants 0/2) or k (variant 1)
          const float* r_s = (a.delta_variant == 1) ? k_s : q_s;
          float part = 0.0f;
#pragma unroll 8
          for (int kk = 0; kk < KDA_D / 2; ++kk) {
            S[kk] *= y_s[k0 + kk];
            part += r_s[k0 + kk] * S[kk];
          }
          sred[khalf][v] = part;
          __syncthreads();
          if (khalf == 0)
            err_s[v] = __bfloat162float(a.convq[(size_t)t * KDA_QKV +
                                                2 * KDA_H * KDA_D +
                                                head * KDA_D + v]) -
                       (sred[0][v] + sred[1][v]);
          __syncthreads();
          if (khalf == 0 && a.delta_variant == 2) {
            // pre-update readout: the output is the retrieval itself
            a.attn[(size_t)t * KDA_OUT + head * KDA_D + v] =
                __float2bfloat16(sred[0][v] + sred[1][v]);
          }
          __syncthreads();  // sred is rewritten below (variant-2 read done)

          float part2 = 0.0f;
#pragma unroll 8
          for (int kk = 0; kk < KDA_D / 2; ++kk) {
            S[kk] += beta * k_s[k0 + kk] * err_s[v];
            part2 += q_s[k0 + kk] * S[kk];
          }
          sred[khalf][v] = part2;
          __syncthreads();
          if (khalf == 0 && a.delta_variant != 2)
            a.attn[(size_t)t * KDA_OUT + head * KDA_D + v] =
                __float2bfloat16(sred[0][v] + sred[1][v]);
          __syncthreads();

          if (j == acc - 1) {  // accepted boundary: write the state back
#pragma unroll 8
            for (int kk = 0; kk < KDA_D / 2; ++kk)
              Sbase[(size_t)v * KDA_D + (k0 + kk)] = S[kk];
          }
        }
      }
    }
  }
  mk_grid_barrier(a.barrier_ctr);

  {  // phase 4: gated RMSNorm -- rmsnorm(attn) * sigmoid(g2), in place
    __shared__ float wred[MK_WARPS];
    __shared__ float inv;
    const int pairs = a.num_tokens * KDA_H;
    for (int i = blockIdx.x; i < pairs; i += MK_GRID) {
      const int t = i / KDA_H, h = i % KDA_H;
      __nv_bfloat16* src = a.attn + (size_t)t * KDA_OUT + h * KDA_D;
      float sq = 0.0f;
      for (int d = threadIdx.x; d < KDA_D; d += MK_THREADS) {
        const float v = __bfloat162float(src[d]);
        sq += v * v;
      }
      for (int off = 16; off; off >>= 1) sq += __shfl_xor_sync(~0u, sq, off);
      if ((threadIdx.x & 31) == 0) wred[threadIdx.x >> 5] = sq;
      __syncthreads();
      if (threadIdx.x == 0) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < MK_WARPS; ++w) v += wred[w];
        inv = rsqrtf(v / (float)KDA_D + a.onorm_eps);
      }
      __syncthreads();
      for (int d = threadIdx.x; d < KDA_D; d += MK_THREADS)
        src[d] = __float2bfloat16(
            __bfloat162float(src[d]) * inv *
            __bfloat162float(a.onorm_w[d]) *
            mk_sigmoid(__bfloat162float(
                a.g2[(size_t)t * KDA_OUT + h * KDA_D + d])));
      __syncthreads();
    }
  }
  mk_grid_barrier(a.barrier_ctr);

  {  // phase 5: o_proj GEMM
    MKGemmCtx c;
    c.x = a.attn;
    c.wq = a.o_wq;
    c.ws = a.o_ws;
    c.out = a.out;
    c.m = a.num_tokens;
    c.n = HIDDEN;
    c.k = KDA_OUT_PAD;
    c.n_orig = HIDDEN;
    mk_gemm_phase(c, smem);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// host entry points
// ---------------------------------------------------------------------------
namespace {

bool g_attrs_set = false;

void set_kernel_attrs() {
  if (g_attrs_set) return;
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM_SMEM));
  {
    int per_sm = 0, sms = 0;
    MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, mk_gemm_kernel, MK_THREADS, GEMM_SMEM));
    MK_CHECK_CUDA(cudaDeviceGetAttribute(
        &sms, cudaDevAttrMultiProcessorCount, 0));
    TORCH_CHECK(per_sm * sms >= MK_GRID, "gemm grid ", MK_GRID,
                " not resident: ", per_sm, " blocks/SM x ", sms, " SMs");
  }
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM_SMEM));
  g_attrs_set = true;
}

}  // namespace

// Geometry probe -- the driver refuses to arm unless the device is exactly
// the part this kernel was drawn for (mirrors the osar init gate: cc 12.1
// and exactly 48 SMs).
std::vector<int64_t> mk_probe_device() {
  int dev = 0;
  MK_CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  MK_CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  return {(int64_t)prop.major, (int64_t)prop.minor,
          (int64_t)prop.multiProcessorCount,
          (int64_t)prop.sharedMemPerBlockOptin};
}

void mk_run_gemm(torch::Tensor x, torch::Tensor wq, torch::Tensor ws,
                 torch::Tensor out, int64_t n_orig) {
  set_kernel_attrs();
  MKGemmCtx c;
  c.x = (const __nv_bfloat16*)x.data_ptr();
  c.wq = (const uint8_t*)wq.data_ptr();
  c.ws = (const float*)ws.data_ptr();
  c.out = (__nv_bfloat16*)out.data_ptr();
  c.m = (int)x.size(0);
  c.n = (int)wq.size(0);
  c.k = (int)x.size(1);
  c.n_orig = (int)n_orig;
  TORCH_CHECK(c.k % KSTEP == 0 && c.k <= KBLK_MAX * KSTEP, "k out of contract");
  TORCH_CHECK(c.n % 128 == 0, "wq rows must be 128-padded");
  TORCH_CHECK(c.m <= 32, "m out of contract");
  auto stream = c10::cuda::getCurrentCUDAStream();
  mk_gemm_kernel<<<MK_GRID, MK_THREADS, GEMM_SMEM, stream>>>(c);
}

void mk_run_gemm_w4(torch::Tensor x, torch::Tensor wq4, torch::Tensor ws4,
                    torch::Tensor out, int64_t n_orig) {
  set_kernel_attrs();
  MKGemmCtx c{};
  c.x = (const __nv_bfloat16*)x.data_ptr();
  c.wq4 = (const uint8_t*)wq4.data_ptr();
  c.ws4 = (const int8_t*)ws4.data_ptr();
  c.out = (__nv_bfloat16*)out.data_ptr();
  c.m = (int)x.size(0);
  c.n = (int)wq4.size(0);      // wq4 rows are 128-padded like the fp8 pack
  c.k = (int)x.size(1);        // k is the ACTIVATION width; wq4 is [n, k/2]
  c.n_orig = (int)n_orig;
  TORCH_CHECK(c.k % KSTEP == 0 && c.k <= KBLK_MAX * KSTEP, "k out of contract");
  TORCH_CHECK(c.n % 128 == 0, "wq4 rows must be 128-padded");
  TORCH_CHECK(c.m <= 32, "m out of contract");
  auto stream = c10::cuda::getCurrentCUDAStream();
  mk_gemm_kernel<<<MK_GRID, MK_THREADS, GEMM_SMEM, stream>>>(c);
}

// ptrs: x, res_in, pm_in, cm_in, fn, hc_scale, hc_base, norm_w, res_out,
//       pm_out, cm_out, layer_in, yp, rp, sq, pmix, ol_stash, barrier
// ints: num_tokens, sinkhorn_repeat
// scalars: rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps
void mk_run_mhc(std::vector<int64_t> ptrs, std::vector<double> scalars,
                std::vector<int64_t> ints) {
  set_kernel_attrs();
  // Ahead of the unpack, not after it: this used to sit below 19 ptrs[]
  // reads, so a short vector was already out of bounds before it fired.
  TORCH_CHECK(ptrs.size() == 18 && ints.size() == 2 && scalars.size() == 5,
              "run_mhc arg contract");
  MKMhcArgs a{};
  a.x_in = (const __nv_bfloat16*)ptrs[0];
  a.residual_in = (const __nv_bfloat16*)ptrs[1];
  a.post_mix_in = (const float*)ptrs[2];
  a.comb_mix_in = (const float*)ptrs[3];
  a.fn = (const float*)ptrs[4];
  a.hc_scale = (const float*)ptrs[5];
  a.hc_base = (const float*)ptrs[6];
  a.norm_weight = (const __nv_bfloat16*)ptrs[7];
  a.residual_out = (__nv_bfloat16*)ptrs[8];
  a.post_mix_out = (float*)ptrs[9];
  a.comb_mix_out = (float*)ptrs[10];
  a.layer_input = (__nv_bfloat16*)ptrs[11];
  a.yp = (float*)ptrs[12];
  a.rp = (float*)ptrs[13];
  a.sq = (float*)ptrs[14];
  a.pmix = (float*)ptrs[15];
  a.ol_stash = (__nv_bfloat16*)ptrs[16];
  a.barrier_ctr = (unsigned long long*)ptrs[17];
  a.num_tokens = (int)ints[0];
  a.sinkhorn_repeat = (int)ints[1];
  a.rms_eps = (float)scalars[0];
  a.pre_eps = (float)scalars[1];
  a.sinkhorn_eps = (float)scalars[2];
  a.post_mult = (float)scalars[3];
  a.norm_eps = (float)scalars[4];

  auto stream = c10::cuda::getCurrentCUDAStream();
  // A persistent grid must be fully resident or the grid barrier deadlocks.
  // Ask the device rather than assume, and clamp to what it answers. Cached
  // because the barrier's ticket arithmetic also needs the grid to be the
  // SAME on every launch.
  static int mhc_grid = 0;
  if (mhc_grid == 0) {
    int per_sm = 0, sms = 0;
    MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, mk_mhc_kernel, MK_THREADS, 0));
    MK_CHECK_CUDA(cudaDeviceGetAttribute(
        &sms, cudaDevAttrMultiProcessorCount, 0));
    mhc_grid = per_sm * sms;
    if (mhc_grid > MK_MHC_GRID_CAP) mhc_grid = MK_MHC_GRID_CAP;
    TORCH_CHECK(mhc_grid > 0, "mhc has no resident blocks");
  }
  a.grid = mhc_grid;
  mk_mhc_kernel<<<mhc_grid, MK_THREADS, 0, stream>>>(a);
}

// ptrs: x, in_wq, in_ws, f_b_w, g_b_w, conv_w, conv_state, rec_state,
//       a_log, dt_bias, cu, state_idx, n_acc, o_wq, o_ws, out,
//       qkv, g1, g2, convq, attn, barrier, onorm_w
// ints: num_tokens, n_spec, mql, delta_variant, conv_width
// scalars: lower_bound, onorm_eps
void mk_run_kda(std::vector<int64_t> ptrs, std::vector<double> scalars,
                std::vector<int64_t> ints) {
  set_kernel_attrs();
  MKKdaArgs a{};
  a.x = (const __nv_bfloat16*)ptrs[0];
  a.in_wq = (const uint8_t*)ptrs[1];
  a.in_ws = (const float*)ptrs[2];
  a.f_b_w = (const __nv_bfloat16*)ptrs[3];
  a.g_b_w = (const __nv_bfloat16*)ptrs[4];
  a.conv_w = (const float*)ptrs[5];
  a.conv_state = (float*)ptrs[6];
  a.rec_state = (float*)ptrs[7];
  a.a_log = (const float*)ptrs[8];
  a.dt_bias = (const float*)ptrs[9];
  a.cu_seqlens = (const int*)ptrs[10];
  a.state_idx = (const int*)ptrs[11];
  a.n_accepted = (const int*)ptrs[12];
  a.o_wq = (const uint8_t*)ptrs[13];
  a.o_ws = (const float*)ptrs[14];
  a.out = (__nv_bfloat16*)ptrs[15];
  a.qkv = (__nv_bfloat16*)ptrs[16];
  a.g1 = (__nv_bfloat16*)ptrs[17];
  a.g2 = (__nv_bfloat16*)ptrs[18];
  a.convq = (__nv_bfloat16*)ptrs[19];
  a.attn = (__nv_bfloat16*)ptrs[20];
  a.barrier_ctr = (unsigned long long*)ptrs[21];
  a.onorm_w = (const __nv_bfloat16*)ptrs[22];
  TORCH_CHECK(ptrs.size() == 23 && ints.size() == 5 && scalars.size() == 2,
              "run_kda arg contract");
  a.num_tokens = (int)ints[0];
  a.n_spec = (int)ints[1];
  a.mql = (int)ints[2];
  a.delta_variant = (int)ints[3];
  a.conv_width = (int)ints[4];
  a.lower_bound = (float)scalars[0];
  a.onorm_eps = (float)scalars[1];
  auto stream = c10::cuda::getCurrentCUDAStream();
  mk_kda_kernel<<<MK_GRID, MK_THREADS, GEMM_SMEM, stream>>>(a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_device", &mk_probe_device, "device geometry probe");
  m.def("run_gemm", &mk_run_gemm, "MK_SEG_GEMM");
  m.def("run_gemm_w4", &mk_run_gemm_w4, "MK_SEG_GEMM (W4 pack)");
  m.def("run_mhc", &mk_run_mhc, "MK_SEG_MHC");
  m.def("run_kda", &mk_run_kda, "MK_SEG_KDA");
}
