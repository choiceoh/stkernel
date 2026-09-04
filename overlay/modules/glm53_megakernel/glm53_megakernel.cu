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
//   MK_SEG_GEMM -- W4A8 skinny GEMM, M <= 32: per-token-group fp8 quant
//                  fused into a mma.sync m16n8k32 e4m3 GEMM whose weights
//                  stream as e2m1 nibbles + per-16-group pow2 exponents and
//                  expand to exact e4m3 bytes in smem, replacing the
//                  per_token_group_quant kernel + deepgemm launch pair.
//                  2 launches -> 1, 0.56x the weight bytes. (The fp8 W8
//                  arm was removed: W4 beat the stock pair on every decode
//                  shape and one lane is one lane to configure.)
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
#include <cstdlib>
#include <vector>
#include <type_traits>

namespace {

// Three tuning constants come in through -D so the probe can sweep them
// without editing this file; the defaults here are the shipped values.
#ifndef MK_GRID_DEF
#define MK_GRID_DEF 96
#endif
#ifndef MK_MHC_GRID_DEF
#define MK_MHC_GRID_DEF 144
#endif
// W raw-record pipeline depth (VLLM_GLM53_MK_NBUF; 3..5, swept by the probe)
#ifndef MK_NBUF_DEF
#define MK_NBUF_DEF 3
#endif
// Ceiling for the gemm and kda persistent grids -- each launch takes the
// smaller of it and what the device reports resident, exactly as mhc
// does. The 69,632 B block only fits once in the SM's 102,400 B, so this
// resolves to 48.
constexpr int MK_GRID_CAP = MK_GRID_DEF;
// mhc only: it takes no dynamic smem, so its occupancy is bounded by
// registers (72 x 256 = 18,432 of 65,536 per SM -> 3 blocks). At the
// gemm/kda grid
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
constexpr int KDA_SPEC = 7;                        // num_speculative_tokens
constexpr int KDA_NQ_MAX = KDA_SPEC + 1;           // query tokens per request
constexpr int KDA_OUT_PAD = KDA_OUT;               // 2048 = 16 x 128

constexpr int KSTEP = 128;               // one scale block of K
constexpr int KBLK_MAX = 32;             // max k blocks (K <= 4096)
// fp8 A tiles: dense 128 B rows with a 16 B-chunk XOR swizzle (mk_swz) keyed
// by the row within the 16-row tile. The old 132 B pitch (33 words) put the
// 8 rows of a fragment load on banks g+q, i.e. 4-way conflicts on 8 of the
// 12 smem loads per k-step: ~1,150 wavefronts per k-block per SM, ~25 us
// at n=6416. Hidden under the W8 stream, exposed in the W4 loop, where
// removing mma_fold measured -15..-27 us per launch. Rows g and g+8 share
// a key, so the 8 lanes of one fragment load land on 8 bank groups.
constexpr int SMEM_A_PITCH = KSTEP;      // 128, dense + swizzled
// fp8 W tile rows are DENSE (128 B). The pack is pre-swizzled: chunk c of
// row r sits at chunk c ^ (r & 7) (build_mk_weight), so the tile copy is a
// straight 16 KB memcpy and the mma's fragment loads read through mk_swz.
// The old 144 B padding kept those loads conflict-free but put every
// cp.async destination row across a 128 B boundary: in the clean regime a
// pure 24 MB stream is 130 us at pitch 144 and 110 us at pitch 128 (srv2,
// second launch of a pair) -- 16%, the whole gap to deepgemm's stream.
constexpr int SMEM_W_PITCH = KSTEP;      // 128, dense + pre-swizzled
constexpr int SMEM_W_ROWS = 128;         // W rows staged per k-block

// W4 raw staging: one (tile, k-block) record is 128 rows x 64 B of e2m1
// pairs plus 128 x 8 B of group exponents. In smem the nibble rows sit on
// an 80 B pitch (a warp's 32 rows of uint4 reads then hit distinct banks)
// and the exponents follow as a flat 1 KB. Three stages, two in flight;
// the expanded e4m3 tiles are the two swb buffers.
// Raw rows on a 64 B pitch: the two uint4 reads per thread per k-block
// conflict 16-way, ~100 cycles, cheaper than the 80 B padding's smem.
// Three stages (two in flight). Deeper measured WORSE at n=1024 m=8, both
// alone and back to back (srv2, 2 reps: 3/4/5 stages = 45/49/52 us single,
// 39/41.5/44 us paired): the raw stream is not the W4 arm's limiter once
// the expansion is register-only, and more records in flight only queue.
constexpr int W4_RAW_PITCH = 64;
constexpr int W4_RAW_NIB = SMEM_W_ROWS * W4_RAW_PITCH;   // 8192
constexpr int W4_RAW_BYTES = W4_RAW_NIB + SMEM_W_ROWS * 8;  // 9216
constexpr int W4_RAW_NBUF = MK_NBUF_DEF;  // 3..5, swept by the probe
static_assert(W4_RAW_NBUF >= 3 && W4_RAW_NBUF <= 5, "W4 raw stages");
constexpr int W4_EXP_NBUF = 2;  // expanded e4m3 tiles, ping-pong
// One kernel, one budget (the fp8 W8 arm and its second budget were
// removed once the W4 arm beat the stock pair on every decode shape: the
// serving lane is W4 or stock, nothing in between to configure).
constexpr int MK_SMEM_ALIGN = 1024;  // runtime alignment of the dynamic base
constexpr int GEMM_SMEM = MK_SMEM_ALIGN + 2 * 16 * SMEM_A_PITCH +
                          W4_EXP_NBUF * SMEM_W_ROWS * SMEM_W_PITCH +
                          KBLK_MAX * KBLK_MAX * 4 +
                          W4_RAW_NBUF * W4_RAW_BYTES;  // 69,632 at 3
static_assert(GEMM_SMEM <= 101376, "over the sm_121 opt-in smem");
static_assert(KDA_D * (KDA_D + 1) * 4 <= GEMM_SMEM,
              "the kda per-position state store stages a padded D x D tile "
              "in the dynamic smem the gemm phases own");
static_assert(W4_RAW_NBUF - 1 <= 4, "mk_cp_wait_upto dispatches up to 4");

#define MK_CHECK_CUDA(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    TORCH_CHECK(_e == cudaSuccess, "megakernel cuda error: ",                \
                cudaGetErrorString(_e));                                     \
  } while (0)

// ---------------------------------------------------------------------------
// graph-safe grid barrier. Never-reset monotonic ticket counter: every
// (launch, phase) adds exactly `grid` arrivals; a block waits for the next
// multiple of `grid` at or after its own ticket, so CUDA-graph replay with
// baked workspace pointers is exact. Device-scope fences make the phases'
// global writes visible across blocks (the osar barrier lesson).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mk_grid_barrier(unsigned long long* ctr,
                                               int grid) {
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
// byte offset within a 128 B tile row -> its swizzled offset (A tiles)
__device__ __forceinline__ int mk_swz(int row, int off) {
  return ((((off >> 4) ^ (row & 7)) << 4) | (off & 15));
}

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
// wait_group takes an immediate; this dispatches a runtime "groups that may
// stay in flight" count (the raw pipeline keeps at most W4_RAW_NBUF - 1).
__device__ __forceinline__ void mk_cp_wait_upto(int n) {
  switch (n) {
    case 0: mk_cp_wait<0>(); break;
    case 1: mk_cp_wait<1>(); break;
    case 2: mk_cp_wait<2>(); break;
    case 3: mk_cp_wait<3>(); break;
    default: mk_cp_wait<4>(); break;
  }
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
  __nv_bfloat16* out;      // [m, n_orig]
  int m, n, k, n_orig;
  int grid;  // resident blocks; see MK_GRID_CAP
  int ksr;   // k-split of the leftover tiles, chosen on the host (mk_choose_ksr)
  // W4 weights: e2m1 nibbles [n, k/2] + per-16-group e4m3 scale bytes
  // [n, k/16] (int8 d = (e << 3) + k, e clamped to [-5, 5] at build). The kernel expands each
  // nibble to an EXACT e4m3 byte (1-bit mantissas always fit; the 2^s
  // product is an exponent-field add) and then runs the same e4m3 mma
  // pipeline -- the DRAM bytes halve and the arithmetic is unchanged.
  // Tile-major: [n/128][k/128][128][64] nibble bytes and
  // [n/128][k/128][128][8] exponents, so one (tile, k-block) record is a
  // contiguous 8 KB + 1 KB run for cp.async (see stage_raw4).
  const uint8_t* wq4;
  const int8_t* ws4;
  // true: the A tiles (g_mk_aq) and their scales (g_mk_axs) were written
  // by the caller's previous phase and published by ITS grid barrier, so
  // the prologue skips the x load / quant / publishing barrier (the caller
  // also resets g_mk_unit_next before that barrier). kda's o_proj: p4
  // emits the normalized attn straight as fp8 k-groups (one head = one
  // 128-group), which retired ~8 us of prologue from the phase.
  bool a_ready = false;
  // Per-tensor pow2 the PACK was normalised by, undone here. The expansion
  // is a byte add, so the group scale can only span the e4m3 exponent
  // field: d in [-5, 5] around a magnitude of 1. Real weights are nowhere
  // near 1 -- GLM-5.3's dense projections need group exponents around 2^-7
  // (median; p1 2^-16) -- so before this every production group clamped at
  // the floor and the pack quantised ~3x worse than the format allows
  // (measured 0.225 rel vs 0.082). The build picks the shift, folds 2^shift
  // into the weights, and passes 2^-shift here; it costs one multiply on
  // the activation scales in the prologue.
  float wgs = 1.0f;
};

// e4m3 encodings of the e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
// Expansion: byte = LUT[mag] + d | sign<<7, where d = (e << 3) + k is the
// group's stored e4m3 SCALE byte -- (1 + k/8) * 2^e. Adding the scale's
// 3-bit mantissa field to an e4m3 byte IS the multiply: the e2m1
// magnitudes carry a 1-bit mantissa, so the product needs at most one
// carry out of the mantissa field and that carry lands on the exponent
// field exactly where it belongs. The three codes whose magnitude mantissa
// is 1.5 (3, 5, 7) need +1 when 1 <= k <= 5, and that correction does not
// depend on k -- hence a second table rather than a select over eight.
// Exhaustively checked against e4m3(magnitude * scale) over every
// (code, k, e in [-5, 5]): 504/504, bytes span 0x08..0x6b, so never a
// denormal, never the sign bit, never the NaN encoding 0x7F.
// The table as one 64-bit immediate, byte c at bits [8c, 8c+8): the expansion
// indexes it with a funnel shift. It was a __device__ __constant__ uint8_t[8]
// first, and that cost more than it saved -- a __constant__ load serialises
// over the distinct addresses in a warp, up to 8 here, 64 lookups per thread
// per k-block, which made the W4 expansion outweigh the DRAM time it was
// meant to hide. The array and a uint8_t accessor around this immediate both
// outlived that change with no callers; they are gone.
constexpr unsigned long long MK_E2M1_LUT64 = 0x4C484440'3C383000ULL;
// same table with +1 on codes 3, 5, 7 (magnitude mantissa 1.5)
constexpr unsigned long long MK_E2M1_LUT64_B = 0x4D484540'3D383000ULL;

// Remainder split-K state. Every block owns one 128-column tile across the
// whole k range, so a tile count that is not a multiple of the grid pays a
// WHOLE extra round for its leftovers: phase 0's 51 tiles take 2 rounds and
// the second carries 3 tiles while 45 blocks idle -- 2 tile-times for 1.06
// tiles of work, and phase 0 is ~43% of the KDA segment.
//
// The leftovers get their k split instead. rem < grid by construction, so
// the accumulator is at most 32 x (47*128) floats; it lives here rather than
// in the ctx so neither host entry point nor the pybind signature changes.
// ksr * rem <= grid no longer holds (see the ksr choice below), so the
// accumulator is sized for the unit cap, not for one round.
constexpr int MK_SPLIT_UNITS_MAX = 2 * MK_GRID_CAP;               // 96
constexpr int MK_SPLIT_MAXCOL = (MK_GRID_CAP - 1) * 128;          // 6016
constexpr int MK_SPLIT_ELEMS = 32 * 128 * MK_SPLIT_UNITS_MAX; // 1.5 MB
__device__ unsigned long long g_mk_gemm_bar = 0ULL;
// kda inlines mk_gemm_phase twice per launch on ITS grid, which need
// not equal the standalone gemm grid. Two grids on one ticket counter
// misalign it and the barrier releases early -- so, two counters.
__device__ unsigned long long g_mk_kda_bar = 0ULL;
__device__ __align__(16) float g_mk_gemm_partial[MK_SPLIT_ELEMS];
// Split-K fold without a grid barrier: every k slice of a leftover tile
// bumps its tile's counter after publishing its partial, and the slice
// that finds the count complete folds the tile (fixed slice order, so the
// sum is bitwise the same whichever block is last) and resets the counter
// for the next launch. Indexed by leftover tile, rem < grid <= MK_GRID_CAP.
// The stamps priced the barrier + cooperative fold at 5-11 us per launch,
// most of it blocks that had finished waiting for the one that had not.
__device__ unsigned int g_mk_tile_arrive[MK_GRID_CAP];
// Dynamic unit hand-out. Static striding (u += grid) gave every block the
// same count of units, but the stamps showed a 25% spread in the loop
// (n=4096: 86..110 us) -- DRAM arbitration is not fair and the slow
// blocks set the wall time. Each block keeps its first unit static (the
// hoisted W fill needs it before the prologue) and then takes the next
// unit from this counter, so the blocks that get served faster do more.
// Block 0 re-arms it ahead of the A-quant barrier, which orders the reset
// before any block's first take.
__device__ unsigned int g_mk_unit_next = 0u;
// A, quantized ONCE per launch. Every n-tile walks all of k, so quantizing
// inside the tile loop redid the same work nblk times -- 51x at n=6416 --
// and read bf16, twice the bytes the mma consumes. Measured ceiling for
// removing it: -10% at n=6416/4096, -22% at n=2048, -14% at n=1024.
// [KBLK_MAX][32 rows][KSTEP] e4m3 + [32 rows][KBLK_MAX] fp32 scales.
__device__ uint8_t g_mk_aq[(size_t)KBLK_MAX * 32 * KSTEP];  // 128 KB
__device__ float g_mk_axs[32 * KBLK_MAX];                   // 4 KB
// Optional phase timestamps, probe builds only (-DMK_PHASE_TS=1): thread 0
// of every block stamps %globaltimer at the phase boundaries marked MK_TS()
// below into g_mk_ts[block][slot]; the host reads (and clears) them with
// read_ts(). Compiled out otherwise -- the shipped kernel is unchanged.
#ifdef MK_PHASE_TS
__device__ unsigned long long g_mk_ts[MK_GRID_CAP * 8];
__device__ unsigned long long g_mk_mhc_ts[MK_MHC_GRID_CAP * 8];  // mhc phases
// kda: [block][16] -- 0 entry, then (phase k end, barrier k exit) pairs
__device__ unsigned long long g_mk_kda_ts[MK_GRID_CAP * 16];
__device__ __forceinline__ unsigned long long mk_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define MK_TS(slot)                                                          \
  do {                                                                       \
    if (threadIdx.x == 0) g_mk_ts[blockIdx.x * 8 + (slot)] = mk_globaltimer(); \
  } while (0)
// accumulated durations (ns), e.g. time spent inside the W pipeline wait
#define MK_KDA_TS(slot)                                                     \
  do {                                                                       \
    if (threadIdx.x == 0)                                                    \
      g_mk_kda_ts[blockIdx.x * 16 + (slot)] = mk_globaltimer();             \
  } while (0)
// mhc tail probes ride the gemm stamp array (idle during mhc), slots 1..7
#define MK_MHC_PROBE(slot)                                                   \
  do {                                                                       \
    if (threadIdx.x == 0) g_mk_ts[blockIdx.x * 8 + (slot)] = mk_globaltimer(); \
  } while (0)
#define MK_TS_ACC_BEGIN(v) const unsigned long long v = mk_globaltimer()
#define MK_TS_ACC_END(acc, v) acc += mk_globaltimer() - (v)
#define MK_TS_STORE(slot, acc)                                               \
  do {                                                                       \
    if (threadIdx.x == 0) g_mk_ts[blockIdx.x * 8 + (slot)] = (acc);         \
  } while (0)
#define MK_MHC_TS(slot)                                                      \
  do {                                                                       \
    if (threadIdx.x == 0)                                                    \
      g_mk_mhc_ts[blockIdx.x * 8 + (slot)] = mk_globaltimer();               \
  } while (0)
#else
#define MK_TS(slot) \
  do {              \
  } while (0)
#define MK_KDA_TS(slot) \
  do {                  \
  } while (0)
#define MK_MHC_PROBE(slot) \
  do {                     \
  } while (0)
#define MK_TS_ACC_BEGIN(v) \
  do {                     \
  } while (0)
#define MK_TS_ACC_END(acc, v) \
  do {                        \
  } while (0)
#define MK_TS_STORE(slot, acc) \
  do {                         \
  } while (0)
#define MK_MHC_TS(slot) \
  do {                  \
  } while (0)
#endif

__device__ void mk_gemm_phase(const MKGemmCtx& c, uint8_t* smem,
                              unsigned long long* bar) {
  const int nblk = c.n / 128;
  const int kblk = c.k / KSTEP;
  const int mtiles = (c.m + 15) / 16;
  MK_TS(0);  // entry
  // Leftover tiles of the last (partial) round, and how many ways to split
  // their k so those blocks are not idle. ksr == 1 leaves the original
  // single-pass path byte for byte -- which is what n/128 == 32 (o_proj)
  // and any exact multiple of c.grid get.
  const int rem = nblk % c.grid;
  const int full = nblk - rem;
  // Chosen on the host (mk_choose_ksr): the cost-model search that used
  // to run here cost every block a 31-step loop with an integer division
  // per step, ~3 us at kernel start on all 48 blocks for every full == 0
  // shape, before the first byte of W moved.
  const int ksr = c.ksr;
  const int pcols = rem * 128;
  // Guard on the accumulator directly rather than on a column count that
  // only bounds it when ksr * rem <= c.grid -- which no longer holds.
  const bool split = (ksr > 1) && (c.m <= 32) &&
                     (rem * 128 <= MK_SPLIT_MAXCOL) &&
                     ((size_t)c.m * pcols * ksr <= MK_SPLIT_ELEMS);
  // One slice per split, summed in a FIXED order below. An atomicAdd
  // accumulator is order-nondeterministic, so back-to-back launches of the
  // same call return bitwise-different results -- the probe's replay-
  // stability check catches exactly that, and this lane cannot afford a
  // kernel whose output depends on scheduling.
  const size_t pslice = (size_t)c.m * pcols;
  // No zero pass, and no barrier for one. Every element of the accumulator
  // is ASSIGNED (not accumulated into) by exactly one unit: the epilogue
  // below writes all rows r < c.m and all 128 columns of its tile, the rem
  // tiles cover pcols, and the ksr splits cover the slices. Zeroing first
  // cost a full pass over pslice * ksr floats plus a grid barrier -- 768 KB
  // against 4 MB of weights at n=1024, i.e. a fifth of the traffic, to
  // pre-set values that are all overwritten before anyone reads them.

  constexpr int NWB = W4_EXP_NBUF;  // expanded e4m3 tiles, ping-pong
  // The dynamic smem region starts right after this kernel's STATIC
  // __shared__ variables (s_last, s_unit below: 8 B -> base at +16), so
  // without this every 128 B tile row straddled a bank-line boundary --
  // the same penalty as the old padded pitch, which is why dense rows
  // measured no faster at first. Align to 1 KB (the constants carry 1 KB
  // of slack for it).
  {
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
    smem += (MK_SMEM_ALIGN - (sb & (MK_SMEM_ALIGN - 1))) & (MK_SMEM_ALIGN - 1);
  }
  uint8_t* saq = smem;  // [2][16][128] fp8 A tiles (single per kb)
  uint8_t* swb = saq + 2 * 16 * SMEM_A_PITCH;  // [NWB][128][128], swizzled
  float* sxs = (float*)(swb + NWB * SMEM_W_ROWS * SMEM_W_PITCH);  // [32][32]
  uint8_t* sraw = (uint8_t*)(sxs + KBLK_MAX * KBLK_MAX);  // W4 raw stages
  __shared__ int s_last;  // "this block completed a leftover tile"
  __shared__ int s_unit;  // next dynamically taken unit, broadcast

  const int units = split ? (full + rem * ksr) : nblk;
  // unit -> (n-tile, k range). Whole tiles first, then the leftover tiles'
  // k slices, ksr per tile.
  auto decode_unit = [&](int u, int& nt, int& kb0, int& kbn) {
    if (!split || u < full) {          // a whole tile, one block, all of k
      nt = u; kb0 = 0; kbn = kblk;
    } else {                           // a leftover tile's k slice
      const int t = (u - full) / ksr, sp = (u - full) % ksr;
      nt = full + t;
      kb0 = (kblk * sp) / ksr;
      kbn = (kblk * (sp + 1)) / ksr;
    }
  };
  // W4: stage one raw (tile, k-block) record -- 512 nibble chunks (two
  // per thread) onto the 80 B pitch, 64 exponent chunks after them. One
  // commit group per stage, which is what the wait counts below rely on.
  auto stage_raw4 = [&](int nt, int kb, int buf) {
    const uint8_t* nsrc =
        c.wq4 + ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * 64);
    const uint8_t* ssrc = (const uint8_t*)c.ws4 +
        ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * 8);
    uint8_t* d = sraw + buf * W4_RAW_BYTES;
    // thread 0 issues no copies: it runs the grid barrier, and the
    // barrier's __threadfence drains that thread's outstanding cp.async
    // (with the first record hoisted above the barrier, the fill's latency
    // moved into the barrier wait when it took part)
    for (int t = (int)threadIdx.x - 1; threadIdx.x != 0 && t < SMEM_W_ROWS * 4;
         t += MK_THREADS - 1)
      mk_cp_async16(d + (t >> 2) * W4_RAW_PITCH + (t & 3) * 16,
                    nsrc + (size_t)t * 16);
    if (threadIdx.x >= 1 && threadIdx.x <= SMEM_W_ROWS * 8 / 16) {
      const int t = (int)threadIdx.x - 1;
      mk_cp_async16(d + W4_RAW_NIB + t * 16, ssrc + (size_t)t * 16);
    }
    mk_cp_commit();
  };
  // W4: expand one landed raw record into an e4m3 tile buffer. Thread ->
  // (row, half): 64 elements = 32 B of nibbles + 4 scale bytes, written as
  // 64 B on the 144 B tile pitch. Exact: the table byte plus the scale byte
  // reproduces e4m3(magnitude * scale) for every code and scale.
  auto expand_w4 = [&](int rawbuf, int expbuf) {
    const int row4 = threadIdx.x & (SMEM_W_ROWS - 1);
    const int half4 = threadIdx.x >> 7;
    const uint8_t* rr = sraw + rawbuf * W4_RAW_BYTES;
    // Registers only. The byte-array form (uint8_t nb[32] / ob[64] filled
    // and drained through uint4 punning) lands in local memory, and its
    // ~150 byte-wide LDL/STL per k-block were the expansion's real cost --
    // ~6 us per k-block on the stamps, more than the record's DRAM time.
    const uint4 n0 = *(const uint4*)(rr + row4 * W4_RAW_PITCH + half4 * 32);
    const uint4 n1 =
        *(const uint4*)(rr + row4 * W4_RAW_PITCH + half4 * 32 + 16);
    const uint32_t sc4 =
        *(const uint32_t*)(rr + W4_RAW_NIB + row4 * 8 + half4 * 4);
    const uint32_t nw[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
    uint8_t* rowb = swb + expbuf * (SMEM_W_ROWS * SMEM_W_PITCH) +
                    row4 * SMEM_W_PITCH;  // chunk offsets go through mk_swz
    // Table lookup by byte permute: a raw word holds 8 nibbles = 8
    // elements in order, and __byte_perm picks output byte i from an
    // 8-byte table by selector nibble i (3 index bits; bit 3 must be
    // clear). The e2m1 magnitude has 8 codes, so the low 16 bits of the
    // word with the sign bits masked ARE the selector for elements 0..3
    // and the high 16 bits for 4..7 -- one prmt per four e4m3 bytes. The
    // group exponent is folded into the table once per 16 elements
    // (MK_E2M1_LUT64's two halves, bytes 1..7 + (s << 3); byte 0 stays
    // 0 so zero stays zero; the sum stays in [8, 124]). The sign is a second prmt over the same
    // nibbles' bit 3 (table {0x00, 0x80}). ~13 ops per raw word against
    // ~22 for the byte-lane arithmetic this replaces; the expansion was
    // 30 of the W4 loop's 71 us at n=6416 (compute-bound, floor ~61).
#pragma unroll
    for (int g4 = 0; g4 < 4; ++g4) {  // one 16-group: 2 raw words -> 4 out words
      const uint32_t eb =
          (uint32_t)((int8_t)((sc4 >> (8 * g4)) & 0xFFu)) & 0xFFu;
      // low three bits are the scale mantissa k for either sign of e, so
      // the table choice is one predicate on the stored byte
      const unsigned long long lutg =
          ((eb & 7u) - 1u) < 5u ? MK_E2M1_LUT64_B : MK_E2M1_LUT64;
      // per-lane adds: a negative scale byte is eb >= 0x80 and the byte sum
      // wraps (0x30 + 0xF8 = 0x28, the right e4m3 for 0.5 * 2^-1); a plain
      // 32-bit add carried that wrap into the next table byte (accuracy
      // gate FAIL on the first run of this form)
      const uint32_t l0 =  // codes 0..3: 0x00 0x30 0x38 0x3C, + d
          __vadd4((uint32_t)lutg, eb * 0x01010100u);
      const uint32_t l1 =  // codes 4..7: 0x40 0x44 0x48 0x4C
          __vadd4((uint32_t)(lutg >> 32), eb * 0x01010101u);
      uint32_t ow[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {   // raw word h of the group: elements 8h..8h+7
        const uint32_t w = nw[2 * g4 + h];
        const uint32_t m0 = __byte_perm(l0, l1, w & 0x7777u);
        const uint32_t m1 = __byte_perm(l0, l1, (w >> 16) & 0x7777u);
        const uint32_t g0 = __byte_perm(0x8000u, 0u, (w >> 3) & 0x1111u);
        const uint32_t g1 = __byte_perm(0x8000u, 0u, (w >> 19) & 0x1111u);
        ow[2 * h] = m0 | g0;
        ow[2 * h + 1] = m1 | g1;
      }
      *(uint4*)(rowb + mk_swz(row4, half4 * 64 + g4 * 16)) =
          make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
  };
  constexpr int RAW_DIST = W4_RAW_NBUF - 1;

  // ---- prologue. Order: the first unit's W fill (independent of the
  // previous kernel, so it goes out before the PDL wait and lands during
  // that kernel's tail), the PDL wait, then this block's A k-block into
  // registers and the amax reduce. Staging x through cp.async ahead of
  // the fill was tried: it made every block wait for the whole fill in
  // the prologue (wait_group cannot skip older groups), which is exactly
  // the exposure a plain launch should not pay.
  const int qw = threadIdx.x >> 5, ql = threadIdx.x & 31;
  constexpr int RPW = 32 / MK_WARPS;  // rows per warp at m = 32
  const int kbq = (int)blockIdx.x;
  int nt0 = 0, kb00 = 0, kbn0 = 0;
  const bool has_u0 = (int)blockIdx.x < units;
  if (has_u0) decode_unit((int)blockIdx.x, nt0, kb00, kbn0);
  bool hoisted = false;
  if (has_u0) {
    if (kb00 < kbn0) {
      stage_raw4(nt0, kb00, kb00 % W4_RAW_NBUF);
#pragma unroll
      for (int d = 1; d < RAW_DIST; ++d)
        if (kb00 + d < kbn0)
          stage_raw4(nt0, kb00 + d, (kb00 + d) % W4_RAW_NBUF);
      hoisted = true;
    }
  }

  // ---- PDL: launched programmatically after the previous kernel, this
  // grid has been running on the SMs that kernel freed, and its W fill
  // above went out during that kernel's tail. From here on it reads x (the
  // previous kernel's output) and touches the shared counters, so it waits
  // for that grid to complete and flush. A no-op for a plain launch.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // x -> registers, after the wait, one unconditional 8 B load per row
  // (rows past m read a clamped row and are never stored). With the W
  // fill already in flight -- or landed, under PDL -- nothing competes
  // with these four loads.
  float v[RPW][4], mx[RPW];
  if (!c.a_ready && kbq < kblk) {
    uint2 raw[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
      const int r = min(qw + i * MK_WARPS, c.m - 1);
      raw[i] = *(const uint2*)(c.x + (size_t)r * c.k + kbq * KSTEP + ql * 4);
    }
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
      const __nv_bfloat16* pv = (const __nv_bfloat16*)&raw[i];
#pragma unroll
      for (int q = 0; q < 4; ++q) v[i][q] = __bfloat162float(pv[q]);
      mx[i] = fmaxf(fmaxf(fabsf(v[i][0]), fabsf(v[i][1])),
                    fmaxf(fabsf(v[i][2]), fabsf(v[i][3])));
#pragma unroll
      for (int off = 16; off; off >>= 1)
        mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], off));
    }
  }
  MK_TS(7);  // x loaded and amax-reduced
  // ---- prologue, part 2: x landed -> amax, scale, convert, publish --
  // once for the WHOLE grid. The barrier that publishes it is the price;
  // the measurement in #209 says it is worth paying.
  auto quant_store = [&](int kb, const float (&vv)[RPW][4],
                         const float (&mm)[RPW]) {
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
      const int r = qw + i * MK_WARPS;
      if (r >= c.m) break;  // rows ascend with i
      const float sc = mk_pow2_scale(mm[i]);
      if (ql == 0) g_mk_axs[r * KBLK_MAX + kb] = sc;
      // sc is a power of two, so the reciprocal is exact and v * rsc is
      // bit-identical to v / sc -- four IEEE divides become one rcp.
      const float rsc = 1.0f / sc;
      uint32_t pack = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        pack |= (uint32_t)mk_f32_to_e4m3(vv[i][q] * rsc) << (8 * q);
      *(uint32_t*)(g_mk_aq + ((size_t)kb * 32 + r) * KSTEP + ql * 4) = pack;
    }
  };
  if (!c.a_ready && kbq < kblk) quant_store(kbq, v, mx);
  for (int kb = c.a_ready ? kblk : kbq + c.grid; kb < kblk;
       kb += c.grid) {  // grid < kblk only
    float v2[RPW][4], mx2[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
      const int r = qw + i * MK_WARPS;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        v2[i][q] = (r < c.m) ? __bfloat162float(
            c.x[(size_t)r * c.k + kb * KSTEP + ql * 4 + q]) : 0.0f;
      mx2[i] = fmaxf(fmaxf(fabsf(v2[i][0]), fabsf(v2[i][1])),
                     fmaxf(fabsf(v2[i][2]), fabsf(v2[i][3])));
#pragma unroll
      for (int off = 16; off; off >>= 1)
        mx2[i] = fmaxf(mx2[i], __shfl_xor_sync(0xffffffffu, mx2[i], off));
    }
    quant_store(kb, v2, mx2);
  }
  if (!c.a_ready) {
    if (blockIdx.x == 0 && threadIdx.x == 0) g_mk_unit_next = 0u;
    MK_TS(1);  // A quantized, before the publishing barrier
    mk_grid_barrier(bar, c.grid);
  }
  MK_TS(2);  // barrier released (or, a_ready: the caller's)
  for (int i = threadIdx.x; i < c.m * KBLK_MAX; i += MK_THREADS)
    sxs[i] = g_mk_axs[i] * c.wgs;  // undo the pack normalisation (ctx)
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t4 = (lane & 3) * 4;
  const int warp = threadIdx.x >> 5;

  // Units beyond the first come from the shared counter (see
  // g_mk_unit_next); the first is static so its W fill could be hoisted.
  auto next_unit = [&]() -> int {
    __syncthreads();  // the previous unit is done with s_unit and smem
    if (threadIdx.x == 0)
      s_unit = c.grid + (int)atomicAdd(&g_mk_unit_next, 1u);
    __syncthreads();
    return s_unit;
  };
#ifdef MK_PHASE_TS
  unsigned long long twait = 0ull;  // ns inside the W pipeline waits
  unsigned long long tmma = 0ull, texp = 0ull;  // ns in mma_fold / expand (W4)
#endif
  for (int u = blockIdx.x; u < units; u = next_unit()) {
    int nt, kb0, kbn;
    decode_unit(u, nt, kb0, kbn);
    const bool to_partial = split && (u >= full);
    if (kb0 >= kbn) continue;
    float acc[2][2][4];  // [m-tile][n8-half][c-frag]
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.0f;

    // Copy one k-block of the pre-quantized A into the padded smem tile.
    // This used to BE the quantization, redone by every block for every
    // k-block it touched; the prologue above now does it once per launch.
    // Plain stores rather than cp.async: the tile is at most m * 128 = 4 KB
    // and L2-hot, and a cp.async here would land in the same group stream
    // the W pipeline's wait counts depend on.
    // Split in two so the L2 loads go out at the top of an iteration and
    // only the smem stores sit between the syncs: the load's latency was
    // exposed once per k-block (the stamps' ~0.9 us "other" per k-block).
    constexpr int A_WORDS = KSTEP / 4;
    constexpr int A_PER_THREAD = 32 * A_WORDS / MK_THREADS;  // 4 at m = 32
    uint32_t areg[A_PER_THREAD];
    auto stage_a_load = [&](int kb) {
#pragma unroll
      for (int i = 0; i < A_PER_THREAD; ++i) {
        const int t = threadIdx.x + i * MK_THREADS;
        const int r = t / A_WORDS, e = (t % A_WORDS) * 4;
        if (r < c.m)
          areg[i] = *(const uint32_t*)(
              g_mk_aq + ((size_t)kb * 32 + r) * KSTEP + e);
      }
    };
    auto stage_a_store = [&]() {
#pragma unroll
      for (int i = 0; i < A_PER_THREAD; ++i) {
        const int t = threadIdx.x + i * MK_THREADS;
        const int r = t / A_WORDS, e = (t % A_WORDS) * 4;
        if (r < c.m) {
          uint8_t* dst = saq + (r >> 4) * 16 * SMEM_A_PITCH +
                         (r & 15) * SMEM_A_PITCH;
          *(uint32_t*)(dst + mk_swz(r & 15, e)) = areg[i];
        }
      }
      // rows >= m keep stale bytes: their output rows are never written and
      // finite e4m3 cannot poison other rows of the same mma.
    };
    auto stage_a = [&](int kb) { stage_a_load(kb); stage_a_store(); };

    // mma + per-k-block activation-scale fold (the weight group scales
    // are already inside the expanded e4m3 bytes).
    // MT = m-tiles actually present (1 for m <= 16, 2 otherwise), a compile-
    // time constant per instantiation: the loops below fully unroll and the
    // second tile's A loads and mmas simply do not exist for m <= 16 (m=8 is
    // in_proj, the biggest shape). A runtime `break` inside the unrolled
    // loops measured 10% slower -- the compiler gave up on the unroll.
    auto mma_fold_t = [&](auto MTC, const uint8_t* sw, int kb) {
      constexpr int MT = decltype(MTC)::value;
      // ---- mma: warp covers n-cols [warp*16, warp*16+16) of the tile,
      // two m16n8 mmas per k-slice; each n8 half keeps its own kacc
      // (a shared accumulator would sum the two halves' products).
      float kacc[2][2][4];
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) kacc[i][j][c] = 0.0f;

#pragma unroll
      for (int ks = 0; ks < KSTEP / 32; ++ks) {
        const int koff = ks * 32;
        uint32_t a[2][4];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const uint8_t* base = saq + i * 16 * SMEM_A_PITCH;
          const int o0 = mk_swz(g, koff + t4), o1 = mk_swz(g, koff + 16 + t4);
          a[i][0] = *(const uint32_t*)(base + g * SMEM_A_PITCH + o0);
          a[i][1] = *(const uint32_t*)(base + (g + 8) * SMEM_A_PITCH + o0);
          a[i][2] = *(const uint32_t*)(base + g * SMEM_A_PITCH + o1);
          a[i][3] = *(const uint32_t*)(base + (g + 8) * SMEM_A_PITCH + o1);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int nrow = warp * 16 + j * 8 + g;
          const uint8_t* wrow = sw + nrow * SMEM_W_PITCH;
          uint32_t b0 = *(const uint32_t*)(wrow + mk_swz(nrow, koff + t4));
          uint32_t b1 =
              *(const uint32_t*)(wrow + mk_swz(nrow, koff + 16 + t4));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
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
      for (int i = 0; i < MT; ++i) {
        const int r0 = i * 16 + g;
        const int r1 = r0 + 8;
        const float s0 = (r0 < c.m) ? sxs[r0 * KBLK_MAX + kb] : 0.0f;
        const float s1 = (r1 < c.m) ? sxs[r1 * KBLK_MAX + kb] : 0.0f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          acc[i][j][0] += kacc[i][j][0] * s0;
          acc[i][j][1] += kacc[i][j][1] * s0;
          acc[i][j][2] += kacc[i][j][2] * s1;
          acc[i][j][3] += kacc[i][j][3] * s1;
        }
      }

    };
    auto mma_fold = [&](const uint8_t* sw, int kb) {
      if (mtiles == 2)
        mma_fold_t(std::integral_constant<int, 2>{}, sw, kb);
      else
        mma_fold_t(std::integral_constant<int, 1>{}, sw, kb);
    };

    const bool prefilled = hoisted && (u == (int)blockIdx.x);
    {
      // ---- cp.async-staged raw records, expanded in smem.
      // The raw (tile, k-block) record streams through the W4_RAW_NBUF
      // cp.async pipeline; the expansion then reads the landed record and
      // fills one of two e4m3 tile buffers (swb[0..1]).
      // This used to be a row-major pack read with synchronous loads --
      // one tile in flight and 128 DRAM pages per tile -- and did 37 GB/s
      // where the W8 arm did 84 at the same shape, on 0.56x the bytes.
      if (!prefilled) stage_raw4(nt, kb0, kb0 % W4_RAW_NBUF);
      stage_a(kb0);
      if (!prefilled) {
#pragma unroll
        for (int d = 1; d < RAW_DIST; ++d)
          if (kb0 + d < kbn) stage_raw4(nt, kb0 + d, (kb0 + d) % W4_RAW_NBUF);
      }
      mk_cp_wait_upto(min(RAW_DIST - 1, kbn - kb0 - 1));
      __syncthreads();
      expand_w4(kb0 % W4_RAW_NBUF, kb0 % 2);  // the loop reads (kb % 2)
      __syncthreads();
      if (u == (int)blockIdx.x) MK_TS(3);
      for (int kb = kb0;; ++kb) {
        if (kb + RAW_DIST < kbn)
          stage_raw4(nt, kb + RAW_DIST, (kb + RAW_DIST) % W4_RAW_NBUF);
        if (kb + 1 < kbn) stage_a_load(kb + 1);
        const uint8_t* sw4t =
            swb + (kb % 2) * (SMEM_W_ROWS * SMEM_W_PITCH);
        MK_TS_ACC_BEGIN(tm);
        mma_fold(sw4t, kb);  // the group scales are inside the bytes
        MK_TS_ACC_END(tmma, tm);
        if (kb + 1 >= kbn) break;
        // raw(kb+1) landed: the groups still allowed in flight are the
        // ones issued after it, min(RAW_DIST - 1, kbn - kb - 2).
        MK_TS_ACC_BEGIN(tw);
        mk_cp_wait_upto(min(RAW_DIST - 1, kbn - kb - 2));
        __syncthreads();  // raw(kb+1) visible; every mma reader of kb done
        MK_TS_ACC_END(twait, tw);
        MK_TS_ACC_BEGIN(te);
        expand_w4((kb + 1) % W4_RAW_NBUF, (kb + 1) % 2);
        MK_TS_ACC_END(texp, te);
        stage_a_store();
        __syncthreads();
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
    if (to_partial) {
      // release: this block's slice is visible device-wide before its
      // arrival is counted (the threadFenceReduction pattern).
      __threadfence();
      __syncthreads();
      const int lt = nt - full;
      if (threadIdx.x == 0) {
        // slices with kb0 == kbn never arrive; there are min(ksr, kblk)
        // that do (the floor boundaries take exactly that many steps).
        const unsigned expect = (unsigned)min(ksr, kblk);
        const unsigned prev = atomicAdd(&g_mk_tile_arrive[lt], 1u);
        s_last = (prev + 1u == expect);
        if (s_last) g_mk_tile_arrive[lt] = 0u;  // all arrived; rearm
      }
      __syncthreads();
      if (s_last) {
        __threadfence();  // acquire side: the other slices' stores
        // Fold this tile: rows r < m, 128 columns as 32 float4, every
        // non-empty slice in index order. __ldcg: the partials were
        // written by other SMs, and L1 is not coherent within a launch.
        for (int i2 = threadIdx.x; i2 < c.m * 32; i2 += MK_THREADS) {
          const int r = i2 >> 5, c4 = (i2 & 31) * 4;
          const float* src =
              g_mk_gemm_partial + (size_t)r * pcols + lt * 128 + c4;
          float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          // ksr <= kblk (every production shape): no slice is empty and no
          // division in the loop -- the two integer divides per slice per
          // element of the general form put ~10 us on the last tile.
          if (ksr <= kblk) {
            for (int spx = 0; spx < ksr; ++spx) {  // fixed order -> reproducible
              const float4 pv = __ldcg((const float4*)(src + (size_t)spx * pslice));
              v4.x += pv.x; v4.y += pv.y; v4.z += pv.z; v4.w += pv.w;
            }
          } else {
            for (int spx = 0; spx < ksr; ++spx) {
              if ((kblk * (spx + 1)) / ksr <= (kblk * spx) / ksr) continue;
              const float4 pv = __ldcg((const float4*)(src + (size_t)spx * pslice));
              v4.x += pv.x; v4.y += pv.y; v4.z += pv.z; v4.w += pv.w;
            }
          }
          const int col = nt * 128 + c4;
          __nv_bfloat16* o = c.out + (size_t)r * c.n_orig + col;
          if (col < c.n_orig) o[0] = __float2bfloat16(v4.x);
          if (col + 1 < c.n_orig) o[1] = __float2bfloat16(v4.y);
          if (col + 2 < c.n_orig) o[2] = __float2bfloat16(v4.z);
          if (col + 3 < c.n_orig) o[3] = __float2bfloat16(v4.w);
        }
      }
      __syncthreads();  // s_last is reused by the next unit
    }
    if (u == (int)blockIdx.x) MK_TS(6);  // first unit (incl. its epilogue) done
  }
  MK_TS(4);  // all units of this block done (no fold barrier any more)
#ifdef MK_PHASE_TS
  MK_TS_STORE(5, twait);
  MK_TS_STORE(3, tmma);
  MK_TS_STORE(7, texp);
#endif
}

__global__ void mk_gemm_kernel(const MKGemmCtx c) {
  extern __shared__ uint8_t smem[];
  // PDL: the next launch in the stream may start on the SMs this grid
  // frees as blocks exit; it prefetches its own weights during this
  // grid's tail and waits (griddepcontrol.wait) before reading anything
  // this grid writes. Harmless when the next launch is not programmatic.
  asm volatile("griddepcontrol.launch_dependents;");
  mk_gemm_phase(c, smem, &g_mk_gemm_bar);
}

// ===========================================================================
// MK-GEMM v2 -- the same W4A8 GEMM as a NON-persistent grid: one block per
// (n-tile, k-slice) unit, no grid barrier, no shared A staging, the W4
// expansion done in registers straight into the mma B fragments.
//
// Why (30차, the 09-04 18:42 armed trace read against the offline bench):
// the persistent kernel above is right for a kernel that owns the GPU and
// wrong for one that shares it. Its 48 x 69.6 KB blocks cannot co-reside
// with the routed MoE kernel (90 KB, 48 blocks), so the shared expert's two
// launches on the side stream queue behind the MoE kernel and the publish
// barrier then holds every landed block for the last one to land: down
// [4096 x 512] is 18 us alone and 135 us in the step (42 launches, 5.7 of
// the lane's 14.1 ms). On the main stream the once-per-launch prologue (x
// load, quant, barrier) and the static first-unit assignment cost 15-35 us
// a launch: dense gate_up [6144 x 4096] is 48 tiles on 48 blocks with
// nothing to balance them and runs 119 us where its bytes are 62.
//
// Here grid = (n/128) x ksr blocks of ~37 KB smem, so two sit on an SM
// (registers capped at 128 by the launch bounds) and the hardware scheduler
// does the balancing and the overlap with whatever else is running. Each
// block quantizes the A k-blocks of ITS slice from x (L2) as it goes --
// same amax, same pow2 scale, same conversion as the shared prologue, so
// the bytes the mma sees are the ones mk_gemm_kernel sees -- and the raw
// (tile, k-block) records stream through the same cp.async ring. The e2m1
// nibbles expand into the B fragments with the same prmt table lookup,
// only per lane instead of per smem tile, which is what frees the 32 KB of
// expanded tiles. ksr > 1 slices go to an fp32 partial and the last slice
// to arrive folds the tile in fixed slice order (the persistent kernel's
// leftover fold, applied to every tile): deterministic, no zero pass.
//
// Shared state: the partials and the per-tile arrival counters below are
// one set for the whole device, like the persistent lane's A tiles and
// unit counter -- so, like every MK GEMM launch, a v2 launch must never
// overlap another MK GEMM launch on a different stream (two launches
// folding the same tile index would sum each other's slices). Serving
// keeps that by stream order: the side-stream pair is joined before the
// next main-stream GEMM.
// ===========================================================================
// v2 raw-record ring depth (VLLM_GLM53_MK_NBUF2; 2..4 keep two blocks per SM)
#ifndef MK_NBUF2_DEF
#define MK_NBUF2_DEF 3
#endif
constexpr int W4_RAW_NBUF2 = MK_NBUF2_DEF;
static_assert(W4_RAW_NBUF2 >= 2 && W4_RAW_NBUF2 <= 5, "v2 raw stages");
static_assert(W4_RAW_NBUF2 - 1 <= 4, "mk_cp_wait_upto dispatches up to 4");
constexpr int GEMM2_SMEM = MK_SMEM_ALIGN + 2 * 32 * SMEM_A_PITCH +
                           2 * 32 * 4 + W4_RAW_NBUF2 * W4_RAW_BYTES;  // 37,120 at 3
// Two blocks per SM is the point of this kernel: the SM has 102,400 B of
// shared memory and reserves ~1 KB per resident block.
static_assert(2 * (GEMM2_SMEM + 1024) <= 102400, "v2 must fit twice per SM");
constexpr int MK2_TILES_MAX = 64;               // n <= 8192
constexpr int MK2_KSR_MAX = 8;
// fp32 partials [ksr][m][n]: the widest per-rank linear (in_proj, 6528)
// at m = 32 and ksr = 4; the host lowers ksr for anything wider.
constexpr int MK2_PART_ELEMS = 32 * KDA_INPROJ_N_PAD * 4;
__device__ __align__(16) float g_mk2_partial[MK2_PART_ELEMS];
// per n-tile slice arrivals, self-rearming like g_mk_tile_arrive
__device__ unsigned int g_mk2_tile_arrive[MK2_TILES_MAX];
#ifdef MK_PHASE_TS
constexpr int MK2_UNITS_MAX = MK2_TILES_MAX * MK2_KSR_MAX;  // 512
// [unit][4]: entry, first record landed, last mma done, exit
__device__ unsigned long long g_mk2_ts[MK2_UNITS_MAX * 4];
#define MK2_TS(slot)                                                          \
  do {                                                                        \
    if (threadIdx.x == 0 && (int)blockIdx.x < MK2_UNITS_MAX)                  \
      g_mk2_ts[blockIdx.x * 4 + (slot)] = mk_globaltimer();                   \
  } while (0)
#else
#define MK2_TS(slot) \
  do {               \
  } while (0)
#endif

struct MKGemm2Ctx {
  const __nv_bfloat16* x;  // [m, k]
  __nv_bfloat16* out;      // [m, n_orig]
  const uint8_t* wq4;      // tile-major W4 pack, see MKGemmCtx
  const int8_t* ws4;
  float wgs;
  int m, n, k, n_orig;
  int ksr;                 // k-slices per tile; grid = (n / 128) * ksr
};

// Four e2m1 nibbles (one 16-bit word, elements ascending) -> four e4m3
// bytes under the group's scale byte: expand_w4's table lookup, one
// fragment register at a time. Same table, same adds, same sign prmt, so
// the byte for every (code, scale) is the byte the smem expansion wrote.
__device__ __forceinline__ uint32_t mk_w4x4(uint32_t w16, uint32_t eb) {
  const unsigned long long lutg =
      ((eb & 7u) - 1u) < 5u ? MK_E2M1_LUT64_B : MK_E2M1_LUT64;
  const uint32_t l0 = __vadd4((uint32_t)lutg, eb * 0x01010100u);
  const uint32_t l1 = __vadd4((uint32_t)(lutg >> 32), eb * 0x01010101u);
  return __byte_perm(l0, l1, w16 & 0x7777u) |
         __byte_perm(0x8000u, 0u, (w16 >> 3) & 0x1111u);
}

__global__ void __launch_bounds__(MK_THREADS, 2)
mk_gemm2_kernel(const MKGemm2Ctx c) {
  extern __shared__ uint8_t smem[];
  // PDL: dependents may start on the SMs this grid frees; this grid's own
  // W fill goes out before its griddepcontrol.wait (below).
  asm volatile("griddepcontrol.launch_dependents;");
  uint8_t* sb0 = smem;
  {  // 1 KB-aligned base (the 128 B A rows and the 64 B raw rows both
     // want bank-line alignment; the static s_last below shifts the base)
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(sb0);
    sb0 += (MK_SMEM_ALIGN - (sb & (MK_SMEM_ALIGN - 1))) & (MK_SMEM_ALIGN - 1);
  }
  uint8_t* saq = sb0;                                   // [2][32][128] swizzled e4m3 A
  float* sxs = (float*)(saq + 2 * 32 * SMEM_A_PITCH);   // [2][32] row scales, wgs folded
  uint8_t* sraw = (uint8_t*)(sxs + 2 * 32);             // [NB][W4_RAW_BYTES]
  __shared__ int s_last;

  const int kblk = c.k / KSTEP;
  const int ksr = c.ksr;
  const int nt = (int)blockIdx.x / ksr, sp = (int)blockIdx.x % ksr;
  // ksr <= kblk (host contract), so every slice is non-empty
  const int kb0 = (kblk * sp) / ksr, kbn = (kblk * (sp + 1)) / ksr;
  const int mtiles = (c.m + 15) / 16;
  MK2_TS(0);
  constexpr int NB = W4_RAW_NBUF2, DIST = NB - 1;

  // One raw (tile, k-block) record -> ring stage `buf`, all 256 threads.
  // The 16 B nibble chunks land XOR-swizzled by (row >> 1) & 3 so the
  // fragment loads below (eight rows, one chunk each, 64 B row pitch) hit
  // eight distinct bank groups instead of two.
  auto stage_raw = [&](int kb, int buf) {
    const uint8_t* nsrc =
        c.wq4 + ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * 64);
    const uint8_t* ssrc = (const uint8_t*)c.ws4 +
        ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * 8);
    uint8_t* d = sraw + buf * W4_RAW_BYTES;
    // two chunks per thread, unrolled: the swizzled destinations are
    // per-thread constants (a runtime loop recomputed them every k-block)
    static_assert((SMEM_W_ROWS * 4) % MK_THREADS == 0, "chunks per thread");
#pragma unroll
    for (int u = 0; u < (SMEM_W_ROWS * 4) / MK_THREADS; ++u) {
      const int t = (int)threadIdx.x + u * MK_THREADS;
      const int r = t >> 2, ch = t & 3;
      mk_cp_async16(d + r * W4_RAW_PITCH + ((ch ^ ((r >> 1) & 3)) << 4),
                    nsrc + (size_t)t * 16);
    }
    if (threadIdx.x < SMEM_W_ROWS * 8 / 16)
      mk_cp_async16(d + W4_RAW_NIB + threadIdx.x * 16,
                    ssrc + (size_t)threadIdx.x * 16);
    mk_cp_commit();
  };

  // x -> registers (one 8 B vector per lane per row, warp w owns rows
  // w, w+8, w+16, w+24) and the quant into an A buffer: quant_store's
  // arithmetic row by row -- warp amax, pow2 scale, SATFINITE e4m3 -- so
  // the tile is byte for byte what the shared prologue publishes.
  const int qw = threadIdx.x >> 5, ql = threadIdx.x & 31;
  constexpr int RPW = 32 / MK_WARPS;  // 4
  uint2 xr[RPW];
  auto load_x = [&](int kb) {
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
      const int r = qw + i * MK_WARPS;
      if (r < c.m)
        xr[i] = *(const uint2*)(c.x + (size_t)r * c.k + kb * KSTEP + ql * 4);
    }
  };
  auto quant_x = [&](int buf) {
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
      const int r = qw + i * MK_WARPS;
      if (r >= c.m) break;  // rows ascend with i (warp-uniform)
      const __nv_bfloat16* pv = (const __nv_bfloat16*)&xr[i];
      float v[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) v[q] = __bfloat162float(pv[q]);
      float mx = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])),
                       fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
      for (int off = 16; off; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      const float sc = mk_pow2_scale(mx);
      // sc is a power of two, so rcp.rn is exact -- the same rsc as the
      // shared prologue's 1.0f / sc, without the IEEE divide's slow-path
      // call that the SASS showed on every row of every k-block
      const float rsc = __frcp_rn(sc);
      uint32_t pack = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        pack |= (uint32_t)mk_f32_to_e4m3(v[q] * rsc) << (8 * q);
      uint8_t* dst = saq + buf * (32 * SMEM_A_PITCH) +
                     (r >> 4) * 16 * SMEM_A_PITCH + (r & 15) * SMEM_A_PITCH;
      *(uint32_t*)(dst + mk_swz(r & 15, ql * 4)) = pack;
      if (ql == 0) sxs[buf * 32 + r] = sc * c.wgs;
    }
  };

  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t4 = (lane & 3) * 4;
  const int warp = threadIdx.x >> 5;
  // this lane's two nibble bytes inside a 16 B chunk: elements t4..t4+3
  // are bytes t4/2, t4/2+1 (b0), the same two of the upper 8 B (b1)
  const uint32_t sel0 = (uint32_t)(t4 >> 1) | ((uint32_t)(t4 >> 1) + 1u) << 4;

  float acc[2][2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.0f;

  // mma over one k-block: B straight from the raw record (nibbles -> e4m3
  // fragments in registers), A from the swizzled tile, then the fold with
  // the row scales. MT = m-tiles present (compile-time per instantiation,
  // like mma_fold_t above).
  auto mma_fold_t = [&](auto MTC, int rbuf, int abuf) {
    constexpr int MT = decltype(MTC)::value;
    const uint8_t* rr = sraw + rbuf * W4_RAW_BYTES;
    const uint8_t* sa = saq + abuf * (32 * SMEM_A_PITCH);
    float kacc[MT][2][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) kacc[i][j][q] = 0.0f;
    // the 8 scale bytes of each of this lane's two W rows
    uint2 scb[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int nrow = warp * 16 + j * 8 + g;
      scb[j] = *(const uint2*)(rr + W4_RAW_NIB + nrow * 8);
    }
#pragma unroll
    for (int ks = 0; ks < KSTEP / 32; ++ks) {
      const int koff = ks * 32;
      uint32_t a[2][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const uint8_t* base = sa + i * 16 * SMEM_A_PITCH;
        const int o0 = mk_swz(g, koff + t4), o1 = mk_swz(g, koff + 16 + t4);
        a[i][0] = *(const uint32_t*)(base + g * SMEM_A_PITCH + o0);
        a[i][1] = *(const uint32_t*)(base + (g + 8) * SMEM_A_PITCH + o0);
        a[i][2] = *(const uint32_t*)(base + g * SMEM_A_PITCH + o1);
        a[i][3] = *(const uint32_t*)(base + (g + 8) * SMEM_A_PITCH + o1);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int nrow = warp * 16 + j * 8 + g;
        // chunk ks of the row = elements 32ks .. 32ks+31 (swizzled slot)
        const uint4 w4 = *(const uint4*)(
            rr + nrow * W4_RAW_PITCH + ((ks ^ ((nrow >> 1) & 3)) << 4));
        const uint32_t w0 = __byte_perm(w4.x, w4.y, sel0) & 0xFFFFu;
        const uint32_t w1 = __byte_perm(w4.z, w4.w, sel0) & 0xFFFFu;
        // groups 2ks (b0) and 2ks+1 (b1): scale bytes 2ks, 2ks+1
        const uint32_t sw = (ks < 2) ? scb[j].x : scb[j].y;
        const uint32_t e0 = (sw >> (16 * (ks & 1))) & 0xFFu;
        const uint32_t e1 = (sw >> (16 * (ks & 1) + 8)) & 0xFFu;
        const uint32_t b0 = mk_w4x4(w0, e0);
        const uint32_t b1 = mk_w4x4(w1, e1);
#pragma unroll
        for (int i = 0; i < MT; ++i) {
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
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      const int r0 = i * 16 + g, r1 = r0 + 8;
      const float s0 = (r0 < c.m) ? sxs[abuf * 32 + r0] : 0.0f;
      const float s1 = (r1 < c.m) ? sxs[abuf * 32 + r1] : 0.0f;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        acc[i][j][0] += kacc[i][j][0] * s0;
        acc[i][j][1] += kacc[i][j][1] * s0;
        acc[i][j][2] += kacc[i][j][2] * s1;
        acc[i][j][3] += kacc[i][j][3] * s1;
      }
    }
  };
  auto mma_fold = [&](int rbuf, int abuf) {
    if (mtiles == 2)
      mma_fold_t(std::integral_constant<int, 2>{}, rbuf, abuf);
    else
      mma_fold_t(std::integral_constant<int, 1>{}, rbuf, abuf);
  };

  // ---- prologue: the W ring first (independent of the previous kernel,
  // so it flies during that kernel's tail under PDL), then the wait, then
  // x(kb0) -> A buffer 0.
#pragma unroll
  for (int d = 0; d < DIST; ++d)
    if (kb0 + d < kbn) stage_raw(kb0 + d, (kb0 + d) % NB);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  load_x(kb0);
  quant_x(0);
  mk_cp_wait_upto(min(DIST - 1, kbn - kb0 - 1));  // raw(kb0) landed
  __syncthreads();
  MK2_TS(1);
  // ---- k loop: one __syncthreads per k-block. The refill targets the
  // stage consumed last iteration (everyone passed that sync); the quant
  // writes the A buffer the current mma is not reading.
  for (int kb = kb0;; ++kb) {
    if (kb + DIST < kbn) stage_raw(kb + DIST, (kb + DIST) % NB);
    if (kb + 1 < kbn) load_x(kb + 1);
    mma_fold(kb % NB, (kb - kb0) & 1);
    if (kb + 1 >= kbn) break;
    quant_x((kb + 1 - kb0) & 1);
    // raw(kb+1) landed: groups issued after it may stay in flight
    mk_cp_wait_upto(min(DIST - 1, kbn - kb - 2));
    __syncthreads();
  }
  MK2_TS(2);

  // ---- epilogue: one walk over the fragment's real rows / cols, two stores
  auto store_tile = [&](auto&& put) {  // put(row, col, value)
#pragma unroll
    for (int i = 0; i < 2; ++i) {  // static index: a runtime bound put acc in local memory
      if (i >= mtiles) break;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int r0 = i * 16 + g, r1 = r0 + 8;
        const int cb = nt * 128 + warp * 16 + j * 8 + (lane & 3) * 2;
        if (r0 < c.m) { put(r0, cb, acc[i][j][0]); put(r0, cb + 1, acc[i][j][1]); }
        if (r1 < c.m) { put(r1, cb, acc[i][j][2]); put(r1, cb + 1, acc[i][j][3]); }
      }
    }
  };
  if (ksr == 1) {  // whole tile: bf16 out
    store_tile([&](int r, int col, float v) {
      if (col < c.n_orig) c.out[(size_t)r * c.n_orig + col] = __float2bfloat16(v);
    });
    MK2_TS(3);
    return;
  }
  // k-slice: assign (never accumulate) this slice's partial, count the
  // arrival, and let the last slice fold the tile in slice order.
  {
    float* pb = g_mk2_partial + (size_t)sp * c.m * c.n;
    store_tile([&](int r, int col, float v) { pb[(size_t)r * c.n + col] = v; });
  }
  __syncthreads();
  __threadfence();  // release: the slice is visible device-wide first
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned prev = atomicAdd(&g_mk2_tile_arrive[nt], 1u);
    s_last = (prev + 1u == (unsigned)ksr);
    if (s_last) g_mk2_tile_arrive[nt] = 0u;  // all slices in; rearm
  }
  __syncthreads();
  if (s_last) {
    __threadfence();  // acquire: the other slices' partials
    for (int i2 = threadIdx.x; i2 < c.m * 32; i2 += MK_THREADS) {
      const int r = i2 >> 5, c4 = (i2 & 31) * 4;
      const float* src = g_mk2_partial + (size_t)r * c.n + nt * 128 + c4;
      float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int s = 0; s < ksr; ++s) {  // fixed order -> reproducible
        const float4 pv = __ldcg((const float4*)(src + (size_t)s * c.m * c.n));
        v4.x += pv.x; v4.y += pv.y; v4.z += pv.z; v4.w += pv.w;
      }
      const int col = nt * 128 + c4;
      __nv_bfloat16* o = c.out + (size_t)r * c.n_orig + col;
      if (col < c.n_orig) o[0] = __float2bfloat16(v4.x);
      if (col + 1 < c.n_orig) o[1] = __float2bfloat16(v4.y);
      if (col + 2 < c.n_orig) o[2] = __float2bfloat16(v4.z);
      if (col + 3 < c.n_orig) o[3] = __float2bfloat16(v4.w);
    }
  }
  MK2_TS(3);
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

// Per-token chunk arrivals (rearmed by the block that runs the token's
// tail), the tail ticket counter, and the exit ticket whose last holder
// rearms the tail counter -- so graph replay needs no host-side reset (the
// tile counters' trick, twice).
__device__ unsigned int g_mk_mhc_tok_arrive[MAX_TOK];
__device__ unsigned int g_mk_mhc_tail_next = 0u;
__device__ unsigned int g_mk_mhc_exit = 0u;

__device__ __forceinline__ float mk_ldcg_bf16(const __nv_bfloat16* p) {
  return __bfloat162float(__ushort_as_bfloat16(__ldcg((const unsigned short*)p)));
}

// p2 for ONE token, by one warp: the 24 chunk reductions, rms, the post /
// comb (sinkhorn) / pre mixes. Reads the other blocks' partials through
// L2 (__ldcg): they were published with a fence + the arrival counter.
__device__ void mk_mhc_p2_token(const MKMhcArgs& a, int t, float* s_pmix) {
  // One warp, and as few DEPENDENT shuffles as possible: on this part a
  // dependent shuffle step in this tail measured ~0.15 us (the 5-step rms
  // reduce 0.75 us, the 16-step lane-parallel sinkhorn ~3.5 us, and a
  // warm second pass cost the same, so it is not cold code). Now: lanes
  // 0..23 sum their output's 16 partials, lane 24 sums all 16 sumsq
  // partials itself and broadcasts rms with ONE shuffle, lane 0 gathers
  // the 16 comb inputs with 16 INDEPENDENT shuffles and runs the whole
  // 4x4 sinkhorn in its registers (fully unrolled, no shuffles), lanes
  // 4..7 / 8..11 do the post / pre sigmoids meanwhile.
  const int lane = threadIdx.x & 31;
  const float hs0 = a.hc_scale[0], hs1 = a.hc_scale[1], hs2 = a.hc_scale[2];
  const float hb_post = a.hc_base[(lane & 3) + HC];
  const float hb_pre = a.hc_base[lane & 3];
  // lane 0's 16 comb biases, issued with the partial-sum loads
  const float4* hb4 = (const float4*)(a.hc_base + 2 * HC);
  const float4 hbc0 = hb4[0], hbc1 = hb4[1], hbc2 = hb4[2], hbc3 = hb4[3];
  float mine = 0.0f;
  if (lane < NOUT) {
#pragma unroll
    for (int c = 0; c < NCHUNK; ++c)
      mine += __ldcg(&a.yp[((size_t)c * MAX_TOK + t) * NOUT + lane]);
  } else if (lane == NOUT) {
#pragma unroll
    for (int c = 0; c < NCHUNK; ++c) mine += __ldcg(&a.rp[c * MAX_TOK + t]);
  }
  MK_MHC_PROBE(1);  // partial sums landed
  const float rms_l = rsqrtf(mine / (float)(HC * HIDDEN) + a.rms_eps);
  const float rms = __shfl_sync(0xffffffffu, rms_l, NOUT);
  const float mixv = mine * rms;
  const float post_in = __shfl_sync(0xffffffffu, mixv, HC + (lane & 3));
  const float pre_in = __shfl_sync(0xffffffffu, mixv, lane & 3);
  float m[HC][HC];
#pragma unroll
  for (int j = 0; j < HC; ++j)
#pragma unroll
    for (int k = 0; k < HC; ++k)
      m[j][k] = __shfl_sync(0xffffffffu, mixv, j * HC + k + 2 * HC);
  MK_MHC_PROBE(2);  // mixes fetched
  if (lane >= HC && lane < 2 * HC) {  // post mixes: hc_scale[1], lanes 4..7
    a.post_mix_out[t * HC + (lane & 3)] =
        mk_sigmoid(post_in * hs1 + hb_post) * a.post_mult;
  }
  if (lane >= 2 * HC && lane < 3 * HC) {  // pre mixes: hc_scale[0], 8..11
    s_pmix[lane & 3] = mk_sigmoid(pre_in * hs0 + hb_pre) + a.pre_eps;
  }
  if (lane == 0) {  // comb mixes: hc_scale[2] + sinkhorn, 4x4 in registers
    const float hb[HC][HC] = {{hbc0.x, hbc0.y, hbc0.z, hbc0.w},
                              {hbc1.x, hbc1.y, hbc1.z, hbc1.w},
                              {hbc2.x, hbc2.y, hbc2.z, hbc2.w},
                              {hbc3.x, hbc3.y, hbc3.z, hbc3.w}};
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      float rm = -INFINITY;
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        m[j][k] = m[j][k] * hs2 + hb[j][k];
        rm = fmaxf(rm, m[j][k]);
      }
      float rs = 0.0f;
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        m[j][k] = __expf(m[j][k] - rm);
        rs += m[j][k];
      }
#pragma unroll
      for (int k = 0; k < HC; ++k)
        m[j][k] = __fdividef(m[j][k], rs) + a.sinkhorn_eps;
    }
#pragma unroll
    for (int k = 0; k < HC; ++k) {
      float cs = 0.0f;
#pragma unroll
      for (int j = 0; j < HC; ++j) cs += m[j][k];
#pragma unroll
      for (int j = 0; j < HC; ++j)
        m[j][k] = __fdividef(m[j][k], cs + a.sinkhorn_eps);
    }
    for (int it = 0; it < a.sinkhorn_repeat - 1; ++it) {
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float rs = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) rs += m[j][k];
#pragma unroll
        for (int k = 0; k < HC; ++k)
          m[j][k] = __fdividef(m[j][k], rs + a.sinkhorn_eps);
      }
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        float cs = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) cs += m[j][k];
#pragma unroll
        for (int j = 0; j < HC; ++j)
          m[j][k] = __fdividef(m[j][k], cs + a.sinkhorn_eps);
      }
    }
    MK_MHC_PROBE(3);  // sinkhorn done
    float4* out4 = (float4*)(a.comb_mix_out + t * HC * HC);
#pragma unroll
    for (int j = 0; j < HC; ++j)
      out4[j] = make_float4(m[j][0], m[j][1], m[j][2], m[j][3]);
  }
}

// p3 + p4 for ONE token, by one block: pre-mix the four residual streams
// (16 elements per thread, kept in registers), one block-wide sum of
// squares, then the normalized layer input -- the ol_stash / sq round
// trips and their two grid barriers are gone.
// p3 + p4 for ONE token, by one block: pre-mix the four residual streams
// (16 elements per thread, kept in registers), one block-wide sum of
// squares, then the normalized layer input -- the ol_stash / sq round
// trips and their two grid barriers are gone. The residual loads do not
// depend on p2, so every warp issues them BEFORE warp 0 runs p2 (load
// phase) and only the mixing runs after (compute phase).
constexpr int MHC_EPT = HIDDEN / MK_THREADS;  // 16 elements per thread
struct MhcTailRegs {
  float res[HC][MHC_EPT];
  float nw[MHC_EPT];
};

__device__ __forceinline__ void mk_mhc_p34_load(const MKMhcArgs& a, int t,
                                                MhcTailRegs& r) {
#pragma unroll
  for (int i = 0; i < MHC_EPT; ++i) {
    const int h = i * MK_THREADS + threadIdx.x;
#pragma unroll
    for (int j = 0; j < HC; ++j)
      r.res[j][i] = mk_ldcg_bf16(a.residual_out + (size_t)t * HC * HIDDEN +
                                 j * HIDDEN + h);
    r.nw[i] = __bfloat162float(a.norm_weight[h]);
  }
}

__device__ void mk_mhc_p34_compute(const MKMhcArgs& a, int t,
                                   const float* s_pmix,
                                   const MhcTailRegs& r) {
  __shared__ float sqred[MK_WARPS];
  float pre[HC];
#pragma unroll
  for (int j = 0; j < HC; ++j) pre[j] = s_pmix[j];
  float vals[MHC_EPT], sq = 0.0f;
#pragma unroll
  for (int i = 0; i < MHC_EPT; ++i) {
    float v = 0.0f;
#pragma unroll
    for (int j = 0; j < HC; ++j) v += pre[j] * r.res[j][i];
    sq += v * v;
    // the old p3 stashed v as bf16 and p4 read it back: same rounding
    vals[i] = __bfloat162float(__float2bfloat16(v));
  }
  MK_MHC_PROBE(5);  // loads consumed, mixing done
#pragma unroll
  for (int off = 16; off; off >>= 1) sq += __shfl_xor_sync(~0u, sq, off);
  if ((threadIdx.x & 31) == 0) sqred[threadIdx.x >> 5] = sq;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int w = 0; w < MK_WARPS; ++w) tot += sqred[w];
  const float rsq = rsqrtf(tot / (float)HIDDEN + a.norm_eps);
#pragma unroll
  for (int i = 0; i < MHC_EPT; ++i) {
    const int h = i * MK_THREADS + threadIdx.x;
    a.layer_input[t * HIDDEN + h] =
        __float2bfloat16(vals[i] * rsq * r.nw[i]);
  }
  __syncthreads();  // sqred reuse
}

__device__ void mk_mhc_p1(const MKMhcArgs& a, int bid) {
  // Block = (chunk, token group). The chunk's fn slice -- 24 outputs x 4
  // streams for this thread's h -- lives in 96 REGISTERS, loaded once, and
  // the group's tokens run against it. At T=8 the old (token, chunk)-pair
  // mapping had all 128 pairs in flight at once, each pulling its 96 KB of
  // fn: 12 MB through L2 -> SM in one round, ~10 us, which is the L2 rate,
  // not DRAM (fn is 1.5 MB). grid / NCHUNK token groups per chunk (6 at
  // the 96 resident blocks this kernel's registers allow) cut that to 8
  // MB and keep every block busy. (The smem-slice form measured worse:
  // its 96-load fill was latency-bound and every token then paid 96 smem
  // reads per thread; registers pay neither. Three groups on 48 blocks
  // left half the grid idle: T=8 span 22.6 us.)
  //  * the 25 per-token partials are reduced through a transposed smem
  //    tile (8 loads + 5 shuffles per output per warp);
  //  * a token's p2 / p3 / p4 ("tail", ~4 us on one block) runs on
  //    whichever block is free: blocks that are done take tail tickets
  //    and wait on the token's chunk-arrival counter -- no grid barrier.
  const int groups = max(1, a.grid / NCHUNK);  // token groups per chunk
  __shared__ float part[MK_THREADS][NOUT + 3];  // pitch 27: conflict-free
  __shared__ float s_pmix[HC];
  __shared__ int s_tok;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // a token's x / residual inputs for this thread's h, loaded one token
  // ahead so the chain does not open with a global round trip
  auto load_tok = [&](int t, int h, float& xv_, float (&res_)[HC],
                      float (&pm_)[HC], float (&cm_)[HC][HC]) {
    xv_ = __bfloat162float(a.x_in[t * HIDDEN + h]);
#pragma unroll
    for (int k = 0; k < HC; ++k)
      res_[k] = __bfloat162float(
          a.residual_in[(size_t)t * HC * HIDDEN + k * HIDDEN + h]);
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      pm_[j] = a.post_mix_in[t * HC + j];
#pragma unroll
      for (int k = 0; k < HC; ++k)
        cm_[k][j] = a.comb_mix_in[t * HC * HC + k * HC + j];
    }
  };
  for (int cg = bid; cg < NCHUNK * groups; cg += a.grid) {
    const int c = cg % NCHUNK, g = cg / NCHUNK;
    const int h = c * HCHUNK + threadIdx.x;  // HCHUNK == MK_THREADS
    float xv = 0.0f, res[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    float pm[HC], cm[HC][HC];
    if (g < a.num_tokens) load_tok(g, h, xv, res, pm, cm);
    float fnr[NOUT][HC];
#pragma unroll
    for (int m = 0; m < NOUT; ++m)
#pragma unroll
      for (int j = 0; j < HC; ++j)
        fnr[m][j] = a.fn[(size_t)m * HC * HIDDEN + j * HIDDEN + h];
    int pend = -1;  // a token whose chunk is done but not yet published
    for (int t = g; t < a.num_tokens; t += groups) {
      float nxv = 0.0f, nres[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
      float npm[HC], ncm[HC][HC];
      if (t + groups < a.num_tokens)
        load_tok(t + groups, h, nxv, nres, npm, ncm);
      float r[HC], sqr = 0.0f;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float v = pm[j] * xv;
#pragma unroll
        for (int k = 0; k < HC; ++k) v += cm[k][j] * res[k];
        r[j] = v;
        a.residual_out[(size_t)t * HC * HIDDEN + j * HIDDEN + h] =
            __float2bfloat16(v);
        sqr += v * v;
      }
#pragma unroll
      for (int m = 0; m < NOUT; ++m) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) v += fnr[m][j] * r[j];
        part[threadIdx.x][m] = v;
      }
      part[threadIdx.x][NOUT] = sqr;
      __syncthreads();
      // 25 columns over 8 warps x 4 lane-groups of 8: warp w, group q
      // owns column w + 8q (< 25); each lane sums 32 rows (8 + 32 i) and
      // three xor-shuffles finish the group -- one pass of 32 loads + 3
      // shuffles per lane instead of 3-4 passes of 8 + 5.
      {
        const int q = lane >> 3, l8 = lane & 7;
        const int m = warp + 8 * q;
        float v = 0.0f;
        if (m <= NOUT) {
#pragma unroll
          for (int i = 0; i < MK_THREADS / 8; ++i)
            v += part[l8 + 8 * i][m];
        }
#pragma unroll
        for (int off = 4; off; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
        if (l8 == 0 && m <= NOUT) {
          if (m < NOUT)
            a.yp[((size_t)c * MAX_TOK + t) * NOUT + m] = v;
          else
            a.rp[c * MAX_TOK + t] = v;
        }
      }
      // publish every second token (and the last): the fence + sync +
      // atomic is ~1 us of the ~2.3 us per-token chain at T=32; a token's
      // tail can start one token later without loss (tails overlap p1 at
      // T=32 and only start after it at T=8 anyway)
      if (pend >= 0) {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
          atomicAdd(&g_mk_mhc_tok_arrive[pend], 1u);
          atomicAdd(&g_mk_mhc_tok_arrive[t], 1u);
        }
        pend = -1;
      } else if (t + groups >= a.num_tokens) {  // last token of this block
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(&g_mk_mhc_tok_arrive[t], 1u);
      } else {
        pend = t;
      }
      xv = nxv;
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        res[k] = nres[k];
        pm[k] = npm[k];
#pragma unroll
        for (int j = 0; j < HC; ++j) cm[k][j] = ncm[k][j];
      }
    }
  }
  MK_MHC_TS(1);
  // ---- tails: take tokens off the ticket counter until it runs past T;
  // wait for the token's 16 chunks (they are all in flight on resident
  // blocks, so the wait is bounded), rearm its counter, run p2 / p3 / p4.
  MK_MHC_TS(5);
  for (;;) {
    if (threadIdx.x == 0) s_tok = (int)atomicAdd(&g_mk_mhc_tail_next, 1u);
    __syncthreads();
    const int t = s_tok;
    if (t >= a.num_tokens) break;
    if (threadIdx.x == 0) {
      volatile unsigned int* v = &g_mk_mhc_tok_arrive[t];
      while (*v < (unsigned int)NCHUNK) __nanosleep(128);
      g_mk_mhc_tok_arrive[t] = 0u;  // rearm for the next launch
      __threadfence();
    }
    __syncthreads();
    MK_MHC_TS(2);  // (probe) last tail's p2 start
    MhcTailRegs tr;
    // warps 1..7 issue their p34 loads under p2; warp 0 loads after its
    // p2 so the sinkhorn chain is not squeezed for registers by them
    if (warp != 0) mk_mhc_p34_load(a, t, tr);
    MK_MHC_PROBE(4);  // p34 loads issued
    if (warp == 0) {
      mk_mhc_p2_token(a, t, s_pmix);
      mk_mhc_p34_load(a, t, tr);
    }
    __syncthreads();
    MK_MHC_TS(3);  // (probe) p2 end / p34 start
    mk_mhc_p34_compute(a, t, s_pmix, tr);  // ends in a __syncthreads
    MK_MHC_TS(4);  // (probe) p34 end
  }
  MK_MHC_TS(6);
  // exit ticket: the last block out rearms the tail counter for the next
  // launch (every block has made its final, failing take by then)
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned int e = atomicAdd(&g_mk_mhc_exit, 1u);
    if (e + 1u == (unsigned int)a.grid) {
      g_mk_mhc_exit = 0u;
      g_mk_mhc_tail_next = 0u;
      __threadfence();
    }
  }
}

__global__ void mk_mhc_kernel(const MKMhcArgs a) {
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  MK_MHC_TS(0);
  // p1 over the (token, chunk) pairs; each token's p2 / p3 / p4 run on
  // the block that completes its last chunk (see mk_mhc_p1) -- no grid
  // barrier anywhere in this kernel. (There used to be three: p1|p2,
  // p2|p3 and the p3|p4 that a p3 storing sumsq per chunk had already
  // retired.)
  mk_mhc_p1(a, blockIdx.x);
  MK_MHC_TS(7);
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
  const uint8_t* in_wq4;   // W4 pack of in_proj (see MKGemmCtx)
  const int8_t* in_ws4;
  float in_wgs = 1.0f;     // pack normalisation, see MKGemmCtx::wgs
  float o_wgs = 1.0f;
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
  const uint8_t* o_wq4;        // W4 pack of o_proj
  const int8_t* o_ws4;
  __nv_bfloat16* out;          // [T, HIDDEN]
  // workspace
  __nv_bfloat16* qkv;   // [MAX_TOK, KDA_INPROJ_N]
  __nv_bfloat16* g1;    // [MAX_TOK, KDA_OUT]
  __nv_bfloat16* g2;    // [MAX_TOK, KDA_OUT]
  __nv_bfloat16* convq; // [MAX_TOK, KDA_QKV]
  __nv_bfloat16* attn;  // [MAX_TOK, KDA_OUT]
  unsigned long long* barrier_ctr;
  int grid;          // resident blocks; see MK_GRID_CAP
  int ksr_in, ksr_out;  // mk_choose_ksr for the in_proj / o_proj phases
  int num_tokens;
  float lower_bound, onorm_eps;
};

__global__ void mk_kda_kernel(const MKKdaArgs a) {
  extern __shared__ uint8_t smem[];
  asm volatile("griddepcontrol.launch_dependents;");  // see mk_gemm_kernel
  MK_KDA_TS(0);

  {  // phase 0: in_proj GEMM into workspace
    MKGemmCtx c;
    c.x = a.x;
    c.wq4 = a.in_wq4;
    c.ws4 = a.in_ws4;
    c.wgs = a.in_wgs;
    c.out = a.qkv;
    c.m = a.num_tokens;
    c.n = KDA_INPROJ_N_PAD;
    c.k = HIDDEN;
    c.n_orig = KDA_INPROJ_N;
    c.grid = a.grid;
    c.ksr = a.ksr_in;
    mk_gemm_phase(c, smem, &g_mk_kda_bar);
  }
  MK_KDA_TS(1);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  MK_KDA_TS(2);

  {  // phase 1: f_b / g_b low-rank gates as a tensor-core GEMM.
    // [T x 128] @ [128 x 2048]^T twice (f, g). The weights (1 MB bf16, the
    // whole cost: ~4.5 us of DRAM) stream through cp.async as 32-row x
    // 128-k tiles (8 KB, double-buffered in the dynamic smem the GEMM
    // phases own), the two activation slices sit in smem once, and each
    // warp runs one (m16, n8) pair over 8 k16 steps of
    // mma.sync.m16n8k16 bf16 -> fp32. The warp-per-dot form before it
    // (one 256 B row per step, five shuffles per token) measured 14-18 us.
    constexpr int GT_ROWS = 32;                    // weight rows per tile
    constexpr int GT_PITCH = KDA_D + 8;            // bf16, 272 B: conflict-free
    constexpr int GT_BYTES = GT_ROWS * GT_PITCH * 2;  // 8704
    constexpr int GT_TILES = 2 * KDA_OUT / GT_ROWS;   // 128 (64 per gate)
    uint8_t* sx = smem;                            // [2][32][GT_PITCH] bf16
    uint8_t* sw = smem + 2 * GT_BYTES;             // [2][GT_ROWS][GT_PITCH]
    const int T = a.num_tokens;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    // activations: 2 gates x 32 rows x 16 chunks of 16 B; rows >= T zeroed
    for (int i = threadIdx.x; i < 2 * GT_ROWS * 16; i += MK_THREADS) {
      const int which = i / (GT_ROWS * 16), rem = i - which * (GT_ROWS * 16);
      const int t = rem >> 4, ck = rem & 15;
      uint8_t* dst = sx + which * GT_BYTES + t * (GT_PITCH * 2) + ck * 16;
      if (t < T)
        mk_cp_async16(dst, a.qkv + (size_t)t * KDA_INPROJ_N + KDA_QKV + KDA_H +
                               which * KDA_D + ck * 8);
      else
        *(uint4*)dst = make_uint4(0u, 0u, 0u, 0u);
    }
    mk_cp_commit();
    auto stage_tile = [&](int tile, int buf) {  // 32 rows x 256 B
      const int which = tile / (KDA_OUT / GT_ROWS);
      const int row0 = (tile - which * (KDA_OUT / GT_ROWS)) * GT_ROWS;
      const __nv_bfloat16* wsrc = (which ? a.g_b_w : a.f_b_w) +
                                  (size_t)row0 * KDA_D;
      uint8_t* dst = sw + buf * GT_BYTES;
      for (int i = threadIdx.x; i < GT_ROWS * 16; i += MK_THREADS) {
        const int r = i >> 4, ck = i & 15;
        mk_cp_async16(dst + r * (GT_PITCH * 2) + ck * 16,
                      wsrc + (size_t)r * KDA_D + ck * 8);
      }
      mk_cp_commit();
    };
    int tile = (int)blockIdx.x;
    if (tile < GT_TILES) stage_tile(tile, 0);
    for (int buf = 0; tile < GT_TILES; tile += a.grid, buf ^= 1) {
      const int next = tile + a.grid;
      if (next < GT_TILES) stage_tile(next, buf ^ 1);
      if (next < GT_TILES) mk_cp_wait<1>(); else mk_cp_wait<0>();
      __syncthreads();  // x + tile `buf` landed for every thread
      const int which = tile / (KDA_OUT / GT_ROWS);
      const int row0 = (tile - which * (KDA_OUT / GT_ROWS)) * GT_ROWS;
      const int n8 = warp & 3, m16 = warp >> 2;
      const uint8_t* xa = sx + which * GT_BYTES + (m16 * 16) * (GT_PITCH * 2);
      const uint8_t* wb = sw + buf * GT_BYTES + (n8 * 8) * (GT_PITCH * 2);
      float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
#pragma unroll
      for (int ks = 0; ks < KDA_D / 16; ++ks) {
        const int k0 = ks * 16 + q * 2;
        const uint32_t a0 = *(const uint32_t*)(xa + g * (GT_PITCH * 2) + k0 * 2);
        const uint32_t a1 =
            *(const uint32_t*)(xa + (g + 8) * (GT_PITCH * 2) + k0 * 2);
        const uint32_t a2 =
            *(const uint32_t*)(xa + g * (GT_PITCH * 2) + (k0 + 8) * 2);
        const uint32_t a3 =
            *(const uint32_t*)(xa + (g + 8) * (GT_PITCH * 2) + (k0 + 8) * 2);
        const uint32_t b0 = *(const uint32_t*)(wb + g * (GT_PITCH * 2) + k0 * 2);
        const uint32_t b1 =
            *(const uint32_t*)(wb + g * (GT_PITCH * 2) + (k0 + 8) * 2);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
      // c0,c1: (row g, cols 2q, 2q+1); c2,c3: row g + 8
      __nv_bfloat16* out = which ? a.g2 : a.g1;
      const int n = row0 + n8 * 8 + q * 2;
      const int t0 = m16 * 16 + g;
      if (t0 < T)
        *(__nv_bfloat162*)(out + (size_t)t0 * KDA_OUT + n) =
            __floats2bfloat162_rn(c0, c1);
      if (t0 + 8 < T)
        *(__nv_bfloat162*)(out + (size_t)(t0 + 8) * KDA_OUT + n) =
            __floats2bfloat162_rn(c2, c3);
      __syncthreads();  // buffer `buf` is refilled two steps from now
    }
  }
  MK_KDA_TS(3);
  // no barrier here: phase 2 (conv) reads only phase 0's qkv, like phase
  // 1 does, so a block goes straight from its gate dots to its conv
  // channels; the barrier that separated them cost ~4 us of wait per
  // launch on the stamps
  MK_KDA_TS(4);

  {  // phase 2: merged short conv (k=4, silu) with accepted-window rollback.
    // hist(pos) is the channel's input at in-request position pos (pos < 0
    // reads the slot's conv state). Outputs are produced for ALL query
    // tokens; the history starts at the accepted boundary and the window
    // written back keeps the accepted drafts (stock's spec conv kernel).
    // Everything a thread touches is indexed by compile-time constants:
    // the token loop is unrolled to the spec window (KDA_NQ_MAX = 8
    // tokens) with a guard, and the history / kept values
    // are selected, never indexed at runtime -- the old form (a lambda
    // over a runtime `pos`) put st[] and kept[] in local memory and the 8
    // token loads behind each other: 8 us for 6144 channels of ~40 FMAs.
    constexpr int NQ_MAX = KDA_NQ_MAX;  // 8 (the host refuses a wider mql)
    for (int r = 0; r < a.n_spec; ++r) {
      const int t0 = a.cu_seqlens[r], t1 = a.cu_seqlens[r + 1];
      const int slot = a.state_idx[r * a.mql + 0];
      const int acc = a.n_accepted[r];
      const int nq_tok = t1 - t0;
      constexpr int CPB = KDA_QKV / 48;  // 128 channels per block at grid 48
      for (int ch = blockIdx.x * CPB + threadIdx.x;
           ch < KDA_QKV && threadIdx.x < CPB; ch += a.grid * CPB) {
        const float* w = a.conv_w + (size_t)ch * CONV_W;
        const size_t sbase = (size_t)slot * KDA_QKV * a.conv_width +
                             (size_t)ch * a.conv_width;
        const int keep = a.conv_width - nq_tok;
        // history = state[acc-1 .. acc+1]; kept = state[acc .. acc+keep-1]
        // (keep <= 2 whenever nq_tok >= 8; guard the general case)
        float st[CONV_W - 1];
#pragma unroll
        for (int i = 0; i < CONV_W - 1; ++i)
          st[i] = a.conv_state[sbase + (acc - 1) + i];
        float kept[CONV_W - 1];
#pragma unroll
        for (int i = 0; i < CONV_W - 1; ++i)
          kept[i] = (i < keep && acc + i < a.conv_width)
                        ? a.conv_state[sbase + acc + i]
                        : 0.0f;
        float xin[NQ_MAX];
#pragma unroll
        for (int j = 0; j < NQ_MAX; ++j)
          xin[j] = (j < nq_tok)
                       ? __bfloat162float(
                             a.qkv[(size_t)(t0 + j) * KDA_INPROJ_N + ch])
                       : 0.0f;
        float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
        // h(pos) for pos in [-3, NQ_MAX): compile-time selection
        auto hist = [&](int pos) -> float {  // pos is a constant after unroll
          return pos < 0 ? st[pos + (CONV_W - 1)] : xin[pos];
        };
#pragma unroll
        for (int j = 0; j < NQ_MAX; ++j) {
          if (j < nq_tok) {
            const float v = w0 * hist(j - 3) + w1 * hist(j - 2) +
                            w2 * hist(j - 1) + w3 * hist(j);
            a.convq[(size_t)(t0 + j) * KDA_QKV + ch] =
                __float2bfloat16(v / (1.0f + expf(-v)));  // silu
          }
        }
        // Write the WHOLE window, matching causal_conv1d_update:
        //   [init[acc] .. init[acc + keep - 1], x[0] .. x[nq - 1]]
        for (int i = 0; i < a.conv_width; ++i) {
          float v;
          if (i < keep) {
            v = (i == 0) ? kept[0] : (i == 1) ? kept[1] : kept[2];
          } else {
            const int q = i - keep;  // 0 .. nq_tok - 1
            v = 0.0f;
#pragma unroll
            for (int j = 0; j < NQ_MAX; ++j) v = (q == j) ? xin[j] : v;
          }
          a.conv_state[sbase + i] = v;
        }
      }
    }
  }
  MK_KDA_TS(5);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  MK_KDA_TS(6);

  {  // phase 3: fine-grained gated delta rule. TWO blocks per head, each
    // owning 64 of the 128 state rows (v) with all of k: the update
    // S[v,k] += beta k[k] err[v], the error err[v] = v_in[v] - sum_k r[k]
    // S[v,k] and the readout out[v] = sum_k q[k] S[v,k] are all row-local,
    // so the rows split across blocks with no cross-block traffic; only
    // the 128-wide q, k, gate inputs are loaded by both. 4 threads per
    // row (k quarters, S[32] register-resident); the per-token chain is 5
    // syncs, shuffle-reduced norms, and the state store staged through
    // smem so it lands as coalesced 128 B rows (a store per token is the
    // stock contract: 8 MB per layer, ~35 us of DRAM at the floor).
    // History: one block per head with S[64] in LOCAL memory (partial
    // unroll) ran ~18 us/token; register-resident S ~8.5; this ~6.
    constexpr int RB = KDA_D / 2;   // rows per block
    constexpr int KQ = 4;           // k quarters per row
    constexpr int KW = KDA_D / KQ;  // 32 k per thread
    __shared__ float sh[3 * KDA_D + RB];  // y, q, k (128 each), err (RB)
    __shared__ float sred[KQ][RB];        // pre-update retrieval parts
    __shared__ float sred2[KQ][RB];       // post-update readout parts
    __shared__ float wsum[MK_WARPS][2];   // per-warp partial |q|^2, |k|^2
    float* y_s = sh;
    float* q_s = sh + KDA_D;
    float* k_s = sh + 2 * KDA_D;
    float* err_s = sh + 3 * KDA_D;

    const int head = blockIdx.x >> 1, rowhalf = blockIdx.x & 1;
    if (blockIdx.x < 2 * KDA_H) {
      const int vl = threadIdx.x & (RB - 1);     // local row 0..63
      const int v = rowhalf * RB + vl;           // state row
      const int kq = threadIdx.x >> 6;           // 0..3
      const int k0 = kq * KW;
      const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
      const bool dth = threadIdx.x < KDA_D;  // "d thread": owns dim d
      const int d = threadIdx.x;             // (valid when dth)
      const bool rth = kq == 0;              // "row thread": owns row v
      // The state multiplier is exp(gate), not the gate. Stock
      // fused_recurrent.py:129-130 computes
      //   b_gk = LOWER_BOUND / (1 + exp(-(a_log_amp * (g + bias))))
      //   b_h *= exp(b_gk)
      // The gate itself is in (lower_bound, 0) = (-5, 0), so using it raw
      // flipped the state's sign every token (rel_err 16 at acc=3, 1570 at
      // acc=8). Per-dim constants hoisted out of the token loop.
      const float alog = expf(a.a_log[head]);
      const float dtb = dth ? a.dt_bias[head * KDA_D + d] : 0.0f;
      // Match the stock kernel exactly (fused_recurrent.py:137-140):
      //   b_q = b_q / sqrt(sum(b_q*b_q) + 1e-6); b_k likewise; b_q *= scale
      // with scale = KDA_D ** -0.5 on q ONLY (kda.py:165). Without the
      // epsilon and the scale the readout ran sqrt(KDA_D) = 11.3x hot.
      constexpr float kda_qk_scale = 0.088388347648318447f;  // 128 ** -0.5

      // One token's inputs straight from global into registers (dims for
      // the d threads, the row's value input for the row threads); the
      // NEXT token's loads are issued at the top of each iteration so
      // their latency hides under this token's sync chain.
      auto load_tok = [&](int t, float& qd, float& kd, float& vd, float& gd,
                          float& bt) {
        const size_t cb = (size_t)t * KDA_QKV + head * KDA_D;
        if (dth) {
          qd = __bfloat162float(a.convq[cb + d]);
          kd = __bfloat162float(a.convq[cb + KDA_H * KDA_D + d]);
          gd = __bfloat162float(a.g1[(size_t)t * KDA_OUT + head * KDA_D + d]);
        }
        if (rth) vd = __bfloat162float(a.convq[cb + 2 * KDA_H * KDA_D + v]);
        bt = __bfloat162float(
            a.qkv[(size_t)t * KDA_INPROJ_N + KDA_QKV + head]);
      };

      for (int r = 0; r < a.n_spec; ++r) {
        const int t0 = a.cu_seqlens[r], t1 = a.cu_seqlens[r + 1];
        const int acc = a.n_accepted[r];
        // The state-index contract, from the stock kernel
        // (fused_recurrent.py, IS_SPEC_DECODING): the state to resume from
        // is the slot of the previous step's last ACCEPTED position,
        // ssm_state_indices[r, acc - 1]; after every token j the running
        // state is stored into ssm_state_indices[r, j] (a slot per query
        // position, so the next step can roll back by picking one); a
        // slot <= 0 is NULL_BLOCK_ID (padding) and the stock program
        // returns without outputs. This kernel used to read [r, 0] and
        // write only at j == acc - 1 -- the fixture (one slot for every
        // position) hid the per-position part and showed the rest as the
        // rec_state mismatch at acc < nq.
        const int slot0 = a.state_idx[r * a.mql + (acc - 1)];
        if (slot0 <= 0 || t1 <= t0) continue;
        float qd = 0.0f, kd = 0.0f, vd = 0.0f, gd = 0.0f, bt = 0.0f;
        load_tok(t0, qd, kd, vd, gd, bt);
        const float* Sbase =
            a.rec_state + (((size_t)slot0 * KDA_H + head) * KDA_D * KDA_D);
        // Element (v, k) lives at v * KDA_D + k, matching the stock
        // writer: fused_recurrent.py stores b_h as
        //   p_ht + o_v[:, None] * K + o_k[None, :]
        // (this kernel once held the transpose; the buffer is shared with
        // the stock path, so the layout is a contract.)
        // S must be REGISTER-resident: with a partial `unroll 8` the index
        // stayed dynamic and the array went to local memory (a 336 B stack
        // frame on the resource dump) -- ~190 L1 round trips per token per
        // thread. Full unrolls, no dynamic indexing anywhere below.
        float S[KW];
#pragma unroll
        for (int kk = 0; kk < KW; ++kk)
          S[kk] = Sbase[(size_t)v * KDA_D + (k0 + kk)];

        for (int j = 0; j < t1 - t0; ++j) {
          const int t = t0 + j;
          float nqd = 0.0f, nkd = 0.0f, nvd = 0.0f, ngd = 0.0f, nbt = 0.0f;
          if (j + 1 < t1 - t0) load_tok(t + 1, nqd, nkd, nvd, ngd, nbt);
          // (A) gate, and |q|^2 / |k|^2 by warp shuffle (warps 4..7 add 0)
          float q2 = 0.0f, k2 = 0.0f;
          if (dth) {
            y_s[d] = expf(a.lower_bound * mk_sigmoid(alog * (gd + dtb)));
            q2 = qd * qd;
            k2 = kd * kd;
          }
#pragma unroll
          for (int off = 16; off; off >>= 1) {
            q2 += __shfl_xor_sync(0xffffffffu, q2, off);
            k2 += __shfl_xor_sync(0xffffffffu, k2, off);
          }
          if (lane == 0) {
            wsum[warp][0] = q2;
            wsum[warp][1] = k2;
          }
          __syncthreads();  // (1) y_s, wsum
          const float nq = rsqrtf(wsum[0][0] + wsum[1][0] + wsum[2][0] +
                                  wsum[3][0] + 1e-6f) * kda_qk_scale;
          const float nk = rsqrtf(wsum[0][1] + wsum[1][1] + wsum[2][1] +
                                  wsum[3][1] + 1e-6f);
          if (dth) {
            q_s[d] = qd * nq;
            k_s[d] = kd * nk;
          }
          __syncthreads();  // (2) q_s, k_s

          const float beta = mk_sigmoid(bt);
          // retrieval operand: q (variants 0/2) or k (variant 1)
          const float* r_s = (a.delta_variant == 1) ? k_s : q_s;
          float part = 0.0f;
#pragma unroll
          for (int kk = 0; kk < KW; ++kk) {
            S[kk] *= y_s[k0 + kk];
            part += r_s[k0 + kk] * S[kk];
          }
          sred[kq][vl] = part;
          __syncthreads();  // (3) sred
          const float ret = sred[0][vl] + sred[1][vl] + sred[2][vl] +
                            sred[3][vl];
          if (rth) {
            err_s[vl] = vd - ret;
            if (a.delta_variant == 2)  // pre-update readout: the retrieval
              a.attn[(size_t)t * KDA_OUT + head * KDA_D + v] =
                  __float2bfloat16(ret);
          }
          __syncthreads();  // (4) err_s

          float part2 = 0.0f;
          const float e = err_s[vl];
#pragma unroll
          for (int kk = 0; kk < KW; ++kk) {
            S[kk] += beta * k_s[k0 + kk] * e;
            part2 += q_s[k0 + kk] * S[kk];
          }
          sred2[kq][vl] = part2;
          // this position's slot takes the state after token j: staged
          // through the (idle here) dynamic smem so the block's 32 KB
          // lands as coalesced 128 B rows (straight from the registers the
          // stores sit 512 B apart across a warp: 8x the sectors)
          const int sj = a.state_idx[r * a.mql + j];  // block-uniform
          float* stg = (float*)smem;  // [RB][KDA_D + 1]
          if (sj > 0) {
#pragma unroll
            for (int kk = 0; kk < KW; ++kk)
              stg[vl * (KDA_D + 1) + (k0 + kk)] = S[kk];
          }
          __syncthreads();  // (5) sred2, stg
          if (rth && a.delta_variant != 2)
            a.attn[(size_t)t * KDA_OUT + head * KDA_D + v] = __float2bfloat16(
                sred2[0][vl] + sred2[1][vl] + sred2[2][vl] + sred2[3][vl]);
          if (sj > 0) {
            float* Sj = a.rec_state +
                        (((size_t)sj * KDA_H + head) * KDA_D * KDA_D) +
                        (size_t)rowhalf * RB * KDA_D;
            for (int idx = threadIdx.x; idx < RB * KDA_D; idx += MK_THREADS)
              Sj[idx] = stg[(idx >> 7) * (KDA_D + 1) + (idx & (KDA_D - 1))];
          }
          // No trailing sync: every buffer the next token writes (y_s,
          // wsum, q_s, k_s, sred, err_s, sred2, stg) is written only after
          // a sync that follows this token's last read of it.
          qd = nqd; kd = nkd; vd = nvd; gd = ngd; bt = nbt;
        }
      }
    }
  }
  MK_KDA_TS(7);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  MK_KDA_TS(8);

  {  // phase 4: gated RMSNorm -- rmsnorm(attn) * sigmoid(g2) -- emitted
    // straight as the o_proj GEMM's fp8 A tiles. One WARP per (token,
    // head): lane l holds dims 4l..4l+3, the sum of squares and the group
    // amax are shuffle reductions, and a head's 128 dims are exactly one
    // 128-wide k-group of the o_proj (KDA_OUT = 16 x 128), so the warp
    // writes g_mk_aq[(kb = head) * 32 + t] and the pow2 scale directly --
    // the same bytes the GEMM prologue would have produced from a bf16
    // attn, minus that prologue (x load, quant, publishing barrier).
    // Was: a whole block per pair, two syncs, half the threads idle.
    const int pairs = a.num_tokens * KDA_H;
    const int lane = threadIdx.x & 31;
    for (int i = blockIdx.x * MK_WARPS + (threadIdx.x >> 5); i < pairs;
         i += a.grid * MK_WARPS) {
      const int t = i / KDA_H, h = i - t * KDA_H;
      const size_t base = (size_t)t * KDA_OUT + h * KDA_D + lane * 4;
      const uint2 ar = *(const uint2*)(a.attn + base);
      const uint2 gr = *(const uint2*)(a.g2 + base);
      const uint2 wr = *(const uint2*)(a.onorm_w + lane * 4);
      const __nv_bfloat16* ap = (const __nv_bfloat16*)&ar;
      const __nv_bfloat16* gp = (const __nv_bfloat16*)&gr;
      const __nv_bfloat16* wp = (const __nv_bfloat16*)&wr;
      float x[4], sq = 0.0f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        x[q] = __bfloat162float(ap[q]);
        sq += x[q] * x[q];
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) sq += __shfl_xor_sync(~0u, sq, off);
      const float inv = rsqrtf(sq / (float)KDA_D + a.onorm_eps);
      float amax = 0.0f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        // bf16 round first: the GEMM prologue quantized the bf16 attn the
        // old p4 stored, so the fp8 bytes stay what they were
        x[q] = __bfloat162float(__float2bfloat16(
            x[q] * inv * __bfloat162float(wp[q]) *
            mk_sigmoid(__bfloat162float(gp[q]))));
        amax = fmaxf(amax, fabsf(x[q]));
      }
#pragma unroll
      for (int off = 16; off; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, off));
      const float sc = mk_pow2_scale(amax);
      const float rsc = 1.0f / sc;  // exact: sc is a power of two
      uint32_t pack = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        pack |= (uint32_t)mk_f32_to_e4m3(x[q] * rsc) << (8 * q);
      *(uint32_t*)(g_mk_aq + ((size_t)h * 32 + t) * KSTEP + lane * 4) = pack;
      if (lane == 0) g_mk_axs[t * KBLK_MAX + h] = sc;
    }
  }
  // p5's dynamic unit hand-out starts from 0; the prologue that used to
  // reset the counter is skipped (a_ready), so reset it under THIS barrier
  if (blockIdx.x == 0 && threadIdx.x == 0) g_mk_unit_next = 0u;
  MK_KDA_TS(9);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  MK_KDA_TS(10);

  {  // phase 5: o_proj GEMM on the A tiles p4 emitted
    MKGemmCtx c;
    c.x = a.attn;  // unused on the a_ready path (documentation only)
    c.a_ready = true;
    c.wq4 = a.o_wq4;
    c.ws4 = a.o_ws4;
    c.wgs = a.o_wgs;
    c.out = a.out;
    c.m = a.num_tokens;
    c.n = HIDDEN;
    c.k = KDA_OUT_PAD;
    c.n_orig = HIDDEN;
    c.grid = a.grid;
    c.ksr = a.ksr_out;
    mk_gemm_phase(c, smem, &g_mk_kda_bar);
  }
  MK_KDA_TS(11);
}

}  // namespace


// ---------------------------------------------------------------- MK_SEG_MLA
// Sparse MLA decode for GLM-5-Next's NoPE shape, replacing FlashInfer's
// BatchMLAPagedAttentionWrapper (the "SM90" FA2 path vLLM picks on sm_12x).
//
// Per query row t and head h, over the indexer's top-k slots of that row:
//     s_j        = sm_scale * dot(q[t,h,:], c[slot_j,:]) * ckv_scale
//     out[t,h,:] = sum_j softmax(s)_j * c[slot_j,:] * ckv_scale
// k and v are the SAME compressed latent (D = kv_lora_rank = 512, one fp8
// scale per layer), so a slot's 512 bytes are read ONCE and serve all H
// heads. That sharing sets the cost model this kernel is shaped around.
//
// GB10 cost model (why it looks like this):
//   * per (row, slot): 512 B read, 16 heads x 512 x 2 = 32,768 flop
//     -> 64 flop/byte. The ledger's sustained stream on this part is
//     135-175 GB/s (225 only on a process's first burst), so the kernel
//     needs ~11 TFLOP/s to stay memory-bound. FP32 FMA peak here is
//     ~19.7 TFLOP/s (48 SM x 128 lanes x 2 x ~1.6 GHz), so PLAIN FMA
//     clears it -- and keeps q in bf16, which the sparse backend requires
//     (no query quantization). Tensor cores would force q to e4m3 or an
//     extra bf16 materialisation of C in smem; neither buys anything
//     against a memory bound.
//   * 512 B / 32 lanes = 16 B per lane: one cp.async.cg per lane per slot,
//     perfectly coalesced, L1-bypassing (the latents are read once and
//     never reused -- keeping them out of L1 leaves it to the q rows).
//   * per-SM bandwidth share is ~3.6 GB/s, so ~2.2 KB in flight covers the
//     ~600 ns DRAM latency; TILE=16 slots x NSTAGE=3 = 24 KB is an order of
//     magnitude past that and still leaves smem for 3 blocks/SM.
//   * q lives in REGISTERS (2 heads x 16 lanes-worth = 32 values), never
//     re-read from smem inside the slot loop.
//   * 48 SMs with T as small as 8 rows: the slot axis is split so
//     T x splits fills the persistent grid, then combined by log-sum-exp
//     behind the same monotonic-ticket barrier the other segments use.
//
// MEASURED STATE (2026-09-03, srv4, W=2048, 300k-slot cache so every read
// misses L2). NOT adopted -- the kernel is correct but slower than the
// wrapper it would replace, and nothing routes to it yet:
//   T= 8: MK 205 us vs FlashInfer run 124 us   (per layer, per step)
//   T=16: MK 317 us vs 194
//   T=32: MK 579 us vs 542
// Roofline probe (VLLM_GLM53_MK_MLA_PROBE): streaming alone 68-91 us
// (92-124 GB/s, i.e. the box's real band), + dot/shuffle 68, + softmax and
// output 52 -- ADDITIVE, so the cross-lane reduction in the score phase is
// the wall, not memory. The fix is to stop reducing across lanes: give the
// dot a lane-per-SLOT layout (no shuffles, unlimited ILP), pass p through
// smem, and keep the lane-per-D layout only for the output accumulation.
// Tile/stage sweep found 32x2 (32 KB) best; 64 KB drops to 1 block/SM and
// costs T=32 65%.
//
// The host-side plan() this was meant to remove is ~38 us/step, not the
// 2.4 ms an earlier synthetic measurement suggested (that built a wrapper
// without reserved buffers). So the whole lever here is ~2% of a step, and
// the kernel has to beat the wrapper outright to be worth arming.
//
// The point of owning this kernel is not the 0.66 ms it spends on the GPU.
// FlashInfer's wrapper bakes each row's kv_len into a HOST-side schedule at
// plan() time and never reads the device-side length buffer, so vLLM's
// builder replans every step outside graph capture -- 2.4 ms of host time
// per step, measured. This kernel reads the row length from device memory,
// so it runs inside the captured graph and that replan disappears with it.
constexpr int MLA_D = 512;                    // kv_lora_rank
constexpr int MLA_H = 16;                     // MLA heads per rank at TP4
constexpr int MLA_WARPS = MK_THREADS / 32;    // 8
constexpr int MLA_HPW = MLA_H / MLA_WARPS;    // 2 heads per warp (softmax owner)
constexpr int MLA_VD = MLA_D / 32;            // 16 latent elements per lane (phase 1)
constexpr int MLA_TILE = 16;   // slots per cp.async tile (2 n-groups x 8)
constexpr int MLA_NSTAGE = 3;  // ring buffers; total smem ~45 KB -> 2 blocks/SM
constexpr int MLA_SPLITS_MAX = 64;

struct MKMlaArgs {
  const __nv_bfloat16* q;      // [T, H, D] bf16, never quantised
  const uint8_t* ckv;          // e4m3 [num_slots, D]
  const int* slots;            // [T, W] global slot ids, valid prefix first
  const int* lens;             // [T] valid count per row -- DEVICE side
  __nv_bfloat16* out;          // [T, H, D]
  float* part;                 // [T, splits, H, D] fp32 split partials
  float* pml;                  // [T, splits, H, 2] running (m, l)
  unsigned long long* barrier_ctr;
  float sm_scale;
  float ckv_scale;
  int T, W, splits, grid;
  int probe;   // 1 = memory pipeline only (roofline), 0 = full
};

// v4: tensor cores, and the e4m3 ring is the ONLY copy of the latent.
//
// v3 materialised a bf16 copy of each tile in shared memory so both mma
// phases could take fragments from it (the output phase with
// ldmatrix.trans). That copy cost 33 KB of the 99 KB budget, which pinned
// the kernel at ONE block per SM -- and a gather microbenchmark on this part
// (scattered 512 B chunks, 1 GiB buffer) reaches 225 GB/s against the 106
// GB/s the kernel was getting, i.e. the access pattern was never the
// problem, occupancy was. v4 converts e4m3 -> bf16 in registers at fragment
// load time, drops the tile, and halves TILE so two blocks fit per SM.
//
// Layout per 16-slot tile:
//   scores  S = Q C^T  -- warp w owns n-group (w & 1) of 8 slots and
//                         k-quarter (w >> 1) of D; 4 partials summed in smem
//   softmax -- warp w owns heads 2w, 2w+1, lane = slot (10 shuffles/tile)
//   output  O += P C   -- warp w owns 64 of the 512 columns, B fragments
//                         assembled from 4 single-byte ring reads each
constexpr int MLA_KQ = 4;                       // k-quarters in the score mma
constexpr int MLA_NG = MLA_WARPS / MLA_KQ;      // 2 n-groups of 8 slots -> TILE 16
constexpr int MLA_CP = MLA_D + 8;               // q pitch (words), bank-skewed
constexpr int MLA_PP = MLA_TILE + 8;            // P pitch
constexpr int MLA_RP = MLA_D + 16;              // e4m3 ring row pitch (bytes)
constexpr int MLA_SMEM_RING = MLA_NSTAGE * MLA_TILE * MLA_RP;
constexpr int MLA_SMEM_Q = MLA_H * MLA_CP * 2;
constexpr int MLA_SMEM_S = MLA_KQ * MLA_H * MLA_TILE * 4;
constexpr int MLA_SMEM_P = MLA_H * MLA_PP * 2;
constexpr int MLA_SMEM_C = MLA_H * 8;   // [H] corr, then [H] l
constexpr int MLA_SMEM = MLA_SMEM_RING + MLA_SMEM_Q + MLA_SMEM_S + MLA_SMEM_P + MLA_SMEM_C;

__device__ __forceinline__ float mla_warp_max(float v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  return v;
}
__device__ __forceinline__ float mla_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}
// two adjacent e4m3 bytes -> one bf16x2 register (an mma fragment half)
__device__ __forceinline__ uint32_t mla_e4m3x2(const uint8_t* p) {
  const __half2 h = __nv_cvt_fp8x2_to_halfraw2(*(const __nv_fp8x2_storage_t*)p, __NV_E4M3);
  const __nv_bfloat162 b = __float22bfloat162_rn(__half22float2(h));
  return *(const uint32_t*)&b;
}
// two e4m3 bytes at a stride (column of the ring) -> one bf16x2 register
__device__ __forceinline__ uint32_t mla_e4m3x2_strided(const uint8_t* p, int stride) {
  const __half2 h = __halves2half2(__nv_cvt_fp8_to_halfraw(p[0], __NV_E4M3),
                                   __nv_cvt_fp8_to_halfraw(p[stride], __NV_E4M3));
  const __nv_bfloat162 b = __float22bfloat162_rn(__half22float2(h));
  return *(const uint32_t*)&b;
}
__device__ __forceinline__ void mla_mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ __launch_bounds__(MK_THREADS) void mk_mla_kernel(const MKMlaArgs a) {
  extern __shared__ __align__(16) char mla_smem[];
  uint8_t* ring = (uint8_t*)mla_smem;
  __nv_bfloat16* sq = (__nv_bfloat16*)(ring + MLA_SMEM_RING);
  float* ss = (float*)((uint8_t*)sq + MLA_SMEM_Q);
  __nv_bfloat16* sp = (__nv_bfloat16*)((uint8_t*)ss + MLA_SMEM_S);
  float* scorr = (float*)((uint8_t*)sp + MLA_SMEM_P);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2, q4 = lane & 3;
  asm volatile("griddepcontrol.launch_dependents;");  // see mk_gemm_kernel
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const float sscale = a.sm_scale * a.ckv_scale;

  const int items = a.T * a.splits;
  for (int item = blockIdx.x; item < items; item += a.grid) {
    const int t = item / a.splits;
    const int sp_i = item - t * a.splits;
    const int len = a.lens[t];
    const int per = (len + a.splits - 1) / a.splits;
    const int j0 = min(len, sp_i * per);
    const int j1 = min(len, j0 + per);

    for (int i = threadIdx.x; i < MLA_H * (MLA_D / 8); i += MK_THREADS) {
      const int h = i / (MLA_D / 8), c8 = i - h * (MLA_D / 8);
      *(uint4*)(sq + h * MLA_CP + c8 * 8) =
          *(const uint4*)(a.q + ((size_t)t * MLA_H + h) * MLA_D + c8 * 8);
    }
    float acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int ntile = (j1 > j0) ? ((j1 - j0 + MLA_TILE - 1) / MLA_TILE) : 0;
    auto issue = [&](int ti) {
      uint8_t* dst = ring + (size_t)(ti % MLA_NSTAGE) * MLA_TILE * MLA_RP;
#pragma unroll
      for (int r = 0; r < MLA_TILE / MLA_WARPS; ++r) {
        const int k = r * MLA_WARPS + warp;
        const int j = j0 + ti * MLA_TILE + k;
        const int sj = (j < j1) ? j : j0;
        const int slot = a.slots[(size_t)t * a.W + sj];
        mk_cp_async16(dst + (size_t)k * MLA_RP + lane * 16,
                      a.ckv + (size_t)slot * MLA_D + lane * 16);
      }
      mk_cp_commit();
    };
#pragma unroll 1
    for (int ti = 0; ti < MLA_NSTAGE - 1; ++ti) { if (ti < ntile) issue(ti); else mk_cp_commit(); }

#pragma unroll 1
    for (int ti = 0; ti < ntile; ++ti) {
      mk_cp_wait<MLA_NSTAGE - 2>();
      __syncthreads();
      if (ti + MLA_NSTAGE - 1 < ntile) issue(ti + MLA_NSTAGE - 1);
      else mk_cp_commit();
      const uint8_t* tile8 = ring + (size_t)(ti % MLA_NSTAGE) * MLA_TILE * MLA_RP;
      const int kmax = min(MLA_TILE, j1 - (j0 + ti * MLA_TILE));

      {  // ---- S = Q C^T, B fragments converted from the ring in registers
        const int n0 = (warp % MLA_NG) * 8, kq = warp / MLA_NG;
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        const __nv_bfloat16* qa = sq + g * MLA_CP;
        const uint8_t* cb = tile8 + (size_t)(n0 + g) * MLA_RP;
#pragma unroll
        for (int ks = 0; ks < (MLA_D / 16) / MLA_KQ; ++ks) {
          const int k0 = kq * (MLA_D / MLA_KQ) + ks * 16 + q4 * 2;
          mla_mma_bf16(c0, c1, c2, c3,
                       *(const uint32_t*)(qa + k0),
                       *(const uint32_t*)(qa + 8 * MLA_CP + k0),
                       *(const uint32_t*)(qa + k0 + 8),
                       *(const uint32_t*)(qa + 8 * MLA_CP + k0 + 8),
                       mla_e4m3x2(cb + k0), mla_e4m3x2(cb + k0 + 8));
        }
        float* sh = ss + (size_t)kq * MLA_H * MLA_TILE;
        sh[g * MLA_TILE + n0 + q4 * 2] = c0;
        sh[g * MLA_TILE + n0 + q4 * 2 + 1] = c1;
        sh[(g + 8) * MLA_TILE + n0 + q4 * 2] = c2;
        sh[(g + 8) * MLA_TILE + n0 + q4 * 2 + 1] = c3;
      }
      __syncthreads();

      {  // ---- online softmax for heads 2w, 2w+1 (lane = slot)
        const int h0 = warp * 2;
        const bool ok = lane < kmax;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int kq = 0; kq < MLA_KQ; ++kq) {
          const float* sh = ss + (size_t)kq * MLA_H * MLA_TILE;
          s0 += sh[h0 * MLA_TILE + (lane % MLA_TILE)];
          s1 += sh[(h0 + 1) * MLA_TILE + (lane % MLA_TILE)];
        }
        s0 = ok ? s0 * sscale : -INFINITY;
        s1 = ok ? s1 * sscale : -INFINITY;
        const float n0 = fmaxf(m0, mla_warp_max(s0));
        const float n1 = fmaxf(m1, mla_warp_max(s1));
        const float cr0 = __expf(m0 - n0), cr1 = __expf(m1 - n1);
        const float p0 = ok ? __expf(s0 - n0) : 0.f;
        const float p1 = ok ? __expf(s1 - n1) : 0.f;
        l0 = fmaf(l0, cr0, mla_warp_sum(p0));
        l1 = fmaf(l1, cr1, mla_warp_sum(p1));
        m0 = n0; m1 = n1;
        if (lane < MLA_TILE) {
          sp[h0 * MLA_PP + lane] = __float2bfloat16(p0 * a.ckv_scale);
          sp[(h0 + 1) * MLA_PP + lane] = __float2bfloat16(p1 * a.ckv_scale);
        }
        if (lane == 0) { scorr[h0] = cr0; scorr[h0 + 1] = cr1; }
      }
      __syncthreads();

      {  // ---- O += P C : 64 columns per warp, B fragments straight from the ring
        const float crg = scorr[g], crg8 = scorr[g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          acc[nt][0] *= crg; acc[nt][1] *= crg; acc[nt][2] *= crg8; acc[nt][3] *= crg8;
        }
        const __nv_bfloat16* pa = sp + g * MLA_PP;
        const uint32_t a0 = *(const uint32_t*)(pa + q4 * 2);
        const uint32_t a1 = *(const uint32_t*)(pa + 8 * MLA_PP + q4 * 2);
        const uint32_t a2 = *(const uint32_t*)(pa + q4 * 2 + 8);
        const uint32_t a3 = *(const uint32_t*)(pa + 8 * MLA_PP + q4 * 2 + 8);
        const int krow = q4 * 2;                     // this lane's two k (slot) rows
        const uint8_t* cb = tile8 + (size_t)krow * MLA_RP + warp * 64;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int n = nt * 8 + g;                  // this lane's output column
          mla_mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3,
                       mla_e4m3x2_strided(cb + n, MLA_RP),
                       mla_e4m3x2_strided(cb + 8 * MLA_RP + n, MLA_RP));
        }
      }
      __syncthreads();
    }

    if (a.splits == 1) {
      // v5 (prefill): one split owns the whole row, so there is nothing for
      // the log-sum-exp phase to combine -- normalise and store bf16 here.
      // Without this a prefill call would need a [T][splits][H][D] fp32
      // scratch: 268 MB at T = 8192.
      //
      // The softmax state lives on the warp that owns heads 2w, 2w+1, while
      // acc rows are heads (g, g+8) of the mma C layout, so the denominator
      // goes through smem first.
      float* sl = scorr + MLA_H;
      if (lane == 0) { sl[warp * 2] = l0; sl[warp * 2 + 1] = l1; }
      __syncthreads();
      const float ig = (sl[g] > 0.f) ? __frcp_rn(sl[g]) : 0.f;
      const float ig8 = (sl[g + 8] > 0.f) ? __frcp_rn(sl[g + 8]) : 0.f;
      __nv_bfloat16* o0 = a.out + ((size_t)t * MLA_H + g) * MLA_D;
      __nv_bfloat16* o8 = a.out + ((size_t)t * MLA_H + g + 8) * MLA_D;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = warp * 64 + nt * 8 + q4 * 2;
        *(__nv_bfloat162*)(o0 + col) = __floats2bfloat162_rn(acc[nt][0] * ig, acc[nt][1] * ig);
        *(__nv_bfloat162*)(o8 + col) = __floats2bfloat162_rn(acc[nt][2] * ig8, acc[nt][3] * ig8);
      }
      __syncthreads();   // sl is rewritten by the next item
    } else {
      float* base = a.part + (((size_t)t * a.splits + sp_i) * MLA_H) * MLA_D;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = warp * 64 + nt * 8 + q4 * 2;
        *(float2*)(base + (size_t)g * MLA_D + col) = make_float2(acc[nt][0], acc[nt][1]);
        *(float2*)(base + (size_t)(g + 8) * MLA_D + col) = make_float2(acc[nt][2], acc[nt][3]);
      }
      if (lane == 0) {
        float* ml = a.pml + (((size_t)t * a.splits + sp_i) * MLA_H + warp * 2) * 2;
        ml[0] = (ntile > 0) ? m0 : -INFINITY;  ml[1] = (ntile > 0) ? l0 : 0.f;
        ml[2] = (ntile > 0) ? m1 : -INFINITY;  ml[3] = (ntile > 0) ? l1 : 0.f;
      }
    }
  }

  if (a.splits == 1) return;   // v5: phase 0 already normalised and stored

  mk_grid_barrier(a.barrier_ctr, a.grid);

  // phase 1 -- log-sum-exp combine, one (row, head) per warp
  const int pairs = a.T * MLA_H;
#pragma unroll 1
  for (int p = blockIdx.x * MLA_WARPS + warp; p < pairs; p += a.grid * MLA_WARPS) {
    const int t = p / MLA_H;
    const int h = p - t * MLA_H;
    float mm = -INFINITY;
    for (int sp = 0; sp < a.splits; ++sp)
      mm = fmaxf(mm, a.pml[(((size_t)t * a.splits + sp) * MLA_H + h) * 2]);
    float ltot = 0.f, o[MLA_VD];
#pragma unroll
    for (int e = 0; e < MLA_VD; ++e) o[e] = 0.f;
    for (int sp = 0; sp < a.splits; ++sp) {
      const float* ml = a.pml + (((size_t)t * a.splits + sp) * MLA_H + h) * 2;
      const float lsp = ml[1];
      if (!(lsp > 0.f)) continue;
      const float w = __expf(ml[0] - mm);
      ltot = fmaf(lsp, w, ltot);
      const float* src = a.part + (((size_t)t * a.splits + sp) * MLA_H + h) * MLA_D
                        + lane * MLA_VD;
#pragma unroll
      for (int e = 0; e < MLA_VD; ++e) o[e] = fmaf(src[e], w, o[e]);
    }
    const float inv = (ltot > 0.f) ? __frcp_rn(ltot) : 0.f;
    __nv_bfloat16* dst = a.out + ((size_t)t * MLA_H + h) * MLA_D + lane * MLA_VD;
#pragma unroll
    for (int e = 0; e < MLA_VD; ++e) dst[e] = __float2bfloat16(o[e] * inv);
  }
}

// ---------------------------------------------------------------------------
// host entry points
// ---------------------------------------------------------------------------
namespace {

bool g_attrs_set = false;
// resident blocks per SM the device reports for mk_gemm2_kernel (2 by
// construction of GEMM2_SMEM; the v2 unit rule sizes its grid from it)
int g_gemm2_bps = 0;
int g_mk_sms = 0;  // multiprocessors, from the device (48 on GB10)

void set_kernel_attrs() {
  if (g_attrs_set) return;
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      GEMM_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      GEMM2_SMEM));
  // not a residency contract (v2 has no barrier): the unit rule wants
  // to know how many blocks share an SM
  MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &g_gemm2_bps, mk_gemm2_kernel, MK_THREADS, GEMM2_SMEM));
  if (g_gemm2_bps < 1) g_gemm2_bps = 1;
  MK_CHECK_CUDA(cudaDeviceGetAttribute(
      &g_mk_sms, cudaDevAttrMultiProcessorCount, 0));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_SMEM));
  g_attrs_set = true;
}

// Resident block count for a persistent grid, clamped to MK_GRID_CAP.
// Cached per kernel: the ticket barrier needs the SAME grid on every launch,
// and the two kernels are asked separately because their occupancy differs.
template <typename K>
int mk_resident_grid(K kernel, int& cache, int smem, int cap = MK_GRID_CAP) {
  if (cache == 0) {
    int per_sm = 0, sms = 0;
    MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, kernel, MK_THREADS, smem));
    MK_CHECK_CUDA(cudaDeviceGetAttribute(
        &sms, cudaDevAttrMultiProcessorCount, 0));
    cache = per_sm * sms;
    if (cache > cap) cache = cap;
    TORCH_CHECK(cache > 0, "persistent grid has no resident blocks");
  }
  return cache;
}

int g_gemm_grid = 0;
int g_kda_grid = 0;
int g_mla_grid = 0;
// MK_SEG_MLA keeps its own ticket counter, so it is not bound by the shared
// 96-block contract the gemm/kda/mhc kernels observe. It still measures 96
// (2 blocks/SM at 45.8 KB smem); the cap is here so a future smem cut is
// not silently thrown away. Moving q to registers frees the smem for a
// third block but pushes registers past the limit -- measured 2 either
// way, and 6-8% SLOWER, so q stays in shared memory.
constexpr int MLA_GRID_CAP = 192;

// VLLM_GLM53_MK_PDL=1: launch with programmatic stream serialization so
// each MK kernel may begin on the SMs its predecessor frees and prefetch
// its weights during the predecessor's tail (the kernels trigger at entry
// and wait before their first dependent read). Default off: the serving
// profile flips it after its bracket, the probe sets it.
bool mk_pdl_enabled() {
  static int v = -1;
  if (v < 0) {
    const char* e = getenv("VLLM_GLM53_MK_PDL");
    v = (e != nullptr && e[0] == '1') ? 1 : 0;
  }
  return v == 1;
}

template <typename K, typename A>
void mk_launch(K kernel, int grid, int smem, cudaStream_t stream,
               const A& args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(MK_THREADS);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = mk_pdl_enabled() ? 1 : 0;
  MK_CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, args));
}

// Leftover-tile k-split for one launch shape. Every block owns one
// 128-column tile across all of k, so a tile count that is not a multiple
// of the grid pays a whole extra round for its leftovers; those get their
// k split ksr ways instead. grid / rem truncates to 1 for nblk in
// (grid/2, grid] -- at grid 48 that is n/128 == 32, the 4096-wide
// projections, which then ran 32 tiles on 48 blocks with 16 idle. When
// full == 0 every unit is the same size, so wall time is exactly
// ceil(nblk*r / grid) / r tile-times and the best r is worth searching
// for: r=3 costs 0.667 where r=1 costs 1.0. (The full > 0 case is left
// alone -- there the units are a MIX of whole tiles and k-slices and this
// model does not hold.) Host-side: this used to run in every block.
int mk_choose_ksr(int m, int n, int k, int grid) {
  const int nblk = n / 128, kblk = k / KSTEP;
  const int rem = nblk % grid, full = nblk - rem;
  int ksr = (rem > 0) ? (grid / rem) : 1;
  if (full == 0 && rem > 0 && m <= 32) {
    int bn = (nblk + grid - 1) / grid, bd = 1;
    // A slice shorter than ~8 k-blocks never gets its cp.async pipeline
    // going (fill latency, then a handful of iterations), so the "fewer
    // rounds" the model counts are not shorter in wall time. Measured on
    // the kda o_proj (k = 2048, 16 k-blocks): r=3 (the model) 40.8 us,
    // r=2 33.4, r=1 35.5, r=4 43.8.
    const int rmax = kblk / 8 > 1 ? kblk / 8 : 1;
    for (int r = 2; r <= kblk && r <= rmax; ++r) {
      if ((size_t)m * rem * 128 * r > MK_SPLIT_ELEMS) break;
      const int rounds = (nblk * r + grid - 1) / grid;
      if (rounds * bd < bn * r) { bn = rounds; bd = r; ksr = r; }
    }
  }
  {  // probe knob: force the standalone lane's split (0 = the cost model).
     // Bench only -- the shapes the model refuses to split (k-blocks <= 8:
     // the shared expert's down [4096 x 512] runs 32 tiles on 48 blocks
     // with 16 idle) are the 30-45 us class of 28차, and the "no slice
     // under 8 k-blocks" rule was measured at k = 2048, not there.
    static int f = -1;
    if (f < 0) { const char* e = getenv("VLLM_GLM53_MK_KSR"); f = e ? atoi(e) : 0; }
    if (f > 0 && full == 0 && rem > 0 && m <= 32) ksr = f < kblk ? f : kblk;
  }
  return ksr;
}

// MK-GEMM v2 lane (mk_gemm2_kernel). VLLM_GLM53_MK_GEMM2=1 routes every
// standalone launch through it; 0 (default) is the persistent kernel, byte
// for byte the lane before it. The bench flips it per call (set_gemm2).
int g_probe_gemm2 = -1;
int g_probe_ksr2 = -1;  // 0 = the rule below; > 0 forces the slice count
bool mk_gemm2_on() {
  if (g_probe_gemm2 < 0) {
    const char* e = getenv("VLLM_GLM53_MK_GEMM2");
    g_probe_gemm2 = (e != nullptr && e[0] == '1') ? 1 : 0;
  }
  return g_probe_gemm2 == 1;
}
// k-slices per tile for one v2 launch, from the 30차 sweeps (srv2, ksr 1/2/3/
// 4/6/8 on every production shape, single and back-to-back): what wins is
// ONE exact wave of the resident slots (blocks/SM x SMs = 96 on GB10) --
// 48 tiles x 2, 32 x 3, 16 x 6 measured best on every k -- or, when the
// tile count does not divide the slots, the longest slices that still fit
// one wave (8 tiles x 8 = 64 units); a small SECOND wave is the worst case
// (51 tiles x 2 = 102 units: +30 us for the six stragglers), so a tile
// count above half the slots takes the finest slices instead (51 x 8 = 408,
// 86 us; x 3 = 153, 87; x 1 = 51, 97). No slice shorter than 4 k-blocks (a
// unit's ring fill and first quant are paid per unit; k = 512 stays whole).
// Then the contract clamps: ksr <= kblk (every slice non-empty), ksr <=
// MK2_KSR_MAX, and the fp32 partial must fit (m x n x ksr floats).
int mk_choose_ksr2(int m, int n, int k) {
  const int nblk = n / SMEM_W_ROWS, kblk = k / KSTEP;
  if (g_probe_ksr2 < 0) {
    const char* e = getenv("VLLM_GLM53_MK_KSR2");
    g_probe_ksr2 = e ? atoi(e) : 0;
  }
  int ksr;
  if (g_probe_ksr2 > 0) {
    ksr = g_probe_ksr2;
  } else {
    const int slots = (g_gemm2_bps > 0 ? g_gemm2_bps : 2) *
                      (g_mk_sms > 0 ? g_mk_sms : 48);
    const int kmax = kblk / 4 > 1 ? kblk / 4 : 1;
    if (slots % nblk == 0 && slots / nblk <= kmax) {
      ksr = slots / nblk;               // one exact wave
    } else if (nblk * 2 > slots) {
      ksr = kmax;                       // would leave a short second wave: slice fine
    } else {
      ksr = slots / nblk;               // under one wave, the longest slices
      if (ksr > kmax) ksr = kmax;
    }
  }
  if (ksr < 1) ksr = 1;
  if (ksr > kblk) ksr = kblk;
  if (ksr > MK2_KSR_MAX) ksr = MK2_KSR_MAX;
  while (ksr > 1 && (size_t)m * n * ksr > (size_t)MK2_PART_ELEMS) --ksr;
  return ksr;
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

#ifdef MK_PHASE_TS
// Read a device stamp array into a host vector and clear it (every probe
// reader below): synchronize first -- the stamps are written by the launch
// the caller just issued.
template <size_t N>
std::vector<int64_t> mk_read_and_clear(unsigned long long (&sym)[N]) {
  std::vector<unsigned long long> h(N);
  MK_CHECK_CUDA(cudaDeviceSynchronize());
  MK_CHECK_CUDA(cudaMemcpyFromSymbol(h.data(), sym, sizeof(unsigned long long) * N));
  void* p = nullptr;
  MK_CHECK_CUDA(cudaGetSymbolAddress(&p, sym));
  MK_CHECK_CUDA(cudaMemset(p, 0, sizeof(unsigned long long) * N));
  return std::vector<int64_t>(h.begin(), h.end());
}
#endif

// Phase timestamps of the last gemm launch, [MK_GRID_CAP][8] ns, then
// cleared. Empty unless built with -DMK_PHASE_TS=1.
std::vector<int64_t> mk_read_ts() {
#ifdef MK_PHASE_TS
  return mk_read_and_clear(g_mk_ts);
#else
  return {};
#endif
}

// Same for the mhc kernel: [MK_MHC_GRID_CAP][8] -- entry, p1 done, barrier,
// p2 done, barrier, p3 done, barrier, p4 done.
std::vector<int64_t> mk_read_kda_ts() {
#ifdef MK_PHASE_TS
  return mk_read_and_clear(g_mk_kda_ts);
#else
  return {};
#endif
}

std::vector<int64_t> mk_read_mhc_ts() {
#ifdef MK_PHASE_TS
  return mk_read_and_clear(g_mk_mhc_ts);
#else
  return {};
#endif
}

void mk_run_gemm(torch::Tensor x, torch::Tensor wq4, torch::Tensor ws4,
                 torch::Tensor out, int64_t n_orig, double wgs) {
  set_kernel_attrs();
  MKGemmCtx c{};
  c.x = (const __nv_bfloat16*)x.data_ptr();
  c.wq4 = (const uint8_t*)wq4.data_ptr();
  c.ws4 = (const int8_t*)ws4.data_ptr();
  c.wgs = (float)wgs;
  c.out = (__nv_bfloat16*)out.data_ptr();
  c.m = (int)x.size(0);
  c.k = (int)x.size(1);        // k is the ACTIVATION width
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 7) == 0 && x.is_contiguous(),
              "x must be 8 B aligned and contiguous");
  // Tile-major packs -- see stage_raw4. The shape is the only thing
  // standing between a stale row-major pack and silently wrong output.
  TORCH_CHECK(wq4.dim() == 4 && wq4.size(2) == SMEM_W_ROWS
                  && wq4.size(3) == 64 && wq4.is_contiguous(),
              "wq4 must be a contiguous [n/128, k/128, 128, 64] pack");
  TORCH_CHECK(ws4.dim() == 4 && ws4.size(0) == wq4.size(0)
                  && ws4.size(1) == wq4.size(1) && ws4.size(2) == SMEM_W_ROWS
                  && ws4.size(3) == 8 && ws4.is_contiguous(),
              "ws4 must be a contiguous [n/128, k/128, 128, 8] pack");
  c.n = (int)wq4.size(0) * SMEM_W_ROWS;
  c.n_orig = (int)n_orig;
  TORCH_CHECK(c.k % KSTEP == 0 && c.k <= KBLK_MAX * KSTEP, "k out of contract");
  TORCH_CHECK((int)wq4.size(1) == c.k / KSTEP, "wq4 k-tiles disagree with x");
  TORCH_CHECK(c.m <= 32, "m out of contract");
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (mk_gemm2_on()) {  // the non-persistent lane
    MKGemm2Ctx c2{};
    c2.x = c.x; c2.out = c.out; c2.wq4 = c.wq4; c2.ws4 = c.ws4;
    c2.wgs = c.wgs; c2.m = c.m; c2.n = c.n; c2.k = c.k;
    c2.n_orig = c.n_orig;
    const int nblk = c2.n / SMEM_W_ROWS;
    c2.ksr = mk_choose_ksr2(c2.m, c2.n, c2.k);
    TORCH_CHECK(nblk <= MK2_TILES_MAX && c2.ksr >= 1
                    && c2.ksr <= c2.k / KSTEP
                    && (size_t)c2.m * c2.n * c2.ksr <= (size_t)MK2_PART_ELEMS,
                "gemm2 plan out of contract");
    mk_launch(mk_gemm2_kernel, nblk * c2.ksr, GEMM2_SMEM, stream, c2);
    return;
  }
  c.grid = mk_resident_grid(mk_gemm_kernel, g_gemm_grid, GEMM_SMEM);
  c.ksr = mk_choose_ksr(c.m, c.n, c.k, c.grid);
  mk_launch(mk_gemm_kernel, c.grid, GEMM_SMEM, stream, c);
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
  mk_launch(mk_mhc_kernel, mhc_grid, 0, stream, a);
}

// ptrs: x, in_wq4, in_ws4, f_b_w, g_b_w, conv_w, conv_state, rec_state,
//       a_log, dt_bias, cu, state_idx, n_acc, o_wq4, o_ws4, out,
//       qkv, g1, g2, convq, attn, barrier, onorm_w
// ints: num_tokens, n_spec, mql, delta_variant, conv_width
// scalars: lower_bound, onorm_eps
void mk_run_kda(std::vector<int64_t> ptrs, std::vector<double> scalars,
                std::vector<int64_t> ints) {
  set_kernel_attrs();
  MKKdaArgs a{};
  a.x = (const __nv_bfloat16*)ptrs[0];
  a.in_wq4 = (const uint8_t*)ptrs[1];
  a.in_ws4 = (const int8_t*)ptrs[2];
  a.f_b_w = (const __nv_bfloat16*)ptrs[3];
  a.g_b_w = (const __nv_bfloat16*)ptrs[4];
  // phase 1 reads the gate weight rows and the token's activation as 8 B
  // vectors per lane (row bases are multiples of 256 B / 8 B by layout)
  TORCH_CHECK((ptrs[3] & 15) == 0 && (ptrs[4] & 15) == 0 && (ptrs[16] & 15) == 0,
              "gate weights and the qkv workspace must be 16 B aligned (cp.async)");
  static_assert(((KDA_QKV + KDA_H) * 2) % 16 == 0 && (KDA_INPROJ_N * 2) % 16 == 0,
                "gate activation slices must be 16 B aligned rows");
  a.conv_w = (const float*)ptrs[5];
  a.conv_state = (float*)ptrs[6];
  a.rec_state = (float*)ptrs[7];
  a.a_log = (const float*)ptrs[8];
  a.dt_bias = (const float*)ptrs[9];
  a.cu_seqlens = (const int*)ptrs[10];
  a.state_idx = (const int*)ptrs[11];
  a.n_accepted = (const int*)ptrs[12];
  a.o_wq4 = (const uint8_t*)ptrs[13];
  a.o_ws4 = (const int8_t*)ptrs[14];
  a.out = (__nv_bfloat16*)ptrs[15];
  a.qkv = (__nv_bfloat16*)ptrs[16];
  a.g1 = (__nv_bfloat16*)ptrs[17];
  a.g2 = (__nv_bfloat16*)ptrs[18];
  a.convq = (__nv_bfloat16*)ptrs[19];
  a.attn = (__nv_bfloat16*)ptrs[20];
  a.barrier_ctr = (unsigned long long*)ptrs[21];
  a.onorm_w = (const __nv_bfloat16*)ptrs[22];
  TORCH_CHECK(ptrs.size() == 23 && ints.size() == 5 && scalars.size() == 4,
              "run_kda arg contract");
  a.num_tokens = (int)ints[0];
  a.n_spec = (int)ints[1];
  a.mql = (int)ints[2];
  TORCH_CHECK(a.mql <= KDA_NQ_MAX,
              "kda: max_query_len over the unrolled conv window (KDA_NQ_MAX)");
  a.delta_variant = (int)ints[3];
  a.conv_width = (int)ints[4];
  a.lower_bound = (float)scalars[0];
  a.onorm_eps = (float)scalars[1];
  a.in_wgs = (float)scalars[2];
  a.o_wgs = (float)scalars[3];
  auto stream = c10::cuda::getCurrentCUDAStream();
  a.grid = mk_resident_grid(mk_kda_kernel, g_kda_grid, GEMM_SMEM);
  a.ksr_in = mk_choose_ksr(a.num_tokens, KDA_INPROJ_N_PAD, HIDDEN, a.grid);
  a.ksr_out = mk_choose_ksr(a.num_tokens, HIDDEN, KDA_OUT_PAD, a.grid);
  {  // probe knobs: force the split-K of either phase (0 = the cost model)
    static int f_in = -1, f_out = -1;
    if (f_in < 0) {
      const char* e = getenv("VLLM_GLM53_MK_KSR_IN");
      f_in = e ? atoi(e) : 0;
      e = getenv("VLLM_GLM53_MK_KSR_OUT");
      f_out = e ? atoi(e) : 0;
    }
    if (f_in > 0) a.ksr_in = f_in;
    if (f_out > 0) a.ksr_out = f_out;
  }
  mk_launch(mk_kda_kernel, a.grid, GEMM_SMEM, stream, a);
}

// MK_SEG_MLA. `splits` is chosen by the caller so T x splits fills the
// persistent grid; the kernel reads every per-row length from device memory,
// which is what lets this run inside the captured graph.
void mk_run_mla(std::vector<int64_t> ptrs, std::vector<double> scalars,
                std::vector<int64_t> ints) {
  set_kernel_attrs();
  MKMlaArgs a{};
  TORCH_CHECK(ptrs.size() == 8 && ints.size() == 3 && scalars.size() == 2,
              "run_mla arg contract");
  a.q = (const __nv_bfloat16*)ptrs[0];
  a.ckv = (const uint8_t*)ptrs[1];
  a.slots = (const int*)ptrs[2];
  a.lens = (const int*)ptrs[3];
  a.out = (__nv_bfloat16*)ptrs[4];
  a.part = (float*)ptrs[5];
  a.pml = (float*)ptrs[6];
  a.barrier_ctr = (unsigned long long*)ptrs[7];
  TORCH_CHECK((ptrs[1] & 15) == 0 && (ptrs[0] & 15) == 0,
              "mla: q and the latent cache must be 16 B aligned (cp.async)");
  a.T = (int)ints[0];
  a.W = (int)ints[1];
  a.splits = (int)ints[2];
  TORCH_CHECK(a.splits >= 1 && a.splits <= MLA_SPLITS_MAX, "mla: split count");
  a.sm_scale = (float)scalars[0];
  a.ckv_scale = (float)scalars[1];
  {  // roofline probe knob (never set in serving)
    static int pv = -1;
    if (pv < 0) { const char* e = getenv("VLLM_GLM53_MK_MLA_PROBE"); pv = e ? atoi(e) : 0; }
    a.probe = pv;
  }
  auto stream = c10::cuda::getCurrentCUDAStream();
  a.grid = mk_resident_grid(mk_mla_kernel, g_mla_grid, MLA_SMEM, MLA_GRID_CAP);
  mk_launch(mk_mla_kernel, a.grid, MLA_SMEM, stream, a);
}

// Blocks the caller can fill: the split count this launch would use.
int64_t mk_mla_grid() {
  set_kernel_attrs();
  return mk_resident_grid(mk_mla_kernel, g_mla_grid, MLA_SMEM, MLA_GRID_CAP);
}

// Bench: the v2 plan one launch of (m, n, k) would use -- {on, ksr, units,
// blocks per SM} -- and the setter behind it (-1 leaves a knob as it is).
std::vector<int64_t> mk_gemm2_plan(int64_t m, int64_t n, int64_t k) {
  set_kernel_attrs();
  const int n_pad = (int)(((n + SMEM_W_ROWS - 1) / SMEM_W_ROWS) * SMEM_W_ROWS);
  const int ksr = mk_choose_ksr2((int)m, n_pad, (int)k);
  return {(int64_t)(mk_gemm2_on() ? 1 : 0), (int64_t)ksr,
          (int64_t)((n_pad / SMEM_W_ROWS) * ksr), (int64_t)g_gemm2_bps};
}
void mk_set_gemm2(int64_t on, int64_t ksr) {
  if (on >= 0) g_probe_gemm2 = (int)on;
  if (ksr >= 0) g_probe_ksr2 = (int)ksr;
}
// v2 unit timestamps of the last launch, [MK2_UNITS_MAX][4] ns, then cleared.
std::vector<int64_t> mk_read_ts2() {
#ifdef MK_PHASE_TS
  return mk_read_and_clear(g_mk2_ts);
#else
  return {};
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_device", &mk_probe_device, "device geometry probe");
  m.def("read_ts", &mk_read_ts, "phase timestamps (MK_PHASE_TS builds)");
  m.def("read_mhc_ts", &mk_read_mhc_ts, "mhc phase timestamps");
  m.def("read_kda_ts", &mk_read_kda_ts, "kda phase timestamps");
  m.def("run_gemm", &mk_run_gemm, "MK_SEG_GEMM (W4 pack)");
  m.def("gemm2_plan", &mk_gemm2_plan, "bench: v2 {on, ksr, units, blocks/SM}");
  m.def("set_gemm2", &mk_set_gemm2, "bench: force the v2 lane and its ksr (-1 = keep)");
  m.def("read_ts2", &mk_read_ts2, "v2 unit timestamps (MK_PHASE_TS builds)");
  m.def("run_mhc", &mk_run_mhc, "MK_SEG_MHC");
  m.def("run_kda", &mk_run_kda, "MK_SEG_KDA");
  m.def("run_mla", &mk_run_mla, "MK_SEG_MLA (sparse MLA decode)");
  m.def("mla_grid", &mk_mla_grid, "MK_SEG_MLA resident grid");
}
