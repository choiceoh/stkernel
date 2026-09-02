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
#include <cstdlib>
#include <vector>

namespace {

// Three tuning constants come in through -D so the probe can sweep them
// without editing this file; the defaults here are the shipped values.
#ifndef MK_GRID_DEF
#define MK_GRID_DEF 96
#endif
#ifndef MK_MHC_GRID_DEF
#define MK_MHC_GRID_DEF 144
#endif
#ifndef MK_W_NBUF_DEF
#define MK_W_NBUF_DEF 3
#endif
// Ceiling for the gemm and kda persistent grids -- each launch takes the
// smaller of it and what the device reports resident, exactly as mhc
// does. At MK_W_NBUF 3 the 63,616 B block only fits once in the SM's
// 102,400 B, so this resolves to 48; a shallower pipeline fits twice and
// it resolves to 96 with no other change.
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
constexpr int NCHUNK = HIDDEN / HCHUNK;  // 16 (p3/p4 slots)
// mhc p1 work item = (64-column sub-chunk, token half). The block loads its
// fn slice ONCE into registers -- 4 output groups x 6 outputs x HC per
// thread -- and walks its tokens, so fn costs 2 x 1.5 MB per op instead of
// T x 1.5 MB. The old (token, 256-chunk) split re-read fn per token: the
// stamps had p1 at 17 us (T=8) and 41 us (T=32), 48 MB of L2 re-reads.
constexpr int P1_HCHUNK = 64;
constexpr int P1_NCHUNK = HIDDEN / P1_HCHUNK;  // 64 yp/rp slots per token
constexpr int P1_TSPLIT = 2;                    // 128 items for 144 blocks
constexpr int P1_MG = MK_THREADS / P1_HCHUNK;   // 4 output groups
constexpr int P1_MPER = NOUT / P1_MG;           // 6 outputs per group
static_assert(NOUT % P1_MG == 0 && P1_HCHUNK * P1_MG == MK_THREADS,
              "p1 thread map: 64 columns x 4 output groups");

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
// W pipeline depth. 3 buffers keep ~2 k-blocks in flight = 32 KB/block;
// 4 keep ~3, 5 keep ~4 (smem 82,048 / 100,480 B, both under the 101,376
// opt-in). Occupancy is NOT the cliff it was written up as: the device
// answers 1 block/SM already at 3 (63,616 B against 102,400 B/SM, #206),
// so 4 and 5 cost nothing there. What made "deepening to 4 measured
// exactly zero" was the wait: cp.async.wait_group<1> is exact only at
// depth 3 -- at 4 it waited for two tiles when one was needed, so the
// extra buffer was never in flight during the wait. The loop now waits
// for exactly the tile it needs (mk_cp_wait_upto). 2 buffers (2 blocks/SM)
// measured worse on every shape.
constexpr int MK_W_NBUF = MK_W_NBUF_DEF;
// W4 raw staging: one (tile, k-block) record is 128 rows x 64 B of e2m1
// pairs plus 128 x 8 B of group exponents. In smem the nibble rows sit on
// an 80 B pitch (a warp's 32 rows of uint4 reads then hit distinct banks)
// and the exponents follow as a flat 1 KB. Three stages, two in flight,
// like the W8 pipeline; the expanded e4m3 tiles reuse swb[0..1].
// Raw rows on a 64 B pitch: the two uint4 reads per thread per k-block
// conflict 16-way, ~100 cycles, where the 80 B padding cost a whole
// pipeline stage of smem. Five stages, four in flight = 36 KB/block, the
// W8 pipeline's 32 KB -- with three (two in flight, 18 KB) the W4 loop ran
// at 86 GB/s against the W8 arm's 150: bytes in flight are the bandwidth.
constexpr int W4_RAW_PITCH = 64;
constexpr int W4_RAW_NIB = SMEM_W_ROWS * W4_RAW_PITCH;   // 8192
constexpr int W4_RAW_BYTES = W4_RAW_NIB + SMEM_W_ROWS * 8;  // 9216
constexpr int W4_RAW_NBUF = 5;
constexpr int W4_EXP_NBUF = 2;  // expanded e4m3 tiles, ping-pong
// Two kernels, two budgets. The W8 kernel is exactly the 63,616 B it was
// before the W4 pipeline: sharing one budget put the W4 raw stages into the
// W8 launch too, and that measured 4-7% slower on every W8 shape (the
// carveout that grew took the L1 the W8 loop was using).
constexpr int GEMM_SMEM = 2 * 16 * SMEM_A_PITCH +
                          MK_W_NBUF * SMEM_W_ROWS * SMEM_W_PITCH +
                          KBLK_MAX * KBLK_MAX * 4;   // 63,616 at NBUF 3
constexpr int GEMM_SMEM_W4 = 2 * 16 * SMEM_A_PITCH +
                             W4_EXP_NBUF * SMEM_W_ROWS * SMEM_W_PITCH +
                             KBLK_MAX * KBLK_MAX * 4 +
                             W4_RAW_NBUF * W4_RAW_BYTES;  // 91,264
static_assert(GEMM_SMEM <= 101376 && GEMM_SMEM_W4 <= 101376,
              "over the sm_121 opt-in smem");
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
// stay in flight" count (the W pipeline keeps at most MK_W_NBUF - 2 = 3).
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
  const uint8_t* wq;       // [n, k] e4m3, n % 128 == 0 (W8 path)
  const float* ws;         // [n/128, k/128] (all-ones tensor on the W4 path)
  __nv_bfloat16* out;      // [m, n_orig]
  int m, n, k, n_orig;
  int grid;  // resident blocks; see MK_GRID_CAP
  int ksr;   // k-split of the leftover tiles, chosen on the host (mk_choose_ksr)
  // W4 path: e2m1 nibbles [n, k/2] + per-16-group scale exponents
  // [n, k/16] (int8, clamped to [-5, 6] at build). The kernel expands each
  // nibble to an EXACT e4m3 byte (1-bit mantissas always fit; the 2^s
  // product is an exponent-field add) and then runs the same e4m3 mma
  // pipeline -- the DRAM bytes halve and the arithmetic is unchanged.
  // Tile-major like wq: [n/128][k/128][128][64] nibble bytes and
  // [n/128][k/128][128][8] exponents, so one (tile, k-block) record is a
  // contiguous 8 KB + 1 KB run for cp.async (see stage_raw4).
  const uint8_t* wq4 = nullptr;  // non-null selects the W4 path
  const int8_t* ws4 = nullptr;
};

// e4m3 encodings of the e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
// Expansion: byte = LUT[mag] + (s << 3) | sign<<7 -- exact while the build
// keeps s in [-5, 6] (exp field stays inside [1, 15), never a denormal,
// never the NaN encoding: LUT mantissas are never 111).
__device__ __constant__ uint8_t mk_e2m1_to_e4m3[8] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C};
// The same table as one 64-bit immediate, byte c at bits [8c, 8c+8): the
// expansion indexes it with a funnel shift instead of a constant-memory
// load. A __constant__ load serialises over the distinct addresses in the
// warp -- up to 8 here, 64 lookups per thread per k-block -- which made
// the W4 expansion cost more than the DRAM time it was meant to hide.
constexpr unsigned long long MK_E2M1_LUT64 = 0x4C484440'3C383000ULL;
__device__ __forceinline__ uint8_t mk_e2m1_byte(int code3) {
  return (uint8_t)(MK_E2M1_LUT64 >> (code3 * 8));
}

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
// The W4 instantiation is a different kernel with its own occupancy, so
// it gets its own ticket counter (#206: grids that can differ must not
// share one).
__device__ unsigned long long g_mk_gemm4_bar = 0ULL;
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
__device__ __forceinline__ unsigned long long mk_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define MK_TS(slot)                                                          \
  do {                                                                       \
    if (threadIdx.x == 0) g_mk_ts[blockIdx.x * 8 + (slot)] = mk_globaltimer(); \
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
#define MK_MHC_TS(slot) \
  do {                  \
  } while (0)
#endif

template <bool W4>
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

  constexpr int NWB = W4 ? W4_EXP_NBUF : MK_W_NBUF;
  uint8_t* saq = smem;  // [2][16][132] fp8 A tiles (single per kb)
  uint8_t* swb = saq + 2 * 16 * SMEM_A_PITCH;  // [NWB][128][144]
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
  // stage one k-block of W rows [nt*128, nt*128+128) into a pipeline
  // buffer (async, 16B copies; both addresses are 16B aligned by
  // construction: pitch 144 and k in {2048, 4096}).
  auto stage_w = [&](int nt, int kb, int buf) {
    // wq is TILE-major: [n/128][k/128][128][128]. Row-major would put the
    // 128 rows of this tile 4096 B apart, so a warp's 32 copies landed on
    // four 128 B segments in four different DRAM pages to fetch 16 KB.
    // Tile-major makes the same tile one contiguous 16 KB run.
    const uint8_t* wsrc =
        c.wq + ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * KSTEP);
    uint8_t* d0 = swb + buf * (SMEM_W_ROWS * SMEM_W_PITCH);
    // Flatten (row, 16B chunk) so ALL MK_THREADS issue copies. The row-
    // strided form left threads >= SMEM_W_ROWS (128 of 256) idle, halving
    // the bytes in flight -- and this stage is latency-bound, so in-flight
    // bytes ARE the bandwidth (Little's law).
    constexpr int MK_W_CHUNKS = KSTEP / 16;
    // Thread 0 issues NO copies (it still commits an empty group, so the
    // per-thread wait counts stay uniform). It is the thread that runs
    // the grid barrier, and the barrier's __threadfence drains that
    // thread's outstanding cp.async: with the first unit's fill hoisted
    // above the A-quant barrier, the stamps showed the barrier wait grow
    // from 1.3 us to 6-12 us -- the fill's latency had simply moved into
    // the barrier. Threads 1..255 cover the 1024 chunks, 4-5 each.
    for (int t = (int)threadIdx.x - 1; threadIdx.x != 0
         && t < SMEM_W_ROWS * MK_W_CHUNKS; t += MK_THREADS - 1) {
      const int r = t / MK_W_CHUNKS;
      const int e = (t % MK_W_CHUNKS) * 16;
      // r * KSTEP + e == t * 16, so the source walk is now linear across
      // the whole block; only the destination keeps the padded pitch.
      mk_cp_async16(d0 + r * SMEM_W_PITCH + e, wsrc + (size_t)t * 16);
    }
    mk_cp_commit();
  };
  // W4: stage one raw (tile, k-block) record -- 512 nibble chunks (two
  // per thread) onto the 80 B pitch, 64 exponent chunks after them. One
  // commit group per stage, the same accounting the W8 waits rely on.
  auto stage_raw4 = [&](int nt, int kb, int buf) {
    const uint8_t* nsrc =
        c.wq4 + ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * 64);
    const uint8_t* ssrc = (const uint8_t*)c.ws4 +
        ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * 8);
    uint8_t* d = sraw + buf * W4_RAW_BYTES;
    // thread 0 issues nothing -- see stage_w
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
  // (row, half): 64 elements = 32 B of nibbles + 4 exponents, written as
  // 64 B on the 144 B tile pitch. Exact: 1-bit mantissas always fit, 2^s
  // is an exponent-field add.
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
    uint4* dv = (uint4*)(swb + expbuf * (SMEM_W_ROWS * SMEM_W_PITCH) +
                         row4 * SMEM_W_PITCH + half4 * 64);
#pragma unroll
    for (int g4 = 0; g4 < 4; ++g4) {  // one 16-group: 2 nibble words -> 4 bytes words
      // exponent-field add; s in [-5, 6] keeps LUT + e inside [8, 124], so
      // it never carries into the sign bit
      const int e = ((int)(int8_t)((sc4 >> (8 * g4)) & 0xFFu)) << 3;
      uint32_t ow[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {  // output word j = elements 4j..4j+3
        const uint32_t w = nw[2 * g4 + (j >> 1)] >> (16 * (j & 1));
        uint32_t o = 0u;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint32_t code = (w >> (4 * q)) & 0xFu;
          const uint32_t c7 = code & 7u;
          uint32_t bt = c7 ? (((uint32_t)mk_e2m1_byte((int)c7) +
                               (uint32_t)e) & 0xFFu)
                           : 0u;
          bt |= (code & 8u) << 4;  // sign
          o |= bt << (8 * q);
        }
        ow[j] = o;
      }
      dv[g4] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
  };
  static_assert(MK_W_NBUF >= 2, "the pipeline needs a spare buffer");
  static_assert(MK_W_NBUF <= 5, "mk_cp_wait_upto dispatches up to 3");
  constexpr int DIST = MK_W_NBUF - 1;
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
      if constexpr (W4) {
        stage_raw4(nt0, kb00, kb00 % W4_RAW_NBUF);
#pragma unroll
        for (int d = 1; d < RAW_DIST; ++d)
          if (kb00 + d < kbn0)
            stage_raw4(nt0, kb00 + d, (kb00 + d) % W4_RAW_NBUF);
      } else {
        stage_w(nt0, kb00, kb00 % MK_W_NBUF);
#pragma unroll
        for (int d = 1; d < DIST; ++d)
          if (kb00 + d < kbn0) stage_w(nt0, kb00 + d, (kb00 + d) % MK_W_NBUF);
      }
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
  if (kbq < kblk) {
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
  if (kbq < kblk) quant_store(kbq, v, mx);
  for (int kb = kbq + c.grid; kb < kblk; kb += c.grid) {  // grid < kblk only
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
  if (blockIdx.x == 0 && threadIdx.x == 0) g_mk_unit_next = 0u;
  MK_TS(1);  // A quantized, before the publishing barrier
  mk_grid_barrier(bar, c.grid);
  MK_TS(2);  // barrier released
  for (int i = threadIdx.x; i < c.m * KBLK_MAX; i += MK_THREADS)
    sxs[i] = g_mk_axs[i];
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
    auto stage_a = [&](int kb) {
      constexpr int WORDS = KSTEP / 4;
      for (int t = threadIdx.x; t < c.m * WORDS; t += MK_THREADS) {
        const int r = t / WORDS, e = (t % WORDS) * 4;
        uint8_t* dst = saq + (r >> 4) * 16 * SMEM_A_PITCH +
                       (r & 15) * SMEM_A_PITCH;
        *(uint32_t*)(dst + e) = *(const uint32_t*)(
            g_mk_aq + ((size_t)kb * 32 + r) * KSTEP + e);
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

    const bool prefilled = hoisted && (u == (int)blockIdx.x);
    if constexpr (W4) {
      // ---- W4 path: cp.async-staged raw records, expanded in smem.
      // The raw (tile, k-block) record streams through the W4_RAW_NBUF
      // pipeline exactly as W8 tiles do; the expansion then reads the
      // landed record and fills one of two e4m3 tile buffers (swb[0..1]).
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
        const uint8_t* sw4t =
            swb + (kb % 2) * (SMEM_W_ROWS * SMEM_W_PITCH);
        mma_fold(sw4t, kb, 1.0f);  // scales already inside the bytes
        if (kb + 1 >= kbn) break;
        // raw(kb+1) landed: the groups still allowed in flight are the
        // ones issued after it, min(RAW_DIST - 1, kbn - kb - 2).
        mk_cp_wait_upto(min(RAW_DIST - 1, kbn - kb - 2));
        __syncthreads();  // raw(kb+1) visible; every mma reader of kb done
        expand_w4((kb + 1) % W4_RAW_NBUF, (kb + 1) % 2);
        stage_a(kb + 1);
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
      if (!prefilled) stage_w(nt, kb0, kb0 % MK_W_NBUF);
      stage_a(kb0);
      if (!prefilled) {
#pragma unroll
        for (int d = 1; d < DIST; ++d)
          if (kb0 + d < kbn) stage_w(nt, kb0 + d, (kb0 + d) % MK_W_NBUF);
      }
      // Wait for exactly W(kb0): the groups allowed to stay in flight are
      // the ones issued after it, min(DIST - 1, kbn - kb0 - 1).
      mk_cp_wait_upto(min(DIST - 1, kbn - kb0 - 1));
      __syncthreads();
      if (u == (int)blockIdx.x) MK_TS(3);  // first unit: W(kb0) landed

      for (int kb = kb0;; ++kb) {
        if (kb + DIST < kbn) stage_w(nt, kb + DIST, (kb + DIST) % MK_W_NBUF);
        const uint8_t* sw =
            swb + (kb % MK_W_NBUF) * (SMEM_W_ROWS * SMEM_W_PITCH);
        mma_fold(sw, kb, c.ws[(size_t)nt * kblk + kb]);

        if (kb + 1 >= kbn) break;
        __syncthreads();  // every mma reader of saq is done first
        stage_a(kb + 1);  // ALU work while W(kb+1) finishes its flight
        // outstanding after W(kb+1): the deeper stages, when they exist --
        // min(DIST - 1, kbn - kb - 2) of them (kb + DIST was just issued).
        mk_cp_wait_upto(min(DIST - 1, kbn - kb - 2));
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
  }
  MK_TS(4);  // all units of this block done (no fold barrier any more)
  MK_TS(6);  // exit
}

template <bool W4>
__global__ void mk_gemm_kernel(const MKGemmCtx c) {
  extern __shared__ uint8_t smem[];
  // PDL: the next launch in the stream may start on the SMs this grid
  // frees as blocks exit; it prefetches its own weights during this
  // grid's tail and waits (griddepcontrol.wait) before reading anything
  // this grid writes. Harmless when the next launch is not programmatic.
  asm volatile("griddepcontrol.launch_dependents;");
  mk_gemm_phase<W4>(c, smem, W4 ? &g_mk_gemm4_bar : &g_mk_gemm_bar);
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
  // thread -> (column hl of the sub-chunk, output group mg); group g is
  // warps 2g and 2g+1, so a group's 64 partials reduce as two warp shuffles
  // plus one smem add. sqr (and residual_out) belong to group 0 only.
  __shared__ float red[MK_WARPS][P1_MPER + 1];
  const int nitems = P1_NCHUNK * P1_TSPLIT;
  const int hl = threadIdx.x & (P1_HCHUNK - 1);
  const int mg = threadIdx.x / P1_HCHUNK;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int it = bid; it < nitems; it += a.grid) {
    const int c1 = it / P1_TSPLIT, th = it % P1_TSPLIT;
    const int h = c1 * P1_HCHUNK + hl;
    const int t0 = (a.num_tokens * th) / P1_TSPLIT;
    const int t1 = (a.num_tokens * (th + 1)) / P1_TSPLIT;
    float fnr[P1_MPER][HC];
#pragma unroll
    for (int mm = 0; mm < P1_MPER; ++mm)
#pragma unroll
      for (int j = 0; j < HC; ++j)
        fnr[mm][j] = a.fn[(size_t)(mg * P1_MPER + mm) * HC * HIDDEN +
                          j * HIDDEN + h];
    for (int t = t0; t < t1; ++t) {
      float cm[HC][HC], pm[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        pm[j] = a.post_mix_in[t * HC + j];
#pragma unroll
        for (int k = 0; k < HC; ++k)
          cm[k][j] = a.comb_mix_in[t * HC * HC + k * HC + j];
      }
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
      float sqr = 0.0f;
      if (mg == 0) {
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          a.residual_out[(size_t)t * HC * HIDDEN + j * HIDDEN + h] =
              __float2bfloat16(r[j]);
          sqr += r[j] * r[j];
        }
      }
      float dot[P1_MPER];
#pragma unroll
      for (int mm = 0; mm < P1_MPER; ++mm) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) v += fnr[mm][j] * r[j];
        dot[mm] = v;
      }
#pragma unroll
      for (int mm = 0; mm < P1_MPER; ++mm)
        for (int off = 16; off; off >>= 1)
          dot[mm] += __shfl_xor_sync(~0u, dot[mm], off);
      for (int off = 16; off; off >>= 1) sqr += __shfl_xor_sync(~0u, sqr, off);
      if (lane == 0) {
#pragma unroll
        for (int mm = 0; mm < P1_MPER; ++mm) red[warp][mm] = dot[mm];
        red[warp][P1_MPER] = sqr;
      }
      __syncthreads();
      if (threadIdx.x < NOUT) {
        const int g = threadIdx.x / P1_MPER, mm = threadIdx.x % P1_MPER;
        a.yp[((size_t)c1 * MAX_TOK + t) * NOUT + threadIdx.x] =
            red[2 * g][mm] + red[2 * g + 1][mm];
      } else if (threadIdx.x == NOUT) {
        a.rp[c1 * MAX_TOK + t] = red[0][P1_MPER] + red[1][P1_MPER];
      }
      __syncthreads();
    }
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
    // The 24 chunk reductions are spread over the warp: lanes 0..23 each
    // sum one output's 16 partials (16 independent loads, one L2 round
    // trip), lanes 24..31 the sumsq partials. Every lane used to run all
    // 384 loads itself, in a dependent chain -- the stamps put this phase
    // at 12.5 us per token-warp while 143 blocks waited at the barrier.
    float mine = 0.0f;
    if (lane < NOUT) {
#pragma unroll 16
      for (int c = 0; c < P1_NCHUNK; ++c)
        mine += a.yp[((size_t)c * MAX_TOK + t) * NOUT + lane];
    } else {
#pragma unroll
      for (int c = lane - NOUT; c < P1_NCHUNK; c += 32 - NOUT)
        mine += a.rp[c * MAX_TOK + t];
    }
    float sqr = (lane >= NOUT) ? mine : 0.0f;
#pragma unroll
    for (int off = 16; off; off >>= 1)
      sqr += __shfl_xor_sync(0xffffffffu, sqr, off);
    const float rms = rsqrtf(sqr / (float)(HC * HIDDEN) + a.rms_eps);
    const float mixv = mine * rms;
    float mixes[NOUT];
#pragma unroll
    for (int m = 0; m < NOUT; ++m) mixes[m] = __shfl_sync(0xffffffffu, mixv, m);

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
  // PDL, no prefetch: this kernel takes no dynamic smem to stage weights
  // into, so it only lets the next launch start on the SMs it frees and
  // itself waits for its predecessor before p1's first read.
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  MK_MHC_TS(0);
  // 3 grid barriers, down from 5, and no prologue. Each surviving one is a
  // real data dependency (partials -> mixes -> pre-mixes -> sumsq). What
  // went: p3 now STORES its sumsq per (chunk, token) instead of accumulating
  // into sq[t], so there is nothing to zero and no barrier to order the
  // zeroing against; and p4 reduces those slots itself, which retired the
  // separate rsqrt phase and its barrier.
  mk_mhc_p1(a, blockIdx.x);
  MK_MHC_TS(1);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  MK_MHC_TS(2);
  mk_mhc_p2(a, blockIdx.x);
  MK_MHC_TS(3);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  MK_MHC_TS(4);
  mk_mhc_p3(a, blockIdx.x);
  MK_MHC_TS(5);
  mk_grid_barrier(a.barrier_ctr, a.grid);
  MK_MHC_TS(6);
  // p4 does its own rsqrt, so the fourth phase boundary (sumsq -> rsqrt)
  // and the single-thread loop that used to sit on it are both gone.
  mk_mhc_p4(a, blockIdx.x);
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
  int grid;          // resident blocks; see MK_GRID_CAP
  int ksr_in, ksr_out;  // mk_choose_ksr for the in_proj / o_proj phases
  int num_tokens;
  float lower_bound, onorm_eps;
};

__global__ void mk_kda_kernel(const MKKdaArgs a) {
  extern __shared__ uint8_t smem[];
  asm volatile("griddepcontrol.launch_dependents;");  // see mk_gemm_kernel

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
    c.grid = a.grid;
    c.ksr = a.ksr_in;
    mk_gemm_phase<false>(c, smem, &g_mk_kda_bar);
  }
  mk_grid_barrier(a.barrier_ctr, a.grid);

  {  // phase 1: f_b / g_b low-rank gates (K = 128, SIMT dot)
    const int total = a.num_tokens * KDA_OUT;
    for (int i = blockIdx.x * MK_THREADS + threadIdx.x; i < 2 * total;
         i += a.grid * MK_THREADS) {
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
  mk_grid_barrier(a.barrier_ctr, a.grid);

  {  // phase 2: merged short conv (k=4, silu) with accepted-window rollback.
    // hist(pos) is the channel's input at in-request position pos (pos < 0
    // reads the slot's conv state). Outputs are produced for ALL query
    // tokens; the state window is rolled to the accepted boundary.
    for (int r = 0; r < a.n_spec; ++r) {
      const int t0 = a.cu_seqlens[r], t1 = a.cu_seqlens[r + 1];
      const int slot = a.state_idx[r * a.mql + 0];
      const int acc = a.n_accepted[r];
      for (int ch = blockIdx.x * MK_THREADS + threadIdx.x; ch < KDA_QKV;
           ch += a.grid * MK_THREADS) {
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
  mk_grid_barrier(a.barrier_ctr, a.grid);

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
  mk_grid_barrier(a.barrier_ctr, a.grid);

  {  // phase 4: gated RMSNorm -- rmsnorm(attn) * sigmoid(g2), in place
    __shared__ float wred[MK_WARPS];
    __shared__ float inv;
    const int pairs = a.num_tokens * KDA_H;
    for (int i = blockIdx.x; i < pairs; i += a.grid) {
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
  mk_grid_barrier(a.barrier_ctr, a.grid);

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
    c.grid = a.grid;
    c.ksr = a.ksr_out;
    mk_gemm_phase<false>(c, smem, &g_mk_kda_bar);
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
      mk_gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      GEMM_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      GEMM_SMEM_W4));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM_SMEM));
  g_attrs_set = true;
}

// Resident block count for a persistent grid, clamped to MK_GRID_CAP.
// Cached per kernel: the ticket barrier needs the SAME grid on every launch,
// and the two kernels are asked separately because their occupancy differs.
template <typename K>
int mk_resident_grid(K kernel, int& cache, int smem) {
  if (cache == 0) {
    int per_sm = 0, sms = 0;
    MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, kernel, MK_THREADS, smem));
    MK_CHECK_CUDA(cudaDeviceGetAttribute(
        &sms, cudaDevAttrMultiProcessorCount, 0));
    cache = per_sm * sms;
    if (cache > MK_GRID_CAP) cache = MK_GRID_CAP;
    TORCH_CHECK(cache > 0, "persistent grid has no resident blocks");
  }
  return cache;
}

int g_gemm_grid = 0;
int g_gemm4_grid = 0;
int g_kda_grid = 0;

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
    for (int r = 2; r <= kblk; ++r) {
      if ((size_t)m * rem * 128 * r > MK_SPLIT_ELEMS) break;
      const int rounds = (nblk * r + grid - 1) / grid;
      if (rounds * bd < bn * r) { bn = rounds; bd = r; ksr = r; }
    }
  }
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

// Phase timestamps of the last gemm launch, [MK_GRID_CAP][8] ns, then
// cleared. Empty unless built with -DMK_PHASE_TS=1.
std::vector<int64_t> mk_read_ts() {
#ifdef MK_PHASE_TS
  std::vector<unsigned long long> h(MK_GRID_CAP * 8);
  MK_CHECK_CUDA(cudaDeviceSynchronize());
  MK_CHECK_CUDA(cudaMemcpyFromSymbol(h.data(), g_mk_ts,
                                     sizeof(unsigned long long) * h.size()));
  void* p = nullptr;
  MK_CHECK_CUDA(cudaGetSymbolAddress(&p, g_mk_ts));
  MK_CHECK_CUDA(cudaMemset(p, 0, sizeof(unsigned long long) * h.size()));
  return std::vector<int64_t>(h.begin(), h.end());
#else
  return {};
#endif
}

// Same for the mhc kernel: [MK_MHC_GRID_CAP][8] -- entry, p1 done, barrier,
// p2 done, barrier, p3 done, barrier, p4 done.
std::vector<int64_t> mk_read_mhc_ts() {
#ifdef MK_PHASE_TS
  std::vector<unsigned long long> h(MK_MHC_GRID_CAP * 8);
  MK_CHECK_CUDA(cudaDeviceSynchronize());
  MK_CHECK_CUDA(cudaMemcpyFromSymbol(h.data(), g_mk_mhc_ts,
                                     sizeof(unsigned long long) * h.size()));
  void* p = nullptr;
  MK_CHECK_CUDA(cudaGetSymbolAddress(&p, g_mk_mhc_ts));
  MK_CHECK_CUDA(cudaMemset(p, 0, sizeof(unsigned long long) * h.size()));
  return std::vector<int64_t>(h.begin(), h.end());
#else
  return {};
#endif
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
  c.k = (int)x.size(1);
  // the A-quant prologue reads x as 8 B vectors (4 bf16 per lane)
  // the A-quant prologue reads x as 8 B vectors (4 bf16 per lane)
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 7) == 0 && x.is_contiguous(),
              "x must be 8 B aligned and contiguous");
  // wq is [n/128, k/128, 128, 128] -- tile-major, see stage_w.
  TORCH_CHECK(wq.dim() == 4 && wq.size(2) == SMEM_W_ROWS
                  && wq.size(3) == KSTEP && wq.is_contiguous(),
              "wq must be a contiguous [n/128, k/128, 128, 128] pack");
  c.n = (int)wq.size(0) * SMEM_W_ROWS;
  c.n_orig = (int)n_orig;
  TORCH_CHECK(c.k % KSTEP == 0 && c.k <= KBLK_MAX * KSTEP, "k out of contract");
  TORCH_CHECK((int)wq.size(1) == c.k / KSTEP, "wq k-tiles disagree with x");
  TORCH_CHECK(c.m <= 32, "m out of contract");
  auto stream = c10::cuda::getCurrentCUDAStream();
  c.grid = mk_resident_grid(mk_gemm_kernel<false>, g_gemm_grid, GEMM_SMEM);
  c.ksr = mk_choose_ksr(c.m, c.n, c.k, c.grid);
  mk_launch(mk_gemm_kernel<false>, c.grid, GEMM_SMEM, stream, c);
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
  c.grid = mk_resident_grid(mk_gemm_kernel<true>, g_gemm4_grid, GEMM_SMEM_W4);
  c.ksr = mk_choose_ksr(c.m, c.n, c.k, c.grid);
  mk_launch(mk_gemm_kernel<true>, c.grid, GEMM_SMEM_W4, stream, c);
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
  a.grid = mk_resident_grid(mk_kda_kernel, g_kda_grid, GEMM_SMEM);
  a.ksr_in = mk_choose_ksr(a.num_tokens, KDA_INPROJ_N_PAD, HIDDEN, a.grid);
  a.ksr_out = mk_choose_ksr(a.num_tokens, HIDDEN, KDA_OUT_PAD, a.grid);
  mk_launch(mk_kda_kernel, a.grid, GEMM_SMEM, stream, a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_device", &mk_probe_device, "device geometry probe");
  m.def("read_ts", &mk_read_ts, "phase timestamps (MK_PHASE_TS builds)");
  m.def("read_mhc_ts", &mk_read_mhc_ts, "mhc phase timestamps");
  m.def("run_gemm", &mk_run_gemm, "MK_SEG_GEMM");
  m.def("run_gemm_w4", &mk_run_gemm_w4, "MK_SEG_GEMM (W4 pack)");
  m.def("run_mhc", &mk_run_mhc, "MK_SEG_MHC");
  m.def("run_kda", &mk_run_kda, "MK_SEG_KDA");
}
