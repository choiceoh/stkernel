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
#include <limits>
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
// One kernel, one budget (the fp8 W8 arm and its second budget were
// removed once the W4 arm beat the stock pair on every decode shape: the
// serving lane is W4 or stock, nothing in between to configure).
constexpr int MK_SMEM_ALIGN = 1024;  // runtime alignment of the dynamic base

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
// 37차 (2026-09-06): every device-side wait carries a deadline. A legitimate
// wait is microseconds; a spin that reaches MK_SPIN_DEADLINE iterations of
// __nanosleep (>= seconds) is a wedged kernel, and a wedged kernel is silent:
// the host's next launch blocks and every stack shows THAT launch (the 10:31
// and 10:47 boots sat 4+ minutes at a stock TileLang launch with the GPU at
// 96 %, and nothing named the spinner). Trapping here names the site and
// kills the process instead of hanging the boot. -DMK_SPIN_DEADLINE=0 disables.
#ifndef MK_SPIN_DEADLINE
#define MK_SPIN_DEADLINE (1u << 24)
#endif
#define MK_SPIN_WAIT(cond, ns, site)                                          \
  do {                                                                        \
    unsigned int mk_spin_n_ = 0;                                              \
    while (cond) {                                                            \
      __nanosleep(ns);                                                        \
      if (MK_SPIN_DEADLINE && ++mk_spin_n_ >= (unsigned int)MK_SPIN_DEADLINE) {   \
        printf("[megakernel] spin deadline at %s (block %d, thread %d)\n",   \
               site, (int)blockIdx.x, (int)threadIdx.x);                      \
        __trap();                                                             \
      }                                                                       \
    }                                                                         \
  } while (0)

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
    MK_SPIN_WAIT(*v < target, 64, "grid barrier");
    __threadfence();
  }
  __syncthreads();
}

__device__ __forceinline__ float mk_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// 33차 lever 1: the EXACT activation scale amax/448 and its correctly
// rounded reciprocal. The pow2 scale above wasted up to one of e4m3's three
// mantissa bits on every activation row (the group's amax landed anywhere
// in [224, 448) of the format); the exact scale puts it at 448. v * rsc is
// no longer bit-identical to v / sc, so the Python twin (_mk_quant_x_ref)
// performs the same three fp32 operations, and the FORM matters: torch
// divides a tensor by a Python scalar as a multiply by the scalar's fp32
// reciprocal (its CUDA div kernel), so this is amax * fp32(1/448), not an
// IEEE divide (that one differed by an ulp on some rows and failed the
// exact gate at 4e-3 with 1526 over-ulp elements, 33차 first build). Then
// 1 / scale (__frcp_rn == torch's reciprocal, round-to-nearest), v * rsc,
// and the conversion is SATFINITE on both sides: the row's own amax times
// a rounded reciprocal can land one ulp over 448. A zero (or subnormal)
// row scales by the floor 1e-30 instead of by zero; its quantized bytes
// are zero either way. The pow2 helpers stay for the fixtures that pin
// them.
__device__ __forceinline__ float mk_act_scale(float amax) {
  return fmaxf(amax * (1.0f / 448.0f), 1.0e-30f);
}
__device__ __forceinline__ float mk_act_rcp(float sc) { return __frcp_rn(sc); }
// The four e4m3 bytes of one lane's four values at a pow2 scale -- ONE
// function for the lane's three A quantizers (the gemm prologue, the local
// path, kda p4), so "the same bytes" is the code, not a promise. rsc is the
// exact reciprocal of the pow2 scale: v * rsc is bit-identical to v / sc.
__device__ __forceinline__ uint8_t mk_f32_to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
// BEGIN MK_FP8_PACK4 -- also compiled verbatim by mk_fp8_pack_bench.py.
#ifndef MK_FP8_PACK2_DEF
#define MK_FP8_PACK2_DEF 0
#endif
__device__ __forceinline__ uint32_t mk_f32x4_to_e4m3(
    float x0, float x1, float x2, float x3) {
#if MK_FP8_PACK2_DEF
  // CUDA's x2 converter puts x in the low byte and y in the high byte.
  // Keep the FP32 inputs and SATFINITE/RN conversion; an intermediate half
  // conversion would double-round values near an e4m3 midpoint.
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(
      make_float2(x0, x1), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(
      make_float2(x2, x3), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
#else
  return (uint32_t)__nv_cvt_float_to_fp8(x0, __NV_SATFINITE, __NV_E4M3)
      | ((uint32_t)__nv_cvt_float_to_fp8(x1, __NV_SATFINITE, __NV_E4M3) << 8)
      | ((uint32_t)__nv_cvt_float_to_fp8(x2, __NV_SATFINITE, __NV_E4M3) << 16)
      | ((uint32_t)__nv_cvt_float_to_fp8(x3, __NV_SATFINITE, __NV_E4M3) << 24);
#endif
}
// END MK_FP8_PACK4
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

#ifdef MK_PHASE_TS
__device__ unsigned long long g_mk_ts[MK_GRID_CAP * 8];
__device__ unsigned long long g_mk_mhc_ts[MK_MHC_GRID_CAP * 8];  // mhc phases
// kda: [block][16] -- 0 entry, then (phase k end, barrier k exit) pairs
__device__ __forceinline__ unsigned long long mk_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// accumulated durations (ns), e.g. time spent inside the W pipeline wait
// mhc tail probes ride the gemm stamp array (idle during mhc), slots 1..7
#define MK_MHC_PROBE(slot)                                                   \
  do {                                                                       \
    if (threadIdx.x == 0) g_mk_ts[blockIdx.x * 8 + (slot)] = mk_globaltimer(); \
  } while (0)
#define MK_MHC_TS(slot)                                                      \
  do {                                                                       \
    if (threadIdx.x == 0)                                                    \
      g_mk_mhc_ts[blockIdx.x * 8 + (slot)] = mk_globaltimer();               \
  } while (0)
#else
#define MK_MHC_PROBE(slot) \
  do {                     \
  } while (0)
#define MK_MHC_TS(slot) \
  do {                  \
  } while (0)
#endif

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
// the bytes the mma sees are the persistent lane's (34차 §8: sunset) -- and the raw
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
#ifndef MK_GEMM_TRANSPOSE_M8_DEF
#define MK_GEMM_TRANSPOSE_M8_DEF 0
#endif
#ifndef MK_GEMM_COMPACT_M8_DEF
#define MK_GEMM_COMPACT_M8_DEF 0
#endif
#ifndef MK_M8_FASTPATH_DEF
#define MK_M8_FASTPATH_DEF 0
#endif
constexpr bool MK_COMPACT_M8 = MK_GEMM_TRANSPOSE_M8_DEF && MK_GEMM_COMPACT_M8_DEF;
constexpr int GEMM2_M8_SMEM = MK_COMPACT_M8
    ? MK_SMEM_ALIGN + 2 * 8 * SMEM_A_PITCH + 2 * 32 * 4 + W4_RAW_NBUF2 * W4_RAW_BYTES
    : GEMM2_SMEM;
// Two blocks per SM is the point of this kernel: the SM has 102,400 B of
// shared memory and reserves ~1 KB per resident block.
static_assert(2 * (GEMM2_SMEM + 1024) <= 102400, "v2 must fit twice per SM");
// n <= 40,960: the vocab head is the widest per-rank linear on the lane
// (154,880 / 4 = 38,720 rows -> 303 tiles, 30차 §13); the arrival counters
// below are the only per-tile state, 320 x 4 B each.
constexpr int MK2_TILES_MAX = 320;
constexpr int MK2_KSR_MAX = 8;
// fp32 partials [ksr][m][n]: the widest per-rank linear (in_proj, 6528)
// at m = 32 and ksr = 4; the host lowers ksr for anything wider.
constexpr int MK2_PART_ELEMS = 32 * KDA_INPROJ_N_PAD * 4;
__device__ __align__(16) float g_mk2_partial[MK2_PART_ELEMS];
// 33차 lever 4 (low-rank correction) scratch, one set per stream context:
// t = x @ lr_b^T (32 rows x LR_MAX fp32), the arrival flag of the LR_CTAS
// reducer blocks, and the count of final stores that rearms them.
constexpr int LR_MAX = 32;
constexpr int LR_CTAS = 16;   // reducer blocks per launch (k / 16 each)
constexpr int LR_KS = 128;    // k per staged chunk: 32 x 128 x + 32 x 128 B bf16 = 16 KB
constexpr int LR_PITCH = LR_KS + 8;  // smem row pitch (bf16): +16 B breaks the 32-way
                                     // bank conflict of 256 B rows (rows differ per lane)
// The epilogue's staging after the main loop (the rings are free then):
// t [32][LR_MAX] fp32, this tile's A rows [128][LR_APITCH] bf16 (pitch 34 =
// 17 words: consecutive columns hit consecutive banks), and the correction
// tile [32][128] fp32 the stores add. 4 + 8.5 + 16 KB must fit the ring.
constexpr int LR_APITCH = LR_MAX + 2;
constexpr int LR_EPI_BYTES = 32 * LR_MAX * 4 + 128 * LR_APITCH * 2 + 32 * 128 * 4;
__device__ __align__(16) float g_mk2_lr_t[2][32 * LR_MAX];
__device__ unsigned g_mk2_lr_flag[2];
__device__ unsigned g_mk2_lr_done[2];

// per n-tile slice arrivals, self-rearming (the completing slice resets it)
__device__ unsigned int g_mk2_tile_arrive[MK2_TILES_MAX];
// MK_SEG_SMLP2 hand-off: the fp8 A groups the gate_up launch's pair
// epilogue emits and the down launch stages (a_ready), plus the (gate,
// up) pair arrivals, self-rearming. One set for the device: the same
// no-overlap contract as the partials above.
// 16 B aligned: the down launch stages it with uint4 loads (a uint8_t
// array carries no alignment on its own)
__device__ __align__(16) uint8_t g_mk2_aq[(size_t)KBLK_MAX * 32 * KSTEP];  // 128 KB
__device__ float g_mk2_axs[32 * KBLK_MAX];
__device__ unsigned int g_mk2_pair_arrive[MK2_TILES_MAX];
#ifdef MK_PHASE_TS
// tail units (MKGemm2Ctx::tail) double the grid
constexpr int MK2_UNITS_MAX = MK2_TILES_MAX * MK2_KSR_MAX * 2;  // 1024
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
  const uint8_t* wq4;      // tile-major W4 pack [n/128, k/128, 128, 64]
  const int8_t* ws4;
  float wgs;
  const float* rgs = nullptr;  // per-row 2^-shift (33차 lever 3), 0 = wgs only
  // 33차 lever 4: low-rank error correction out += (x @ lr_b^T) @ lr_a^T.
  // lr_a bf16 [n_pad, lr_r] (row = output column), lr_b bf16 [lr_r, k];
  // lr_r = 0 is off. LR_CTAS extra blocks at the FRONT of the grid reduce
  // t = x @ lr_b^T (m x lr_r, fp32) into g_mk2_lr_t[lr_slot] and raise
  // g_mk2_lr_flag; a tile's final store waits for the flag, adds
  // t[row] . lr_a[col] and the last final store of the launch rearms both
  // (graph replay bakes the same pointers, so the launch cleans up after
  // itself). lr_slot = the launch's stream context (bg).
  const __nv_bfloat16* lr_a = nullptr;
  const __nv_bfloat16* lr_b = nullptr;
  int lr_r = 0;
  int lr_slot = 0;
  int m, n, k, n_orig;
  int ksr;                 // k-slices per tile; grid = (n / 128) * ksr
  // MK_SEG_SMLP2 (the shared-expert MLP as two PDL-chained v2 launches, no
  // grid barrier): the gate_up launch runs with pair_act -- whichever block
  // stores the SECOND final tile of a (gate, up) pair computes the clamped
  // SwiGLU over that pair's 128 columns from the just-stored bf16 rows and
  // emits the fp8 A group + per-row scale into g_mk2_aq / g_mk2_axs -- and
  // the down launch runs with a_ready, staging those groups instead of
  // quantizing x. Its griddepcontrol.wait orders it after the gate_up grid.
  int a_ready = 0;
  int pair_act = 0;
  int n_int = 0;               // gate width = up width = the down launch's k
  float act_limit = 0.0f, act_alpha = 1.0f, act_beta = 0.0f;
};


// RQ = rows each warp quantizes per k-block (1, 2, 4 for m <= 8, 16, 32):
// MT (m-tiles in the mma) and the x lane mapping follow it at compile time,
// so the m <= 16 instantiations carry neither the second m-tile nor the
// four-row quant. The host picks the instantiation from m.
// One reducer block of a corrected v2 launch: its k-range of
// t = x @ lr_b^T, staged LR_KS columns at a time (x rows and lr_b rows,
// bf16), each thread owning (row, four j); atomically added into the
// launch's t scratch, then the arrival flag. Waits on the previous grid
// first (PDL): x is that grid's output and the scratch was rearmed by it.
__device__ __forceinline__ void mk2_lr_partial(const MKGemm2Ctx& c,
                                               uint8_t* smem, int part) {
  MK2_TS(0);  // dispatched (before the PDL wait)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  MK2_TS(1);  // the previous grid is done: x is readable
  __nv_bfloat16* sx = (__nv_bfloat16*)smem;          // [32][LR_PITCH]
  __nv_bfloat16* sB = sx + 32 * LR_PITCH;            // [LR_MAX][LR_PITCH]
  const int r = c.lr_r;
  const int jgs = r / 4;                             // j groups of four
  const int jg = (int)threadIdx.x % jgs;
  const int row = (int)threadIdx.x / jgs;
  const bool active = row < c.m;                     // row < 32 always
  const int kper = c.k / LR_CTAS;                    // k % 128 == 0 -> % 8 == 0
  const int kb = part * kper, ke = kb + kper;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k0 = kb; k0 < ke; k0 += LR_KS) {
    const int kn = min(LR_KS, ke - k0);
    const int q4 = kn / 4;
    __syncthreads();  // the previous chunk is consumed
    for (int i = threadIdx.x; i < c.m * q4; i += MK_THREADS) {
      const int rr = i / q4, cc = (i % q4) * 4;
      *(uint2*)(sx + rr * LR_PITCH + cc) =
          *(const uint2*)(c.x + (size_t)rr * c.k + k0 + cc);
    }
    for (int i = threadIdx.x; i < r * q4; i += MK_THREADS) {
      const int jj = i / q4, cc = (i % q4) * 4;
      *(uint2*)(sB + jj * LR_PITCH + cc) =
          *(const uint2*)(c.lr_b + (size_t)jj * c.k + k0 + cc);
    }
    __syncthreads();
    if (active) {
      const __nv_bfloat16* xr = sx + row * LR_PITCH;
      for (int kk = 0; kk < kn; kk += 2) {
        const __nv_bfloat162 xv = *(const __nv_bfloat162*)(xr + kk);
        const float x0 = __low2float(xv), x1 = __high2float(xv);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const __nv_bfloat162 bv =
              *(const __nv_bfloat162*)(sB + (jg * 4 + q) * LR_PITCH + kk);
          acc[q] += x0 * __low2float(bv) + x1 * __high2float(bv);
        }
      }
    }
  }
  if (active) {
    float* t = g_mk2_lr_t[c.lr_slot] + row * LR_MAX + jg * 4;
#pragma unroll
    for (int q = 0; q < 4; ++q) atomicAdd(t + q, acc[q]);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(&g_mk2_lr_flag[c.lr_slot], 1u);
  MK2_TS(3);  // partial t published
}

// LR: the low-rank-correction instantiation (33차 lever 4). The plain one
// carries none of its code, scratch or waits -- the production path must
// not pay for a lever that is off.
template <int RQ, bool LR, bool COMPACT = false>
__global__ void __launch_bounds__(MK_THREADS, (MK_COMPACT_M8 && COMPACT) ? 3 : 2)
mk_gemm2_kernel(const MKGemm2Ctx c) {
  static_assert(RQ == 1 || RQ == 2 || RQ == 4, "rows per warp");
  constexpr int MT = (RQ == 4) ? 2 : 1;   // m-tiles present
  // C=1 verifies six tokens. Put W on the 16-row MMA operand and X on
  // the 8-column operand: one W[16,32] @ X[8,32]^T instead of two
  // X[16,32] @ W[8,32]^T operations. Quantization and K order are shared.
  constexpr bool TRANSPOSE = MK_GEMM_TRANSPOSE_M8_DEF && RQ == 1 && !LR;
  constexpr int NJ = TRANSPOSE ? 1 : 2;
  static_assert(!COMPACT || (RQ == 1 && !LR), "compact specialization is ordinary M<=8 only");
  constexpr int A_ROWS = (TRANSPOSE && MK_COMPACT_M8 && COMPACT) ? 8 : 32;
  constexpr int LPR = 32 / RQ;            // lanes per quantized row
  constexpr int EPL = KSTEP / LPR;        // x elements per lane: 4, 8, 16
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
  uint8_t* saq = sb0;                                   // [2][A_ROWS][128] swizzled e4m3 A
  float* sxs = (float*)(saq + 2 * A_ROWS * SMEM_A_PITCH);   // [2][32] row scales, wgs folded
  uint8_t* sraw = (uint8_t*)(sxs + 2 * 32);             // [NB][W4_RAW_BYTES]
  __shared__ int s_last;

  // 33차 lever 4 (LR only): the first LR_CTAS blocks of a corrected launch
  // reduce t = x @ lr_b^T and leave; they are the lowest block indices so
  // they are dispatched before any block that could wait on them. The
  // staged t reuses the A ring (saq) after the main loop; s_last is reused
  // as the launch's "last final store" flag.
  if constexpr (LR) {
    if ((int)blockIdx.x < LR_CTAS) {
      mk2_lr_partial(c, sb0, (int)blockIdx.x);
      return;
    }
  }
  static_assert(!LR || GEMM2_SMEM - MK_SMEM_ALIGN >= LR_EPI_BYTES,
                "the low-rank epilogue staging must fit the v2 smem (raise VLLM_GLM53_MK_NBUF2 to 3)");
  float* s_lr_t = (float*)sb0;                                  // [32][LR_MAX]
  __nv_bfloat16* s_lr_a = (__nv_bfloat16*)(sb0 + 32 * LR_MAX * 4);   // [128][LR_APITCH]
  float* s_lr_c = (float*)(sb0 + 32 * LR_MAX * 4 + 128 * LR_APITCH * 2);  // [32][128]
  const int bid = (int)blockIdx.x - (LR ? LR_CTAS : 0);
  const int kblk = c.k / KSTEP;
  const int ksr = c.ksr;
  const int nt = bid / ksr, sp = bid % ksr;
  // ksr <= kblk (host contract), so every slice is non-empty
  const int kb0 = (kblk * sp) / ksr, kbn = (kblk * (sp + 1)) / ksr;
  const int nslices = ksr;      // partials per tile
  const int slice = sp;
  MK2_TS(0);
  constexpr int NB = W4_RAW_NBUF2, DIST = NB - 1;

  // One raw (tile, k-block) record -> ring stage `buf`, all 256 threads.
  // The 16 B nibble chunks land XOR-swizzled by (row >> 1) & 3 so the
  // fragment loads below (eight rows, one word each, 64 B row pitch) hit
  // 32 distinct banks.
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

  // x -> registers and the quant into an A buffer. Warp w quantizes rows
  // w + 8 i (i < RQ); the LPR lanes of group i share row w + 8 i, lane u of
  // the group holds elements [u EPL, u EPL + EPL) -- so the amax is one
  // xor-shuffle chain over LPR lanes that reduces all RQ rows at once
  // (four separate 32-lane chains at m = 32 before). quant_store's
  // arithmetic per element: warp amax, pow2 scale, SATFINITE e4m3, so the
  // tile is byte for byte what the shared prologue publishes.
  const int qw = threadIdx.x >> 5, ql = threadIdx.x & 31;
  const int qrow = qw + MK_WARPS * (ql / LPR);   // this lane's row
  const int qu = ql % LPR;                       // lane within the row group
  uint2 xr[EPL / 4];                             // bf16 x2 per word
  // a_ready: the A group is already e4m3 in g_mk2_aq (16 B per thread:
  // row t >> 3, chunk t & 7) with its pow2 scale in g_mk2_axs
  const int arow = (int)threadIdx.x >> 3, achunk = (int)threadIdx.x & 7;
  uint4 areg = make_uint4(0u, 0u, 0u, 0u);
  float asc = 1.0f;
  auto load_x = [&](int kb) {
    if (c.a_ready) {
      if (arow < c.m) {
        areg = *(const uint4*)(g_mk2_aq + ((size_t)kb * 32 + arow) * KSTEP + achunk * 16);
        if (achunk == 0) asc = g_mk2_axs[arow * KBLK_MAX + kb];
      }
      return;
    }
    if (qrow < c.m) {
      const __nv_bfloat16* src = c.x + (size_t)qrow * c.k + kb * KSTEP + qu * EPL;
#pragma unroll
      for (int w = 0; w < EPL / 4; ++w) xr[w] = *(const uint2*)(src + 4 * w);
    }
  };
  auto quant_x = [&](int buf) {
    if (c.a_ready) {  // stage the published group; nothing to quantize
      if (arow < c.m) {
        uint8_t* dst = saq + buf * (A_ROWS * SMEM_A_PITCH) +
                       (arow >> 4) * 16 * SMEM_A_PITCH + (arow & 15) * SMEM_A_PITCH;
        *(uint4*)(dst + mk_swz(arow & 15, achunk * 16)) = areg;
        if (achunk == 0) sxs[buf * 32 + arow] = asc * c.wgs;
      }
      return;
    }
    float v[EPL];
    float mx = 0.0f;  // rows past m reduce a zero and store nothing
    if (qrow < c.m) {
#pragma unroll
      for (int w = 0; w < EPL / 4; ++w) {
        const __nv_bfloat16* pv = (const __nv_bfloat16*)&xr[w];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          v[4 * w + q] = __bfloat162float(pv[q]);
          mx = fmaxf(mx, fabsf(v[4 * w + q]));
        }
      }
    }
    if constexpr (MK_M8_FASTPATH_DEF && TRANSPOSE) {
      // mx starts at +0 and fmax(fabs(x)) is nonnegative and not NaN.
      // Unsigned IEEE bit order therefore equals float order, including
      // +inf. All 32 lanes of this one-row warp participate, even past m.
      mx = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(mx)));
    } else {
#pragma unroll
      for (int off = LPR / 2; off; off >>= 1)  // stays inside the row's lane group
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    if (qrow >= c.m) return;
    const float sc = mk_act_scale(mx);
    // one rcp.rn per row (the IEEE divide's slow-path call showed on every
    // row of every k-block in the SASS); the twin does v * (1 / sc) too
    const float rsc = mk_act_rcp(sc);
    uint8_t* dst = saq + buf * (A_ROWS * SMEM_A_PITCH) +
                   (qrow >> 4) * 16 * SMEM_A_PITCH + (qrow & 15) * SMEM_A_PITCH +
                   mk_swz(qrow & 15, qu * EPL);  // EPL <= 16: inside one chunk
#pragma unroll
    for (int w = 0; w < EPL / 4; ++w) {
      const uint32_t pack = mk_f32x4_to_e4m3(
          v[4 * w] * rsc, v[4 * w + 1] * rsc,
          v[4 * w + 2] * rsc, v[4 * w + 3] * rsc);
      *(uint32_t*)(dst + 4 * w) = pack;
    }
    if (qu == 0) sxs[buf * 32 + qrow] = sc * c.wgs;
  };

  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q = lane & 3;
  const int warp = threadIdx.x >> 5;

  float acc[MT][NJ][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

  // mma over one k-block. The mma's k axis is a permutation of the block's
  // 128 elements, the same on A and B: lane q of a quad owns natural
  // elements [32 q, 32 q + 32) -- W's raw chunk q, e2m1 groups 2q and
  // 2q+1 -- and at step ks feeds the 8-element word (ks + q) & 3 of them as
  // b0/b1 (a0..a3 read the same elements of A). Two groups per lane per
  // row means two LUT pairs per row per k-block instead of eight (the
  // fragment-major mapping rebuilt one per fragment: 216 of the ~670
  // instructions per warp per k-block on the SASS), and the rotated word
  // puts the quad's four raw loads on four different banks. The in-mma
  // summation order changes with the permutation, so v2 is no longer
  // bit-identical to the persistent lane on unsplit shapes; the exact gate
  // (<= 1 bf16 ulp of the fp32 reference) is the contract.
  auto mma_fold = [&](int rbuf, int abuf) {
    const uint8_t* rr = sraw + rbuf * W4_RAW_BYTES;
    const uint8_t* sa = saq + abuf * (A_ROWS * SMEM_A_PITCH);
    float kacc[MT][NJ][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) kacc[i][j][e] = 0.0f;
    // per W row: the LUT pairs of groups 2q (words 0, 1) and 2q+1 (2, 3)
    uint32_t l0a[2], l1a[2], l0b[2], l1b[2];
    int slot[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int nrow = warp * 16 + j * 8 + g;
      uint32_t ea, eb;
      if constexpr (MK_M8_FASTPATH_DEF && TRANSPOSE) {
        // Each lane needs only the two exponents for groups 2q and 2q+1.
        // This aligned halfword avoids a 64-bit load, select and shift.
        const uint32_t ex = *(const uint16_t*)(rr + W4_RAW_NIB + nrow * 8 + 2 * q);
        ea = ex & 0xFFu;
        eb = ex >> 8;
      } else {
        const uint2 sb = *(const uint2*)(rr + W4_RAW_NIB + nrow * 8);
        const uint32_t sw = (q < 2) ? sb.x : sb.y;
        ea = (sw >> (16 * (q & 1))) & 0xFFu;
        eb = (sw >> (16 * (q & 1) + 8)) & 0xFFu;
      }
      const unsigned long long la =
          ((ea & 7u) - 1u) < 5u ? MK_E2M1_LUT64_B : MK_E2M1_LUT64;
      const unsigned long long lb =
          ((eb & 7u) - 1u) < 5u ? MK_E2M1_LUT64_B : MK_E2M1_LUT64;
      l0a[j] = __vadd4((uint32_t)la, ea * 0x01010100u);
      l1a[j] = __vadd4((uint32_t)(la >> 32), ea * 0x01010101u);
      l0b[j] = __vadd4((uint32_t)lb, eb * 0x01010100u);
      l1b[j] = __vadd4((uint32_t)(lb >> 32), eb * 0x01010101u);
      slot[j] = nrow * W4_RAW_PITCH + ((q ^ ((nrow >> 1) & 3)) << 4);
    }
#pragma unroll
    for (int ks = 0; ks < KSTEP / 32; ++ks) {
      const int wsel = (ks + q) & 3;             // this lane's word this step
      const int koff = 32 * q + 8 * wsel;        // its natural elements
      uint32_t a[MT][4];
      uint32_t wb[2][2];
      if constexpr (!TRANSPOSE) {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const uint8_t* base = sa + i * 16 * SMEM_A_PITCH;
          const int o0 = mk_swz(g, koff), o1 = mk_swz(g, koff + 4);
          a[i][0] = *(const uint32_t*)(base + g * SMEM_A_PITCH + o0);
          a[i][1] = *(const uint32_t*)(base + (g + 8) * SMEM_A_PITCH + o0);
          a[i][2] = *(const uint32_t*)(base + g * SMEM_A_PITCH + o1);
          a[i][3] = *(const uint32_t*)(base + (g + 8) * SMEM_A_PITCH + o1);
        }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint32_t w = *(const uint32_t*)(rr + slot[j] + 4 * wsel);
        const uint32_t l0 = (wsel < 2) ? l0a[j] : l0b[j];
        const uint32_t l1 = (wsel < 2) ? l1a[j] : l1b[j];
        const uint32_t b0 = __byte_perm(l0, l1, w & 0x7777u) |
                            __byte_perm(0x8000u, 0u, (w >> 3) & 0x1111u);
        const uint32_t b1 = __byte_perm(l0, l1, (w >> 16) & 0x7777u) |
                            __byte_perm(0x8000u, 0u, (w >> 19) & 0x1111u);
        if constexpr (TRANSPOSE) {
          wb[j][0] = b0;
          wb[j][1] = b1;
        } else {
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
      if constexpr (TRANSPOSE) {
        // Invalid columns must be zero; the quantizer publishes only m
        // rows and CUDA graph replay may leave older rows in this buffer.
        uint32_t x0 = 0, x1 = 0;
        if (g < c.m) {
          x0 = *(const uint32_t*)(sa + g * SMEM_A_PITCH + mk_swz(g, koff));
          x1 = *(const uint32_t*)(sa + g * SMEM_A_PITCH + mk_swz(g, koff + 4));
        }
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(kacc[0][0][0]), "+f"(kacc[0][0][1]),
              "+f"(kacc[0][0][2]), "+f"(kacc[0][0][3])
            : "r"(wb[0][0]), "r"(wb[1][0]), "r"(wb[0][1]), "r"(wb[1][1]),
              "r"(x0), "r"(x1));
      }
    }
    if constexpr (TRANSPOSE) {
      const float s0 = (2 * q < c.m) ? sxs[abuf * 32 + 2 * q] : 0.0f;
      const float s1 = (2 * q + 1 < c.m) ? sxs[abuf * 32 + 2 * q + 1] : 0.0f;
      acc[0][0][0] += kacc[0][0][0] * s0;
      acc[0][0][1] += kacc[0][0][1] * s1;
      acc[0][0][2] += kacc[0][0][2] * s0;
      acc[0][0][3] += kacc[0][0][3] * s1;
    } else {
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
    }
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

  // pair_act: this block just stored tile nt's FINAL bf16 rows (a whole
  // tile, or the fold as the last-arriving slice). Count the pair's
  // arrival; the block completing the pair reads both tiles' rows back
  // (__ldcg: the other tile came from another SM) and emits the fp8 A
  // group for the down launch. Release/acquire as in the slice fold:
  // fence before the arrival, fence after winning it. The activation is
  // clamp, fp32 silu x (up + beta), bf16 round -- the rounding the stock
  // chain has --
  // then the per-row pow2 quant.
  __shared__ int s_pair_last;
  auto pair_finish = [&](int tile) {
    __syncthreads();
    __threadfence();
    __syncthreads();
    const int groups = c.n_int / KSTEP;
    const int pair = (tile < groups) ? tile : tile - groups;
    if (threadIdx.x == 0) {
      const unsigned prev = atomicAdd(&g_mk2_pair_arrive[pair], 1u);
      s_pair_last = (prev + 1u == 2u);
      if (s_pair_last) g_mk2_pair_arrive[pair] = 0u;  // both arrived; rearm
    }
    __syncthreads();
    if (s_pair_last) {
      __threadfence();
      for (int t = warp; t < c.m; t += MK_WARPS) {  // one warp per row
        const size_t gb = (size_t)t * c.n_orig + (size_t)pair * KSTEP + lane * 4;
        const uint2 gr = __ldcg((const uint2*)(c.out + gb));
        const uint2 ur = __ldcg((const uint2*)(c.out + gb + c.n_int));
        const __nv_bfloat16* gp = (const __nv_bfloat16*)&gr;
        const __nv_bfloat16* up = (const __nv_bfloat16*)&ur;
        float v[4], amax = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float gv = __bfloat162float(gp[e]), uv = __bfloat162float(up[e]);
          if (c.act_limit > 0.0f) {
            gv = fminf(gv, c.act_limit);
            uv = fminf(fmaxf(uv, -c.act_limit), c.act_limit);
          }
          v[e] = __bfloat162float(__float2bfloat16(
              gv * mk_sigmoid(c.act_alpha * gv) * (uv + c.act_beta)));
          amax = fmaxf(amax, fabsf(v[e]));
        }
#pragma unroll
        for (int off = 16; off; off >>= 1)
          amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, off));
        const float sc = mk_act_scale(amax);
        const float rsc = mk_act_rcp(sc);  // 33차 lever 1: exact scale  // exact: sc is a power of two
        const uint32_t pack = mk_f32x4_to_e4m3(
            v[0] * rsc, v[1] * rsc, v[2] * rsc, v[3] * rsc);
        *(uint32_t*)(g_mk2_aq + ((size_t)pair * 32 + t) * KSTEP + lane * 4) = pack;
        if (lane == 0) g_mk2_axs[t * KBLK_MAX + pair] = sc;
      }
    }
  };

  // ---- epilogue: one walk over the fragment's real rows / cols, two stores
  auto store_tile = [&](auto&& put) {  // put(row, col, value)
    if constexpr (TRANSPOSE) {
      const int cb = nt * 128 + warp * 16 + g;
      if (2 * q < c.m) {
        put(2 * q, cb, acc[0][0][0]);
        put(2 * q, cb + 8, acc[0][0][2]);
      }
      if (2 * q + 1 < c.m) {
        put(2 * q + 1, cb, acc[0][0][1]);
        put(2 * q + 1, cb + 8, acc[0][0][3]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int r0 = i * 16 + g, r1 = r0 + 8;
          const int cb = nt * 128 + warp * 16 + j * 8 + (lane & 3) * 2;
          if (r0 < c.m) { put(r0, cb, acc[i][j][0]); put(r0, cb + 1, acc[i][j][1]); }
          if (r1 < c.m) { put(r1, cb, acc[i][j][2]); put(r1, cb + 1, acc[i][j][3]); }
        }
      }
    }
  };
  // ---- 33차 lever 4: wait for the reducer blocks' t, stage it, and add
  // t[row] . lr_a[col] to every final store; the launch's last final store
  // rearms the scratch for the next launch (graph replay reuses it).
  // After the main loop (every thread past it: the first __syncthreads
  // below), the smem rings are free: stage t and THIS tile's 128 rows of
  // lr_a with coalesced loads, then compute the correction tile
  // corr[r][cc] = t[r] . A[nt*128 + cc] cooperatively from smem, so the
  // stores add one smem read each. (The first form -- each store walking
  // its A row from L2 -- was latency-bound: 30-50 us per tile on the
  // stamps, more than the main loop itself.)
  auto lr_wait = [&]() {
    if (threadIdx.x == 0) {
      MK_SPIN_WAIT(*((volatile unsigned*)&g_mk2_lr_flag[c.lr_slot]) < (unsigned)LR_CTAS, 64, "gemm2 lr flag");
      __threadfence();
    }
    __syncthreads();   // flag seen; the main loop's smem reads are all done
    const float* t = g_mk2_lr_t[c.lr_slot];
    for (int i = threadIdx.x; i < 32 * LR_MAX; i += MK_THREADS) s_lr_t[i] = __ldcg(t + i);
    const int rq = c.lr_r / 2;   // bf16 pairs per A row
    for (int i = threadIdx.x; i < 128 * rq; i += MK_THREADS) {
      const int row = i / rq, q = i - row * rq;
      *(__nv_bfloat162*)(s_lr_a + row * LR_APITCH + q * 2) =
          *(const __nv_bfloat162*)(c.lr_a + ((size_t)nt * 128 + row) * c.lr_r + q * 2);
    }
    __syncthreads();
    const int cc = (int)threadIdx.x & 127, r0 = (int)threadIdx.x >> 7;
    const __nv_bfloat16* a = s_lr_a + cc * LR_APITCH;
    for (int r = r0; r < c.m; r += 2) {
      const float* t_r = s_lr_t + r * LR_MAX;
      float acc_lr = 0.0f;
      for (int j = 0; j < c.lr_r; j += 2) {
        const __nv_bfloat162 av = *(const __nv_bfloat162*)(a + j);
        acc_lr += t_r[j] * __low2float(av) + t_r[j + 1] * __high2float(av);
      }
      s_lr_c[r * 128 + cc] = acc_lr;
    }
    __syncthreads();
  };
  auto lr_term = [&](int r, int col) -> float {
    return s_lr_c[r * 128 + (col - nt * 128)];
  };
  auto lr_done = [&]() {
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned prev = atomicAdd(&g_mk2_lr_done[c.lr_slot], 1u);
      s_last = (prev + 1u == (unsigned)(c.n / SMEM_W_ROWS)) ? 1 : 0;
    }
    __syncthreads();
    if (s_last) {
      float* t = g_mk2_lr_t[c.lr_slot];
      for (int i = threadIdx.x; i < 32 * LR_MAX; i += MK_THREADS) t[i] = 0.0f;
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) {
        g_mk2_lr_done[c.lr_slot] = 0u;
        g_mk2_lr_flag[c.lr_slot] = 0u;
      }
    }
  };
  if (nslices == 1) {  // whole tile: bf16 out
    if constexpr (LR) lr_wait();
    store_tile([&](int r, int col, float v) {
      if (col < c.n_orig) {
        float o = v * (c.rgs ? c.rgs[col] : 1.0f);
        if constexpr (LR) o += lr_term(r, col);
        c.out[(size_t)r * c.n_orig + col] = __float2bfloat16(o);
      }
    });
    if (c.pair_act) pair_finish(nt);  // the tile's final store was just made
    if constexpr (LR) lr_done();
    MK2_TS(3);
    return;
  }
  // k-slice: assign (never accumulate) this slice's partial, count the
  // arrival, and let the last slice fold the tile in slice order.
  {
    float* pb = g_mk2_partial + (size_t)slice * c.m * c.n;
    store_tile([&](int r, int col, float v) { pb[(size_t)r * c.n + col] = v; });
  }
  __syncthreads();
  __threadfence();  // release: the slice is visible device-wide first
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned prev = atomicAdd(&g_mk2_tile_arrive[nt], 1u);
    s_last = (prev + 1u == (unsigned)nslices);
    if (s_last) g_mk2_tile_arrive[nt] = 0u;  // all slices in; rearm
  }
  __syncthreads();
  if (s_last) {
    __threadfence();  // acquire: the other slices' partials
    if constexpr (LR) lr_wait();
    for (int i2 = threadIdx.x; i2 < c.m * 32; i2 += MK_THREADS) {
      const int r = i2 >> 5, c4 = (i2 & 31) * 4;
      const float* src = g_mk2_partial + (size_t)r * c.n + nt * 128 + c4;
      float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int s = 0; s < nslices; ++s) {  // fixed order -> reproducible
        const float4 pv = __ldcg((const float4*)(src + (size_t)s * c.m * c.n));
        v4.x += pv.x; v4.y += pv.y; v4.z += pv.z; v4.w += pv.w;
      }
      const int col = nt * 128 + c4;
      const float4 rg = c.rgs ? *(const float4*)(c.rgs + col)
                              : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
      __nv_bfloat16* o = c.out + (size_t)r * c.n_orig + col;
      v4.x *= rg.x; v4.y *= rg.y; v4.z *= rg.z; v4.w *= rg.w;
      if constexpr (LR) {
        v4.x += lr_term(r, col); v4.y += lr_term(r, col + 1);
        v4.z += lr_term(r, col + 2); v4.w += lr_term(r, col + 3);
      }
      if (col < c.n_orig) o[0] = __float2bfloat16(v4.x);
      if (col + 1 < c.n_orig) o[1] = __float2bfloat16(v4.y);
      if (col + 2 < c.n_orig) o[2] = __float2bfloat16(v4.z);
      if (col + 3 < c.n_orig) o[3] = __float2bfloat16(v4.w);
    }
    if (c.pair_act) pair_finish(nt);  // the fold was this tile's final store
    if constexpr (LR) lr_done();
  }
  MK2_TS(3);
}

// ===========================================================================
// MK_SEG_MHC -- fused hc_post + hc_pre (+ RMSNorm), T <= MAX_TOK.
// Port of mhc_fused_tilelang + mhc_pre_big_fuse_with_norm_tilelang
// (overlay/modules/glm53_kernels/tilelang_kernels.py). Every rounding
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
        // Publication can wait, but reuse of part[][] cannot: every
        // warp must finish reducing this token before any warp stores
        // the next token's rows into the same shared tile.
        __syncthreads();
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
      MK_SPIN_WAIT(*v < (unsigned int)NCHUNK, 128, "mhc token arrive");
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
  asm volatile("griddepcontrol.launch_dependents;");  // PDL: dependents may launch
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

// Exact-selection prefill pair reuse (selection exact; tile order differs
// from per-query attention, so output is not bit-exact). The former UNION
// lane scanned the physical cache span and synchronized min/max to the CPU.
// This schedule is bounded by two top-k lists: shared hashing, multiset
// union, independent membership, and no materialized KV or host read of
// device metadata.
constexpr int MLA_PAIR_HASH = 8192;
constexpr int MLA_PAIR_MAX_W = 2176;
// keys + counts + three control words: total, cursor, failed.
constexpr int MLA_PAIR_PREP_SMEM = (2 * MLA_PAIR_HASH + 3) * 4;
constexpr int MLA_PAIR_SMEM = MLA_SMEM_RING + 2 * MLA_SMEM_S
                             + 2 * MLA_SMEM_P + 2 * MLA_SMEM_C;
constexpr int MLA_GROUP4_THREADS = 2 * MK_THREADS;
constexpr int MLA_GROUP4_PREP_SMEM = MLA_PAIR_HASH * 12 + 12;
// Weak-overlap groups process two independent rows at a time in two
// disjoint pair-sized regions. Strong groups share one FP8 ring and need
// only 45KB, but reserve this 70.7KB maximum for the exact fallback.
constexpr int MLA_GROUP4_SMEM = 2 * MLA_PAIR_SMEM;
static_assert(MLA_GROUP4_PREP_SMEM <= 99 * 1024,
              "four-row hash must fit the SM121 opt-in shared-memory cap");
static_assert(MLA_SMEM_RING + 4 * (MLA_SMEM_S + MLA_SMEM_P + MLA_SMEM_C)
                  <= MLA_GROUP4_SMEM,
              "four-row attention must fit the fallback shared allocation");
static_assert(MLA_PAIR_MAX_W < 65536, "each packed multiplicity field is 16 bits");
static_assert(2 * MLA_PAIR_MAX_W < MLA_PAIR_HASH,
              "two full-width top-k lists must leave a free probe slot; a "
              "wider W needs a larger table (or the bounded-probe fallback)");

struct MKMlaPairArgs {
  MKMlaArgs a;
  int* pair_slots;          // [ceil(T/group_width), group_width*W], multiset union
  int* membership;          // one independent bit per group row and occurrence
  int* pair_lens;           // -1: insufficient reuse, process original rows
  int groups;
};

__global__ __launch_bounds__(MK_THREADS) void mk_mla_pair_prepare(const MKMlaPairArgs p) {
  extern __shared__ __align__(16) unsigned int table[];
  unsigned int* keys = table;
  unsigned int* counts = table + MLA_PAIR_HASH;
  unsigned int* total = counts + MLA_PAIR_HASH;
  unsigned int* cursor = total + 1;
  unsigned int* failed = cursor + 1;
  const int group = blockIdx.x, t = group * 2;
  const int lane = threadIdx.x & 31;
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int n0 = p.a.lens[t];
  const int n1 = t + 1 < p.a.T ? p.a.lens[t + 1] : 0;
  if (n0 <= 0 || n1 <= 0) {
    if (threadIdx.x == 0) p.pair_lens[group] = -1;
    return;
  }
  for (int i = threadIdx.x; i < MLA_PAIR_HASH; i += MK_THREADS) {
    keys[i] = 0xffffffffu;
    counts[i] = 0;
  }
  if (threadIdx.x == 0) { *total = 0; *cursor = 0; *failed = 0; }
  __syncthreads();
  for (int j = threadIdx.x; j < n0 + n1; j += MK_THREADS) {
    const int row = j >= n0;
    const int col = row ? j - n0 : j;
    const unsigned int slot = (unsigned int)p.a.slots[(size_t)(t + row) * p.a.W + col];
    unsigned int h = (slot * 2654435761u) & (MLA_PAIR_HASH - 1);
    // Bounded like group4: static_assert above proves the table cannot fill
    // under the enforced W cap, but a raised cap must degrade to original-list
    // fallback here, never hang the device on a full-table probe.
    bool inserted = false;
#pragma unroll 1
    for (int probe = 0; probe < 64; ++probe) {
      const unsigned int old = atomicCAS(keys + h, 0xffffffffu, slot);
      if (old == 0xffffffffu || old == slot) {
        atomicAdd(counts + h, row ? 65536u : 1u);
        inserted = true;
        break;
      }
      h = (h + 1) & (MLA_PAIR_HASH - 1);
    }
    if (!inserted) atomicExch(failed, 1u);
  }
  __syncthreads();
  if (*failed) {
    if (threadIdx.x == 0) p.pair_lens[group] = -1;
    return;
  }
  unsigned int local = 0;
  for (int i = threadIdx.x; i < MLA_PAIR_HASH; i += MK_THREADS) {
    const unsigned int c = counts[i];
    local += max(c & 65535u, c >> 16);
  }
  // A multiset union preserves repeated selected slots rather than silently
  // deduplicating their probability mass. Width <=2176 keeps each count
  // safely inside its 16-bit half.
  atomicAdd(total, local);
  __syncthreads();
  // Pay the wider score/output work only if at least 25% of KV loads vanish.
  // Weak-overlap pairs keep their original lists and reduction order.
  if (*total * 4 > (unsigned int)(n0 + n1) * 3) {
    if (threadIdx.x == 0) p.pair_lens[group] = -1;
    return;
  }
  for (int i = threadIdx.x; i < MLA_PAIR_HASH; i += MK_THREADS) {
    const unsigned int c = counts[i];
    const unsigned int c0 = c & 65535u, c1 = c >> 16;
    const unsigned int copies = max(c0, c1);
    unsigned int prefix = copies;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      const unsigned int prev = __shfl_up_sync(0xffffffff, prefix, offset);
      if (lane >= offset) prefix += prev;
    }
    const unsigned int warp_count = __shfl_sync(0xffffffff, prefix, 31);
    unsigned int base = 0;
    if (lane == 0 && warp_count) base = atomicAdd(cursor, warp_count);
    base = __shfl_sync(0xffffffff, base, 0) + prefix - copies;
    for (unsigned int repeat = 0; repeat < copies; ++repeat) {
      const size_t dst = (size_t)group * (2 * p.a.W) + base + repeat;
      p.pair_slots[dst] = (int)keys[i];
      p.membership[dst] = (repeat < c0 ? 1 : 0) | (repeat < c1 ? 2 : 0);
    }
  }
  if (threadIdx.x == 0) p.pair_lens[group] = (int)*total;
}

// Four-query scheduling uses 64-bit packed counts (four 16-bit fields),
// with an 8192-key table bounded independently of cache slot IDs. Four
// disjoint W2176 rows can exceed table capacity: bounded probing marks the
// entire group for original-list fallback rather than hanging or dropping
// a selected occurrence. The 98,316-byte workspace fits SM121's 99KiB cap.
__global__ __launch_bounds__(MK_THREADS) void mk_mla_group4_prepare(const MKMlaPairArgs p) {
  extern __shared__ __align__(16) unsigned int table4[];
  unsigned int* keys = table4;
  unsigned long long* counts = (unsigned long long*)(keys + MLA_PAIR_HASH);
  unsigned int* total = (unsigned int*)(counts + MLA_PAIR_HASH);
  unsigned int* cursor = total + 1;
  unsigned int* failed = total + 2;
  const int group = blockIdx.x, t = group * 4;
  const int lane = threadIdx.x & 31;
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (t + 3 >= p.a.T) {
    if (threadIdx.x == 0) p.pair_lens[group] = -1;
    return;
  }
  const int n0 = p.a.lens[t], n1 = p.a.lens[t + 1];
  const int n2 = p.a.lens[t + 2], n3 = p.a.lens[t + 3];
  if (n0 <= 0 || n1 <= 0 || n2 <= 0 || n3 <= 0) {
    if (threadIdx.x == 0) p.pair_lens[group] = -1;
    return;
  }
  for (int i = threadIdx.x; i < MLA_PAIR_HASH; i += MK_THREADS) {
    keys[i] = 0xffffffffu; counts[i] = 0;
  }
  if (threadIdx.x == 0) { *total = 0; *cursor = 0; *failed = 0; }
  __syncthreads();
  const int selected_count = n0 + n1 + n2 + n3;
  for (int j = threadIdx.x; j < selected_count; j += MK_THREADS) {
    int row, col;
    if (j < n0) { row = 0; col = j; }
    else if (j < n0 + n1) { row = 1; col = j - n0; }
    else if (j < n0 + n1 + n2) { row = 2; col = j - n0 - n1; }
    else { row = 3; col = j - n0 - n1 - n2; }
    const unsigned int slot = (unsigned int)p.a.slots[(size_t)(t + row) * p.a.W + col];
    unsigned int h = (slot * 2654435761u) & (MLA_PAIR_HASH - 1);
    bool inserted = false;
#pragma unroll 1
    for (int probe = 0; probe < 64; ++probe) {
      const unsigned int old = atomicCAS(keys + h, 0xffffffffu, slot);
      if (old == 0xffffffffu || old == slot) {
        atomicAdd(counts + h, 1ull << (16 * row));
        inserted = true;
        break;
      }
      h = (h + 1) & (MLA_PAIR_HASH - 1);
    }
    if (!inserted) atomicExch(failed, 1u);
  }
  __syncthreads();
  if (*failed) {
    if (threadIdx.x == 0) p.pair_lens[group] = -1;
    return;
  }
  unsigned int local = 0;
  for (int i = threadIdx.x; i < MLA_PAIR_HASH; i += MK_THREADS) {
    const unsigned long long c = counts[i];
    const unsigned int c0 = c & 65535ull, c1 = (c >> 16) & 65535ull;
    const unsigned int c2 = (c >> 32) & 65535ull, c3 = c >> 48;
    local += max(max(c0, c1), max(c2, c3));
  }
  atomicAdd(total, local);
  __syncthreads();
  if (*total * 4 > (unsigned int)selected_count * 3) {
    if (threadIdx.x == 0) p.pair_lens[group] = -1;
    return;
  }
  for (int i = threadIdx.x; i < MLA_PAIR_HASH; i += MK_THREADS) {
    const unsigned long long c = counts[i];
    const unsigned int c0 = c & 65535ull, c1 = (c >> 16) & 65535ull;
    const unsigned int c2 = (c >> 32) & 65535ull, c3 = c >> 48;
    const unsigned int copies = max(max(c0, c1), max(c2, c3));
    unsigned int prefix = copies;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      const unsigned int prev = __shfl_up_sync(0xffffffff, prefix, offset);
      if (lane >= offset) prefix += prev;
    }
    const unsigned int warp_count = __shfl_sync(0xffffffff, prefix, 31);
    unsigned int base = 0;
    if (lane == 0 && warp_count) base = atomicAdd(cursor, warp_count);
    base = __shfl_sync(0xffffffff, base, 0) + prefix - copies;
    for (unsigned int repeat = 0; repeat < copies; ++repeat) {
      const size_t dst = (size_t)group * (4 * p.a.W) + base + repeat;
      p.pair_slots[dst] = (int)keys[i];
      p.membership[dst] = (repeat < c0 ? 1 : 0) | (repeat < c1 ? 2 : 0)
                         | (repeat < c2 ? 4 : 0) | (repeat < c3 ? 8 : 0);
    }
  }
  if (threadIdx.x == 0) p.pair_lens[group] = (int)*total;
}

template <bool SUBGROUP>
__device__ __forceinline__ void mla_group_sync() {
  if constexpr (SUBGROUP) {
    // Fallback query pairs may have different loop lengths. Each half of
    // the 512-thread CTA owns a distinct named barrier and shared region.
    // The arrival count is a PTX immediate, so it is pinned to MK_THREADS
    // rather than derived from MLA_GROUP4_THREADS.
    static_assert(MK_THREADS == 256,
                  "subgroup barrier arrival count must match half the CTA");
    const int barrier_id = 1 + (threadIdx.x / MK_THREADS);
    asm volatile("bar.sync %0, 256;" :: "r"(barrier_id) : "memory");
  } else {
    __syncthreads();
  }
}

// Keep the only KV copy as FP8, as v5 does. Two shared Q tiles would force
// one CTA/SM; Q fragments instead use the read-only cache. This adds ~T*16KB
// of unique Q traffic versus T*W*512B of original KV traffic. ROWS is static
// so the two output accumulator arrays stay register-indexed.
template <int ROWS, int GROUP_ROWS = ROWS, bool SUBGROUP = false>
__device__ __forceinline__ void mla_pair_attention(
    const MKMlaPairArgs& p, int t, int length, const int* selected,
    const int* bits, unsigned char* smem) {
  const MKMlaArgs& a = p.a;
  constexpr int shared_rows = GROUP_ROWS == 4 ? 4 : 2;
  constexpr int load_warps = GROUP_ROWS == 4 ? 16 : MLA_WARPS;
  uint8_t* ring = smem;
  float* ss = (float*)(ring + MLA_SMEM_RING);
  __nv_bfloat16* sp = (__nv_bfloat16*)((uint8_t*)ss + shared_rows * MLA_SMEM_S);
  float* scorr = (float*)((uint8_t*)sp + shared_rows * MLA_SMEM_P);
  const int lane = threadIdx.x & 31, cta_warp = threadIdx.x >> 5;
  const int warp = cta_warp % MLA_WARPS;
  const int row_base = GROUP_ROWS == 4 ? (cta_warp / MLA_WARPS) * ROWS : 0;
  const int load_warp = GROUP_ROWS == 4 ? cta_warp : warp;
  const int g = lane >> 2, q4 = lane & 3;
  float acc[ROWS][8][4];
  float m0[ROWS], m1[ROWS], l0[ROWS], l1[ROWS];
#pragma unroll
  for (int row = 0; row < ROWS; ++row) {
    m0[row] = m1[row] = -INFINITY;
    l0[row] = l1[row] = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
      acc[row][nt][0] = acc[row][nt][1] = acc[row][nt][2] = acc[row][nt][3] = 0.f;
  }
  const int ntile = (length + MLA_TILE - 1) / MLA_TILE;
  auto issue = [&](int ti) {
    uint8_t* dst = ring + (size_t)(ti % MLA_NSTAGE) * MLA_TILE * MLA_RP;
#pragma unroll
    for (int r = 0; r < MLA_TILE / load_warps; ++r) {
      const int k = r * load_warps + load_warp;
      const int j = ti * MLA_TILE + k;
      const int slot = selected[j < length ? j : 0];
      mk_cp_async16(dst + (size_t)k * MLA_RP + lane * 16,
                    a.ckv + (size_t)slot * MLA_D + lane * 16);
    }
    mk_cp_commit();
  };
#pragma unroll 1
  for (int ti = 0; ti < MLA_NSTAGE - 1; ++ti) {
    if (ti < ntile) issue(ti); else mk_cp_commit();
  }
#pragma unroll 1
  for (int ti = 0; ti < ntile; ++ti) {
    mk_cp_wait<MLA_NSTAGE - 2>();
    mla_group_sync<SUBGROUP>();
    if (ti + MLA_NSTAGE - 1 < ntile) issue(ti + MLA_NSTAGE - 1);
    else mk_cp_commit();
    const uint8_t* tile8 = ring + (size_t)(ti % MLA_NSTAGE) * MLA_TILE * MLA_RP;
    const int kmax = min(MLA_TILE, length - ti * MLA_TILE);
#pragma unroll
    for (int row = 0; row < ROWS; ++row) {
      const int n0 = (warp % MLA_NG) * 8, kq = warp / MLA_NG;
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
      const __nv_bfloat16* qa = a.q + ((size_t)(t + row_base + row) * MLA_H + g) * MLA_D;
      const uint8_t* cb = tile8 + (size_t)(n0 + g) * MLA_RP;
#pragma unroll
      for (int ks = 0; ks < (MLA_D / 16) / MLA_KQ; ++ks) {
        const int k0 = kq * (MLA_D / MLA_KQ) + ks * 16 + q4 * 2;
        mla_mma_bf16(c0, c1, c2, c3,
          __ldg((const uint32_t*)(qa + k0)),
          __ldg((const uint32_t*)(qa + 8 * MLA_D + k0)),
          __ldg((const uint32_t*)(qa + k0 + 8)),
          __ldg((const uint32_t*)(qa + 8 * MLA_D + k0 + 8)),
          mla_e4m3x2(cb + k0), mla_e4m3x2(cb + k0 + 8));
      }
      float* sh = ss + (row_base + row) * (MLA_SMEM_S / sizeof(float)) + kq * MLA_H * MLA_TILE;
      sh[g * MLA_TILE + n0 + q4 * 2] = c0;
      sh[g * MLA_TILE + n0 + q4 * 2 + 1] = c1;
      sh[(g + 8) * MLA_TILE + n0 + q4 * 2] = c2;
      sh[(g + 8) * MLA_TILE + n0 + q4 * 2 + 1] = c3;
    }
    mla_group_sync<SUBGROUP>();
#pragma unroll
    for (int row = 0; row < ROWS; ++row) {
      const int h0 = warp * 2;
      bool ok = lane < kmax;
      if constexpr (ROWS == 2) {
        const int owned = ok ? bits[ti * MLA_TILE + lane] : 0;
        ok = ok && ((owned & (1 << (row_base + row))) != 0);
      }
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int kq = 0; kq < MLA_KQ; ++kq) {
        const float* sh = ss + (row_base + row) * (MLA_SMEM_S / sizeof(float)) + kq * MLA_H * MLA_TILE;
        s0 += sh[h0 * MLA_TILE + lane % MLA_TILE];
        s1 += sh[(h0 + 1) * MLA_TILE + lane % MLA_TILE];
      }
      s0 = ok ? s0 * (a.sm_scale * a.ckv_scale) : -INFINITY;
      s1 = ok ? s1 * (a.sm_scale * a.ckv_scale) : -INFINITY;
      const float n0 = fmaxf(m0[row], mla_warp_max(s0));
      const float n1 = fmaxf(m1[row], mla_warp_max(s1));
      // A union tile may contain no entries for this row. Avoid inf-inf
      // while retaining zero mass until its first selected slot appears.
      const float safe0 = n0 == -INFINITY ? 0.f : n0;
      const float safe1 = n1 == -INFINITY ? 0.f : n1;
      const float cr0 = __expf(m0[row] - safe0), cr1 = __expf(m1[row] - safe1);
      const float p0 = ok ? __expf(s0 - safe0) : 0.f;
      const float p1 = ok ? __expf(s1 - safe1) : 0.f;
      l0[row] = fmaf(l0[row], cr0, mla_warp_sum(p0));
      l1[row] = fmaf(l1[row], cr1, mla_warp_sum(p1));
      m0[row] = n0; m1[row] = n1;
      __nv_bfloat16* spr = sp + (row_base + row) * (MLA_SMEM_P / sizeof(__nv_bfloat16));
      if (lane < MLA_TILE) {
        spr[h0 * MLA_PP + lane] = __float2bfloat16(p0 * a.ckv_scale);
        spr[(h0 + 1) * MLA_PP + lane] = __float2bfloat16(p1 * a.ckv_scale);
      }
      if (lane == 0) { scorr[(row_base + row) * MLA_H + h0] = cr0; scorr[(row_base + row) * MLA_H + h0 + 1] = cr1; }
    }
    mla_group_sync<SUBGROUP>();
#pragma unroll
    for (int row = 0; row < ROWS; ++row) {
      const float crg = scorr[(row_base + row) * MLA_H + g], crg8 = scorr[(row_base + row) * MLA_H + g + 8];
      const __nv_bfloat16* pa = sp + (row_base + row) * (MLA_SMEM_P / sizeof(__nv_bfloat16)) + g * MLA_PP;
      const uint32_t a0 = *(const uint32_t*)(pa + q4 * 2);
      const uint32_t a1 = *(const uint32_t*)(pa + 8 * MLA_PP + q4 * 2);
      const uint32_t a2 = *(const uint32_t*)(pa + q4 * 2 + 8);
      const uint32_t a3 = *(const uint32_t*)(pa + 8 * MLA_PP + q4 * 2 + 8);
      const uint8_t* cb = tile8 + (size_t)(q4 * 2) * MLA_RP + warp * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        acc[row][nt][0] *= crg; acc[row][nt][1] *= crg;
        acc[row][nt][2] *= crg8; acc[row][nt][3] *= crg8;
        const int n = nt * 8 + g;
        mla_mma_bf16(acc[row][nt][0], acc[row][nt][1], acc[row][nt][2], acc[row][nt][3],
          a0, a1, a2, a3, mla_e4m3x2_strided(cb + n, MLA_RP),
          mla_e4m3x2_strided(cb + 8 * MLA_RP + n, MLA_RP));
      }
    }
    mla_group_sync<SUBGROUP>();
  }
  mk_cp_wait<0>();
  mla_group_sync<SUBGROUP>();
  float* sl = scorr + shared_rows * MLA_H;
#pragma unroll
  for (int row = 0; row < ROWS; ++row) {
    if (lane == 0) { sl[(row_base + row) * MLA_H + warp * 2] = l0[row]; sl[(row_base + row) * MLA_H + warp * 2 + 1] = l1[row]; }
  }
  mla_group_sync<SUBGROUP>();
#pragma unroll
  for (int row = 0; row < ROWS; ++row) {
    const float ig = sl[(row_base + row) * MLA_H + g] > 0.f ? __frcp_rn(sl[(row_base + row) * MLA_H + g]) : 0.f;
    const float ig8 = sl[(row_base + row) * MLA_H + g + 8] > 0.f ? __frcp_rn(sl[(row_base + row) * MLA_H + g + 8]) : 0.f;
    __nv_bfloat16* o0 = a.out + ((size_t)(t + row_base + row) * MLA_H + g) * MLA_D;
    __nv_bfloat16* o8 = a.out + ((size_t)(t + row_base + row) * MLA_H + g + 8) * MLA_D;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int col = warp * 64 + nt * 8 + q4 * 2;
      *(__nv_bfloat162*)(o0 + col) = __floats2bfloat162_rn(acc[row][nt][0] * ig, acc[row][nt][1] * ig);
      *(__nv_bfloat162*)(o8 + col) = __floats2bfloat162_rn(acc[row][nt][2] * ig8, acc[row][nt][3] * ig8);
    }
  }
  mla_group_sync<SUBGROUP>();
}

__global__ __launch_bounds__(MK_THREADS) void mk_mla_pair_kernel(const MKMlaPairArgs p) {
  extern __shared__ __align__(16) unsigned char smem[];
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  for (int group = blockIdx.x; group < p.groups; group += p.a.grid) {
    const int t = group * 2, n = p.pair_lens[group];
    if (n >= 0) {
      const size_t base = (size_t)group * (2 * p.a.W);
      mla_pair_attention<2>(p, t, n, p.pair_slots + base, p.membership + base, smem);
    } else {
      mla_pair_attention<1>(p, t, p.a.lens[t], p.a.slots + (size_t)t * p.a.W, nullptr, smem);
      if (t + 1 < p.a.T)
        mla_pair_attention<1>(p, t + 1, p.a.lens[t + 1], p.a.slots + (size_t)(t + 1) * p.a.W, nullptr, smem);
    }
  }
}

__global__ __launch_bounds__(MLA_GROUP4_THREADS) void mk_mla_group4_kernel(const MKMlaPairArgs p) {
  extern __shared__ __align__(16) unsigned char smem4[];
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int group = blockIdx.x, t = group * 4;
  const int n = p.pair_lens[group];
  if (n >= 0) {
    const size_t base = (size_t)group * (4 * p.a.W);
    // Each 256-thread half owns two queries and 64 FP32 accumulator
    // registers per thread. All 512 threads share the one KV ring.
    mla_pair_attention<2, 4>(p, t, n, p.pair_slots + base, p.membership + base, smem4);
  } else {
    const int subgroup = threadIdx.x / MK_THREADS;
    const int first = t + subgroup * 2;
    unsigned char* local_smem = smem4 + subgroup * MLA_PAIR_SMEM;
    if (first < p.a.T)
      mla_pair_attention<1, 1, true>(p, first, p.a.lens[first],
          p.a.slots + (size_t)first * p.a.W, nullptr, local_smem);
    if (first + 1 < p.a.T)
      mla_pair_attention<1, 1, true>(p, first + 1, p.a.lens[first + 1],
          p.a.slots + (size_t)(first + 1) * p.a.W, nullptr, local_smem);
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// host entry points
// ---------------------------------------------------------------------------
namespace {

bool g_attrs_set = false;
// resident blocks per SM the device reports for mk_gemm2_kernel (2 by
// construction of GEMM2_SMEM; the v2 unit rule sizes its grid from it)
int g_gemm2_bps = 0;
int g_gemm2_m8_bps = 0;
int g_mk_sms = 0;  // multiprocessors, from the device (48 on GB10)

void set_kernel_attrs() {
  if (g_attrs_set) return;
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm2_kernel<1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM2_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm2_kernel<2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM2_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm2_kernel<4, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM2_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm2_kernel<1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM2_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm2_kernel<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM2_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(
      mk_gemm2_kernel<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM2_SMEM));
  // not a residency contract (v2 has no barrier): the unit rule wants
  // to know how many blocks share an SM
  // the widest instantiation (two m-tiles, four quant rows) bounds the
  // others' occupancy
  MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &g_gemm2_bps, mk_gemm2_kernel<4, false>, MK_THREADS, GEMM2_SMEM));
  if (g_gemm2_bps < 1) g_gemm2_bps = 1;
  if constexpr (MK_COMPACT_M8) {
    MK_CHECK_CUDA(cudaFuncSetAttribute(
        mk_gemm2_kernel<1, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM2_M8_SMEM));
    MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &g_gemm2_m8_bps, mk_gemm2_kernel<1, false, true>, MK_THREADS, GEMM2_M8_SMEM));
    if (g_gemm2_m8_bps < 1) g_gemm2_m8_bps = 1;
  } else {
    g_gemm2_m8_bps = g_gemm2_bps;
  }
  {  // the CURRENT device, the one the occupancy above and the launches
     // use -- not ordinal 0 (mk_probe_device asks the same way)
    int dev = 0;
    MK_CHECK_CUDA(cudaGetDevice(&dev));
    MK_CHECK_CUDA(cudaDeviceGetAttribute(
        &g_mk_sms, cudaDevAttrMultiProcessorCount, dev));
  }
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

int g_probe_ksr2 = -1;  // 0 = the rule below; > 0 forces the slice count
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
// Only the two C=1 shapes that won both warm and cold same-source A/B.
// More residency alone lost on the small shared-expert GEMMs and on the
// 51-tile in-projection. They retain the two-block transposed kernel.
bool mk_use_compact_m8(int m, int n, int k, bool lr = false) {
  return MK_COMPACT_M8 && !lr && m == 6 &&
         ((n == 4096 && k == 2048) || (n == 6144 && k == 4096));
}

int mk_choose_ksr2(int m, int n, int k, bool lr = false) {
  const int nblk = n / SMEM_W_ROWS, kblk = k / KSTEP;
  if (g_probe_ksr2 < 0) {
    const char* e = getenv("VLLM_GLM53_MK_KSR2");
    g_probe_ksr2 = e ? atoi(e) : 0;
  }
  int ksr;
  if (g_probe_ksr2 > 0) {
    ksr = g_probe_ksr2;
  } else {
    const int bps = mk_use_compact_m8(m, n, k, lr) ? g_gemm2_m8_bps : g_gemm2_bps;
    const int slots = (bps > 0 ? bps : 2) *
                      (g_mk_sms > 0 ? g_mk_sms : 48);
    const int kmax = kblk / 4 > 1 ? kblk / 4 : 1;
    if (slots % nblk == 0 && slots / nblk <= kmax) {
      ksr = slots / nblk;               // one exact wave
    } else if (nblk >= 2 * slots) {
      // two or more full waves of tiles on its own (the vocab head: 303
      // tiles, 3.2 waves): the last wave's stragglers are a small fraction
      // of a launch already at the DRAM floor, and every split adds m x n
      // fp32 partials per slice (2.5 MB at n = 38,784) plus their fold.
      // Head sweep (30차 §13): ksr 1 vs 2 on the real shape decides.
      ksr = 1;
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

// One v2 launch: the instantiation follows m (rows quantized per warp).
void mk_launch_gemm2(const MKGemm2Ctx& c2, cudaStream_t stream) {
  const int grid2 = (c2.n / SMEM_W_ROWS) * c2.ksr
                    + (c2.lr_r > 0 ? LR_CTAS : 0);
  if (c2.lr_r > 0) {
    if (c2.m <= 8)
      mk_launch(mk_gemm2_kernel<1, true>, grid2, GEMM2_SMEM, stream, c2);
    else if (c2.m <= 16)
      mk_launch(mk_gemm2_kernel<2, true>, grid2, GEMM2_SMEM, stream, c2);
    else
      mk_launch(mk_gemm2_kernel<4, true>, grid2, GEMM2_SMEM, stream, c2);
    return;
  }
  if (mk_use_compact_m8(c2.m, c2.n, c2.k))
    mk_launch(mk_gemm2_kernel<1, false, true>, grid2, GEMM2_M8_SMEM, stream, c2);
  else if (c2.m <= 8)
    mk_launch(mk_gemm2_kernel<1, false>, grid2, GEMM2_SMEM, stream, c2);
  else if (c2.m <= 16)
    mk_launch(mk_gemm2_kernel<2, false>, grid2, GEMM2_SMEM, stream, c2);
  else
    mk_launch(mk_gemm2_kernel<4, false>, grid2, GEMM2_SMEM, stream, c2);
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
std::vector<int64_t> mk_read_mhc_ts() {
#ifdef MK_PHASE_TS
  return mk_read_and_clear(g_mk_mhc_ts);
#else
  return {};
#endif
}

void mk_run_gemm(torch::Tensor x, torch::Tensor wq4, torch::Tensor ws4,
                 torch::Tensor out, int64_t n_orig, double wgs, int64_t bg,
                 int64_t rgs_ptr, int64_t lr_a_ptr, int64_t lr_b_ptr,
                 int64_t lr_r) {
  set_kernel_attrs();
  MKGemm2Ctx c2{};
  c2.x = (const __nv_bfloat16*)x.data_ptr();
  c2.wq4 = (const uint8_t*)wq4.data_ptr();
  c2.ws4 = (const int8_t*)ws4.data_ptr();
  c2.wgs = (float)wgs;
  c2.rgs = (const float*)rgs_ptr;   // 33차 lever 3 (0 = per-tensor wgs only)
  TORCH_CHECK(lr_r >= 0 && lr_r <= LR_MAX && lr_r % 8 == 0,
              "low-rank correction rank out of contract (0..32, x8)");
  TORCH_CHECK(lr_r == 0 || (lr_a_ptr && lr_b_ptr),
              "low-rank correction needs both factors");
  c2.out = (__nv_bfloat16*)out.data_ptr();
  c2.m = (int)x.size(0);
  c2.k = (int)x.size(1);        // k is the ACTIVATION width
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 7) == 0 && x.is_contiguous(),
              "x must be 8 B aligned and contiguous");
  // Tile-major packs -- see stage_raw. The shape is the only thing
  // standing between a stale row-major pack and silently wrong output.
  TORCH_CHECK(wq4.dim() == 4 && wq4.size(2) == SMEM_W_ROWS
                  && wq4.size(3) == 64 && wq4.is_contiguous(),
              "wq4 must be a contiguous [n/128, k/128, 128, 64] pack");
  TORCH_CHECK(ws4.dim() == 4 && ws4.size(0) == wq4.size(0)
                  && ws4.size(1) == wq4.size(1) && ws4.size(2) == SMEM_W_ROWS
                  && ws4.size(3) == 8 && ws4.is_contiguous(),
              "ws4 must be a contiguous [n/128, k/128, 128, 8] pack");
  c2.n = (int)wq4.size(0) * SMEM_W_ROWS;
  c2.n_orig = (int)n_orig;
  TORCH_CHECK(c2.k % KSTEP == 0 && c2.k <= KBLK_MAX * KSTEP, "k out of contract");
  TORCH_CHECK((int)wq4.size(1) == c2.k / KSTEP, "wq4 k-tiles disagree with x");
  TORCH_CHECK(c2.m <= 32, "m out of contract");
  auto stream = c10::cuda::getCurrentCUDAStream();
  c2.lr_a = (const __nv_bfloat16*)lr_a_ptr;
  c2.lr_b = (const __nv_bfloat16*)lr_b_ptr;
  c2.lr_r = (int)lr_r;
  c2.lr_slot = bg != 0 ? 1 : 0;
  const int nblk = c2.n / SMEM_W_ROWS;
  c2.ksr = mk_choose_ksr2(c2.m, c2.n, c2.k, c2.lr_r > 0);
  // one slice per tile stores bf16 straight from the accumulators (no
  // partial is read or written), so the partial bound is a split's
  // contract only: m = 32 on the head (32 x 38,784 floats) is served whole
  TORCH_CHECK(nblk <= MK2_TILES_MAX && c2.ksr >= 1
                  && c2.ksr <= c2.k / KSTEP
                  && (c2.ksr == 1
                      || (size_t)c2.m * c2.n * c2.ksr <= (size_t)MK2_PART_ELEMS),
              "gemm2 plan out of contract");
  mk_launch_gemm2(c2, stream);
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

void mk_run_mla_prefill_pair(std::vector<int64_t> ptrs, std::vector<double> scalars,
                            std::vector<int64_t> ints) {
  TORCH_CHECK(ptrs.size() == 8 && scalars.size() == 2 && ints.size() == 2,
              "run_mla_prefill_pair arg contract");
  TORCH_CHECK(ints[0] >= 128 && ints[0] <= 8192 && ints[1] > 0 && ints[1] <= MLA_PAIR_MAX_W,
              "mla pair requires bounded prefill T and W");
  TORCH_CHECK((ptrs[0] & 15) == 0 && (ptrs[1] & 15) == 0 && (ptrs[4] & 3) == 0,
              "mla pair requires aligned Q, FP8 cache and BF16 output");
  static bool attrs = false;
  static int grid = 0;
  if (!attrs) {
    MK_CHECK_CUDA(cudaFuncSetAttribute(
        mk_mla_pair_prepare, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_PAIR_PREP_SMEM));
    MK_CHECK_CUDA(cudaFuncSetAttribute(
        mk_mla_pair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_PAIR_SMEM));
    attrs = true;
  }
  MKMlaPairArgs p{};
  p.a.q = (const __nv_bfloat16*)ptrs[0];
  p.a.ckv = (const uint8_t*)ptrs[1];
  p.a.slots = (const int*)ptrs[2];
  p.a.lens = (const int*)ptrs[3];
  p.a.out = (__nv_bfloat16*)ptrs[4];
  p.pair_slots = (int*)ptrs[5];
  p.membership = (int*)ptrs[6];
  p.pair_lens = (int*)ptrs[7];
  p.a.sm_scale = (float)scalars[0];
  p.a.ckv_scale = (float)scalars[1];
  p.a.T = (int)ints[0]; p.a.W = (int)ints[1]; p.a.splits = 1;
  p.groups = (p.a.T + 1) / 2;
  p.a.grid = mk_resident_grid(mk_mla_pair_kernel, grid, MLA_PAIR_SMEM, MLA_GRID_CAP);
  auto stream = c10::cuda::getCurrentCUDAStream();
  mk_launch(mk_mla_pair_prepare, p.groups, MLA_PAIR_PREP_SMEM, stream, p);
  mk_launch(mk_mla_pair_kernel, p.a.grid, MLA_PAIR_SMEM, stream, p);
}

void mk_run_mla_prefill_group4(std::vector<int64_t> ptrs, std::vector<double> scalars,
                              std::vector<int64_t> ints) {
  TORCH_CHECK(ptrs.size() == 8 && scalars.size() == 2 && ints.size() == 2,
              "run_mla_prefill_group4 arg contract");
  TORCH_CHECK(ints[0] >= 128 && ints[0] <= 8192 && ints[1] > 0 && ints[1] <= MLA_PAIR_MAX_W,
              "mla group4 requires bounded prefill T and W");
  TORCH_CHECK((ptrs[0] & 15) == 0 && (ptrs[1] & 15) == 0 && (ptrs[4] & 3) == 0,
              "mla group4 requires aligned Q, FP8 cache and BF16 output");
  static bool attrs = false;
  if (!attrs) {
    MK_CHECK_CUDA(cudaFuncSetAttribute(mk_mla_group4_prepare,
        cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_GROUP4_PREP_SMEM));
    MK_CHECK_CUDA(cudaFuncSetAttribute(mk_mla_group4_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_GROUP4_SMEM));
    attrs = true;
  }
  MKMlaPairArgs p{};
  p.a.q = (const __nv_bfloat16*)ptrs[0];
  p.a.ckv = (const uint8_t*)ptrs[1];
  p.a.slots = (const int*)ptrs[2];
  p.a.lens = (const int*)ptrs[3];
  p.a.out = (__nv_bfloat16*)ptrs[4];
  p.pair_slots = (int*)ptrs[5];
  p.membership = (int*)ptrs[6];
  p.pair_lens = (int*)ptrs[7];
  p.a.sm_scale = (float)scalars[0]; p.a.ckv_scale = (float)scalars[1];
  p.a.T = (int)ints[0]; p.a.W = (int)ints[1]; p.a.splits = 1;
  p.groups = (p.a.T + 3) / 4;
  // No inter-CTA barrier: one ordinary block per group lets CUDA schedule
  // the 512-thread register/shared-memory footprint without assuming that
  // its residency equals the older 256-thread pair kernel.
  p.a.grid = p.groups;
  auto stream = c10::cuda::getCurrentCUDAStream();
  mk_launch(mk_mla_group4_prepare, p.groups, MLA_GROUP4_PREP_SMEM, stream, p);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.groups);
  cfg.blockDim = dim3(MLA_GROUP4_THREADS);
  cfg.dynamicSmemBytes = MLA_GROUP4_SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = mk_pdl_enabled() ? 1 : 0;
  MK_CHECK_CUDA(cudaLaunchKernelEx(&cfg, mk_mla_group4_kernel, p));
}

// Blocks the caller can fill: the split count this launch would use.
int64_t mk_mla_grid() {
  set_kernel_attrs();
  return mk_resident_grid(mk_mla_kernel, g_mla_grid, MLA_SMEM, MLA_GRID_CAP);
}

// MK_SEG_SMLP2: the shared-expert / dense MLP as two PDL-chained v2
// launches -- gate_up with the pair-activation epilogue into the caller's
// bf16 scratch, down on the a_ready path -- no grid barrier anywhere.
// ptrs: x, gu_wq4, gu_ws4, d_wq4, d_ws4, gu_scratch, out, gu_rgs, d_rgs (0 = none)
// scalars: gu_wgs, d_wgs, limit, alpha, beta
// ints: T, k_gu, n_gu, n_int, n_out, gu_tiles_n, gu_tiles_k, d_tiles_n, d_tiles_k
void mk_run_smlp2(std::vector<int64_t> ptrs, std::vector<double> scalars,
                  std::vector<int64_t> ints) {
  set_kernel_attrs();
  TORCH_CHECK(ptrs.size() == 9 && scalars.size() == 5 && ints.size() == 9,
              "run_smlp2 arg contract");
  const int T = (int)ints[0], k_gu = (int)ints[1], n_gu = (int)ints[2];
  const int n_int = (int)ints[3], n_out = (int)ints[4];
  TORCH_CHECK(T >= 1 && T <= 32, "smlp2: T out of contract");
  TORCH_CHECK(k_gu % KSTEP == 0 && k_gu <= KBLK_MAX * KSTEP, "smlp2: k_gu out of contract");
  TORCH_CHECK(n_int % KSTEP == 0 && n_int <= KBLK_MAX * KSTEP, "smlp2: n_int out of contract");
  TORCH_CHECK(n_gu == 2 * n_int, "smlp2: gate_up width is not 2 x n_int");
  TORCH_CHECK(((uintptr_t)ptrs[0] & 7) == 0, "smlp2: x must be 8 B aligned");
  const int n_gu_pad = ((n_gu + SMEM_W_ROWS - 1) / SMEM_W_ROWS) * SMEM_W_ROWS;
  const int n_out_pad = ((n_out + SMEM_W_ROWS - 1) / SMEM_W_ROWS) * SMEM_W_ROWS;
  TORCH_CHECK((int)ints[5] * SMEM_W_ROWS == n_gu_pad && (int)ints[6] == k_gu / KSTEP,
              "smlp2: gate_up pack tiles disagree with (n_gu, k_gu)");
  TORCH_CHECK((int)ints[7] * SMEM_W_ROWS == n_out_pad && (int)ints[8] == n_int / KSTEP,
              "smlp2: down pack tiles disagree with (n_out, n_int)");
  TORCH_CHECK(n_gu_pad / SMEM_W_ROWS <= MK2_TILES_MAX && n_out_pad / SMEM_W_ROWS <= MK2_TILES_MAX,
              "smlp2: too many tiles");
  auto stream = c10::cuda::getCurrentCUDAStream();
  MKGemm2Ctx g{};  // gate_up -> scratch, the pair epilogue emits the A groups
  g.x = (const __nv_bfloat16*)ptrs[0];
  g.wq4 = (const uint8_t*)ptrs[1];
  g.ws4 = (const int8_t*)ptrs[2];
  g.wgs = (float)scalars[0];
  g.rgs = (const float*)ptrs[7];
  g.out = (__nv_bfloat16*)ptrs[5];
  g.m = T; g.n = n_gu_pad; g.k = k_gu; g.n_orig = n_gu;
  g.ksr = mk_choose_ksr2(g.m, g.n, g.k);
  g.pair_act = 1; g.n_int = n_int;
  g.act_limit = (float)scalars[2]; g.act_alpha = (float)scalars[3]; g.act_beta = (float)scalars[4];
  TORCH_CHECK((size_t)g.m * g.n * g.ksr <= (size_t)MK2_PART_ELEMS, "smlp2: gate_up plan out of contract");
  mk_launch_gemm2(g, stream);
  MKGemm2Ctx d{};  // down on the published groups
  d.x = g.x;  // unused on the a_ready path
  d.wq4 = (const uint8_t*)ptrs[3];
  d.ws4 = (const int8_t*)ptrs[4];
  d.wgs = (float)scalars[1];
  d.rgs = (const float*)ptrs[8];
  d.out = (__nv_bfloat16*)ptrs[6];
  d.m = T; d.n = n_out_pad; d.k = n_int; d.n_orig = n_out;
  d.ksr = mk_choose_ksr2(d.m, d.n, d.k);
  d.a_ready = 1;
  TORCH_CHECK((size_t)d.m * d.n * d.ksr <= (size_t)MK2_PART_ELEMS, "smlp2: down plan out of contract");
  mk_launch_gemm2(d, stream);
}

// Bench: the plan one launch of (m, n, k) would use -- {ksr, units, blocks
// per SM} -- and the setter behind it (-1 leaves the knob as it is).
std::vector<int64_t> mk_gemm2_plan(int64_t m, int64_t n, int64_t k) {
  set_kernel_attrs();
  const int n_pad = (int)(((n + SMEM_W_ROWS - 1) / SMEM_W_ROWS) * SMEM_W_ROWS);
  const int ksr = mk_choose_ksr2((int)m, n_pad, (int)k);
  return {(int64_t)ksr, (int64_t)((n_pad / SMEM_W_ROWS) * ksr),
          (int64_t)(mk_use_compact_m8((int)m, n_pad, (int)k) ? g_gemm2_m8_bps : g_gemm2_bps)};
}
void mk_set_gemm2(int64_t ksr) {
  if (ksr >= 0) g_probe_ksr2 = (int)ksr;
}
// Boot checks temporarily force splits. Preserve the raw value, including
// the -1 sentinel that defers reading the environment knob, so a check
// cannot change the plan the caller configured.
std::vector<int64_t> mk_probe_state() {
  return {g_probe_ksr2};
}
void mk_restore_probe_state(const std::vector<int64_t>& state) {
  TORCH_CHECK(state.size() == 1, "restore_probe_state: expected one knob");
  const int64_t imax = std::numeric_limits<int>::max();
  const int64_t imin = std::numeric_limits<int>::min();
  TORCH_CHECK(state[0] >= imin && state[0] <= imax,
              "restore_probe_state: the knob is outside its stored range");
  g_probe_ksr2 = (int)state[0];
}
// v2 unit timestamps of the last launch, [MK2_UNITS_MAX][4] ns, then cleared.
std::vector<int64_t> mk_read_ts2() {
#ifdef MK_PHASE_TS
  return mk_read_and_clear(g_mk2_ts);
#else
  return {};
#endif
}

// ---------------------------------------------------------------------------
// MK_PREP (37차, 2026-09-06): the fused decode-step preparation kernel in
// CUDA -- the same contract as glm53_prep_fused's Triton kernel, one block
// per program (num_reqs request blocks, one fill block, one pad block per
// KV group), with the request's Q tokens as plain loops: no power-of-two
// lane constraint, no masks to forget (the k=5 boot's self-check caught a
// padded-lane store crossing into the next request's rows).
// Pointer element types are the runner's (asserted on the Python side):
// int32 input_ids/qsl/seq_lens/num_computed/prefill_len/num_accepted/
// block tables/gdn state/req_id/exp_bt/dec_*/idx_bt; int64 positions/
// idx_mapping/last_sampled/draft_tokens/slot mappings/expanded_idx/
// comp_slot/strides; uint64 pointer tables; int8 is_padding/gdn masks.
// ---------------------------------------------------------------------------
struct MKPrepArgs {
  int num_reqs, num_tokens, max_num_reqs, max_num_tokens;
  const int64_t* idx_mapping; const int32_t* num_computed; const int32_t* prefill_len;
  const int64_t* last_sampled; const int64_t* draft_tokens; int64_t draft_stride;
  const int32_t* num_accepted;
  int32_t* input_ids; int64_t* positions; int32_t* qsl; int32_t* seq_lens; int8_t* is_padding;
  int64_t* expanded_idx; int32_t* expanded_pos;
  const uint64_t* src_bt_ptrs; const uint64_t* dst_bt_ptrs; const int64_t* bt_strides;
  const int32_t* block_sizes; const int32_t* num_blocks; int64_t num_blocks_stride;
  int64_t* slot; int64_t slot_stride;
  const int32_t* gdn_group_idx; const uint64_t* gdn_state_ptrs; const int64_t* gdn_state_strides;
  const uint64_t* gdn_mask_ptrs; const uint64_t* gdn_tok_ptrs; const uint64_t* gdn_qsl_ptrs;
  const uint64_t* gdn_nacc_ptrs;
  int32_t* req_id; int64_t req_id_cap; int32_t* exp_bt; int64_t exp_bt_stride;
  int32_t* dec_seq_lens; int64_t dec_seq_cap; int32_t* dec_lens; int32_t* per_req_dec_lens;
  int32_t* idx_bt; int64_t idx_bt_stride; int64_t idx_bt_cols;
  int64_t* comp_slot; int64_t comp_slot_cap;
  int Q, NUM_SPEC, NS, G, N_GDN, ATTN_G, FACTOR, RATIO, SBS; int64_t PAD_ID;
};

#define MK_PREP_THREADS 256

template <typename T>
__device__ __forceinline__ void mk_prep_fill(T* p, int64_t start, int64_t end, T v) {
  for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) p[i] = v;
}

__global__ void __launch_bounds__(MK_PREP_THREADS) mk_prep_kernel(MKPrepArgs a) {
  const int pid = blockIdx.x;
  if (pid >= a.num_reqs) {
    const int role = pid - a.num_reqs;
    if (role == 0) {
      mk_prep_fill<int32_t>(a.qsl, a.num_reqs, a.max_num_reqs + 1, a.num_tokens);
      mk_prep_fill<int32_t>(a.seq_lens, a.num_reqs, a.max_num_reqs, 0);
      mk_prep_fill<int8_t>(a.is_padding, 0, a.num_tokens, 0);
      if (threadIdx.x < a.N_GDN) {
        int32_t* qp = reinterpret_cast<int32_t*>(a.gdn_qsl_ptrs[threadIdx.x]);
        qp[a.num_reqs] = a.num_tokens;
      }
      mk_prep_fill<int32_t>(a.req_id, a.num_tokens, a.req_id_cap, 0);
      mk_prep_fill<int32_t>(a.dec_seq_lens, a.num_tokens, a.dec_seq_cap, 0);
      mk_prep_fill<int64_t>(a.comp_slot, a.num_tokens, a.comp_slot_cap, a.PAD_ID);
    } else {
      const int g = role - 1;
      mk_prep_fill<int64_t>(a.slot + (int64_t)g * a.slot_stride, a.num_tokens, a.max_num_tokens,
                            a.PAD_ID);
    }
    return;
  }
  const int r = pid;
  const int64_t rs = a.idx_mapping[r];
  const int32_t ncomp = a.num_computed[rs];
  const int Q = a.Q;
  const int64_t qs = (int64_t)r * Q;
  const int32_t seq_len = ncomp + Q;
  const int t = threadIdx.x;
  if (t == 0) {
    a.seq_lens[r] = seq_len;
    a.qsl[r] = (int32_t)qs;
    a.per_req_dec_lens[r] = Q;
  }
  // per-token scalars (Q <= 32 threads)
  if (t < Q) {
    const int64_t pos = (int64_t)ncomp + t;
    a.positions[qs + t] = pos;
    a.expanded_idx[qs + t] = rs;
    a.expanded_pos[qs + t] = t;
    a.req_id[qs + t] = r;
    a.dec_lens[qs + t] = 1;
    a.dec_seq_lens[qs + t] = (seq_len - Q + t + 1) / a.RATIO;
    // combine_sampled_and_draft_tokens, NUM_NEW_SAMPLED_TOKENS=1
    const int32_t prefill_len = a.prefill_len[rs];
    if (seq_len > prefill_len) {
      if (t == 0 && seq_len - Q >= prefill_len)
        a.input_ids[qs] = (int32_t)a.last_sampled[rs];
      if (t < a.NUM_SPEC)
        a.input_ids[qs + 1 + t] = (int32_t)a.draft_tokens[rs * a.draft_stride + t];
    }
  }
  // gather_block_tables + compute_slot_mappings, every group
  for (int g = 0; g < a.G; ++g) {
    const int32_t* src = reinterpret_cast<const int32_t*>(a.src_bt_ptrs[g]);
    int32_t* dst = reinterpret_cast<int32_t*>(a.dst_bt_ptrs[g]);
    const int64_t stride = a.bt_strides[g];
    const int32_t bs = a.block_sizes[g];
    const int32_t nb = a.num_blocks[(int64_t)g * a.num_blocks_stride + rs];
    const int32_t* src_row = src + rs * stride;
    int32_t* dst_row = dst + (int64_t)r * stride;
    for (int i = t; i < nb; i += blockDim.x) dst_row[i] = src_row[i];
    if (t < Q) {
      const int64_t pos = (int64_t)ncomp + t;
      const int64_t bsz = bs;
      const int32_t bn = src_row[pos / bsz];
      a.slot[(int64_t)g * a.slot_stride + qs + t] = (int64_t)bn * bs + pos % bsz;
    }
  }
  __syncthreads();  // the gathered rows are read back below
  // GDN builders (FULL-graph branch): spec rows only, no padding
  const int32_t nacc = a.num_accepted[rs];
  for (int k = 0; k < a.N_GDN; ++k) {
    const int32_t gm = a.gdn_group_idx[k];
    const int32_t* dst = reinterpret_cast<const int32_t*>(a.dst_bt_ptrs[gm]);
    const int64_t stride = a.bt_strides[gm];
    int32_t* sp = reinterpret_cast<int32_t*>(a.gdn_state_ptrs[k]);
    const int64_t ss = a.gdn_state_strides[k];
    if (t < a.NS) sp[(int64_t)r * ss + t] = dst[(int64_t)r * stride + t];
    if (t == 0) {
      reinterpret_cast<int8_t*>(a.gdn_mask_ptrs[k])[r] = 1;
      reinterpret_cast<int32_t*>(a.gdn_qsl_ptrs[k])[r] = (int32_t)qs;
      reinterpret_cast<int32_t*>(a.gdn_nacc_ptrs[k])[r] = nacc;
    }
    if (t < Q) reinterpret_cast<int32_t*>(a.gdn_tok_ptrs[k])[qs + t] = (int32_t)(qs + t);
  }
  // attention group: sparse MLA req ids + indexer decode buffers
  const int32_t* dsta = reinterpret_cast<const int32_t*>(a.dst_bt_ptrs[a.ATTN_G]);
  const int64_t wa = a.bt_strides[a.ATTN_G];
  const int32_t* row = dsta + (int64_t)r * wa;
  if (t < Q) {
    // get_compressed_slot_mapping over indexer_block_table = bt[:, ::F] // F
    const int32_t pos32 = seq_len - Q + t;
    const bool valid = ((pos32 + 1) % a.RATIO) == 0;
    int64_t cslot = a.PAD_ID;
    if (valid) {
      const int32_t pc = pos32 / a.RATIO;
      const int32_t bid = pc / a.SBS;
      const int32_t bn = row[(int64_t)bid * a.FACTOR] / a.FACTOR;
      cslot = (int64_t)bn * a.SBS + pc % a.SBS;
    }
    a.comp_slot[qs + t] = cslot;
  }
  // expanded_block_table_buffer[t] = the gathered row, full width, Q times;
  // indexer_decode_block_table_buffer[t, c] = row[c*F] // F
  for (int j = 0; j < Q; ++j) {
    int32_t* erow = a.exp_bt + (qs + j) * a.exp_bt_stride;
    int32_t* irow = a.idx_bt + (qs + j) * a.idx_bt_stride;
    for (int64_t c = t; c < wa; c += blockDim.x) erow[c] = row[c];
    for (int64_t c = t; c < a.idx_bt_cols; c += blockDim.x) irow[c] = row[c * a.FACTOR] / a.FACTOR;
  }
}

// ptrs: 33 device pointers in the Python order (see PrepPlan._cuda_args);
// ints: num_reqs, num_tokens, max_num_reqs, max_num_tokens, draft_stride,
// num_blocks_stride, slot_stride, req_id_cap, exp_bt_stride, dec_seq_cap,
// idx_bt_stride, idx_bt_cols, comp_slot_cap, Q, NUM_SPEC, NS, G, N_GDN,
// ATTN_G, FACTOR, RATIO, SBS, PAD_ID
void mk_run_prep(std::vector<int64_t> ptrs, std::vector<int64_t> ints) {
  TORCH_CHECK(ptrs.size() == 33 && ints.size() == 23, "mk_run_prep: 33 ptrs + 23 ints");
  MKPrepArgs a;
  int p = 0;
  auto P = [&](auto** dst) { *dst = reinterpret_cast<std::remove_reference_t<decltype(*dst)>>(ptrs[p++]); };
  P(&a.idx_mapping); P(&a.num_computed); P(&a.prefill_len); P(&a.last_sampled); P(&a.draft_tokens);
  P(&a.num_accepted); P(&a.input_ids); P(&a.positions); P(&a.qsl); P(&a.seq_lens); P(&a.is_padding);
  P(&a.expanded_idx); P(&a.expanded_pos); P(&a.src_bt_ptrs); P(&a.dst_bt_ptrs); P(&a.bt_strides);
  P(&a.block_sizes); P(&a.num_blocks); P(&a.slot); P(&a.gdn_group_idx); P(&a.gdn_state_ptrs);
  P(&a.gdn_state_strides); P(&a.gdn_mask_ptrs); P(&a.gdn_tok_ptrs); P(&a.gdn_qsl_ptrs);
  P(&a.gdn_nacc_ptrs); P(&a.req_id); P(&a.exp_bt); P(&a.dec_seq_lens); P(&a.dec_lens);
  P(&a.per_req_dec_lens); P(&a.idx_bt); P(&a.comp_slot);
  int q = 0;
  a.num_reqs = (int)ints[q++]; a.num_tokens = (int)ints[q++]; a.max_num_reqs = (int)ints[q++];
  a.max_num_tokens = (int)ints[q++]; a.draft_stride = ints[q++]; a.num_blocks_stride = ints[q++];
  a.slot_stride = ints[q++]; a.req_id_cap = ints[q++]; a.exp_bt_stride = ints[q++];
  a.dec_seq_cap = ints[q++]; a.idx_bt_stride = ints[q++]; a.idx_bt_cols = ints[q++];
  a.comp_slot_cap = ints[q++]; a.Q = (int)ints[q++]; a.NUM_SPEC = (int)ints[q++]; a.NS = (int)ints[q++];
  a.G = (int)ints[q++]; a.N_GDN = (int)ints[q++]; a.ATTN_G = (int)ints[q++]; a.FACTOR = (int)ints[q++];
  a.RATIO = (int)ints[q++]; a.SBS = (int)ints[q++]; a.PAD_ID = ints[q++];
  TORCH_CHECK(a.Q > 0 && a.Q <= 32 && a.NS <= MK_PREP_THREADS && a.N_GDN <= MK_PREP_THREADS,
              "mk_run_prep: Q/NS/N_GDN out of range");
  auto stream = c10::cuda::getCurrentCUDAStream();
  mk_prep_kernel<<<a.num_reqs + 1 + a.G, MK_PREP_THREADS, 0, stream>>>(a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_device", &mk_probe_device, "device geometry probe");
  m.def("read_ts", &mk_read_ts, "phase timestamps (MK_PHASE_TS builds)");
  m.def("read_mhc_ts", &mk_read_mhc_ts, "mhc phase timestamps");
  m.def("run_gemm", &mk_run_gemm, "MK_SEG_GEMM (W4 pack)");
  m.def("gemm2_plan", &mk_gemm2_plan, "bench: {ksr, units, blocks/SM} of (m, n, k)");
  m.def("set_gemm2", &mk_set_gemm2, "bench: force the GEMM's ksr (-1 = keep)");
  m.def("probe_state", &mk_probe_state, "snapshot the raw GEMM probe knob");
  m.def("restore_probe_state", &mk_restore_probe_state, "restore the raw GEMM probe knob");
  m.def("read_ts2", &mk_read_ts2, "v2 unit timestamps (MK_PHASE_TS builds)");
  m.def("run_mhc", &mk_run_mhc, "MK_SEG_MHC");
  m.def("run_prep", &mk_run_prep, "MK_PREP: fused decode-step preparation (CUDA form of glm53_prep_fused)");
  m.def("run_mla", &mk_run_mla, "MK_SEG_MLA (sparse MLA decode)");
  m.def("run_mla_prefill_pair", &mk_run_mla_prefill_pair, "MK MLA exact-selection (not bit-exact output) prefill pair reuse");
  m.def("run_mla_prefill_group4", &mk_run_mla_prefill_group4, "MK MLA exact-selection (not bit-exact output) four-query reuse");
  m.def("run_smlp2", &mk_run_smlp2, "MK_SEG_SMLP2 (two PDL-chained v2 launches, no barrier)");
  m.def("mla_grid", &mk_mla_grid, "MK_SEG_MLA resident grid");
}
