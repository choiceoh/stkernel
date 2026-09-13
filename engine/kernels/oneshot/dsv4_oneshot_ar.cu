// dsv4_oneshot_ar — torch custom-op wrapping the Stage-2 one-shot cross-node
// AllReduce (probes/oneshot_ar2.cu) for in-engine use. bf16, variable size,
// torch tensor in/out, torch.distributed bootstrap.
//
// Exposes (pybind):
//   local_infos() -> bytes        # serialize my per-peer QP info (after QP create)
//   connect(all: List[bytes])     # match + RTR/RTS + start proxy
//   oneshot_ar(Tensor) -> Tensor  # graph-capturable AllReduce (out-of-place)
//   healthy() -> bool             # proxy watchdog status
//   shutdown()
#define AT_PER_OPERATOR_HEADERS
#include <ATen/core/Tensor.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "dsv4_oneshot_transport.h"

#ifndef OSAR_COMPACT_CTA
#define OSAR_COMPACT_CTA 0
#endif
#ifndef OSAR_PROXY_INLINE
#define OSAR_PROXY_INLINE 0
#endif

#define NPEER 3
#define RING 4
// 131072 bf16 elements cover 32 hidden-4096 tokens (DSpark C<=5). Keep the
// shipped value as the default, but allow a per-build override so the unknown
// one-shot-vs-NCCL crossover can be measured at higher concurrency. All peers
// must use the same value because it participates in the remote rx stride.
// Half the 256 KiB cap: decode messages are hidden(4096) x tokens x 2 B, so
// this puts C=1-shaped traffic (8 tokens = 64 KiB) on one side and a full
// 4-sequence spec batch (32 tokens = 256 KiB) on the other.
#define SPLIT_BYTES (128 * 1024)
#ifndef MAXEL
#define MAXEL 131072
#endif
#define ARGRID 48     // GB10 / SM121a has exactly 48 SMs
#define ARTHREADS 256
static_assert(ARGRID == OSAR_PUBLICATION_TICKETS && ARTHREADS == 256);
// 16B (8 x bf16) vector lanes for the copy/reduce phases. #99's phase timers
// put copy+reduce at ~11.8us of the ~100us collective with 2-byte scalar
// accesses. VECITER bounds the per-thread vector trip count: n <= MAXEL and
// the grid is fixed at ARGRID x ARTHREADS (the done_ctr protocol), so
// ceil((MAXEL/8)/(ARGRID*ARTHREADS)) == 2 covers every launch this build can
// issue. That bound is what lets the reduce phase reuse the values a thread
// copied (identical mapping in both loops) instead of re-reading src.
#define VECITER ((MAXEL / 8 + ARGRID * ARTHREADS - 1) / (ARGRID * ARTHREADS))
#define PROXY_CORE 18
// clock64() counts SM cycles. The SM clock is pinned at 1592 MHz on this
// fleet (nvidia-smi clocks.sm, flat across load), so one constant converts.
// The absolute us are only as good as that assumption; the RATIO between the
// stall watchdog uses that clock; normal serving does not collect phase counters.
#define SM_CLK_MHZ 1592

#define CHK(x)                                                            \
  do {                                                                    \
    if (!(x)) {                                                           \
      fprintf(stderr, "[oneshot] FAIL %s:%d %s errno=%d(%s)\n", __FILE__, \
              __LINE__, #x, errno, strerror(errno));                      \
      throw std::runtime_error("oneshot verbs failure");                  \
    }                                                                     \
  } while (0)

#include <ctime>

typedef __nv_bfloat16 bf16;

struct Ctrl {
  volatile uint64_t tx_seq;
  volatile uint64_t ack_seq;
  volatile uint64_t stop;
  volatile uint64_t proxy_beat;          // watchdog heartbeat
  volatile unsigned long long done_ctr;  // monotonic block-completion count
  uint64_t flag_src[NPEER];
  uint64_t nbytes[RING];                 // payload size per slot (GPU sets)
  volatile uint64_t rxf[RING][NPEER];    // inbound flags (slot-major)
  // Phase timers. These live INSIDE the old pad[8] so every offset after them
  // -- tx, rx -- is unchanged and the peers' rx_base/rxf_base stay valid.
  // Monotonic and never reset, for the same reason done_ctr is: a cudagraph
  // replay cannot see a reset. Written by block 0 thread 0 only, one store
  // per collective, so the cost is a handful of cycles on one thread.
  volatile uint64_t t_guard;             // SM cycles spinning for ring space
  volatile uint64_t t_copy;              // copy + fence + counter + publish
  volatile uint64_t t_wait;              // SM cycles spinning for peer flags
  volatile uint64_t t_reduce;            // the summation
  volatile uint64_t t_calls;             // samples behind the four above
  // The same wait, counted only for messages at or below SPLIT_BYTES. The
  // wait is "until all three peers' data landed", so it carries the RDMA
  // transfer as well as any arrival skew: at the 256 KiB cap a rank takes in
  // 3 x 256 KiB, which is ~43 us at the fabric's measured 17.8 GB/s and
  // ~22 us if both HCAs serve it -- against 38.7 us measured. Splitting the
  // same counter by size separates the two: transfer scales with the
  // message, skew does not. Large = total - small, so this costs two fields,
  // which is what the padding has left (the peers' rx_base/rxf_base offsets
  // must not move).
  volatile uint64_t t_wait_sm;
  volatile uint64_t t_calls_sm;
  // pad[0] doubles as the STALL WORD (32차 §9 hang forensics): block 0's
  // thread 0 writes it once when a spin passes OSAR_STALL_S seconds --
  // (seq << 8) | (phase << 6) | (missing-peer mask << 3) | slot -- and the
  // proxy thread, which is alive while the kernel spins, prints it with
  // ack_seq / tx_seq / rxf[slot]. After OSAR_STALL_TRAP_S the kernel traps so
  // the engine dies in seconds with the line above, not after a 5-minute
  // RPC timeout with nothing. Offsets after it are untouched.
  uint64_t pad[1];
  bf16 tx[RING][MAXEL];
  bf16 rx[RING][NPEER][MAXEL];
};

struct Info {
  uint32_t qpn, psn, rkey;
  uint32_t mtu;
  uint8_t gid[16];
  uint64_t rx_base, rxf_base;
};

static Ctrl *g_ctrl = nullptr;
static struct ibv_qp *g_qp[NPEER];
static struct ibv_cq *g_cq = nullptr;
static struct ibv_mr *g_mr = nullptr;
static struct ibv_pd *g_pd = nullptr;
static struct ibv_context *g_ctx = nullptr;
static Info g_local[NPEER], g_remote[NPEER];
static int g_rank = -1, g_world = 0, g_sgid = -1, g_peers[NPEER];
static pthread_t g_proxy;
static bool g_started = false;
static unsigned g_inline_cap[NPEER] = {};
static const char *DEVNAME = "rocep1s0f0";

// ---------------- kernels (device-side slot from tx_seq) ----------------
// One launch per collective. #89 folded the guard into k_copy_in and the wait
// into k_reduce (both spin on globals -- tx_seq/ack_seq or the peers' rxf
// flags -- so every block can poll independently and a not-yet-resident block
// can never deadlock us) but kept k_signal separate: signalling needs ALL
// copy blocks done, which "needs" an atomic counter, and a counter would have
// to be reset somewhere a cudagraph replay cannot see.
//
// A counter that is never reset kills that objection. done_ctr is monotonic
// and every launch adds exactly ARGRID to it (fixed grid, see py_oneshot), so
// stream ordering keeps it a multiple of ARGRID at every kernel entry: the
// block whose atomicAdd returns old % ARGRID == ARGRID-1 is the ARGRID-th block
// of THIS launch to finish copying -- meaning every block already read
// tx_seq, so publishing tx_seq = s0+1 cannot be misread as a later sequence.
// The counter lives past any replay; nothing to reset.
// Spin discipline. Both loops below read flags that a PEER writes -- ack_seq
// through the proxy, rxf through the peer NIC -- into RDMA-registered memory.
// A thread reading them at full rate puts continuous traffic on the same path
// the write has to travel to become visible, so hammering the flag competes
// with the arrival it is waiting for. Stay hot for a few reads (a short wait
// then pays nothing) and afterwards sleep, doubling to a small cap so a long
// wait stops competing. The cap bounds the latency this can add: 4 us per
// collective worst case, against a wait believed to be two orders larger.
#define SPIN_HOT 8
#define SPIN_NS0 128u
#define SPIN_NS_MAX 4096u
#ifndef OSAR_STALL_S
#define OSAR_STALL_S 3          // seconds of one spin before the stall word is written
#endif
#ifndef OSAR_STALL_TRAP_S
#define OSAR_STALL_TRAP_S 30    // seconds of one spin before the kernel traps (0 = never)
#endif
#define OSAR_SM_HZ 1592000000ull
#define STALL_GUARD 1ull
#define STALL_WAIT 2ull

// Bounded-spin bookkeeping for the two loops below (block 0 / thread 0 only:
// it is the timer block and the one writer of pad[0]). Checked every 256
// backoffs so the hot path pays nothing measurable.
__device__ __forceinline__ void osar_stall_check(Ctrl *c, int n, long long t_start,
                                                 uint64_t phase, uint64_t seq, int slot,
                                                 unsigned missing) {
  if ((n & 255) != 0) return;
  // The bootstrap collectives wait for peers to connect (seconds, legitimately):
  // no verdict before the protocol has run a few rounds.
  if (seq < 16) return;
  const long long el = clock64() - t_start;
  if (el > (long long)(OSAR_STALL_S * OSAR_SM_HZ) && c->pad[0] == 0) {
    c->pad[0] = (seq << 8) | (phase << 6) | ((uint64_t)(missing & 7u) << 3) | (uint64_t)slot;
    __threadfence_system();
  }
#if OSAR_STALL_TRAP_S > 0
  if (el > (long long)(OSAR_STALL_TRAP_S * OSAR_SM_HZ)) __trap();
#endif
}

__device__ __forceinline__ void osar_backoff(int &n, unsigned &ns) {
  if (++n <= SPIN_HOT) return;
#if __CUDA_ARCH__ >= 700
  __nanosleep(ns);
  ns = ns < SPIN_NS_MAX ? (ns << 1) : SPIN_NS_MAX;
#endif
}

// L2 prefetch hints for the peer-wait window (VLLM_GLM53_AR_PREFETCH): byte
// ranges of the weights the NEXT kernel after this collective streams. The
// wait is 20-40 us of idle DRAM on every rank (MEASUREMENTS 19차: wait 38.7
// of 45.5 us per collective, ~100 per step); warps 1..7 of every block walk
// these ranges with prefetch.global.L2 while thread 0 polls the peer flags,
// so the consumer finds its first megabytes in L2 (24 MB on this part).
// Passed by value: a CUDA-graph capture bakes the hint with the launch, and
// the shim learns the ranges from the consumers that follow each collective
// during the eager warmups that precede capture. n == 0 is exactly the old
// kernel -- no branch of it touches memory.
#define OSAR_MAXHINT 8
struct HintArgs {
  unsigned long long ptr[OSAR_MAXHINT];
  unsigned int len[OSAR_MAXHINT];
  int n;
};

__device__ __forceinline__ void osar_prefetch(const HintArgs &h,
                                              volatile int *landed) {
  // One 32 B sector index space over the concatenated ranges, interleaved
  // across the grid: block b, thread t takes sectors b*224 + (t-32) + k*10752,
  // so every block warms a uniform slice of every range. A prefix walk warms
  // a few blocks' tiles of the consumer and leaves its slowest block cold
  // (MEASUREMENTS 11차). Owning blocks stop as soon as the peers landed; the
  // work is bounded either way (budget <= 20 MB is ~2 us of issue per block).
  const int tid = (int)threadIdx.x - 32;
  const int stride = ARGRID * (ARTHREADS - 32);
  int idx = (int)blockIdx.x * (ARTHREADS - 32) + tid;
  for (int r = 0; r < h.n; ++r) {
    const unsigned long long base = h.ptr[r];
    const int nsec = (int)((h.len[r] + 31u) >> 5);
    for (; idx < nsec; idx += stride) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(
          base + ((unsigned long long)idx << 5)));
      if (*landed) return;
    }
    idx -= nsec;
  }
}

// Keep this arithmetic host-testable against an independent enumeration of
// the vector and scalar-tail accesses. Block zero also guards an empty call.
template <bool VECTOR_EXACT>
__host__ __device__ constexpr bool osar_block_owns(unsigned block, unsigned threads, int n) {
  return block == 0 || block * threads < (VECTOR_EXACT ? (n >> 3) : n);
}

__host__ __device__ constexpr int64_t osar_max_int64(int64_t a, int64_t b,
                                                    int64_t c, int64_t d) {
  int64_t ab = a > b ? a : b;
  int64_t cd = c > d ? c : d;
  return ab > cd ? ab : cd;
}

template <bool CONSUMER_PDL, bool COMPACT = false, bool WRAP_SAFE = false,
          bool MAX_INT64 = false, bool PACKETS = false, bool GATHER_INT64 = false>
__device__ __forceinline__ void k_oneshot_impl(Ctrl *c, const bf16 *src,
                                              bf16 *dst, int n, int nbytes,
                                              const HintArgs h, int rank) {
  // In a PDL chain this collective is also a consumer. Neither the input
  // nor the protocol's previous sequence may be read before its predecessor
  // has completed. No forward progress relies on concurrent residency.
  if constexpr (CONSUMER_PDL)
    asm volatile("griddepcontrol.wait;" ::: "memory");
  // The ordinary grid is fixed at ARGRID for the counter invariant; the
  // opt-in compact variant contributes four tickets per CTA, still 48 total.
  // At decode sizes
  // the smallest plain call has n = hidden and many blocks fall entirely past
  // the payload. The 16B lanes make this starker: with blockDim 256 and
  // n = 4096 there are only nv = 512 vectors, so threads 0..511 (blocks 0-1)
  // do all the copying and blocks 2..47 copy nothing. The ordinary path keeps
  // its conservative element-granular ownership. The PDL candidate instead
  // counts actual 16B lanes: at T=6 only 12 blocks, rather than all 48, need
  // the ring guard, writer fences and peer wait. Empty blocks retire sooner
  // and leave resources for the dependent MHC weight preparation.
  //
  // The peer wait is the expensive one. Data-owning blocks polling the same
  // three volatile flags for the whole RDMA latency window generate traffic
  // aimed at the very cache lines whose update they are waiting to observe.
  //
  // A block that owns no element needs neither spin: the guard protects a slot
  // it never writes, and the peer flags gate data it never reads. Only the
  // atomicAdd stays unconditional -- the invariant is that every launch adds
  // exactly ARGRID, and that is what makes the last-block test sound.
  //
  // Block 0 owns unconditionally: it covers the n % 8 scalar tail and ensures
  // an n == 0 collective cannot publish into a slot nobody guarded. Every
  // block still contributes its ticket and may become the final publisher.
  const bool owns = osar_block_owns<CONSUMER_PDL>(blockIdx.x, blockDim.x, n);
  // Block 0 always owns, so it always walks every phase and is the sample.
  const bool timer = (blockIdx.x == 0) && (threadIdx.x == 0);
  long long t0 = timer ? clock64() : 0;

  if (owns && threadIdx.x == 0) {
    // tx_seq cannot move while we spin: this launch has not published yet and
    // the previous launch on this stream already retired. Reading it once
    // halves the loop's traffic -- only ack_seq, which the proxy advances,
    // has to be re-read.
    const uint64_t want = c->tx_seq + 1;
    int sp = 0;
    unsigned ns = SPIN_NS0;
    while (want > c->ack_seq + RING) {
      osar_backoff(sp, ns);
      if (timer) osar_stall_check(c, sp, t0, STALL_GUARD, want, (int)(want % RING), 0u);
    }
    // a spin that ended is not a stall: clear the word so the proxy stops
    // repeating a bootstrap-era wait (K5 boot 05:28: seq=2 printed for 20 min)
    if (timer && c->pad[0] != 0) c->pad[0] = 0;
  }
  __syncthreads();
  uint64_t nxt = c->tx_seq + 1;
  int slot = (int)(nxt % RING);
  // Vector phase: 16B lanes, own values stashed per thread for the reduce.
  // Alignment is guaranteed ring-side by construction (MAXEL and every rx/tx
  // stride are multiples of 8 bf16) and source-side by a 16B check in
  // py_oneshot_impl; the scalar tail below covers n % 8.
  const uint4 *src4 = reinterpret_cast<const uint4 *>(src);
  uint4 *tx4 = reinterpret_cast<uint4 *>(c->tx[slot]);
  const int nv = n >> 3;
  uint4 mine[COMPACT ? OSAR_COMPACT_VECITER : VECITER];
  for (int v = blockIdx.x * blockDim.x + threadIdx.x, k = 0; v < nv;
       v += gridDim.x * blockDim.x, k++) {
    // Round 2 cache policy: __ldg is the read-only path for the producer's
    // L2-hot output; __stwt writes tx THROUGH instead of allocating L2 lines
    // -- the GPU never re-reads tx (the host proxy and our NIC, as the DMA
    // source of the RDMA write, are the readers, both from host memory), so
    // 64KB/collective stops churning the 24MB L2 the rest of the step
    // depends on, and a through-store may drain at the protocol fence for
    // free.
    uint4 val = __ldg(&src4[v]);
    __stwt(&tx4[v], val);
    mine[k] = val;
  }
  for (int i = (nv << 3) + blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    c->tx[slot][i] = src[i];
  // tx[slot] must be RDMA-readable before the last block publishes tx_seq.
  // Every WRITING block fences its own copy, and each fence precedes that
  // block's counter increment, so when the counter wraps this launch's ARGRID
  // all writes are already visible system-wide (a compact CTA contributes
  // four tickets only after every one of its writers fenced). A block that copied nothing
  // has nothing to order, so its fence is vacuous and is skipped.
  if (owns)
    __threadfence_system();
  // A thread fence orders only the calling thread's writes. Thread 0 must not
  // announce this block complete until every warp has copied and fenced its
  // own grid-stride elements; otherwise another block can publish tx_seq while
  // this block still has RDMA-visible payload writes in flight.
  __syncthreads();
  __shared__ bool last;
  if (threadIdx.x == 0) {
    if constexpr (WRAP_SAFE || MAX_INT64) {
      // Small integer packets fit one CTA. It contributes the same
      // 48 tickets as a whole BF16 grid, after every writer has fenced.
      constexpr unsigned weight = MAX_INT64 ? ARGRID : (COMPACT ? ARGRID / OSAR_COMPACT_GRID : 1);
      const auto old = atomicAdd((unsigned long long *)&c->done_ctr,
                                 (unsigned long long)weight);
      last = osar_publication_last(old, weight, nxt);
    } else {
      last = atomicAdd((unsigned long long *)&c->done_ctr, 1ULL) %
               ARGRID == ARGRID - 1;
    }
  }
  __syncthreads();
  if (last && threadIdx.x == 0) {
    c->nbytes[slot] = (uint64_t)nbytes;
    __threadfence_system();
    c->tx_seq = nxt;
    __threadfence_system();
  }
  // Every block has copied, fenced and contributed its publication ticket
  // before releasing the dependent grid. The consumer may load immutable
  // weights now; its dependency wait still gates every reduced-input read.
  if constexpr (CONSUMER_PDL)
    asm volatile("griddepcontrol.launch_dependents;");
  // Peer wait: rxf is only ever written by the peers' NICs, never by a block
  // of this kernel -- same independence argument as the guard above. Fence
  // stays where it always was: after the wait, before reading peer data.
  // The prefetch hints ride the wait: thread 0 polls, warps 1..7 warm L2
  // with the next kernel's weights until the peers land (s_landed) or their
  // slice is done. A non-owning block has no flag to wait for and simply
  // issues its slice; warp 0's other lanes go straight to the barrier. The
  // phase timer below still brackets thread 0's poll alone, so t_wait keeps
  // measuring the collective -- and shows any DRAM contention the prefetch
  // puts on the NIC's writes.
  __shared__ volatile int s_landed;
  if (threadIdx.x == 0) s_landed = 0;
  __syncthreads();
  long long t2 = timer ? clock64() : 0;
  if (owns && threadIdx.x == 0) {
    // The old form re-read every peer's flag on every pass, including peers
    // that had already landed. Remember who arrived and read only the first
    // one still missing -- same short-circuit shape as before, minus the
    // repeated reads of flags whose answer cannot change.
    bool got[NPEER] = {false, false, false};
    int left = NPEER, sp = 0;
    unsigned ns = SPIN_NS0;
    while (left) {
      for (int q = 0; q < NPEER; q++) {
        if (got[q]) continue;
        if (c->rxf[slot][q] >= nxt) {
          got[q] = true;
          --left;
        } else {
          osar_backoff(sp, ns);
          if (timer)
            osar_stall_check(c, sp, t2, STALL_WAIT, nxt, slot,
                             (got[0] ? 0u : 1u) | (got[1] ? 0u : 2u) | (got[2] ? 0u : 4u));
          break;
        }
      }
    }
    if (timer && c->pad[0] != 0) c->pad[0] = 0;
    s_landed = 1;
  } else if (h.n > 0 && threadIdx.x >= 32) {
    osar_prefetch(h, &s_landed);
  }
  __syncthreads();
  if (owns)
    __threadfence_system();
  if constexpr (GATHER_INT64) {
    // Preserve candidate bits and canonical rank order. Rank-major rows may
    // begin at an odd int64 offset, so the output uses aligned 8-byte stores.
    // The same CTA publishes all 48 tickets and consumes peers before the
    // next collective can reuse a slot; no NCCL event enters a bounded body.
    auto* output = reinterpret_cast<int64_t*>(dst);
    const int keys = n / 4;
    for (int i = threadIdx.x; i < keys; i += blockDim.x) {
      int peer = 0;
      for (int r = 0; r < 4; ++r) {
        output[r * keys + i] = r == rank
            ? reinterpret_cast<const int64_t*>(src)[i]
            : __ldcs(reinterpret_cast<const long long*>(c->rx[slot][peer++]) + i);
      }
    }
    return;
  }
  if constexpr (PACKETS) {
    // The immediate same-stream MHC consumer owns these addresses until it
    // completes. It folds ranks 0,1,2,3 and rounds to BF16 at its input load.
    // No collective may be enqueued between exchange and that consumer.
    if (timer) {
      auto **addresses = reinterpret_cast<const bf16 **>(dst);
      int peer = 0;
      for (int r = 0; r < 4; ++r)
        addresses[r] = r == rank ? src : c->rx[slot][peer++];
    }
    return;
  }
  // Reduce on the same 16B lanes: own values come from the registers filled
  // during the copy (identical mapping), peers' from the rx ring read with
  // __ldcs -- each rx line is consumed exactly once per collective and
  // overwritten by the NIC four collectives later, so evict-first streaming
  // keeps 192KB/collective out of L2. Conversions pack bf16x2 -> float2
  // (half the convert instructions). Vector lanes and the scalar tail fold
  // ranks in the same canonical order before the final BF16 rounding.
  const uint4 *rx4[NPEER] = {
      reinterpret_cast<const uint4 *>(c->rx[slot][0]),
      reinterpret_cast<const uint4 *>(c->rx[slot][1]),
      reinterpret_cast<const uint4 *>(c->rx[slot][2]),
  };
  uint4 *dst4 = reinterpret_cast<uint4 *>(dst);
  for (int v = blockIdx.x * blockDim.x + threadIdx.x, k = 0; v < nv;
       v += gridDim.x * blockDim.x, k++) {
    union {
      uint4 v4;
      __nv_bfloat162 b2[4];
      int64_t i64[2];
    } a, r0, r1, r2, o;
    a.v4 = mine[k];
    r0.v4 = __ldcs(&rx4[0][v]);
    r1.v4 = __ldcs(&rx4[1][v]);
    r2.v4 = __ldcs(&rx4[2][v]);
    if constexpr (MAX_INT64) {
#pragma unroll
      for (int p = 0; p < 2; p++)
        o.i64[p] = osar_max_int64(a.i64[p], r0.i64[p], r1.i64[p], r2.i64[p]);
    } else {
#pragma unroll
      for (int p = 0; p < 4; p++) {
        float2 fa = __bfloat1622float2(a.b2[p]);
        float2 f0 = __bfloat1622float2(r0.b2[p]);
        float2 f1 = __bfloat1622float2(r1.b2[p]);
        float2 f2 = __bfloat1622float2(r2.b2[p]);
        float2 acc;
        acc.x = osar_sum_rank_order(fa.x, f0.x, f1.x, f2.x, rank);
        acc.y = osar_sum_rank_order(fa.y, f0.y, f1.y, f2.y, rank);
        o.b2[p] = __float22bfloat162_rn(acc);
      }
    }
    dst4[v] = o.v4;
  }
  if constexpr (MAX_INT64) {
    // n counts two-byte transport cells. An odd key count leaves one
    // complete int64 after the 16-byte vectors; never convert its bits.
    for (int i = (nv << 1) + blockIdx.x * blockDim.x + threadIdx.x; i < n / 4;
         i += gridDim.x * blockDim.x) {
      reinterpret_cast<int64_t *>(dst)[i] = osar_max_int64(
          reinterpret_cast<const int64_t *>(src)[i],
          reinterpret_cast<const int64_t *>(c->rx[slot][0])[i],
          reinterpret_cast<const int64_t *>(c->rx[slot][1])[i],
          reinterpret_cast<const int64_t *>(c->rx[slot][2])[i]);
    }
  } else {
    for (int i = (nv << 3) + blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
      float acc = osar_sum_rank_order(__bfloat162float(src[i]),
                  __bfloat162float(c->rx[slot][0][i]),
                  __bfloat162float(c->rx[slot][1][i]),
                  __bfloat162float(c->rx[slot][2][i]), rank);
      dst[i] = __float2bfloat16(acc);
    }
  }

}

// Distinct entry points compile away the mode test and retain the original
// ordinary/consumer mode split. Rank selects only the FP32 summation order.
__global__ void k_oneshot(Ctrl *c, const bf16 *src, bf16 *dst, int n,
                          int nbytes, const HintArgs h, int rank) {
  k_oneshot_impl<false>(c, src, dst, n, nbytes, h, rank);
}

__global__ void k_oneshot_consumer(Ctrl *c, const bf16 *src, bf16 *dst, int n,
                                   int nbytes, const HintArgs h, int rank) {
  k_oneshot_impl<true>(c, src, dst, n, nbytes, h, rank);
}

__global__ void k_oneshot_packets(Ctrl *c, const bf16 *src, bf16 *addresses,
                                int n, int rank) {
  const HintArgs hints{};
  k_oneshot_impl<true, false, true, false, true>(c, src, addresses, n, n * 2, hints, rank);
}

__global__ void k_oneshot_max_int64(Ctrl *c, const bf16 *src, bf16 *dst, int n,
                                   int nbytes, const HintArgs h, int rank) {
  k_oneshot_impl<false, false, OSAR_COMPACT_CTA != 0, true>(c, src, dst, n, nbytes, h, rank);
}

__global__ void k_oneshot_gather_int64(Ctrl* c, const bf16* src, bf16* dst,
                                       int keys, int rank) {
  const HintArgs hints{};
  k_oneshot_impl<false, false, true, true, false, true>(
      c, src, dst, keys * 4, keys * 8, hints, rank);
}

#if OSAR_COMPACT_CTA
// Keep the disabled mode's entry points and generated instructions unchanged.
// The companion contributes one ticket per CTA but uses the same wrap-safe
// publication predicate as compact launches: all calls share done_ctr.
template <bool CONSUMER_PDL, bool COMPACT>
__global__ void k_oneshot_compact_mode(Ctrl *c, const bf16 *src, bf16 *dst,
                                      int n, int nbytes, const HintArgs h, int rank) {
  k_oneshot_impl<CONSUMER_PDL, COMPACT, true>(c, src, dst, n, nbytes, h, rank);
}
#endif

// ---------------- proxy ----------------
static void *proxy_fn(void *) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(PROXY_CORE, &set);
  sched_setaffinity(0, sizeof(set), &set);
#if OSAR_PROXY_INLINE
  // Stable descriptor addresses for the life of the proxy. post_send copies
  // the inline flag before returning; payloads remain protected by ring ACKs.
  OsarProxyInlineWrs prepared[RING][NPEER];
  for (int slot = 0; slot < RING; ++slot) {
    for (int p = 0; p < NPEER; ++p) {
      if (!prepared[slot][p].init(
              (uintptr_t)g_ctrl->tx[slot], g_mr->lkey,
              g_remote[p].rx_base + (uint64_t)slot * NPEER * MAXEL * 2,
              g_remote[p].rxf_base + (uint64_t)slot * NPEER * 8,
              g_remote[p].rkey, g_inline_cap[p])) {
        fprintf(stderr, "[oneshot] insufficient inline capability; proxy exiting\n");
        return nullptr;
      }
    }
  }
#endif
  uint64_t sent = 0, done[64] = {0};
  while (!g_ctrl->stop) {
    g_ctrl->proxy_beat++;
    uint64_t s = g_ctrl->tx_seq;
    while (sent < s) {
      sent++;
      int slot = (int)(sent % RING);
      uint32_t nb = (uint32_t)g_ctrl->nbytes[slot];
      for (int p = 0; p < NPEER; p++) {
#if OSAR_PROXY_INLINE
        if (prepared[slot][p].post(g_qp[p], sent, (unsigned)p, nb)) {
          fprintf(stderr, "[oneshot] inline post_send failed; proxy exiting\n");
          return nullptr;
        }
#else
        g_ctrl->flag_src[p] = sent;
        struct ibv_sge sge[2];
        struct ibv_send_wr wr[2], *bad;
        memset(wr, 0, sizeof(wr));
        sge[0].addr = (uintptr_t)g_ctrl->tx[slot];
        sge[0].length = nb;
        sge[0].lkey = g_mr->lkey;
        wr[0].wr_id = (sent << 4) | (unsigned)p;
        wr[0].sg_list = &sge[0];
        wr[0].num_sge = 1;
        wr[0].opcode = IBV_WR_RDMA_WRITE;
        wr[0].send_flags = 0;
        wr[0].wr.rdma.remote_addr =
            g_remote[p].rx_base + (uint64_t)slot * NPEER * MAXEL * 2;
        wr[0].wr.rdma.rkey = g_remote[p].rkey;
        wr[0].next = &wr[1];
        sge[1].addr = (uintptr_t)&g_ctrl->flag_src[p];
        sge[1].length = 8;
        sge[1].lkey = g_mr->lkey;
        wr[1].wr_id = (sent << 4) | 0x8 | (unsigned)p;
        wr[1].sg_list = &sge[1];
        wr[1].num_sge = 1;
        wr[1].opcode = IBV_WR_RDMA_WRITE;
        wr[1].send_flags = IBV_SEND_SIGNALED;
        wr[1].wr.rdma.remote_addr =
            g_remote[p].rxf_base + (uint64_t)slot * NPEER * 8;
        wr[1].wr.rdma.rkey = g_remote[p].rkey;
        if (ibv_post_send(g_qp[p], wr, &bad)) {
          fprintf(stderr, "[oneshot] post_send failed; proxy exiting\n");
          return nullptr;
        }
#endif
      }
#if OSAR_PROXY_INLINE
      if (sent == 1) {
        fprintf(stderr, "[oneshot] inline proxy serving peers=%d inline_bytes=8 "
                        "wr_reuse=1\n", NPEER);
      }
#endif
    }
    struct ibv_wc wc[16];
    int n = ibv_poll_cq(g_cq, 16, wc);
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr, "[oneshot] WC error %d; proxy exiting\n", wc[i].status);
        return nullptr;
      }
      uint64_t cs = wc[i].wr_id >> 4;
      if (++done[cs % 64] == NPEER) {
        done[cs % 64] = 0;
        if (cs > g_ctrl->ack_seq) g_ctrl->ack_seq = cs;
      }
    }
    {
      // The stall word: the kernel is spinning past OSAR_STALL_S. Say who is
      // waited for, with the counters the kernel spins on, every 5 s.
      static time_t last_stall = 0;
      const uint64_t sw = *(volatile uint64_t *)&g_ctrl->pad[0];
      if (sw != 0 && time(nullptr) - last_stall >= 5) {
        last_stall = time(nullptr);
        const int slot = (int)(sw & 7u);
        fprintf(stderr,
                "[oneshot] STALL rank=%d phase=%s seq=%llu slot=%d missing_peer_mask=0x%x "
                "tx_seq=%llu ack_seq=%llu rxf[slot]={%llu,%llu,%llu} beat=%llu\n",
                g_rank, ((sw >> 6) & 3u) == STALL_GUARD ? "guard(ring-space)" : "wait(peer-flags)",
                (unsigned long long)(sw >> 8), slot, (unsigned)((sw >> 3) & 7u),
                (unsigned long long)g_ctrl->tx_seq, (unsigned long long)g_ctrl->ack_seq,
                (unsigned long long)g_ctrl->rxf[slot][0], (unsigned long long)g_ctrl->rxf[slot][1],
                (unsigned long long)g_ctrl->rxf[slot][2], (unsigned long long)g_ctrl->proxy_beat);
      }

    }
  }
  return nullptr;
}

// ---------------- setup ----------------
static int find_gid(const char *myip, union ibv_gid *out) {
  unsigned a, b, c, d;
  CHK(sscanf(myip, "%u.%u.%u.%u", &a, &b, &c, &d) == 4);
  for (int i = 0; i < 16; i++) {
    union ibv_gid g;
    if (ibv_query_gid(g_ctx, 1, i, &g)) continue;
    if (g.raw[10] != 0xff || g.raw[11] != 0xff) continue;
    if (g.raw[12] != a || g.raw[13] != b || g.raw[14] != c || g.raw[15] != d)
      continue;
    char path[256], buf[64] = {0};
    snprintf(path, sizeof(path),
             "/sys/class/infiniband/%s/ports/1/gid_attrs/types/%d", DEVNAME, i);
    FILE *f = fopen(path, "r");
    if (!f) continue;
    size_t r = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    (void)r;
    if (!strstr(buf, "RoCE v2")) continue;
    *out = g;
    return i;
  }
  return -1;
}

static void init_ctx(int rank, int world, const std::string &myip) {
  g_rank = rank;
  g_world = world;
  int np = 0;
  for (int r = 0; r < world; r++)
    if (r != rank) g_peers[np++] = r;

  void *hp = aligned_alloc(4096, sizeof(Ctrl));
  CHK(hp != nullptr);
  memset(hp, 0, sizeof(Ctrl));
  // host-register: RDMA-registrable + GPU reads via ATS at cache speed
  CHK(cudaHostRegister(hp, sizeof(Ctrl), cudaHostRegisterDefault) ==
      cudaSuccess);
  g_ctrl = (Ctrl *)hp;

  int nd = 0;
  struct ibv_device **devs = ibv_get_device_list(&nd);
  for (int i = 0; i < nd; i++)
    if (!strcmp(ibv_get_device_name(devs[i]), DEVNAME))
      g_ctx = ibv_open_device(devs[i]);
  CHK(g_ctx != nullptr);
  struct ibv_port_attr pa;
  CHK(ibv_query_port(g_ctx, 1, &pa) == 0);
  union ibv_gid mygid;
  g_sgid = find_gid(myip.c_str(), &mygid);
  CHK(g_sgid >= 0);
  g_pd = ibv_alloc_pd(g_ctx);
  CHK(g_pd);
  g_cq = ibv_create_cq(g_ctx, 4096, nullptr, nullptr, 0);
  CHK(g_cq);
  g_mr = ibv_reg_mr(g_pd, hp, sizeof(Ctrl),
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  CHK(g_mr);

  srand((unsigned)(time(nullptr) ^ (rank * 7919)));
  for (int s = 0; s < NPEER; s++) {
    struct ibv_qp_init_attr qia;
    memset(&qia, 0, sizeof(qia));
    qia.send_cq = g_cq;
    qia.recv_cq = g_cq;
    qia.cap.max_send_wr = 1024;
    qia.cap.max_recv_wr = 4;
    qia.cap.max_send_sge = 1;
    qia.cap.max_inline_data = 16;
    qia.qp_type = IBV_QPT_RC;
    g_qp[s] = ibv_create_qp(g_pd, &qia);
    CHK(g_qp[s]);
    // ibv_create_qp returns the actual capabilities in qia.cap. An enabled
    // mode must not silently claim inline service on an incapable QP: local
    // setup failure participates in the shim's all-rank pre-connect vote.
    g_inline_cap[s] = qia.cap.max_inline_data;
#if OSAR_PROXY_INLINE
    CHK(g_inline_cap[s] >= sizeof(uint64_t));
#endif
    g_local[s].qpn = g_qp[s]->qp_num;
    g_local[s].psn = (uint32_t)(rand() & 0xffffff);
    g_local[s].rkey = g_mr->rkey;
    g_local[s].mtu = (uint32_t)pa.active_mtu;
    memcpy(g_local[s].gid, mygid.raw, 16);
    g_local[s].rx_base = (uintptr_t)&g_ctrl->rx[0][s][0];
    g_local[s].rxf_base = (uintptr_t)&g_ctrl->rxf[0][s];
  }
}

static void to_rts(struct ibv_qp *q, const Info *rem, uint32_t my_psn,
                   enum ibv_mtu mtu) {
  struct ibv_qp_attr a;
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = 1;
  a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
  CHK(ibv_modify_qp(q, &a,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS) == 0);
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = mtu;
  a.dest_qp_num = rem->qpn;
  a.rq_psn = rem->psn;
  a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;
  a.ah_attr.port_num = 1;
  memcpy(a.ah_attr.grh.dgid.raw, rem->gid, 16);
  a.ah_attr.grh.sgid_index = g_sgid;
  a.ah_attr.grh.hop_limit = 64;
  CHK(ibv_modify_qp(q, &a,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0);
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.sq_psn = my_psn;
  CHK(ibv_modify_qp(q, &a,
                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC) == 0);
}

// peer p's slot index for me = position of my rank in p's peer list
static int slot_of_me(int peer) {
  int cnt = 0;
  for (int r = 0; r < g_world; r++) {
    if (r == peer) continue;
    if (r == g_rank) return cnt;
    cnt++;
  }
  return -1;
}

// ---------------- pybind ----------------
static void py_init(int rank, int world, const std::string &myip) {
  // This fixed grid is part of the cudagraph-safe monotonic-counter protocol,
  // not a generic launch hint. Match one block to each SM on the only device
  // it was designed for. The Python bootstrap turns any throw here into a
  // lockstep all-rank vote for the NCCL fallback.
  int device = -1;
  cudaError_t err = cudaGetDevice(&device);
  TORCH_CHECK(err == cudaSuccess, "oneshot: cudaGetDevice failed: ",
              cudaGetErrorString(err));
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  TORCH_CHECK(err == cudaSuccess, "oneshot: cudaGetDeviceProperties failed: ",
              cudaGetErrorString(err));
  TORCH_CHECK(prop.major == 12 && prop.minor == 1 &&
                  prop.multiProcessorCount == ARGRID,
              "oneshot: expected GB10 SM121a with ", ARGRID,
              " SMs, got sm_", prop.major, prop.minor, " with ",
              prop.multiProcessorCount, " SMs");
  init_ctx(rank, world, myip);
}
static py::bytes py_local_infos() {
  // serialize NPEER Info blocks
  std::string s(reinterpret_cast<const char *>(g_local), sizeof(g_local));
  return py::bytes(s);
}
static void py_connect(std::vector<std::string> all) {
  for (int s = 0; s < NPEER; s++) {
    int peer = g_peers[s];
    const Info *pi = reinterpret_cast<const Info *>(all[peer].data());
    int myslot = slot_of_me(peer);
    g_remote[s] = pi[myslot];
    enum ibv_mtu mtu = (enum ibv_mtu)(g_local[s].mtu < g_remote[s].mtu
                                          ? g_local[s].mtu
                                          : g_remote[s].mtu);
    to_rts(g_qp[s], &g_remote[s], g_local[s].psn, mtu);
  }
  pthread_create(&g_proxy, nullptr, proxy_fn, nullptr);
  g_started = true;
}
static at::Tensor py_oneshot_impl(at::Tensor input,
                                     const std::vector<int64_t> &ptrs,
                                     const std::vector<int64_t> &lens,
                                     bool consumer_pdl = false) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16);
  TORCH_CHECK(input.is_contiguous());
  // The copy/reduce phases use 16B vectors; the ring side is aligned by
  // construction (every stride is a multiple of 8 bf16), the tensor side by
  // this check. The shim's eligibility check routes odd shapes to NCCL
  // before we get here, so this is a last-resort guard.
  TORCH_CHECK((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0,
              "oneshot: input not 16B aligned");
  int64_t n = input.numel();
  TORCH_CHECK(n <= MAXEL, "oneshot: tensor too large");
  TORCH_CHECK(ptrs.size() == lens.size() && ptrs.size() <= OSAR_MAXHINT,
              "oneshot: hint lists must pair up, at most OSAR_MAXHINT");
  auto out = torch::empty_like(input);
  const bf16 *src = reinterpret_cast<const bf16 *>(input.data_ptr());
  bf16 *dst = reinterpret_cast<bf16 *>(out.data_ptr());
  cudaStream_t st = c10::cuda::getCurrentCUDAStream();
  // Prefetch hints: (device pointer, bytes) pairs the shim learned for this
  // collective's ordinal. Baked into the launch, so a captured graph replays
  // them without any host code.
  HintArgs h;
  h.n = 0;
  for (size_t i = 0; i < ptrs.size(); ++i) {
    if (ptrs[i] == 0 || lens[i] <= 0) continue;
    h.ptr[h.n] = (unsigned long long)ptrs[i];
    h.len[h.n] = (unsigned int)std::min<int64_t>(lens[i], 0x7fffffff);
    ++h.n;
  }
  // Disabled mode: one launch, grid FIXED at ARGRID however small n is. The
  // last-block detection in k_oneshot is (done_ctr % ARGRID == ARGRID-1),
  // which is only sound if every launch contributes exactly ARGRID
  // increments. Enabled mode keeps total tickets at 48 with a separate
  // wrap-safe companion for ordinary launches. Only 1..32768-element PDL
  // consumers use 12 CTAs, each contributing four tickets. Their fixed
  // two-vector stash is independent of the optional MAXEL build override.
  if (consumer_pdl) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(ARGRID);
    cfg.blockDim = dim3(ARTHREADS);
    cfg.stream = st;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
#if OSAR_COMPACT_CTA
    const bool compact = osar_compact_eligible((uint64_t)n);
    if (compact) cfg.gridDim = dim3(OSAR_COMPACT_GRID);
    const auto err = compact
        ? cudaLaunchKernelEx(&cfg, k_oneshot_compact_mode<true, true>, g_ctrl,
                              src, dst, (int)n, (int)(n * 2), h, g_rank)
        : cudaLaunchKernelEx(&cfg, k_oneshot_compact_mode<true, false>, g_ctrl,
                              src, dst, (int)n, (int)(n * 2), h, g_rank);
#else
    const auto err = cudaLaunchKernelEx(&cfg, k_oneshot_consumer, g_ctrl, src,
                                         dst, (int)n, (int)(n * 2), h, g_rank);
#endif
    TORCH_CHECK(err == cudaSuccess, "oneshot PDL launch: ",
                cudaGetErrorString(err));
  } else {
#if OSAR_COMPACT_CTA
    k_oneshot_compact_mode<false, false><<<ARGRID, ARTHREADS, 0, st>>>(
        g_ctrl, src, dst, (int)n, (int)(n * 2), h, g_rank);
#else
    k_oneshot<<<ARGRID, ARTHREADS, 0, st>>>(g_ctrl, src, dst, (int)n,
                                          (int)(n * 2), h, g_rank);
#endif
  }
  return out;
}
static at::Tensor py_oneshot(at::Tensor input) {
  return py_oneshot_impl(input, {}, {});
}
static at::Tensor py_oneshot_hint(at::Tensor input,
                                      std::vector<int64_t> ptrs,
                                      std::vector<int64_t> lens) {
  return py_oneshot_impl(input, ptrs, lens);
}
static at::Tensor py_oneshot_consumer(at::Tensor input) {
  return py_oneshot_impl(input, {}, {}, true);
}

static at::Tensor py_oneshot_packets(at::Tensor input) {
  TORCH_CHECK(g_started && input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
              input.is_contiguous() && input.dim() == 2 && input.size(1) == 4096 &&
              input.size(0) > 0 && input.size(0) <= 64 && input.numel() <= MAXEL &&
              (reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0,
              "one-shot packets require a live TP4 transport and aligned BF16 [1..64,4096]");
  auto addresses = torch::empty({4}, input.options().dtype(at::kLong));
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(ARGRID);
  cfg.blockDim = dim3(ARTHREADS);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  auto err = cudaLaunchKernelEx(&cfg, k_oneshot_packets, g_ctrl,
      reinterpret_cast<const bf16 *>(input.data_ptr()),
      reinterpret_cast<bf16 *>(addresses.data_ptr()), (int)input.numel(), g_rank);
  TORCH_CHECK(err == cudaSuccess, "one-shot packet launch: ", cudaGetErrorString(err));
  return addresses;
}

// Reserve before the producer, publish only after its entire grid has retired.
// The publication kernel waits for the producer's whole grid before publishing.
// Only then may its dependent MHC begin preparing immutable weights.
__global__ void k_reserve_packets(Ctrl* c, int64_t* reservation) {
  const uint64_t next = c->tx_seq + 1;
  int sp = 0;
  unsigned ns = SPIN_NS0;
  const auto start = clock64();
  while (next > c->ack_seq + RING) {
    osar_backoff(sp, ns);
    osar_stall_check(c, sp, start, STALL_GUARD, next, next % RING, 0u);
  }
  c->pad[0] = 0;
  reservation[0] = reinterpret_cast<int64_t>(c->tx[next % RING]);
  reservation[1] = next;
}

__global__ void k_publish_packets(Ctrl* c, const int64_t* reservation,
                                  int64_t* addresses, int nbytes, int rank) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint64_t next = reservation[1];
  const int slot = next % RING;
  // Keep the existing ticket count aligned even when the next collective is
  // an int64 MAX or an ordinary copied BF16 packet.
  atomicAdd((unsigned long long*)&c->done_ctr, (unsigned long long)ARGRID);
  c->nbytes[slot] = nbytes;
  __threadfence_system();
  c->tx_seq = next;
  __threadfence_system();
  asm volatile("griddepcontrol.launch_dependents;");
  const auto start = clock64();
  int sp = 0;
  unsigned ns = SPIN_NS0;
  for (int p = 0; p < NPEER; ++p) {
    while (c->rxf[slot][p] < next) {
      osar_backoff(sp, ns);
      osar_stall_check(c, sp, start, STALL_WAIT, next, slot, 1u << p);
    }
  }
  c->pad[0] = 0;
  __threadfence_system();
  int peer = 0;
  for (int r = 0; r < 4; ++r)
    addresses[r] = reinterpret_cast<int64_t>(r == rank ? c->tx[slot] : c->rx[slot][peer++]);
}

static void check_producer_template(const at::Tensor& input) {
  TORCH_CHECK(g_started && input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
              input.is_contiguous() && input.dim() == 2 && input.size(1) == 4096 &&
              input.size(0) > 0 && input.size(0) <= 32 &&
              (reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0,
              "one-shot producer requires live TP4 and BF16 [1..32,4096] metadata");
}

static at::Tensor py_reserve_packets(at::Tensor input) {
  check_producer_template(input);
  auto reservation = torch::empty({2}, input.options().dtype(at::kLong));
  k_reserve_packets<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(g_ctrl, reservation.data_ptr<int64_t>());
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "one-shot reservation: ", cudaGetErrorString(err));
  return reservation;
}

static at::Tensor py_publish_packets(at::Tensor input, at::Tensor reservation) {
  check_producer_template(input);
  TORCH_CHECK(reservation.device() == input.device() && reservation.scalar_type() == at::kLong &&
              reservation.is_contiguous() && reservation.numel() == 2,
              "invalid one-shot producer reservation");
  auto addresses = torch::empty({4}, input.options().dtype(at::kLong));
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(1);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  auto err = cudaLaunchKernelEx(&cfg, k_publish_packets, g_ctrl, reservation.data_ptr<int64_t>(),
      addresses.data_ptr<int64_t>(), int(input.numel() * 2), g_rank);
  TORCH_CHECK(err == cudaSuccess, "one-shot publication: ", cudaGetErrorString(err));
  return addresses;
}

static at::Tensor py_oneshot_max_int64(at::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kLong && input.is_contiguous(),
              "oneshot MAX requires contiguous CUDA int64");
  TORCH_CHECK(input.numel() > 0 && input.numel() <= 64 &&
                  (reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0,
              "oneshot MAX requires 1..64 aligned keys");
  auto *data = reinterpret_cast<bf16 *>(input.data_ptr());
  HintArgs hints{};
  k_oneshot_max_int64<<<1, ARTHREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      g_ctrl, data, data, (int)input.numel() * 4, (int)input.numel() * 8, hints, g_rank);
  return input;
}

static at::Tensor py_oneshot_gather_int64(at::Tensor input) {
  TORCH_CHECK(g_ctrl && g_started && input.is_cuda() && input.scalar_type() == at::kLong &&
              input.is_contiguous() && input.numel() > 0 && input.numel() <= 512 &&
              (reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0,
              "oneshot gather requires live TP4 and 1..512 aligned CUDA int64 keys");
  // <=4096 bytes means at most one uint4 per thread in the shared publisher.
  std::vector<int64_t> shape{4};
  shape.insert(shape.end(), input.sizes().begin(), input.sizes().end());
  auto output = torch::empty(shape, input.options());
  k_oneshot_gather_int64<<<1, ARTHREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      g_ctrl, reinterpret_cast<const bf16*>(input.data_ptr()),
      reinterpret_cast<bf16*>(output.data_ptr()), int(input.numel()), g_rank);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "one-shot int64 gather launch failed");
  return output;
}
// The phase counters (SM cycles, monotonic) for a probe that wants the wait
// per collective with and without hints: [guard, copy, wait, reduce, calls].
static std::vector<int64_t> py_phase_counters() {
  if (!g_ctrl) return {};
  return {(int64_t)g_ctrl->t_guard, (int64_t)g_ctrl->t_copy,
          (int64_t)g_ctrl->t_wait, (int64_t)g_ctrl->t_reduce,
          (int64_t)g_ctrl->t_calls};
}
static bool py_healthy() {
  if (!g_started) return false;
  static uint64_t last = 0;
  uint64_t b = g_ctrl->proxy_beat;
  bool ok = b != last || b == 0;
  last = b;
  return ok;
}
static void py_shutdown() {
  if (!g_started) return;
  g_ctrl->stop = 1;
  pthread_join(g_proxy, nullptr);
  g_started = false;
}

#ifndef ST_ONESHOT_LOCAL_TEST
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &py_init);
  m.def("local_infos", &py_local_infos);
  m.def("connect", &py_connect);
  m.def("oneshot_ar", &py_oneshot);
  m.def("oneshot_ar_consumer", &py_oneshot_consumer);
  m.def("oneshot_packets", &py_oneshot_packets);
  m.def("reserve_packets", &py_reserve_packets);
  m.def("publish_packets", &py_publish_packets);
  m.def("oneshot_max_int64", &py_oneshot_max_int64);
  m.def("oneshot_gather_int64", &py_oneshot_gather_int64);
  m.def("oneshot_ar_hint", &py_oneshot_hint);
  m.def("phase_counters", &py_phase_counters);
  m.def("transport_modes", []() {
    return std::vector<int64_t>{OSAR_COMPACT_CTA, OSAR_PROXY_INLINE,
        (int64_t)g_inline_cap[0], (int64_t)g_inline_cap[1],
        (int64_t)g_inline_cap[2]};
  });
  m.def("healthy", &py_healthy);
  m.def("shutdown", &py_shutdown);
}

#endif  // ST_ONESHOT_LOCAL_TEST
