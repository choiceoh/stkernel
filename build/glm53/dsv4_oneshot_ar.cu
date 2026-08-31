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
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define NPEER 3
#define RING 4
// Size gate. 131072 bf16 = 256 KB = 32 tokens at hidden 4096, which is GLM's
// max captured verify (MAX_SEQS 4 * (1+7)). It is NOT dsv4's: that stack
// captures MAX_NUM_SEQS 32 * (1+5) = 192 tokens, so above C=5 every AllReduce
// falls out of one-shot and back onto the NCCL ring -- silently, because the
// gate is a size test, not an error. Overridable at build time so the
// latency-vs-bandwidth crossover can be swept instead of assumed; the buffers
// below scale linearly with it (RING + RING*NPEER = 16 copies), so 786432
// (C=32) costs 24 MiB/rank against 4.
#ifndef MAXEL
#define MAXEL 131072
#endif
#define ARGRID 48     // GB10 / SM121a has exactly 48 SMs
#define ARTHREADS 256
#define PROXY_CORE 18
// clock64() counts SM cycles. The SM clock is pinned at 1592 MHz on this
// fleet (nvidia-smi clocks.sm, flat across load), so one constant converts.
// The absolute us are only as good as that assumption; the RATIO between the
// four phases does not depend on it at all, and the ratio is the answer we
// are after.
#define SM_CLK_MHZ 1592
#define REPORT_SEC 10

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
  uint64_t pad[3];
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

__device__ __forceinline__ void osar_backoff(int &n, unsigned &ns) {
  if (++n <= SPIN_HOT) return;
#if __CUDA_ARCH__ >= 700
  __nanosleep(ns);
  ns = ns < SPIN_NS_MAX ? (ns << 1) : SPIN_NS_MAX;
#endif
}

__global__ void k_oneshot(Ctrl *c, const bf16 *src, bf16 *dst, int n,
                          int nbytes) {
  // The grid is fixed at ARGRID for the counter invariant, so at decode sizes
  // the smallest plain call has n = hidden and many blocks fall entirely past
  // the payload: with blockDim 256 and n 4096, blocks 16..47 copy nothing and
  // reduce nothing. They still pay the launch/sync/counter cost.
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
  // Block 0 owns unconditionally so an n == 0 collective still takes the ring
  // guard rather than publishing into a slot nobody checked.
  const bool owns = (blockIdx.x == 0) || (blockIdx.x * blockDim.x < n);
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
    while (want > c->ack_seq + RING) osar_backoff(sp, ns);
  }
  long long t1 = timer ? clock64() : 0;
  __syncthreads();
  uint64_t nxt = c->tx_seq + 1;
  int slot = (int)(nxt % RING);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    c->tx[slot][i] = src[i];
  // tx[slot] must be RDMA-readable before the last block publishes tx_seq.
  // Every WRITING block fences its own copy, and each fence precedes that
  // block's counter increment, so when the counter wraps this launch's ARGRID
  // all writes are already visible system-wide. A block that copied nothing
  // has nothing to order, so its fence is vacuous and is skipped.
  if (owns)
    __threadfence_system();
  // A thread fence orders only the calling thread's writes. Thread 0 must not
  // announce this block complete until every warp has copied and fenced its
  // own grid-stride elements; otherwise another block can publish tx_seq while
  // this block still has RDMA-visible payload writes in flight.
  __syncthreads();
  __shared__ bool last;
  if (threadIdx.x == 0)
    last = atomicAdd((unsigned long long *)&c->done_ctr, 1ULL) %
               ARGRID == ARGRID - 1;
  __syncthreads();
  if (last && threadIdx.x == 0) {
    c->nbytes[slot] = (uint64_t)nbytes;
    __threadfence_system();
    c->tx_seq = nxt;
    __threadfence_system();
  }
  // Peer wait: rxf is only ever written by the peers' NICs, never by a block
  // of this kernel -- same independence argument as the guard above. Fence
  // stays where it always was: after the wait, before reading peer data.
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
          break;
        }
      }
    }
  }
  long long t3 = timer ? clock64() : 0;
  __syncthreads();
  if (owns)
    __threadfence_system();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    float acc = __bfloat162float(src[i]) +
                __bfloat162float(c->rx[slot][0][i]) +
                __bfloat162float(c->rx[slot][1][i]) +
                __bfloat162float(c->rx[slot][2][i]);
    dst[i] = __float2bfloat16(acc);
  }
  if (timer) {
    long long t4 = clock64();
    c->t_guard += (uint64_t)(t1 - t0);
    c->t_copy += (uint64_t)(t2 - t1);
    c->t_wait += (uint64_t)(t3 - t2);
    c->t_reduce += (uint64_t)(t4 - t3);
    c->t_calls += 1;
  }
}

// ---------------- proxy ----------------
static void *proxy_fn(void *) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(PROXY_CORE, &set);
  sched_setaffinity(0, sizeof(set), &set);
  uint64_t sent = 0, done[64] = {0};
  time_t last_report = 0;
  uint64_t last_guard = 0, last_copy = 0, last_wait = 0, last_reduce = 0,
           last_calls = 0;
  while (!g_ctrl->stop) {
    g_ctrl->proxy_beat++;
    uint64_t s = g_ctrl->tx_seq;
    while (sent < s) {
      sent++;
      int slot = (int)(sent % RING);
      uint32_t nb = (uint32_t)g_ctrl->nbytes[slot];
      for (int p = 0; p < NPEER; p++) {
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
      }
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
    // Phase report. The reader has to live here and not in the shim: under a
    // full-decode cudagraph the Python entry runs once at capture and never
    // again, while the kernel keeps accumulating on every replay. The proxy
    // is the only host code that runs per collective for the life of the boot.
    // Rank 0 only, every REPORT_SEC seconds, deltas since the last report so a
    // slow warmup does not smear the steady state.
    if (g_rank == 0) {
      time_t now = time(nullptr);
      if (last_report == 0) last_report = now;
      if (now - last_report >= REPORT_SEC) {
        uint64_t calls = g_ctrl->t_calls, dn = calls - last_calls;
        if (dn > 0) {
          double k = 1.0 / (double)dn / (double)SM_CLK_MHZ;  // cycles -> us
          fprintf(stderr,
                  "[osar] phase us/collective (n=%llu): guard=%.1f copy=%.1f "
                  "wait=%.1f reduce=%.1f | total=%.1f  @%d MHz assumed\n",
                  (unsigned long long)dn,
                  (double)(g_ctrl->t_guard - last_guard) * k,
                  (double)(g_ctrl->t_copy - last_copy) * k,
                  (double)(g_ctrl->t_wait - last_wait) * k,
                  (double)(g_ctrl->t_reduce - last_reduce) * k,
                  (double)(g_ctrl->t_guard - last_guard +
                           g_ctrl->t_copy - last_copy +
                           g_ctrl->t_wait - last_wait +
                           g_ctrl->t_reduce - last_reduce) * k,
                  SM_CLK_MHZ);
          last_guard = g_ctrl->t_guard;
          last_copy = g_ctrl->t_copy;
          last_wait = g_ctrl->t_wait;
          last_reduce = g_ctrl->t_reduce;
          last_calls = calls;
        }
        last_report = now;
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
static torch::Tensor py_oneshot(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.is_contiguous());
  int64_t n = input.numel();
  TORCH_CHECK(n <= MAXEL, "oneshot: tensor too large");
  auto out = torch::empty_like(input);
  const bf16 *src = reinterpret_cast<const bf16 *>(input.data_ptr());
  bf16 *dst = reinterpret_cast<bf16 *>(out.data_ptr());
  cudaStream_t st = c10::cuda::getCurrentCUDAStream();
  // One launch, and the grid is FIXED at ARGRID however small n is: the
  // last-block detection in k_oneshot is (done_ctr % ARGRID == ARGRID-1),
  // which is only sound if every launch contributes exactly ARGRID
  // increments. The 48-block grid fills GB10 once and covers MAXEL through
  // the kernel's grid-stride loops; empty decode blocks only sync/increment.
  k_oneshot<<<ARGRID, ARTHREADS, 0, st>>>(g_ctrl, src, dst, (int)n,
                                          (int)(n * 2));
  return out;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &py_init);
  m.def("local_infos", &py_local_infos);
  m.def("connect", &py_connect);
  m.def("oneshot_ar", &py_oneshot);
  m.def("healthy", &py_healthy);
  m.def("shutdown", &py_shutdown);
}
