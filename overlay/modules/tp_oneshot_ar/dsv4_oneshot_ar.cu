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
#define MAXEL 131072  // 256KB bf16 — size gate keeps callers <= this
#define PROXY_CORE 18

#define CHK(x)                                                            \
  do {                                                                    \
    if (!(x)) {                                                           \
      fprintf(stderr, "[oneshot] FAIL %s:%d %s errno=%d(%s)\n", __FILE__, \
              __LINE__, #x, errno, strerror(errno));                      \
      throw std::runtime_error("oneshot verbs failure");                  \
    }                                                                     \
  } while (0)

typedef __nv_bfloat16 bf16;

struct Ctrl {
  volatile uint64_t tx_seq;
  volatile uint64_t ack_seq;
  volatile uint64_t stop;
  volatile uint64_t proxy_beat;          // watchdog heartbeat
  uint64_t flag_src[NPEER];
  uint64_t nbytes[RING];                 // payload size per slot (GPU sets)
  volatile uint64_t rxf[RING][NPEER];    // inbound flags (slot-major)
  uint64_t pad[8];
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
// k_guard used to be its own <<<1,32>>> launch. Its condition reads only
// globals (tx_seq, ack_seq), identical for every block, so each block can spin
// on it independently -- no cross-block coordination, and nothing here waits on
// a block of this same kernel, so a block that is not resident yet cannot
// deadlock us. Folding it in removes one launch per collective; at ~104
// collectives a decode step that is ~104 kernels of the 2,210 a step runs.
__global__ void k_copy_in(Ctrl *c, const bf16 *src, int n) {
  if (threadIdx.x == 0)
    while ((c->tx_seq + 1) > (c->ack_seq + RING)) {
    }
  __syncthreads();
  uint64_t nxt = c->tx_seq + 1;
  int slot = (int)(nxt % RING);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    c->tx[slot][i] = src[i];
}
__global__ void k_signal(Ctrl *c, int nbytes) {
  if (threadIdx.x == 0) {
    uint64_t nxt = c->tx_seq + 1;
    int slot = (int)(nxt % RING);
    c->nbytes[slot] = (uint64_t)nbytes;
    __threadfence_system();
    c->tx_seq = nxt;
    __threadfence_system();
  }
}
// k_wait used to be its own <<<1,32>>> launch, for the same reason and with
// the same argument: it polls the peers' inbound flags, which are globals, so
// every block can wait on them by itself. The fence stays where it was -- after
// the wait, before anything reads peer data.
__global__ void k_reduce(Ctrl *c, const bf16 *src, bf16 *dst, int n) {
  uint64_t s = c->tx_seq;
  int slot = (int)(s % RING);
  if (threadIdx.x == 0) {
    while (c->rxf[slot][0] < s || c->rxf[slot][1] < s || c->rxf[slot][2] < s) {
    }
  }
  __syncthreads();
  __threadfence_system();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    float acc = __bfloat162float(src[i]) +
                __bfloat162float(c->rx[slot][0][i]) +
                __bfloat162float(c->rx[slot][1][i]) +
                __bfloat162float(c->rx[slot][2][i]);
    dst[i] = __float2bfloat16(acc);
  }
}

// ---------------- proxy ----------------
static void *proxy_fn(void *) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(PROXY_CORE, &set);
  sched_setaffinity(0, sizeof(set), &set);
  uint64_t sent = 0, done[64] = {0};
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
  int grid = (int)((n + 255) / 256);
  if (grid > 256) grid = 256;
  // Three launches, not five: the ring-space guard rides in k_copy_in and the
  // peer wait rides in k_reduce. k_signal has to stay separate -- it may only
  // run once every block of k_copy_in has finished, and there is no grid-wide
  // "all blocks done" inside a kernel without a cooperative launch or an atomic
  // counter, and a counter would have to be reset somewhere a cudagraph replay
  // cannot see.
  k_copy_in<<<grid, 256, 0, st>>>(g_ctrl, src, (int)n);
  k_signal<<<1, 32, 0, st>>>(g_ctrl, (int)(n * 2));
  k_reduce<<<grid, 256, 0, st>>>(g_ctrl, src, dst, (int)n);
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
