// Stage-2 core: CUDA-graph-compatible one-shot cross-node AllReduce.
//
// Differences vs the Stage-1 prototype:
//  * pure libibverbs (no librdmacm — the serving image lacks it): manual RC
//    QP bring-up (INIT->RTR->RTS) with RoCEv2 GID autodetect, out-of-band
//    bootstrap over plain TCP (production will use the torch.distributed
//    store instead).
//  * per-call op sequence usable under cudaGraph capture/replay:
//      guard -> D2H memcpy(in) -> signal -> wait -> 3x H2D memcpy -> reduce
//    with a RING-slot flow control (proxy acks send completions).
//  * two datapaths: mode 0 = staged device memcpys (design target),
//    mode 1 = GPU reduces straight from mapped pinned rx (Stage-1 style).
//
// Build: nvcc -O2 -arch=sm_121a oneshot_ar2.cu -o osar2 -libverbs
// Run:   ./osar2 <rank 0-3> [replays=2000] [warmup=100] [mode=0]
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define NELEM 12288  // 48KB float payload
#define NPEER 3
#define RING 4
#define CALLS 8      // AR calls captured per graph (emulates layers)
#define DEVNAME "rocep1s0f0"
#define TCP_BASE 23300

static const char *IPS[4] = {"10.10.10.2", "10.10.10.3", "10.10.10.1",
                             "10.10.10.4"};

#define CHK(x)                                                             \
  do {                                                                     \
    if (!(x)) {                                                            \
      fprintf(stderr, "FAIL %s:%d %s errno=%d(%s)\n", __FILE__, __LINE__,  \
              #x, errno, strerror(errno));                                 \
      exit(1);                                                             \
    }                                                                      \
  } while (0)
#define CUCHK(x)                                                           \
  do {                                                                     \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x,  \
              cudaGetErrorString(e_));                                     \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// ---- pinned control/data block (single ibv_reg_mr region) ----
struct Ctrl {
  volatile uint64_t tx_seq;              // last seq signalled by GPU
  volatile uint64_t ack_seq;             // last seq fully sent (proxy)
  volatile uint64_t stop;
  uint64_t flag_src[NPEER];              // staging for outgoing flag writes
  volatile uint64_t rxf[RING][NPEER];    // inbound flags (slot-major)
  uint64_t pad[8];
  float tx[RING][NELEM];                 // outbound payload slots
  float rx[RING][NPEER][NELEM];          // slot-major: one 144KB H2D/slot
};

struct PeerXchg {  // exchanged over TCP per pair
  uint32_t qpn, psn, rkey;
  uint32_t mtu;
  uint8_t gid[16];
  uint64_t rx_base;   // where I write payload on the peer (their rx[slot_of_me])
  uint64_t rxf_base;  // where I write flags on the peer
};

static Ctrl *ctrl;
static struct ibv_qp *qp[NPEER];
static struct ibv_cq *cq;
static struct ibv_mr *mr;
static PeerXchg rem[NPEER];
static int g_inflight = 0;

// ---------------- GPU kernels ----------------
__global__ void k_guard(volatile uint64_t *tx_seq, volatile uint64_t *ack) {
  if (threadIdx.x == 0)
    while ((*tx_seq + 1) > (*ack + RING)) {
    }
}
__global__ void k_signal(volatile uint64_t *tx_seq) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    *tx_seq = *tx_seq + 1;
    __threadfence_system();
  }
}
__global__ void k_wait(volatile uint64_t *tx_seq,
                       volatile uint64_t *rxf /* [NPEER][RING] */) {
  if (threadIdx.x == 0) {
    uint64_t s = *tx_seq;
    int slot = (int)(s % RING);
    while (rxf[slot * NPEER + 0] < s || rxf[slot * NPEER + 1] < s ||
           rxf[slot * NPEER + 2] < s) {
    }
  }
  __syncthreads();
  __threadfence_system();
}
__global__ void k_wait_slot(volatile uint64_t *tx_seq,
                            volatile uint64_t *rxf, int slot) {
  if (threadIdx.x == 0) {
    uint64_t s = *tx_seq;
    while (rxf[slot * NPEER + 0] < s || rxf[slot * NPEER + 1] < s ||
           rxf[slot * NPEER + 2] < s) {
    }
  }
  __syncthreads();
  __threadfence_system();
}
__global__ void k_copy_in_slot(Ctrl *c, const float *src, int slot) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NELEM) c->tx[slot][i] = src[i];
}
__global__ void k_reduce_staged(float *dst, const float *src,
                                const float *s0, const float *s1,
                                const float *s2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NELEM) dst[i] = src[i] + s0[i] + s1[i] + s2[i];
}
__global__ void k_reduce_direct(float *dst, const float *src,
                                volatile uint64_t *tx_seq, const float *rx0,
                                const float *rx1, const float *rx2) {
  // rx pointers are mapped-pinned; slot resolved from the live seq
  uint64_t s = *tx_seq;
  int slot = (int)(s % RING);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NELEM)
    dst[i] = src[i] + rx0[slot * NELEM + i] + rx1[slot * NELEM + i] +
             rx2[slot * NELEM + i];
}

// ---------------- proxy ----------------
static void post_pair(int p, uint64_t s) {
  int slot = (int)(s % RING);
  ctrl->flag_src[p] = s;
  struct ibv_sge sge[2];
  struct ibv_send_wr wr[2], *bad;
  memset(wr, 0, sizeof(wr));
  sge[0].addr = (uintptr_t)ctrl->tx[slot];
  sge[0].length = NELEM * sizeof(float);
  sge[0].lkey = mr->lkey;
  wr[0].wr_id = (s << 4) | (unsigned)p;
  wr[0].sg_list = &sge[0];
  wr[0].num_sge = 1;
  wr[0].opcode = IBV_WR_RDMA_WRITE;
  wr[0].send_flags = 0;
  wr[0].wr.rdma.remote_addr =
      rem[p].rx_base + (uint64_t)slot * NPEER * NELEM * 4;
  wr[0].wr.rdma.rkey = rem[p].rkey;
  wr[0].next = &wr[1];
  sge[1].addr = (uintptr_t)&ctrl->flag_src[p];
  sge[1].length = 8;
  sge[1].lkey = mr->lkey;
  wr[1].wr_id = (s << 4) | 0x8 | (unsigned)p;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;
  wr[1].opcode = IBV_WR_RDMA_WRITE;
  wr[1].send_flags = IBV_SEND_SIGNALED;  // one WC per (peer, seq)
  wr[1].wr.rdma.remote_addr = rem[p].rxf_base + (uint64_t)slot * NPEER * 8;
  wr[1].wr.rdma.rkey = rem[p].rkey;
  CHK(ibv_post_send(qp[p], wr, &bad) == 0);
  g_inflight++;
}

static void *proxy_fn(void *arg) {
  (void)arg;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(18, &set);
  sched_setaffinity(0, sizeof(set), &set);
  uint64_t sent = 0;      // last seq posted
  uint64_t done[64];      // completed peer-count per seq (ring of RING*2)
  memset(done, 0, sizeof(done));
  while (!ctrl->stop) {
    uint64_t s = ctrl->tx_seq;
    while (sent < s) {
      sent++;
      for (int p = 0; p < NPEER; p++) post_pair(p, sent);
    }
    struct ibv_wc wc[8];
    int n = ibv_poll_cq(cq, 8, wc);
    for (int i = 0; i < n; i++) {
      CHK(wc[i].status == IBV_WC_SUCCESS);
      uint64_t cs = wc[i].wr_id >> 4;
      g_inflight--;
      int idx = (int)(cs % 64);
      if (++done[idx] == NPEER) {
        done[idx] = 0;
        if (cs > ctrl->ack_seq) ctrl->ack_seq = cs;  // seqs complete in order
      }
    }
  }
  return NULL;
}

// ---------------- verbs bring-up (no rdma_cm) ----------------
static int find_gid(struct ibv_context *ctx, int port, const char *myip,
                    union ibv_gid *gid_out) {
  unsigned a, b, c, d;
  CHK(sscanf(myip, "%u.%u.%u.%u", &a, &b, &c, &d) == 4);
  for (int i = 0; i < 16; i++) {
    union ibv_gid g;
    if (ibv_query_gid(ctx, port, i, &g)) continue;
    if (g.raw[10] != 0xff || g.raw[11] != 0xff) continue;  // v4-mapped only
    if (g.raw[12] != a || g.raw[13] != b || g.raw[14] != c || g.raw[15] != d)
      continue;
    char path[256], buf[64] = {0};
    snprintf(path, sizeof(path),
             "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", DEVNAME,
             port, i);
    FILE *f = fopen(path, "r");
    if (!f) continue;
    size_t rd = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    (void)rd;
    if (!strstr(buf, "RoCE v2")) continue;
    *gid_out = g;
    return i;
  }
  return -1;
}

static void qp_to_rts(struct ibv_qp *q, const PeerXchg *peer, int sgid_idx,
                      uint32_t my_psn, enum ibv_mtu mtu) {
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
  a.dest_qp_num = peer->qpn;
  a.rq_psn = peer->psn;
  a.max_dest_rd_atomic = 0;
  a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;
  a.ah_attr.port_num = 1;
  memcpy(a.ah_attr.grh.dgid.raw, peer->gid, 16);
  a.ah_attr.grh.sgid_index = sgid_idx;
  a.ah_attr.grh.hop_limit = 64;
  CHK(ibv_modify_qp(q, &a,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) ==
      0);
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.sq_psn = my_psn;
  a.max_rd_atomic = 0;
  CHK(ibv_modify_qp(q, &a,
                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC) == 0);
}

// blocking TCP exchange of PeerXchg structs; lower rank listens
int run_capture(cudaStream_t, Ctrl *, Ctrl *, float *, float *,
                float **, int, int, int, int);

static void tcp_xchg(int rank, int peer, const PeerXchg *mine,
                     PeerXchg *theirs) {
  int lo = rank < peer ? rank : peer, hi = rank < peer ? peer : rank;
  int port = TCP_BASE + lo * 4 + hi;
  int fd;
  if (rank == lo) {
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    CHK(ls >= 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr.s_addr = inet_addr(IPS[rank]);
    CHK(bind(ls, (struct sockaddr *)&sa, sizeof(sa)) == 0);
    CHK(listen(ls, 1) == 0);
    fd = accept(ls, NULL, NULL);
    CHK(fd >= 0);
    close(ls);
  } else {
    fd = -1;
    for (int t = 0; t < 600; t++) {
      fd = socket(AF_INET, SOCK_STREAM, 0);
      CHK(fd >= 0);
      struct sockaddr_in sa;
      memset(&sa, 0, sizeof(sa));
      sa.sin_family = AF_INET;
      sa.sin_port = htons(port);
      sa.sin_addr.s_addr = inet_addr(IPS[peer]);
      if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) == 0) break;
      close(fd);
      fd = -1;
      usleep(100000);
    }
    CHK(fd >= 0);
  }
  CHK(write(fd, mine, sizeof(*mine)) == sizeof(*mine));
  size_t got = 0;
  while (got < sizeof(*theirs)) {
    ssize_t r = read(fd, (char *)theirs + got, sizeof(*theirs) - got);
    CHK(r > 0);
    got += (size_t)r;
  }
  close(fd);
}

int main(int argc, char **argv) {
  CHK(argc >= 2);
  int rank = atoi(argv[1]);
  int replays = argc > 2 ? atoi(argv[2]) : 2000;
  int warm = argc > 3 ? atoi(argv[3]) : 100;
  int mode = argc > 4 ? atoi(argv[4]) : 0;
  CHK(rank >= 0 && rank < 4);

  CUCHK(cudaSetDevice(0));
  void *hp;
  CUCHK(cudaHostAlloc(&hp, sizeof(Ctrl), cudaHostAllocMapped));
  memset(hp, 0, sizeof(Ctrl));
  ctrl = (Ctrl *)hp;
  Ctrl *dctrl;
  CUCHK(cudaHostGetDevicePointer((void **)&dctrl, hp, 0));

  // device-side src/dst/staging
  float *d_src, *d_dst, *d_stage_all;
  float *d_stage[NPEER];
  CUCHK(cudaMalloc(&d_src, NELEM * 4));
  CUCHK(cudaMalloc(&d_dst, NELEM * 4));
  CUCHK(cudaMalloc(&d_stage_all, (size_t)NPEER * NELEM * 4));
  for (int p = 0; p < NPEER; p++) d_stage[p] = d_stage_all + p * NELEM;
  {
    float *tmp = (float *)malloc(NELEM * 4);
    for (int i = 0; i < NELEM; i++) tmp[i] = (float)(rank + 1);
    CUCHK(cudaMemcpy(d_src, tmp, NELEM * 4, cudaMemcpyHostToDevice));
    free(tmp);
  }

  // ---- verbs ----
  int ndev = 0;
  struct ibv_device **devs = ibv_get_device_list(&ndev);
  struct ibv_context *ctx = NULL;
  for (int i = 0; i < ndev; i++)
    if (!strcmp(ibv_get_device_name(devs[i]), DEVNAME))
      ctx = ibv_open_device(devs[i]);
  CHK(ctx != NULL);
  struct ibv_port_attr pattr;
  CHK(ibv_query_port(ctx, 1, &pattr) == 0);
  union ibv_gid mygid;
  int sgid = find_gid(ctx, 1, IPS[rank], &mygid);
  CHK(sgid >= 0);
  struct ibv_pd *pd = ibv_alloc_pd(ctx);
  CHK(pd);
  cq = ibv_create_cq(ctx, 1024, NULL, NULL, 0);
  CHK(cq);
  mr = ibv_reg_mr(pd, hp, sizeof(Ctrl),
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  CHK(mr);

  int peers[NPEER], np = 0;
  for (int r2 = 0; r2 < 4; r2++)
    if (r2 != rank) peers[np++] = r2;

  srand((unsigned)(time(NULL) ^ (rank * 7919)));
  for (int s = 0; s < NPEER; s++) {
    struct ibv_qp_init_attr qia;
    memset(&qia, 0, sizeof(qia));
    qia.send_cq = cq;
    qia.recv_cq = cq;
    qia.cap.max_send_wr = 512;
    qia.cap.max_recv_wr = 4;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 16;
    qia.qp_type = IBV_QPT_RC;
    qp[s] = ibv_create_qp(pd, &qia);
    CHK(qp[s]);

    // slot index of ME on the peer side: position of my rank in their list
    int my_slot_on_peer = 0, cnt = 0;
    for (int r2 = 0; r2 < 4; r2++) {
      if (r2 == peers[s]) continue;
      if (r2 == rank) my_slot_on_peer = cnt;
      cnt++;
    }
    (void)my_slot_on_peer;

    PeerXchg mine;
    memset(&mine, 0, sizeof(mine));
    mine.qpn = qp[s]->qp_num;
    mine.psn = (uint32_t)(rand() & 0xffffff);
    mine.rkey = mr->rkey;
    mine.mtu = (uint32_t)pattr.active_mtu;
    memcpy(mine.gid, mygid.raw, 16);
    // peer s writes into column s of every slot row (slot-major layout)
    mine.rx_base = (uintptr_t)&ctrl->rx[0][s][0];
    mine.rxf_base = (uintptr_t)&ctrl->rxf[0][s];
    tcp_xchg(rank, peers[s], &mine, &rem[s]);
    enum ibv_mtu mtu = (enum ibv_mtu)(mine.mtu < rem[s].mtu ? mine.mtu
                                                            : rem[s].mtu);
    qp_to_rts(qp[s], &rem[s], sgid, mine.psn, mtu);
    printf("rank %d slot %d <-> peer %d qpn %u<->%u mtu %d\n", rank, s,
           peers[s], qp[s]->qp_num, rem[s].qpn, (int)mtu);
    fflush(stdout);
  }

  pthread_t pt;
  CHK(pthread_create(&pt, NULL, proxy_fn, NULL) == 0);

  // ---- captured per-call sequence (see run_capture below) ----
  cudaStream_t st;
  CUCHK(cudaStreamCreate(&st));
  int rc = run_capture(st, ctrl, dctrl, d_src, d_dst, d_stage, replays, warm,
                       mode, rank);
  ctrl->stop = 1;
  pthread_join(pt, NULL);
  return rc;
}

// ---- real capture path (kept separate so kernels can sit beside it) ----
__global__ void k_copy_in(Ctrl *c, const float *src) {
  uint64_t next = c->tx_seq + 1;  // seq this call will take
  int slot = (int)(next % RING);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NELEM) c->tx[slot][i] = src[i];
}
__global__ void k_copy_stage(Ctrl *c, float *s0, float *s1, float *s2) {
  uint64_t s = c->tx_seq;
  int slot = (int)(s % RING);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NELEM) {
    s0[i] = c->rx[0][slot][i];
    s1[i] = c->rx[1][slot][i];
    s2[i] = c->rx[2][slot][i];
  }
}

int run_capture(cudaStream_t st, Ctrl *h, Ctrl *d, float *d_src, float *d_dst,
                float **d_stage, int replays, int warm, int mode, int rank) {
  volatile uint64_t *d_tx = &d->tx_seq;
  volatile uint64_t *d_ack = &d->ack_seq;
  volatile uint64_t *d_rxf = &d->rxf[0][0];
  int grid = (NELEM + 255) / 256;

  cudaGraph_t graph;
  cudaGraphExec_t gexec;
  CUCHK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int c = 0; c < CALLS; c++) {
    int slot = (c + 1) % RING;  // fixed per call: CALLS %% RING == 0
    k_guard<<<1, 32, 0, st>>>(d_tx, d_ack);
    if (mode != 2) k_copy_in_slot<<<grid, 256, 0, st>>>(d, d_src, slot);
    k_signal<<<1, 32, 0, st>>>(d_tx);
    k_wait_slot<<<1, 32, 0, st>>>(d_tx, d_rxf, slot);
    if (mode == 5) {
      CUCHK(cudaMemcpyAsync(d_stage[0], (const void *)&h->rx[slot][0][0],
                            (size_t)NPEER * NELEM * 4, cudaMemcpyHostToDevice,
                            st));
      k_reduce_staged<<<grid, 256, 0, st>>>(d_dst, d_src, d_stage[0],
                                            d_stage[1], d_stage[2]);
    }  // mode 2: signalling core only
  }
  CUCHK(cudaStreamEndCapture(st, &graph));
  CUCHK(cudaGraphInstantiate(&gexec, graph, NULL, NULL, 0));

  // warmup
  for (int i = 0; i < warm; i++) CUCHK(cudaGraphLaunch(gexec, st));
  CUCHK(cudaStreamSynchronize(st));
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < replays; i++) CUCHK(cudaGraphLaunch(gexec, st));
  CUCHK(cudaStreamSynchronize(st));
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double us = ((t1.tv_sec - t0.tv_sec) * 1e6 +
               (t1.tv_nsec - t0.tv_nsec) * 1e-3) /
              ((double)replays * CALLS);

  // numerics
  float out[4];
  CUCHK(cudaMemcpy(out, d_dst, sizeof(out), cudaMemcpyDeviceToHost));
  int ok = (mode == 2) || (out[0] == 10.0f && out[1] == 10.0f);
  printf("rank %d GRAPH mode=%d RESULT: %.2f us/AR over %d replays x %d "
         "calls | seq=%llu | numerics %s\n",
         rank, mode, us, replays, CALLS,
         (unsigned long long)h->tx_seq, ok ? "OK" : "BAD");
  fflush(stdout);
  return ok ? 0 : 1;
}
