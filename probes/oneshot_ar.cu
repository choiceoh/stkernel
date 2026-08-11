// One-shot cross-node AllReduce prototype for 4x GB10 (Stage-1 gate).
// Per round: GPU writes 48KB payload + seq flag into pinned unified memory;
// a core-pinned host proxy busy-polls the flag and RDMA-writes payload+flag
// to all 3 peers; the GPU polls the 3 inbound flags and reduces locally.
// Measures steady-state end-to-end AllReduce latency (GPU-visible to
// GPU-visible). Gate: < 25us vs NCCL's ~65us.
//
// Build: nvcc -O2 -arch=native oneshot_ar.cu -o oneshot_ar -lrdmacm -libverbs
// Run:   ./oneshot_ar <rank 0-3> [rounds=5000] [warmup=500]
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NELEM 12288          // 48KB of float
#define NPEER 3
#define MAXROUNDS 20000
#define PORT_BASE 22300      // pair (lo,hi): port = PORT_BASE + lo*4 + hi

struct PeerInfo {
  uint64_t rx_addr;    // where I should write payload on the peer
  uint64_t flag_addr;  // where I should write the flag on the peer
  uint32_t rkey;
  uint32_t pad;
};

static const char *IPS[4] = {"10.10.10.2", "10.10.10.3", "10.10.10.1",
                             "10.10.10.4"};

#define CHK(x)                                                              \
  do {                                                                      \
    if (!(x)) {                                                             \
      fprintf(stderr, "FAIL %s:%d %s errno=%d(%s)\n", __FILE__, __LINE__,   \
              #x, errno, strerror(errno));                                  \
      exit(1);                                                              \
    }                                                                       \
  } while (0)
#define CUCHK(x)                                                            \
  do {                                                                      \
    cudaError_t e_ = (x);                                                   \
    if (e_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x,   \
              cudaGetErrorString(e_));                                      \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

struct Shm {
  volatile uint64_t tx_seq;        // GPU -> proxy: round ready
  volatile uint64_t rx_flag[NPEER];  // peers' RDMA flag writes land here
  volatile uint64_t stop;
  uint64_t flag_src[NPEER];        // staging for outgoing flag writes
  struct PeerInfo xmine[NPEER];    // in-MR staging for info exchange
  struct PeerInfo xrbuf[NPEER];
  uint64_t pad[8];
  float tx[NELEM];
  float rx[NPEER][NELEM];
  float result[NELEM];
  long long tick[MAXROUNDS];       // per-round clock64 delta
};

static struct rdma_cm_id *conn[NPEER];  // slot s = peer list order
static struct PeerInfo remote_info[NPEER];
static struct ibv_mr *mrc[NPEER];  // per-connection MR (per-id default pd)
static Shm *shm;

// ---------------- GPU side ----------------
__global__ void ar_kernel(Shm *s, int rank, int rounds, int sigonly) {
  const int tid = threadIdx.x;
  const int nthr = blockDim.x;
  for (int r = 1; r <= rounds; r++) {
    if (!sigonly)
      for (int i = tid; i < NELEM; i += nthr) s->tx[i] = (float)(rank + 1);
    __syncthreads();
    __threadfence_system();
    long long t0;
    if (tid == 0) {
      t0 = clock64();
      s->tx_seq = (uint64_t)r;
      __threadfence_system();
      while (s->rx_flag[0] < (uint64_t)r || s->rx_flag[1] < (uint64_t)r ||
             s->rx_flag[2] < (uint64_t)r) {
      }
    }
    __syncthreads();
    __threadfence_system();
    if (!sigonly)
      for (int i = tid; i < NELEM; i += nthr)
        s->result[i] = s->tx[i] + s->rx[0][i] + s->rx[1][i] + s->rx[2][i];
    __syncthreads();
    if (tid == 0 && r < MAXROUNDS) s->tick[r] = clock64() - t0;
  }
  if (tid == 0) s->stop = 1;
}

// ---------------- proxy ----------------
static void drain_cq(struct ibv_cq *cq) {
  struct ibv_wc wc[8];
  int n;
  while ((n = ibv_poll_cq(cq, 8, wc)) > 0) {
    for (int i = 0; i < n; i++)
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr, "WC error %d (%s)\n", wc[i].status,
                ibv_wc_status_str(wc[i].status));
        exit(1);
      }
  }
}

static void *proxy_fn(void *arg) {
  (void)arg;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(18, &set);  // dedicated big core
  sched_setaffinity(0, sizeof(set), &set);

  uint64_t last = 0;
  while (!shm->stop) {
    uint64_t s = shm->tx_seq;
    if (s > last) {
      last = s;
      for (int p = 0; p < NPEER; p++) {
        shm->flag_src[p] = s;
        struct ibv_sge sge[2];
        struct ibv_send_wr wr[2], *bad;
        memset(wr, 0, sizeof(wr));
        sge[0].addr = (uintptr_t)shm->tx;
        sge[0].length = NELEM * sizeof(float);
        sge[0].lkey = mrc[p]->lkey;
        wr[0].wr_id = 1;
        wr[0].sg_list = &sge[0];
        wr[0].num_sge = 1;
        wr[0].opcode = IBV_WR_RDMA_WRITE;
        wr[0].send_flags = IBV_SEND_SIGNALED;
        wr[0].wr.rdma.remote_addr = remote_info[p].rx_addr;
        wr[0].wr.rdma.rkey = remote_info[p].rkey;
        wr[0].next = &wr[1];
        sge[1].addr = (uintptr_t)&shm->flag_src[p];
        sge[1].length = 8;
        sge[1].lkey = mrc[p]->lkey;
        wr[1].wr_id = 2;
        wr[1].sg_list = &sge[1];
        wr[1].num_sge = 1;
        wr[1].opcode = IBV_WR_RDMA_WRITE;
        wr[1].send_flags = IBV_SEND_SIGNALED;
        wr[1].wr.rdma.remote_addr = remote_info[p].flag_addr;
        wr[1].wr.rdma.rkey = remote_info[p].rkey;
        CHK(ibv_post_send(conn[p]->qp, wr, &bad) == 0);
      }
      for (int p = 0; p < NPEER; p++) drain_cq(conn[p]->send_cq);
    }
  }
  return NULL;
}

// ---------------- connection setup ----------------
static struct rdma_cm_id *ep_new(const char *src, const char *dst, int port,
                                 struct ibv_pd *pd, int passive) {
  struct rdma_addrinfo hints, *res;
  memset(&hints, 0, sizeof(hints));
  hints.ai_port_space = RDMA_PS_TCP;
  if (passive) hints.ai_flags = RAI_PASSIVE;
  char ps[16];
  snprintf(ps, sizeof(ps), "%d", port);
  CHK(rdma_getaddrinfo(passive ? src : dst, ps, &hints, &res) == 0);

  struct ibv_qp_init_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.cap.max_send_wr = 256;
  attr.cap.max_recv_wr = 16;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  attr.cap.max_inline_data = 16;
  attr.sq_sig_all = 0;

  struct rdma_cm_id *id;
  CHK(rdma_create_ep(&id, res, pd, passive ? NULL : &attr) == 0);
  rdma_freeaddrinfo(res);
  return id;
}

static void xchg_info(struct rdma_cm_id *id, int slot) {
  // my info for this peer: where they should write into MY shm.
  // Buffers must live INSIDE the registered MR (rdma_verbs asserts).
  shm->xmine[slot].rx_addr = (uintptr_t)shm->rx[slot];
  shm->xmine[slot].flag_addr = (uintptr_t)&shm->rx_flag[slot];
  shm->xmine[slot].rkey = mrc[slot]->rkey;
  CHK(rdma_post_send(id, NULL, &shm->xmine[slot], sizeof(PeerInfo),
                     mrc[slot],
                     IBV_SEND_SIGNALED) == 0);
  struct ibv_wc wc;
  CHK(rdma_get_send_comp(id, &wc) >= 0 && wc.status == IBV_WC_SUCCESS);
  CHK(rdma_get_recv_comp(id, &wc) >= 0 && wc.status == IBV_WC_SUCCESS);
}

int main(int argc, char **argv) {
  CHK(argc >= 2);
  int rank = atoi(argv[1]);
  int rounds = argc > 2 ? atoi(argv[2]) : 5000;
  int warmup = argc > 3 ? atoi(argv[3]) : 500;
  int sigonly = argc > 4 ? atoi(argv[4]) : 0;
  CHK(rank >= 0 && rank < 4 && rounds + warmup < MAXROUNDS);
  int total = rounds + warmup;

  CUCHK(cudaSetDevice(0));
  void *host;
  CUCHK(cudaHostAlloc(&host, sizeof(Shm), cudaHostAllocMapped));
  memset(host, 0, sizeof(Shm));
  shm = (Shm *)host;
  Shm *dshm;
  CUCHK(cudaHostGetDevicePointer((void **)&dshm, host, 0));

  // peer list in global-rank order, slot index = position in this list
  int peers[NPEER], np = 0;
  for (int r2 = 0; r2 < 4; r2++)
    if (r2 != rank) peers[np++] = r2;

  for (int s = 0; s < NPEER; s++) {
    int peer = peers[s];
    int lo = rank < peer ? rank : peer, hi = rank < peer ? peer : rank;
    int port = PORT_BASE + lo * 4 + hi;
    if (rank < peer) {  // I am the listener for this pair
      struct rdma_cm_id *lid = ep_new(IPS[rank], NULL, port, NULL, 1);
      CHK(rdma_listen(lid, 1) == 0);
      struct rdma_cm_id *cid;
      CHK(rdma_get_request(lid, &cid) == 0);
      if (!cid->qp) {
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.cap.max_send_wr = 256;
        attr.cap.max_recv_wr = 16;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.qp_type = IBV_QPT_RC;
        CHK(rdma_create_qp(cid, NULL, &attr) == 0);
      }
      mrc[s] = ibv_reg_mr(cid->qp->pd, host, sizeof(Shm),
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      CHK(mrc[s] != NULL);
      CHK(rdma_post_recv(cid, NULL, &shm->xrbuf[s], sizeof(PeerInfo),
                         mrc[s]) == 0);
      CHK(rdma_accept(cid, NULL) == 0);
      conn[s] = cid;
    } else {  // I connect
      struct rdma_cm_id *cid = NULL;
      for (int tries = 0; tries < 300; tries++) {
        cid = ep_new(NULL, IPS[peer], port, NULL, 0);
        CHK(cid->qp != NULL);
        mrc[s] = ibv_reg_mr(cid->qp->pd, host, sizeof(Shm),
                            IBV_ACCESS_LOCAL_WRITE |
                                IBV_ACCESS_REMOTE_WRITE);
        CHK(mrc[s] != NULL);
        CHK(rdma_post_recv(cid, NULL, &shm->xrbuf[s], sizeof(PeerInfo),
                           mrc[s]) == 0);
        if (rdma_connect(cid, NULL) == 0) break;
        ibv_dereg_mr(mrc[s]);
        rdma_destroy_ep(cid);
        cid = NULL;
        usleep(200000);
      }
      CHK(cid != NULL);
      conn[s] = cid;
    }
    xchg_info(conn[s], s);
    remote_info[s] = shm->xrbuf[s];
    printf("rank %d slot %d <-> peer %d connected (rkey %u)\n", rank, s,
           peers[s], remote_info[s].rkey);
    fflush(stdout);
  }

  pthread_t pt;
  CHK(pthread_create(&pt, NULL, proxy_fn, NULL) == 0);

  ar_kernel<<<1, 256>>>(dshm, rank, total, sigonly);
  CUCHK(cudaGetLastError());

  // wall clock over the measured window via tx_seq polling
  while (shm->tx_seq < (uint64_t)warmup) usleep(50);
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  while (shm->tx_seq < (uint64_t)total) usleep(50);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  CUCHK(cudaDeviceSynchronize());
  double wall_us = (t1.tv_sec - t0.tv_sec) * 1e6 +
                   (t1.tv_nsec - t0.tv_nsec) * 1e-3;

  // numeric check: 1+2+3+4 = 10
  CHK(sigonly || (shm->result[0] == 10.0f && shm->result[NELEM - 1] == 10.0f));

  // clock64 percentiles (nominal 2.0 GHz, informational)
  long long *ts = (long long *)shm->tick + warmup;
  int n = total - warmup;
  for (int i = 0; i < n; i++)  // insertion-lite: full sort
    for (int j = i + 1; j < n; j++)
      if (ts[j] < ts[i]) {
        long long tmp = ts[i];
        ts[i] = ts[j];
        ts[j] = tmp;
      }
  double ghz = 2.0;
  printf("rank %d %s RESULT: wall %.2f us/round over %d rounds | clk64 "
         "p50 %.1f p90 %.1f p99 %.1f us (nominal %.1fGHz) | numerics OK\n",
         rank, sigonly ? "SIGONLY" : "FULL", wall_us / n, n, ts[n / 2] / ghz / 1e3,
         ts[(int)(n * 0.9)] / ghz / 1e3, ts[(int)(n * 0.99)] / ghz / 1e3,
         ghz);
  fflush(stdout);
  return 0;
}
