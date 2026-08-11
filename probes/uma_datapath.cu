// Datapath gate for one-shot AR Stage-2: can the GPU read an RDMA-registered
// buffer at cache speed on GB10 UMA, eliminating the H2D staging copy?
//
// Compares 4 buffer types as (a) ibv_reg_mr registrable? (b) GPU sequential
// read bandwidth (large) (c) GPU read latency for a 48KB reduce (small):
//   A cudaHostAlloc(Mapped)      — current mode-1 path (uncached, 210us)
//   B cudaMallocManaged          — unified; GPU-native cached access?
//   C managed + MemAdvise(GPU) + Prefetch(GPU)
//   D malloc + cudaHostRegister(default)
//
// Build: nvcc -O2 -arch=sm_121a uma_datapath.cu -o umadp -libverbs
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define BIG (16 * 1024 * 1024)      // 64MB float: bandwidth
#define SMALL 12288                 // 48KB float: AR-sized latency
#define DEVNAME "rocep1s0f0"

#define CUCHK(x)                                                            \
  do {                                                                      \
    cudaError_t e_ = (x);                                                   \
    if (e_ != cudaSuccess) {                                                \
      printf("    CUDA FAIL %s: %s\n", #x, cudaGetErrorString(e_));         \
      return;                                                               \
    }                                                                       \
  } while (0)

__global__ void k_reduce(float *dst, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += a[i];
}
__global__ void k_reduce_small(float *dst, const float *a) {
  int i = threadIdx.x;
  for (int j = i; j < SMALL; j += blockDim.x) dst[j] = a[j] + 1.0f;
}

static struct ibv_pd *pd;

static void trial(const char *name, int type) {
  printf("[%s]\n", name);
  float *buf = NULL, *dst = NULL;
  size_t bytes = (size_t)BIG * sizeof(float);
  if (type == 0) {
    CUCHK(cudaHostAlloc((void **)&buf, bytes, cudaHostAllocMapped));
  } else if (type == 1) {
    CUCHK(cudaMallocManaged((void **)&buf, bytes));
  } else if (type == 2) {
    CUCHK(cudaMallocManaged((void **)&buf, bytes));
    int dev = 0;
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id = dev;
    cudaMemAdvise(buf, bytes, cudaMemAdviseSetPreferredLocation, loc);
    cudaMemAdvise(buf, bytes, cudaMemAdviseSetAccessedBy, loc);
    CUCHK(cudaMemPrefetchAsync(buf, bytes, loc, 0, 0));
    CUCHK(cudaDeviceSynchronize());
  } else {
    buf = (float *)malloc(bytes);
    CUCHK(cudaHostRegister(buf, bytes, cudaHostRegisterDefault));
  }
  CUCHK(cudaMalloc((void **)&dst, bytes));
  for (size_t i = 0; i < 64; i++) buf[i] = 1.0f;

  // (a) RDMA registration
  struct ibv_mr *mr =
      ibv_reg_mr(pd, buf, bytes,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  printf("    ibv_reg_mr: %s%s\n", mr ? "OK rkey=" : "FAILED",
         mr ? "" : "");
  if (mr) printf("    (rkey %u lkey %u)\n", mr->rkey, mr->lkey);

  // GPU device pointer for the read
  float *rdptr = buf;
  if (type == 0) CUCHK(cudaHostGetDevicePointer((void **)&rdptr, buf, 0));

  // (b) bandwidth: sum BIG floats into dst
  int grid = (BIG + 255) / 256;
  k_reduce<<<grid, 256>>>(dst, rdptr, BIG);
  CUCHK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int r = 0; r < 20; r++) k_reduce<<<grid, 256>>>(dst, rdptr, BIG);
  cudaEventRecord(e1);
  CUCHK(cudaEventSynchronize(e1));
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  double gbps = (double)bytes * 20 / (ms / 1e3) / 1e9;
  printf("    GPU read BW: %.1f GB/s (%.2f ms/64MB)\n", gbps, ms / 20);

  // (c) 48KB reduce latency (steady, 5000x)
  k_reduce_small<<<1, 256>>>(dst, rdptr);
  CUCHK(cudaDeviceSynchronize());
  cudaEventRecord(e0);
  for (int r = 0; r < 5000; r++) k_reduce_small<<<1, 256>>>(dst, rdptr);
  cudaEventRecord(e1);
  CUCHK(cudaEventSynchronize(e1));
  cudaEventElapsedTime(&ms, e0, e1);
  printf("    48KB reduce: %.2f us/call\n", ms * 1e3 / 5000);

  if (mr) ibv_dereg_mr(mr);
}

int main() {
  int n = 0;
  struct ibv_device **devs = ibv_get_device_list(&n);
  struct ibv_context *ctx = NULL;
  for (int i = 0; i < n; i++)
    if (!strcmp(ibv_get_device_name(devs[i]), DEVNAME))
      ctx = ibv_open_device(devs[i]);
  if (!ctx) {
    printf("no %s\n", DEVNAME);
    return 1;
  }
  pd = ibv_alloc_pd(ctx);
  trial("A cudaHostAlloc(Mapped) [current]", 0);
  trial("B cudaMallocManaged", 1);
  trial("C managed + Advise(GPU) + Prefetch", 2);
  trial("D malloc + cudaHostRegister", 3);
  return 0;
}
