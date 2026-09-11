// Bounded capability probe: <= 8 CTAs, 32 threads/CTA and 64 output bytes.
// 64 KiB dynamic SMEM forces different CTAs onto different GB10 SMs.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <initializer_list>

__global__ void read_cluster_neighbor(int* out) {
  auto cluster = cooperative_groups::this_cluster();
  __shared__ int value;
  if (threadIdx.x == 0) value = 100 + cluster.block_rank();
  cluster.sync();
  if (threadIdx.x == 0) {
    int next = (cluster.block_rank() + 1) % cluster.num_blocks();
    out[blockIdx.x] = *cluster.map_shared_rank(&value, next);
    unsigned smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[8 + blockIdx.x] = smid;
  }
  cluster.sync();
}

int main() {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(48);
  config.blockDim = dim3(32);
  config.dynamicSmemBytes = 64 * 1024;
  auto status = cudaFuncSetAttribute(read_cluster_neighbor,
      cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
  if (status != cudaSuccess) return 1;
  int maximum = 0;
  status = cudaOccupancyMaxPotentialClusterSize(
      &maximum, read_cluster_neighbor, &config);
  std::printf("max_potential_cluster_size=%d status=%s\n", maximum,
              cudaGetErrorString(status));
  if (status != cudaSuccess) return 1;
  int* out = nullptr;
  status = cudaMalloc(&out, 16 * sizeof(int));
  if (status != cudaSuccess) return 2;
  bool failed = false;
  for (int blocks : {1, 2, 4, 8}) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {static_cast<unsigned>(blocks), 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;
    config.gridDim = dim3(blocks);
    int active = 0;
    status = cudaOccupancyMaxActiveClusters(&active, read_cluster_neighbor, &config);
    std::printf("cluster_size=%d active_clusters=%d occupancy_status=%s\n",
                blocks, active, cudaGetErrorString(status));
    if (blocks > maximum || status != cudaSuccess) {
      std::printf("cluster_size=%d execution_skipped_unsupported_configuration\n", blocks);
      continue;
    }
    status = cudaLaunchKernelEx(&config, read_cluster_neighbor, out);
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    std::printf("cluster_size=%d execution_status=%s\n", blocks, cudaGetErrorString(status));
    if (status != cudaSuccess) { failed = true; break; }
    int host[16]{};
    status = cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost);
    bool correct = status == cudaSuccess;
    for (int i = 0; i < blocks; ++i) correct &= host[i] == 100 + (i + 1) % blocks;
    bool distinct = true;
    for (int i = 0; i < blocks; ++i)
      for (int j = 0; j < i; ++j) distinct &= host[8 + i] != host[8 + j];
    std::printf("cluster_size=%d neighbor_read=%s\n", blocks, correct ? "PASS" : "FAIL");
    std::printf("cluster_size=%d distinct_sms=%s smids=", blocks, distinct ? "PASS" : "FAIL");
    for (int i = 0; i < blocks; ++i) std::printf("%s%d", i ? "," : "", host[8 + i]);
    std::printf("\n");
    failed |= !correct || !distinct;
  }
  cudaFree(out);
  return failed ? 3 : 0;
}
