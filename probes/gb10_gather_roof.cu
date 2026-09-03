// LPDDR gather ceiling on GB10: sequential vs scattered fixed-size chunks.
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
__global__ void seq_read(const uint4* __restrict__ p, size_t n4, float* out) {
  uint4 acc = make_uint4(0,0,0,0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if (out) out[0] = (float)(acc.x ^ acc.y ^ acc.z ^ acc.w);
}
// each warp pulls one CHUNK-byte run whose base is scattered
template <int CHUNK>
__global__ void gather(const uint8_t* __restrict__ base, const int* __restrict__ idx,
                       int nchunk, float* out) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int warps = (gridDim.x * blockDim.x) >> 5;
  uint4 acc = make_uint4(0,0,0,0);
  for (int c = warp; c < nchunk; c += warps) {
    const uint8_t* src = base + (size_t)idx[c] * CHUNK;
#pragma unroll
    for (int off = 0; off < CHUNK / 16; off += 32) {
      const int j = off + lane;
      if (j * 16 < CHUNK) { uint4 v = *(const uint4*)(src + (size_t)j * 16);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    }
  }
  if (out) out[0] = (float)(acc.x ^ acc.y ^ acc.z ^ acc.w);
}
int main() {
  const size_t BYTES = 1ull << 30;             // 1 GiB, far past L2
  uint8_t* buf; cudaMalloc(&buf, BYTES); cudaMemset(buf, 1, BYTES);
  float* out; cudaMalloc(&out, 4);
  const int B = 48 * 8, T = 256;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  auto run = [&](const char* nm, double bytes, auto&& f) {
    f(); cudaDeviceSynchronize();
    cudaEventRecord(a); for (int r = 0; r < 5; ++r) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("  %-34s %7.1f GB/s\n", nm, bytes * 5 / (ms * 1e-3) / 1e9);
  };
  run("sequential uint4 stream", (double)BYTES, [&]{ seq_read<<<B,T>>>((const uint4*)buf, BYTES/16, out); });
  for (int chunk : {256, 512, 1024, 2048, 4096}) {
    int n = (int)(BYTES / chunk);
    int* idx; cudaMalloc(&idx, sizeof(int) * n);
    int* h = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) h[i] = i;
    for (int i = n - 1; i > 0; --i) { int j = rand() % (i + 1); int t = h[i]; h[i] = h[j]; h[j] = t; }
    cudaMemcpy(idx, h, sizeof(int) * n, cudaMemcpyHostToDevice); free(h);
    char nm[64]; snprintf(nm, sizeof nm, "scattered %d B chunks", chunk);
    double bytes = (double)n * chunk;
    if (chunk == 256) run(nm, bytes, [&]{ gather<256><<<B,T>>>(buf, idx, n, out); });
    else if (chunk == 512) run(nm, bytes, [&]{ gather<512><<<B,T>>>(buf, idx, n, out); });
    else if (chunk == 1024) run(nm, bytes, [&]{ gather<1024><<<B,T>>>(buf, idx, n, out); });
    else if (chunk == 2048) run(nm, bytes, [&]{ gather<2048><<<B,T>>>(buf, idx, n, out); });
    else run(nm, bytes, [&]{ gather<4096><<<B,T>>>(buf, idx, n, out); });
    cudaFree(idx);
  }
  return 0;
}
