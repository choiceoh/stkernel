// What does an sm_121a tensor core actually issue, per format?
//
// The ledger recorded "fp4 gives no compute gain: e2m1 kind::f8f6f4 = 155.1
// TFLOP/s, the same as e4m3".  But the serving trace has the b12x NVFP4 MoE
// running at 226 TFLOP/s, which is 46 pct ABOVE that supposed ceiling -- so
// either the FLOP count is wrong (it is not: 9 expert passes x 50.3 MFLOP x
// 7370 tokens = 3.34 TFLOP per layer-chunk, matching the trace's 3.3) or the
// measured instruction was not the fast fp4 path.  226/155 = 1.458, close
// enough to 3/2 to be structural rather than kernel efficiency, so measure
// each candidate instruction directly.
//
//   nvcc -O3 -gencode arch=compute_121a,code=sm_121a gb10_mma_rates.cu -o r
//
// -gencode, not -arch: under -c, `-arch=sm_121a` silently produces a plain
// sm_121 target and the arch-specific instructions never make it in.
#include <cstdio>
#include <cuda_runtime.h>

#define ITERS 4096

__device__ __forceinline__ void mma_bf16_16x8x16(float *d, const unsigned *a,
                                                 const unsigned *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_e4m3_16x8x32(float *d, const unsigned *a,
                                                 const unsigned *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// fp4 through the SHARED f8f6f4 datapath: e2m1 operands, but the same k32
// shape and the same register footprint as e4m3 -- each nibble occupies a
// byte-wide lane.  This is what the ledger measured.
__device__ __forceinline__ void mma_e2m1_k32(float *d, const unsigned *a,
                                             const unsigned *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// fp4 PACKED: k64 in the same registers, two nibbles per byte.  If this
// assembles, fp4 is 2x fp8 and the ledger's "no gain" was measuring the
// unpacked form.
__device__ __forceinline__ void mma_e2m1_k64(float *d, const unsigned *a,
                                             const unsigned *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.f32.e2m1.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 2:4 structured sparsity on e4m3: half the k operands, a metadata word
// naming which two of every four survive.  The only 2x on this part that
// does not need a different number format.
__device__ __forceinline__ void mma_sp_e4m3(float *d, const unsigned *a,
                                            const unsigned *b, unsigned e) {
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4"
      ".f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(b[2]), "r"(b[3]), "r"(e));
}

template <int KIND>
__global__ void rate(float *sink, int reps) {
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  unsigned a[4], b[4];
  const unsigned t = threadIdx.x + 1;
  for (int i = 0; i < 4; ++i) {
    a[i] = 0x3c003c00u ^ (t * (i + 1));
    b[i] = 0x3c003c00u ^ (t * (i + 3));
  }
  const unsigned e = 0x44444444u;              // 2:4 metadata: keep lanes 0,1
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if (KIND == 0) mma_bf16_16x8x16(d, a, b);
      else if (KIND == 1) mma_e4m3_16x8x32(d, a, b);
      else if (KIND == 2) mma_e2m1_k32(d, a, b);
      else if (KIND == 3) mma_e2m1_k64(d, a, b);
      else                mma_sp_e4m3(d, a, b, e);
    }
  }
  if (threadIdx.x == 1024) sink[0] = d[0] + d[1] + d[2] + d[3];
}

template <int KIND>
void run(const char *name, double k_per_mma) {
  int blocks = 48 * 4, threads = 128;         // 4 blocks/SM, one warp group
  float *sink;
  cudaMalloc(&sink, 4);
  rate<KIND><<<blocks, threads>>>(sink, 64);  // warm
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("  %-26s  UNSUPPORTED (%s)\n", name, cudaGetErrorString(err));
    cudaFree(sink);
    return;
  }
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  rate<KIND><<<blocks, threads>>>(sink, ITERS);
  cudaEventRecord(t1);
  cudaDeviceSynchronize();
  float ms = 0.f;
  cudaEventElapsedTime(&ms, t0, t1);
  // 16x8xK MACs per mma per warp, 4 warps per block.
  double mmas = (double)blocks * (threads / 32) * ITERS * 8;
  double tflops = mmas * 16 * 8 * k_per_mma * 2 / (ms * 1e-3) / 1e12;
  printf("  %-26s  %8.1f TFLOP/s\n", name, tflops);
  cudaFree(sink);
}

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int clk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  printf("sm_%d%d  %d SMs  %.0f MHz nominal\n", p.major, p.minor,
         p.multiProcessorCount, clk / 1000.0);
  run<0>("bf16      m16n8k16", 16);
  run<1>("e4m3      m16n8k32", 32);
  run<2>("e2m1 k32  (f8f6f4)", 32);
  run<3>("e2m1 k64  (mxf4)  ", 64);
  run<4>("e4m3 2:4  m16n8k64", 64);
  return 0;
}
