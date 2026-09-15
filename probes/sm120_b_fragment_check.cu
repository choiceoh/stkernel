// Why the sm_120 fallbacks in engine/kernels/{mla,dense} are the arithmetic they replace.
//
// glm53_megakernel.cu and kernels.cu carry `#else` branches for __CUDA_ARCH__ < 1210,
// because sm_120 has neither `add.u8x4` nor ldmatrix's `.b8`/`.m16n16`. A fallback that
// compiles but lays its fragments out differently would not fail -- it would return
// plausible, wrong numbers, which in a check lane is worse than no lane at all. So the
// claim is measured here, on the card itself, against a CPU reference:
//
//   check 1  the mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 fragment layouts the
//            fallback assumes (A row-major m16k16, B col-major k16n8, C m16n8)
//   check 2  that the megakernel's own mla_e4m3x2_strided form -- which the fallback reuses
//            in place of ldmatrix.trans.b8 -- puts the tile's bytes exactly where check 1
//            says B's elements live, over a tile with a non-square row pitch
//
// Build and run on any sm_120 box; it needs no engine, no model and no fleet:
//
//   nvcc -O2 -gencode arch=compute_120,code=sm_120 -std=c++17 //        probes/sm120_b_fragment_check.cu -o /tmp/check && /tmp/check
//
// Recorded 2026-09-15 on an RTX 5050: check 1 128/128, check 2 256/256.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M 16
#define N 8
#define K 16

__device__ __forceinline__ void mma_m16n8k16_one(float& c0, float& c1, float& c2, float& c3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_one(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *(const uint32_t*)&v;
}

// The layouts under test. group = lane>>2, q = lane&3.
//   A  a0:(group, 2q+{0,1})  a1:(group+8, 2q+{0,1})  a2:(group, 2q+{0,1}+8)  a3:(group+8, ...+8)
//   B  b0:(k=2q+{0,1},   n=group)   b1:(k=2q+{0,1}+8, n=group)
//   C  c0,c1:(m=group,   n=2q+{0,1})  c2,c3:(m=group+8, n=2q+{0,1})
__global__ void run_one(const float* A, const float* B, float* C) {
  const int lane = threadIdx.x, group = lane >> 2, q = lane & 3;
  const uint32_t a0 = pack_one(A[group * K + 2 * q], A[group * K + 2 * q + 1]);
  const uint32_t a1 = pack_one(A[(group + 8) * K + 2 * q], A[(group + 8) * K + 2 * q + 1]);
  const uint32_t a2 = pack_one(A[group * K + 2 * q + 8], A[group * K + 2 * q + 9]);
  const uint32_t a3 = pack_one(A[(group + 8) * K + 2 * q + 8], A[(group + 8) * K + 2 * q + 9]);
  const uint32_t b0 = pack_one(B[(2 * q) * N + group], B[(2 * q + 1) * N + group]);
  const uint32_t b1 = pack_one(B[(2 * q + 8) * N + group], B[(2 * q + 9) * N + group]);
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  mma_m16n8k16_one(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
  C[group * N + 2 * q] = c0;
  C[group * N + 2 * q + 1] = c1;
  C[(group + 8) * N + 2 * q] = c2;
  C[(group + 8) * N + 2 * q + 1] = c3;
}

static int check_one() {
  float *A, *B, *C;
  cudaMallocManaged(&A, M * K * sizeof(float));
  cudaMallocManaged(&B, K * N * sizeof(float));
  cudaMallocManaged(&C, M * N * sizeof(float));
  srand(7);
  // bf16 has 8 mantissa bits; small integers are exact, so any mismatch is layout, not rounding.
  for (int i = 0; i < M * K; ++i) A[i] = (float)(rand() % 7 - 3);
  for (int i = 0; i < K * N; ++i) B[i] = (float)(rand() % 7 - 3);
  for (int i = 0; i < M * N; ++i) C[i] = 0.f;
  run_one<<<1, 32>>>(A, B, C);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("LAUNCH FAILED: %s\n", cudaGetErrorString(cudaGetLastError())); return 2; }

  int bad = 0; float worst = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float ref = 0;
      for (int k = 0; k < K; ++k) ref += A[m * K + k] * B[k * N + n];
      float d = fabsf(ref - C[m * N + n]);
      if (d > 1e-3f) { ++bad; if (d > worst) worst = d; }
    }
  printf(bad ? "LAYOUT WRONG: %d of %d elements differ (worst %.1f)\n"
             : "LAYOUT CONFIRMED: all %d elements match CPU matmul (worst %.1f)\n",
         bad ? bad : M * N, bad ? M * N : (int)worst, bad ? worst : 0.f);
  return bad ? 1 : 0;
}



#define M 16
#define N 8
#define K 16
#define RP (K + 3)          // a deliberately non-square row pitch: MLA_RP is MLA_D + 16
#define NCOL 32             // the tile is wider than one mma tile, as the ring is

__device__ __forceinline__ void mla_mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t mla_e4m3x2_strided(const uint8_t* p, int stride) {
  const __half2 h = __halves2half2(__nv_cvt_fp8_to_halfraw(p[0], __NV_E4M3),
                                   __nv_cvt_fp8_to_halfraw(p[stride], __NV_E4M3));
  const __nv_bfloat162 b = __float22bfloat162_rn(__half22float2(h));
  return *(const uint32_t*)&b;
}

__device__ __forceinline__ uint32_t pack_two(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *(const uint32_t*)&v;
}

// One warp, one nt step: acc = P(16x16 bf16) x C(16x8 e4m3, taken from column nt*8+g)
__global__ void run_two(const float* P, const uint8_t* tileIn, float* out, int nt) {
  __shared__ uint8_t tile[K * RP];
  for (int i = threadIdx.x; i < K * RP; i += 32) tile[i] = tileIn[i];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q4 = lane & 3;          // exactly as glm53_megakernel.cu names them
  const uint32_t a0 = pack_two(P[g * K + 2 * q4], P[g * K + 2 * q4 + 1]);
  const uint32_t a1 = pack_two(P[(g + 8) * K + 2 * q4], P[(g + 8) * K + 2 * q4 + 1]);
  const uint32_t a2 = pack_two(P[g * K + 2 * q4 + 8], P[g * K + 2 * q4 + 9]);
  const uint32_t a3 = pack_two(P[(g + 8) * K + 2 * q4 + 8], P[(g + 8) * K + 2 * q4 + 9]);

  const uint8_t* cb = tile + (size_t)(q4 * 2) * RP;   // the megakernel's strided form
  const int n = nt * 8 + g;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  mla_mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3,
               mla_e4m3x2_strided(cb + n, RP),
               mla_e4m3x2_strided(cb + 8 * RP + n, RP));
  out[g * N + 2 * q4] = c0;
  out[g * N + 2 * q4 + 1] = c1;
  out[(g + 8) * N + 2 * q4] = c2;
  out[(g + 8) * N + 2 * q4 + 1] = c3;
}

static int check_two() {
  float *P, *out; uint8_t* tile;
  cudaMallocManaged(&P, M * K * sizeof(float));
  cudaMallocManaged(&tile, K * RP);
  cudaMallocManaged(&out, M * N * sizeof(float));
  srand(11);
  for (int i = 0; i < M * K; ++i) P[i] = (float)(rand() % 5 - 2);
  // e4m3 codes that are small exact integers, so a mismatch is layout and not rounding
  static const uint8_t code[7] = {0x00, 0x38, 0x40, 0x44, 0xB8, 0xC0, 0xC4};  // 0,1,2,3,-1,-2,-3
  static const float val[7]    = {0.f,  1.f,  2.f,  3.f,  -1.f, -2.f, -3.f};
  float ref_tile[K][NCOL];
  for (int k = 0; k < K; ++k)
    for (int c = 0; c < RP; ++c) {
      int pick = rand() % 7;
      tile[k * RP + c] = code[pick];
      if (c < NCOL) ref_tile[k][c] = val[pick];
    }

  int bad = 0; float worst = 0;
  for (int nt = 0; nt < 2; ++nt) {                  // two column blocks: nt*8 + g
    for (int i = 0; i < M * N; ++i) out[i] = 0.f;
    run_two<<<1, 32>>>(P, tile, out, nt);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      printf("LAUNCH FAILED: %s\n", cudaGetErrorString(cudaGetLastError())); return 2;
    }
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        float ref = 0;
        for (int k = 0; k < K; ++k) ref += P[m * K + k] * ref_tile[k][nt * 8 + n];
        float d = fabsf(ref - out[m * N + n]);
        if (d > 1e-3f) { ++bad; if (d > worst) worst = d; }
      }
  }
  if (bad) printf("STRIDED FORM WRONG: %d of %d differ (worst %.1f)\n", bad, 2 * M * N, worst);
  else     printf("STRIDED FORM CONFIRMED: all %d elements match the CPU reference\n", 2 * M * N);
  return bad ? 1 : 0;
}

int main() {
  int rc = 0;
  puts("check 1: mma m16n8k16 fragment layouts");
  fputs("  ", stdout);
  rc |= check_one();
  puts("check 2: the strided B form the sm_120 fallback reuses");
  fputs("  ", stdout);
  rc |= check_two();
  return rc;
}
