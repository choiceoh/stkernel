// Probe-only compensated TF32 projection. FP32 residual computation and
// BF16 rounding are retained; the dot-product has a different error/order.
// PTX m16n8k8 fragment mapping: NVIDIA PTX ISA, matrix-fragments-for-mma-m16n8k8.
__device__ __forceinline__ uint32_t mhc_tf32(float v) {
  uint32_t r;
  asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
  return r;
}

__device__ __forceinline__ void mhc_mma_tf32(float (&acc)[4],
                                           const uint32_t (&a)[4],
                                           const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int TOK, int CHUNK>
__global__ void mk_mhc_tf32x3_p1(const MKMhcArgs a) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  __shared__ uint32_t hi[8][HC][CHUNK];
  __shared__ uint32_t lo[8][HC][CHUNK];
  __shared__ float partial[HC][8][32];
  __shared__ float sq[TOK][MK_THREADS];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x, j = warp / 2, out_base = (warp & 1) * 16;
  // One thread owns each hidden coordinate and computes all four streams.
  for (int h0 = tid; h0 < CHUNK; h0 += MK_THREADS) {
    const int h = c * CHUNK + h0;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      float s = 0.0f;
      float x = 0.0f, res[HC] = {};
      if (t < TOK) {
        x = __bfloat162float(a.x_in[t * HIDDEN + h]);
#pragma unroll
        for (int k = 0; k < HC; ++k)
          res[k] = __bfloat162float(a.residual_in[(size_t)t * HC * HIDDEN + k * HIDDEN + h]);
      }
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        float v = 0.0f;
        if (t < TOK) {
          v = a.post_mix_in[t * HC + k] * x;
#pragma unroll
          for (int i = 0; i < HC; ++i)
            v += a.comb_mix_in[t * HC * HC + i * HC + k] * res[i];
          a.residual_out[(size_t)t * HC * HIDDEN + k * HIDDEN + h] = __float2bfloat16(v);
          s += v * v;
        }
        const uint32_t vh = mhc_tf32(v);
        hi[t][k][h0] = vh;
        lo[t][k][h0] = mhc_tf32(v - __uint_as_float(vh));
      }
      if (t < TOK) sq[t][h0] = s;
    }
  }
  __syncthreads();
  const int g = lane >> 2, q = lane & 3;
  float acc[4] = {};
#pragma unroll
  for (int k = 0; k < CHUNK; k += 8) {
    uint32_t ah[4], al[4], bh[2], bl[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = out_base + g + ((i & 1) ? 8 : 0);
      const int col = k + q + ((i & 2) ? 4 : 0);
      const float f = row < NOUT ? a.fn[(size_t)row * HC * HIDDEN + j * HIDDEN + c * CHUNK + col] : 0.0f;
      ah[i] = mhc_tf32(f);
      al[i] = mhc_tf32(f - __uint_as_float(ah[i]));
    }
    bh[0] = hi[g][j][k + q]; bh[1] = hi[g][j][k + q + 4];
    bl[0] = lo[g][j][k + q]; bl[1] = lo[g][j][k + q + 4];
    // Low products first to avoid discarding both correction terms when
    // adding them to the much larger high product in a single instruction.
    mhc_mma_tf32(acc, ah, bl);
    mhc_mma_tf32(acc, al, bh);
    mhc_mma_tf32(acc, ah, bh);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = out_base + g + ((i & 2) ? 8 : 0);
    const int token = 2 * q + (i & 1);
    partial[j][token][row] = acc[i];
  }
  __syncthreads();
  for (int i = tid; i < TOK * NOUT; i += MK_THREADS) {
    const int t = i / NOUT, m = i % NOUT;
    float v = 0.0f;
#pragma unroll
    for (int k = 0; k < HC; ++k) v += partial[k][t][m];
    a.yp[((size_t)c * MAX_TOK + t) * NOUT + m] = v;
  }
  // One warp per token, using the same eight-lane reduction grouping as
  // the SIMT prototype; no inter-block atomic state.
  if (warp < TOK) {
    float s = 0.0f;
#pragma unroll
    for (int i = lane; i < CHUNK; i += 32) s += sq[warp][i];
#pragma unroll
    for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(~0u, s, off);
    if (lane == 0) a.rp[c * MAX_TOK + warp] = s;
  }
}
