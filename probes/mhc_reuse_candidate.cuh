// Experimental FP32 MHC: read each fn element once for the entire C=1
// token group. A separate token-tail launch removes all arrival counters.
// Insert after mk_mhc_p34_compute in the same-source probe extension.
template <int TOK, int CHUNK>
__global__ void mk_mhc_reuse_p1(const MKMhcArgs a) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  constexpr int CHUNKS = HIDDEN / CHUNK;
  __shared__ float residual[TOK][HC][CHUNK];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;
  // Each residual output is computed once, with the original FP32 order
  // and original BF16 rounding. The projection consumes its FP32 value.
  for (int i = tid; i < HC * CHUNK; i += MK_THREADS) {
    const int j = i / CHUNK, h = c * CHUNK + i % CHUNK;
#pragma unroll
    for (int t = 0; t < TOK; ++t) {
      const float x = __bfloat162float(a.x_in[t * HIDDEN + h]);
      float v = a.post_mix_in[t * HC + j] * x;
#pragma unroll
      for (int k = 0; k < HC; ++k)
        v += a.comb_mix_in[t * HC * HC + k * HC + j] *
             __bfloat162float(a.residual_in[(size_t)t * HC * HIDDEN + k * HIDDEN + h]);
      residual[t][j][i % CHUNK] = v;
      a.residual_out[(size_t)t * HC * HIDDEN + j * HIDDEN + h] = __float2bfloat16(v);
    }
  }
  __syncthreads();
  const int lane = tid & 31, l8 = lane & 7;
  const int m = (tid >> 5) + 8 * (lane >> 3);
  float sums[TOK] = {};
  if (m <= NOUT) {
#pragma unroll
    for (int i = 0; i < CHUNK / 8; ++i) {
      const int h = l8 + i * 8;
      float fn[HC] = {};
      if (m < NOUT) {
#pragma unroll
        for (int j = 0; j < HC; ++j)
          fn[j] = a.fn[(size_t)m * HC * HIDDEN + j * HIDDEN + c * CHUNK + h];
      }
#pragma unroll
      for (int t = 0; t < TOK; ++t) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          const float r = residual[t][j][h];
          v += (m < NOUT ? fn[j] : r) * r;
        }
        sums[t] += v;
      }
    }
  }
#pragma unroll
  for (int t = 0; t < TOK; ++t) {
#pragma unroll
    for (int off = 4; off; off >>= 1)
      sums[t] += __shfl_xor_sync(0xffffffffu, sums[t], off);
    if (l8 == 0 && m <= NOUT) {
      if (m < NOUT) a.yp[((size_t)c * MAX_TOK + t) * NOUT + m] = sums[t];
      else a.rp[c * MAX_TOK + t] = sums[t];
    }
  }
}

template <int CHUNKS>
__global__ void mk_mhc_reuse_tail(const MKMhcArgs a) {
  // The ordinary stream dependency waits for every producer CTA; no
  // persistent-grid assumptions, global counters, or spin waiting.
  __shared__ float pmix[HC];
  const int t = blockIdx.x, warp = threadIdx.x >> 5;
  MhcTailRegs r;
  if (warp != 0) mk_mhc_p34_load(a, t, r);
  if (warp == 0) {
    mk_mhc_p2_token<CHUNKS>(a, t, pmix);
    mk_mhc_p34_load(a, t, r);
  }
  __syncthreads();
  mk_mhc_p34_compute(a, t, pmix, r);
}
