#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void mhc_post_tilelang_kernel(const float* __restrict__ a, __grid_constant__ const CUtensorMap b_desc, const float* __restrict__ c, const bfloat16_t* __restrict__ d, bfloat16_t* __restrict__ x, int num_tokens);
extern "C" __global__ void __launch_bounds__(128, 1) mhc_post_tilelang_kernel(const float* __restrict__ a, __grid_constant__ const CUtensorMap b_desc, const float* __restrict__ c, const bfloat16_t* __restrict__ d, bfloat16_t* __restrict__ x, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* d_shared = ((void*)((char*)buf_dyn_shmem + 4096));
  __shared__ __align__(16) uint64_t ready_mem[1];
  auto ready = reinterpret_cast<Barrier*>(ready_mem);
  float a_local[16];
  float c_local[4];
  float b_local[16];
  bfloat16_t b_shared_local_cast[4];
  bfloat16_t d_shared_local_cast_1[4];
  float d_local[4];
  float x_local[16];
  bfloat16_t x_local_cast_2[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(b_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    ready[0].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(ulonglong4*)(a_local + (i * 8)) = tl::load_global_256(&(*(ulonglong4*)(a + ((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)i) * (int64_t)8)))));
  }
  *(float4*)(c_local + 0) = *(float4*)(c + (((int64_t)((int)blockIdx.x)) * (int64_t)4));
  if (tl::tl_shuffle_elect<128>()) {
    ready[0].expect_transaction(4096);
    tl::tma_load(b_desc, ready[0], (&(((bfloat16_t*)b_shared)[0])), (((int)blockIdx.y) * 512), 0, ((int)blockIdx.x));
    tl::tma_load(b_desc, ready[0], (&(((bfloat16_t*)b_shared)[1024])), ((((int)blockIdx.y) * 512) + 256), 0, ((int)blockIdx.x));
    ready[0].arrive_and_expect_tx(1024);
    tl::tma_load((&(((bfloat16_t*)d_shared)[0])), (&(d[((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int64_t)((int)blockIdx.y)) * (int64_t)512))])), ready[0], 1024);
  }
  ready[0].wait(0);
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint2*)(b_shared_local_cast + 0) = *(uint2*)(((bfloat16_t*)b_shared) + ((((((int)threadIdx.x) >> 6) * 1024) + (i_1 * 256)) + ((((int)threadIdx.x) & 63) * 4)));
    float4 __1;
    uint2 v_ = *(uint2*)(b_shared_local_cast + 0);
    ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
    ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
    *(float4*)(b_local + (i_1 * 4)) = __1;
  }
  *(uint2*)(d_shared_local_cast_1 + 0) = *(uint2*)(((bfloat16_t*)d_shared) + (((int)threadIdx.x) * 4));
  float4 __2;
  uint2 v__1 = *(uint2*)(d_shared_local_cast_1 + 0);
  ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
  ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
  *(float4*)(d_local + 0) = __2;
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    x_local[i_2] = (c_local[(i_2 >> 2)] * d_local[(i_2 & 3)]);
    for (int i_hci = 0; i_hci < 4; ++i_hci) {
      x_local[i_2] = (x_local[i_2] + (a_local[((i_hci * 4) + (i_2 >> 2))] * b_local[((i_hci * 4) + (i_2 & 3))]));
    }
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    uint2 __3;
    float4 v__2 = *(float4*)(x_local + (i_3 * 4));
    (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
    *(uint2*)(x_local_cast_2 + 0) = __3;
    *(uint2*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + (((int64_t)i_3) * (int64_t)4096)) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4))) = *(uint2*)(x_local_cast_2 + 0);
  }
}
