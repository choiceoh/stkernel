// Delayed input producer for testing PDL read ordering without RDMA.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void ar_consumer_delay(const __nv_bfloat16* src,
                                 __nv_bfloat16* dst, int n,
                                 unsigned long long cycles, bool early) {
  if (early) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
  }
  const auto start = clock64();
  if (threadIdx.x == 0)
    while (clock64() - start < cycles) __nanosleep(64);
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) dst[i] = src[i];
}

void produce(torch::Tensor src, torch::Tensor dst, int64_t cycles, bool early) {
  TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.is_contiguous() && dst.is_contiguous());
  TORCH_CHECK(src.scalar_type() == torch::kBFloat16 && dst.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(src.sizes() == dst.sizes() && src.numel() <= 131072 && cycles >= 0);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(48); cfg.blockDim = dim3(256);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr; cfg.numAttrs = early ? 1 : 0;
  const auto err = cudaLaunchKernelEx(&cfg, ar_consumer_delay,
      (const __nv_bfloat16*)src.data_ptr(), (__nv_bfloat16*)dst.data_ptr(),
      (int)src.numel(), (unsigned long long)cycles, early);
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("produce", &produce); }
