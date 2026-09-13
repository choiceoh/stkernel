// One page-aligned pinned allocation, two owned tensor aliases on GB10 UMA.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <memory>

std::vector<at::Tensor> mapped_pair(int64_t bytes) {
  TORCH_CHECK(bytes > 0 && bytes % 4096 == 0, "mapped staging requires positive page-aligned bytes");
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp p{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&p, device));
  TORCH_CHECK(p.major == 12 && p.minor == 1 && p.multiProcessorCount == 48 &&
              p.unifiedAddressing && p.canMapHostMemory,
              "mapped staging requires GB10 SM121 with host mapping and UVA");
  void *host = nullptr, *mapped = nullptr;
  C10_CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
  auto owner = std::shared_ptr<void>(host, [](void* ptr) { cudaFreeHost(ptr); });
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&mapped, host, 0));
  TORCH_CHECK(reinterpret_cast<uintptr_t>(host) % 4096 == 0, "host staging is not page aligned");
  // Either alias can outlive the other. Their deleters share the allocation;
  // the GPU alias is not an independently allocated CUDA storage block.
  auto cpu = at::from_blob(host, {bytes}, [owner](void*) {},
                           at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  auto gpu = at::from_blob(mapped, {bytes}, [owner](void*) {},
                           at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, device)));
  return {cpu, gpu};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mapped_pair", &mapped_pair, "Owned CPU and GPU aliases of GB10 pinned staging");
}
