// Regression producer for a dynamically computed NVFP4 scale. It permits a
// PDL dependent to launch before publishing the scale, exactly the contract
// that requires the consumer to wait before reading any dependent data.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

__global__ void publish_scale(float* scale, float value, unsigned long long cycles) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const auto start = clock64();
  while (clock64() - start < cycles) {}
  if (threadIdx.x == 0) *scale = value;
}

void publish(torch::Tensor scale, double value, int64_t cycles) {
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat32 && scale.numel() == 1);
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(32);
  config.stream = c10::cuda::getCurrentCUDAStream();
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, publish_scale, scale.data_ptr<float>(),
                                  static_cast<float>(value), static_cast<unsigned long long>(cycles)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("publish", &publish); }
