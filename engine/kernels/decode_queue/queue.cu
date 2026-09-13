#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

using Word = unsigned long long;
using SystemAtomic = cuda::atomic_ref<Word, cuda::thread_scope_system>;

static uint8_t* host_page(const at::Tensor& host) {
  TORCH_CHECK(host.device().is_cpu() && host.scalar_type() == at::kByte &&
              host.is_contiguous() && host.numel() >= 4096 &&
              reinterpret_cast<uintptr_t>(host.data_ptr()) % 64 == 0,
              "queue control must be the aligned owned host page");
  return host.data_ptr<uint8_t>();
}

static void check_device() {
  int device, native = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&native, cudaDevAttrHostNativeAtomicSupported, device));
  TORCH_CHECK(native, "shared decode queue requires native CPU/GPU system atomics");
}

__global__ void publish_rows(uint8_t* page, const int64_t* tokens, const int64_t* count,
                             const bool* done, const int64_t* accepted, const int64_t* before,
                             const int64_t* index, int rows, int t, int max_rows, int stride) {
  int j = *index;
  if (j < 0 || j >= 4) return;  // caller detects a missing publication after graph retirement
  auto* dst = reinterpret_cast<int64_t*>(page + 128) + j * max_rows * stride;
  for (int p = threadIdx.x; p < rows * (t + 4); p += blockDim.x) {
    int r = p / (t + 4), c = p % (t + 4);
    dst[r * stride + c] = c < t ? tokens[r * t + c] :
      c == t ? count[r] : c == t + 1 ? int64_t(done[r]) : c == t + 2 ? accepted[r] : before[r];
  }
  // Every writer orders its own mapped stores before the publishing thread's release.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0)
    SystemAtomic(*reinterpret_cast<Word*>(page)).store(j + 1, cuda::memory_order_release);
}

__global__ void interrupt_word(uint8_t* page, int64_t* into) {
  *into = SystemAtomic(*reinterpret_cast<Word*>(page + 64)).load(cuda::memory_order_acquire);
}

static void publish(at::Tensor page, at::Tensor tokens, at::Tensor count, at::Tensor done,
                    at::Tensor accepted, at::Tensor before, at::Tensor index, int max_rows, int stride) {
  TORCH_CHECK(page.is_cuda() && page.scalar_type() == at::kByte && page.is_contiguous() &&
              page.numel() >= 128 + 4 * max_rows * stride * 8 &&
              reinterpret_cast<uintptr_t>(page.data_ptr()) % 64 == 0 &&
              1 <= max_rows && max_rows <= 4 && stride % 8 == 0 &&
              tokens.dim() == 2 && 1 <= tokens.size(0) && tokens.size(0) <= max_rows &&
              1 <= tokens.size(1) && tokens.size(1) <= 8 && stride >= tokens.size(1) + 4,
              "invalid shared decode payload geometry");
  for (const auto& tensor : {tokens, count, accepted, before, index})
    TORCH_CHECK(tensor.device() == page.device() && tensor.scalar_type() == at::kLong && tensor.is_contiguous(),
                "shared decode inputs must be contiguous CUDA int64");
  TORCH_CHECK(done.device() == page.device() && done.scalar_type() == at::kBool && done.is_contiguous() &&
              count.numel() == tokens.size(0) && done.numel() == count.numel() &&
              accepted.numel() == count.numel() && before.numel() == count.numel() && index.numel() == 1,
              "shared decode row metadata differs");
  publish_rows<<<1, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
    page.data_ptr<uint8_t>(), tokens.data_ptr<int64_t>(), count.data_ptr<int64_t>(),
    done.data_ptr<bool>(), accepted.data_ptr<int64_t>(), before.data_ptr<int64_t>(),
    index.data_ptr<int64_t>(), tokens.size(0), tokens.size(1), max_rows, stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void read_interrupt(at::Tensor page, at::Tensor into) {
  TORCH_CHECK(page.is_cuda() && page.scalar_type() == at::kByte && page.is_contiguous() && page.numel() >= 128 &&
              reinterpret_cast<uintptr_t>(page.data_ptr()) % 64 == 0 &&
              into.device() == page.device() && into.scalar_type() == at::kLong && into.numel() == 1,
              "invalid shared interrupt word");
  interrupt_word<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(page.data_ptr<uint8_t>(), into.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("check_device", &check_device);
  m.def("reset", [](at::Tensor host) {
    auto* page = host_page(host);
    __atomic_store_n(reinterpret_cast<Word*>(page + 64), 0ULL, __ATOMIC_RELEASE);
    __atomic_store_n(reinterpret_cast<Word*>(page), 0ULL, __ATOMIC_RELEASE);
  });
  m.def("cancel", [](at::Tensor host) {
    __atomic_store_n(reinterpret_cast<Word*>(host_page(host) + 64), 1ULL, __ATOMIC_RELEASE);
  });
  m.def("published", [](at::Tensor host) {
    return __atomic_load_n(reinterpret_cast<Word*>(host_page(host)), __ATOMIC_ACQUIRE);
  });
  m.def("publish", &publish);
  m.def("read_interrupt", &read_interrupt);
}
