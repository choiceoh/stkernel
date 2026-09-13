// Bounded WHILE over an owned deterministic iteration graph, GB10 only.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

__global__ void reset_count(int64_t* count) { *count = 0; }
__global__ void next_iteration(cudaGraphConditionalHandle handle, int64_t* count,
                                const int64_t* stop, int limit) {
  int64_t done = ++(*count);
  cudaGraphSetConditional(handle, done < limit && *stop == 0);
}

static void validate_body(cudaGraph_t graph) {
  size_t size = 0;
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &size));
  TORCH_CHECK(size > 0, "bounded graph body is empty");
  std::vector<cudaGraphNode_t> nodes(size);
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &size));
  for (auto node : nodes) {
    cudaGraphNodeType type;
    C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child = nullptr;
      C10_CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(node, &child));
      validate_body(child);
    } else {
      TORCH_CHECK(type == cudaGraphNodeTypeKernel || type == cudaGraphNodeTypeEmpty ||
                  type == cudaGraphNodeTypeMemcpy || type == cudaGraphNodeTypeMemset,
                  "bounded graph rejects host callbacks, event/semaphore nodes, allocation and nested conditions");
    }
  }
  // CUDA instantiation checks device accessibility, mapped-copy operands,
  // forbidden dynamic/device launches and all conditional body constraints.
}

class BoundedGraph {
  // The cloned graph retains addresses, not Python allocation owners. Keep the
  // captured body/pool and all caller buffers alive until the last launch ends.
  pybind11::object owners_;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  cudaEvent_t done_ = nullptr;
  at::Tensor count_, stop_;
  int device_;
  cudaStream_t stream_ = nullptr;
  bool launched_ = false;
 public:
  BoundedGraph(uintptr_t body, at::Tensor count, at::Tensor stop, int limit, pybind11::object owners)
      : owners_(std::move(owners)), count_(count), stop_(stop),
        device_(count.is_cuda() ? count.get_device() : -1) {
    TORCH_CHECK(limit == 1 || limit == 2 || limit == 4, "bounded graph permits 1, 2 or 4 iterations");
    for (const auto& t : {count_, stop_})
      TORCH_CHECK(t.is_cuda() && t.device() == count_.device() && t.scalar_type() == at::kLong &&
                  t.is_contiguous() && t.dim() == 1 && t.numel() == 1,
                  "bounded graph needs same-device contiguous CUDA int64[1] controls");
    TORCH_CHECK(count_.data_ptr() != stop_.data_ptr(), "count and stop must not alias");
    TORCH_CHECK(body != 0, "bounded graph needs a retained CUDA graph");
    c10::cuda::CUDAGuard guard(count_.device());
    stream_ = c10::cuda::getCurrentCUDAStream(device_);
    cudaDeviceProp p{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&p, device_));
    TORCH_CHECK(p.major == 12 && p.minor == 1 && p.multiProcessorCount == 48, "bounded graph requires GB10");
    validate_body(reinterpret_cast<cudaGraph_t>(body));
    try {
      C10_CUDA_CHECK(cudaGraphCreate(&graph_, 0));
      cudaGraphConditionalHandle handle;
      C10_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph_, 1, cudaGraphCondAssignDefault));
      auto* counter = count_.data_ptr<int64_t>();
      auto* stopping = stop_.data_ptr<int64_t>();
      void* reset_args[] = {&counter};
      cudaKernelNodeParams reset{};
      reset.func = reinterpret_cast<void*>(reset_count);
      reset.gridDim = reset.blockDim = dim3(1);
      reset.kernelParams = reset_args;
      cudaGraphNode_t reset_node, condition, child, advance;
      C10_CUDA_CHECK(cudaGraphAddKernelNode(&reset_node, graph_, nullptr, 0, &reset));
      cudaGraphNodeParams params{};
      params.type = cudaGraphNodeTypeConditional;
      params.conditional.handle = handle;
      params.conditional.type = cudaGraphCondTypeWhile;
      params.conditional.size = 1;
      C10_CUDA_CHECK(cudaGraphAddNode(&condition, graph_, &reset_node, nullptr, 1, &params));
      auto loop = params.conditional.phGraph_out[0];
      C10_CUDA_CHECK(cudaGraphAddChildGraphNode(&child, loop, nullptr, 0, reinterpret_cast<cudaGraph_t>(body)));
      void* next_args[] = {&handle, &counter, &stopping, &limit};
      cudaKernelNodeParams next{};
      next.func = reinterpret_cast<void*>(next_iteration);
      next.gridDim = next.blockDim = dim3(1);
      next.kernelParams = next_args;
      C10_CUDA_CHECK(cudaGraphAddKernelNode(&advance, loop, &child, 1, &next));
      C10_CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    } catch (...) {
      release();
      throw;
    }
  }
  void replay() {
    TORCH_CHECK(exec_, "bounded graph is closed");
    c10::cuda::CUDAGuard guard(count_.device());
    auto stream = c10::cuda::getCurrentCUDAStream(device_);
    TORCH_CHECK(stream.stream() == stream_, "bounded graph controls and replay require the construction stream");
    C10_CUDA_CHECK(cudaGraphLaunch(exec_, stream));
    launched_ = true;
    auto status = cudaEventRecord(done_, stream);
    if (status != cudaSuccess) {
      // Never rely on the previous replay's event after a successful launch.
      cudaStreamSynchronize(stream_);
      launched_ = false;
      C10_CUDA_CHECK(status);
    }
  }
  void close() {
    c10::cuda::CUDAGuard guard(count_.device());
    if (launched_) C10_CUDA_CHECK(cudaEventSynchronize(done_));
    launched_ = false;
    release();
    owners_ = pybind11::none();
  }
  void release() noexcept {
    if (exec_) { cudaGraphExecDestroy(exec_); exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
    if (done_) { cudaEventDestroy(done_); done_ = nullptr; }
  }
  ~BoundedGraph() {
    // Explicit close reports CUDA errors. Destruction must not release tensor
    // owners while the last submitted bounded launch still reads them.
    int previous = device_;
    cudaGetDevice(&previous);
    cudaSetDevice(device_);
    if (launched_ && done_) cudaEventSynchronize(done_);
    release();
    cudaSetDevice(previous);
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<BoundedGraph>(m, "BoundedGraph")
      .def(pybind11::init<uintptr_t, at::Tensor, at::Tensor, int, pybind11::object>())
      .def("replay", &BoundedGraph::replay)
      .def("close", &BoundedGraph::close);
}
