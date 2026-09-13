// Bounded WHILE over an owned deterministic iteration graph, GB10 only.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

__global__ void reset_count(int64_t* count) { *count = 0; }
__device__ unsigned long long timer_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__global__ void begin_iteration(const int64_t* count, int64_t* times) {
  times[2 * *count] = timer_ns();
}
__global__ void stamp_point(int64_t* times, const int64_t* count, int width, int column) {
  times[width * *count + column] = timer_ns();
}
void stamp(at::Tensor times, at::Tensor count, int column) {
  TORCH_CHECK(times.is_cuda() && count.is_cuda() && times.device() == count.device() &&
              times.scalar_type() == at::kLong && count.scalar_type() == at::kLong &&
              times.is_contiguous() && count.is_contiguous() && count.numel() == 1 &&
              times.dim() == 2 && times.size(0) == 4 && column >= 0 && column < times.size(1),
              "stage timestamps require same-device int64[4,width] and an iteration index");
  c10::cuda::CUDAGuard guard(times.device());
  stamp_point<<<1, 1, 0, c10::cuda::getCurrentCUDAStream(times.get_device())>>>(
      times.data_ptr<int64_t>(), count.data_ptr<int64_t>(), times.size(1), column);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void next_iteration(cudaGraphConditionalHandle handle, int64_t* count,
                                const int64_t* stop, int limit, int64_t* times) {
  times[2 * *count + 1] = timer_ns();
  int64_t done = ++(*count);
  cudaGraphSetConditional(handle, done < limit && *stop == 0);
}

static void validate_body(cudaGraph_t graph, const std::string& path = "body") {
  size_t size = 0;
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &size));
  TORCH_CHECK(size > 0, "bounded graph body is empty");
  std::vector<cudaGraphNode_t> nodes(size);
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &size));
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto node = nodes[i];
    const auto location = path + "/" + std::to_string(i);
    cudaGraphNodeType type;
    C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child = nullptr;
      C10_CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(node, &child));
      validate_body(child, location);
    } else {
      TORCH_CHECK(type == cudaGraphNodeTypeKernel || type == cudaGraphNodeTypeEmpty ||
                  type == cudaGraphNodeTypeMemcpy || type == cudaGraphNodeTypeMemset,
                  "bounded graph rejects node type ", static_cast<int>(type), " at ", location,
                  ": host callbacks, event/semaphore nodes, allocation and nested conditions are forbidden");
    }
  }
  // CUDA instantiation checks device accessibility, mapped-copy operands,
  // forbidden dynamic/device launches and all conditional body constraints.
}

void append_child(uintptr_t body) {
  TORCH_CHECK(body != 0, "composition needs a retained child graph");
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus status;
  cudaGraph_t parent = nullptr;
  const cudaGraphNode_t* dependencies = nullptr;
  const cudaGraphEdgeData* edge_data = nullptr;
  size_t count = 0;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &parent,
                                         &dependencies, &edge_data, &count));
  TORCH_CHECK(status == cudaStreamCaptureStatusActive && parent,
              "child graph composition requires an active stream capture");
  cudaGraphNode_t child;
  // Full completion edges deliberately serialize the borrowed child against
  // the preceding input copies/stamp. Its internal stream edges are cloned.
  C10_CUDA_CHECK(cudaGraphAddChildGraphNode(&child, parent, dependencies, count,
                                            reinterpret_cast<cudaGraph_t>(body)));
  C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream, &child, nullptr, 1,
                                                     cudaStreamSetCaptureDependencies));
}

class BoundedGraph {
  // The cloned graph retains addresses, not Python allocation owners. Keep the
  // captured body/pool and all caller buffers alive until the last launch ends.
  pybind11::object owners_;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  cudaEvent_t done_ = nullptr;
  at::Tensor count_, stop_, times_;
  int device_;
  cudaStream_t stream_ = nullptr;
  bool launched_ = false;
 public:
  BoundedGraph(uintptr_t body, at::Tensor count, at::Tensor stop, int limit,
               at::Tensor times, pybind11::object owners)
      : owners_(std::move(owners)), count_(count), stop_(stop), times_(times),
        device_(count.is_cuda() ? count.get_device() : -1) {
    TORCH_CHECK(limit == 1 || limit == 2 || limit == 4, "bounded graph permits 1, 2 or 4 iterations");
    for (const auto& t : {count_, stop_})
      TORCH_CHECK(t.is_cuda() && t.device() == count_.device() && t.scalar_type() == at::kLong &&
                  t.is_contiguous() && t.dim() == 1 && t.numel() == 1,
                  "bounded graph needs same-device contiguous CUDA int64[1] controls");
    TORCH_CHECK(count_.data_ptr() != stop_.data_ptr(), "count and stop must not alias");
    TORCH_CHECK(times_.is_cuda() && times_.device() == count_.device() &&
                times_.scalar_type() == at::kLong && times_.is_contiguous() &&
                times_.dim() == 2 && times_.size(0) == 4 && times_.size(1) == 2,
                "bounded graph needs CUDA int64[4,2] iteration timestamps");
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
      auto* times_ptr = times_.data_ptr<int64_t>();
      void* reset_args[] = {&counter};
      cudaKernelNodeParams reset{};
      reset.func = reinterpret_cast<void*>(reset_count);
      reset.gridDim = reset.blockDim = dim3(1);
      reset.kernelParams = reset_args;
      cudaGraphNode_t reset_node, condition, begin, child, advance;
      C10_CUDA_CHECK(cudaGraphAddKernelNode(&reset_node, graph_, nullptr, 0, &reset));
      cudaGraphNodeParams params{};
      params.type = cudaGraphNodeTypeConditional;
      params.conditional.handle = handle;
      params.conditional.type = cudaGraphCondTypeWhile;
      params.conditional.size = 1;
      C10_CUDA_CHECK(cudaGraphAddNode(&condition, graph_, &reset_node, nullptr, 1, &params));
      auto loop = params.conditional.phGraph_out[0];
      void* begin_args[] = {&counter, &times_ptr};
      cudaKernelNodeParams stamp{};
      stamp.func = reinterpret_cast<void*>(begin_iteration);
      stamp.gridDim = stamp.blockDim = dim3(1);
      stamp.kernelParams = begin_args;
      C10_CUDA_CHECK(cudaGraphAddKernelNode(&begin, loop, nullptr, 0, &stamp));
      C10_CUDA_CHECK(cudaGraphAddChildGraphNode(&child, loop, &begin, 1, reinterpret_cast<cudaGraph_t>(body)));
      void* next_args[] = {&handle, &counter, &stopping, &limit, &times_ptr};
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
  m.def("append_child", &append_child);
  m.def("stamp", &stamp);
  pybind11::class_<BoundedGraph>(m, "BoundedGraph")
      .def(pybind11::init<uintptr_t, at::Tensor, at::Tensor, int, at::Tensor, pybind11::object>())
      .def("replay", &BoundedGraph::replay)
      .def("close", &BoundedGraph::close);
}
