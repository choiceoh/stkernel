// Exercise the production reservation, publication and mixed-collective kernels
// against owned mapped memory. The test CPU stands in for the NIC/proxy.
#define ST_ONESHOT_LOCAL_TEST
#include "dsv4_oneshot_ar.cu"
#include <optional>
#include <pybind11/stl.h>

static Ctrl* host_control(const at::Tensor& host) {
  TORCH_CHECK(host.device().is_cpu() && host.scalar_type() == at::kByte &&
              host.is_contiguous() && host.numel() >= sizeof(Ctrl), "owned host control required");
  return reinterpret_cast<Ctrl*>(host.data_ptr());
}

static void prepare(at::Tensor host, at::Tensor device, int rank, int64_t sequence, int64_t ack) {
  auto* c = host_control(host);
  TORCH_CHECK(device.is_cuda() && device.numel() == host.numel() && 0 <= rank && rank < 4,
              "invalid test device/rank");
  memset(c, 0, sizeof(Ctrl));
  c->tx_seq = sequence;
  c->ack_seq = ack;
  c->done_ctr = sequence * ARGRID;
  g_ctrl = reinterpret_cast<Ctrl*>(device.data_ptr());
  g_started = true;
  g_rank = rank;
}

static uint64_t published(at::Tensor host) {
  return __atomic_load_n(&host_control(host)->tx_seq, __ATOMIC_ACQUIRE);
}

static at::Tensor payload(at::Tensor host, uint64_t sequence) {
  auto* c = host_control(host);
  auto bytes = c->nbytes[sequence % RING];
  TORCH_CHECK(bytes > 0 && bytes <= MAXEL * 2, "invalid published bytes");
  return at::from_blob(c->tx[sequence % RING], {(int64_t)bytes}, [host](void*) {},
                       at::TensorOptions().dtype(at::kByte).device(at::kCPU));
}

static void land(at::Tensor host, uint64_t sequence) {
  auto* c = host_control(host);
  const auto slot = sequence % RING;
  for (int p = 0; p < NPEER; ++p) {
    memcpy(c->rx[slot][p], c->tx[slot], c->nbytes[slot]);
    __atomic_store_n(&c->rxf[slot][p], sequence, __ATOMIC_RELEASE);
  }
  __atomic_store_n(&c->ack_seq, sequence, __ATOMIC_RELEASE);
}

static void land_peers(at::Tensor host, uint64_t sequence, at::Tensor peers) {
  auto* c = host_control(host);
  const auto slot = sequence % RING;
  TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kByte && peers.is_contiguous() &&
              peers.dim() == 2 && peers.size(0) == NPEER && peers.size(1) == c->nbytes[slot],
              "oracle needs three exact rank-ordered byte packets");
  for (int p = 0; p < NPEER; ++p) {
    memcpy(c->rx[slot][p], peers[p].data_ptr(), c->nbytes[slot]);
    __atomic_store_n(&c->rxf[slot][p], sequence, __ATOMIC_RELEASE);
  }
  __atomic_store_n(&c->ack_seq, sequence, __ATOMIC_RELEASE);
}

// Peers that already answered: packets, flags and the all-peer ACK for `count` future sequences, so a timed
// chain measures only the GPU side of each collective. Later sequences overwrite a reused slot's flag upward.
static void land_ahead(at::Tensor host, uint64_t first, int64_t count, at::Tensor peers) {
  auto* c = host_control(host);
  TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kByte && peers.is_contiguous() &&
              peers.dim() == 2 && peers.size(0) == NPEER && peers.size(1) <= MAXEL * 2 && count > 0 &&
              count <= 64 && first > __atomic_load_n(&c->tx_seq, __ATOMIC_ACQUIRE),
              "landing ahead needs future sequences and three rank-ordered packets");
  for (uint64_t sequence = first; sequence < first + (uint64_t)count; ++sequence) {
    const auto slot = sequence % RING;
    for (int p = 0; p < NPEER; ++p) {
      memcpy(c->rx[slot][p], peers[p].data_ptr(), peers.size(1));
      __atomic_store_n(&c->rxf[slot][p], sequence, __ATOMIC_RELEASE);
    }
  }
  __atomic_store_n(&c->ack_seq, first + (uint64_t)count - 1, __ATOMIC_RELEASE);
}

// A PDL neighbour on the collective's stream. As its producer it releases the next launch before a late
// write, so a sum that read before its own dependency wait would see the previous input; as its successor it
// launches at the sum's release, before the peers land, and copies only after its wait.
// `stamps`, when given, receives block 0 thread 0's SM clock at entry and after the dependency wait: a
// successor that launched at its predecessor's release shows the predecessor's remaining time between them.
__global__ void staged_copy(const bf16* src, bf16* dst, int n, long long cycles, long long* stamps) {
  if (stamps != nullptr && blockIdx.x == 0 && threadIdx.x == 0) stamps[0] = clock64();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (stamps != nullptr && blockIdx.x == 0 && threadIdx.x == 0) stamps[1] = clock64();
  asm volatile("griddepcontrol.launch_dependents;");
  const long long start = clock64();
  if (threadIdx.x == 0)
    while (clock64() - start < cycles) __nanosleep(64);
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

static void launch_staged_copy(at::Tensor src, at::Tensor dst, int64_t cycles, std::optional<at::Tensor> stamps) {
  TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.scalar_type() == at::kBFloat16 &&
              dst.scalar_type() == at::kBFloat16 && src.is_contiguous() && dst.is_contiguous() &&
              src.numel() == dst.numel() && src.numel() <= MAXEL && cycles >= 0,
              "staged copy needs matching contiguous BF16 tensors within MAXEL");
  TORCH_CHECK(!stamps || (stamps->is_cuda() && stamps->scalar_type() == at::kLong && stamps->is_contiguous() &&
                          stamps->numel() == 2), "staged copy stamps are a contiguous CUDA int64[2]");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(ARGRID);
  cfg.blockDim = dim3(ARTHREADS);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  const auto err = cudaLaunchKernelEx(&cfg, staged_copy, reinterpret_cast<const bf16*>(src.data_ptr()),
                                       reinterpret_cast<bf16*>(dst.data_ptr()), int(src.numel()), (long long)cycles,
                                       stamps ? reinterpret_cast<long long*>(stamps->data_ptr()) : nullptr);
  TORCH_CHECK(err == cudaSuccess, "staged copy launch: ", cudaGetErrorString(err));
}

__global__ void fold(const int64_t* addresses, bf16* out, int n) {
  auto ranks = reinterpret_cast<const bf16* const*>(addresses);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float v = __bfloat162float(ranks[0][i]) + __bfloat162float(ranks[1][i]);
    v += __bfloat162float(ranks[2][i]);
    v += __bfloat162float(ranks[3][i]);
    out[i] = __float2bfloat16(v);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bytes", []() { return (sizeof(Ctrl)+4095)/4096*4096; });
  m.def("prepare", &prepare);
  m.def("published", &published);
  m.def("payload", &payload);
  m.def("land", &land);
  m.def("land_peers", &land_peers);
  m.def("land_ahead", &land_ahead);
  m.def("ack", [](at::Tensor host, uint64_t sequence) {
    __atomic_store_n(&host_control(host)->ack_seq, sequence, __ATOMIC_RELEASE);
  });
  m.def("tickets", [](at::Tensor host) { return host_control(host)->done_ctr; });
  m.def("reserve_packets", &py_reserve_packets);
  m.def("publish_packets", &py_publish_packets);
  m.def("oneshot_packets", &py_oneshot_packets);
  m.def("oneshot_ar", &py_oneshot);
  m.def("oneshot_ar_consumer", &py_oneshot_consumer);
  m.def("staged_copy", &launch_staged_copy, pybind11::arg("src"), pybind11::arg("dst"), pybind11::arg("cycles"),
        pybind11::arg("stamps") = pybind11::none());
  m.def("moe_packets", &py_moe_packets);
  m.def("moe_gated_packets", &py_moe_gated_packets);
  m.def("oneshot_max_int64", &py_oneshot_max_int64);
  m.def("oneshot_gather_int64", &py_oneshot_gather_int64);
  m.def("consume", [](at::Tensor addresses, at::Tensor out) {
    fold<<<48, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(addresses.data_ptr<int64_t>(),
        reinterpret_cast<bf16*>(out.data_ptr()), out.numel());
  });
}
