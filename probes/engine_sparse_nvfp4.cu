// Diagnostic only: native SM121a dense/sparse NVFP4 GEMM on the same values.
// CUTLASS remains an external dependency; no serving dispatch is changed.
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"

using namespace cute;
using FP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using BF16 = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Problem = Shape<int, int, int, int>;
using ProbeTile = Shape<_128, _128, _256>;
using Cluster = Shape<_1, _1, _1>;

template<bool Sparse> struct Config {
  using Op = std::conditional_t<Sparse, cutlass::arch::OpClassBlockScaledSparseTensorOp,
                                      cutlass::arch::OpClassBlockScaledTensorOp>;
  using EpiSchedule = std::conditional_t<Sparse,
      cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
      cutlass::epilogue::TmaWarpSpecializedCooperative>;
  using Schedule = std::conditional_t<Sparse,
      cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120,
      cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, Op, ProbeTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, void, Col, 8, BF16, Col, 8, EpiSchedule>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, Op, FP4, Row, Sparse ? 64 : 32, FP4, Col, 32,
      float, ProbeTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Problem, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using SparseConfig = Config<true>::Main::SparseConfig;
using Utility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
    Problem, FP4::DataType, Row, SparseConfig>;
using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
    Problem, FP4::DataType, Row, SparseConfig, cutlass::arch::Sm120>;
using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

static thread_local std::string last_error;
static void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
static void check(cutlass::Status s) {
  if (s != cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));
}
struct Buffer {
  void* ptr = nullptr;
  explicit Buffer(size_t n = 0) { if (n) check(cudaMalloc(&ptr, n)); }
  ~Buffer() { if (ptr) cudaFree(ptr); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
struct Context {
  virtual void run(cudaStream_t) = 0;
  virtual ~Context() = default;
};

template<int DestinationGroup, class Layout>
__global__ void pack_scales(const uint8_t* src, uint8_t* dst, Layout layout,
                            int rows, int k, int batches) {
  int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  int blocks = k / 32;
  if (i < rows * blocks * batches) {
    int scale = i % blocks, row = (i / blocks) % rows, batch = i / (blocks * rows);
    for (int j = 0; j < 32; j += DestinationGroup)
      dst[layout(row, scale * 32 + j, batch)] = src[i];
  }
}

template<bool Sparse> struct GemmContext : Context {
  using Cfg = Config<Sparse>;
  using Gemm = typename Cfg::Gemm;
  Gemm gemm;
  Buffer workspace;
  GemmContext(typename Gemm::Arguments args, cudaStream_t stream)
      : workspace(Gemm::get_workspace_size(args)) {
    check(gemm.can_implement(args));
    check(gemm.initialize(args, workspace.ptr, stream));
  }
  void run(cudaStream_t stream) override { check(gemm.run(stream)); }
};

// All data buffers belong to Python and must outlive their contexts/graphs.
template<bool Sparse>
Context* make_context(Problem shape, void* a, void* b, void* sf_a, void* sf_b,
                       void* metadata, void* output, cudaStream_t stream) {
  using Cfg = Config<Sparse>;
  using Kernel = typename Cfg::Kernel;
  using SF = typename Cfg::Main::Sm1xxBlkScaledConfig;
  auto [m, n, k, batches] = shape;
  auto da = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, batches});
  auto db = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, batches});
  auto dc = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {m, n, batches});
  auto dd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, batches});
  typename Cfg::Main::Arguments main;
  main.ptr_A = static_cast<FP4::DataType*>(a);
  main.ptr_B = static_cast<FP4::DataType*>(b);
  main.dB = db;
  main.ptr_SFA = static_cast<FP4::ScaleFactorType*>(sf_a);
  main.ptr_SFB = static_cast<FP4::ScaleFactorType*>(sf_b);
  main.layout_SFA = SF::tile_atom_to_shape_SFA(shape);
  main.layout_SFB = SF::tile_atom_to_shape_SFB(shape);
  if constexpr (Sparse) {
    main.layout_a = SparseConfig::fill_layoutA(shape);
    main.ptr_E = static_cast<uint8_t*>(metadata);
    main.layout_e = SparseConfig::fill_layoutE(shape);
  } else {
    main.dA = da;
  }
  typename Cfg::Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, shape, main,
    {{1.f, 0.f}, nullptr, dc, static_cast<BF16*>(output), dd}
  };
  return new GemmContext<Sparse>(args, stream);
}

extern "C" const char* sparse_probe_error() { return last_error.c_str(); }

// Sizes are bytes. Sparse uses logical K32 scales; dense repeats each at K16.
extern "C" int sparse_probe_sizes(int m, int n, int k, int batches, int64_t* out) {
  try {
    if (m <= 0 || m % 128 || n <= 0 || k <= 0 || k % 256 || batches <= 0)
      throw std::runtime_error("requires positive M%128=0, K%256=0, N and batch");
    auto p = make_shape(m,n,k,batches);
    auto da = cutlass::make_cute_packed_stride(Config<true>::Kernel::StrideA{}, {m,k,batches});
    Utility u(p, da);
    out[0] = u.get_compressed_tensor_A_bytes();
    out[1] = u.get_tensor_E_bytes();
    using SF = Config<true>::Main::Sm1xxBlkScaledConfig;
    using DenseSF = Config<false>::Main::Sm1xxBlkScaledConfig;
    out[2] = size(filter_zeros(SF::tile_atom_to_shape_SFA(p)));
    out[3] = size(filter_zeros(SF::tile_atom_to_shape_SFB(p)));
    out[4] = size(filter_zeros(DenseSF::tile_atom_to_shape_SFA(p)));
    out[5] = size(filter_zeros(DenseSF::tile_atom_to_shape_SFB(p)));
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

extern "C" int sparse_probe_prepare(int m, int n, int k, int batches,
    void* a, void* compressed, void* metadata, void* source_sf_a, void* source_sf_b,
    void* sf_a, void* sf_b, void* dense_sf_a, void* dense_sf_b, void* stream_ptr) {
  try {
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    auto p = make_shape(m,n,k,batches);
    auto da = cutlass::make_cute_packed_stride(Config<true>::Kernel::StrideA{}, {m,k,batches});
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    Compressor::Arguments args{p, {static_cast<FP4::DataType*>(a), da,
                                   static_cast<FP4::DataType*>(compressed),
                                   static_cast<uint8_t*>(metadata)}, {hw}};
    Compressor compressor;
    check(compressor.can_implement(args));
    Buffer workspace(Compressor::get_workspace_size(args));
    check(compressor.initialize(args, workspace.ptr, stream));
    check(compressor.run(stream));
    using SF = Config<true>::Main::Sm1xxBlkScaledConfig;
    using DenseSF = Config<false>::Main::Sm1xxBlkScaledConfig;
    pack_scales<32><<<(m*k/32*batches+255)/256,256,0,stream>>>(
        static_cast<uint8_t*>(source_sf_a), static_cast<uint8_t*>(sf_a),
        SF::tile_atom_to_shape_SFA(p), m,k,batches);
    pack_scales<32><<<(n*k/32*batches+255)/256,256,0,stream>>>(
        static_cast<uint8_t*>(source_sf_b), static_cast<uint8_t*>(sf_b),
        SF::tile_atom_to_shape_SFB(p), n,k,batches);
    pack_scales<16><<<(m*k/32*batches+255)/256,256,0,stream>>>(
        static_cast<uint8_t*>(source_sf_a), static_cast<uint8_t*>(dense_sf_a),
        DenseSF::tile_atom_to_shape_SFA(p), m,k,batches);
    pack_scales<16><<<(n*k/32*batches+255)/256,256,0,stream>>>(
        static_cast<uint8_t*>(source_sf_b), static_cast<uint8_t*>(dense_sf_b),
        DenseSF::tile_atom_to_shape_SFB(p), n,k,batches);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

extern "C" void* sparse_probe_create(int sparse, int m, int n, int k, int batches,
    void* a, void* b, void* sf_a, void* sf_b, void* metadata, void* output, void* stream) {
  try {
    auto p = make_shape(m,n,k,batches);
    if (sparse) return make_context<true>(p,a,b,sf_a,sf_b,metadata,output,static_cast<cudaStream_t>(stream));
    return make_context<false>(p,a,b,sf_a,sf_b,metadata,output,static_cast<cudaStream_t>(stream));
  } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
extern "C" int sparse_probe_run(void* context, void* stream) {
  try {
    static_cast<Context*>(context)->run(static_cast<cudaStream_t>(stream));
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
extern "C" void sparse_probe_destroy(void* context) { delete static_cast<Context*>(context); }
