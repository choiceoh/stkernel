// Prepared cuBLASLt MXFP8 plans. No allocation, search or transpose in run().
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace py = pybind11;

static void check(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation, " failed: cuBLAS status ", int(status));
}

struct Context {
  int device;
  cublasLtHandle_t handle = nullptr;
  Context(int device_, int major, int minor) : device(device_) {
    c10::cuda::CUDAGuard guard(device);
    const auto* properties = at::cuda::getDeviceProperties(device);
    TORCH_CHECK(major == 12 && (minor == 0 || minor == 1)
                && properties->major == major && properties->minor == minor,
                "ST cuBLASLt device differs from the declared SM120/SM121 target");
    check(cublasLtCreate(&handle), "cublasLtCreate");
  }
  ~Context() { if (handle) cublasLtDestroy(handle); }
};

struct Descriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  ~Descriptors() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (d) cublasLtMatrixLayoutDestroy(d);
    if (operation) cublasLtMatmulDescDestroy(operation);
  }
};

class Plan : public std::enable_shared_from_this<Plan> {
  std::shared_ptr<Context> context;
  Descriptors desc;
  int64_t m, n, k;
  size_t workspace_limit;
  std::vector<cublasLtMatmulHeuristicResult_t> choices;
  std::set<std::string> visited;
  static constexpr size_t MAX_CHOICES = 192;

  template <class T> static T config(const cublasLtMatmulAlgo_t& algo,
                                     cublasLtMatmulAlgoConfigAttributes_t key) {
    T value{}; size_t written = 0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&algo, key, &value, sizeof(value), &written), "algo config");
    TORCH_CHECK(written == sizeof(value), "unexpected cuBLAS config size");
    return value;
  }

  static std::vector<uint32_t> capability(const cublasLtMatmulAlgo_t& algo,
                                         cublasLtMatmulAlgoCapAttributes_t key) {
    if (key != CUBLASLT_ALGO_CAP_TILE_IDS && key != CUBLASLT_ALGO_CAP_STAGES_IDS) {
      uint32_t value = 0; size_t bytes = 0;
      auto status = cublasLtMatmulAlgoCapGetAttribute(&algo, key, &value, sizeof(value), &bytes);
      if (status == CUBLAS_STATUS_NOT_SUPPORTED) return {};
      check(status, "algo scalar capability");
      TORCH_CHECK(bytes == sizeof(value), "unexpected cuBLAS scalar capability size");
      return {value};
    }
    size_t bytes = 0;
    auto status = cublasLtMatmulAlgoCapGetAttribute(&algo, key, nullptr, 0, &bytes);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) return {};
    check(status, "algo capability size");
    TORCH_CHECK(bytes % sizeof(uint32_t) == 0, "unexpected cuBLAS capability size");
    std::vector<uint32_t> values(bytes / sizeof(uint32_t));
    if (bytes) check(cublasLtMatmulAlgoCapGetAttribute(&algo, key, values.data(), bytes, &bytes), "algo capability");
    return values;
  }

  static bool set(cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t key, uint32_t value) {
    auto status = cublasLtMatmulAlgoConfigSetAttribute(&algo, key, &value, sizeof(value));
    if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_INVALID_VALUE) return false;
    check(status, "set algo config");
    return true;
  }

  void admit(const cublasLtMatmulAlgo_t& algo) {
    if (choices.size() >= MAX_CHOICES) return;
    std::string key(reinterpret_cast<const char*>(&algo), sizeof(algo));
    if (!visited.insert(key).second) return;
    cublasLtMatmulHeuristicResult_t result{};
    auto status = cublasLtMatmulAlgoCheck(context->handle, desc.operation, desc.a, desc.b, desc.d, desc.d, &algo, &result);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_INVALID_VALUE
        || status == CUBLAS_STATUS_ARCH_MISMATCH) return;
    check(status, "cublasLtMatmulAlgoCheck");
    if (result.state != CUBLAS_STATUS_SUCCESS || result.workspaceSize > workspace_limit) return;
    // BF16 partial reductions would change the requested FP32 accumulation.
    auto reduction = config<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
    if (reduction != CUBLASLT_REDUCTION_SCHEME_NONE && reduction != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return;
    for (auto attr : {CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
                      CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES}) {
      auto values = capability(algo, attr);
      if (values.size() != 1 || values[0] > 16) return;
    }
    result.algo = algo;
    choices.push_back(result);
  }

  static std::vector<uint32_t> spread(const std::vector<uint32_t>& values, size_t count) {
    if (values.size() <= count) return values;
    std::vector<uint32_t> result;
    for (size_t i = 0; i < count; ++i) result.push_back(values[i * (values.size() - 1) / (count - 1)]);
    return result;
  }

  std::vector<cublasLtMatmulAlgo_t> variants(const cublasLtMatmulAlgo_t& seed) {
    // Interleave families as well as seeds. A global cap must not spend every
    // slot on the first algorithm's smallest tile IDs before trying SplitK.
    std::array<std::vector<cublasLtMatmulAlgo_t>, 5> families;
    auto tiles = spread(capability(seed, CUBLASLT_ALGO_CAP_TILE_IDS), 8);
    auto stages = spread(capability(seed, CUBLASLT_ALGO_CAP_STAGES_IDS), 8);
    for (auto tile : tiles) {
      auto algo = seed;
      if (set(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile)) families[0].push_back(algo);
    }
    for (auto stage : stages) {
      auto algo = seed;
      if (set(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, stage)) families[1].push_back(algo);
    }
    for (auto tile : spread(tiles, 4)) for (auto stage : spread(stages, 4)) {
      auto algo = seed;
      if (set(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile)
          && set(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, stage)) families[2].push_back(algo);
    }
    auto split = capability(seed, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);
    auto reduction = capability(seed, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
    if (split.size() == 1 && split[0] && reduction.size() == 1
        && (reduction[0] & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
      auto split_tiles = spread(tiles, 3);
      split_tiles.insert(split_tiles.begin(), config<uint32_t>(seed, CUBLASLT_ALGO_CONFIG_TILE_ID));
      for (uint32_t count : {2, 4, 8, 16}) for (auto tile : split_tiles) {
        if (k < int64_t(count) * 128) continue;
        auto algo = seed;
        if (set(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile)
            && set(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, count)
            && set(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE))
          families[3].push_back(algo);
      }
    }
    auto swizzle = capability(seed, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
    if (swizzle.size() == 1 && swizzle[0] == 1) {
      auto algo = seed;
      if (set(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
              1 - config<uint32_t>(seed, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING))) families[4].push_back(algo);
    }
    auto custom = capability(seed, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
    if (custom.size() == 1) for (uint32_t option = 0; option <= std::min(custom[0], 4u); ++option) {
      auto algo = seed;
      if (set(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, option)) families[4].push_back(algo);
    }
    std::vector<cublasLtMatmulAlgo_t> result;
    for (size_t round = 0;; ++round) {
      bool more = false;
      for (const auto& family : families) if (round < family.size()) {
        result.push_back(family[round]); more = true;
      }
      if (!more) break;
    }
    return result;
  }

  void tensor(const torch::Tensor& value, at::ScalarType type, const char* name) const {
    TORCH_CHECK(value.is_cuda() && value.get_device() == context->device && value.scalar_type() == type
                && value.is_contiguous() && reinterpret_cast<uintptr_t>(value.data_ptr()) % 16 == 0,
                name, " must be aligned, contiguous and on the plan's CUDA device");
  }

  static bool overlaps(const torch::Tensor& a, const torch::Tensor& b) {
    if (!a.numel() || !b.numel()) return false;
    auto a0 = reinterpret_cast<uintptr_t>(a.data_ptr()), b0 = reinterpret_cast<uintptr_t>(b.data_ptr());
    return a0 < b0 + b.nbytes() && b0 < a0 + a.nbytes();
  }

  static void setup_operation(cublasLtMatmulDesc_t& op) {
    check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "matmul descriptor");
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    int8_t fast = 0;
    auto scaling = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transpose A");
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transpose B");
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)), "FP32 accumulation");
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaling, sizeof(scaling)), "MX scale A");
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaling, sizeof(scaling)), "MX scale B");
  }

 public:
  Plan(std::shared_ptr<Context> context_, int64_t m_, int64_t n_, int64_t k_, size_t limit)
      : context(std::move(context_)), m(m_), n(n_), k(k_), workspace_limit(limit) {
    TORCH_CHECK(m > 0 && n > 0 && k > 0 && n % 128 == 0 && k % 128 == 0
                && m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX, "invalid ST MXFP8 shape");
    c10::cuda::CUDAGuard guard(context->device);
    setup_operation(desc.operation);
    // Column-major TN computes Y^T from the existing row-major W and X.
    check(cublasLtMatrixLayoutCreate(&desc.a, CUDA_R_8F_E4M3, k, n, k), "W layout");
    check(cublasLtMatrixLayoutCreate(&desc.b, CUDA_R_8F_E4M3, k, m, k), "X layout");
    check(cublasLtMatrixLayoutCreate(&desc.d, CUDA_R_16BF, n, m, n), "Y layout");
    check(cublasLtMatmulPreferenceCreate(&desc.preference), "matmul preference");
    uint32_t alignment = 16, reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
      check(cublasLtMatmulPreferenceSetAttribute(desc.preference, attr, &alignment, sizeof(alignment)), "pointer alignment");
    check(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                              &reduction, sizeof(reduction)), "reduction precision");
    for (size_t budget : {size_t(0), size_t(1) << 20, size_t(8) << 20, size_t(32) << 20, size_t(64) << 20}) {
      if (budget > workspace_limit) continue;
      check(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &budget, sizeof(budget)), "workspace bound");
      std::array<cublasLtMatmulHeuristicResult_t, 16> result{};
      int count = 0;
      auto status = cublasLtMatmulAlgoGetHeuristic(context->handle, desc.operation, desc.a, desc.b,
                       desc.d, desc.d, desc.preference, result.size(), result.data(), &count);
      if (status == CUBLAS_STATUS_NOT_SUPPORTED) continue;
      check(status, "cuBLAS heuristics");
      for (int i = 0; i < count; ++i) if (result[i].state == CUBLAS_STATUS_SUCCESS) admit(result[i].algo);
    }
    auto seeds = choices;
    std::vector<cublasLtMatmulAlgo_t> selected;
    std::set<int32_t> ids;
    // Cover distinct implementations first, then alternate heuristic configs.
    for (const auto& seed : seeds)
      if (selected.size() < 16 && ids.insert(config<int32_t>(seed.algo, CUBLASLT_ALGO_CONFIG_ID)).second)
        selected.push_back(seed.algo);
    for (const auto& seed : seeds) {
      if (selected.size() >= 16) break;
      if (std::none_of(selected.begin(), selected.end(), [&](const auto& a) {
            return std::memcmp(&a, &seed.algo, sizeof(a)) == 0; })) selected.push_back(seed.algo);
    }
    std::vector<std::vector<cublasLtMatmulAlgo_t>> proposals;
    for (const auto& seed : selected) proposals.push_back(variants(seed));
    for (size_t round = 0; choices.size() < MAX_CHOICES; ++round) {
      bool more = false;
      for (const auto& family : proposals) if (round < family.size()) {
        admit(family[round]); more = true;
      }
      if (!more) break;
    }
  }

  py::list candidates() const {
    py::list result;
    for (size_t i = 0; i < choices.size(); ++i) {
      const auto& c = choices[i];
      py::dict row;
      row["index"] = i; row["workspace"] = c.workspaceSize;
      row["id"] = config<int32_t>(c.algo, CUBLASLT_ALGO_CONFIG_ID);
      row["tile"] = config<uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
      row["stages"] = config<uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
      row["split_k"] = config<int32_t>(c.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
      row["reduction"] = config<uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
      row["swizzle"] = config<uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
      row["custom"] = config<uint32_t>(c.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
      result.append(row);
    }
    return result;
  }

  void validate(size_t index, const torch::Tensor& q, const torch::Tensor& weight,
                const torch::Tensor& s_q, const torch::Tensor& s_weight,
                const torch::Tensor& out, const torch::Tensor& workspace) const {
    TORCH_CHECK(index < choices.size(), "unknown cuBLAS algorithm");
    tensor(q, at::ScalarType::Float8_e4m3fn, "Q"); tensor(weight, at::ScalarType::Float8_e4m3fn, "W");
    tensor(s_q, at::kByte, "activation scales"); tensor(s_weight, at::kByte, "weight scales");
    tensor(out, at::kBFloat16, "output"); tensor(workspace, at::kByte, "workspace");
    TORCH_CHECK(q.dim() == 2 && q.size(0) == m && q.size(1) == k
                && weight.dim() == 2 && weight.size(0) == n && weight.size(1) == k
                && out.dim() == 2 && out.size(0) == m && out.size(1) == n
                && s_q.dim() == 1 && s_q.numel() == ((m + 127) / 128) * (k / 128) * 512
                && s_weight.dim() == 1 && s_weight.numel() == n * (k / 32), "MXFP8 plan shape mismatch");
    const auto& choice = choices[index];
    TORCH_CHECK(workspace.dim() == 1 && size_t(workspace.numel()) >= choice.workspaceSize
                && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 256 == 0, "insufficient/alignment-invalid cuBLAS workspace");
    for (const auto& input : {q, weight, s_q, s_weight}) {
      TORCH_CHECK(!overlaps(input, out) && !overlaps(input, workspace), "cuBLAS write buffer overlaps an input");
    }
    TORCH_CHECK(!overlaps(out, workspace), "cuBLAS workspace overlaps output");
  }

  static void set_scales(cublasLtMatmulDesc_t op, const torch::Tensor& s_q, const torch::Tensor& s_weight) {
    const void* a_scale = s_weight.data_ptr(); const void* b_scale = s_q.data_ptr();
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)), "scale pointer A");
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)), "scale pointer B");
  }

  void launch(cublasLtMatmulDesc_t op, size_t index, const torch::Tensor& q,
              const torch::Tensor& weight, const torch::Tensor& out, const torch::Tensor& workspace) {
    c10::cuda::CUDAGuard guard(context->device);
    const auto& choice = choices[index];
    const float alpha = 1.f, beta = 0.f;
    check(cublasLtMatmul(context->handle, op, &alpha, weight.data_ptr(), desc.a,
                        q.data_ptr(), desc.b, &beta, out.data_ptr(), desc.d, out.data_ptr(), desc.d,
                        &choice.algo, workspace.data_ptr(), choice.workspaceSize,
                        at::cuda::getCurrentCUDAStream(context->device)), "MXFP8 matmul");
  }

  void run(size_t index, const torch::Tensor& q, const torch::Tensor& weight,
           const torch::Tensor& s_q, const torch::Tensor& s_weight,
           const torch::Tensor& out, const torch::Tensor& workspace) {
    validate(index, q, weight, s_q, s_weight, out, workspace);
    set_scales(desc.operation, s_q, s_weight);
    launch(desc.operation, index, q, weight, out, workspace);
  }

  struct Bound {
    std::shared_ptr<Plan> plan;
    Descriptors desc;  // own operation; matrix layouts remain owned by plan
    size_t index;
    torch::Tensor q, weight, s_q, s_weight, out, workspace;
    Bound(std::shared_ptr<Plan> owner, size_t selected, torch::Tensor q_, torch::Tensor weight_,
          torch::Tensor s_q_, torch::Tensor s_weight_, torch::Tensor out_, torch::Tensor workspace_)
        : plan(std::move(owner)), index(selected), q(q_), weight(weight_), s_q(s_q_),
          s_weight(s_weight_), out(out_), workspace(workspace_) {
      plan->validate(index, q, weight, s_q, s_weight, out, workspace);
      setup_operation(desc.operation);
      set_scales(desc.operation, s_q, s_weight);
    }
    void run() { plan->launch(desc.operation, index, q, weight, out, workspace); }
  };

  std::shared_ptr<Bound> bind(size_t index, const torch::Tensor& q, const torch::Tensor& weight,
                              const torch::Tensor& s_q, const torch::Tensor& s_weight,
                              const torch::Tensor& out, const torch::Tensor& workspace) {
    return std::make_shared<Bound>(shared_from_this(), index, q, weight, s_q, s_weight, out, workspace);
  }

  py::dict statistics() const {
    py::dict result;
    result["checked_configurations"] = visited.size();
    result["admitted_configurations"] = choices.size();
    result["candidate_limit"] = MAX_CHOICES;
    return result;
  }

};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("version", [] { return cublasLtGetVersion(); });
  py::class_<Context, std::shared_ptr<Context>>(module, "Context").def(py::init<int, int, int>());
  py::class_<Plan::Bound, std::shared_ptr<Plan::Bound>>(module, "BoundMatmul").def("run", &Plan::Bound::run);
  py::class_<Plan, std::shared_ptr<Plan>>(module, "Plan")
      .def(py::init<std::shared_ptr<Context>, int64_t, int64_t, int64_t, size_t>())
      .def("candidates", &Plan::candidates).def("run", &Plan::run)
      .def("bind", &Plan::bind).def("statistics", &Plan::statistics);
}
