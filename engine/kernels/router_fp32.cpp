// SPDX-License-Identifier: Apache-2.0
// Keep the router's FP32 multiply independent of ambient TF32/autocast policy.
#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/mm.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/utils/pybind.h>

at::Tensor router_logits(const at::Tensor& x, const at::Tensor& weight) {
  TORCH_CHECK(x.is_cuda() && x.device() == weight.device() &&
              x.dim() == 2 && weight.dim() == 2 && x.size(1) == weight.size(1) &&
              (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat) &&
              weight.scalar_type() == at::kFloat,
              "FP32 router requires CUDA BF16/FP32 inputs and FP32 [experts, hidden] weights");
  // Both guards are thread-local. Never change the process-wide matmul flags:
  // the shared-expert stream and other callers keep their own precision policy.
  at::NoTF32Guard ieee;
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastCUDA);
  return at::mm(x.to(at::kFloat), weight.t());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &router_logits);
}
