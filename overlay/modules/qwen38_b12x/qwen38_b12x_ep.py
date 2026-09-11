"""b12x under expert parallelism on Qwen3.8-Flash-Next: the vLLM side.

vLLM's `FlashInferB12xExperts` refuses EP -- `_supports_parallel_config` is
`not use_ep` -- and with reason: the wrapper it builds insists
`num_local_experts == num_experts`, because the SM120 kernel indexes its
weights and its routing state with the ids it is handed, and under EP the ids
are global while the weights are local (flashinfer #3383). The first TEP=4
boot died on exactly that oracle check:

    NvFp4 MoE backend 'FLASHINFER_B12X' does not support the deployment
    configuration since kernel does not support parallel config ...

With the unrouted-slot guard in this module's kernel override
(moe_dynamic_generic.py), what is missing is one line: map the global top-k
ids through vLLM's `expert_map` -- local slot, or -1 for an expert another
rank owns -- before the kernel sees them. A -1 slot then owns no row, a token
none of whose experts live here comes out zero, and the TP all-reduce that
already follows the MoE sums the ranks. No dummy expert, no GEMM rows for the
3/4 of routes that are not this rank's.

So, installed like qwen38_b12x_bounds through a meta-path hook:

  * `_supports_parallel_config` -> True
  * the `B12xMoEWrapper` is built for the LOCAL expert count (the weights' E),
    ONCE per geometry and shared by all 48 MoE layers (13.5-20.8 GiB of
    per-layer workspaces otherwise)
  * `apply()`: `topk_ids = expert_map[topk_ids]` when a map is given; for
    the static/micro kernels (small batches) a -1 slot becomes expert 0 at
    weight 0 instead, since only the dynamic kernel carries the guard

Nothing else in the class changes. `DENEB_B12X_EP=0` leaves the stock class
alone, in which case an EP boot fails the oracle exactly as before.
"""

from __future__ import annotations

import os
import sys

import torch

TARGET = "vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe"
_DONE = False
_SHARED_WRAPPERS: dict = {}     # geometry -> B12xMoEWrapper, process-wide


def _log(msg: str) -> None:
    sys.stderr.write(f"[qwen38-b12x-ep] {msg}\n")
    sys.stderr.flush()


def _small_batch_backend(experts, num_tokens: int) -> str:
    """'static' (static/micro kernels) or 'dynamic', as the wrapper decides."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        select_sm120_moe_backend,
    )
    return select_sm120_moe_backend(
        num_tokens=int(num_tokens), num_topk=experts.topk,
        activation_precision="fp4", quant_mode="nvfp4")


def _patch(mod) -> None:
    global _DONE
    if _DONE or not hasattr(mod, "FlashInferB12xExperts"):
        return
    cls = mod.FlashInferB12xExperts

    cls._supports_parallel_config = staticmethod(lambda moe_parallel_config: True)

    def _ensure_wrapper(self) -> None:
        if self._wrapper is not None:
            return
        from flashinfer.fused_moe import B12xMoEWrapper

        # ONE wrapper per geometry for the whole process, not one per layer.
        # A wrapper built with use_cuda_graph=True preallocates its static and
        # dynamic workspaces and its output buffer for max_num_tokens; measured
        # on this model's local shape (E=128, K=2560, N=640, top-10) that is
        # 0.28 GiB at 8192 tokens and 0.43 GiB at 16384 -- times 48 MoE layers,
        # 13.5 / 20.8 GiB per rank of scratch that is never live in two layers
        # at once. run() takes the weights as arguments and keys its
        # weight-view cache on their pointers, so the layers can share.
        # The weights on this rank are [num_local_experts, ...]; the kernel's
        # E is that, and the ids reaching it are local (or -1) by apply().
        key = (self.num_local_experts, self.topk, self.hidden_dim,
               self.intermediate_size_per_partition, self.max_num_tokens,
               self._activation_str)
        wrapper = _SHARED_WRAPPERS.get(key)
        if wrapper is None:
            wrapper = B12xMoEWrapper(
                num_experts=self.num_local_experts,
                top_k=self.topk,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                use_cuda_graph=True,
                max_num_tokens=self.max_num_tokens,
                num_local_experts=self.num_local_experts,
                activation=self._activation_str,
            )
            _SHARED_WRAPPERS[key] = wrapper
            _log(f"b12x wrapper for {key} built once, shared by every MoE layer")
        self._wrapper = wrapper

    cls._ensure_wrapper = _ensure_wrapper

    inner_apply = cls.apply

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        if expert_map is not None:
            # global expert id -> this rank's slot, or -1: vLLM's unrouted
            # sentinel, which the guarded DYNAMIC kernel skips.
            topk_ids = expert_map[topk_ids.long()]
            if _small_batch_backend(self, topk_ids.shape[0]) == "static":
                # The static and micro kernels (routed_rows <= the static
                # cutover: decode, short prompts) carry no unrouted-slot guard
                # -- the first TEP=4 request died there with an IMA. For them
                # the sentinel becomes expert 0 at weight 0, which the pair
                # test holds bit-identical to -1: a few extra rows through one
                # expert per step, and nothing else.
                neg = topk_ids < 0
                topk_ids = torch.where(neg, torch.zeros_like(topk_ids), topk_ids)
                topk_weights = torch.where(
                    neg, torch.zeros_like(topk_weights), topk_weights)
        return inner_apply(self, output, hidden_states, w1, w2, topk_weights,
                           topk_ids, activation, global_num_experts, None,
                           a1q_scale, a2_scale, workspace13, workspace2,
                           expert_tokens_meta, apply_router_weight_on_input)

    cls.apply = apply
    _DONE = True
    _log(f"armed on {mod.__name__}: EP allowed, wrapper at local E, ids mapped")


class _Hook:
    def find_module(self, fullname, path=None):
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET or _DONE:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _Hook) or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            inner = spec.loader.exec_module

            def exec_module(module, _i=inner):
                _i(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec
        return None


def install() -> None:
    if os.environ.get("DENEB_B12X_EP", "1").strip() in ("0", "false", "no"):
        return
    try:
        if TARGET in sys.modules:
            _patch(sys.modules[TARGET])
        elif not any(isinstance(f, _Hook) for f in sys.meta_path):
            sys.meta_path.insert(0, _Hook())
    except Exception as exc:                                  # noqa: BLE001
        _log(f"install failed: {exc!r}")
