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
  * the `B12xMoEWrapper` is built for the LOCAL expert count (the weights' E)
  * `apply()`: `topk_ids = expert_map[topk_ids]` when a map is given

Nothing else in the class changes. `DENEB_B12X_EP=0` leaves the stock class
alone, in which case an EP boot fails the oracle exactly as before.
"""

from __future__ import annotations

import os
import sys

TARGET = "vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe"
_DONE = False


def _log(msg: str) -> None:
    sys.stderr.write(f"[qwen38-b12x-ep] {msg}\n")
    sys.stderr.flush()


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

        # The weights on this rank are [num_local_experts, ...]; the kernel's
        # E is that, and the ids reaching it are local (or -1) by apply().
        self._wrapper = B12xMoEWrapper(
            num_experts=self.num_local_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            max_num_tokens=self.max_num_tokens,
            num_local_experts=self.num_local_experts,
            activation=self._activation_str,
        )

    cls._ensure_wrapper = _ensure_wrapper

    inner_apply = cls.apply

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        if expert_map is not None:
            # global expert id -> this rank's slot, or -1: vLLM's unrouted
            # sentinel, which the guarded kernel skips.
            topk_ids = expert_map[topk_ids.long()]
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
