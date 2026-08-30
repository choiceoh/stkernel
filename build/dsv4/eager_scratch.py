# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager-break scratch pool — deneb fork port of upstream PR #49236, slimmed
for the TP4 GB10 stack.

Upstream plumbs one pool through nvidia/model.py into every attention layer and
also pools the padded-q output. This port diverges deliberately:
- the padded-q bucket is DROPPED: it requires the `_insert_out` variant of the
  fused qnorm/rope/insert CUDA op, which this image's `_C` library predates;
- the FP4 indexer bucket is DROPPED (this stack runs the FP8 indexer,
  use_fp4_kv=False) and replaced with an FP8 bucket sized for it;
- instead of model.py plumbing, the pool is a lazily-created per-device
  singleton (`get_eager_scratch_pool`), so the non-overlaid image model.py
  stays untouched. All 43 layers share one pool — eager bodies are serialized,
  and per-size views are cached so cudagraph capture sees stable addresses.

Every consumer passes buffers optionally; `None` falls back to the original
per-call allocation, so any single file can be rolled back independently.
"""
from math import prod

import torch

from vllm.utils.math_utils import round_up


class DeepseekV4EagerScratchPool:
    """Model-wide scratch reused inside the attention eager break."""

    _ALIGNMENT = 256

    def __init__(
        self,
        max_num_tokens: int,
        q_head_dim: int,
        index_q_heads: int,
        index_q_head_dim: int,
        index_topk: int,
        device: torch.device | str,
    ) -> None:
        self.max_num_tokens = max_num_tokens
        self.index_topk = index_topk

        # FP8 indexer outputs (C4A layers): quantized index_q + fp32 weights.
        fp8_specs = (
            ((max_num_tokens, index_q_heads, index_q_head_dim), torch.uint8),
            ((max_num_tokens, index_q_heads), torch.float32),
        )
        # Global top-k mapping (C4A decode/prefill globalization).
        global_specs = (
            ((max_num_tokens, index_topk), torch.int32),
            ((max_num_tokens,), torch.int32),
        )
        # C128 two-pass compressor intermediate.
        compressor_specs = (((max_num_tokens, q_head_dim), torch.float32),)

        # Layout: [FP8 indexer bucket][global-topk bucket]. The FP8 scratch is
        # written on the indexer aux stream while the default stream later runs
        # the globalization, so those two must NOT alias. The compressor bucket
        # is C128-only (those layers have no indexer) and overlays the FP8
        # region.
        fp8_bytes = self._packed_size(fp8_specs)
        assert self._packed_size(compressor_specs) <= fp8_bytes
        storage = torch.empty(
            fp8_bytes + self._packed_size(global_specs),
            dtype=torch.uint8,
            device=device,
        )

        fp8_q, fp8_w = self._views(storage, fp8_specs, base=0)
        self._fp8_template = (fp8_q.view(torch.float8_e4m3fn), fp8_w)
        self._fp8_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        g_idx, g_len = self._views(storage, global_specs, base=fp8_bytes)
        self._global_template = (g_idx, g_len)
        self._global_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        (comp,) = self._views(storage, compressor_specs, base=0)
        self._compressor_template = comp
        self._compressor_outputs: dict[int, torch.Tensor] = {}
        self._storage = storage

    @classmethod
    def _packed_size(
        cls, specs: tuple[tuple[tuple[int, ...], torch.dtype], ...]
    ) -> int:
        offset = 0
        for shape, dtype in specs:
            offset = round_up(offset, cls._ALIGNMENT) + prod(shape) * dtype.itemsize
        return round_up(offset, cls._ALIGNMENT)

    @classmethod
    def _views(
        cls,
        storage: torch.Tensor,
        specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
        base: int = 0,
    ) -> list[torch.Tensor]:
        offset = base
        views = []
        for shape, dtype in specs:
            offset = round_up(offset, cls._ALIGNMENT)
            num_bytes = prod(shape) * dtype.itemsize
            views.append(
                storage[offset : offset + num_bytes].view(dtype).view(shape)
            )
            offset += num_bytes
        return views

    def indexer_q_outputs(
        self, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(index_q_fp8, index_weights_out) for fused_indexer_q_rope_quant."""
        output = self._fp8_outputs.get(num_tokens)
        if output is None:
            q, w = self._fp8_template
            output = (q[:num_tokens], w[:num_tokens])
            self._fp8_outputs[num_tokens] = output
        return output

    def compressor_scratch(self, num_tokens: int) -> torch.Tensor:
        output = self._compressor_outputs.get(num_tokens)
        if output is None:
            output = self._compressor_template[:num_tokens]
            self._compressor_outputs[num_tokens] = output
        return output

    def global_topk_outputs(
        self, topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, topk = topk_indices.shape
        assert topk == self.index_topk
        output = self._global_outputs.get(num_tokens)
        if output is None:
            indices, lens = self._global_template
            output = (indices[:num_tokens], lens[:num_tokens])
            self._global_outputs[num_tokens] = output
        return output


_POOLS: dict[torch.device, DeepseekV4EagerScratchPool] = {}


def get_eager_scratch_pool(
    max_num_tokens: int,
    q_head_dim: int,
    index_q_heads: int,
    index_q_head_dim: int,
    index_topk: int,
    device: torch.device,
) -> DeepseekV4EagerScratchPool:
    """Per-device singleton so all layers share one pool without model.py
    plumbing. First caller's sizes win; later callers assert compatibility."""
    pool = _POOLS.get(device)
    if pool is None:
        pool = DeepseekV4EagerScratchPool(
            max_num_tokens,
            q_head_dim,
            index_q_heads,
            index_q_head_dim,
            index_topk,
            device,
        )
        _POOLS[device] = pool
    else:
        assert pool.max_num_tokens >= max_num_tokens
        assert pool.index_topk == index_topk
    return pool
