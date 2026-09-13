# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
from functools import lru_cache

import torch

import triton

from .utils import tensor_cache


@lru_cache(maxsize=8)
def single_sequence_bounds(tokens: int, device: torch.device) -> torch.Tensor:
    """Immutable bounds reused across layers, including the tensor-cache identity.

    Recreating [0, T] on every layer misses prepare_chunk_indices' cache and
    forces its GPU-to-CPU tolist() every time. Keep only eight recent shapes.
    """
    return torch.tensor([0, tokens], dtype=torch.int32, device=device)


@lru_cache(maxsize=8)
def chunk_mark_indices(marks: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Read-only chunk marks shared by all KDA layers at these boundaries."""
    return torch.tensor(marks, dtype=torch.int32, device=device)


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
