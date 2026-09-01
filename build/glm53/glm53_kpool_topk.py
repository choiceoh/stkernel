# SPDX-License-Identifier: Apache-2.0
"""Optional fused radix top-k/expand/tail path for GLM-5.3 KPool.

The CUDA source is adapted from SGLang's Apache-2.0
``kpool_topk_transform`` kernel.  Keep compilation and execution fail-closed:
the stock vLLM top-k and Triton expansion remain the authoritative fallback.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

logger = logging.getLogger("vllm.glm53.kpool_topk")

_ENV = "VLLM_GLM53_KPOOL_FUSED_TOPK"
_SOURCE = os.path.join(os.path.dirname(__file__), "glm53_kpool_topk.cu")
_LOCK = threading.Lock()
_EXT = None
_FAILED = False


def _enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _load_extension():
    global _EXT, _FAILED
    if _EXT is not None or _FAILED or not _enabled():
        return _EXT
    with _LOCK:
        if _EXT is not None or _FAILED:
            return _EXT
        try:
            from torch.utils.cpp_extension import load

            build_directory = os.environ.get(
                "VLLM_GLM53_KPOOL_BUILD_DIR",
                "/tmp/glm53_kpool_topk_build",
            )
            os.makedirs(build_directory, exist_ok=True)
            _EXT = load(
                name="glm53_kpool_topk_ext",
                sources=[_SOURCE],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_121a"],
                build_directory=build_directory,
                verbose=False,
            )
        except Exception:
            _FAILED = True
            logger.warning(
                "GLM fused KPool top-k unavailable; using stock path.",
                exc_info=True,
            )
    return _EXT


def prepare_glm53_kpool_topk() -> bool:
    """Compile at boot when explicitly armed, never on the first request."""
    armed = _load_extension() is not None
    # Say it either way. The failure path already warns, but a silent success
    # is indistinguishable from a knob that never parsed -- and this lane has
    # measured that mistake more than once.
    if _enabled():
        logger.info("glm53 kpool fused top-k: %s",
                    "ARMED" if armed else "requested but NOT armed")
    return armed


def glm53_kpool_topk_expand_tail(
    scores: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Select 512 pools, expand to 2048 tokens, and append a <=3 tail."""
    if not _enabled():
        return False
    if not (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and scores.stride(1) == 1
        and row_starts.is_cuda
        and row_starts.dtype == torch.int32
        and row_starts.ndim == 1
        and row_ends.is_cuda
        and row_ends.dtype == torch.int32
        and row_ends.shape == row_starts.shape
        and seq_lens.is_cuda
        and seq_lens.dtype == torch.int32
        and seq_lens.shape == row_starts.shape
        and output.is_cuda
        and output.dtype == torch.int32
        and output.ndim == 2
        and output.shape == (scores.shape[0], 2051)
        and output.stride(1) == 1
        and scores.shape[0] == row_starts.numel()
    ):
        return False
    extension = _load_extension()
    if extension is None:
        return False
    try:
        extension.run(scores, row_starts, row_ends, seq_lens, output)
    except Exception:
        logger.warning(
            "GLM fused KPool top-k launch failed; using stock path.",
            exc_info=True,
        )
        return False
    return True
