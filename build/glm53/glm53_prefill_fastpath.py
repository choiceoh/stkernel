# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 prefill metadata warm-up (VLLM_GLM53_PREFILL_METADATA_WARMUP).

Compiles the exact pooled-prefill metadata launch signatures at boot, before
the first request, so the first prefill pays no Triton JIT for them (33차).
The fused K+gate indexer path and the SM121 dense-prefill cache-only path
this module once carried (VLLM_GLM53_FUSED_K_GATE, VLLM_GLM53_SM121_MLA_PREFILL)
were sunset in 34차 §8: opt-in, never judged, and MK-MLA v5 routes the prefill
rows the dense arm targeted.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger


logger = init_logger(__name__)

_GLM53_PREFILL_METADATA_WARMUP_ENV = "VLLM_GLM53_PREFILL_METADATA_WARMUP"


def warm_glm53_prefill_metadata_runtime(model: torch.nn.Module) -> int:
    """Populate Triton's real runtime cache for GLM's pooled metadata key.

    ``VllmJitKernel.warmup`` compiles synthetic pointer variants, but Triton's
    launch cache still distinguishes aligned and sliced int32 pointers.  Run
    one valid row for both variants of GLM's pooled compression ratio so
    the first user prefill cannot become the compilation request.  This is
    best-effort and independent of the dense-prefill experiment.
    """
    if os.environ.get(_GLM53_PREFILL_METADATA_WARMUP_ENV, "1") != "1":
        return 0
    if not torch.cuda.is_available():
        return 0

    vllm_config = getattr(model, "vllm_config", None)
    hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if (
        getattr(hf_config, "model_type", None) != "glm5_next_text"
        or getattr(hf_config, "index_kpool", None) != 4
        or getattr(parallel_config, "tensor_parallel_size", None) != 4
    ):
        return 0

    parameter = next(model.parameters(), None)
    if parameter is None or parameter.device.type != "cuda":
        return 0

    # Resolve dynamically because another image profile owns a different
    # overlay for the same vLLM module; only the GLM profile guarantees this
    # private kernel symbol.
    import importlib

    indexer_module = importlib.import_module(
        "vllm.v1.attention.backends.mla.indexer"
    )
    metadata_kernel = getattr(
        indexer_module, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL"
    )

    device = parameter.device
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_compressed_seq_lens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    token_to_seq = torch.empty(1, dtype=torch.int32, device=device)
    cu_seqlen_ks = torch.empty(1, dtype=torch.int32, device=device)
    cu_seqlen_ke = torch.empty(1, dtype=torch.int32, device=device)
    aligned = torch.tensor([4], dtype=torch.int32, device=device)
    unaligned_storage = torch.tensor([-1, 4], dtype=torch.int32, device=device)

    launches = 0
    for uncompressed_seq_lens in (aligned, unaligned_storage[1:]):
        metadata_kernel(
            query_start_loc,
            uncompressed_seq_lens,
            cu_compressed_seq_lens,
            cu_compressed_seq_lens,
            token_to_seq,
            cu_seqlen_ks,
            cu_seqlen_ke,
            0,
            1,
            0,
            1,
            1,
            num_reqs=1,
            COMPRESS_RATIO=4,
        )
        launches += 1
    torch.cuda.synchronize(device)
    return launches

