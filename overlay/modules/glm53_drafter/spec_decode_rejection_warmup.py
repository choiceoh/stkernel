# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up spec-decode rejection-sampler Triton kernels.

The rejection sampler kernels (``_compute_local_logits_stats_kernel``,
``_rejection_kernel``, ``_resample_kernel``) are JIT-compiled by Triton on
first use. Without warmup, the first spec-decode request pays a multi-second
compilation cost. This pre-compiles them with dummy data matching the
server's vocab size and speculative config.
"""
# deneb fork (2026-09-01, glm53_dflash_warmup): byte-identical to the
# glm53:v13-b12x image's vllm/model_executor/warmup/
# spec_decode_rejection_warmup.py (preimage sha256
# cd3bce82d185b46fdb92b37292176c7d6ee9488e8bbed4cfde0fe079bf0b6f53) EXCEPT
# the appended _deneb_prepare_dflash_inputs_warmup: the DFlash input-prep
# triton kernel JIT-specializes on its BLOCK_SIZE bucket (min(256,
# next_pow2(max_query_len + num_query_per_req))) and the boot log showed it
# compiling during inference -- a multi-second spike on the first real
# request. This warms every bucket the runtime can hit, before serving.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@torch.inference_mode()
def spec_decode_rejection_warmup(worker: Worker) -> None:
    spec_config = worker.vllm_config.speculative_config
    if spec_config is None:
        return

    # deneb fork: warm the DFlash input-prep kernel across its BLOCK_SIZE
    # buckets before anything can trip a first-use JIT compile in serving.
    try:
        _deneb_prepare_dflash_inputs_warmup(worker)
    except Exception:
        logger.warning(
            "Skipping DFlash input-prep kernel warmup.", exc_info=True)

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    model_config = worker.vllm_config.model_config
    vocab_size = model_config.get_vocab_size()
    num_spec = spec_config.num_speculative_tokens
    if num_spec <= 0 or vocab_size <= 0:
        return

    # Mirror the constexpr-relevant flags the runtime uses.
    rejection_method = getattr(spec_config, "rejection_sample_method", None)
    use_block_verification = rejection_method == "block"
    use_synthetic = rejection_method == "synthetic"

    device = torch.device("cuda")
    num_reqs = 1
    tokens_per_req = num_spec + 1
    num_logits = num_reqs * tokens_per_req

    # Triton JIT-specializes on tensor dtypes. The target logits may be fp32
    # (apply_sampling_params copies to fp32 when processing is needed) or the
    # model dtype (pass-through otherwise), while draft logits are always the
    # model dtype. Warm every (target, draft) combination the runtime can hit.
    model_dtype = model_config.dtype
    warmup_dtype_pairs = {
        (model_dtype, model_dtype),
        (torch.float32, torch.float32),
        (torch.float32, model_dtype),
        (model_dtype, torch.float32),
    }

    logger.info(
        "Warming up spec-decode rejection sampler kernels "
        "(vocab=%d, num_spec=%d, dtype_pairs=%s, block_verify=%s).",
        vocab_size,
        num_spec,
        [(str(t), str(d)) for t, d in warmup_dtype_pairs],
        use_block_verification,
    )
    for tgt_dtype, draft_dtype in warmup_dtype_pairs:
        target_logits = torch.zeros(
            (num_logits, vocab_size), dtype=tgt_dtype, device=device
        )
        draft_logits = torch.zeros(
            (num_reqs, num_spec, vocab_size), dtype=draft_dtype, device=device
        )
        synthetic_rates = (
            torch.full((num_spec,), 0.5, dtype=torch.float32, device=device)
            if use_synthetic
            else None
        )
        try:
            rejection_sample(
                target_logits=target_logits,
                draft_logits=draft_logits,
                draft_sampled=torch.zeros(num_logits, dtype=torch.int64, device=device),
                cu_num_logits=torch.tensor(
                    [0, num_logits], dtype=torch.int32, device=device
                ),
                pos=torch.zeros(num_logits, dtype=torch.int64, device=device),
                idx_mapping=torch.zeros(num_reqs, dtype=torch.int32, device=device),
                expanded_idx_mapping=torch.zeros(
                    num_logits, dtype=torch.int32, device=device
                ),
                expanded_local_pos=torch.arange(
                    num_logits, dtype=torch.int32, device=device
                ),
                temperature=torch.zeros(num_reqs, dtype=torch.float32, device=device),
                seed=torch.full((num_reqs,), 42, dtype=torch.int64, device=device),
                num_speculative_steps=num_spec,
                synthetic_conditional_rates=synthetic_rates,
                use_fp64=False,
                use_block_verification=use_block_verification,
            )
        except Exception:
            logger.warning(
                "Skipping spec-decode rejection sampler warmup.", exc_info=True
            )
            return


_DENEB_WARMUP_ENV = "VLLM_DFLASH_PREP_WARMUP"


def _deneb_warmup_query_lens(num_query_per_req: int, max_num_tokens: int):
    """Query lengths hitting every BLOCK_SIZE bucket the runtime can take.

    BLOCK_SIZE = min(256, next_pow2(max_query_len + num_query_per_req)), so a
    query length of B - num_query_per_req lands exactly on bucket B for each
    power of two B >= num_query_per_req + 1. Pure; extracted for tests.
    """
    import os

    if (os.environ.get(_DENEB_WARMUP_ENV) or "1").strip().lower() in (
            "0", "false", "no", "off"):
        return []
    lens = []
    bucket = 8
    while bucket <= 256:
        qlen = bucket - num_query_per_req
        if 1 <= qlen <= max_num_tokens:
            lens.append(qlen)
        bucket *= 2
    return lens


def _deneb_prepare_dflash_inputs_warmup(worker) -> None:
    """JIT-warm _prepare_dflash_inputs_kernel across its BLOCK_SIZE buckets.

    Calls the production wrapper with a one-request dummy batch (context of
    one token, one sampled token) so every kernel load stays in bounds.
    Scalars use production values because triton specializes ints on ==1 and
    divisibility; the per-request variable is the query length, swept here.
    """
    import os
    import types

    from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # noqa: F401
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        prepare_dflash_inputs,
    )

    vc = worker.vllm_config
    spec_config = vc.speculative_config
    dflash_config = (
        getattr(spec_config.draft_model_config.hf_config,
                "dflash_config", None) or {})
    if "mask_token_id" not in dflash_config:
        return

    num_spec = spec_config.num_speculative_tokens
    num_query_per_req = num_spec + 1
    max_num_reqs = vc.scheduler_config.max_num_seqs
    max_num_tokens = vc.scheduler_config.max_num_batched_tokens
    max_model_len = vc.model_config.max_model_len
    block_size = vc.cache_config.block_size
    mask_id = int(dflash_config["mask_token_id"])

    device = torch.device("cuda")
    i64, i32, f32 = torch.int64, torch.int32, torch.float32
    input_buffers = types.SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=i64, device=device),
        positions=torch.zeros(max_num_tokens, dtype=i64, device=device),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=i32, device=device),
        seq_lens=torch.zeros(max_num_reqs, dtype=i32, device=device),
    )
    input_batch = types.SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=torch.zeros(max_num_reqs, dtype=i64,
                                         device=device),
        positions=torch.zeros(max_num_tokens, dtype=i64, device=device),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=i32,
                                    device=device),
        idx_mapping=torch.zeros(max_num_reqs, dtype=i32, device=device),
    )
    max_blocks = max(1, max_model_len // block_size + 1)
    block_table = torch.zeros(max_num_reqs, max_blocks, dtype=i32,
                              device=device)

    lens = _deneb_warmup_query_lens(num_query_per_req, max_num_tokens)
    for qlen in lens:
        input_batch.num_scheduled_tokens.fill_(qlen)
        # one request: context of one token, one sampled token -- keeps every
        # kernel load in bounds (a zero context would index positions[-1])
        input_batch.query_start_loc[1] = 1
        input_batch.positions[0] = 0
        prepare_dflash_inputs(
            input_buffers=input_buffers,
            query_slot_mapping=torch.zeros(max_num_tokens, dtype=i64,
                                           device=device),
            context_positions=torch.zeros(max_num_tokens, dtype=i64,
                                           device=device),
            context_slot_mapping=torch.zeros(max_num_tokens, dtype=i64,
                                             device=device),
            sample_indices=torch.zeros(max_num_reqs * max(num_spec, 1),
                                       dtype=i64, device=device),
            sample_pos=torch.zeros(max_num_reqs * max(num_spec, 1),
                                   dtype=i64, device=device),
            sample_idx_mapping=torch.zeros(max_num_reqs * max(num_spec, 1),
                                           dtype=i32, device=device),
            temperature=torch.zeros(1, dtype=f32, device=device),
            seeds=torch.full((1,), 42, dtype=i64, device=device),
            input_batch=input_batch,
            num_sampled=torch.ones(1, dtype=i64, device=device),
            num_rejected=torch.zeros(1, dtype=i64, device=device),
            last_sampled=torch.zeros(max_num_reqs, dtype=i64, device=device),
            next_prefill_tokens=torch.zeros(max_num_reqs, dtype=i64,
                                            device=device),
            input_temperature=torch.zeros(max_num_reqs, dtype=f32,
                                          device=device),
            input_seeds=torch.zeros(max_num_reqs, dtype=i64, device=device),
            block_table=block_table,
            block_size=block_size,
            parallel_drafting_token_id=mask_id,
            num_query_per_req=num_query_per_req,
            num_speculative_steps=num_spec,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            max_model_len=max_model_len,
            sample_from_anchor=False,
        )
    logger.info(
        "Warmed DFlash input-prep kernel over %d BLOCK_SIZE buckets "
        "(num_query_per_req=%d).", len(lens), num_query_per_req)
