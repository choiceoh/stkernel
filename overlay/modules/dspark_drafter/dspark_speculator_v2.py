# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.

Opt-in two-pass refinement (``VLLM_DSPARK_REFINE_PASS``, default off):
the DSpark paper (arXiv:2607.05147) identifies "rapid acceptance decay" as the
parallel-drafter bottleneck — base logits at noise positions never see the
tokens actually sampled before them, and only the first-order Markov bias
injects intra-block dependency. The refinement pass writes the pass-1 draft
tokens into the noise slots, re-runs the parallel backbone (context KV is
reused; query KV slots are simply overwritten), and re-runs the sequential
sampling. Notes:

  * Same Gumbel keys on both passes — REQUIRED, not merely safe. This fork's
    verifier shares the draft's per-position (seed, pos) Gumbel noise, so a
    position is accepted iff argmax(log q + g) == argmax(log p + g). The
    emitted token is always the verifier's own argmax over the TARGET
    distribution, so q (and therefore this whole feature) cannot change the
    output distribution — it only moves the match rate. Re-keying pass 2
    would break the coupling and collapse acceptance.
  * Jacobi semantics: one refinement iteration; if pass 2 re-samples an
    earlier position differently, later positions were conditioned on the
    stale pass-1 token. Still strictly more information than noise inputs.
  * OOD caveat: the backbone is trained with mask/noise inputs at non-anchor
    positions (anchor-bounded packing per the paper), so real-token inputs
    are out of the training distribution — acceptance may not improve.
    Default-off knob; adopt only on measured bench-dec acceptance + 9/9.
  * Cost: one extra 3-layer backbone forward + draft-head logits per draft
    step (roughly doubles the draft segment of the captured graph).
"""

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils_v2 import load_dspark_model

logger = init_logger(__name__)


def _parse_draft_topk(raw: str, vocab_size: int) -> int | None:
    """Parse the opt-in top-k, returning ``None`` for the disabled path."""
    value = raw.strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return None
    try:
        topk = int(value)
    except ValueError as exc:
        raise ValueError(
            "VLLM_DSPARK_DRAFT_TOPK must be an integer or 0, "
            f"got {raw!r}"
        ) from exc
    if not 1 <= topk <= vocab_size:
        raise ValueError(
            "VLLM_DSPARK_DRAFT_TOPK must be in "
            f"[1, {vocab_size}], got {topk}"
        )
    return topk


def _parse_refine_pass(raw: str) -> bool:
    """Parse the opt-in two-pass refinement knob (fail closed on garbage)."""
    value = raw.strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"VLLM_DSPARK_REFINE_PASS must be a boolean value, got {raw!r}"
    )


def _refine_feedback_indices(max_num_reqs: int, num_query_per_req: int) -> list[int]:
    """Flat input-buffer indices that receive pass-1 draft tokens.

    Query offset 0 is the anchor (the last verified token — never rewritten);
    offset j>=1 is a noise slot whose true autoregressive input is the token
    drafted AT offset j-1 (offset k predicts position query_pos(k)+1, i.e. the
    input of offset k+1). Flattening is row-major by request so element m of
    ``draft_tokens[:num_reqs, :N-1].reshape(-1)`` lands at index m here.
    """
    return [
        r * num_query_per_req + j
        for r in range(max_num_reqs)
        for j in range(1, num_query_per_req)
    ]


_SPEC_WARMUP_ENV = "VLLM_DSV4_SPEC_WARMUP"


def _warmup_query_lens(num_query_per_req: int, max_num_tokens: int) -> list[int]:
    """Return one target-query length for every input-prep JIT bucket."""
    if os.getenv(_SPEC_WARMUP_ENV, "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return []
    lens = []
    bucket = 8
    while bucket <= 256:
        qlen = bucket - num_query_per_req
        if 1 <= qlen <= max_num_tokens:
            lens.append(qlen)
        bucket *= 2
    return lens


def _warmup_rejection_sampler(speculator: "DSparkSpeculator") -> int:
    """Compile the rejection-sampler dtype signatures DSpark can serve."""
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    vocab_size = int(speculator.vllm_config.model_config.get_vocab_size())
    num_spec = speculator.num_speculative_steps
    if vocab_size <= 0 or num_spec <= 0:
        return 0

    model_dtype = speculator.dtype
    dtype_pairs = list(
        dict.fromkeys(
            (
                (model_dtype, model_dtype),
                (torch.float32, torch.float32),
                (torch.float32, model_dtype),
                (model_dtype, torch.float32),
            )
        )
    )
    num_logits = num_spec + 1
    for target_dtype, draft_dtype in dtype_pairs:
        target_logits = torch.zeros(
            num_logits,
            vocab_size,
            dtype=target_dtype,
            device=speculator.device,
        )
        draft_logits = torch.zeros(
            1,
            num_spec,
            vocab_size,
            dtype=draft_dtype,
            device=speculator.device,
        )
        rejection_sample(
            target_logits=target_logits,
            draft_logits=draft_logits,
            draft_sampled=torch.zeros(
                num_logits, dtype=torch.int64, device=speculator.device
            ),
            cu_num_logits=torch.tensor(
                [0, num_logits], dtype=torch.int32, device=speculator.device
            ),
            pos=torch.arange(
                num_logits, dtype=torch.int64, device=speculator.device
            ),
            idx_mapping=torch.zeros(
                1, dtype=torch.int32, device=speculator.device
            ),
            expanded_idx_mapping=torch.zeros(
                num_logits, dtype=torch.int32, device=speculator.device
            ),
            expanded_local_pos=torch.arange(
                num_logits, dtype=torch.int32, device=speculator.device
            ),
            temperature=torch.full(
                (1,), 0.8, dtype=torch.float32, device=speculator.device
            ),
            seed=torch.full(
                (1,), 42, dtype=torch.int64, device=speculator.device
            ),
            num_speculative_steps=num_spec,
        )
        del target_logits, draft_logits
    return len(dtype_pairs)


def _warmup_input_prep(speculator: "DSparkSpeculator") -> int:
    """Compile every BLOCK_SIZE specialization of DSpark input preparation."""
    import types

    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        prepare_dflash_inputs,
    )

    lens = _warmup_query_lens(
        speculator.num_query_per_req, speculator.max_num_tokens
    )
    if not lens:
        return 0

    device = speculator.device
    max_reqs = speculator.max_num_reqs
    max_tokens = speculator.max_num_tokens
    num_spec = speculator.num_speculative_steps
    block_size = speculator.draft_block_size
    max_blocks = max(1, speculator.max_model_len // block_size + 1)
    i64, i32 = torch.int64, torch.int32

    input_buffers = types.SimpleNamespace(
        input_ids=torch.zeros(max_tokens, dtype=i64, device=device),
        positions=torch.zeros(max_tokens, dtype=i64, device=device),
        query_start_loc=torch.zeros(max_reqs + 1, dtype=i32, device=device),
        seq_lens=torch.zeros(max_reqs, dtype=i32, device=device),
    )
    input_batch = types.SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=torch.zeros(max_reqs, dtype=i64, device=device),
        positions=torch.zeros(max_tokens, dtype=i64, device=device),
        query_start_loc=torch.zeros(max_reqs + 1, dtype=i32, device=device),
        idx_mapping=torch.zeros(max_reqs, dtype=i32, device=device),
    )
    query_slot_mapping = torch.zeros(max_tokens, dtype=i64, device=device)
    context_positions = torch.zeros(max_tokens, dtype=i64, device=device)
    context_slot_mapping = torch.zeros(max_tokens, dtype=i64, device=device)
    sample_count = max_reqs * max(num_spec, 1)
    sample_indices = torch.zeros(sample_count, dtype=i64, device=device)
    sample_pos = torch.zeros(sample_count, dtype=i64, device=device)
    sample_idx_mapping = torch.zeros(sample_count, dtype=i32, device=device)
    block_table = torch.zeros(
        max_reqs, max_blocks, dtype=i32, device=device
    )
    num_sampled = torch.ones(1, dtype=i64, device=device)
    num_rejected = torch.zeros(1, dtype=i64, device=device)
    last_sampled = torch.zeros(max_reqs, dtype=i64, device=device)
    next_prefill_tokens = torch.zeros(max_reqs, dtype=i64, device=device)

    # Keep one valid context position; only num_scheduled_tokens is varied to
    # select the constexpr bucket. This avoids touching an uninitialised tail.
    input_batch.query_start_loc[1] = 1
    for qlen in lens:
        input_batch.num_scheduled_tokens.fill_(qlen)
        prepare_dflash_inputs(
            input_buffers=input_buffers,
            query_slot_mapping=query_slot_mapping,
            context_positions=context_positions,
            context_slot_mapping=context_slot_mapping,
            sample_indices=sample_indices,
            sample_pos=sample_pos,
            sample_idx_mapping=sample_idx_mapping,
            input_batch=input_batch,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            block_table=block_table,
            block_size=block_size,
            parallel_drafting_token_id=speculator.parallel_drafting_token_id,
            num_query_per_req=speculator.num_query_per_req,
            num_speculative_steps=num_spec,
            max_num_reqs=max_reqs,
            max_num_tokens=max_tokens,
            max_model_len=speculator.max_model_len,
            sample_from_anchor=True,
        )
    return len(lens)


def _warmup_dspark_spec_decode(speculator: "DSparkSpeculator") -> None:
    """Best-effort JIT warmup after the draft block table is initialised."""
    if getattr(speculator, "_deneb_spec_warmup_done", False):
        return
    speculator._deneb_spec_warmup_done = True
    if not _warmup_query_lens(
        speculator.num_query_per_req, speculator.max_num_tokens
    ):
        logger.info("DSpark spec-decode JIT warmup disabled by %s.", _SPEC_WARMUP_ENV)
        return

    rejection_count = 0
    prep_count = 0
    try:
        rejection_count = _warmup_rejection_sampler(speculator)
    except Exception:
        logger.warning("Skipping DSpark rejection-sampler warmup.", exc_info=True)
    try:
        prep_count = _warmup_input_prep(speculator)
    except Exception:
        logger.warning("Skipping DSpark input-prep warmup.", exc_info=True)
    torch.cuda.synchronize(speculator.device)
    logger.info(
        "DSpark spec-decode JIT warmup complete: rejection dtype pairs=%d, "
        "input-prep buckets=%d.",
        rejection_count,
        prep_count,
    )


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            raise RuntimeError("DSpark requires speculative_config")
        draft_vocab_size = int(
            speculative_config.draft_model_config.hf_config.vocab_size
        )
        self._draft_topk = _parse_draft_topk(
            os.getenv("VLLM_DSPARK_DRAFT_TOPK", ""), draft_vocab_size
        )
        self._refine_pass = _parse_refine_pass(
            os.getenv("VLLM_DSPARK_REFINE_PASS", "")
        )

        # DFlash initialization loads the draft model through our override, so
        # the opt-in value must exist before entering super().__init__().
        super().__init__(vllm_config, device)

        # Anchor-first: N query tokens per request (anchor + N-1 noise), not 1+N.
        self.num_query_per_req = self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self.dflash_causal = False

        # The anchor query position is itself a prediction (see module docstring).
        self.sample_from_anchor = True

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        self._refine_input_idx: torch.Tensor | None = None
        if self._refine_pass:
            if self.num_speculative_steps < 2:
                raise RuntimeError(
                    "VLLM_DSPARK_REFINE_PASS=1 requires num_speculative_tokens"
                    " >= 2 (there are no noise slots to refine at "
                    f"{self.num_speculative_steps})"
                )
            self._refine_input_idx = torch.tensor(
                _refine_feedback_indices(
                    self.max_num_reqs, self.num_query_per_req
                ),
                dtype=torch.int64,
                device=device,
            )
            logger.info_once(
                "DSpark(v2) two-pass draft refinement enabled: pass-1 tokens "
                "replace the %d noise slots per request, same Gumbel keys.",
                self.num_query_per_req - 1,
            )

    def set_attn(self, model_state, kv_cache_config, block_tables) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)
        _warmup_dspark_spec_decode(self)

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        if self._draft_topk is None:
            return model

        draft_vocab_size = int(model.config.vocab_size)
        target_vocab_size = int(
            self.vllm_config.model_config.hf_config.vocab_size
        )
        if target_vocab_size != draft_vocab_size:
            raise RuntimeError(
                "VLLM_DSPARK_DRAFT_TOPK requires identical target/draft "
                f"vocabularies; target={target_vocab_size}, "
                f"draft={draft_vocab_size}"
            )

        markov_head = getattr(getattr(model, "model", None), "markov_head", None)
        markov_w2 = getattr(markov_head, "markov_w2", None)
        markov_weight = getattr(markov_w2, "weight", None)
        markov_rank = int(model.config.dspark_markov_rank)
        expected_w2_shape = (draft_vocab_size, markov_rank)
        if (
            markov_head is None
            or not getattr(markov_head, "_replicate_w2", False)
            or markov_weight is None
            or tuple(markov_weight.shape) != expected_w2_shape
        ):
            actual_shape = (
                None if markov_weight is None else tuple(markov_weight.shape)
            )
            raise RuntimeError(
                "VLLM_DSPARK_DRAFT_TOPK requires full replicated Markov W2; "
                f"expected={expected_w2_shape}, actual={actual_shape}, "
                f"replicated={getattr(markov_head, '_replicate_w2', False)}"
            )

        logits_processor = getattr(model, "logits_processor", None)
        if (
            get_tensor_model_parallel_world_size() > 1
            and not getattr(logits_processor, "use_all_gather", False)
        ):
            raise RuntimeError(
                "VLLM_DSPARK_DRAFT_TOPK with TP>1 requires full-vocabulary "
                "logits on every rank (LogitsProcessor.use_all_gather=True)"
            )
        return model

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_logits(sample_hidden)
        if self._draft_topk is not None and (
            base_logits is None
            or base_logits.ndim != 2
            or base_logits.shape[0] != num_sample
            or base_logits.shape[1] != int(self.model.config.vocab_size)
            or self._draft_topk > base_logits.shape[1]
        ):
            actual_shape = None if base_logits is None else tuple(base_logits.shape)
            raise RuntimeError(
                "invalid full-vocabulary DSpark base logits for top-k: "
                f"expected=({num_sample}, {self.model.config.vocab_size}), "
                f"actual={actual_shape}, topk={self._draft_topk}"
            )
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        if self._draft_topk is not None:
            base_values, token_indices = torch.topk(
                base_logits, self._draft_topk, dim=-1, sorted=False
            )
            base_logits.fill_(-float("inf"))

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self._draft_topk is None:
                bias = self.model.markov_bias(markov_embed)
                logits_i = base_logits[:, i] + bias
            else:
                logits_i = self.model.apply_markov_bias_gathered(
                    markov_embed,
                    base_logits[:, i],
                    base_values[:, i],
                    token_indices[:, i],
                )
            if self.draft_logits is not None:
                # sample_pos is the predicted token's position Q; the target
                # verifies it with the predecessor's Gumbel key (Q-1). Pass Q-1.
                draft_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                draft_i = logits_i.argmax(dim=-1)
            self.draft_tokens[:num_reqs, i] = draft_i
            prev = draft_i

    def _feed_back_draft_tokens(self, num_reqs: int) -> None:
        """Write the pass-1 draft tokens into the per-request noise slots.

        Query offset j >= 1 receives the token drafted at offset j-1 (offset k
        predicts position query_pos+1 = the input of offset k+1); anchors at
        offset 0 are never touched. The prepare-inputs kernel rewrites every
        query slot on the next step, so this in-place, in-graph mutation of
        the shared input buffer cannot leak across steps. Padded CG-replay
        requests write junk tokens into their own (PAD-slot-mapped, unsampled)
        query rows — harmless by the same argument as their pass-1 rows.
        """
        assert self._refine_input_idx is not None
        n_feedback = self.num_query_per_req - 1
        idx = self._refine_input_idx[: num_reqs * n_feedback]
        input_ids = self.input_buffers.input_ids
        src = self.draft_tokens[:num_reqs, :n_feedback].reshape(-1)
        input_ids.index_copy_(0, idx, src.to(dtype=input_ids.dtype))

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
        if not self._refine_pass:
            return
        # Refinement pass (module docstring): identical metadata and slot
        # mappings — query KV slots are overwritten with the refined inputs'
        # KV, context KV is untouched. Re-running _sample_sequential replays
        # the SAME (seed, pos) Gumbel keys, which the verifier-side coupling
        # requires, and overwrites draft_tokens plus the recorded proposal q
        # in draft_logits with the pass-2 values the verifier will consume.
        self._feed_back_draft_tokens(num_reqs)
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
