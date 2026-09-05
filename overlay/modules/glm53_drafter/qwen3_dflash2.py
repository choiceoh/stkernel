# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
# deneb fork: the drafter's only GEMM into the head goes through
# _apply_head, so an fp8 copy is swapped in there and the top-k reduction
# above it is left alone. See overlay/modules/fp8_lm_head.
from vllm.model_executor.layers.fp8_lm_head import (
    Fp8HeadLogitsProcessor,
    decodable_vocab_size,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix


logger = init_logger(__name__)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    tap_valid: torch.Tensor,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += (
            coefficients[:, tap]
            * shifted
            * tap_valid[:, tap].view(-1, 1, 1)
        )
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        max_num_tokens: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        # Row position within a speculative block depends only on the static
        # flattened row index. Precompute every tap's boundary mask once instead
        # of allocating arange + compare tensors in all four grouped-conv calls
        # per layer/step. Non-persistent: this is derived runtime state, not a
        # checkpoint weight, and register_buffer moves it with the module.
        position = torch.arange(max_num_tokens, dtype=torch.int32)
        if block_size & (block_size - 1) == 0:
            position.bitwise_and_(block_size - 1)
        else:
            position.remainder_(block_size)
        tap_ids = torch.arange(taps, dtype=torch.int32)
        self.register_buffer(
            "_tap_valid",
            position[:, None] >= tap_ids[None, :],
            persistent=False,
        )
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self._tap_valid[: hidden_states.shape[0]],
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    # Keep the selector's final bilinear score in fp32.  The codebooks and
    # projected hidden state are bf16, but returning the einsum in bf16 rounds
    # distinct rank-256 path scores onto the same value before the selector
    # walk sees them.  A tie here can pick the wrong successor even though the
    # trained codebooks assigned it a lower score.  Cast before the multiply as
    # well as the reduction so neither the modulation nor the final score loses
    # that ordering.  This tensor is tiny for DFlash2 (B x 7 x 16 x 16), and
    # _selector_walk already stores/consumes its scores as fp32.
    successors = successor_table[candidate_ids].float()
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids].float()
    return unary_logits.float()[:, :, None] + torch.einsum(
        "blpr,blcr->blpc",
        predecessors * hidden.float()[:, :, None],
        successors,
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )



_EARLY_FC = None


def _early_fc_take():
    """`take_early_fc` of glm53_dflash_early_fc, resolved once; None when that
    module is not mounted (stock projection every step)."""
    global _EARLY_FC
    if _EARLY_FC is None:
        try:
            from vllm.models.glm5next.nvidia.glm53_dflash_early_fc import (
                take_early_fc,
            )

            _EARLY_FC = take_early_fc
        except Exception:
            _EARLY_FC = False
    return _EARLY_FC or None


_OSAR = None


def _osar_shim():
    """The one-shot AR shim, resolved once; None when it is not mounted or
    predates the prefetch hints (the same resolver the target model uses)."""
    global _OSAR
    if _OSAR is None:
        try:
            from vllm.distributed.device_communicators import (
                dsv4_oneshot_shim as shim,
            )

            _OSAR = shim if hasattr(shim, "begin_forward") else False
        except Exception:
            _OSAR = False
    return _OSAR or None


def dflash2_selector_load_verdict(stats):
    """Did the path selector's weights actually load? Pure predicate.

    `stats` is a mapping name -> (finite_fraction, abs_max, abs_mean). The
    codebooks are declared with `torch.empty`, so a name the loader never
    matched keeps whatever was in that memory -- and DFlash2's whole claim
    rests on them: the blog reports ~86 pct conditional acceptance held across
    ALL draft positions, where an unweighted selector decays hard toward the
    block end. This fleet measures 78.6 / 57.5 / 35.1 / 22.7 / 16.8 / 10.3 /
    6.7 pct, i.e. exactly the suffix decay the selector exists to remove, and a
    mean accept length of 2.4-3.3 against a designed 4.8.

    Returns (ok, reasons). Uninitialised memory shows up as non-finite values
    or an absurd magnitude; an all-zero tensor means the name matched nothing
    and something zeroed it.
    """
    reasons = []
    for name, (finite, amax, amean) in sorted(stats.items()):
        if finite < 1.0:
            reasons.append(f"{name}: {100 * (1 - finite):.2f}% non-finite")
        elif amax == 0.0:
            reasons.append(f"{name}: all zero")
        elif amax > 1e4 or amean > 1e3:
            reasons.append(f"{name}: |max|={amax:.3g} |mean|={amean:.3g} "
                           "-- not a trained magnitude")
    return (not reasons), tuple(reasons)


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale

    # deneb fork (34차, EXP-24 item 6): the context K/V projection of
    # `precompute_and_store_context_kv` -- one fused bf16 GEMM over the five
    # layers' k/v rows ([L x 2 x kv, hidden] = [2560 x 4096] per rank, 21 MB)
    # -- was the drafter's last linear outside the W4 lane ("30 of 31"): 101 us
    # per step in the 09-05 trace. With VLLM_GLM53_DRAFTER_CTX_KV_W4=1 the
    # fused weight gets a W4 pack at buffer-build time and decode-sized
    # batches (M <= 32) run it on the lane; larger batches (prefill) and a
    # disarmed lane keep the stock F.linear. The query path's k/v rows already
    # serve from the same kind of pack (VLLM_DFLASH2_FP8_DENSE=1), so this
    # makes the context K/V numerics match the query K/V numerics. Drafter
    # only (acceptance profile is the gate; the target reads none of this).
    def _build_fused_kv_buffers(self) -> None:
        super()._build_fused_kv_buffers()
        self._deneb_ctx_kv_pack = None
        if (os.environ.get("VLLM_GLM53_DRAFTER_CTX_KV_W4") or "0").strip() != "1":
            return
        try:
            from vllm.model_executor.layers import glm53_megakernel as _mk

            w = self._fused_kv_weight
            if (getattr(self, "_fused_kv_bias", None) is None and w.dim() == 2
                    and w.dtype == torch.bfloat16 and w.shape[1] % 128 == 0
                    and w.shape[1] <= _mk.MK_GEMM_KMAX):
                self._deneb_ctx_kv_pack = _mk.build_mk_weight_w4(w)
                logger.warning("[drafter-ctx-kv] W4 pack built for the context K/V "
                               "projection [%d x %d]; decode batches take the lane",
                               int(w.shape[0]), int(w.shape[1]))
            else:
                logger.warning("[drafter-ctx-kv] shape/bias not admitted -> stock bf16 "
                               "F.linear (bias=%s shape=%s)",
                               getattr(self, "_fused_kv_bias", None) is not None,
                               tuple(w.shape))
        except Exception:
            self._deneb_ctx_kv_pack = None
            logger.exception("[drafter-ctx-kv] pack build failed -> stock bf16 F.linear")

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pack = getattr(self, "_deneb_ctx_kv_pack", None)
        if pack is None:
            return super()._project_context_kv(
                context_states, num_ctx, num_layers, num_kv_heads, head_dim)
        from vllm.model_executor.layers import glm53_megakernel as _mk

        normed = torch.empty_like(context_states)
        ops.rms_norm(normed, context_states, self._hidden_norm_weight, self._rms_norm_eps)
        _mk.maybe_arm()
        all_kv_flat = _mk.gemm_w4a8(normed, pack, int(self._fused_kv_weight.shape[0]))
        if all_kv_flat is None:
            # prefill-sized batch or a disarmed lane: the stock projection
            all_kv_flat = F.linear(normed, self._fused_kv_weight, None)
        elif not getattr(self, "_deneb_ctx_kv_said", False):
            self._deneb_ctx_kv_said = True
            logger.warning("[drafter-ctx-kv] serving: context K/V projection on the W4 "
                           "lane (rows=%d, was bf16 F.linear)", int(num_ctx))
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        return all_kv[0], all_kv[1]


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = Fp8HeadLogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
            selector_top_k=int(draft_config["selector_top_k"]),
            valid_vocab_size=decodable_vocab_size(
                vllm_config.model_config.tokenizer
            ),
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # AR prefetch hints for the drafter's own collectives (tp_oneshot_ar,
        # VLLM_GLM53_AR_PREFETCH): the drafter runs TP=4 with two collectives
        # per layer, and each one can warm the next GEMM's W4 pack while it
        # waits for the peers. The shim keys its learned hints by "which
        # collective of which model's forward", so the boundary has to come
        # from THIS class -- above the compiled DFlashQwen3Model, like the
        # target's boundary sits above its compiled region -- and under its
        # own scope, so the target's richer table never overwrites the
        # drafter's. No-op unless the knob is set or the shim is not mounted.
        osar = _osar_shim()
        if osar is not None:
            osar.begin_forward("drafter")
        try:
            return super().forward(input_ids, positions, inputs_embeds)
        finally:
            if osar is not None:
                osar.end_forward()

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # deneb fork (glm53_dflash_early_fc): the fc may already have run on a
        # side stream under the target's head + sampler. Take that result for
        # this step's token count, else the stock projection. Waits on the
        # producer's event first, before precompute_and_store_context_kv and
        # the drafter graph -- no megakernel launch overlaps the fc's.
        early = _early_fc_take()
        if early is not None and hidden_states.dim() == 2:
            got = early(self, int(hidden_states.shape[0]))
            if got is not None:
                return got
        return super().combine_hidden_states(hidden_states)

    def verify_selector_loaded(self) -> None:
        """Say out loud whether the path selector's weights actually arrived.

        The codebooks are declared with `torch.empty`, so a name the loader
        never matched keeps whatever was in that memory. This lane has now hit
        that exact failure twice -- the drafter's fp8 lm_head (#132) and the
        EP pair buffer (#146) -- and both were silent: no error, just wrong
        numbers downstream.

        Here the downstream symptom would be suffix decay, which is precisely
        what DFlash2's selector exists to remove. The design reports ~86 pct
        conditional acceptance held across all draft positions; this fleet
        measures 78.6 / 57.5 / 35.1 / 22.7 / 16.8 / 10.3 / 6.7 and a mean
        accept length of 2.4-3.3 against a designed 4.8.
        """
        # The selector lives on the inner model (see compute_candidates), not
        # on this wrapper -- reading it off `self` would silently find None and
        # report nothing, which is the failure this check exists to catch.
        selector = getattr(
            getattr(self, "model", None), "candidate_selector", None
        )
        if selector is None:
            logger.warning(
                "dflash2 path selector: not found on the model -- cannot "
                "verify that its codebooks loaded"
            )
            return
        stats = {}
        for name in ("predecessor_codebook", "successor_codebook"):
            w = getattr(selector, name, None)
            if w is None:
                stats[name] = (0.0, 0.0, 0.0)
                continue
            with torch.no_grad():
                f = w.float()
                finite = torch.isfinite(f)
                frac = float(finite.float().mean())
                clean = f[finite]
                amax = float(clean.abs().max()) if clean.numel() else 0.0
                amean = float(clean.abs().mean()) if clean.numel() else 0.0
            stats[name] = (frac, amax, amean)
        ok, reasons = dflash2_selector_load_verdict(stats)
        detail = " ".join(
            f"{n}(finite={fr:.4f} |max|={mx:.3g} |mean|={mn:.3g})"
            for n, (fr, mx, mn) in sorted(stats.items())
        )
        if ok:
            logger.info("dflash2 path selector: loaded -- %s", detail)
        else:
            logger.warning(
                "dflash2 path selector looks UNLOADED (%s) -- %s. The selector "
                "is what holds acceptance flat across draft positions; without "
                "it the block suffix decays and mean accept length collapses.",
                "; ".join(reasons), detail,
            )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # First call only: the selector is loaded by now, and its codebooks
        # are declared with torch.empty -- a name the loader never matched
        # would look like suffix decay, not like an error.
        if not getattr(self, "_deneb_selector_checked", False):
            self._deneb_selector_checked = True
            self.verify_selector_loaded()
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )


EntryClass = DFlash2Qwen3ForCausalLM
