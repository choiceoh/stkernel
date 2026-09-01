# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
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


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
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
            self.block_size,
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
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
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
