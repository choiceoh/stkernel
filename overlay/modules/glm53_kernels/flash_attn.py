# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend for MLA prefill."""

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, zip_inputs
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import (
    compile_flash_attn_varlen_func_from_specs,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
        MLADims,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


FA4_STANDARD_DTYPES = (torch.bfloat16, torch.float16)

# Current vLLM MLA prefill expands K/V to num_heads before FA4, so this plan
# covers qhead_per_kvhead=1. Batch is not a current FA4 MLA-prefill key field.
# Use b1 for compile-only specs because it is the conservative case for
# Split-KV shape heuristics.
# TODO(roberto): FA4 also has direct-GQA and qv/top-k absorbed-MLA paths, but
# vLLM does not use them in this backend yet; they need a separate
# num_kv_heads/qv/top-k-aware warmup plan if wired in later.
FA4_MLA_PREFILL_COMPILE_BATCH_SIZE = 1
FA4_MLA_PREFILL_Q_TILE = 128
FA4_MLA_PREFILL_K_TILE = 128
FA4_MLA_PREFILL_LONG_K_BLOCKS = 32
FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS = 64
FA4_MLA_PREFILL_CAUSAL_OPTIONS = (False, True)
FA4_MLA_PREFILL_LSE_OPTIONS = (False, True)

# deneb fork: opt-in GLM-5.3/GB10 short-prefill admission. Keep the stock
# dimension list unchanged unless every runtime contract below is exact.
_GLM53_SM121_MLA_PREFILL_ENV = "VLLM_GLM53_SM121_MLA_PREFILL"


def _glm53_sm121_dense_prefill_gate(
    *,
    flag: str | None,
    capability: tuple[int, int] | None,
    dimensions: tuple[int, int, int],
    model_type: str | None,
    kv_lora_rank: int | None,
    num_attention_heads: int | None,
    index_topk: int | None,
    index_kpool: int | None,
    kv_cache_dtype: str | None,
    outer_backend_name: str | None,
    flash_attn_version: int | None,
) -> bool:
    """Fail-closed admission for GLM-5.3's SM121 dense-prefill arm.

    This is intentionally narrower than ``FlashAttnPrefillBackend``'s normal
    dimension allowlist. The production GLM image currently runs the SM90
    sparse implementation with the standard HND fp8 cache on GB10. That
    implementation inherits ``SparseMLACommonImpl.forward_mha`` and can use
    FlashAttention for the dense (``seq_len <= topk``) prefill subset.

    Do not admit the SM120 sparse implementation here. Its upstream contract
    explicitly has no dense-MHA prefill, and packed ``fp8_ds_mla`` context
    gathers have had correctness failures. Every mismatch therefore retains
    the existing top-k MQA path.
    """
    return (
        flag == "1"
        and capability == (12, 1)
        and dimensions == (256, 0, 256)
        and model_type == "glm5_next_text"
        and kv_lora_rank == 512
        and num_attention_heads == 64
        and index_topk == 2048
        and index_kpool == 4
        and kv_cache_dtype == "fp8_e4m3"
        and outer_backend_name == "FLASHINFER_MLA_SPARSE_SM90"
        and flash_attn_version == 2
    )


def _glm53_sm121_dense_prefill_enabled(mla_dimensions: MLADimensions) -> bool:
    """Read the live layer/config state and apply the pure admission gate."""
    capability = current_platform.get_device_capability()
    if capability is None:
        return False

    # Import lazily: this module is resolved while MLAAttention itself is being
    # constructed, so a top-level config import would add a cycle to the stock
    # prefill backend.
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return False

    model_config = vllm_config.model_config
    hf_text_config = model_config.hf_text_config
    cache_config = vllm_config.cache_config
    if cache_config is None:
        return False

    # MLAAttention registers itself in static_forward_context before asking the
    # prefill selector for a backend. Read that just-created layer to bind this
    # opt-in to the current HND/SM90 sparse implementation rather than merely
    # assuming the selector's candidate list will stay ordered the same way.
    outer_backend_name = None
    static_context = vllm_config.compilation_config.static_forward_context
    for layer in reversed(tuple(static_context.values())):
        if (
            getattr(layer, "qk_nope_head_dim", None)
            == mla_dimensions.qk_nope_head_dim
            and getattr(layer, "qk_rope_head_dim", None)
            == mla_dimensions.qk_rope_head_dim
            and getattr(layer, "v_head_dim", None) == mla_dimensions.v_head_dim
        ):
            backend = getattr(layer, "attn_backend", None)
            get_name = getattr(backend, "get_name", None)
            if callable(get_name):
                outer_backend_name = get_name()
            break

    return _glm53_sm121_dense_prefill_gate(
        flag=os.environ.get(_GLM53_SM121_MLA_PREFILL_ENV),
        capability=(capability.major, capability.minor),
        dimensions=(
            mla_dimensions.qk_nope_head_dim,
            mla_dimensions.qk_rope_head_dim,
            mla_dimensions.v_head_dim,
        ),
        model_type=getattr(hf_text_config, "model_type", None),
        kv_lora_rank=getattr(hf_text_config, "kv_lora_rank", None),
        num_attention_heads=getattr(hf_text_config, "num_attention_heads", None),
        index_topk=getattr(hf_text_config, "index_topk", None),
        index_kpool=getattr(hf_text_config, "index_kpool", None),
        kv_cache_dtype=str(cache_config.cache_dtype),
        outer_backend_name=outer_backend_name,
        flash_attn_version=get_flash_attn_version(
            head_size=mla_dimensions.qk_nope_head_dim
            + mla_dimensions.qk_rope_head_dim,
            head_size_v=mla_dimensions.v_head_dim,
        ),
    )


def _glm53_sm121_instance_dimensions(
    *,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
) -> bool:
    """Identify only the non-stock GLM dimension arm from instance fields."""
    return (qk_nope_head_dim, qk_rope_head_dim, v_head_dim) == (
        256,
        0,
        256,
    )


def _glm53_sm121_filter_prefill_metadata(
    prefill_metadata: "MLACommonPrefillMetadata", *, optin: bool
) -> None:
    """Keep SM121 FA2 away from the unsafe non-causal context call."""
    if optin and prefill_metadata.chunked_context is not None:
        prefill_metadata.use_dense_mha = False


class FA4MLAPrefillKernel(VllmJitKernel["FA4MLAPrefillKernel.CompileKey"]):
    """FA4 MLA prefill compile-key wrapper used by generic JIT warmup."""

    @dataclass(frozen=True)
    class CompileKey:
        """High-level FA4 compile-only key used by vLLM warmup.

        This is not the CuTeDSL cache key. FA4 owns the selector that maps these
        serving inputs to the actual compile-static fields: tile sizes, q_stage,
        Split-KV, scheduler choice, layout-presence booleans, dtype/head dims,
        arch, and related fields.
        """

        q_shape: tuple[int, ...]
        k_shape: tuple[int, ...]
        v_shape: tuple[int, ...]
        q_dtype: torch.dtype
        max_seqlen_q: int
        max_seqlen_k: int
        softmax_scale: float
        causal: bool
        fa_version: int
        v_stride: tuple[int, ...] | None = None
        cu_seqlens_q_shape: tuple[int, ...] | None = None
        cu_seqlens_k_shape: tuple[int, ...] | None = None
        window_size: tuple[int, int] | None = None
        return_softmax_lse: bool = False
        num_splits: int = 0

    @staticmethod
    def kernel(
        *args: Any,
        runtime_kernel: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        kernel = flash_attn_varlen_func if runtime_kernel is None else runtime_kernel
        assert kernel is not None
        return kernel(*args, **kwargs)

    def dispatch(  # type: ignore[override]
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        num_heads: int,
        mla_dims: "MLADims",
        max_seqlen_q: int,
        max_seqlen_k: int,
        requires_v_padding: bool,
        causal: bool,
        fa_version: int,
        window_size: tuple[int, int] | None = None,
        return_lse: bool = False,
        num_splits: int = 0,
    ) -> CompileKey:
        return self.CompileKey(
            q_shape=(
                batch_size * max_seqlen_q,
                num_heads,
                mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim,
            ),
            k_shape=(
                batch_size * max_seqlen_k,
                num_heads,
                mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim,
            ),
            v_shape=(
                batch_size * max_seqlen_k,
                num_heads,
                (
                    mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim
                    if requires_v_padding
                    else mla_dims.v_head_dim
                ),
            ),
            q_dtype=dtype,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=(mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim)
            ** -0.5,
            causal=causal,
            fa_version=fa_version,
            v_stride=(
                None
                if requires_v_padding
                else (
                    num_heads * (mla_dims.qk_nope_head_dim + mla_dims.v_head_dim),
                    mla_dims.qk_nope_head_dim + mla_dims.v_head_dim,
                    1,
                )
            ),
            cu_seqlens_q_shape=(batch_size + 1,),
            cu_seqlens_k_shape=(batch_size + 1,),
            window_size=window_size,
            return_softmax_lse=return_lse,
            num_splits=num_splits,
        )

    def _is_valid_warmup_shape_probe(
        self,
        *,
        max_seqlen_k: int,
        num_splits: int,
        is_sm90: bool,
        qk_head_dim: int,
        effective_v_head_dim: int,
    ) -> bool:
        long_k = FA4_MLA_PREFILL_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE
        very_long_k = FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE
        return (
            max_seqlen_k < long_k
            or (num_splits != 1 and max_seqlen_k == long_k)
            or (
                num_splits != 1
                and max_seqlen_k == very_long_k
                and not is_sm90
                and qk_head_dim != effective_v_head_dim
            )
        )

    def get_warmup_keys(self, vllm_config: "VllmConfig") -> list[CompileKey]:
        from vllm.model_executor.layers.attention.mla_attention import (
            get_mla_dims,
        )

        mla_dims = get_mla_dims(vllm_config.model_config)
        qk_head_dim = mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim
        fa_version = get_flash_attn_version(head_size=qk_head_dim)
        if fa_version != 4:
            return []

        dtype = vllm_config.model_config.dtype
        if dtype not in FA4_STANDARD_DTYPES:
            dtype = torch.bfloat16

        is_sm90 = current_platform.is_device_capability(90)
        is_sm100_family = current_platform.is_device_capability_family(100)
        is_sm120 = current_platform.is_device_capability_family(120)
        if not (is_sm90 or is_sm100_family or is_sm120):
            return []

        num_splits = 1 if envs.VLLM_BATCH_INVARIANT else 0
        if is_sm120 and num_splits != 1:
            return []

        num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        if num_heads <= 0:
            return []

        requires_v_padding = False
        effective_v_head_dim = (
            qk_head_dim if requires_v_padding else mla_dims.v_head_dim
        )
        shape_probes = (
            dict(max_seqlen_q=1, max_seqlen_k=FA4_MLA_PREFILL_K_TILE),
            dict(
                max_seqlen_q=FA4_MLA_PREFILL_Q_TILE + 1,
                max_seqlen_k=4 * FA4_MLA_PREFILL_K_TILE,
            ),
            dict(
                max_seqlen_q=1,
                max_seqlen_k=FA4_MLA_PREFILL_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE,
            ),
            dict(
                max_seqlen_q=FA4_MLA_PREFILL_Q_TILE + 1,
                max_seqlen_k=FA4_MLA_PREFILL_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE,
            ),
            dict(
                max_seqlen_q=1,
                max_seqlen_k=FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS
                * FA4_MLA_PREFILL_K_TILE,
            ),
            dict(
                max_seqlen_q=FA4_MLA_PREFILL_Q_TILE + 1,
                max_seqlen_k=FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS
                * FA4_MLA_PREFILL_K_TILE,
            ),
        )

        return self._trace_dispatch(self.dispatch)(
            zip_inputs(*shape_probes),
            batch_size=FA4_MLA_PREFILL_COMPILE_BATCH_SIZE,
            dtype=dtype,
            num_heads=num_heads,
            mla_dims=mla_dims,
            requires_v_padding=requires_v_padding,
            is_sm90=is_sm90,
            qk_head_dim=qk_head_dim,
            effective_v_head_dim=effective_v_head_dim,
            causal=FA4_MLA_PREFILL_CAUSAL_OPTIONS,
            return_lse=FA4_MLA_PREFILL_LSE_OPTIONS,
            num_splits=num_splits,
            fa_version=fa_version,
            _when=self._is_valid_warmup_shape_probe,
        )

    def compile(self, compile_key: CompileKey) -> None:
        assert compile_flash_attn_varlen_func_from_specs is not None
        window_size = (
            list(compile_key.window_size)
            if compile_key.window_size is not None
            else None
        )
        compile_flash_attn_varlen_func_from_specs(
            q_shape=compile_key.q_shape,
            k_shape=compile_key.k_shape,
            v_shape=compile_key.v_shape,
            q_dtype=compile_key.q_dtype,
            v_stride=compile_key.v_stride,
            cu_seqlens_q_shape=compile_key.cu_seqlens_q_shape,
            cu_seqlens_k_shape=compile_key.cu_seqlens_k_shape,
            max_seqlen_q=compile_key.max_seqlen_q,
            max_seqlen_k=compile_key.max_seqlen_k,
            softmax_scale=compile_key.softmax_scale,
            causal=compile_key.causal,
            window_size=window_size,
            return_softmax_lse=compile_key.return_softmax_lse,
            fa_version=compile_key.fa_version,
            num_splits=compile_key.num_splits,
        )

    def __call__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        runtime_kernel: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.kernel(
            q=q,
            k=k,
            v=v,
            runtime_kernel=runtime_kernel,
            **kwargs,
        )


class FlashAttnPrefillBackend(MLAPrefillBackend):
    """FlashAttention backend for MLA prefill."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def is_available(cls) -> bool:
        return is_flash_attn_varlen_func_available()

    @classmethod
    def supports_mla_dimensions(cls, mla_dimensions: MLADimensions) -> bool:
        stock_supported = mla_dimensions in [
            MLADimensions(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
            MLADimensions(
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
            ),
            MLADimensions(
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
        ]
        return stock_supported or _glm53_sm121_dense_prefill_enabled(mla_dimensions)

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        mla_dimensions = MLADimensions(
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        )
        self._glm53_sm121_optin = _glm53_sm121_instance_dimensions(
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        ) and _glm53_sm121_dense_prefill_enabled(mla_dimensions)

        # Handle the differences between the flash_attn_varlen from
        # flash_attn and the one from vllm_flash_attn
        assert flash_attn_varlen_func is not None, (
            "FlashAttnPrefillBackend requires flash_attn_varlen_func. "
            "Ensure FlashAttnPrefillBackend.is_available() is checked first."
        )
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version(
            head_size=qk_head_dim, head_size_v=v_head_dim
        )
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = functools.partial(
                flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version
            )

        # Determine if we need to pad V
        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims.
        # FA3 on Hopper (SM90) and FA4 natively handle diff headdims.
        device_capability = current_platform.get_device_capability()
        self.requires_v_padding = self.vllm_flash_attn_version is None or not (
            (
                self.vllm_flash_attn_version == 3
                and device_capability is not None
                and device_capability[0] == 9
            )
            or self.vllm_flash_attn_version == 4
        )

        # Track whether we're using vllm's FA or upstream (for ROCm)
        self._is_vllm_fa = current_platform.is_cuda() or current_platform.is_xpu()

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        super().prepare_metadata(prefill_metadata)
        # vLLM #50707 reports an SM121 FA2 device assert for non-causal varlen.
        # Cached context reaches run_prefill_context_chunk(causal=False), so
        # route the entire mixed/cached prefill batch through exact MQA instead.
        _glm53_sm121_filter_prefill_metadata(
            prefill_metadata,
            optin=self._glm53_sm121_optin,
        )

    def supports_quant_output(self, quant_key: "QuantKey") -> bool:
        device_capability = current_platform.get_device_capability()
        return (
            self.vllm_flash_attn_version == 4
            and self._is_vllm_fa
            and device_capability is not None
            and device_capability[0] in (10, 11)
            and quant_key == kFp8StaticTensorSym
        )

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
            kwargs["out"] = out
            kwargs["output_scale"] = output_scale
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            kwargs["return_attn_probs"] = return_softmax_lse
            assert out is None and output_scale is None
        if envs.VLLM_BATCH_INVARIANT:
            kwargs["num_splits"] = 1

        attn_out = FA4_MLA_PREFILL_KERNEL(
            q=q,
            k=k,
            v=maybe_padded_v,
            runtime_kernel=self.flash_attn_varlen_func,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there are multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def supports_out(self) -> bool:
        # A padded V produces a qk_head_dim output that cannot be written into
        # a v_head_dim `out`; only the unpadded path honors `out`.
        return not self.requires_v_padding

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
            out=out,
            output_scale=output_scale,
        )

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=chunk.query_start_loc,
            cu_seqlens_k=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len,
            max_seqlen_k=chunk.max_seq_len,
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
            out=out,
        )


FA4_MLA_PREFILL_KERNEL = FA4MLAPrefillKernel()
