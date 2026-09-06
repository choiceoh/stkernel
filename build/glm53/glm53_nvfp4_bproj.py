# SPDX-License-Identifier: Apache-2.0
"""Opt-in NVFP4 for the BF16 MLA B projections during eager pure prefill.

The source weight and original linear method remain the decode path. In
particular, MLA's absorbed W_UK_T/W_UV copies still come directly from BF16:
get_and_maybe_dequant_weights recognizes this UnquantizedLinearMethod
subclass. The sparse MQA path uses those absorbed copies, so installing a
kv_b_proj pack does not imply that sparse MQA will execute it.
"""

from functools import cache
import os
import re

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

logger = init_logger(__name__)
_ENABLED = os.environ.get("VLLM_GLM53_PREFILL_NVFP4_BPROJ") == "1"
_MIN_ROWS = 128
_BPROJ = re.compile(r"^(.*\.self_attn)\.(q_b_proj|kv_b_proj|indexer\.wq_b)$")


@cache
def _device_supported(device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (12, 1)


def _pure_prefill(metadata, layer_name, num_tokens):
    """Accept only host counts for the exact MLA layer and full token batch."""
    if not isinstance(metadata, dict):
        return False
    item = metadata.get(layer_name)
    counts = (
        getattr(item, "num_actual_tokens", None),
        getattr(item, "num_prefills", None),
        getattr(item, "num_decodes", None),
        getattr(item, "num_decode_tokens", None),
    )
    if not all(type(value) is int for value in counts):
        return False
    actual, prefills, decodes, decode_tokens = counts
    if actual != num_tokens or prefills <= 0 or decodes != 0 or decode_tokens != 0:
        return False
    # MLA currently folds speculative rows into decode counts. Also reject
    # explicit speculative counts if a future metadata version adds them.
    for field in ("num_spec_decodes", "num_spec_decode_tokens"):
        value = getattr(item, field, 0)
        if type(value) is not int or value != 0:
            return False
    prefill_tokens = getattr(item, "num_prefill_tokens", num_tokens)
    return type(prefill_tokens) is int and prefill_tokens == num_tokens


class PrefillNvfp4BprojMethod(UnquantizedLinearMethod):
    """Keep BF16 extraction/decode semantics and route only proven prefill."""

    def __init__(self, base, pair, weight, name, metadata_name):
        self._base = base
        self._pair = pair
        self._source = weight
        self._source_ptr = weight.data_ptr()
        self._device = weight.device
        self._shape = tuple(weight.shape)
        self._name = name
        self._metadata_name = metadata_name
        self._reported = False

    def apply(self, layer, x, bias=None):
        pair = self._pair
        weight = getattr(layer, "weight", None)
        if (
            pair is None
            or torch.compiler.is_compiling()
            or bias is not None
            or getattr(layer, "bias", None) is not None
            or weight is not self._source
            or tuple(weight.shape) != self._shape
            or weight.dtype != torch.bfloat16
            or weight.device != self._device
            or weight.data_ptr() != self._source_ptr
            or not weight.is_contiguous()
            or x.ndim != 2
            or x.shape[0] < _MIN_ROWS
            or x.shape[1] != self._shape[1]
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or x.device != weight.device
            or not x.is_contiguous()
            or torch.cuda.is_current_stream_capturing()
            or not is_forward_context_available()
        ):
            return self._base.apply(layer, x, bias)
        if not _pure_prefill(
            get_forward_context().attn_metadata, self._metadata_name, x.shape[0]
        ):
            return self._base.apply(layer, x, bias)
        from .glm53_fp8_dense import _nvfp4_dense_gemm

        # Once an admitted CUDA launch starts, errors must propagate. A
        # local retry cannot recover an asynchronous fault or safely choose
        # a different path ahead of the other TP ranks' collectives.
        out = _nvfp4_dense_gemm(x, *pair)
        if not self._reported:
            self._reported = True
            logger.warning(
                "[nvfp4-bproj] serving %s M=%d [N,K]=%s (pure prefill; "
                "decode and absorbed MLA weights stay BF16)",
                self._name, x.shape[0], self._shape,
            )
        return out


def maybe_build_nvfp4_bproj(model):
    """Rebuild checked packs from loaded BF16 weights; never change a default.

    Already-quantized methods are deliberately left as configured, including
    the existing all-token FP8_DENSE_BPROJ extension. Each new pack must pass
    a real direct-GEMM boot check; an unavailable check cannot arm this path.
    This is a source/scale check, not the serving quality or throughput gate.
    """
    if not _ENABLED:
        return False
    from . import glm53_fp8_dense as dense

    built = []
    skipped = []
    for name, mod in model.named_modules():
        # Prefixes are the metadata's runtime names, whereas named_modules
        # may be rooted at either the model or its enclosing causal LM.
        prefix = getattr(mod, "prefix", None)
        match = _BPROJ.fullmatch(prefix) if isinstance(prefix, str) else None
        if match is None:
            continue
        base = getattr(mod, "quant_method", None)
        if isinstance(base, PrefillNvfp4BprojMethod):
            base = base._base
            mod.quant_method = base
        weight = getattr(mod, "weight", None)
        if (
            type(base) is not UnquantizedLinearMethod
            or getattr(mod, "bias", None) is not None
            or not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.dtype != torch.bfloat16
            or not weight.is_cuda
            or not weight.is_contiguous()
            or min(weight.shape) < 512
            or weight.shape[0] % 128 != 0
            or weight.shape[1] % 128 != 0
            or not _device_supported(weight.device)
        ):
            skipped.append(name)
            continue
        try:
            wq, wsf, w_gs = dense._quantize_nvfp4(weight)
            rows = weight.shape[0]
            known_alpha = dense._NVFP4_ALPHA[0]
            candidates = (known_alpha,) if known_alpha is not None else (1.0, -1.0)
            pair = None
            for alpha in candidates:
                checked = dense._copy_matches_source(
                    mod,
                    dense.NvFp4DenseMethod(base, wq, wsf, w_gs, rows, alpha),
                    weight,
                    rtol=4 * dense._STALE_RTOL,
                    got_fn=lambda xx: dense._nvfp4_dense_gemm(
                        xx, wq, wsf, w_gs, rows, alpha
                    ),
                )
                if checked is True:
                    pair = (wq, wsf, w_gs, rows, alpha)
                    dense._NVFP4_ALPHA[0] = alpha
                    break
            if pair is None:
                skipped.append(name)
                logger.warning(
                    "[nvfp4-bproj] %s kept original method: boot check "
                    "unavailable or failed", name,
                )
                continue
            mod.quant_method = PrefillNvfp4BprojMethod(
                base, pair, weight, prefix, f"{match.group(1)}.attn"
            )
            built.append((prefix, tuple(weight.shape)))
        except Exception as exc:
            skipped.append(name)
            logger.warning("[nvfp4-bproj] %s kept original method: %r", name, exc)
        finally:
            # The GB10 shares host and device memory. Bound loading's cached
            # transient to one weight, as the enclosing dense pass does.
            torch.cuda.empty_cache()
    logger.warning(
        "[nvfp4-bproj] %d checked packs, %d skipped; BF16 sources retained; "
        "actual invocation is logged separately: %s",
        len(built), len(skipped), built,
    )
    return bool(built)
