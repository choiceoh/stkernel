# SPDX-License-Identifier: Apache-2.0
"""Block-quantized fp8 (W8A16) vocabulary head for a speculative drafter.

The draft head is read once per drafted token, so halving its bytes is worth
more than the quantization costs. Adopted on this fleet for DSV4's drafter as
VLLM_DSPARK_FP8_LM_HEAD; the quantize/GEMM pair below is that implementation,
moved here so a second drafter can use it instead of growing a copy.

Not to be confused with the rowwise `_scaled_mm` draft head, which was measured
on the same fleet and rejected (60.6 against 61.7, acceptance unmoved).

`build_fp8_lm_head` runs after weights load and before capture; it attaches the
quantized pair to the head module, so `Fp8HeadLogitsProcessor._apply_head` finds
them from the argument it is already given and everything else in
`get_top_k_tokens` -- padding mask, top-k reduction, its all-gather, scale, soft
cap -- stays untouched.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor

logger = init_logger(__name__)


def _read_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# deepgemm packs only the exponent of an fp32 scale (UE8M0) and asserts the
# sign and mantissa are zero -- smxx_layout.cuh:131,
# `(values[j] & 0x807fffffu) == 0`. That assert is a printf followed by
# `asm("trap;")`: it destroys the CUDA context, so the boot dies later at an
# unrelated sync (empty_cache) with a traceback that never names this file.
# We hit it on this fleet 768 times in one boot. Enforce the precondition here
# instead of trusting the upstream rounding to have covered every value.
_SF_SIGN_AND_MANTISSA = 0x807FFFFF
_SMALLEST_NORMAL_F32 = 2.0**-126
_DECODABLE_VOCAB_CACHE: dict[tuple[str, str], int] = {}


def decodable_vocab_size(
    model_path: str,
    override_env: str = "VLLM_GLM53_DECODABLE_VOCAB",
) -> int | None:
    """Return the number of ids the tokenizer can decode, cached per path."""
    override = os.environ.get(override_env, "").strip()
    key = (model_path, override)
    if key in _DECODABLE_VOCAB_CACHE:
        return _DECODABLE_VOCAB_CACHE[key] or None
    if override:
        value = int(override)
        _DECODABLE_VOCAB_CACHE[key] = value
        return value

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        value = int(tokenizer.get_vocab_size(with_added_tokens=True))
        del tokenizer
    except Exception as exc:  # a missing/odd tokenizer must not stop a boot
        logger.warning(
            "[vocab-mask] could not read the tokenizer at %s (%r) -- "
            "orphan LM head rows stay reachable",
            model_path,
            exc,
        )
        value = 0
    _DECODABLE_VOCAB_CACHE[key] = value
    return value or None


def _local_valid_vocab_end(
    valid_vocab_size: int | None,
    vocab_start: int,
    local_width: int,
) -> int:
    """Map a global valid-vocab bound onto one vocab-parallel shard."""
    if valid_vocab_size is None:
        return local_width
    return max(0, min(local_width, valid_vocab_size - vocab_start))


def _ue8m0_violations(scales: "torch.Tensor") -> "torch.Tensor":
    """The kernel's assert, evaluated on the host."""
    return (scales.view(torch.int32) & _SF_SIGN_AND_MANTISSA) != 0


def _describe_ue8m0_scales(scales: "torch.Tensor") -> int:
    """Describe post-requant scales without changing them; return bad count."""
    if scales.dtype != torch.float32:
        logger.warning("fp8 lm_head: scales are %s, not float32", scales.dtype)
        return int(scales.numel())
    bad = _ue8m0_violations(scales)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return 0
    finite = torch.isfinite(scales)
    nan = int(torch.isnan(scales).sum())
    inf = int((~finite & ~torch.isnan(scales)).sum())
    neg = int((finite & (scales < 0)).sum())
    denorm = int((finite & (scales > 0) & (scales < _SMALLEST_NORMAL_F32)).sum())
    normal_bad = n_bad - nan - inf - neg - denorm
    sample = scales.reshape(-1)[bad.reshape(-1)][:4].tolist()
    logger.warning(
        "fp8 lm_head: %d of %d post-requant scales fail deepgemm's UE8M0 "
        "check (nan=%d inf=%d negative=%d denormal=%d normal-but-not-pow2=%d); "
        "first offenders %s. Left untouched -- rewriting them would change the "
        "weights the requant already matched to them.",
        n_bad, scales.numel(), nan, inf, neg, denorm, normal_bad, sample,
    )
    return n_bad


def _repair_ue8m0_scales(scales: "torch.Tensor") -> tuple["torch.Tensor", int]:
    """Force every scale onto an exact power of two. Returns (fixed, count).

    Positive normals round UP to the next power of two -- the safe direction
    for a quantization scale, and at most 2x.

    Denormals are the case rounding cannot fix: their exponent field is
    already zero, so exp2(ceil(log2(x))) is still denormal and still trips the
    assert. They come from a weight block that is all but zero, so flushing
    them (with 0, negatives, inf and NaN) to zero is both what the kernel
    accepts -- 0x00000000 passes -- and numerically what that block meant.
    """
    if scales.dtype != torch.float32:
        return scales, 0
    bad = _ue8m0_violations(scales)
    count = int(bad.sum())
    if count == 0:
        return scales, 0
    out = torch.zeros_like(scales)
    keep = torch.isfinite(scales) & (scales >= _SMALLEST_NORMAL_F32)
    rounded = torch.exp2(torch.ceil(torch.log2(scales[keep])))
    # log2/exp2 round-off can land just under the smallest normal; drop those.
    rounded = torch.where(
        rounded >= _SMALLEST_NORMAL_F32, rounded, torch.zeros_like(rounded)
    )
    out[keep] = rounded
    return out, count


def _quantize_fp8_deepgemm(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a bf16/fp32 weight to the deepgemm fp8 layout (chunked
    over rows so the fp32 staging copy never exceeds ~1/8 of the weight)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    with torch.no_grad():
        w = weight.detach()
        rows = w.shape[0]
        step = max(128, (rows // 8) // 128 * 128)
        chunks_q, chunks_s = [], []
        for r0 in range(0, rows, step):
            cq, cs = per_block_cast_to_fp8(
                w[r0 : r0 + step].float(), [128, 128], use_ue8m0=True
            )
            chunks_q.append(cq)
            chunks_s.append(cs)
        # use_e8m0=True does the requant AND the layout transform in one
        # call, and the trapping assert lives between them -- neither before
        # nor after that call can reach it. Do the requant ourselves, inspect
        # what it leaves behind, then run the transform alone.
        wq = torch.cat(chunks_q, dim=0)
        ws = torch.cat(chunks_s, dim=0)
        try:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                requant_weight_ue8m0_inplace,
            )
        except ImportError:
            return deepgemm_post_process_fp8_weight_block(
                wq, ws, (128, 128), use_e8m0=True
            )
        requant_weight_ue8m0_inplace(wq, ws, block_size=(128, 128))
        # Report without rewriting (#131), then fail before the layout kernel
        # can execute its device-side trap. build_fp8_lm_head_weight catches
        # this and keeps the loaded bf16 head intact.
        bad_scales = _describe_ue8m0_scales(ws)
        if bad_scales:
            raise RuntimeError(
                f"{bad_scales} of {ws.numel()} UE8M0 scales are unsafe to pack"
            )
        return deepgemm_post_process_fp8_weight_block(
            wq, ws, (128, 128), use_e8m0=False
        )


def _fp8_gemm(
    x: torch.Tensor, dg_w: torch.Tensor, dg_ws: torch.Tensor
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt

    xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(
        x.to(torch.bfloat16), 128
    )
    out = torch.empty(
        x.shape[0], dg_w.shape[0], dtype=torch.bfloat16, device=x.device
    )
    fp8_gemm_nt((xq, xs), (dg_w, dg_ws), out)
    return out


def build_fp8_lm_head_weight(head) -> bool:
    """Quantize one head module in place. Returns whether it took."""
    weight = getattr(head, "weight", None)
    if weight is None or weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    try:
        dg_w, dg_ws = _quantize_fp8_deepgemm(weight)
    except Exception as exc:
        # Name the exception. A bare "failed; staying on bf16" reads as a
        # healthy fallback, but a device-side assert inside deepgemm kills the
        # CUDA context on the way out -- the boot then dies somewhere else
        # entirely (empty_cache) with a traceback that never mentions this.
        logger.warning_once(
            "fp8 lm_head: quantization failed (%s: %s) on weight %s %s; "
            "staying on bf16.",
            type(exc).__name__,
            exc,
            tuple(weight.shape),
            weight.dtype,
        )
        return False
    head._deneb_fp8_w = dg_w
    head._deneb_fp8_ws = dg_ws
    logger.info_once("fp8 lm_head: quantized %s.", tuple(dg_w.shape))
    return True


def build_fp8_lm_head(model) -> bool:
    """Quantize `model.lm_head` in place. Returns whether it took.

    Any failure leaves the head untouched and the caller on the bf16 path: this
    is an optimization, and a drafter that runs slower is better than one that
    does not run.
    """
    if not _read_bool_env("VLLM_SPEC_FP8_LM_HEAD"):
        return False
    return build_fp8_lm_head_weight(getattr(model, "lm_head", None))


class Fp8HeadLogitsProcessor(LogitsProcessor):
    """LogitsProcessor whose head projection uses an fp8 copy of the weight.

    `fp8_env` names the knob that arms it, because the two ends of speculative
    decoding do not carry the same risk. A badly quantized draft head costs
    acceptance and nothing else; rejection sampling still reproduces the
    target's distribution. The target's logits decide the sampled token and the
    accept/reject, so they are outside that guarantee.

    The copy is built on first use when it was not built at load time, which is
    how the target head is handled elsewhere in this stack -- the first call is
    the eager warmup, before capture.
    """

    def __init__(
        self,
        *args,
        fp8_env: str = "VLLM_SPEC_FP8_LM_HEAD",
        valid_vocab_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._deneb_fp8_env = fp8_env
        if valid_vocab_size is not None and valid_vocab_size <= 0:
            raise ValueError("valid_vocab_size must be positive when provided")
        if valid_vocab_size is not None and valid_vocab_size > self.org_vocab_size:
            raise ValueError(
                "valid_vocab_size cannot exceed the logits processor vocabulary"
            )
        self._deneb_valid_vocab_size = valid_vocab_size

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        # The target and drafter may share one lm_head object.  The FP8 copy is
        # shared storage, but the endpoint gates are intentionally independent:
        # an off endpoint must not inherit the other endpoint's FP8 copy.
        use_fp8 = _read_bool_env(self._deneb_fp8_env)
        dg_w = getattr(lm_head, "_deneb_fp8_w", None) if use_fp8 else None
        if dg_w is None and use_fp8:
            if build_fp8_lm_head_weight(lm_head):
                dg_w = getattr(lm_head, "_deneb_fp8_w", None)
        if (
            dg_w is None
            or embedding_bias is not None
            or (self.head_dtype is not None
                and self.head_dtype != hidden_states.dtype)
        ):
            out = super()._apply_head(lm_head, hidden_states, embedding_bias)
        else:
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            out = _fp8_gemm(flat, dg_w, lm_head._deneb_fp8_ws)
            out = out.view(*hidden_states.shape[:-1], -1)

        # DFlash2 selects top-k from the local shard before the cross-rank
        # reduction.  Mask tokenizer-orphan rows here so they cannot occupy a
        # candidate slot and become a guaranteed rejection (target p=0).
        if self._deneb_valid_vocab_size is not None:
            shard = getattr(lm_head, "shard_indices", None)
            vocab_start = int(getattr(shard, "org_vocab_start_index", 0))
            valid_end = _local_valid_vocab_end(
                self._deneb_valid_vocab_size, vocab_start, out.shape[-1]
            )
            if valid_end < out.shape[-1]:
                out[..., valid_end:] = -float("inf")
        return out
