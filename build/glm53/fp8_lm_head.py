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


def _contiguous_tokenizer_vocab_size(vocab: dict[str, int]) -> int:
    """Return the decodable prefix length, rejecting holes and empty vocabs.

    The candidate mask is a suffix mask, so a vocabulary *count* is a valid
    boundary only when tokenizer IDs are exactly ``[0, count)``.  Silently
    accepting a hole would keep one non-decodable row reachable and mask one
    real high ID instead.
    """
    ids = {int(token_id) for token_id in vocab.values()}
    if not ids:
        raise ValueError("tokenizer vocabulary is empty")
    if min(ids) != 0 or max(ids) + 1 != len(ids):
        raise ValueError(
            "tokenizer ids must form a contiguous prefix starting at zero "
            f"(unique={len(ids)}, min={min(ids)}, max={max(ids)})"
        )
    return len(ids)


def decodable_vocab_size(
    model_path: str,
    override_env: str = "VLLM_GLM53_DECODABLE_VOCAB",
) -> int:
    """Return the contiguous tokenizer-ID prefix, cached per path.

    This is a correctness boundary for speculative candidates.  A tokenizer
    that cannot be read or represented by a suffix mask must stop model load;
    fail-open would make orphan LM-head rows reachable again.
    """
    override = os.environ.get(override_env, "").strip()
    key = (model_path, override)
    if key in _DECODABLE_VOCAB_CACHE:
        return _DECODABLE_VOCAB_CACHE[key]
    if override:
        try:
            value = int(override)
        except ValueError as exc:
            raise ValueError(f"{override_env} must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"{override_env} must be a positive integer")
        _DECODABLE_VOCAB_CACHE[key] = value
        return value

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        value = _contiguous_tokenizer_vocab_size(
            tokenizer.get_vocab(with_added_tokens=True)
        )
        del tokenizer
    except Exception as exc:
        raise RuntimeError(
            f"cannot establish a safe decodable vocabulary for {model_path}"
        ) from exc
    _DECODABLE_VOCAB_CACHE[key] = value
    return value


def _validate_decodable_top_k(
    valid_vocab_size: int | None,
    selector_top_k: int | None,
) -> None:
    """Reject candidate configurations that cannot return ``top_k`` IDs."""
    if selector_top_k is None:
        return
    if selector_top_k <= 0:
        raise ValueError("selector_top_k must be positive when provided")
    if valid_vocab_size is None:
        raise ValueError("selector_top_k requires a decodable vocabulary bound")
    if valid_vocab_size < selector_top_k:
        raise ValueError(
            "decodable vocabulary is smaller than selector_top_k "
            f"({valid_vocab_size} < {selector_top_k})"
        )


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
    # The packer's bitmask accepts zero and +inf because both have a clear
    # sign/mantissa.  They are not usable quantization scales, so also enforce
    # the value-level finite-positive contract before launching the kernel.
    bit_bad = _ue8m0_violations(scales)
    finite = torch.isfinite(scales)
    nan_mask = torch.isnan(scales)
    inf_mask = torch.isinf(scales)
    zero_mask = finite & (scales == 0)
    neg_mask = finite & (scales < 0)
    denorm_mask = finite & (scales > 0) & (scales < _SMALLEST_NORMAL_F32)
    normal_bad_mask = finite & (scales >= _SMALLEST_NORMAL_F32) & bit_bad
    bad = (
        nan_mask
        | inf_mask
        | zero_mask
        | neg_mask
        | denorm_mask
        | normal_bad_mask
    )
    n_bad = int(bad.sum())
    if n_bad == 0:
        return 0
    nan = int(nan_mask.sum())
    inf = int(inf_mask.sum())
    zero = int(zero_mask.sum())
    neg = int(neg_mask.sum())
    denorm = int(denorm_mask.sum())
    normal_bad = int(normal_bad_mask.sum())
    sample = scales.reshape(-1)[bad.reshape(-1)][:4].tolist()
    logger.warning(
        "fp8 lm_head: %d of %d post-requant scales are unsafe for UE8M0 "
        "(nan=%d inf=%d zero=%d negative=%d denormal=%d "
        "normal-but-not-pow2=%d); first offenders %s. Left untouched -- "
        "rewriting them would change the weights the requant already matched "
        "to them.",
        n_bad,
        scales.numel(),
        nan,
        inf,
        zero,
        neg,
        denorm,
        normal_bad,
        sample,
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
        except ImportError as exc:
            raise RuntimeError(
                "vLLM lacks requant_weight_ue8m0_inplace; refusing an "
                "unvalidated DeepGEMM scale pack"
            ) from exc
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
    """Quantize one head at most once. Returns whether an fp8 copy exists."""
    cached = (
        getattr(head, "_deneb_fp8_w", None) is not None
        and getattr(head, "_deneb_fp8_ws", None) is not None
    )
    if cached:
        return True
    if getattr(head, "_deneb_fp8_build_attempted", False):
        return False
    weight = getattr(head, "weight", None)
    if weight is None or weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    # A failed online quantization is deterministic for this loaded weight and
    # runtime. Latch the attempt before allocating so the logits hot path never
    # retries the full vocabulary head after falling back to bf16.
    head._deneb_fp8_build_attempted = True
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
    # Publish the weight last: `_deneb_fp8_w` is the hot-path readiness marker,
    # so any observer that sees it must also be able to read the scale layout.
    head._deneb_fp8_ws = dg_ws
    head._deneb_fp8_w = dg_w
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
        selector_top_k: int | None = None,
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
        _validate_decodable_top_k(valid_vocab_size, selector_top_k)
        self._deneb_valid_vocab_size = valid_vocab_size

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        # The target and drafter may share one lm_head object.  The FP8 copy is
        # shared storage, but the endpoint gates are intentionally independent:
        # an off endpoint must not inherit the other endpoint's FP8 copy.
        use_fp8 = _read_bool_env(self._deneb_fp8_env)
        dg_w = getattr(lm_head, "_deneb_fp8_w", None) if use_fp8 else None
        attempted = getattr(lm_head, "_deneb_fp8_build_attempted", False)
        if dg_w is None and use_fp8 and not attempted:
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
