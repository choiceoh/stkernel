"""E8M0 block scales that Triton cannot name, converted where -- and only
where -- the Triton kernel is the one consuming them.

V4.1's dense fp8 weights carry `F8_E8M0` scales (`scale_fmt: ue8m0`) at
[32, 32] granularity. The [32, 32] is what makes vLLM select
`TritonFp8BlockScaledMMKernel` instead of the DeepGEMM block-scaled kernel,
and that Triton kernel cannot even NAME the dtype:

    File "triton/_utils.py", line 119, in canonicalize_dtype
        return type_canonicalisation_dict[dtype_str]
    KeyError: 'float8_e8m0fnu'

V4 does not hit this: its [128, 128] scales go to DeepGEMM, which consumes
e8m0 natively. So this is the same root cause as the o-projection -- block
granularity choosing a different kernel -- surfacing in the dense linears.

## Why casting is exact, not a compromise

E8M0 is a bare 8-bit exponent: every representable value is a power of two,
and every one of them is representable in float32. `t.to(torch.float32)` is
lossless, verified in the probe over the whole 8-bit domain rather than
asserted. The cost is 4 bytes instead of 1 on the SCALES only, which for a
[32, 32] block scheme is one scale per 1024 weights -- under 0.4% of the
tensor it belongs to.

## Why only the dense linears

The MoE is on the B12X MXFP4 lane, and MXFP4 wants e8m0 scales: that is the
format, not an accident of storage. Converting those would break the very
path this bring-up wants. So the walk is scoped to modules whose quant method
is the fp8 LINEAR method, and everything under a fused-MoE module is left
alone. A blanket "cast every e8m0 tensor" would boot and then feed the MXFP4
kernel scales in a layout it does not expect.
"""

from __future__ import annotations

import logging
import sys

import torch

logger = logging.getLogger(__name__)

E8M0 = getattr(torch, "float8_e8m0fnu", None)
_CONVERTED = [0]

# Substrings of class names that own MXFP4 experts. Their scales stay e8m0.
_MOE_MARKERS = ("moe", "experts", "fusedmoe")


def _is_moe(module) -> bool:
    name = type(module).__name__.lower()
    return any(marker in name for marker in _MOE_MARKERS)


def convert_module(module) -> int:
    """Cast this module's own e8m0 parameters/buffers to float32. Not recursive."""
    if E8M0 is None:
        return 0
    converted = 0
    for holder, items in ((module._parameters, module._parameters.items()),
                          (module._buffers, module._buffers.items())):
        for name, value in list(items):
            if value is None or value.dtype is not E8M0:
                continue
            dense = value.detach().to(torch.float32)
            if isinstance(value, torch.nn.Parameter):
                new = torch.nn.Parameter(dense, requires_grad=False)
                # vLLM hangs loader metadata off the parameter object; a fresh
                # Parameter without it is a parameter the loader can no longer
                # place, so carry every non-dunder attribute across.
                for attr, val in vars(value).items():
                    if not attr.startswith("__"):
                        try:
                            setattr(new, attr, val)
                        except AttributeError:
                            pass
                holder[name] = new
            else:
                holder[name] = dense
            converted += 1
    return converted


def install(model) -> int:
    """Convert dense-linear e8m0 scales after weights land. Returns the count.

    Wraps `process_weights_after_loading` rather than running once: vLLM calls
    it for both real and dummy loads, and running before it would convert
    scales the quant method is about to overwrite.
    """
    wrapped = 0
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None or _is_moe(module) or _is_moe(quant_method):
            continue
        inner = getattr(quant_method, "process_weights_after_loading", None)
        if inner is None or getattr(inner, "_dsv41_scales", False):
            continue

        def process_weights_after_loading(layer, _inner=inner):
            _inner(layer)
            n = convert_module(layer)
            if n:
                _CONVERTED[0] += n
                if _CONVERTED[0] == n:          # first one only
                    sys.stderr.write(
                        f"[dsv41_scales] first conversion: {n} E8M0 scale(s) "
                        f"-> fp32 on {type(layer).__name__}\n")
                    sys.stderr.flush()

        process_weights_after_loading._dsv41_scales = True
        quant_method.process_weights_after_loading = process_weights_after_loading
        wrapped += 1
    # stderr, not logger.info: vLLM's logging config does not adopt loggers
    # outside the vllm namespace, so an INFO line from here is invisible --
    # which is how this nearly shipped with no evidence that it ran at all.
    sys.stderr.write(
        f"[dsv41_scales] wrapped {wrapped} dense-linear quant method(s)\n")
    sys.stderr.flush()
    if wrapped:
        logger.info(
            "DeepSeek-V4.1: %d dense-linear quant methods will hand their "
            "E8M0 block scales to Triton as float32 (exact; Triton cannot "
            "name float8_e8m0fnu). MXFP4 expert scales are left alone.",
            wrapped)
    return wrapped
