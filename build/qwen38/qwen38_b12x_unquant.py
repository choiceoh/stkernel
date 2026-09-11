"""`--moe-backend flashinfer_b12x` names the NVFP4 lane; the MTP head's MoE is bf16.

The Qwen3.8-Flash-Next NVFP4 checkpoint leaves its MTP draft layer unquantized,
and vLLM's moe_backend is one global name. The unquantized oracle maps that
name through a fixed table and raises for anything else -- which is exactly
what ended the first MTP boot after a full weight load:

    ValueError: moe_backend='flashinfer_b12x' is not supported for unquantized
    MoE. Expected one of ['triton', 'batched_triton', 'flashinfer_trtllm',
    'flashinfer_cutlass', 'aiter'].

An unquantized layer cannot run the FP4 kernel whatever the flag says, so the
honest reading of the flag for that layer is "the default": this hook makes
`map_unquantized_backend` return TRITON for the b12x name (logged once) and
leaves every other name to the stock table. `DENEB_B12X_UNQUANT=0` disables.
"""

from __future__ import annotations

import os
import sys

TARGET = "vllm.model_executor.layers.fused_moe.oracle.unquantized"
_DONE = False


def _log(msg: str) -> None:
    sys.stderr.write(f"[qwen38-b12x-unquant] {msg}\n")
    sys.stderr.flush()


def _patch(mod) -> None:
    global _DONE
    if _DONE or not hasattr(mod, "map_unquantized_backend"):
        return
    inner = mod.map_unquantized_backend
    fallback = mod.UnquantizedMoeBackend.TRITON

    def map_unquantized_backend(runner_backend):
        name = getattr(runner_backend, "value", runner_backend)
        if str(name) == "flashinfer_b12x":
            _log(f"unquantized MoE under moe_backend={name}: using {fallback.value} "
                 f"(the FP4 lane has no bf16 kernel; this is the MTP head)")
            return fallback
        return inner(runner_backend)

    mod.map_unquantized_backend = map_unquantized_backend
    _DONE = True
    _log(f"armed on {mod.__name__}")


class _Hook:
    def find_module(self, fullname, path=None):
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET or _DONE:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _Hook) or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            inner = spec.loader.exec_module

            def exec_module(module, _i=inner):
                _i(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec
        return None


def install() -> None:
    if os.environ.get("DENEB_B12X_UNQUANT", "1").strip() in ("0", "false", "no"):
        return
    try:
        if TARGET in sys.modules:
            _patch(sys.modules[TARGET])
        elif not any(isinstance(f, _Hook) for f in sys.meta_path):
            sys.meta_path.insert(0, _Hook())
    except Exception as exc:                                  # noqa: BLE001
        _log(f"install failed: {exc!r}")
