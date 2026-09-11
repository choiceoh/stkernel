"""Make the PLE lookup a graph boundary: add its op to vLLM's splitting ops.

vLLM cuts the piecewise CUDA graph at `CompilationConfig._attention_ops` and
runs those ops eagerly between the pieces; the qwen3_8_flash_next plugin
already lists its short conv there. The n-gram lookup needs the same cut:
`Qwen3_8FlashNextNGramEmbedding.forward_impl` is dynamo-disabled (the
2026-08-28 width patch), which torch.compile's full-graph capture reports as

    torch._dynamo.exc.Unsupported: Skip inlining `torch.compiler.disable()`d function

-- the first TEP=4 boot's compile error -- and the SSD path does host I/O in
it, which no captured graph can replay. ple_layer.py wraps the lookup in
`vllm::qwen3_8_flash_next_ple_embed`; this hook appends that name to the
class-level list before any VllmConfig is built (the list is copied into
`splitting_ops` in `set_splitting_ops_for_v1`), installed from a .pth like the
b12x hooks so it is in place in every process. `DENEB_PLE_SPLIT=0` leaves the
list alone, in which case compiled boots fail exactly as before.
"""

from __future__ import annotations

import os
import sys

TARGET = "vllm.config.compilation"
OP = "vllm::qwen3_8_flash_next_ple_embed"
_DONE = False


def _log(msg: str) -> None:
    sys.stderr.write(f"[qwen38-ple-split] {msg}\n")
    sys.stderr.flush()


def _patch(mod) -> None:
    global _DONE
    if _DONE or not hasattr(mod, "CompilationConfig"):
        return
    ops = mod.CompilationConfig._attention_ops
    if OP not in ops:
        ops.append(OP)
    _DONE = True
    _log(f"{OP} is a splitting op ({len(ops)} listed)")


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
    if os.environ.get("DENEB_PLE_SPLIT", "1").strip() in ("0", "false", "no"):
        return
    try:
        if TARGET in sys.modules:
            _patch(sys.modules[TARGET])
        elif not any(isinstance(f, _Hook) for f in sys.meta_path):
            sys.meta_path.insert(0, _Hook())
    except Exception as exc:                                  # noqa: BLE001
        _log(f"install failed: {exc!r}")
