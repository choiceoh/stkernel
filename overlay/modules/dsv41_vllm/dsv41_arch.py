"""Register DeepseekV41ForCausalLM, without an image that knows about it.

Adding an architecture does NOT need a rebuilt image: `ModelRegistry.
register_model` takes a "module:Class" string, and a .pth at the package root
runs early enough to arm it. Overriding an existing file needs a preimage;
adding one does not. That distinction is the whole reason this profile can be
brought up at all.

The registration cannot happen at .pth time -- vllm is not imported yet, and
importing it from a .pth would drag the whole framework into every unrelated
python process on the node. So this uses the meta-path finder that
`deneb_boot_stamps` already proves out: sit at the front of sys.meta_path,
let the rest of the path resolve the registry module, and register the moment
that module finishes executing. Everything else falls through untouched.

The class itself is named as a STRING, so nothing about V4.1 -- or torch, or
the V4 model it derives from -- is imported until vLLM actually instantiates
the architecture.
"""

from __future__ import annotations

import os
import sys

ARCH = "DeepseekV41ForCausalLM"
TARGET = "vllm.model_executor.models.registry"
# transformers has never heard of `deepseek_v41`, so ModelConfig fails before
# architectures are ever consulted. Registering the config type is a separate
# hook because it must land on `transformers`, which vllm imports first.
CONFIG_TARGET = "transformers"
IMPL = "dsv41_impl:DeepseekV41ForCausalLM"

_DONE = False
_CONFIG_DONE = False


def _log(msg):
    sys.stderr.write(f"[dsv41_arch] {msg}\n")
    sys.stderr.flush()


def _register_config():
    """AutoConfig.register for deepseek_v41. Safe to call more than once."""
    global _CONFIG_DONE
    if _CONFIG_DONE:
        return
    try:
        import dsv41_config
        added = dsv41_config.register()
        _CONFIG_DONE = True
        if added:
            _log(f"registered config type(s) {added}")
    except Exception as exc:                                  # noqa: BLE001
        _log(f"FAILED to register the deepseek_v41 config type: {exc!r}")


def _register(module=None):
    global _DONE
    if _DONE:
        return
    # Read ModelRegistry off the module we just watched execute. `from vllm
    # import ModelRegistry` cannot work here: vllm/__init__.py is still part
    # way through importing this very submodule, so the name it re-exports
    # does not exist yet and the import fails with a message about vllm being
    # broken rather than about import order.
    ModelRegistry = getattr(module, "ModelRegistry", None)
    if ModelRegistry is None:
        try:
            from vllm.model_executor.models.registry import ModelRegistry
        except Exception as exc:                              # noqa: BLE001
            _log(f"ModelRegistry unavailable, not registering: {exc!r}")
            return
    try:
        if ARCH in ModelRegistry.get_supported_archs():
            # The image grew its own support while this overlay was still
            # mounted. Registering over it would silently prefer ours, and
            # which one is better is not a decision to make by import order.
            _log(f"{ARCH} already supported by the image; leaving it alone")
            _DONE = True
            return
        ModelRegistry.register_model(ARCH, IMPL)
        _DONE = True
        _log(f"registered {ARCH} -> {IMPL}")
    except Exception as exc:                                  # noqa: BLE001
        # Loud, not silent: an unregistered architecture fails at model
        # resolution with a message about V4.1 not being supported, which
        # reads like a missing image rather than a failed overlay.
        _log(f"FAILED to register {ARCH}: {exc!r}")


class _PostImport:
    """Register the instant the registry module finishes executing."""

    def find_module(self, fullname, path=None):               # legacy API
        return None

    def find_spec(self, fullname, path=None, target=None):
        want_config = fullname == CONFIG_TARGET and not _CONFIG_DONE
        want_arch = fullname == TARGET and not _DONE
        if not (want_config or want_arch):
            return None
        for finder in sys.meta_path:
            # By CLASS, not identity: if install() ran twice (two site dirs,
            # a re-exec, an overlay copied into more than one root) there are
            # two of these on the path, and an identity check lets each
            # delegate to the other forever. That is a RecursionError at the
            # first vllm import, which reads like a broken image.
            if isinstance(finder, _PostImport) or not hasattr(
                    finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            loader = spec.loader
            inner = loader.exec_module

            def exec_module(module, _inner=inner, _arch=want_arch):
                _inner(module)
                if _arch:
                    _register(module)
                else:
                    _register_config()

            loader.exec_module = exec_module
            return spec
        return None


def install():
    if os.environ.get("DSV41_REGISTER", "1").strip() in ("0", "false", "no"):
        return
    try:
        if CONFIG_TARGET in sys.modules:
            _register_config()
        if TARGET in sys.modules:
            _register(sys.modules[TARGET])  # already imported: register now
        if not (_DONE and _CONFIG_DONE) and not any(
                isinstance(f, _PostImport) for f in sys.meta_path):
            sys.meta_path.insert(0, _PostImport())
    except Exception as exc:                                  # noqa: BLE001
        _log(f"install failed: {exc!r}")
