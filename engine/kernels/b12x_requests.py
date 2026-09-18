"""What a boot asked b12x to compile, recorded -- and the same kernels compiled for another tree before it boots.

A b12x CuTe-DSL kernel takes 2.4-9.5 s of CPU to compile, and a tree that changes any of the dispatcher's key files
compiles all of them again at its first boot. That boot runs inside the fleet window, which is production's downtime:
production opened its door after 150 s instead of 105 s when it recompiled six of them, and Qwen3.8's first fleet boot
was ready in 107.4 s cold against 40.1 s warm (measurements/qwen38_boot_20260918). The compile needs no GPU. The CuTe DSL
lowers to sm_121a, ptxas assembles, and the object is exported to the flashinfer module a boot loads from
(`moe_dispatch._cute_dsl_module` names it by its sources' contents, so what one tree built stays that tree's). What the
compile does need is the list of kernels, and nothing knows that list better than the last boot. So:

    record(md, path, profile)   at boot: every getter call that adds a kernel to the process (built, or read from disk)
                                becomes one JSON line holding the getter, its arguments, the three `configure_*` settings
                                the getters read, and the device's SM and cluster counts
    prebuild                    engine/runtime/b12x_prebuild.py, in a CPU container with the NEW tree: each line replayed
                                through the same getter, so the key and module name are the new tree's own, and that
                                tree's first boot finds the objects (it reads the environment it runs in, so it lives
                                with the runtime, not here: the kernel package reads none, D11)

Nothing here can serve a wrong kernel. A line that no longer fits its getter (an argument renamed) fails alone and is
reported. A replay that lands on another key than the boot's is a wasted compile, because the boot looks up its own key.
The direct micro kernel is not recorded: its disk entry carries a block-dim verdict that only a module loaded on the GPU
can give (moe_dispatch._build_direct_micro_on_disk).

This module lives outside the b12x package on purpose. Importing that package imports the dispatcher, and the prebuild
has to patch the CUDA queries before it does; this module imports nothing but the standard library at its top.
"""
from __future__ import annotations

import functools
import json
import sys
import time
from pathlib import Path

GETTERS = {"_get_static_kernel": "_STATIC_KERNEL_CACHE", "_get_static_kernel_v2": "_STATIC_V2_KERNEL_CACHE",
           "_get_micro_kernel": "_MICRO_KERNEL_CACHE", "_get_dynamic_kernel": "_DYNAMIC_KERNEL_CACHE"}
"""The dispatcher's disk-cached getters and the in-process cache each one fills: a call that grows its cache added a
kernel to this process, and that call is the request."""
CONFIG = ("_GLM53_B12X_STATIC_V2", "_TP_SF6_Q0_ENABLED", "_EP_ZERO_WEIGHT_MICRO_CELL")
"""The settings a profile's lanes put on the dispatcher (`configure_static_v2`, `configure_tp_sf6_q0`,
`configure_ep_zero_weight_micro`) and the getters read."""
ROOT = "st-b12x-requests"
KEYED = ("getter", "args", "kwargs", "config", "device")


def path_under(base, profile: str) -> "Path | None":
    """<base>/st-b12x-requests/<profile>.jsonl, beside the modules it lists, for base = the boot's
    FLASHINFER_WORKSPACE_BASE (the caller reads it: the kernel package reads no environment, D11); None without one --
    a boot the launcher gave no cache records nothing."""
    return Path(base) / ROOT / f"{profile}.jsonl" if base else None


def encode(value):
    """A getter argument as JSON; tuples, dicts, dtypes and devices tagged so they come back as themselves."""
    import torch
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return {"__t": "dtype", "v": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"__t": "device", "v": str(value)}
    if isinstance(value, tuple):
        return {"__t": "tuple", "v": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {"__t": "dict", "v": [[encode(k), encode(v)] for k, v in value.items()]}
    raise TypeError(f"no request encoding for {type(value).__name__}")


def decode(value):
    import torch
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        kind, body = value.get("__t"), value.get("v")
        if kind == "dtype":
            dtype = getattr(torch, body, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"not a torch dtype: {body!r}")
            return dtype
        if kind == "device":
            return torch.device(body)
        if kind == "tuple":
            return tuple(decode(v) for v in body)
        if kind == "dict":
            return {decode(k): decode(v) for k, v in body}
        raise ValueError(f"unknown request encoding {kind!r}")
    return value


def key(line: dict) -> str:
    return json.dumps({name: line[name] for name in KEYED}, sort_keys=True)


def read(paths) -> "list[dict]":
    """The distinct requests in `paths`, in first-seen order; a file that cannot be read and a line that does not parse
    are skipped."""
    lines, seen = [], set()
    for path in paths:
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        for raw in text.splitlines():
            try:
                line = json.loads(raw)
                k = key(line)
            except (ValueError, KeyError, TypeError):
                continue
            if k not in seen:
                seen.add(k)
                lines.append(line)
    return lines


class Recorder:
    """Appends each new request to `path`. It never stops a boot: the first write that fails is said once and ends
    the recording."""

    def __init__(self, md, path, profile: str):
        self.md, self.path, self.profile = md, Path(path), profile
        self.seen = {key(line) for line in read([self.path])}
        self.written, self.failed = 0, None
        self._device = None

    def device(self) -> dict:
        """The SM and cluster counts the getters read, and the arch directory flashinfer keeps this boot's objects under
        (`<base>/.cache/flashinfer/<version>/<arch>/cached_ops`): a prebuild must write where the boot will look."""
        if self._device is None:
            import torch
            try:
                from flashinfer.jit import env as jit_env
                arch = jit_env.FLASHINFER_WORKSPACE_DIR.name
            except ImportError:
                arch = None
            self._device = {"sm": int(self.md.get_num_sm(torch.device("cuda"))),
                            "clusters": {"1": int(self.md.get_max_active_clusters(1))}, "jit_arch": arch}
        return self._device

    def note(self, getter: str, args, kwargs) -> None:
        if self.failed is not None:
            return
        try:
            line = {"getter": getter, "args": encode(list(args)), "kwargs": {k: encode(v) for k, v in kwargs.items()},
                    "config": {name: encode(getattr(self.md, name, None)) for name in CONFIG}, "device": self.device()}
            k = key(line)
            if k in self.seen:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(dict(line, profile=self.profile, when=time.strftime("%Y-%m-%dT%H:%M:%S")),
                                    sort_keys=True) + "\n")
            self.seen.add(k)
            self.written += 1
        except (OSError, TypeError, ValueError) as exc:
            self.failed = f"{type(exc).__name__}: {exc}"
            print(f"  b12x requests: recording stopped at {getter} ({self.failed})", flush=True)


def _wrap(recorder: Recorder, name: str, getter, cache):
    @functools.wraps(getter)
    def call(*args, **kwargs):
        before = len(cache)
        out = getter(*args, **kwargs)
        if len(cache) != before:
            recorder.note(name, args, kwargs)
        return out
    call.b12x_request_recorder = recorder
    return call


def record(md, path, profile: str) -> "Recorder | None":
    """Record this process's kernel requests to `path` (None: nothing). The getters are the dispatcher's module globals
    and every launch reaches them by name, so wrapping the attribute covers every call; installing twice keeps the
    first recorder."""
    if path is None:
        return None
    first = getattr(md, next(iter(GETTERS)))
    if hasattr(first, "b12x_request_recorder"):
        return first.b12x_request_recorder
    recorder = Recorder(md, path, profile)
    for name, cache in GETTERS.items():
        setattr(md, name, _wrap(recorder, name, getattr(md, name), getattr(md, cache)))
    return recorder


DISPATCHER = "engine.kernels.b12x.moe_dispatch"


def record_loaded(profile: str, path) -> "Recorder | None":
    """`record` on the dispatcher this process's lanes imported. A boot whose lanes never loaded b12x -- a test's
    stand-in table -- records nothing and imports nothing (the dispatcher needs flashinfer)."""
    md = sys.modules.get(DISPATCHER)
    return record(md, path, profile) if md is not None else None


__all__ = ["GETTERS", "CONFIG", "ROOT", "DISPATCHER", "path_under", "encode", "decode", "key", "read",
           "Recorder", "record", "record_loaded"]
