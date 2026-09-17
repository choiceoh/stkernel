"""The native CUDA extensions a GLM-5.3 fleet boot loads, built on every rank before the ranks' first device collective.

A native used to build at its first use, inside the phase that needed it: prefill top-k inside the 32K prefill pass,
the bounded graph inside target capture, mapped staging and the decode queue inside the burst pipeline's capture, the
dense lane at weight preparation. Each rank compiles at its own speed, so on a cold cache the fastest rank waits for
the slowest inside a collective. Main's first cold boot (2026-09-15, 3acae017, cache timestamps on all four nodes)
finished the same builds 17-34 s apart across ranks; rank 1 finished the decode queue 34 s before rank 0, waited in a
one-shot sum, the RoCE retries ran out (WC error 12) and every rank died before the door opened.

Here each rank starts every build at once, one thread per native, while its comm initialises; the ranks then meet at
a preparation rendezvous (base/comm.wait_prepared, 1800 s) before the one-shot transport's first sum. A kept build is a
key check and a load; a new key compiles in parallel with the others instead of one after another. Nothing here touches
the device: the lanes' own entry points (`dense.extension`, `mla.maybe_arm`, ...) still probe and qualify at first use,
and return the module built here.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib
import time

# (name, module, zero-argument entry point that compiles a new key and loads the module without touching the device)
MODULES = (("cublaslt", "engine.kernels.dense.cublaslt", "_build"),
           ("dense", "engine.kernels.dense", "build"),
           ("mla", "engine.kernels.mla", "_build"),
           ("prefill-topk", "engine.kernels.prefill_topk", "_build"),
           ("router-fp32", "engine.kernels.router_fp32", "build"),
           ("decode-topk", "engine.kernels.decode_topk", "_build"),
           ("mapped-staging", "engine.kernels.mapped_staging", "build"),
           ("bounded-graph", "engine.kernels.bounded_graph", "build"),
           ("decode-queue", "engine.kernels.decode_queue", "build"))
ONESHOT = ("one-shot", "engine.kernels.oneshot", "build")        # its sources take the served rails and flag mode

# Natives that exist under engine/kernels but that no serving module binds: a probe's own cell. A cell never runs
# inside a boot, so it cannot make one rank wait for another's compile, and building it here would only lengthen
# every cold boot. Binding one from a lane means moving it into MODULES above.
PROBE_MODULES = (("router-fused", "engine.kernels.router_fused", "build"),)


def builds(oneshot_rails: int, oneshot_inline: bool):
    """(name, build) of every native the fleet boot loads. The modules are imported here, on the caller's thread:
    concurrent first imports of packages that import each other can deadlock the import system."""
    modules = {module: importlib.import_module(module) for _, module, _ in MODULES + (ONESHOT,)}
    importlib.import_module("torch.utils.cpp_extension")
    importlib.import_module("engine.kernels.common.native_cache")
    importlib.import_module("engine.kernels.native_root")
    out = [(name, getattr(modules[module], entry)) for name, module, entry in MODULES]
    oneshot = getattr(modules[ONESHOT[1]], ONESHOT[2])
    out.append((ONESHOT[0], lambda: oneshot(oneshot_rails, inline_flags=oneshot_inline)))
    return out


class NativeBuilds:
    """Every build running at once, one thread each; `wait` returns each one's seconds or raises naming the failures."""

    def __init__(self, builds):
        self.seconds = {}
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(builds)), thread_name_prefix="st-native-build")
        self._futures = {name: self._pool.submit(self._timed, name, build) for name, build in builds}

    def _timed(self, name, build):
        start = time.perf_counter()
        try:
            return build()
        finally:
            self.seconds[name] = round(time.perf_counter() - start, 3)

    def wait(self) -> dict:
        errors = []
        for name, future in self._futures.items():
            try:
                future.result()
            except Exception as exc:                          # noqa: BLE001 -- every failure is named below
                errors.append(f"{name}: {type(exc).__name__}: {str(exc)[-600:]}")
        self._pool.shutdown(wait=True)
        if errors:
            raise RuntimeError("native builds failed -- " + " | ".join(errors))
        return {name: self.seconds[name] for name in self._futures}


def line(rank: int, seconds: dict, wall: float) -> str:
    return (f"  rank{rank}: native builds in {wall:.1f} s (" +
            ", ".join(f"{name} {s:.1f}s" for name, s in seconds.items()) + ")")
