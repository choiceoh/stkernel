# SPDX-License-Identifier: Apache-2.0
# deneb fork (2026-09-03, glm53_boot_stamps): boot phase timing.
#
# Why: `init engine (profile, create kv cache, warmup model) took 192.52 s`
# is one number for a window that holds at least eight things. The 2026-09-02
# boot could only be split where some component happened to log -- osar 60 s,
# the megakernel 43 s, torch.compile 10 s, cudagraph memory profiling 12 s --
# leaving 63 s in two gaps (23:30:16-23:30:58 and 23:31:16-23:31:31) with no
# marker in them at all. Those gaps are now the largest boot item this repo
# has not attributed, and guessing at them is how the fan investigation went
# wrong. So: stamp the phases themselves.
#
# This file is ADDITIVE -- it replaces nothing in the image. A .pth beside it
# imports it at interpreter start (the image already ships its own
# sitecustomize.py, which must not be shadowed), and it patches the vLLM
# worker's phase methods once they are importable.
#
# It only measures: every wrapper calls the original, returns its value, and
# any failure anywhere leaves the original in place. DENEB_BOOT_STAMPS=0
# disables it outright.
import os
import sys
import time

_T0 = time.monotonic()
_DONE = False
_ORIG = {}


def _log(msg):
    # vllm's logger is not up at .pth time; stderr is what the supervisor
    # captures into glm53.log either way.
    sys.stderr.write(f"[boot-stamp] {msg}\n")
    sys.stderr.flush()


# 2026-09-06 (37차): a phase that runs long gets its stack logged while it
# runs. The 2026-09-05 19:30 boot spent 302 s between the TP group and the
# EP group inside initialize_model_parallel with nothing logged on any rank
# (the 2026-09-04 20:16 and 2026-09-05 00:03 boots: 1 s), and a gap with
# no marker cannot be attributed afterwards. Every DENEB_BOOT_STAMP_SLOW
# seconds (default 60; 0 disables) that a stamped phase is still running,
# the calling thread's stack goes to the log, at most _SLOW_MAX_DUMPS times
# per phase. The watcher is a daemon thread that only reads frames.
try:
    _SLOW = float(os.environ.get("DENEB_BOOT_STAMP_SLOW", "60") or 0)
except ValueError:
    _SLOW = 60.0
_SLOW_MAX_DUMPS = 10


def _dump_stack(label, t_start, tid):
    import traceback

    frame = sys._current_frames().get(tid)
    where = ("".join(traceback.format_stack(frame)[-14:]) if frame is not None
             else "<thread frame not found>\n")
    _log(f"{label} still running after {time.monotonic() - t_start:.0f}s; "
         f"its thread is at:\n{where.rstrip()}")


def _watch(label, t_start, tid, done):
    n = 0
    while not done.wait(_SLOW) and n < _SLOW_MAX_DUMPS:
        n += 1
        try:
            _dump_stack(label, t_start, tid)
        except Exception:
            return


def _start_watch(label, t_start):
    """The Event that ends the watcher, or None when watching is off."""
    if _SLOW <= 0:
        return None
    try:
        import threading

        done = threading.Event()
        threading.Thread(target=_watch, name="boot-stamp-watch", daemon=True,
                         args=(label, t_start, threading.get_ident(), done)).start()
        return done
    except Exception:
        return None


def _wrap_weights_iter(cls, name):
    """Split `Loading weights took Ns` into read and apply.

    That one number covers `model.load_weights(get_all_weights(...))`: the
    loader yields tensors and the model's weight_loader consumes them. On
    2026-09-02 it read 296 s and looked like the biggest item in the boot;
    on 2026-09-03 the same code read 63.5 s, because the first was measured
    while this repo's own benches had all four hosts busy. Timing the
    generator from both sides makes that visible in the log instead of in a
    later argument: time inside next() is the loader, wall time minus that
    is the model.
    """
    fn = getattr(cls, name, None)
    if fn is None or (cls, name) in _ORIG:
        return
    _ORIG[(cls, name)] = fn

    def timed(*a, **kw):
        it = fn(*a, **kw)
        produce = 0.0
        nbytes = 0
        count = 0
        t_start = time.monotonic()
        while True:
            t = time.monotonic()
            try:
                item = next(it)
            except StopIteration:
                produce += time.monotonic() - t
                break
            except Exception:
                produce += time.monotonic() - t
                raise
            produce += time.monotonic() - t
            count += 1
            try:
                tensor = item[1]
                nbytes += tensor.numel() * tensor.element_size()
            except Exception:
                pass
            yield item
        total = time.monotonic() - t_start
        gib = nbytes / float(1 << 30)
        _log(f"weights: {count} tensors, {gib:.1f} GiB | read {produce:.1f}s "
             f"({gib / max(produce, 1e-9):.2f} GiB/s) | "
             f"apply {max(total - produce, 0.0):.1f}s")

    try:
        setattr(cls, name, timed)
    except Exception:
        pass


def _wrap(cls, name, label):
    fn = getattr(cls, name, None)
    if fn is None or (cls, name) in _ORIG:
        return
    _ORIG[(cls, name)] = fn

    def timed(*a, **kw):
        t = time.monotonic()
        lab = label
        group = kw.get("group_name")  # init_model_parallel_group(..., group_name="tp")
        if group:
            lab = f"{label} {group}"
        done = _start_watch(lab, t)
        try:
            return fn(*a, **kw)
        finally:
            if done is not None:
                done.set()
            now = time.monotonic()
            _log(f"{lab} took {now - t:.1f}s "
                 f"(at {now - _T0:.1f}s since interpreter start)")

    try:
        setattr(cls, name, timed)
    except Exception:
        pass


def _patch():
    """Wrap the phases that make up `init engine`. Idempotent, best effort.

    Each module is patched the moment IT appears -- waiting for both would
    hold the whole install hostage to whichever imports last, and the audit
    hook only fires on the NEXT import after a module lands.
    """
    global _DONE
    want = 0
    for mod, pairs in (
        ("vllm.distributed.parallel_state", (
            # module-level functions (cls None = the module itself): the
            # world group, then every model-parallel group by name
            (None, "init_distributed_environment", "dist-world"),
            (None, "init_model_parallel_group", "dist-group"),
            (None, "initialize_model_parallel", "dist-model-parallel"),
        )),
        ("vllm.v1.worker.gpu_worker", (
            ("Worker", "init_device", "init-device"),
            ("Worker", "load_model", "load-model"),
            ("Worker", "determine_available_memory", "profile/determine-memory"),
            ("Worker", "initialize_from_config", "kv-cache-alloc"),
            ("Worker", "compile_or_warm_up_model", "compile+warmup"),
        )),
        ("vllm.v1.worker.gpu.model_runner", (
            ("GPUModelRunner", "profile_run", "profile-run"),
            ("GPUModelRunner", "capture_model", "cudagraph-capture"),
        )),
    ):
        want += len(pairs)
        m = sys.modules.get(mod)
        if m is None:
            continue
        for cls_name, fn_name, label in pairs:
            cls = m if cls_name is None else getattr(m, cls_name, None)
            if cls is not None:
                _wrap(cls, fn_name, label)
    loader = sys.modules.get("vllm.model_executor.model_loader.default_loader")
    want += 1
    if loader is not None:
        cls = getattr(loader, "DefaultModelLoader", None)
        if cls is not None:
            _wrap_weights_iter(cls, "get_all_weights")
    if len(_ORIG) >= want and not _DONE:
        _DONE = True
        _log(f"installed, {len(_ORIG)} phases wrapped")


class _PostImport:
    """Patch each target module the instant it finishes executing.

    An audit hook on "import" was tried first and is not enough: CPython
    raises that event from _find_and_load, so a module already in
    sys.modules never fires one. The patch then rode whatever new import
    happened next, which in a boot is soon but is not a guarantee -- and
    "soon" is exactly the kind of assumption this module exists to replace.

    A meta-path finder at the front sees every fresh import. For the two
    modules that hold the phases it asks the REST of the path for the spec
    (skipping itself, so no recursion), then wraps the loader's exec_module
    to run the patch immediately after the module's own code. Everything
    else falls through untouched by returning None.
    """

    TARGETS = ("vllm.distributed.parallel_state",
               "vllm.v1.worker.gpu_worker", "vllm.v1.worker.gpu.model_runner",
               "vllm.model_executor.model_loader.default_loader")

    def find_spec(self, name, path=None, target=None):
        if _DONE or name not in self.TARGETS:
            return None
        try:
            import importlib.machinery as _m

            rest = [f for f in sys.meta_path if not isinstance(f, _PostImport)]
            spec = None
            for finder in rest:
                found = getattr(finder, "find_spec", None)
                if found is None:
                    continue
                spec = found(name, path, target)
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            real_exec = loader.exec_module

            def exec_module(module, _real=real_exec):
                _real(module)
                try:
                    _patch()
                except Exception:
                    pass

            loader.exec_module = exec_module
            return _m.ModuleSpec(name, loader, origin=spec.origin,
                                 is_package=spec.submodule_search_locations
                                 is not None)
        except Exception:
            return None  # never stand between the boot and its imports


def install():
    if os.environ.get("DENEB_BOOT_STAMPS", "1").strip() in ("0", "false", "no"):
        return
    try:
        # already imported (a re-exec, or an odd import order)? patch now.
        _patch()
        if not _DONE:
            sys.meta_path.insert(0, _PostImport())
    except Exception as e:
        _log(f"install failed, continuing unmeasured: {e!r}")
