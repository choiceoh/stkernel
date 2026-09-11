"""Turn b12x's illegal address into an inequality with numbers.

`launch_sm120_dynamic_moe` hands the SM120 CuteDSL kernel three CAPACITIES and
lets it index with them:

    num_tokens, workspace.max_rows,
    workspace.physical_tiles_capacity * workspace.tile_m,
    workspace.task_capacity

If the workspace was sized for a smaller call than this one, the kernel walks
off its own buffers. The only symptom is `cudaErrorIllegalAddress`, and because
CUDA errors are asynchronous it surfaces at whichever kernel launches NEXT --
on Qwen3.8-Flash-Next that was the hyper-connection combine, which had nothing
to do with it. Two hours went into the wrong kernel before
`CUDA_LAUNCH_BLOCKING=1` named this one.

So the check is not a fix. It is the instrument that makes the next step
possible: an exception naming which capacity was short and by how much, raised
before the launch, instead of an illegal address somewhere else.

## Why a wrapper and not an override

This fleet already overrides `moe_dispatch.py` -- for the GLM-5.3 image, whose
FlashInfer is a different build (5,514 lines against this image's 3,047, and a
different preimage). Mounting that file here would replace a 0.6.17 module with
one written against another version. A wrapper needs no preimage, works on
whatever build is present, and reads the geometry from the module it is
wrapping rather than from a copy that can drift.

## Why it can stay on

Host-side arithmetic on integers already in hand -- no device sync, no extra
allocation. `DENEB_B12X_BOUNDS=0` turns it off and restores the illegal
address.
"""

from __future__ import annotations

import os
import sys

TARGET = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
_DONE = False


def _log(msg: str) -> None:
    sys.stderr.write(f"[qwen38_b12x_bounds] {msg}\n")
    sys.stderr.flush()


def check(mod, workspace, *, routed_rows: int, n: int, where: str) -> None:
    """Raise if this call needs more than the workspace was allocated for.

    The geometry comes from the wrapped module's own `_dynamic_task_geometry`,
    so the formula is whatever that build uses. A copy here would be a second
    source of truth that silently stops matching.
    """
    geometry = getattr(mod, "_dynamic_task_geometry", None)
    if geometry is None:
        return
    tile_n = getattr(mod, "_level_tile_n", None)
    kwargs = {"tile_m": workspace.tile_m}
    if tile_n is not None:
        kwargs["tile_n"] = tile_n(workspace.activation_precision)
    want_tiles, _, want_tasks = geometry(workspace.state_E, n, routed_rows,
                                         **kwargs)
    want_rows = want_tiles * workspace.tile_m

    short = []
    cap = getattr(workspace, "routed_rows_capacity", None)
    if cap is not None and routed_rows > cap:
        short.append(f"routed rows {routed_rows} > capacity {cap}")
    if want_rows > workspace.max_rows:
        short.append(f"padded rows {want_rows} > max_rows {workspace.max_rows}")
    if want_tiles > workspace.physical_tiles_capacity:
        short.append(f"tiles {want_tiles} > physical_tiles_capacity "
                     f"{workspace.physical_tiles_capacity}")
    if want_tasks > workspace.task_capacity:
        short.append(f"tasks {want_tasks} > task_capacity "
                     f"{workspace.task_capacity}")
    if not short:
        return
    raise ValueError(
        f"b12x dynamic MoE workspace too small [{where}]: "
        + "; ".join(short)
        + f". state_E={workspace.state_E} n={n} tile_m={workspace.tile_m} "
          f"routed_rows={routed_rows}. Launching would index past the "
          f"workspace, which reports as cudaErrorIllegalAddress at whichever "
          f"kernel runs next. DENEB_B12X_BOUNDS=0 restores that.")


def _patch(mod) -> None:
    global _DONE
    if _DONE or not hasattr(mod, "launch_sm120_dynamic_moe"):
        return
    inner = mod.launch_sm120_dynamic_moe

    def launch_sm120_dynamic_moe(*args, **kwargs):
        ws = kwargs.get("workspace")
        tokens, top_k, n = (kwargs.get("num_tokens"), kwargs.get("top_k"),
                            kwargs.get("n"))
        if ws is not None and None not in (tokens, top_k, n):
            check(mod, ws, routed_rows=int(tokens) * int(top_k), n=int(n),
                  where="launch_sm120_dynamic_moe")
        return inner(*args, **kwargs)

    mod.launch_sm120_dynamic_moe = launch_sm120_dynamic_moe
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
    if os.environ.get("DENEB_B12X_BOUNDS", "1").strip() in ("0", "false", "no"):
        return
    try:
        if TARGET in sys.modules:
            _patch(sys.modules[TARGET])
        elif not any(isinstance(f, _Hook) for f in sys.meta_path):
            sys.meta_path.insert(0, _Hook())
    except Exception as exc:                                  # noqa: BLE001
        _log(f"install failed: {exc!r}")
