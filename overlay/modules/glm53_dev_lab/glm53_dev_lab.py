"""29차 item 5 -- the dev lab: a kernel iteration loop that needs no boot.

Worker side. A boot with VLLM_GLM53_DEV_LAB=1 installs, in every rank:
  * a wrapper on CudaGraphManager.run_fullgraph that remembers the last FULL
    batch descriptor a real step replayed (the served decode graph), and
  * Worker.glm53_lab(op, **kw), reached through the API server's
    POST /glm53/lab (glm53_lab_middleware -> engine_client.collective_rpc),
    so every rank runs the op together -- the one-shot AR inside the graphs
    needs all four.

ops:
  info                       -> last descriptor, graphs captured, armed lanes
  replay  {n: 50}            -> replay the served decode graph n times on
                                every rank; us per step from CUDA events.
                                THIS MUTATES KV/state slots of the last
                                batch: dev boots only, no live traffic.
  reload  {src: path}        -> rebuild the megakernel extension from a .cu on
                                the node, swap it in, re-run the self-tests
  recapture                  -> drop the captured graphs and capture again
                                (the new kernels bake into the graphs)
A loop: edit .cu -> scp to /overlays/... on the 4 nodes -> reload -> recapture
-> replay: ~1-2 minutes instead of a 25-minute bracket boot.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)
_INSTALLED = False
_LAST: dict = {"desc": None, "mgr": None, "n_steps": 0}


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:
        logger.exception("[dev-lab] install failed (runner API moved?)")
        return False
    orig = CudaGraphManager.run_fullgraph

    def run_fullgraph(self, desc):
        _LAST["desc"], _LAST["mgr"] = desc, self
        _LAST["n_steps"] += 1
        return orig(self, desc)

    CudaGraphManager.run_fullgraph = run_fullgraph
    Worker.glm53_lab = _lab
    _INSTALLED = True
    logger.warning("[dev-lab] installed: POST /glm53/lab {op: info|replay|reload|recapture}")
    return True


def _rank() -> int:
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def _lab(worker, op: str = "info", **kw) -> dict:
    import torch

    out = {"rank": _rank(), "op": op}
    try:
        runner = worker.model_runner
        mgr = getattr(runner, "cudagraph_manager", None)
        if op == "info":
            from vllm.model_executor.layers import glm53_megakernel as mk
            out.update(desc=str(_LAST["desc"]), steps_seen=_LAST["n_steps"],
                       graphs=len(mgr.graphs) if mgr is not None else None,
                       armed=dict(mk._ARMED), ext=getattr(mk._EXT, "__name__", None))
            return out
        if op == "replay":
            n = int(kw.get("n", 50))
            desc = _LAST["desc"]
            if desc is None or mgr is None or desc not in mgr.graphs:
                raise RuntimeError("no served FULL graph seen yet (send one decode request first)")
            torch.cuda.synchronize()
            for _ in range(3):
                mgr.run_fullgraph(desc)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            e0.record()
            for _ in range(n):
                mgr.run_fullgraph(desc)
            e1.record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            out.update(n=n, desc=str(desc), us_per_step=round(1e3 * e0.elapsed_time(e1) / n, 1),
                       wall_us_per_step=round(1e6 * wall / n, 1))
            return out
        if op == "reload":
            from vllm.model_executor.layers import glm53_megakernel as mk
            src = kw["src"]
            info = mk.rebuild(src)
            out.update(info)
            return out
        if op == "recapture":
            if mgr is None:
                raise RuntimeError("no cudagraph manager")
            # every manager that capture_model() re-captures must be emptied:
            # the target's, and the speculator's own (DEVLAB 01:48: the target
            # re-captured, then the drafter's manager asserted "already
            # captured" because only the target's dict had been cleared)
            managers = [mgr]
            spec = getattr(runner, "speculator", None)
            for name in dir(spec) if spec is not None else []:
                if name.endswith("cudagraph_manager"):
                    m2 = getattr(spec, name, None)
                    if m2 is not None and hasattr(m2, "graphs"):
                        managers.append(m2)
            for m2 in managers:
                m2.graphs.clear()
                if hasattr(m2, "_graphs_captured"):
                    m2._graphs_captured = False
            out["managers_cleared"] = len(managers)
            t0 = time.perf_counter()
            runner.capture_model()
            out.update(graphs=len(mgr.graphs), seconds=round(time.perf_counter() - t0, 1))
            _LAST["desc"] = None   # the next real step re-records it
            return out
        raise ValueError(f"unknown op {op!r}")
    except Exception as e:
        logger.exception("[dev-lab] %s failed", op)
        out["error"] = repr(e)
        return out
