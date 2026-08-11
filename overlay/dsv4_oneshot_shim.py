# SPDX-License-Identifier: Apache-2.0
"""DENEB one-shot cross-node AllReduce shim (Stage-3 integration).

Routes small decode bf16 AllReduce through the host-register RDMA one-shot
kernel (probes/oneshot_ar2.cu, standalone-verified 29us vs NCCL 67us) instead
of NCCL. Wired from the cuda_communicator.py overlay's all_reduce wrapper.

THREE independent fallbacks keep production safe:
  1. env gate VLLM_DSV4_ONESHOT_AR (default 0 = shim is a no-op)
  2. any build/bootstrap/runtime exception -> permanent disable, NCCL path
  3. proxy watchdog: unhealthy -> disable, NCCL path

Modes (VLLM_DSV4_ONESHOT_SHADOW, default 1):
  shadow=1: NCCL stays the REAL path. In eager (non-capture) calls the
            one-shot result is computed alongside and compared (divergence
            logged); under CUDA-graph capture the one-shot path is skipped
            entirely so the decode graph is byte-identical to today.
  shadow=0: one-shot IS the real path (goes into the captured decode graph);
            NCCL still serves prefill/large tensors via the size gate.
"""
import logging
import os
import time

logger = logging.getLogger("vllm.dsv4.osar")

_MAXEL = 131072  # 256KB bf16 — matches the extension's MAXEL size gate
IPS = ["10.10.10.2", "10.10.10.3", "10.10.10.1", "10.10.10.4"]
_SRC = (
    "/opt/venv/lib/python3.12/site-packages/vllm/distributed/"
    "device_communicators/dsv4_oneshot_ar.cu"
)


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off")


_ENABLED = _flag("VLLM_DSV4_ONESHOT_AR", "0")
_SHADOW = _flag("VLLM_DSV4_ONESHOT_SHADOW", "1")

_ext = None
_connected = False
_disabled = not _ENABLED
_stats = {"n": 0, "maxdiv": 0.0}


def _build():
    from torch.utils.cpp_extension import load

    return load(
        name="dsv4_oneshot_ar",
        sources=[_SRC],
        extra_cuda_cflags=["-O2", "-arch=sm_121a"],
        extra_ldflags=["-libverbs"],
        build_directory="/root/.osar_build",
        verbose=False,
    )


def _bootstrap(comm):
    global _ext, _connected, _disabled
    if _disabled or _connected:
        return
    try:
        import torch  # noqa: F401
        import torch.distributed as dist

        if "tp" not in getattr(comm, "unique_name", ""):
            return  # only the TP group
        if comm.world_size != 4:
            _disabled = True
            return
        rank = comm.rank_in_group
        os.makedirs("/root/.osar_build", exist_ok=True)
        _ext = _build()
        _ext.init(rank, comm.world_size, IPS[rank])
        gathered = [None] * comm.world_size
        dist.all_gather_object(gathered, _ext.local_infos(), group=comm.cpu_group)
        _ext.connect(gathered)
        _connected = True
        logger.warning(
            "[osar] connected rank=%d world=%d shadow=%s", rank,
            comm.world_size, _SHADOW)
    except Exception as e:
        _disabled = True
        logger.warning("[osar] bootstrap failed -> NCCL fallback: %r", e)


def _eligible(t):
    import torch

    return (
        t.dtype == torch.bfloat16
        and t.is_cuda
        and t.is_contiguous()
        and t.numel() <= _MAXEL
    )


def maybe_all_reduce(comm, input_, orig):
    """Return a reduced tensor if handled here, else None (caller uses NCCL)."""
    global _disabled
    if _disabled:
        return None
    import torch

    if not _connected:
        _bootstrap(comm)
        if not _connected:
            return None
    if not _eligible(input_):
        return None
    if not _ext.healthy():
        _disabled = True
        logger.warning("[osar] proxy unhealthy -> NCCL fallback")
        return None
    try:
        capturing = torch.cuda.is_current_stream_capturing()
        if _SHADOW:
            if capturing:
                return None  # graph: NCCL only, one-shot stays out of the graph
            nccl_out = orig(input_)
            osar_out = _ext.oneshot_ar(input_)
            torch.cuda.synchronize()
            div = (nccl_out.float() - osar_out.float()).abs().max().item()
            _stats["n"] += 1
            if div > _stats["maxdiv"]:
                _stats["maxdiv"] = div
            if _stats["n"] % 500 == 0:
                logger.warning(
                    "[osar] shadow n=%d maxdiv=%.4g", _stats["n"],
                    _stats["maxdiv"])
            return nccl_out
        # real path (shadow=0): one-shot serves, incl. inside the decode graph
        return _ext.oneshot_ar(input_)
    except Exception as e:
        _disabled = True
        logger.warning("[osar] runtime failure -> NCCL fallback: %r", e)
        return None
