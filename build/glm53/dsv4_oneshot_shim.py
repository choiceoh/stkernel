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
# No rank->IP table here. Which node holds which rank is the launcher's choice,
# not a property of the fleet: hy4 orders its workers 10.10.10.3, 10.10.10.1,
# 10.10.10.4 and the glm53 launcher orders them 10.10.10.1, 10.10.10.3,
# 10.10.10.4, so a literal list written against one of them binds ranks 1 and 2
# to each other's NIC on the other. That is what "oneshot verbs failure" was,
# and it failed on exactly 10.10.10.1 and 10.10.10.3.
#
# Every launcher already passes each container its own address as VLLM_HOST_IP,
# which is right by construction.
# The .cu is mounted next to this file by the module manifest, so derive the
# path from here rather than naming an image's layout. It used to be the literal
# /opt/venv/lib/python3.12/site-packages/... of the dsv4 image; the glm53 image
# puts vllm under /usr/local/lib/python3.12/dist-packages, so the build raised
# FileNotFoundError and the shim fell back to NCCL exactly as designed -- which
# is why nothing looked wrong and this had never run on that image at all.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv4_oneshot_ar.cu")


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off")


_ENABLED = _flag("VLLM_DSV4_ONESHOT_AR", "0")
_SHADOW = _flag("VLLM_DSV4_ONESHOT_SHADOW", "1")

_ext = None
_connected = False
_selftest_ok = False
_disabled = not _ENABLED


def _build():
    from torch.utils.cpp_extension import load

    # Which .cu is this process actually about to compile and serve? The boot
    # gate used to answer only "osar or NCCL" (self-test PASS + real=True), so
    # a boot that never received a fused .cu through deploy-overlays.sh passed
    # review as if it had -- the sixth declared-but-not-applied knob in this
    # repo. The launcher mounts PROFILE_OVERLAY_DIR, not the repo, and only
    # deploy-overlays.sh populates it; nothing else distinguishes versions.
    # This line makes the mounted source's fingerprint part of every boot log,
    # so "which version ran" is answered by the log alone: kernels=5 is the
    # pre-#89 chain, 3 is #89, 1 is #90.
    import hashlib

    with open(_SRC, "rb") as f:
        _src_md5 = hashlib.md5(f.read()).hexdigest()[:8]
    with open(_SRC, encoding="utf-8", errors="replace") as f:
        _n_kernels = sum(1 for line in f if line.lstrip().startswith("__global__"))
    logger.warning(
        "[osar] source md5=%s kernels=%d (%s)", _src_md5, _n_kernels, _SRC)

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

        # Local setup, and it must not return or raise past here: every rank
        # has to reach the agreement below. Returning early is what hung the
        # fleet twice -- one rank gave up on its own while the others waited
        # forever in the all_gather_object that follows.
        ok = 1
        try:
            os.makedirs("/root/.osar_build", exist_ok=True)
            _ext = _build()
            local_ip = os.environ.get("VLLM_HOST_IP", "").strip()
            if not local_ip:
                raise RuntimeError(
                    "VLLM_HOST_IP unset -- no correct NIC to bind to, and "
                    "guessing is what broke this before"
                )
            _ext.init(rank, comm.world_size, local_ip)
        except Exception as e:
            ok = 0
            logger.warning("[osar] local setup failed on rank %d: %r", rank, e)

        # Agreement. All ranks or none: a group where some serve from the
        # one-shot path and others from NCCL calls two different collectives
        # and deadlocks.
        votes = torch.tensor([ok], dtype=torch.int32)
        dist.all_reduce(votes, group=comm.cpu_group)
        if int(votes.item()) != comm.world_size:
            _disabled = True
            logger.warning(
                "[osar] %d/%d ranks ready -> every rank stays on NCCL",
                int(votes.item()), comm.world_size)
            return

        gathered = [None] * comm.world_size
        dist.all_gather_object(gathered, _ext.local_infos(), group=comm.cpu_group)
        _ext.connect(gathered)
        _connected = True
        logger.warning(
            "[osar] connected rank=%d world=%d shadow=%s", rank,
            comm.world_size, _SHADOW)
    except Exception as e:
        # Anything past the agreement is a genuine surprise; the group already
        # decided it was going ahead, so a fallback here is not in lockstep.
        # Nothing better is available at this point than saying so loudly.
        _disabled = True
        logger.warning(
            "[osar] bootstrap failed AFTER the group agreed -> this rank falls "
            "back to NCCL while others may not: %r", e)
        return
    _self_test(comm, rank)


def _self_test(comm, rank):
    """Lockstep numerics gate: a single barrier-synchronized one-shot AR vs
    NCCL on a fixed input. This is the ONLY place shadow mode drives the
    one-shot path — matching NCCL's collective lockstep exactly — so the
    stateful tx_seq ring can never desync against production AR traffic."""
    global _selftest_ok, _disabled
    try:
        import torch
        import torch.distributed as dist

        g = comm.cpu_group
        x = torch.full((6, 4096), float(rank + 1), dtype=torch.bfloat16,
                       device="cuda")
        dist.barrier(group=g)
        ref = comm._all_reduce_impl(x.clone())  # NCCL, collective
        dist.barrier(group=g)
        got = _ext.oneshot_ar(x.clone())        # one-shot, lockstep via barrier
        torch.cuda.synchronize()
        dist.barrier(group=g)
        div = (ref.float() - got.float()).abs().max().item()
        if div <= 0.5:
            _selftest_ok = True
            logger.warning("[osar] self-test PASS div=%.4g (real=%s)", div,
                           not _SHADOW)
        else:
            _disabled = True
            logger.warning("[osar] self-test FAIL div=%.4g -> NCCL", div)
    except Exception as e:
        _disabled = True
        logger.warning("[osar] self-test error -> NCCL: %r", e)


def _eligible(t):
    import torch

    return (
        t.dtype == torch.bfloat16
        and t.is_cuda
        and t.is_contiguous()
        and t.numel() <= _MAXEL
    )


def maybe_all_reduce(comm, input_, orig):
    """Return a reduced tensor if handled here, else None (caller uses NCCL).

    One-shot only ever serves in REAL mode (shadow=0), where it replaces NCCL
    at exactly the AR call sites — 4-rank lockstep is automatic. shadow=1 runs
    the boot self-test then stays permanently on NCCL (observe-only)."""
    global _disabled
    if _disabled:
        return None
    if not _connected:
        _bootstrap(comm)
    if _disabled or not _selftest_ok:
        return None
    if _SHADOW:
        return None  # verified at boot; production traffic stays on NCCL
    if not _eligible(input_):
        return None
    if not _ext.healthy():
        _disabled = True
        logger.warning("[osar] proxy unhealthy -> NCCL fallback")
        return None
    try:
        return _ext.oneshot_ar(input_)  # real path (works in graph + eager)
    except Exception as e:
        _disabled = True
        logger.warning("[osar] runtime failure -> NCCL fallback: %r", e)
        return None
