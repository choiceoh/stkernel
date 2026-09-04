# SPDX-License-Identifier: Apache-2.0
"""DENEB one-shot cross-node AllReduce shim (Stage-3 integration).

Routes small decode bf16 AllReduce through the host-register RDMA one-shot
kernel (probes/oneshot_ar2.cu, standalone-verified 29us vs NCCL 67us) instead
of NCCL. Wired from the cuda_communicator.py overlay's all_reduce wrapper.

Safety is collective, never rank-local:
  1. env gate VLLM_DSV4_ONESHOT_AR (default 0 = shim is a no-op)
  2. local setup, RDMA connect, and the numerics self-test each use an all-rank
     vote; any failed vote sends every rank back to NCCL together
  3. after the real one-shot path is committed, a local watchdog/runtime fault
     is fatal instead of silently mixing one rank's NCCL with its peers' OSAR

Modes (VLLM_DSV4_ONESHOT_SHADOW, default 1):
  shadow=1: NCCL stays the REAL path. In eager (non-capture) calls the
            one-shot result is computed alongside and compared (divergence
            logged); under CUDA-graph capture the one-shot path is skipped
            entirely so the decode graph is byte-identical to today.
  shadow=0: one-shot IS the real path (goes into the captured decode graph);
            NCCL still serves prefill/large tensors via the size gate.
"""
import hashlib
import logging
import os
import shutil
import time

logger = logging.getLogger("vllm.dsv4.osar")

_MAXEL_DEFAULT = 131072  # 256 KiB bf16; matches the CUDA built-in default


def _resolve_maxel() -> int:
    """Resolve the opt-in one-shot size gate without changing its default."""
    raw = (os.environ.get("VLLM_DSV4_OSAR_MAXEL") or "").strip()
    if not raw:
        return _MAXEL_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "[osar] VLLM_DSV4_OSAR_MAXEL=%r is not an integer; using %d",
            raw,
            _MAXEL_DEFAULT,
        )
        return _MAXEL_DEFAULT
    if value < 1024 or value > 8 << 20 or value % 1024:
        logger.warning(
            "[osar] VLLM_DSV4_OSAR_MAXEL=%d is outside [1024,8388608] "
            "or not a multiple of 1024; using %d",
            value,
            _MAXEL_DEFAULT,
        )
        return _MAXEL_DEFAULT
    return value


_MAXEL = _resolve_maxel()
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


class OneShotFatal(RuntimeError):
    """A post-agreement fault where rank-local NCCL fallback would deadlock."""


_ext = None
_connected = False
_selftest_ok = False
_boot_agreed = False
_disabled = not _ENABLED


def _build_dir(src_md5: str, flags: list) -> str:
    """Where nvcc builds the extension -- on the PERSISTENT cache mount when
    the container has one.

    This build cost 60 s of the 2026-09-02 boot (23:28:27 source md5 ->
    23:29:27 connected) and it used to land in /root/.osar_build, inside the
    container, so every restart paid it again for a .cu that changes on
    deploys, not on restarts. TRITON_CACHE_DIR and VLLM_CACHE_ROOT already
    point at the host mount; this follows them, as glm53_megakernel does.

    The directory name hashes everything the built .so is only valid for:
    the source, the nvcc flags (MAXEL rides in them, and an object built for
    another peer-buffer stride must never be reused), and the torch / CUDA
    pair (an image bump changes the ABI). Any of those moving picks a fresh
    directory and an honest recompile.
    """
    import torch

    root = os.environ.get("VLLM_DSV4_OSAR_BUILD_ROOT")
    if not root:
        for cand in ("/cache", "/root/.cache"):
            if os.path.isdir(cand) and os.access(cand, os.W_OK):
                root = os.path.join(cand, "osar_build")
                break
        else:
            root = "/root/.osar_build"  # no mount: container-local, as before
    key = hashlib.md5("|".join(
        [src_md5, *flags, torch.__version__, str(torch.version.cuda)]
    ).encode()).hexdigest()[:12]
    path = os.path.join(root, key)
    os.makedirs(path, exist_ok=True)
    # Siblings are builds of other sources or MAXEL values, ~2 MB each.
    # Anything untouched for a week is not coming back; failing to prune is
    # never a reason to fail the build.
    try:
        cutoff = time.time() - 7 * 86400
        for name in os.listdir(root):
            stale = os.path.join(root, name)
            if stale != path and os.path.isdir(stale) \
                    and os.path.getmtime(stale) < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
    except OSError:
        pass
    return path


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
        "[osar] source md5=%s kernels=%d maxel=%d (%s)",
        _src_md5,
        _n_kernels,
        _MAXEL,
        _SRC,
    )

    cuda_flags = ["-O2", "-arch=sm_121a"]
    if _MAXEL != _MAXEL_DEFAULT:
        cuda_flags.append(f"-DMAXEL={_MAXEL}")
    # The key covers MAXEL through the flags: never reuse an object whose
    # peer-buffer stride was compiled for a different value, even if torch's
    # extension cache misses the flag.
    build_directory = _build_dir(_src_md5, cuda_flags)

    return load(
        name="dsv4_oneshot_ar",
        sources=[_SRC],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=["-libverbs"],
        build_directory=build_directory,
        verbose=False,
    )


def _bootstrap(comm):
    global _ext, _connected, _boot_agreed, _disabled
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
        # MAXEL agreement, same all-or-none rule and for the same reason.
        # MAXEL is compiled into the kernel and participates in the remote rx
        # stride, and `_eligible` gates on `numel <= _MAXEL` -- so a rank built
        # with a larger value takes the one-shot path for a tensor its peers
        # send through NCCL. That is the split collective this vote exists to
        # prevent, and it cannot be seen locally: every rank boots fine and the
        # md5 line matches, because the source is identical and only -DMAXEL
        # differs. Vote before connect: the peer buffers are sized from it.
        maxel_bounds = torch.tensor([_MAXEL, -_MAXEL], dtype=torch.int64)
        dist.all_reduce(maxel_bounds, op=dist.ReduceOp.MAX, group=comm.cpu_group)
        maxel_hi = int(maxel_bounds[0].item())
        maxel_lo = -int(maxel_bounds[1].item())
        if maxel_hi != maxel_lo:
            _disabled = True
            logger.warning(
                "[osar] MAXEL differs across ranks (%d..%d; this rank %d) -> "
                "every rank stays on NCCL. Set VLLM_DSV4_OSAR_MAXEL "
                "identically on every node, or unset it everywhere.",
                maxel_lo, maxel_hi, _MAXEL,
            )
            return
        _boot_agreed = True

        gathered = [None] * comm.world_size
        dist.all_gather_object(gathered, _ext.local_infos(), group=comm.cpu_group)
        connect_ok = 1
        connect_error = None
        try:
            _ext.connect(gathered)
        except Exception as e:
            connect_ok = 0
            connect_error = e
            logger.warning("[osar] connect failed on rank %d: %r", rank, e)

        # QP transition/proxy startup can fail on only one node. Decide again
        # before any rank runs the one-shot self-test; a local fallback here
        # would split the next collective into OSAR and NCCL.
        connect_votes = torch.tensor([connect_ok], dtype=torch.int32)
        dist.all_reduce(connect_votes, group=comm.cpu_group)
        connected_ranks = int(connect_votes.item())
        if connected_ranks != comm.world_size:
            if connect_ok:
                try:
                    _ext.shutdown()
                except Exception:
                    logger.exception("[osar] shutdown after failed connect vote")
            _boot_agreed = False
            _disabled = True
            logger.warning(
                "[osar] %d/%d ranks connected -> every rank stays on NCCL%s",
                connected_ranks,
                comm.world_size,
                "" if connect_error is None else f" ({connect_error!r})",
            )
            return

        _connected = True
        logger.warning(
            "[osar] connected rank=%d world=%d shadow=%s", rank,
            comm.world_size, _SHADOW)
    except Exception as e:
        if _boot_agreed:
            raise OneShotFatal(
                "one-shot bootstrap failed after all ranks agreed; refusing "
                "rank-local NCCL fallback"
            ) from e
        _disabled = True
        logger.warning("[osar] bootstrap failed before agreement -> NCCL: %r", e)
        return
    _self_test(comm, rank)


def _self_test(comm, rank):
    """Lockstep numerics gate: a single barrier-synchronized one-shot AR vs
    NCCL on a fixed input. This is the ONLY place shadow mode drives the
    one-shot path — matching NCCL's collective lockstep exactly — so the
    stateful tx_seq ring can never desync against production AR traffic."""
    global _selftest_ok, _connected, _boot_agreed, _disabled
    local_ok = 0
    div = float("inf")
    error = None
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
        div = (ref.float() - got.float()).abs().max().item()
        local_ok = int(div <= 0.5)
    except Exception as e:
        error = e
        logger.warning("[osar] self-test error on rank %d: %r", rank, e)

    # A numerical mismatch can be rank-specific. No rank may enter real mode
    # until every rank reports the same successful OSAR result.
    try:
        import torch
        import torch.distributed as dist

        test_votes = torch.tensor([local_ok], dtype=torch.int32)
        dist.all_reduce(test_votes, group=comm.cpu_group)
        passed_ranks = int(test_votes.item())
    except Exception as e:
        raise OneShotFatal(
            "one-shot self-test vote failed after RDMA connect"
        ) from e

    if passed_ranks == comm.world_size:
        _selftest_ok = True
        logger.warning("[osar] self-test PASS div=%.4g (real=%s)", div,
                       not _SHADOW)
        return

    _disabled = True
    _connected = False
    _boot_agreed = False
    try:
        _ext.shutdown()
    except Exception:
        logger.exception("[osar] shutdown after failed self-test vote")
    logger.warning(
        "[osar] self-test passed on %d/%d ranks -> every rank stays on NCCL%s",
        passed_ranks,
        comm.world_size,
        "" if error is None else f" ({error!r})",
    )


def _eligible(t):
    import torch

    return (
        t.dtype == torch.bfloat16
        and t.is_cuda
        and t.is_contiguous()
        and t.numel() <= _MAXEL
    )


# ---- L2 prefetch hints for the peer wait (VLLM_GLM53_AR_PREFETCH) ----------
# The collective's wait (38.7 of 45.5 us, MEASUREMENTS 19차, ~100 per decode
# step) is idle DRAM on every rank. The kernel can spend it pulling the NEXT
# kernel's weights into L2 (k_oneshot's HintArgs); what it should pull is
# learned here, not declared: every weight-streaming launch of the megakernel
# driver notes the tensors it reads, the note is attributed to the most
# recent collective of the current target forward, and the table so learned
# in the eager warmups is what the captured launches bake in. Nothing here
# runs inside a compiled region: begin/end come from the model class above
# the traced graph, and the notes come from custom-op bodies that execute
# eagerly during capture.
def _resolve_prefetch_budget() -> int:
    """Bytes of the next consumer's weights one collective may warm. 0 = off
    (the kernel is byte-identical to before), 1 = 12 MB, or an integer MB in
    1..20 -- L2 is 24 MB and the consumer's activations must fit beside it."""
    raw = (os.environ.get("VLLM_GLM53_AR_PREFETCH") or "0").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return 0
    if raw == "1":
        return 12 << 20
    try:
        mb = int(raw)
    except ValueError:
        logger.warning(
            "[osar] VLLM_GLM53_AR_PREFETCH=%r is not 0, 1 or an MB count; off",
            raw,
        )
        return 0
    if not 1 <= mb <= 20:
        logger.warning(
            "[osar] VLLM_GLM53_AR_PREFETCH=%d MB is outside 1..20; off", mb
        )
        return 0
    return mb << 20


_PREFETCH_BUDGET = _resolve_prefetch_budget()
_HINT_MAX = 8  # OSAR_MAXHINT in the .cu
_in_forward = False
_scope = "target"  # which model's forward: "target" or "drafter"
_ordinal = 0  # collectives seen so far in the current forward
# scope -> ordinal -> [(ptr, nbytes)], adopted; what capture bakes. One table
# per model: the drafter's ten collectives have their own consumers, and a
# shared table would let the richer target forward overwrite them (or the
# drafter's ordinals read the target's rows).
_tables: dict = {}
_cand: dict = {}  # the current forward's candidate, ordinal -> ranges
_cand_bytes: dict = {}


def begin_forward(scope: str = "target"):
    """Called by a model class above its compiled region, once per forward
    (eager warmup and capture alike). This forward's collectives use the
    scope's adopted table; the consumers noting themselves build the
    candidate. `scope` names the model -- the target and the drafter each
    keep their own table."""
    global _in_forward, _scope, _ordinal, _cand, _cand_bytes
    if not _PREFETCH_BUDGET:
        return
    _in_forward = True
    _scope = scope
    _ordinal = 0
    _cand = {}
    _cand_bytes = {}


def end_forward():
    global _in_forward
    if not _PREFETCH_BUDGET:
        return
    _in_forward = False
    table = _tables.get(_scope, {})
    if sum(len(v) for v in _cand.values()) > sum(len(v) for v in table.values()):
        # a decode-shaped forward saw more consumers than the last one did
        # (a prefill forward routes M > 32 to stock and notes nothing)
        _tables[_scope] = _cand
        logger.warning(
            "[osar] prefetch hints learned (%s): %d collectives, %.1f MB in total",
            _scope,
            len(_cand),
            sum(sum(n for _, n in v) for v in _cand.values()) / 1e6,
        )


def note_consumer(tensors):
    """A weight-streaming launch reports the tensors it is about to read.
    Attributed to the most recent collective of this forward, up to the
    budget; the first notes after a collective are the ones that count."""
    if not _PREFETCH_BUDGET or not _in_forward or _ordinal == 0:
        return
    lst = _cand.setdefault(_ordinal, [])
    used = _cand_bytes.get(_ordinal, 0)
    for t in tensors:
        if t is None or len(lst) >= _HINT_MAX or used >= _PREFETCH_BUDGET:
            break
        try:
            if not t.is_cuda:
                continue
            nb = t.numel() * t.element_size()
        except AttributeError:
            continue
        if nb <= 0:
            continue
        take = min(nb, _PREFETCH_BUDGET - used)
        lst.append((int(t.data_ptr()), int(take)))
        used += take
    _cand_bytes[_ordinal] = used


def prefetch_hint_table(scope: str = "target"):
    """The adopted table of one scope, for probes and the boot log."""
    return dict(_tables.get(scope, {}))


def maybe_all_reduce(comm, input_, orig):
    """Return a reduced tensor if handled here, else None (caller uses NCCL).

    One-shot only ever serves in REAL mode (shadow=0), where it replaces NCCL
    at exactly the AR call sites — 4-rank lockstep is automatic. shadow=1 runs
    the boot self-test then stays permanently on NCCL (observe-only)."""
    global _disabled, _ordinal
    if _in_forward:
        # every collective of the forward counts, whichever path serves it:
        # the ordinal is the key the learned hints are filed under
        _ordinal += 1
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
        raise OneShotFatal(
            "one-shot proxy unhealthy after real-mode commit; refusing "
            "rank-local NCCL fallback"
        )
    try:
        hint = _tables.get(_scope, {}).get(_ordinal) if _in_forward else None
        if hint:
            return _ext.oneshot_ar_hint(  # real path + L2 hints, graph + eager
                input_, [p for p, _ in hint], [n for _, n in hint])
        return _ext.oneshot_ar(input_)  # real path (works in graph + eager)
    except Exception as e:
        raise OneShotFatal(
            "one-shot runtime failure after real-mode commit; refusing "
            "rank-local NCCL fallback"
        ) from e
