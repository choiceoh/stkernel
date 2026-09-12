"""This fleet's process group and collectives (base). Four Sparks, one head,
one fabric -- and nothing else.

vLLM's parallel_state.py carries 2,376 lines to serve every topology; the
overlay changes 17 of them. The engine serves one: TP=4 across srv1..srv4
over RoCE v2, head srv2 (10.10.10.2), bootstrapped by env exactly as the
launchers do it (lib/common-tp4.sh, 44th ledger):

    NCCL_IB_GID_INDEX   picked per node: the GID whose type is "RoCE v2" and
                        whose address is IPv4-mapped (::ffff:) -- it differs
                        by node and by boot, so it is detected, never assumed
    GLOO_SOCKET_IFNAME  the port-1 interface, so the first gloo collective
                        does not sit 302 s on the wrong port (dev-loop memory)
    MASTER_ADDR/PORT    the head

World size 1 is a real mode, not a stub: every collective is the identity,
which is how the base self-checks run on one node. A PROFILE, though, is
written for TP=4 and should be checked at TP=4 on one node too: `LocalTP`
runs the four ranks as four threads on the one GB10, and its collectives are
the real thing (barrier, sum, broadcast) -- so a wrong row/column split shows
up on one box, not on the fleet. NCCL supplies sum and integer MAX.
Vocabulary-parallel greedy selection uses one MAX candidate per token.
Profiles may prepare the owned one-shot transport for aligned BF16 sums;
other dtypes/shapes keep NCCL. Both return the resulting tensor.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import timedelta

NODES = ("10.10.10.2", "10.10.10.1", "10.10.10.3", "10.10.10.4")   # rank 0 owns the rendezvous store
HEAD = NODES[0]
GLOO_IFNAME = "enP2p1s0f0np0"
LANES = {"tp.allreduce.nccl": "NCCL sum/MAX over the TP process group"}


def pick_gid_index(show_gids: str) -> "int | None":
    """The launcher's CT_GID_PRELUDE rule over `show_gids` output: first line
    whose type says RoCE v2 and whose GID is IPv4-mapped."""
    for line in show_gids.splitlines():
        if "RoCE v2" in line and re.search(r"0000:0000:0000:0000:0000:ffff:", line):
            cols = line.split()                 # DEV PORT INDEX GID IPv4 VER NETDEV
            if len(cols) > 2 and cols[2].isdigit():
                return int(cols[2])             # INDEX, not PORT -- both are digits
    return None


def fleet_env(rank: int, world: int = 4, port: int = 29555) -> "dict[str, str]":
    env = {"MASTER_ADDR": HEAD, "MASTER_PORT": str(port), "RANK": str(rank),
           "WORLD_SIZE": str(world), "LOCAL_RANK": "0", "GLOO_SOCKET_IFNAME": GLOO_IFNAME,
           "NCCL_ASYNC_ERROR_HANDLING": "1"}
    try:
        out = subprocess.run(["show_gids"], capture_output=True, text=True, timeout=10).stdout
        gid = pick_gid_index(out)
        if gid is not None:
            env["NCCL_IB_GID_INDEX"] = str(gid)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return env


@dataclass
class Comm:
    world_size: int = 1
    rank: int = 0
    group: object = None
    control: object = None            # a gloo group beside NCCL: the loop's arrivals and votes are host objects (45차 §23 B3)
    transport: object = None

    def prepare_oneshot(self):
        if self.transport is not None:
            raise RuntimeError("one-shot transport is already bound")
        from engine.kernels.oneshot import OneShot
        self.transport = OneShot(self, NODES)

    @classmethod
    def init(cls, rank: "int | None" = None, world: "int | None" = None, *, timeout_s: float = 120.):
        """One process per node. World 1 needs no process group at all."""
        import torch.distributed as dist

        world = world if world is not None else int(os.getenv("WORLD_SIZE", "1"))
        rank = rank if rank is not None else int(os.getenv("RANK", "0"))
        if world <= 0 or not 0 <= rank < world or timeout_s <= 0:
            raise ValueError("comm requires a positive world/timeout and a rank inside that world")
        if world == 1:
            return cls(1, 0, None)
        for k, v in fleet_env(rank, world).items():
            os.environ.setdefault(k, v)
        dist.init_process_group("nccl", world_size=world, rank=rank, timeout=timedelta(seconds=timeout_s))
        # The control plane is gloo over the same sockets: a broadcast of arrivals or a vote on the NCCL group is a
        # device collective, ordered behind every kernel of the step in flight and synchronised to read back -- which
        # is exactly the wait an asynchronous step loop exists to remove.
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_s))
        return cls(world, rank, dist.group.WORLD, control)

    # A collective of this class is device work on the caller's stream (NCCL), or at
    # world 1 no work at all, so a CUDA graph may capture it. LocalTP's cannot --
    # see its own flag. Capture sites must read this before they begin.
    graph_capture_safe = True

    def all_reduce(self, t):
        if self.world_size == 1:
            return t
        if self.transport is not None and self.transport.eligible(t):
            return self.transport.reduce(t)
        import torch.distributed as dist
        dist.all_reduce(t, group=self.group)
        return t

    def reduce_scatter_rows(self, t):
        """TP sum with each rank retaining one contiguous token shard."""
        if self.world_size == 1:
            return t
        import torch
        import torch.distributed as dist
        if t.shape[0] % self.world_size:
            raise ValueError("reduce-scatter requires equally sized token shards")
        out = torch.empty((t.shape[0]//self.world_size, *t.shape[1:]), device=t.device, dtype=t.dtype)
        dist.reduce_scatter_tensor(out,t.contiguous(),group=self.group)
        return out

    def all_gather(self, t, dim=-1):
        if self.world_size == 1:
            return t
        import torch
        import torch.distributed as dist
        if t.ndim == 0 or not -t.ndim <= dim < t.ndim:
            raise IndexError('all_gather dimension is outside the tensor rank')
        dim = dim % t.ndim
        # NCCL writes directly into one rank-major allocation. The list API
        # otherwise creates per-rank outputs before torch.cat copies them again.
        gathered = torch.empty((self.world_size, *t.shape), device=t.device, dtype=t.dtype)
        dist.all_gather_into_tensor(gathered.flatten(0, 1), t.contiguous(), group=self.group)
        return gathered.movedim(0, dim).flatten(dim, dim + 1)

    def all_reduce_max(self, t):
        if self.world_size > 1:
            if self.transport is not None and self.transport.eligible_max(t):
                return self.transport.reduce_max(t)
            import torch.distributed as dist
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.group)
        return t

    def barrier(self):
        if self.world_size > 1:
            import torch.distributed as dist
            dist.barrier(group=self.group)

    def broadcast_object(self, obj):
        """rank 0's `obj` on every rank (the arrivals of a step): on the control group, so it neither waits for the
        device nor makes the device wait."""
        if self.world_size == 1:
            return obj
        import torch.distributed as dist
        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self.control if self.control is not None else self.group)
        return box[0]

    def all_reduce_host(self, values) -> "list[int]":
        """Sum small integer vectors across ranks on the control group (a vote), without touching the device."""
        values = [int(v) for v in values]
        if self.world_size == 1 or not values:
            return values
        import torch
        import torch.distributed as dist
        t = torch.tensor(values, dtype=torch.int64)
        dist.all_reduce(t, group=self.control if self.control is not None else self.group)
        return [int(v) for v in t.tolist()]

    def close(self):
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        if self.world_size > 1:
            import torch.distributed as dist
            dist.destroy_process_group()


class RankLeft(RuntimeError):
    """A collective broke because another rank died; the cause is on that rank."""


class _LocalRun:
    """One invocation owns its barrier, exchange buffers and queued kernels."""

    def __init__(self, world, timeout):
        import queue
        import threading
        self.owner = threading.current_thread()
        self.barrier = threading.Barrier(world, timeout=timeout)
        self.slots, self.results = [None] * world, [None] * world
        self.jobs = queue.Queue()
        self.workers = ()
        self.failed = False

    def abort(self):
        self.failed = True
        self.barrier.abort()

    def meet(self):
        import threading
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError as e:
            raise RankLeft("LocalTP: a rank left or timed out at the collective") from e


class LocalTP:
    # NOT capturable. Every collective below crosses ranks through a host
    # threading.Barrier and a Python slot assignment: under capture the barrier runs
    # once, at capture time, and leaves no node in the graph, while the peer tensors
    # are read at whatever addresses they held then. A replay would race with no
    # ordering at all. Graph capture must refuse this comm (D3: die, never fall back).
    graph_capture_safe = False

    """TP=`world` inside one process: rank r runs on thread r, and every
    collective meets at a barrier. Sums are taken in fp32 in rank order by
    every rank alike, so the four copies of a reduced tensor are identical --
    which is what the fleet's NCCL guarantees and what a check can assert."""

    def __init__(self, world: int = 4, timeout_s: float = 600.0):
        import math
        import threading
        if type(world) is not int or world <= 0 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("LocalTP needs a positive world and finite positive timeout")
        self.world_size = world
        self.timeout = timeout_s
        self._gate = threading.Lock()
        self._active = None

    def on_main(self, fn, *args, **kwargs):
        """Run fn on the thread that called `run` and return its result.
        DeepGEMM's JIT runtime (the served mHC lane) raises
        CUDA_ERROR_INVALID_VALUE from any other thread -- measured in
        probes/mhc_lane_isolate.py -- so a served lane goes through here."""
        return self._dispatch(self._active, fn, *args, **kwargs)

    def _dispatch(self, run, fn, *args, **kwargs):
        import threading
        if run is None or run is not self._active:
            raise RuntimeError("LocalTP dispatch requires its active run")
        caller = threading.current_thread()
        if caller is not run.owner and caller not in run.workers:
            raise RuntimeError("LocalTP dispatch called from a foreign thread")
        if run.failed:
            raise RankLeft("LocalTP: this run has failed")
        if caller is run.owner:
            return fn(*args, **kwargs)
        done, box = threading.Event(), {}

        def job():
            try:
                if run.failed:
                    raise RankLeft("LocalTP: queued kernel cancelled after run failure")
                box["out"] = fn(*args, **kwargs)
            except BaseException as e:            # noqa: BLE001
                run.abort()
                box["err"] = e
            finally:
                done.set()
        run.jobs.put(job)
        done.wait()
        if "err" in box:
            raise box["err"]
        return box["out"]

    def rank(self, r: int):
        if type(r) is not int or not 0 <= r < self.world_size:
            raise ValueError("LocalTP rank is outside its world")
        return _LocalRank(self, r, self._active)

    def run(self, fn, *args):
        """fn(rank_comm, *args) on every rank at once; returns the four results
        (any rank's exception is re-raised here)."""
        import queue
        import threading
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("LocalTP.run is already active; overlapping or nested runs are forbidden")
        run = None
        try:
            run = _LocalRun(self.world_size, self.timeout)
            self._active = run
            errors = [None] * self.world_size

            def body(r):
                try:
                    run.results[r] = fn(self.rank(r), *args)
                except BaseException as e:        # noqa: BLE001
                    errors[r] = e
                    run.abort()

            def drain():
                while any(t.is_alive() for t in run.workers):
                    try:
                        run.jobs.get(timeout=0.02)()
                    except queue.Empty:
                        pass
                while not run.jobs.empty():
                    run.jobs.get()()
                for t in run.workers:
                    if t.ident is not None:
                        t.join()

            run.workers = tuple(threading.Thread(target=body, args=(r,), name=f"rank{r}")
                                for r in range(self.world_size))
            try:
                for t in run.workers:
                    t.start()
                drain()
            except BaseException:
                run.abort()
                drain()                         # release queued callers and join started ranks before retiring
                raise
            culprits = [(r, e) for r, e in enumerate(errors) if e is not None and not isinstance(e, RankLeft)]
            victims = [(r, e) for r, e in enumerate(errors) if isinstance(e, RankLeft)]
            for r, e in culprits + victims:
                raise RuntimeError(f"rank {r} failed") from e
            return list(run.results)
        finally:
            self._active = None
            if run is not None:
                run.slots.clear()
                run.results.clear()
                run.workers = ()
            self._gate.release()


class _LocalRank:
    def __init__(self, tp: LocalTP, rank: int, run):
        self.tp, self.rank, self.world_size = tp, rank, tp.world_size
        self.group = None
        self._run = run

    def _state(self):
        import threading
        run = self._run
        if run is None or run is not self.tp._active:
            raise RuntimeError("LocalTP rank belongs to an inactive or completed run")
        if threading.current_thread() is not run.workers[self.rank]:
            raise RuntimeError("LocalTP rank used from a foreign thread")
        if run.failed:
            raise RankLeft("LocalTP: this run has failed")
        return run

    def all_reduce(self, t):
        return self._reduce(t, maximum=False)

    def all_reduce_max(self, t):
        return self._reduce(t, maximum=True)

    def _reduce(self, t, maximum):
        import torch
        run = self._state()
        run.slots[self.rank] = t
        run.meet()
        total = run.slots[0] if maximum else run.slots[0].float()
        for r in range(1, self.world_size):
            total = torch.maximum(total, run.slots[r]) if maximum else total + run.slots[r].float()
        out = total.to(t.dtype)
        run.meet()                                # everyone has read; slots may be reused
        t.copy_(out)
        return t

    def all_gather(self, t, dim=-1):
        import torch
        run = self._state()
        run.slots[self.rank] = t
        run.meet()
        out = torch.cat(list(run.slots), dim=dim)
        run.meet()
        return out

    def barrier(self):
        self._state().meet()

    def broadcast_object(self, obj):
        run = self._state()
        if self.rank == 0:
            run.slots[0] = obj
        run.meet()
        out = run.slots[0]
        run.meet()
        return out

    def on_main(self, fn, *args, **kwargs):
        return self.tp._dispatch(self._state(), fn, *args, **kwargs)

    def close(self):
        pass


def _selfcheck() -> None:
    import torch
    sample = """DEV     PORT    INDEX   GID                                     IPv4            VER     DEV
---     ----    -----   ---                                     ------------    ---     ---
mlx5_0  1       0       fe80:0000:0000:0000:0000:0000:0000:0001                 v1      enp1s0
mlx5_0  1       1       fe80:0000:0000:0000:0000:0000:0000:0001                 v2      enp1s0
mlx5_0  1       2       0000:0000:0000:0000:0000:ffff:0a0a:0a04 10.10.10.4      v1      enp1s0
mlx5_0  1       3       0000:0000:0000:0000:0000:ffff:0a0a:0a04 10.10.10.4      RoCE v2 enp1s0
"""
    assert pick_gid_index(sample) == 3, pick_gid_index(sample)
    assert pick_gid_index("nothing here") is None
    c = Comm.init(rank=0, world=1)
    x = torch.arange(4.0)
    assert torch.equal(c.all_reduce(x.clone()), x) and torch.equal(c.all_gather(x), x)
    env = fleet_env(3)
    assert env["MASTER_ADDR"] == HEAD and env["RANK"] == "3" and env["GLOO_SOCKET_IFNAME"] == GLOO_IFNAME
    assert "tp.allreduce.nccl" in LANES
    # TP=4 on one box: four threads, real sums, identical copies
    tp = LocalTP(4)
    def rank_fn(comm, base):
        mine = base + comm.rank                            # rank r holds base + r
        red = comm.all_reduce(mine.clone())
        gat = comm.all_gather(torch.full((2,), float(comm.rank)), dim=0)
        return red, gat
    outs = tp.run(rank_fn, torch.ones(3))
    assert all(torch.equal(o[0], torch.full((3,), 4.0 + 6.0)) for o in outs), [o[0] for o in outs]   # 4*1 + (0+1+2+3)
    assert all(torch.equal(o[1], torch.tensor([0., 0., 1., 1., 2., 2., 3., 3.])) for o in outs)
    import threading
    main = threading.current_thread()
    def needs_main(comm, _):
        where = comm.on_main(lambda: threading.current_thread())
        return where is main and threading.current_thread() is not main
    assert all(LocalTP(4).run(needs_main, None)), "on_main must run on the calling thread of run()"
    assert LocalTP(4).run(lambda comm, _: comm.broadcast_object({"from": comm.rank}), None) == [{"from": 0}] * 4
    def bad(comm, _):
        if comm.rank == 2:
            raise ValueError("rank 2 dies")
        return comm.all_reduce(torch.ones(1))
    try:
        LocalTP(4).run(bad, None); raise AssertionError("a dead rank must surface")
    except RuntimeError as e:
        assert "rank 2" in str(e)
    print(f"  comm: GID rule picks index 3 from a RoCE v2 IPv4-mapped line, world-1 identity, fleet env for rank 3 "
          f"(GID detected: {env.get('NCCL_IB_GID_INDEX', 'n/a')}); LocalTP(4): sums and gathers identical on all four ranks, on_main runs on the main thread, a dead rank surfaces OK")


if __name__ == "__main__":
    _selfcheck()
