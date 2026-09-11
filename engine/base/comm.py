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
up on one box, not on the fleet. The one-shot all-reduce (tp_oneshot_ar,
ours) is a LANE: registered here by name so proof can demand it reported
serving; its kernel is not ported into this file.
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
LANES = {"tp.allreduce.oneshot": "tp_oneshot_ar (ours): one-shot AR over RoCE, prefetch-hinted"}


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
        return cls(world, rank, dist.group.WORLD)

    def all_reduce(self, t):
        if self.world_size == 1:
            return t
        import torch.distributed as dist
        dist.all_reduce(t, group=self.group)
        return t

    def all_gather(self, t, dim=-1):
        if self.world_size == 1:
            return t
        import torch
        import torch.distributed as dist
        parts = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(parts, t, group=self.group)
        return torch.cat(parts, dim=dim)

    def barrier(self):
        if self.world_size > 1:
            import torch.distributed as dist
            dist.barrier(group=self.group)

    def close(self):
        if self.world_size > 1:
            import torch.distributed as dist
            dist.destroy_process_group()


class RankLeft(RuntimeError):
    """A collective broke because another rank died; the cause is on that rank."""


class LocalTP:
    """TP=`world` inside one process: rank r runs on thread r, and every
    collective meets at a barrier. Sums are taken in fp32 in rank order by
    every rank alike, so the four copies of a reduced tensor are identical --
    which is what the fleet's NCCL guarantees and what a check can assert."""

    def __init__(self, world: int = 4, timeout_s: float = 600.0):
        import threading
        self.world_size = world
        self.timeout = timeout_s
        self._barrier = threading.Barrier(world)
        self._slots = [None] * world
        self._results = [None] * world

    def rank(self, r: int):
        return _LocalRank(self, r)

    def _meet(self):
        try:
            self._barrier.wait(timeout=self.timeout)
        except Exception as e:                     # BrokenBarrierError: another rank died -- die too, loudly
            raise RankLeft("LocalTP: a rank left the collective (see its traceback)") from e

    def run(self, fn, *args):
        """fn(rank_comm, *args) on every rank at once; returns the four results
        (any rank's exception is re-raised here)."""
        import threading
        errors = [None] * self.world_size

        def body(r):
            try:
                self._results[r] = fn(self.rank(r), *args)
            except BaseException as e:            # noqa: BLE001
                errors[r] = e
                self._barrier.abort()

        threads = [threading.Thread(target=body, args=(r,), name=f"rank{r}") for r in range(self.world_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        culprits = [(r, e) for r, e in enumerate(errors) if e is not None and not isinstance(e, RankLeft)]
        victims = [(r, e) for r, e in enumerate(errors) if isinstance(e, RankLeft)]
        for r, e in culprits + victims:
            raise RuntimeError(f"rank {r} failed") from e
        return list(self._results)


class _LocalRank:
    def __init__(self, tp: LocalTP, rank: int):
        self.tp, self.rank, self.world_size = tp, rank, tp.world_size
        self.group = None

    def all_reduce(self, t):
        tp = self.tp
        tp._slots[self.rank] = t
        tp._meet()
        total = tp._slots[0].float()
        for r in range(1, tp.world_size):
            total = total + tp._slots[r].float()
        out = total.to(t.dtype)
        tp._meet()                                 # everyone has read; slots may be reused
        t.copy_(out)
        return t

    def all_gather(self, t, dim=-1):
        import torch
        tp = self.tp
        tp._slots[self.rank] = t
        tp._meet()
        out = torch.cat(list(tp._slots), dim=dim)
        tp._meet()
        return out

    def barrier(self):
        self.tp._meet()

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
    assert "tp.allreduce.oneshot" in LANES
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
    def bad(comm, _):
        if comm.rank == 2:
            raise ValueError("rank 2 dies")
        return comm.all_reduce(torch.ones(1))
    try:
        LocalTP(4).run(bad, None); raise AssertionError("a dead rank must surface")
    except RuntimeError as e:
        assert "rank 2" in str(e)
    print(f"  comm: GID rule picks index 3 from a RoCE v2 IPv4-mapped line, world-1 identity, fleet env for rank 3 "
          f"(GID detected: {env.get('NCCL_IB_GID_INDEX', 'n/a')}); LocalTP(4): sums and gathers identical on all four ranks, a dead rank surfaces OK")


if __name__ == "__main__":
    _selfcheck()
