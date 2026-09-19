"""Probe: record reads the step loop makes itself, per rank, when a conversation's record is not held in memory.

Four ranks (LocalTP, CPU). Nine conversations park; then conversation 0 is continued by name and by a hint.
From a repo root inside stk-test:
    python3 measurements/parked_record_bound_20260919/probe_loopreads.py as-is   (whatever the tree does)
    python3 measurements/parked_record_bound_20260919/probe_loopreads.py naive   (that, plus a trim at every park)
"""
import sys
import threading

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
import test_engine_serve as T  # noqa: E402
from test_engine_tier import MemoryTier  # noqa: E402
from engine.base.runner import Runner  # noqa: E402

MODE = sys.argv[1]
if MODE == "naive":
    moved = Runner._moved

    def trimmed(self, key, record=None):
        moved(self, key, record)
        if record is not None:
            with self._book:
                while len(self.parked) > self.PARKED_RECORDS_KEPT:
                    self.parked.popitem(last=False)
    Runner._moved = trimmed


class Watched(MemoryTier):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.loop_reads, self.tier_reads, self.other_reads, self.in_loop = [], [], [], False
        self.local = threading.local()

    def run_async(self, fn, *args):
        def job(*a):
            self.local.on = True
            try:
                return fn(*a)
            finally:
                self.local.on = False
        return super().run_async(job, *args)

    def record(self, seq):
        if getattr(self.local, "on", False):
            self.tier_reads.append(seq)
        elif self.in_loop:
            self.loop_reads.append(seq)
        else:
            self.other_reads.append(seq)                  # the continuation scan in `submit`, on rank 0
        return super().record(seq)


def steps(s, tier, n):
    tier.in_loop = True
    for _ in range(n):
        s.once()
    tier.in_loop = False


def rank_main(comm, how):
    tier = Watched()
    s = T.server(comm=comm, rows=2, keep_idle=True, tier=tier)
    kept = Runner.PARKED_RECORDS_KEPT
    for i in range(kept + 1):
        if comm.rank == 0:
            s.submit([3, 4, 5] if i == 0 else [9, 10 + i], 2, 0)
        steps(s, tier, 20)
    held = 0 in s.runner.parked
    if comm.rank == 0:
        if how == "hint":
            again, _ = s.submit([3, 4, 5, 7], 2, 0, continue_history=True)   # rank 0's scan puts it back in its memory
        else:
            again, _ = s.submit([6], 2, 0, conversation=0)
    steps(s, tier, 30)
    out = None
    if comm.rank == 0:
        out = (s.take_result(again), dict(s.continuation_fallbacks))
        s.alive = False
    s.once()
    return comm.rank, held, len(s.runner.parked), tier.loop_reads, tier.tier_reads, tier.other_reads, out


if __name__ == "__main__":
    from engine.base.comm import LocalTP
    for how in ("named", "hint"):
        rows = LocalTP(4, timeout_s=60).run(rank_main, how)
        print(f"[{MODE}] continuation {how}:")
        for rank, held, size, loop_reads, tier_reads, other_reads, out in rows:
            print(f"  rank {rank}: conversation 0 held before: {held}; records held: {size}; "
                  f"record reads on the loop: {loop_reads}; on the tier's thread: {tier_reads}"
                  + (f"; by the scan: {other_reads}; answer {out[0]}, fallbacks {out[1]}" if out is not None else ""))
