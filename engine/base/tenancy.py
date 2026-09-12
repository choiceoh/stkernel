"""What the next tenant of the fleet must not inherit.

D16 keeps idle conversations on NVMe so they survive a boot, and that is right for a restart: the same session's
clients come back and find their turn where they left it. A HANDOVER is not a restart. The previous holder's
conversations belong to clients that are gone, and its prefix tier is warm with boundaries the new holder never
computed -- which is harmless for correctness (a boundary is salted by tenant, so nobody reads another's) and
poison for measurement, because a run that inherits a warm tier is not the run anybody thinks they are timing.

So a boot that finds the fleet in different hands than the last boot on this node starts that node's tenant state
empty. Compile caches are NOT touched: they belong to the hardware and the image, not to whoever reserved it, and
rebuilding them costs a boot for nothing.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

MARKER = ".fleet-tenant"
DISCARD = ".fleet-discard"      # where a handover puts the old tenant's bytes until a thread eats them


def held_by(directory: "str | Path") -> "str | None":
    """Who this node's tenant state was last written by, or None if nobody has claimed it."""
    mark = Path(directory) / MARKER
    try:
        return mark.read_text().strip() or None
    except OSError:
        return None


def claim(directory: "str | Path", owner: str, *, clear=shutil.rmtree, background=True) -> "str | None":
    """Take this node's tenant directory for `owner`, emptying it first if it was somebody else's.

    Returns the owner it was taken from, or None if it was already ours or unclaimed. `clear` is injected so
    the decision can be tested without a filesystem full of KV.

    The emptying is a RENAME, and the bytes go on a thread. Deleting them here took the fleet down: every rank
    runs this, the amount each one has to delete is different (they spilled different things), the ones that
    finish first enter the next collective and wait, and the slow ones are still in `rmtree` -- so NCCL's
    watchdog timed the collective out and all four ranks aborted (2026-09-12, the first boot after a real
    serving run handed over: 34 parked conversations and 3.38 GiB of prefix tier). A rename is one inode
    operation and takes the same negligible time on every node, which is what the ranks need of each other.

    A discard directory left by a boot that died before its thread finished is swept by the next claim, so
    nothing accumulates. If the rename cannot be done -- a filesystem that will not take it -- this falls
    back to deleting in place, because a slow handover is better than a dirty one.
    """
    directory = Path(directory)
    previous = held_by(directory)
    directory.mkdir(parents=True, exist_ok=True)
    doomed = [child for child in sorted(directory.iterdir()) if not child.name.startswith(DISCARD)]
    stale = [child for child in sorted(directory.iterdir()) if child.name.startswith(DISCARD)]
    if previous is not None and previous != owner and doomed:
        bin_ = directory / f"{DISCARD}.{int(time.time() * 1000):x}"
        try:
            bin_.mkdir()
            for child in doomed:
                child.rename(bin_ / child.name)
            stale.append(bin_)
        except OSError:                       # cannot rename here: pay for it now rather than inherit it
            for child in doomed:
                clear(child) if child.is_dir() else child.unlink()
    _discard(stale, clear, background)
    (directory / MARKER).write_text(owner)
    return previous if previous != owner else None


def _discard(paths, clear, background: bool) -> None:
    """Delete what was renamed aside, off the boot's critical path unless a caller wants it synchronous."""
    if not paths:
        return

    def run():
        for path in paths:
            try:
                clear(path)
            except OSError:
                pass                          # the next claim sweeps it; a handover does not fail on cleanup
    if background:
        threading.Thread(target=run, name="tenant-discard", daemon=True).start()
    else:
        run()
