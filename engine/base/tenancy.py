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
from pathlib import Path

MARKER = ".fleet-tenant"


def held_by(directory: "str | Path") -> "str | None":
    """Who this node's tenant state was last written by, or None if nobody has claimed it."""
    mark = Path(directory) / MARKER
    try:
        return mark.read_text().strip() or None
    except OSError:
        return None


def claim(directory: "str | Path", owner: str, *, clear=shutil.rmtree) -> "str | None":
    """Take this node's tenant directory for `owner`, emptying it first if it was somebody else's.

    Returns the owner it was taken from, or None if it was already ours or unclaimed. `clear` is injected so
    the decision can be tested without a filesystem full of KV.
    """
    directory = Path(directory)
    previous = held_by(directory)
    if previous is not None and previous != owner:
        for child in sorted(directory.iterdir()):
            clear(child) if child.is_dir() else child.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MARKER).write_text(owner)
    return previous if previous != owner else None
