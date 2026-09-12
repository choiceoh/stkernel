#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pin the control scripts before queueing; a common checkout may advance later."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


def source_files(repo):
    source = repo/'bench'
    files = {str(p.relative_to(repo)):p.read_bytes() for p in sorted(source.iterdir())
             if p.suffix in ('.py', '.sh') and p.is_file()}
    # Canonical entrypoints outside bench/ participate in the same policy identity.
    # Execution re-validates against THIS snapshot, so an entry the queue admits must
    # also be pinned here or it passes admission and then stalls before it runs
    # (2026-09-12: three ST reservations paused on 'cannot verify ... No such file').
    import fleet_onepass
    canonical = ('probes/run_ar_consumer_campaign.sh',
                 # Occupancy and yield execute from the pinned control tree.
                 # Without both files, a free lease looks unreadable forever.
                 'launchers/lib/fleet-lease.sh', 'engine/base/fleet_lease.py',
                 *fleet_onepass.ST_ENTRIES, *fleet_onepass.ST_PROBES)
    for relative in canonical:
        if (repo / relative).is_file():
            files[relative] = (repo / relative).read_bytes()
    return files


def file_identity(files):
    return hashlib.sha256(json.dumps({name:hashlib.sha256(data).hexdigest()
        for name,data in files.items()}, sort_keys=True).encode()).hexdigest()


def pin(repo, directory):
    files = source_files(repo)
    key = file_identity(files)
    if any((repo/name).read_bytes() != data for name,data in files.items()):
        raise ValueError('runner changed while pinning; resubmit the committed version')
    runners = directory/'runners'
    runners.mkdir(parents=True, exist_ok=True)
    target = runners/key
    if target.exists():
        if any((target/name).read_bytes() != data for name,data in files.items()):
            raise ValueError('pinned runner integrity check failed')
        return target
    temporary = Path(tempfile.mkdtemp(prefix='.pin-', dir=runners))
    try:
        for name,data in files.items():
            (temporary/name).parent.mkdir(parents=True, exist_ok=True)
            (temporary/name).write_bytes(data)
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists() or any((target/name).read_bytes() != data for name,data in files.items()):
                raise
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == '__main__':
    print(pin(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
