#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pin the control scripts before queueing; a common checkout may advance later."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


def pin(repo, directory):
    source = repo/'bench'
    files = {p.name:p.read_bytes() for p in sorted(source.iterdir()) if p.suffix in ('.py', '.sh') and p.is_file()}
    key = hashlib.sha256(json.dumps({name:hashlib.sha256(data).hexdigest() for name,data in files.items()}, sort_keys=True).encode()).hexdigest()
    if any((source/name).read_bytes() != data for name,data in files.items()):
        raise ValueError('runner changed while pinning; resubmit the committed version')
    runners = directory/'runners'
    runners.mkdir(parents=True, exist_ok=True)
    target = runners/key
    if target.exists():
        if any((target/'bench'/name).read_bytes() != data for name,data in files.items()):
            raise ValueError('pinned runner integrity check failed')
        return target
    temporary = Path(tempfile.mkdtemp(prefix='.pin-', dir=runners))
    try:
        (temporary/'bench').mkdir()
        for name,data in files.items():
            (temporary/'bench'/name).write_bytes(data)
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists() or any((target/'bench'/name).read_bytes() != data for name,data in files.items()):
                raise
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == '__main__':
    print(pin(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
