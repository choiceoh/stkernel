"""Fetch the x86_64 check image's inputs into one directory; never uses a GPU.

Two kinds of input, and they are not equally pinned:

  the LOCK          cuda132.x86_64.lock.json's 42 wheels, each verified against its
                    recorded sha256 before it is kept -- the same contract fetch_cuda132.py
                    applies to the ARM64 seed.
  the CLOSURE       the locked wheels' own unpinned requirements (filelock, sympy, tvm_ffi, ...).
                    The ARM64 seed never names these because its bootstrap image already
                    had them; a base image does not, and `pip install --no-deps` over the
                    lock alone leaves an unimportable torch. They are resolved by pip here,
                    at fetch time, so the image build itself can still run --network none.
                    They are NOT hash-pinned yet: they are recorded in closure.json with
                    the digest each one actually had, which is what a later lock would use.

Usage:
    python3 engine/runtime/fetch_x86_64.py ~/.cache/st/cuda132-x86_64
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

LOCK = Path(__file__).with_name('cuda132.x86_64.lock.json')
PYTHON_TAG = ('--only-binary=:all:', '--python-version', '3.12', '--implementation', 'cp',
              '--platform', 'manylinux_2_28_x86_64', '--platform', 'manylinux2014_x86_64',
              '--platform', 'any')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry: dict, directory: Path) -> None:
    name = entry['filename']
    if Path(name).name != name or not name.endswith('.whl'):
        raise ValueError(f'invalid wheel filename: {name}')
    destination = directory / name
    if destination.is_file() and sha256(destination) == entry['sha256']:
        return
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'st-runtime-fetch/1'})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix='.download-', delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(request, timeout=600) as response:
                for block in iter(lambda: response.read(1 << 20), b''):
                    stream.write(block)
        if sha256(temporary) != entry['sha256']:
            raise RuntimeError(f'checksum mismatch: {name}')
        temporary.replace(destination)
        print(f'verified {name}', flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def missing_requirements(directory: Path, lock: dict) -> list[str]:
    """Every locked wheel's Requires-Dist that the lock does not itself pin.

    Not torch's alone: flashinfer needs tvm_ffi, and a lock built only from torch's
    requirements produces an image where `import flashinfer` raises ModuleNotFoundError
    after a clean build (2026-09-15). Anything the lock DOES pin is filtered out, which is
    also what keeps flashinfer's `nvidia-cutlass-dsl==4.7.0` from fighting the pinned 4.6.2
    -- the same discrepancy engine/runtime/README.md records for the ARM64 seed.
    """
    locked = {w['name'].lower().replace('_', '-') for w in lock['wheels']}
    wanted: dict[str, str] = {}
    for entry in lock['wheels']:
        path = directory / entry['filename']
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if not names:
                continue
            text = archive.read(names[0]).decode('utf-8', 'replace')
        for line in text.splitlines():
            if not line.startswith('Requires-Dist:'):
                continue
            requirement = line.split(':', 1)[1].strip()
            if ';' in requirement:                   # an extra or an environment marker
                requirement, _, marker = requirement.partition(';')
                if 'extra ==' in marker:
                    continue
            requirement = requirement.strip()
            name = re.split(r'[<>=!~\[ ]', requirement, 1)[0].lower().replace('_', '-')
            if name and name not in locked:
                wanted.setdefault(name, requirement)
    return sorted(wanted.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    wheels = args.directory / 'wheels'
    wheels.mkdir(parents=True, exist_ok=True)

    lock = json.loads(LOCK.read_text())
    print(f'lock: {len(lock["wheels"])} wheels for {lock["platform"]}', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda entry: fetch(entry, wheels), lock['wheels']))

    requirements = missing_requirements(wheels, lock)
    print(f'closure: the locked set needs {len(requirements)} packages the lock does not pin: '
          f'{", ".join(requirements)}', flush=True)
    if requirements:
        subprocess.run([sys.executable, '-m', 'pip', 'download', *PYTHON_TAG,
                        '--dest', str(wheels), *requirements], check=True)
    # Everything in the directory that the lock does not name -- not "whatever pip added just
    # now". A second run finds the closure already downloaded, and a delta would then be empty:
    # the image would build, install 42 wheels, and have no importable torch.
    named = {entry['filename'] for entry in lock['wheels']}
    closure = sorted(p.name for p in wheels.glob('*.whl') if p.name not in named)
    (args.directory / 'closure.json').write_text(json.dumps(
        {'note': 'resolved by pip at fetch time; digests recorded as fetched, not pinned upstream',
         'files': {name: sha256(wheels / name) for name in closure}}, indent=2) + '\n')
    print(f'closure: {len(closure)} wheels -> closure.json', flush=True)

    for name in ('cuda132.x86_64.lock.json', 'install_x86_64.py'):
        (args.directory / name).write_bytes((Path(__file__).with_name(name)).read_bytes())
    print(f'ready: {args.directory}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
