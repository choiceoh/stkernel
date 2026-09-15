"""Fetch the x86_64 check image's inputs into one directory; never uses a GPU.

Two kinds of input, and they are not equally pinned:

  the LOCK          cuda132.x86_64.lock.json's 42 wheels, each verified against its
                    recorded sha256 before it is kept -- the same contract fetch_cuda132.py
                    applies to the ARM64 seed.
  the CLOSURE       the locked wheels' own unpinned requirements (filelock, sympy, tvm_ffi,
                    ...). The ARM64 seed never names these because its bootstrap image
                    already had them; a base image does not, and `pip install --no-deps`
                    over the lock alone leaves an unimportable torch. They are resolved
                    here, at fetch time, so the image build itself can still run
                    --network none. They are NOT hash-pinned: closure.json records the
                    digest each one actually had, which is what a later lock would use.

The closure is resolved WITHOUT letting pip resolve anything the lock pins. That is not
fussiness. Asked to satisfy flashinfer's tvm_ffi with dependency resolution on, pip walks
into torch, decides PyPI's cu12 torch will do, and drags the whole cu12 NVIDIA stack in
behind it; the install then *uninstalls* torch-2.13.0+cu132 to make room. On 2026-09-15
that is exactly what happened, and only install_x86_64.py's post-install version check
caught it. So each round here downloads with --no-deps, reads what arrived, and asks again
for whatever that now needs, minus every name the lock already owns. It closes on its own
and cannot reach a pinned package.

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
ROUNDS = 12          # a dependency graph this shallow closes in three or four; the cap is a stop, not a budget


def canon(name: str) -> str:
    """PEP 503 normalisation: Jinja2, jinja-2 and jinja_2 are one package."""
    return re.sub(r'[-_.]+', '-', name).strip().lower()


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


def requires(wheel: Path) -> list[str]:
    """One wheel's base-install Requires-Dist -- no extras, markers kept as pip wrote them."""
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if not names:
            return []
        text = archive.read(names[0]).decode('utf-8', 'replace')
    out = []
    for line in text.splitlines():
        if not line.startswith('Requires-Dist:'):
            continue
        requirement = line.split(':', 1)[1].strip()
        if ';' in requirement:
            requirement, _, marker = requirement.partition(';')
            if 'extra ==' in marker:              # an optional extra: not part of a base install
                continue
        if requirement.strip():
            out.append(requirement.strip())
    return out


def resolve_closure(wheels: Path, lock: dict) -> list[str]:
    """Everything the locked set needs and does not pin, downloaded transitively but never
    through pip's resolver -- see the module docstring for what that costs when it is on."""
    locked_names = {canon(entry['name']) for entry in lock['wheels']}
    locked_files = {entry['filename'] for entry in lock['wheels']}
    have = {canon(path.name.split('-')[0]): path.name
            for path in wheels.glob('*.whl') if path.name not in locked_files}
    frontier = [path for path in wheels.glob('*.whl') if path.name in locked_files]
    asked: set[str] = set()

    for _ in range(ROUNDS):
        wanted: dict[str, str] = {}
        for path in frontier:
            for requirement in requires(path):
                name = canon(re.split(r'[<>=!~\[ (]', requirement, 1)[0])
                if name and name not in locked_names and name not in have and name not in asked:
                    wanted[name] = requirement
        if not wanted:
            break
        asked |= set(wanted)
        print(f'  closure round: {len(wanted)} new -- {", ".join(sorted(wanted))}', flush=True)
        before = {path.name for path in wheels.glob('*.whl')}
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', *PYTHON_TAG,
                        '--dest', str(wheels), *wanted.values()], check=True)
        frontier = [path for path in wheels.glob('*.whl') if path.name not in before]
        for path in frontier:
            have[canon(path.name.split('-')[0])] = path.name
    else:
        raise RuntimeError(f'the closure did not settle in {ROUNDS} rounds')

    stray = sorted(set(have.values()) - {p.name for p in wheels.glob('*.whl')})
    if stray:
        raise RuntimeError(f'closure names wheels that are not here: {stray}')
    return sorted(have.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--resolve-closure', action='store_true',
                        help='re-resolve the closure from the index instead of taking the one the '
                             'lock pins; then re-run make_x86_64_lock.py --closure-from to write '
                             'the new one down')
    parser.add_argument('--prune', action='store_true',
                        help='delete wheels that are neither locked nor in the resolved closure '
                             '(use after a resolver has polluted the directory)')
    args = parser.parse_args(argv)
    wheels = args.directory / 'wheels'
    wheels.mkdir(parents=True, exist_ok=True)

    lock = json.loads(LOCK.read_text())
    print(f'lock: {len(lock["wheels"])} wheels for {lock["platform"]}', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda entry: fetch(entry, wheels), lock['wheels']))

    if args.prune:
        for path in wheels.glob('*.whl'):
            if path.name not in {entry['filename'] for entry in lock['wheels']}:
                path.unlink()
        print('pruned: only the locked wheels remain; the closure is resolved from scratch', flush=True)

    pinned = lock.get('closure') or []
    if pinned and not args.resolve_closure:
        # The lock names the closure, so this is verification, not resolution: one lock fetches
        # the same bytes today and next month. An unpinned closure made two builds of one lock
        # two different images, which is not a thing this repository tolerates anywhere else.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda entry: fetch(entry, wheels), pinned))
        closure = sorted(entry['filename'] for entry in pinned)
        note = 'pinned by the lock and verified by sha256 on the way in'
    else:
        closure = resolve_closure(wheels, lock)
        note = ('resolved at fetch time with --no-deps per round and NOT pinned; run '
                'make_x86_64_lock.py --closure-from to write it into the lock')
    (args.directory / 'closure.json').write_text(json.dumps(
        {'note': note + '. Installed with --no-deps: nothing here may replace a locked wheel.',
         'files': {name: sha256(wheels / name) for name in closure}}, indent=2) + '\n')
    print(f'closure: {len(closure)} wheels, '
          f'{"pinned by the lock" if pinned and not args.resolve_closure else "resolved fresh"}', flush=True)

    for name in ('cuda132.x86_64.lock.json', 'install_x86_64.py'):
        (args.directory / name).write_bytes((Path(__file__).with_name(name)).read_bytes())
    print(f'ready: {args.directory}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
