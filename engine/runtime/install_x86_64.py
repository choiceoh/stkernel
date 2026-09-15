"""Install the x86_64 check image's packages inside the image, never on the host.

The ARM64 installer (install_cuda132.py) edits a vLLM bootstrap image: it lifts DeepGEMM
out of it, removes vLLM, and deletes the CUDA 13.0 toolkit the base shipped. None of that
applies here -- the base is a stock Ubuntu with nothing CUDA on it -- so this installs the
locked set onto an empty tree and then does the one thing the ARM64 installer also does and
the wheels do not: give NVIDIA's runtime wheels the developer SDK link names (`libcudart.so`
beside `libcudart.so.13`) that torch and nvcc link against, and put the toolkit where
CUDA_HOME points.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('the ST x86_64 check image requires Linux x86_64 / Python 3.12')
    lock = json.loads((ROOT / 'cuda132.x86_64.lock.json').read_text())
    wheels = ROOT / 'wheels'
    for entry in lock['wheels']:
        if sha256(wheels / entry['filename']) != entry['sha256']:
            raise RuntimeError(f"corrupt runtime input: {entry['filename']}")

    # The lock, by hash. --no-deps because the lock IS the dependency decision.
    requirements = ROOT / 'locked-requirements.txt'
    requirements.write_text(''.join(
        f"{e['name']}=={e['version']} --hash=sha256:{e['sha256']}\n" for e in lock['wheels']))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-index',
                    '--no-deps', '--require-hashes', '--find-links', str(wheels),
                    '-r', str(requirements)], check=True)
    # The closure, fetched beside the lock. --no-deps here too, and for the same reason it is
    # used to fetch them: with resolution on, pip reads some transitive `torch` requirement,
    # decides a PyPI cu12 torch satisfies it, and UNINSTALLS torch-2.13.0+cu132 to install it
    # (2026-09-15). The closure is already transitively complete; it needs no resolver.
    closure = json.loads((ROOT / 'closure.json').read_text())['files']
    if closure:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-index',
                        '--no-deps', '--find-links', str(wheels),
                        *[f'{wheels}/{name}' for name in closure]], check=True)
    # After the closure, not before: this is the check that catches a resolver that reached
    # past the lock, and it only means anything once everything has been installed.
    for entry in lock['wheels']:
        found = importlib.metadata.version(entry['name'])
        if found != entry['version']:
            raise RuntimeError(f"the lock did not survive installation: {entry['name']} is "
                               f"{found}, the lock pins {entry['version']}")

    packages = Path(sysconfig.get_path('purelib'))
    toolkit = packages / 'nvidia/cu13'
    for relative in ('bin/nvcc', 'bin/ptxas', 'include/cuda_runtime.h',
                     'nvvm/libdevice/libdevice.10.bc', 'lib/libcudart.so.13'):
        if not (toolkit / relative).is_file():
            raise RuntimeError(f'incomplete CUDA SDK: {relative}')
    if not (toolkit / 'lib64').exists():
        (toolkit / 'lib64').symlink_to('lib', target_is_directory=True)
    # SONAMEs only (libcudart.so.13); torch and nvcc link with -lcudart. Shortest name wins
    # when both a SONAME and a fully versioned file exist -- as install_cuda132.py does.
    for library in sorted((toolkit / 'lib').glob('*.so.*'), key=lambda p: (len(p.name), p.name)):
        link = library.with_name(library.name.split('.so.')[0] + '.so')
        if not link.exists():
            link.symlink_to(library.name)
    for name, target in (('cuda-13.2', toolkit), ('cuda-13', Path('cuda-13.2')), ('cuda', Path('cuda-13.2'))):
        path = Path('/usr/local') / name
        if path.is_symlink() or path.exists():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        path.symlink_to(target, target_is_directory=True)
    print(f'installed {len(lock["wheels"])} locked wheels + {len(closure)} closure wheels; '
          f'CUDA_HOME -> {toolkit}')


if __name__ == '__main__':
    main()
