"""Write the x86_64 twin of cuda132.lock.json; never uses a GPU, installs nothing.

The ARM64 lock is half the seed: the other half -- Triton, TileLang, CuTe DSL, FlashInfer,
cuDNN, NCCL, NVSHMEM and DeepGEMM -- is inherited from the bootstrap image
(glm53:v13-b12x-it), a locally built ARM64 vLLM dev image with no registry digest and
`ai.vllm.build.commit=unknown`. That image cannot be re-pulled, rebuilt for x86_64, or
have its compiled DeepGEMM extension reproduced byte-for-byte, so an x86_64 seed cannot
inherit anything: every package it needs has to be named here, from an index.

That is why this lock is NOT the production image's dependency set and its output must not
be tagged as one. What it can carry is the layer the RTX 5050 can actually run:
engine/kernels/b12x declares `@supported_compute_capability([120, 121])` and imports only
torch and flashinfer -- no DeepGEMM -- so an sm_120 box can exercise that path while every
native lane (mla, dense, oneshot: `-gencode arch=compute_121a`) and the device gate in
engine/kernels/cells.py stay a GB10's.

Three entries cannot match the ARM64 lock, and each is recorded in `deviations`:
  nvidia-cudla     dropped -- Tegra's Deep Learning Accelerator; no x86_64 build exists
                   and no x86_64 host has the hardware
  flashinfer       0.6.18.dev20260819 is a dev build that was never published; the nearest
                   published release is used. FlashInfer is a py3-none-any wheel that JITs
                   its kernels, so the substitution is a version difference, not an arch one
  tilelang         0.1.12 publishes no x86_64 wheel; the nearest version that does is used

Usage:
    python3 engine/runtime/make_x86_64_lock.py engine/runtime/cuda132.x86_64.lock.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import urllib.parse
import urllib.request

PLATFORM = 'linux-x86_64-cp312'
ARM_LOCK = Path(__file__).with_name('cuda132.lock.json')
TORCH_INDEX = 'https://download.pytorch.org/whl/cu132/{project}/'
PYPI = 'https://pypi.org/pypi/{name}/{version}/json'

# Inherited from the bootstrap image on ARM64; named from an index here. Versions are
# engine/runtime/dependencies.json's unless a deviation below says otherwise.
INHERITED = [
    ('triton', '3.7.1'),
    ('nvidia-cutlass-dsl', '4.6.2'),
    ('nvidia-cutlass-dsl-libs-cu13', '4.6.2'),
    ('nvidia-cudnn-cu13', '9.20.0.48'),
    ('nvidia-nccl-cu13', '2.29.7'),
    ('nvidia-nvshmem-cu13', '3.4.5'),
    ('nvidia-cuda-nvdisasm', '13.3.73'),
    ('flashinfer-python', '0.6.18.post1'),
    ('tilelang', '0.1.14'),
]
DROPPED = {'nvidia-cudla'}
DEVIATIONS = [
    {'name': 'nvidia-cudla', 'fleet': '13.2.75', 'here': None,
     'why': 'Tegra-only accelerator: no x86_64 wheel exists and no x86_64 host has the hardware'},
    {'name': 'flashinfer-python', 'fleet': '0.6.18.dev20260819', 'here': '0.6.18.post1',
     'why': 'the fleet pins an unpublished dev build; this is the nearest published release, '
            'and the wheel is py3-none-any (kernels are JIT) so nothing about it is arch-specific'},
    {'name': 'tilelang', 'fleet': '0.1.12', 'here': '0.1.14',
     'why': '0.1.12 publishes no x86_64 wheel; 0.1.14 is the nearest version that does'},
    {'name': 'deep_gemm', 'fleet': '2.6.1 (extracted from the bootstrap image)', 'here': None,
     'why': 'a compiled extension lifted byte-for-byte out of an ARM64 image; not reproducible '
            'on x86_64 at any version. engine/kernels/b12x does not import it'},
]


def open_url(url: str, timeout: float):
    """urlopen with a User-Agent: the cu132 CDN answers a header-less request with 403."""
    request = urllib.request.Request(url, headers={'User-Agent': 'st-runtime-lock/1 (python-urllib)'})
    return urllib.request.urlopen(request, timeout=timeout)


def sha256_of(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


def prefer(filenames: list[str]) -> str | None:
    """The one x86_64 wheel to take: a cp312 build before an abi3 one before a pure-Python one."""
    def rank(name: str) -> tuple:
        return (0 if 'cp312' in name else 1 if 'abi3' in name else 2, len(name), name)
    x86 = sorted((f for f in filenames if 'x86_64' in f and f.endswith('.whl')), key=rank)
    if x86:
        return x86[0]
    pure = sorted((f for f in filenames if f.endswith('-any.whl')), key=len)
    return pure[0] if pure else None


def from_pypi(name: str, version: str) -> dict:
    with open_url(PYPI.format(name=name, version=version), 60) as response:
        data = json.load(response)
    files = {u['filename']: u for u in data.get('urls', []) if u['filename'].endswith('.whl')}
    chosen = prefer(list(files))
    if chosen is None:
        raise RuntimeError(f'{name} {version}: no x86_64 or pure-Python wheel on PyPI')
    entry = files[chosen]
    return {'name': name, 'version': version, 'filename': chosen,
            'url': entry['url'], 'sha256': entry['digests']['sha256']}


def from_torch_index(name: str, version: str) -> dict:
    """The cu132 index publishes no digest, so the wheel is fetched once and hashed here."""
    project = name.replace('-', '_')
    index = TORCH_INDEX.format(project=project)
    with open_url(index, 60) as response:
        page = response.read().decode('utf-8', 'replace')
    wanted = re.escape(version.replace('+', '%2B'))
    pattern = rf'href="([^"]*{re.escape(project)}-{wanted}-cp312-cp312-[^"]*x86_64\.whl)[^"]*"'
    found = re.findall(pattern, page)
    if not found:
        raise RuntimeError(f'{name} {version}: no cp312 x86_64 wheel on the cu132 index')
    # Some entries are absolute (the R2 CDN), others are index-relative: both occur on this page.
    url = urllib.parse.urljoin(index, found[0])
    filename = url.rsplit('/', 1)[-1].split('#')[0].replace('%2B', '+')
    print(f'  hashing {filename} (the cu132 index publishes no digest)', file=sys.stderr, flush=True)
    with tempfile.TemporaryFile() as scratch:
        with open_url(url, 600) as response:
            for block in iter(lambda: response.read(1 << 20), b''):
                scratch.write(block)
        scratch.seek(0)
        digest = sha256_of(scratch)
    return {'name': name, 'version': version, 'filename': filename, 'url': url, 'sha256': digest}


def resolve(entry: tuple[str, str]) -> dict:
    name, version = entry
    return from_torch_index(name, version) if '+cu' in version else from_pypi(name, version)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)

    arm = json.loads(ARM_LOCK.read_text())
    wanted = [(w['name'], w['version']) for w in arm['wheels'] if w['name'] not in DROPPED]
    substitute = {d['name']: d['here'] for d in DEVIATIONS if d['here']}
    wanted = [(n, substitute.get(n, v)) for n, v in wanted] + INHERITED

    with ThreadPoolExecutor(max_workers=6) as pool:
        wheels = list(pool.map(resolve, wanted))
    wheels.sort(key=lambda w: w['name'])

    lock = {
        'schema': arm['schema'],
        'platform': PLATFORM,
        'cuda': arm['cuda'],
        'bootstrap_seed_image_id': None,
        'note': ('The x86_64 set for an sm_120 CHECK image, not the production runtime. It '
                 'inherits nothing from glm53:v13-b12x-it -- that image is ARM64, locally built '
                 'and unreproducible -- so it names every package from an index and carries no '
                 'DeepGEMM. See make_x86_64_lock.py and bench/OST_97X_LANE.md.'),
        'deviations': DEVIATIONS,
        'wheels': wheels,
    }
    args.output.write_text(json.dumps(lock, indent=2) + '\n')
    print(f'{args.output}: {len(wheels)} wheels for {PLATFORM}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
