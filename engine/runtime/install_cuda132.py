"""Install the pinned overlay inside the new image, never on the host."""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

from fetch_cuda132 import sha256


def main():
    if platform.machine() != 'aarch64' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('ST CUDA 13.2 image requires Linux ARM64 / Python 3.12')
    root = Path(__file__).resolve().parent
    lock = json.loads((root / 'cuda132.lock.json').read_text())
    for entry in lock['wheels']:
        if sha256(root / 'wheels' / entry['filename']) != entry['sha256']:
            raise RuntimeError(f"corrupt runtime input: {entry['filename']}")
    # Keep the seed's patched DeepGEMM extension and JIT headers together.
    subprocess.run([sys.executable, str(root / 'promote_deep_gemm.py')], check=True)
    # Torchaudio is unused by ST and its cu130 build requires a different Torch
    # release. Vision and video dependencies have matching cu132 wheels below.
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'vllm', 'torchaudio'], check=True)
    packages = Path(sysconfig.get_path('purelib'))
    shutil.rmtree(packages / 'vllm', ignore_errors=True)
    requirements = root / 'locked-requirements.txt'
    requirements.write_text(''.join(
        f"{e['name']}=={e['version']} --hash=sha256:{e['sha256']}\n" for e in lock['wheels']))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-index',
                    '--no-deps', '--force-reinstall', '--require-hashes', '--find-links', str(root / 'wheels'),
                    '-r', str(requirements)], check=True)
    for entry in lock['wheels']:
        if importlib.metadata.version(entry['name']) != entry['version']:
            raise RuntimeError(f"runtime install did not apply {entry['name']}")
    toolkit = packages / 'nvidia/cu13'
    for relative in ('bin/nvcc', 'bin/ptxas', 'include/cuda_runtime.h',
                     'nvvm/libdevice/libdevice.10.bc', 'lib/libcudart.so.13'):
        if not (toolkit / relative).is_file():
            raise RuntimeError(f'incomplete CUDA SDK: {relative}')
    if not (toolkit / 'lib64').exists():
        (toolkit / 'lib64').symlink_to('lib', target_is_directory=True)
    # NVIDIA's runtime wheels contain SONAMEs (e.g. libcudart.so.13), while
    # Torch/NVCC link using -lcudart. Supply the normal devel SDK link names.
    # Prefer the shortest versioned name if both a SONAME and full version exist.
    for library in sorted((toolkit / 'lib').glob('*.so.*'), key=lambda p: (len(p.name), p.name)):
        link = library.with_name(library.name.split('.so.')[0] + '.so')
        if not link.exists():
            link.symlink_to(library.name)
    # Remove the obsolete toolkit from the new image's filesystem. The seed
    # image and the host toolkit are untouched; no 13.0 library can win PATH or
    # LD_LIBRARY_PATH resolution in the migrated image.
    shutil.rmtree('/usr/local/cuda-13.0')
    for name, target in (('cuda-13.2', toolkit), ('cuda-13', Path('cuda-13.2')),
                         ('cuda', Path('cuda-13.2'))):
        path = Path('/usr/local') / name
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            raise RuntimeError(f'refusing to replace unexpected toolkit directory: {path}')
        path.symlink_to(target, target_is_directory=True)
    Path('/etc/ld.so.conf.d/st-cuda132.conf').write_text(
        str(toolkit / 'lib') + '\n' + str(toolkit / 'nvvm/lib64') + '\n')
    subprocess.run(['ldconfig'], check=True)
    Path('/opt/st-runtime').mkdir(exist_ok=True)
    shutil.copy2(root / 'cuda132.lock.json', '/opt/st-runtime/cuda132.lock.json')


if __name__ == '__main__':
    main()
