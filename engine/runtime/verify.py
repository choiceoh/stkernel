"""Fail closed on runtime ABI drift and report the deployed source identity."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sysconfig

PACKAGES = ('torch', 'torchvision', 'torchcodec', 'triton', 'flashinfer-python', 'tilelang',
            'nvidia-cutlass-dsl', 'nvidia-cutlass-dsl-libs-cu13', 'nvidia-cuda-nvdisasm',
            'nvidia-cudnn-cu13', 'nvidia-nccl-cu13', 'nvidia-cusparselt-cu13', 'nvidia-nvshmem-cu13')


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_versions(expected):
    actual = {name: importlib.metadata.version(name) for name in expected}
    for name, version in actual.items():
        if version != expected[name]:
            raise RuntimeError(f'ST ABI mismatch: {name}={version}, expected {expected[name]}')
    return actual


def compiler_report(expected):
    """Check actual selected tools/libraries without creating a CUDA context."""
    from torch.utils.cpp_extension import CUDA_HOME
    from triton import knobs
    from engine.kernels.common.native_cache import cuda_toolchain_identity
    toolkit = Path(sysconfig.get_path('purelib')) / 'nvidia/cu13'
    if not CUDA_HOME or Path(CUDA_HOME).resolve() != toolkit.resolve():
        raise RuntimeError(f'ST CUDA_HOME must select the pinned CUDA 13.2 SDK: {CUDA_HOME}')
    override = os.environ.get('PYTORCH_NVCC')
    if override and Path(override).resolve() != (toolkit / 'bin/nvcc').resolve():
        raise RuntimeError(f'ST refuses a redirected NVCC: {override}')
    tools = cuda_toolchain_identity(CUDA_HOME)
    for path, version in tools:
        if not Path(path).is_relative_to(toolkit) or not re.search(
                r'V' + re.escape(expected['cuda_compiler']) + r'\b', version):
            raise RuntimeError(f'ST CUDA compiler mismatch: {path}: {version}')
    assembler = (toolkit / 'bin/ptxas').resolve()
    for tool in (knobs.nvidia.ptxas, knobs.nvidia.ptxas_blackwell):
        if Path(tool.path).resolve() != assembler:
            raise RuntimeError(f'ST Triton selected an unpinned assembler: {tool.path}')
    from tilelang.env import CUDA_HOME as tilelang_home
    import deep_gemm
    for name, home in (('TileLang', tilelang_home), ('DeepGEMM', deep_gemm._find_cuda_home())):
        if not home or Path(home).resolve() != toolkit.resolve():
            raise RuntimeError(f'ST {name} CUDA_HOME mismatch: {home}')
    # These version queries are host-only. They also force the loader to reveal
    # the runtime/NVRTC libraries it selected, not just pip's metadata labels.
    runtime = ctypes.CDLL('libcudart.so.13')
    runtime_version = ctypes.c_int()
    if runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version)) or runtime_version.value != 13020:
        raise RuntimeError(f'ST loaded an obsolete CUDA runtime: {runtime_version.value}')
    nvrtc = ctypes.CDLL('libnvrtc.so.13')
    major, minor = ctypes.c_int(), ctypes.c_int()
    if nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) or (major.value, minor.value) != (13, 2):
        raise RuntimeError(f'ST loaded an obsolete NVRTC: {major.value}.{minor.value}')
    loaded = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/' in line and any(name in line for name in
                         ('libcudart.so', 'libnvrtc.so', 'libnvrtc-builtins.so', 'libnvJitLink.so',
                          'libnvvm.so', 'libcublas.so', 'libcublasLt.so'))})
    for path in loaded:
        if not Path(path).resolve().is_relative_to(toolkit.resolve()):
            raise RuntimeError(f'ST loaded a CUDA library outside the pinned SDK: {path}')
    return dict(cuda_home=str(toolkit), tools=tools, triton_ptxas=str(assembler),
                cuda_runtime=runtime_version.value, nvrtc=[major.value, minor.value], libraries=loaded)


def verify(gpu=False):
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "runtime/dependencies.json").read_text())
    lock_path = root / 'runtime/cuda132.lock.json'
    lock = json.loads(lock_path.read_text())
    if sha256(lock_path) != sha256(Path('/opt/st-runtime/cuda132.lock.json')):
        raise RuntimeError('ST runtime seed was built from a different CUDA package lock')
    pins = {name: expected[name] for name in PACKAGES}
    pins.update({e['name']: e['version'] for e in lock['wheels']})
    versions = check_versions(pins)
    if not platform.python_version().startswith(expected["python"] + "."):
        raise RuntimeError("ST Python ABI differs from the pinned runtime")
    if importlib.util.find_spec("vllm") is not None:
        raise RuntimeError("standalone ST runtime must not contain vLLM")
    import torch
    import deep_gemm
    if torch.version.cuda != expected["cuda"]:
        raise RuntimeError(f"ST CUDA ABI mismatch: {torch.version.cuda}")
    compilers = compiler_report(expected)
    library = Path(deep_gemm.__file__).resolve().parent
    provenance = json.loads((library / "ST_SOURCE.json").read_text())
    for name, digest in provenance["files"].items():
        if sha256(library / name) != digest:
            raise RuntimeError(f"DeepGEMM extension/header provenance changed: {name}")
    files = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    source = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    report = dict(passed=True, python=platform.python_version(), cuda=torch.version.cuda,
                  cuda_toolkit=expected['cuda_toolkit'], compiler=compilers,
                  cuda_lock_sha256=sha256(lock_path), torch_git=torch.version.git_version,
                  packages=versions, seed_image=expected["seed_image_id"],
                  engine_source_sha256=source, source_files=files,
                  deep_gemm_files=len(provenance["files"]), vllm_present=False)
    if gpu:
        from engine.profiles.glm53.facts import check_box
        report["device"] = check_box()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.gpu), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
