#!/usr/bin/env python3
"""Normal-fleet payload: complete MHC extension AOT in a device-free container.

Execution requires a clean committed source and >=12 GiB host MemAvailable.
No model/weight mounts, image pulls, device flags, inference, or serving changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
INPUTS = (
    'overlay/modules/glm53_megakernel/glm53_megakernel.py',
    'overlay/modules/glm53_megakernel/glm53_megakernel.cu',
    'probes/mk_mhc_geometry_bench.py',
    'tests/test_megakernel_mhc_geometry.py',
    'tests/test_megakernel_mhc_geometry_regressions.py',
    'measurements/dsv41_mhc_20260910/cpu_compile_runner.py',
)

INNER = r'''
import hashlib, json, os, pathlib, runpy, subprocess, sys
root = pathlib.Path('/repo')
out = pathlib.Path('/evidence')
assert os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
devices = sorted(str(p) for pattern in ('nvidia*', 'dri/*', 'kfd')
                 for p in pathlib.Path('/dev').glob(pattern))
assert not devices, ('unexpected device nodes', devices)
import torch
import torch.utils.cpp_extension as ce
assert not torch.cuda.is_initialized()
build_calls = []
original_load = ce.load
expected = ['-O2', '-gencode', 'arch=compute_121a,code=sm_121a',
            '-DMK_GRID_DEF=96', '-DMK_MHC_GRID_DEF=144', '-DMK_NBUF2_DEF=3',
            '-DMK_FP8_PACK2_DEF=0', '-DMK_GEMM_TRANSPOSE_M8_DEF=0',
            '-DMK_GEMM_COMPACT_M8_DEF=0', '-DMK_M8_FASTPATH_DEF=0']
diagnostics = ['-Xptxas=-v', '-Xptxas=--warn-on-spills']
def observed_load(**kw):
    flags = kw['extra_cuda_cflags']
    assert [f for f in flags if f not in diagnostics] == expected, flags
    assert kw['sources'] == [str(root / 'overlay/modules/glm53_megakernel/glm53_megakernel.cu')]
    assert pathlib.Path(kw['build_directory']).is_relative_to(out / 'build')
    kw['extra_cuda_cflags'] = expected + diagnostics
    kw['verbose'] = True
    build_calls.append({k: kw[k] for k in ('name', 'sources', 'build_directory', 'extra_cuda_cflags')})
    return original_load(**kw)
ce.load = observed_load
probe = runpy.run_path(str(root / 'probes/mk_mhc_geometry_bench.py'))
assert probe['main'](['--compile-only', '--output', str(out / 'compile.json')]) == 0
assert len(build_calls) == 1, build_calls
assert not torch.cuda.is_initialized()
assert not any(k == 'vllm' or k.startswith('vllm.') for k in sys.modules)
receipt = json.loads((out / 'compile.json').read_text())
assert receipt['passed'] is True and receipt['compile_only'] is True
assert receipt['gpu_numerics'] is False and receipt['v41_model_equivalence'] is False
assert receipt['rows'] == []
assert receipt['compile']['exports'] == ['run_mhc', 'run_mhc_v41']
assert receipt['compile']['cuda_initialized_before'] is False
assert receipt['compile']['cuda_initialized_after'] is False
assert receipt['compile']['device_nodes_absent'] is True
assert receipt['compile']['load']['cuda_flags'] == expected + diagnostics
assert receipt['compile']['load']['sources'] == build_calls[0]['sources']
artifacts = []
for p in sorted((out / 'build').rglob('*')):
    if p.is_file() and p.suffix in ('.so', '.o', '.ninja'):
        artifacts.append({'path': str(p.relative_to(out)), 'bytes': p.stat().st_size,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
assert any(p['path'].endswith('.so') for p in artifacts)
assert any(p['path'].endswith('.o') for p in artifacts)
report = dict(schema=1, passed=True, devices=devices, cuda_initialized=False,
              torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
              nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'], text=True),
              build_calls=build_calls, artifacts=artifacts,
              gpu_numerics=False, gpu_performance=False, model_equivalence=False)
(out / 'aot.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS full-TU AOT/export admission; no GPU numerics/performance/model claim', flush=True)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_text(argv):
    return subprocess.check_output(argv, text=True).strip()


def source_receipt(source, revision):
    actual = run_text(['git', '-C', str(source), 'rev-parse', 'HEAD'])
    assert actual == revision, ('source revision mismatch', actual, revision)
    assert not run_text(['git', '-C', str(source), 'status', '--porcelain']), 'source is not clean'
    files = []
    for relative in INPUTS:
        path = source / relative
        assert path.is_file() and not path.is_symlink(), relative
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha(path)))
    return dict(revision=actual, clean=True, files=files)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.revision), 'frozen full revision required'
    source, out = args.source.resolve(strict=True), args.output.resolve()
    assert not out.is_relative_to(source), 'evidence must be outside the source tree'
    before = source_receipt(source, args.revision)
    assert sha(Path(__file__)) == sha(source / INPUTS[-1]), 'runner differs from frozen source'
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))) * 1024
    assert available >= 12 * 1024**3, ('need 12 GiB before CPU compile', available)
    image = run_text(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE])
    assert image == IMAGE, ('unexpected local image', image)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inner.py').write_text(INNER)
    name = 'dsv41-mhc-cpu-' + str(os.getpid())
    env = {'NVIDIA_VISIBLE_DEVICES': 'void', 'CUDA_VISIBLE_DEVICES': '',
           'MK_PROBE_NO_GPU': '1', 'MAX_JOBS': '1', 'OMP_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_MODULE_LOADING': 'LAZY',
           'VLLM_GLM53_MK_BUILD_ROOT': '/evidence/build/mk',
           'VLLM_GLM53_MEGAKERNEL': '0', 'VLLM_GLM53_MK_MHC': '0',
           'VLLM_GLM53_MK_GRID': '96', 'VLLM_GLM53_MK_MHC_GRID': '144',
           'VLLM_GLM53_MK_NBUF2': '3', 'VLLM_GLM53_MK_FP8_PACK2': '0',
           'VLLM_GLM53_MK_GEMM_TRANSPOSE_M8': '0', 'VLLM_GLM53_MK_M8_FASTPATH': '0',
           'VLLM_GLM53_MK_PHASE_TS': '0'}
    cmd = ['docker', 'run', '--pull=never', '--rm', '--runtime=runc', '--network=none',
           '--name', name, '--cpuset-cpus=14-15', '--memory=6g', '--memory-swap=6g']
    for key, value in env.items():
        cmd += ['-e', key + '=' + value]
    cmd += ['--mount', f'type=bind,src={source},dst=/repo,readonly',
            '--mount', f'type=bind,src={out},dst=/evidence',
            '--workdir', '/repo', '--entrypoint', 'python3', IMAGE, '-B', '/evidence/inner.py']
    report = dict(schema=1, passed=False, image=image, source_before=before,
                  wrapper_sha256=sha(Path(__file__)), inner_sha256=sha(out / 'inner.py'),
                  mem_available_bytes=available, command=cmd, started_at=time.time(),
                  gpu_numerics=False, gpu_performance=False, model_equivalence=False)
    (out / 'execution.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        with (out / 'compile.log').open('x') as log:
            report['returncode'] = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                   timeout=600).returncode
        assert report['returncode'] == 0, ('AOT failed; preserve compile.log', report['returncode'])
        aot = json.loads((out / 'aot.json').read_text())
        assert aot['passed'] is True and aot['cuda_initialized'] is False
        receipt = json.loads((out / 'compile.json').read_text())
        hashes = {f['path']: f['sha256'] for f in before['files']}
        assert receipt['driver_sha256'] == hashes[INPUTS[0]]
        assert receipt['cuda_sha256'] == hashes[INPUTS[1]]
        report['source_after'] = source_receipt(source, args.revision)
        assert report['source_after'] == before
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        # Only this uniquely named, device-free compiler container is affected.
        subprocess.run(['docker', 'stop', '-t', '1', name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=10)
        report['finished_at'] = time.time()
        (out / 'execution.json').write_text(json.dumps(report, indent=2) + '\n')
        files = [p for p in sorted(out.rglob('*')) if p.is_file()]
        (out / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    print(json.dumps({'passed': True, 'output': str(out), 'execution_sha256': sha(out / 'execution.json')}))


if __name__ == '__main__':
    main()
