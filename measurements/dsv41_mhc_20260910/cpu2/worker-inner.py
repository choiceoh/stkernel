
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
