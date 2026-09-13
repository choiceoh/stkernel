"""Compile the actual SF6 expansion and MoE readers without a CUDA device.

Run in a resource-capped runc container through fleet --cpu. This proves
lowering and resource allocation only; it cannot prove numerics or speed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-m64', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise ValueError('compile evidence must use a fresh directory')
    assert not list(Path('/dev').glob('nvidia*')), 'CPU compilation must expose no GPU'
    os.environ.update(CUTE_DSL_ARCH='sm_121a', CUTE_DSL_KEEP='ptx,cubin',
                      CUTE_DSL_DUMP_DIR=str(output / 'cute'),
                      CUTE_DSL_CACHE_DIR=str(output / 'cute-cache'),
                      CUTE_DSL_DISABLE_FILE_CACHING='1',
                      TRITON_CACHE_DIR=str(output / 'triton-cache'))
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    assert not torch.cuda.is_initialized()
    # Import-time availability/capability queries do not create a context.
    # The runner denies device nodes and the final assertion catches initialization.
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *a, **kw: (12, 1)
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels.b12x.moe_sf6_prefill_scales_kernel import expand
    report = dict(status='RUNNING', scope='CPU compilation only', gpu_used=False,
                  torch=torch.__version__, triton=triton.__version__, variants=[])
    started = time.monotonic()
    try:
        for kind, k_tiles in (('fc1', 16), ('fc2', 4)):
            compiled = triton.compile(ASTSource(expand, {'Packed': '*u8', 'Out': '*u8'},
                constexprs={'K_TILES': k_tiles, 'FC2': kind == 'fc2'}),
                target=GPUTarget('cuda', 121, 32), options={'num_warps': 4})
            for extension in ('ptx', 'cubin'):
                data = compiled.asm[extension]
                path = output / (kind + '.' + extension)
                path.write_bytes(data.encode() if isinstance(data, str) else data)
            report['variants'].append(dict(kind=kind, cubin_sha256=digest(output / (kind+'.cubin')),
                                           shared_bytes=compiled.metadata.shared))
        md.get_num_sm = lambda *a: 48
        md.get_max_active_clusters = lambda *a: 48
        def build_reader(module, name, build, **kw):
            # Create and pass each reader's artifact directory explicitly so
            # identical entry names cannot overwrite another variant's evidence.
            artifact_dir = output / 'cute' / name
            artifact_dir.mkdir(parents=True, exist_ok=False)
            compile_reader = md.cute.compile

            def compile_with_artifacts(*args, **kwargs):
                kwargs['options'] = (kwargs.get('options', '') +
                    f' --keep-ptx --keep-cubin --dump-dir={artifact_dir}')
                return compile_reader(*args, **kwargs)

            md.cute.compile = compile_with_artifacts
            try:
                compiled = build()
            finally:
                md.cute.compile = compile_reader
            cubins = sorted(artifact_dir.rglob('*.cubin'))
            ptx_files = sorted(artifact_dir.rglob('*.ptx'))
            assert cubins and ptx_files, f'missing reader artifacts: {name}'
            return compiled

        md.build_and_load_cute_dsl_kernel = build_reader
        md.configure_static_v2('t,r,sf6')
        md.configure_tp_sf6_q0(True)
        for rows in (2672, 32256):
            for expansion in (False, True):
                md._get_dynamic_kernel(288, rows, 4096, 512, 8, rows,
                    activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                    swiglu_limit=10., tiled=True, reform_sf_pack=not expansion,
                    _prefill_scale_expansion=expansion)
                report['variants'].append(dict(rows=rows, expanded_scales=expansion,
                                               cache_key=repr(list(md._DYNAMIC_KERNEL_CACHE)[-1])))
                print(f'compiled rows={rows} expanded_scales={expansion}', flush=True)
        if args.include_m64:
            md._get_dynamic_kernel(288, 2672, 4096, 512, 8, 2672,
                activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                swiglu_limit=10., tiled=True, reform_sf_pack=True, tile_m=64,
                _prefill_tile64=True)
            report['variants'].append(dict(rows=2672, tile_m=64,
                cache_key=repr(list(md._DYNAMIC_KERNEL_CACHE)[-1])))
            print('compiled private M64 short prefill', flush=True)
        resources = []
        for cubin in sorted((output / 'cute').rglob('*.cubin')):
            result = subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(cubin)],
                                    capture_output=True, text=True, check=True)
            cubin.with_suffix('.resources.log').write_text(result.stdout)
            resources.append(dict(path=str(cubin.relative_to(output)), sha256=digest(cubin),
                                  resources=result.stdout))
        readers = 5 if args.include_m64 else 4
        assert len(md._DYNAMIC_KERNEL_CACHE) == readers, 'all requested readers must compile'
        assert len(resources) >= readers, 'fresh cubin evidence missing'
        report.update(status='PASS', resources=resources)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if report['cuda_initialized']:
            report['status'] = 'FAIL'
        report['elapsed_s'] = time.monotonic() - started
        paths = sorted((root / 'engine/kernels/b12x').rglob('*.py'))
        report['source_sha256'] = {str(p.relative_to(root)): digest(p) for p in paths}
        (output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ('status', 'cuda_initialized', 'elapsed_s')}), flush=True)
    assert not report['cuda_initialized']


if __name__ == '__main__':
    main()
