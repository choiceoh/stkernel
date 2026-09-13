"""Compile the actual oracle candidate bodies, without a CUDA context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'result.json').exists() and not list(Path('/dev').glob('nvidia*'))
    os.environ.update(CUTE_DSL_ARCH='sm_121a', CUTE_DSL_DISABLE_FILE_CACHING='1',
                      CUTE_DSL_CACHE_DIR=str(output/'cute-cache'), TRITON_CACHE_DIR=str(output/'triton-cache'))
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from engine.kernels.prefill_mhc import _post_prenorm
    started = time.monotonic()
    report = dict(status='RUNNING', gpu_used=False, variants=[])
    try:
        for bm, bk, splits in ((32,128,1), (16,128,1), (16,64,4), (16,32,4)):
            signature = {name: dtype for name,dtype in zip(
                ('Comb','Residual','Post','X','Fn','ResidualOut','GemmOut','Sqrsum','M'),
                ('*fp32','*bf16','*fp32','*bf16','*bf16','*bf16','*fp32','*fp32','i32'))}
            compiled = triton.compile(ASTSource(_post_prenorm, signature,
                constexprs=dict(HIDDEN=4096,NSPLITS=splits,BM=bm,BK=bk,BN=32)),
                target=GPUTarget('cuda',121,32), options=dict(num_warps=4))
            name = f'mhc-bm{bm}-bk{bk}-s{splits}'
            (output/(name+'.cubin')).write_bytes(compiled.asm['cubin'])
            (output/(name+'.ptx')).write_text(compiled.asm['ptx'])
            usage = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(output/(name+'.cubin'))],text=True)
            report['variants'].append(dict(kind=name,shared_bytes=compiled.metadata.shared,resources=usage))
            print(name,usage.strip(),flush=True)
        assert not torch.cuda.is_initialized()
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda *a,**k: (12,1)
        from engine.kernels.b12x import moe_dispatch as md
        md.get_num_sm = lambda *a:48
        md.get_max_active_clusters = lambda *a:48
        def build_reader(module,name,build,**kwargs):
            dest = output/'cute'/name
            dest.mkdir(parents=True,exist_ok=False)
            original = md.cute.compile
            def compile(*a,**kw):
                kw['options'] = kw.get('options','') + f' --keep-ptx --keep-cubin --dump-dir={dest}'
                return original(*a,**kw)
            md.cute.compile = compile
            try:
                return build()
            finally:
                md.cute.compile = original
        md.build_and_load_cute_dsl_kernel = build_reader
        md.configure_static_v2('t,r,sf6')
        md.configure_tp_sf6_q0(True)
        for rows in (2672,32256):
            md._get_dynamic_kernel(288,rows,4096,512,8,rows,
                activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.,
                tiled=True,reform_sf_pack=False,_prefill_scale_expansion=True,_prefill_n128=True)
            report['variants'].append(dict(kind='moe_n128',rows=rows,key=repr(list(md._DYNAMIC_KERNEL_CACHE)[-1])))
            print('compiled N128',rows,flush=True)
        for cubin in sorted((output/'cute').rglob('*.cubin')):
            usage = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cubin)],text=True)
            report.setdefault('cute_resources',[]).append(dict(path=str(cubin.relative_to(output)),resources=usage))
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        report.update(cuda_initialized=torch.cuda.is_initialized(),elapsed_s=time.monotonic()-started)
        files = [*root.glob('engine/kernels/b12x/**/*.py'),root/'engine/kernels/prefill_mhc.py',
                 root/'engine/kernels/dense/mhc.py',root/'engine/profiles/glm53/net.py']
        report['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('status','cuda_initialized','elapsed_s')}),flush=True)
    assert not report['cuda_initialized']


if __name__=='__main__':
    main()
