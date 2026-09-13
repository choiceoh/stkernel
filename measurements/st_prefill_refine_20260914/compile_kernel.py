"""Compile bounded MLA addresses in the ST CPU image, without a CUDA context."""
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
    parser.add_argument('--output', required=True, type=Path)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    assert not list(Path('/dev').glob('nvidia*'))
    os.environ['TRITON_CACHE_DIR'] = str(output/'cache')
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from engine.kernels.mla.prefill_absorb import _absorb
    from engine.kernels.mla.prefill_dense import _dense_prefix
    from engine.profiles.glm53.facts import architecture
    from engine.profiles.glm53.caches import layout
    config = root/'measurements/st_prefill_dense_prefix_20260913/model-config.json'
    facts = architecture(json.loads(config.read_text()))
    cache = layout(facts, range(facts.layers))
    assert (facts.heads//4, facts.qk_nope, facts.v_dim, facts.kv_lora) == (16, 256, 256, 512)
    variants = []
    for transpose in (False, True):
        inner, outer = (512, 256) if transpose else (256, 512)
        constants = dict(HEADS=16, INPUT=inner, OUTPUT=outer, WH=512*512, WR=512,
                         TRANSPOSE=transpose, BM=64, BN=64, BK=32)
        variants.append(('absorb-output' if transpose else 'absorb-query', _absorb,
                         dict(X='*bf16', W='*bf16', Y='*bf16', ROWS='i32'), constants, 4, 2))
    for layer, offset in cache.token_offsets.items():
        constants = dict(SCALE=facts.mla_scale, KV_SCALE=1., BLOCK=facts.block,
                         STRIDE=cache.block_bytes//facts.kv_lora, OFFSET=offset//facts.kv_lora,
                         IDENTITY=False, HEADS=16, DIM=512, BM=32, BN=32)
        variants.append((f'dense-prefix-L{layer}', _dense_prefix,
                         dict(Q='*bf16', KV='*fp8e4nv', Blocks='*i32', Out='*bf16', ROWS='i32', CONTEXT='i32'),
                         constants, 8, 1))
    started = time.monotonic()
    report = dict(status='RUNNING', gpu_used=False, variants=[],
                  model_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    try:
        for name, kernel, signature, constants, warps, stages in variants:
            compiled = triton.compile(ASTSource(kernel, signature, constexprs=constants),
                                      target=GPUTarget('cuda', 121, 32),
                                      options=dict(num_warps=warps, num_stages=stages))
            cubin = output/(name+'.cubin')
            cubin.write_bytes(compiled.asm['cubin'])
            (output/(name+'.ptx')).write_text(compiled.asm['ptx'])
            usage = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(cubin)], text=True)
            report['variants'].append(dict(name=name, constants=constants,
                                           shared_bytes=compiled.metadata.shared, resources=usage))
            print(name, usage.strip(), flush=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        paths = ['engine/kernels/mla/prefill_absorb.py', 'engine/kernels/mla/prefill_dense.py',
                 'engine/modules/prefill_indexer.py', 'engine/profiles/glm53/net.py', 'engine/profiles/glm53/boot.py']
        report.update(cuda_initialized=torch.cuda.is_initialized(), elapsed_s=time.monotonic()-started,
                      source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ('status', 'cuda_initialized', 'elapsed_s')}), flush=True)
    assert not report['cuda_initialized']


if __name__ == '__main__':
    main()
