"""Compile the actual dense-prefix MLA body without a CUDA device/context."""
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
    output.mkdir(parents=True, exist_ok=False)
    assert not list(Path('/dev').glob('nvidia*'))
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from engine.kernels.mla.prefill_dense import _dense_prefix
    from engine.profiles.glm53.facts import architecture
    from engine.profiles.glm53.caches import layout
    config_path = Path(__file__).with_name('model-config.json')
    facts = architecture(json.loads(config_path.read_text()))
    cache = layout(facts, range(facts.layers))
    offsets = {L: offset//facts.kv_lora for L, offset in cache.token_offsets.items()}
    last_layer = max(offsets)
    started = time.monotonic()
    report = dict(status='RUNNING', gpu_used=False, variants=[],
                  model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  geometry=dict(block=facts.block, stride=cache.block_bytes//facts.kv_lora,
                                token_offsets=offsets, scale=facts.mla_scale))
    try:
        geometries = [(last_layer, *g) for g in ((32,32,8), (32,32,4), (64,32,8), (32,64,8), (32,32,16))]
        geometries += [(L, 32, 32, 8) for L in offsets if L != last_layer]
        for layer, bm, bn, warps in geometries:
            constants = dict(SCALE=facts.mla_scale, KV_SCALE=1., BLOCK=facts.block,
                             STRIDE=cache.block_bytes//facts.kv_lora, OFFSET=offsets[layer], IDENTITY=False,
                             HEADS=16, DIM=512, BM=bm, BN=bn)
            signature = dict(Q='*bf16', KV='*fp8e4nv', Blocks='*i32', Out='*bf16', ROWS='i32', CONTEXT='i32')
            compiled = triton.compile(ASTSource(_dense_prefix, signature, constexprs=constants),
                                      target=GPUTarget('cuda', 121, 32),
                                      options=dict(num_warps=warps, num_stages=1))
            name = f'L{layer}-m{bm}-n{bn}-w{warps}'
            cubin = output / (name + '.cubin')
            cubin.write_bytes(compiled.asm['cubin'])
            (output / (name + '.ptx')).write_text(compiled.asm['ptx'])
            usage = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(cubin)], text=True)
            report['variants'].append(dict(name=name, shared_bytes=compiled.metadata.shared, resources=usage))
            print(name, usage.strip(), flush=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        report.update(cuda_initialized=torch.cuda.is_initialized(), elapsed_s=time.monotonic()-started)
        paths = ['engine/kernels/mla/prefill_dense.py', 'engine/modules/prefill_attention.py',
                 'engine/modules/sparse_attention.py',
                 'engine/profiles/glm53/net.py', 'engine/profiles/glm53/lanes.py',
                 'engine/profiles/glm53/execution.py', 'engine/profiles/glm53/boot.py']
        report['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('status', 'cuda_initialized', 'elapsed_s')}), flush=True)
    assert not report['cuda_initialized']


if __name__ == '__main__':
    main()
