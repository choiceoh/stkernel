"""Compile the actual MLA contraction body without a CUDA device/context."""
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
    from engine.kernels.mla.prefill_absorb import _absorb
    from engine.profiles.glm53.facts import architecture
    config_path = root / 'measurements/st_prefill_dense_prefix_20260913/model-config.json'
    facts = architecture(json.loads(config_path.read_text()))
    heads = facts.heads // 4
    assert (heads, facts.qk_nope, facts.v_dim, facts.kv_lora) == (16, 256, 256, 512)
    started = time.monotonic()
    report = dict(status='RUNNING', gpu_used=False, variants=[],
                  model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  geometry=dict(heads=heads, qk_nope=facts.qk_nope, v_dim=facts.v_dim, kv_lora=facts.kv_lora))
    try:
        geometries = [(64,64,32,4,2), (64,64,64,4,2), (32,64,64,4,2), (64,128,32,4,2)]
        for transpose in (False, True):
            for bm, bn, bk, warps, stages in geometries:
              inner, outer = (512, 256) if transpose else (256, 512)
              constants = dict(HEADS=heads, INPUT=inner, OUTPUT=outer, WH=512*512,
                               WR=512, TRANSPOSE=transpose, BM=bm, BN=bn, BK=bk)
              signature = dict(X='*bf16', W='*bf16', Y='*bf16', ROWS='i32')
              compiled = triton.compile(ASTSource(_absorb, signature, constexprs=constants),
                                        target=GPUTarget('cuda', 121, 32),
                                        options=dict(num_warps=warps, num_stages=stages))
              name = f'{"output" if transpose else "query"}-m{bm}-n{bn}-k{bk}-w{warps}-s{stages}'
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
        paths = ['engine/kernels/mla/prefill_absorb.py', 'engine/modules/mla_absorb.py',
                 'engine/profiles/glm53/net.py', 'engine/profiles/glm53/lanes.py',
                 'engine/profiles/glm53/execution.py', 'engine/profiles/glm53/boot.py']
        report['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('status', 'cuda_initialized', 'elapsed_s')}), flush=True)
    assert not report['cuda_initialized']


if __name__ == '__main__':
    main()
