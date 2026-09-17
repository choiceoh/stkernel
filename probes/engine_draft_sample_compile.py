"""Compile sampled candidate walk for SM121 without GPU access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('offline compile requires hidden GPUs and no device nodes')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.draft_sample import _sample_walk
    cells = []
    for k, c in ((7,16), (8,16), (5,4), (7,8)):
        for greedy, single in ((False,False),(False,True),(True,False),(True,True)):
            constants = dict(sUn=2*k+1, sUk=1, K=k, C=c, BC=triton.next_power_of_2(c), GREEDY_ROWS=greedy, SINGLE_ROW=single, LAST_MASS=not greedy)
            signature = {name: '*fp32' for name in ('SCORES','PROBS','CDF','MASS','TEMPS','UNIFORMS','Q')}
            signature.update({name:'*i64' for name in ('CAND','OUT','SUPPORT')})
            kernel = triton.compile(ASTSource(_sample_walk, signature, constexprs=constants),
                                    target=GPUTarget('cuda',121,32), options=dict(num_warps=1))
            cells.append(dict(k=k, candidates=c, greedy_rows=greedy, single_row=single, last_mass=not greedy, shared_bytes=kernel.metadata.shared,
                              ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest(),
                              cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    if torch.cuda.is_initialized():
        raise RuntimeError('offline compile initialized CUDA')
    paths = ('engine/kernels/draft_sample.py', 'probes/engine_draft_sample_compile.py')
    report = dict(status='PASS', gpu_used=False, target='sm_121', torch=torch.__version__,
                  cuda=torch.version.cuda, triton=triton.__version__, cells=cells,
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
