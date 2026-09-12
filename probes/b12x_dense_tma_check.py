"""Compile the E=1/M=1 NVFP4 input TMA before/after row padding, without CUDA.

Run inside the ST image without --gpus. Requires nvdisasm on PATH. This
checks the real workspace allocator and generated SM121a instructions;
numerical correctness and graph replay still require a separate GPU run.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    os.environ['CUTE_DSL_KEEP'] = 'ptx,cubin'
    artifacts = Path(a.out).with_suffix('.artifacts')
    artifacts.mkdir(parents=True, exist_ok=True)
    os.environ['CUTE_DSL_DUMP_DIR'] = str(artifacts)
    import torch
    assert not torch.cuda.is_initialized()
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *args, **kwargs: (12, 1)
    torch.cuda.current_device = lambda: 0
    from engine.kernels.b12x import moe_dispatch as md
    md.get_num_sm = lambda *args: 48
    md.get_max_active_clusters = lambda *args: 48
    md.build_and_load_cute_dsl_kernel = lambda module, name, build, **kwargs: build()
    workspace = md.allocate_sm120_static_workspace(
        state_E=1, weight_E=1, max_rows=1, k=4096, n=3072, num_topk=1,
        device=torch.device('cpu'))
    assert workspace.max_rows >= 128
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for rows in (1, workspace.max_rows):
            kernel, mac = md._get_micro_kernel(
                1, 1, 1, 4096, 3072, 1, rows, single_token=True, mac_override=24,
                activation='swigluoai_uninterleave', swiglu_alpha=1.,
                swiglu_beta=0., swiglu_limit=10.)
            cubin = Path(directory) / f'rows-{rows}.cubin'
            cubin.write_bytes(kernel.__cubin__)
            sass = subprocess.check_output(['nvdisasm', str(cubin)], text=True)
            counts = {f'{d}D': len(re.findall(rf'UTMALDG\.{d}D\b', sass)) for d in (1, 2, 3)}
            results.append(dict(rows=rows, mac=mac, input_tma=counts))
    assert results[0]['input_tma']['1D'] > 0, results
    assert results[1]['input_tma']['1D'] == 0, results
    assert results[1]['input_tma']['2D'] > results[0]['input_tma']['2D'], results
    assert not torch.cuda.is_initialized()
    receipt = dict(passed=True, cuda_initialized=False, target='sm_121a', checks=results)
    Path(a.out).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
