"""Compile and load the single-GPU transport oracle (production one-shot source plus the PDL neighbour) with CUDA
hidden, and record what it binds; the GPU lane then only has to run it.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=/repo python3 measurements/st_c2_oneshot_consumer_20260915/oracle_compile.py --output /out/oracle.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import torch
    assert not torch.cuda.is_initialized()
    from tests.test_engine_direct_producer_cuda import build_oracle
    ext = build_oracle()
    bound = sorted(name for name in dir(ext) if not name.startswith('_'))
    for name in ('oneshot_ar', 'oneshot_ar_consumer', 'staged_copy', 'moe_packets', 'oneshot_max_int64', 'land_peers'):
        assert name in bound, name
    assert not torch.cuda.is_initialized()
    sources = ('probes/oneshot_producer_oracle.cu', 'engine/kernels/oneshot/dsv4_oneshot_ar.cu',
               'engine/kernels/oneshot/dsv4_oneshot_transport.h')
    report = dict(status='PASS', evidence='oracle Torch extension compile/load only', gpu_used=False,
                  torch=torch.__version__, cuda=torch.version.cuda, extension=ext.__name__,
                  extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(), bindings=bound,
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
