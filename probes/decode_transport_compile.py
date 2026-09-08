#!/usr/bin/env python3
"""Compile one production OSAR mode without a CUDA device or RDMA connection."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_hashes(paths):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact', choices=('0', '1'), required=True)
    parser.add_argument('--inline', choices=('0', '1'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        parser.error('run in the device-free CPU container')
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error('fresh output required; refusing to overwrite compile evidence')
    os.environ.update(CUDA_VISIBLE_DEVICES='', MAX_JOBS='1',
        VLLM_DSV4_OSAR_MAXEL='131072',
        VLLM_GLM53_AR_CONSUMER_PDL='1', VLLM_GLM53_MK_PDL='1',
        VLLM_GLM53_AR_COMPACT_CTA=args.compact,
        VLLM_GLM53_AR_PROXY_INLINE=args.inline)
    directory = ROOT / 'overlay/modules/tp_oneshot_ar'
    source = directory / 'dsv4_oneshot_shim.py'
    paths = (source, directory/'dsv4_oneshot_ar.cu', directory/'dsv4_oneshot_transport.h')
    before = source_hashes(paths)
    import torch
    assert not torch.cuda.is_initialized(), 'CPU compiler entered with initialized CUDA'
    spec = importlib.util.spec_from_file_location('decode_transport_cpu', source)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    extension = shim._build()
    modes = list(extension.transport_modes())
    assert modes == [int(args.compact), int(args.inline), 0, 0, 0], modes
    assert hasattr(extension, 'oneshot_ar') and hasattr(extension, 'oneshot_ar_consumer')
    assert not torch.cuda.is_initialized(), 'CPU compilation initialized CUDA'
    after = source_hashes(paths)
    assert before == after, 'source changed during compilation'
    artifact = Path(extension.__file__).resolve()
    assert artifact.is_file(), ('compiled extension missing', str(artifact))
    report = dict(status='PASS', evidence='compile-only', modes=modes,
        torch=torch.__version__, cuda=torch.version.cuda, cuda_initialized=False,
        maxel=shim._MAXEL, source_sha256=after,
        extension_path=str(artifact), extension_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
