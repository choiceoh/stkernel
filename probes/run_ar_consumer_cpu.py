#!/usr/bin/env python3
"""Compile the AR/MHC consumer pair in a device-free, capped container."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))) * 1024
    assert available >= 12 * 1024**3, ('need 12 GiB before CPU compile', available)
    args.out.mkdir(parents=True, exist_ok=False)
    name = 'ar-consumer-cpu-' + str(os.getpid())
    cmd = ['docker', 'run', '--rm', '--runtime=runc', '--network=none', '--name', name,
        '--cpuset-cpus=14-15', '--memory=6g', '--memory-swap=6g',
        '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
        '-e', 'MAX_JOBS=1', '-e', 'OMP_NUM_THREADS=1',
        '-e', 'AR_CONSUMER_BUILD=/evidence/build',
        '-e', 'VLLM_GLM53_MK_BUILD_ROOT=/evidence/build/mk',
        '-e', 'VLLM_DSV4_OSAR_BUILD_ROOT=/evidence/build/osar',
        '--mount', f'type=bind,src={ROOT},dst=/repo,readonly',
        '--mount', f'type=bind,src={args.out},dst=/evidence',
        '--workdir', '/repo', '--entrypoint', 'python3', IMAGE,
        '/repo/probes/ar_consumer_probe.py', '--compile-only', '--out', '/evidence/compile.json']
    try:
        with (args.out / 'compile.log').open('w') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
        assert rc == 0, ('compile failed', rc, str(args.out / 'compile.log'))
        assert json.loads((args.out / 'compile.json').read_text())['status'] == 'PASS'
    finally:
        subprocess.run(['docker', 'stop', '-t', '1', name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    print('PASS device-free production MK/OSAR compilation', flush=True)


if __name__ == '__main__':
    main()
