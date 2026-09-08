#!/usr/bin/env python3
"""Device-free CuTe compile in an isolated, memory-capped serving image."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'


def mounts():
    entries = (ROOT/'build/glm53/manifest.tsv').read_text().splitlines()
    wanted = {p.name for p in (ROOT/'overlay/modules/glm53_moe').glob('*.py')}
    result = []
    for line in entries:
        parts = line.split('\t')
        if parts[0] in wanted:
            result += ['--mount', f'type=bind,src={ROOT}/build/glm53/{parts[0]},dst={parts[1]},readonly']
            wanted.remove(parts[0])
    assert not wanted, wanted
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rows', nargs='+', type=int, default=[6,16])
    args = ap.parse_args()
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    args.out.mkdir(parents=True, exist_ok=False)
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:'))) * 1024
    assert available >= 12*1024**3, ('need 12 GiB before CPU compile',available)
    for rows in args.rows:
        variant = "m" + str(rows)
        name = 'moe-reform-cpu-' + variant + '-' + str(os.getpid())
        cmd = ['docker','run','--rm','--runtime=runc','--network=none','--name',name,
               '--cpuset-cpus=14-15','--memory=2g','--memory-swap=2g',
               '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
               '-e','CUTE_DSL_ARCH=sm_121a','-e','MAX_JOBS=1','-e','OMP_NUM_THREADS=1',
               '-e','MK_PKG_PATH=/usr/local/lib/python3.12/dist-packages',
               '--mount',f'type=bind,src={ROOT},dst=/repo,readonly',*mounts(),
               '--workdir','/repo','--entrypoint','python3',IMAGE,
               '/repo/probes/b12x_static_compile_check.py','--specs','t|t,r',
               '--m',str(rows),'--max-rows','640']
        try:
            with (args.out/(variant+'.log')).open('w') as log:
                result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=360)
            print(variant,result.returncode,flush=True)
            assert result.returncode == 0,(variant,result.returncode)
        finally:
            subprocess.run(['docker','stop','-t','1',name],stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,timeout=10)


if __name__ == '__main__':
    main()
