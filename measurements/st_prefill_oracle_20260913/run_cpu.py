"""Bounded CPU-only compiler run in the already-qualified ST runtime image."""
import argparse
import json
from pathlib import Path
import subprocess
import os

IMAGE = 'sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('compiler output must be separate from immutable source')
    assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    assert available >= 12*1024*1024, 'CPU compile requires 12 GiB available without reclaiming serving memory'
    output.mkdir(parents=True, exist_ok=False)
    name = 'st-prefill-oracle-cpu-' + str(os.getpid())
    command = ['docker', 'run', '--rm', '--runtime=runc', '--network=none', '--name', name,
               '--cpus=2', '--memory=4g', '--memory-swap=4g', '--pids-limit=256',
               '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
               '-e', 'MAX_JOBS=1', '-e', 'OMP_NUM_THREADS=1', '-e', 'PYTHONPATH=/repo',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-v', str(root)+':/repo:ro',
               '-v', str(output)+':/evidence', '-w', '/repo', '--entrypoint=python3', IMAGE]
    compile_args = ['measurements/st_prefill_oracle_20260913/compile_candidates.py', '--output', '/evidence']
    try:
        tests = ['tests.test_prefill_oracle_candidates', 'tests.test_moe_prefill_m64',
                 'tests.test_moe_prefill_scale_expansion', 'tests.test_engine_mhc_contract',
                 'tests.test_engine_direct_mhc']
        with (output/'cpu-tests.log').open('w') as log:
            subprocess.run(command + ['-m', 'unittest', *tests], stdout=log,
                           stderr=subprocess.STDOUT, timeout=240, check=True)
        with (output/'compile.log').open('w') as log:
            completed = subprocess.run(command + compile_args, stdout=log, stderr=subprocess.STDOUT, timeout=840)
        report = json.loads((output/'result.json').read_text())
        assert completed.returncode==0 and report['status']=='PASS' and not report['cuda_initialized'], report.get('error')
        print(json.dumps(dict(status='PASS', image=IMAGE, elapsed_s=report['elapsed_s'],
                              variants=len(report['variants']), output=str(output))), flush=True)
    finally:
        subprocess.run(['docker', 'stop', '-t', '1', name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15)


if __name__=='__main__':
    main()
