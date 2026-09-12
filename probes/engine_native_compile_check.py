"""CPU-only cold NVCC A/B for the one-shot extension's Torch includes.

Use a fresh cache and Python process per sample in the ST image without GPUs.
The only source difference is torch/extension.h versus Tensor/pybind headers.
All device SASS must match exactly; no transport or GPU kernel is executed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from probes.engine_native_cache_check import ONESHOT_API, ONESHOT_FILES, ROOT

FULL = '#include <torch/extension.h>\n'
LEAN = '#include <torch/types.h>\n#include <torch/csrc/utils/pybind.h>\n'


def run(output, repeats):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'engine/kernels/oneshot'
    code = (source / ONESHOT_FILES[1]).read_text()
    if (code.count(FULL), code.count(LEAN)) not in ((1, 0), (0, 1)):
        raise ValueError('expected exactly one supported Torch include block')
    baseline = code.replace(LEAN, FULL)
    candidate = code.replace(FULL, LEAN)
    variants = dict(full=baseline, lean=candidate)
    for arm, content in variants.items():
        destination = output / arm
        destination.mkdir()
        for name in ONESHOT_FILES:
            shutil.copy2(source / name, destination / name)
        (destination / ONESHOT_FILES[1]).write_text(content)
    report = dict(scope=__doc__, gpu_used=False, complete=False, rows=[],
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__).resolve(), ROOT / 'probes/engine_native_cache_check.py',
                                           ROOT / 'engine/kernels/native_cache.py',
                                           *(source / name for name in ONESHOT_FILES))},
                  variant_sha256={arm: hashlib.sha256(code.encode()).hexdigest() for arm, code in variants.items()})
    sass_hash = None
    for sample in range(repeats):
        for arm in (('full', 'lean') if sample % 2 == 0 else ('lean', 'full')):
            command = [sys.executable, str(ROOT / 'probes/engine_native_cache_check.py'),
                       '--worker', '--fixture', 'oneshot', '--arm', 'stable',
                       '--input', str(output / arm), '--cache', str(output / f'cache-{sample}-{arm}')]
            child = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', MAX_JOBS='2'),
                                   text=True, capture_output=True, timeout=300)
            stem = output / f'{sample}-{arm}'
            stem.with_suffix('.log').write_text(child.stdout + child.stderr)
            if child.returncode:
                raise RuntimeError(f'{sample}/{arm} failed; see its saved compiler log')
            row = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith('RESULT ')))
            if row['value'] != ONESHOT_API or row['nvcc_compilations'] != 1 or row['cuda_initialized']:
                raise RuntimeError(f'cold compilation or exported API differs: {row}')
            binary = Path(row['directory']) / f"st_oneshot_{row['key']}.so"
            sass = subprocess.check_output(['cuobjdump', '--dump-sass', str(binary)])
            stem.with_suffix('.sass').write_bytes(sass)
            if b'Function :' not in sass:
                raise RuntimeError('cuobjdump returned no device functions')
            digest = hashlib.sha256(sass).hexdigest()
            if sass_hash is None:
                sass_hash = digest
            row.update(headers=arm, sample=sample, sass_sha256=digest)
            report['rows'].append(row)
            (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            if digest != sass_hash:
                raise RuntimeError('device SASS differs; inspect the saved dumps before adopting the header change')
    medians = {arm: statistics.median(row['load_seconds'] for row in report['rows'] if row['headers'] == arm)
               for arm in variants}
    report.update(complete=True, median_load_seconds=medians,
                  reduction_percent=100 * (1 - medians['lean'] / medians['full']))
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('complete', 'median_load_seconds', 'reduction_percent')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args.output.resolve(), args.repeats)
