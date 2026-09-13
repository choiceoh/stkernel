"""CPU-only cold NVCC A/B for the one-shot extension's Torch includes.

Use a fresh cache and Python process per sample in the ST image without GPUs.
Compare full versus Tensor/pybind headers, or Tensor/pybind versus narrower
per-operator factory headers. The latter retains torch::empty_like and uses
the same underlying ATen Tensor and dtype aliases explicitly.
Extracted device cubins must match after normalizing only NVCC's source-file
IDs in internal symbol strings. Code, constants and launch metadata remain
byte-exact. No transport or GPU kernel is executed. cuobjdump extraction does
not require nvdisasm or a GPU driver.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import sys

from probes.engine_native_cache_check import ONESHOT_API, ONESHOT_FILES, ROOT

FULL = '#include <torch/extension.h>\n'
LEAN = '#include <torch/types.h>\n#include <torch/csrc/utils/pybind.h>\n'
OPERATORS = ('#define AT_PER_OPERATOR_HEADERS\n#include <ATen/core/Tensor.h>\n'
             '#include <torch/csrc/autograd/generated/variable_factories.h>\n'
             '#include <torch/csrc/utils/pybind.h>\n')


def source_variants(code, comparison):
    counts = tuple(code.count(block) for block in (FULL, LEAN, OPERATORS))
    if counts not in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        raise ValueError('expected exactly one supported Torch include block')
    if counts[2]:
        code = code.replace(OPERATORS, LEAN).replace('at::Tensor', 'torch::Tensor')
        code = code.replace('at::kBFloat16', 'torch::kBFloat16').replace('at::kLong', 'torch::kInt64')
    lean = code.replace(FULL, LEAN)
    if comparison == 'full-lean':
        return dict(full=lean.replace(LEAN, FULL), lean=lean)
    if comparison != 'lean-operators':
        raise ValueError(f'unknown comparison: {comparison}')
    operators = lean.replace(LEAN, OPERATORS).replace('torch::Tensor', 'at::Tensor')
    operators = operators.replace('torch::kBFloat16', 'at::kBFloat16').replace('torch::kInt64', 'at::kLong')
    return dict(lean=lean, operators=operators)


def canonical_cubin(path):
    """Normalize path-derived internal names, preserving every other ELF byte.

    NVCC changes these IDs for identical code compiled at different paths.
    Only two observed name prefixes, inside .strtab, are eligible. In
    particular no instruction, symbol offset, constant or ELF metadata is
    masked. A new compiler format fails the equality gate for inspection.
    """
    data = bytearray(path.read_bytes())
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected a little-endian ELF64 cubin')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, names_index = struct.unpack_from('<HHH', data, 58)
    rows = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size) for i in range(count)]
    names = rows[names_index]
    strings = bytes(data[names[4]:names[4] + names[5]])
    replacements, text_sections = 0, 0
    for row in rows:
        name = strings[row[0]:].split(b'\x00', 1)[0]
        text_sections += name.startswith(b'.text.')
        if name == b'.strtab':
            start, end = row[4], row[4] + row[5]
            normalized, replacements = re.subn(
                rb'(_INTERNAL_|_GLOBAL__N__)[0-9a-f]{8}(_18_dsv4_oneshot_ar_cu_)',
                rb'\g<1>00000000\g<2>', bytes(data[start:end]))
            assert len(normalized) == end - start
            data[start:end] = normalized
    if not text_sections:
        raise ValueError('cubin has no device text sections')
    return hashlib.sha256(data).hexdigest(), replacements, text_sections


def run(output, repeats, comparison):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'engine/kernels/oneshot'
    code = (source / ONESHOT_FILES[1]).read_text()
    variants = source_variants(code, comparison)
    arms = tuple(variants)
    for arm, content in variants.items():
        destination = output / arm
        destination.mkdir()
        for name in ONESHOT_FILES:
            shutil.copy2(source / name, destination / name)
        (destination / ONESHOT_FILES[1]).write_text(content)
    report = dict(scope=__doc__, comparison=comparison, gpu_used=False, complete=False, rows=[],
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__).resolve(), ROOT / 'probes/engine_native_cache_check.py',
                                           ROOT / 'engine/kernels/common/native_cache.py',
                                           *(source / name for name in ONESHOT_FILES))},
                  variant_sha256={arm: hashlib.sha256(code.encode()).hexdigest() for arm, code in variants.items()})
    cubin_hashes = None
    for sample in range(repeats):
        for arm in (arms if sample % 2 == 0 else tuple(reversed(arms))):
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
            extracted = output / f'{sample}-{arm}-cubins'
            extracted.mkdir()
            subprocess.run(['cuobjdump', '--extract-elf', 'all', str(binary)],
                           cwd=extracted, check=True, capture_output=True)
            cubins = sorted(extracted.glob('*.cubin'))
            if not cubins:
                raise RuntimeError('cuobjdump extracted no device cubins')
            digests = sorted(canonical_cubin(p) for p in cubins)
            if cubin_hashes is None:
                cubin_hashes = digests
            row.update(headers=arm, sample=sample,
                       cubin_sha256=sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in cubins),
                       canonical_cubins=[dict(sha256=sha, normalized_file_ids=ids, text_sections=sections)
                                         for sha, ids, sections in digests])
            report['rows'].append(row)
            (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            if digests != cubin_hashes:
                raise RuntimeError('device cubins differ; inspect the saved binaries before adopting the header change')
    medians = {arm: statistics.median(row['load_seconds'] for row in report['rows'] if row['headers'] == arm)
               for arm in variants}
    report.update(complete=True, median_load_seconds=medians,
                  median_compiler_cpu_seconds={arm: statistics.median(row['compiler_cpu_seconds']
                                               for row in report['rows'] if row['headers'] == arm)
                                               for arm in arms},
                  reduction_percent=100 * (1 - medians[arms[1]] / medians[arms[0]]))
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('complete', 'median_load_seconds', 'reduction_percent')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--comparison', choices=('full-lean', 'lean-operators'), default='full-lean')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args.output.resolve(), args.repeats, args.comparison)
