"""Compile and dlopen every native ST extension without GPU device access.

The dense builder's final device-properties query is replaced by its declared
GB10 shape. Compilation and library loading are real; device execution,
numerics, graph replay and throughput are outside this probe's scope.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUILDERS = {'mla': '_build', 'dense': 'extension', 'oneshot': 'build',
            'mapped_staging': 'build', 'decode_queue': 'build',
            'bounded_graph': 'build', 'prefill_topk': '_build'}


def worker(name, output):
    import torch
    from torch.utils import cpp_extension
    from engine.kernels.common.native_cache import cuda_toolchain_identity
    if torch.cuda.is_initialized():
        raise RuntimeError('unexpected CUDA initialization')
    loaded = []
    real_load = cpp_extension.load

    def load(**kwargs):
        module = real_load(**kwargs)
        loaded.append(module)
        if name != 'dense':
            return module
        class DeclaredDevice:
            def probe_device(self):
                return (12, 1, 48)
        return DeclaredDevice()

    start = time.monotonic()
    with patch.object(cpp_extension, 'load', load):
        module = importlib.import_module('engine.kernels.' + name)
        getattr(module, BUILDERS[name])()
    if torch.cuda.is_initialized() or len(loaded) != 1:
        raise RuntimeError('native compile must load one extension without initializing CUDA')
    path = Path(loaded[0].__file__)
    resources = subprocess.check_output([str(Path(cpp_extension.CUDA_HOME) / 'bin/cuobjdump'),
                                         '--dump-resource-usage', str(path)], text=True)
    output.with_suffix('.resources.txt').write_text(resources)
    result = dict(status='PASS', gpu_used=False, extension=name, torch=torch.__version__,
                  torch_git=torch.version.git_version, cuda=torch.version.cuda,
                  compiler=cuda_toolchain_identity(cpp_extension.CUDA_HOME),
                  binary=str(path), binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  seconds=time.monotonic() - start)
    output.write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=BUILDERS)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('run with CUDA_VISIBLE_DEVICES= and no /dev/nvidia* devices')
    if args.worker:
        worker(args.worker, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name in BUILDERS:
        output = args.output / (name + '.json')
        with output.with_suffix('.log').open('w') as log:
            result = subprocess.run([sys.executable, __file__, '--worker', name, '--output', str(output)],
                                    stdout=log, stderr=subprocess.STDOUT)
        print(f'{name}: exit {result.returncode}', flush=True)
        if result.returncode:
            print(output.with_suffix('.log').read_text()[-6000:])
            raise SystemExit(result.returncode)
    (args.output / 'results.json').write_text(json.dumps(dict(status='PASS', gpu_used=False,
        extensions=[json.loads((args.output / (name + '.json')).read_text()) for name in BUILDERS]), indent=2) + '\n')


if __name__ == '__main__':
    main()
