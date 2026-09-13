"""Run in the ST image with CUDA hidden, repository cwd and --out DIR."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    import torch
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'tests.test_engine_oneshot_health', 'tests.test_engine_oneshot_integer',
        'tests.test_engine_oneshot_sum', 'tests.test_engine_comm_layout'])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and not result.skipped
    from engine.kernels.oneshot import build, MAX_ELEMENTS
    ext = build()
    assert not torch.cuda.is_initialized()
    files = ['engine/kernels/oneshot/' + p for p in [
        'SOURCE.json', '__init__.py', 'dsv4_oneshot_ar.cu', 'dsv4_oneshot_transport.h']]
    report = dict(status='PASS', evidence='full Torch extension compile/load only',
                  gpu_used=False, torch=torch.__version__, cuda=torch.version.cuda,
                  max_elements=MAX_ELEMENTS, tests=result.testsRun,
                  extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / 'compile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
