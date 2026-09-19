"""Run a clean origin/main onepass with the two already-tested harness repairs.

The engine, launcher, workload and grading files remain the selected main's
bytes. Only deleted metric-helper imports and the serving-container label are
adapted; this wrapper is recorded separately from the measured source SHA.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--helpers', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    args, forwarded = p.parse_known_args()
    sys.path.insert(0, str(args.tree / 'bench'))
    import onepass
    spec = importlib.util.spec_from_file_location('gptq_compatible_metrics', args.helpers / 'bench/onepass_metrics.py')
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.tree, text=True).strip()
    metrics._git_sha = lambda: sha
    original = onepass._load

    def load(filename, name):
        if filename in ('bench-dec.py', 'bracket.py') and not (args.tree / 'bench' / filename).exists():
            return metrics
        return original(filename, name)

    def served_build(_repo):
        names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).split()
        return onepass._st_build(names, name='st-qwen38') if 'st-qwen38' in names else {}

    onepass._load = load
    onepass._served_build = served_build
    if args.preflight:
        for filename in ('korean-corruption.py', 'check-quality.py', 'bench-dec.py', 'bracket.py'):
            load(filename, 'main_preflight_' + filename.replace('-', '_').replace('.', '_'))
        print(json.dumps(dict(main_sha=sha, harness='deleted metrics helpers and Qwen container identity only')))
        return 0
    sys.argv = [str(args.tree / 'bench/onepass.py'), *forwarded]
    return onepass._main()


if __name__ == '__main__':
    raise SystemExit(main())
