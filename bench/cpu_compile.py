# SPDX-License-Identifier: Apache-2.0
"""Compile one CUDA translation unit to an object without loading a CUDA context."""
import argparse
from pathlib import Path
import re
import subprocess
from prepared_artifacts import output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--arch', required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / args.source).resolve()
    if not source.is_relative_to(root) or source.suffix != '.cu' or not source.is_file():
        ap.error('source must be a repository CUDA translation unit')
    if not re.fullmatch(r'sm_[0-9]{2,3}a?', args.arch):
        ap.error('arch must be an explicit sm_ target')
    out = output_path(root, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.call(['nvcc', '--compile', '-arch=' + args.arch, str(source), '-o', str(out)])


if __name__ == '__main__':
    raise SystemExit(main())
