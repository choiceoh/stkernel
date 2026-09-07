#!/usr/bin/env python3
"""Replay publication of an already deployed revision with identical bytes only.

The PRIME uses the official deployer and its current-main admission. Timed arms
never admit a revision: every file on every node must already match this exact
manifest and source commit before any write. This lets one admitted runtime be
measured while other PRs merge, like the other fixed-runtime startup brackets.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from startup_deploy_receipts import sample

ROOT = Path(__file__).resolve().parents[1]
TARGET = '/home/choiceoh/overlays/glm53/'
SSH = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']


def require_identical(expected, snapshots):
    if set(snapshots) != {'srv1', 'srv2', 'srv3', 'srv4'}:
        raise ValueError('all four deployed nodes are required')
    for node, snapshot in snapshots.items():
        actual = {name: row['sha256'] for name, row in snapshot['files'].items()}
        if actual != expected:
            raise ValueError(f'{node}: this replay cannot deploy different source bytes or a different manifest')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--preserve', choices=('0', '1'), required=True)
    args = ap.parse_args()
    session = os.environ['FLEET_SESSION']
    holder = Path(os.environ.get('FLEET_DIR', '/home/choiceoh/glm53-logs/fleet'))/'holder'

    def held():
        if holder.read_text().split('|')[0] != session:
            raise RuntimeError('only the current fleet holder may replay publication')

    held()
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(ROOT), *argv], text=True).strip()
    if git('rev-parse', 'HEAD') != args.source_commit or git('status', '--porcelain'):
        raise RuntimeError('the admitted source checkout changed')
    build = ROOT/'build/glm53'
    with tempfile.TemporaryDirectory(prefix='startup-identical-publication-') as directory:
        manifest = Path(directory)/'manifest.tsv'
        manifest.write_text(f'# source_commit={args.source_commit}\n' + (build/'manifest.tsv').read_text())
        names = [line.split('\t')[0] for line in manifest.read_text().splitlines()
                 if line and not line.startswith('#')]
        paths = [manifest, *(build/name for name in names)]
        expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        require_identical(expected, dict(sample(n) for n in (1, 2, 3, 4)))
        for node in (2, 1, 3, 4):
            held()
            destination = TARGET if node == 2 else f'choiceoh@10.10.10.{node}:{TARGET}'
            if args.preserve == '1':
                argv = ['bash', '-c', '. "$1"; shift; glm53_sync_overlays "$@"',
                        'same-source-publication', str(ROOT/'launchers/lib/glm53-overlay-sync.sh')]
                if node != 2:
                    argv += ['-e', 'ssh -o BatchMode=yes -o ConnectTimeout=8']
                subprocess.run([*argv, *map(str, paths), destination], check=True)
            elif node == 2:
                for path in paths:
                    subprocess.run(['install', '-m', '0644', str(path), TARGET+path.name], check=True)
            else:
                subprocess.run(['scp', *SSH, '-q', *map(str, paths), destination], check=True)
        require_identical(expected, dict(sample(n) for n in (1, 2, 3, 4)))
    print(f'identical-source publication preserve={args.preserve}; all-rank SHA256 parity verified')


if __name__ == '__main__':
    main()
