#!/usr/bin/env python3
"""Read-only source timestamps and Ninja build receipts on all four nodes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys

READ = '''
from pathlib import Path
import hashlib,json
overlay=Path('/home/choiceoh/overlays/glm53')
cache=Path('/home/choiceoh/glm53-cache')
def receipt(p):
 s=p.stat()
 return dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mtime_ns=s.st_mtime_ns,inode=s.st_ino,size=s.st_size)
names=[line.split('\\t')[0] for line in (overlay/'manifest.tsv').read_text().splitlines() if line and not line.startswith('#')]
files={name:receipt(overlay/name) for name in ['manifest.tsv',*names]}
ninja={str(p.relative_to(cache)):dict(**receipt(p),rows=len(p.read_text().splitlines())-1) for root in ('osar_build','mk_build') for p in (cache/root).glob('*/.ninja_log')}
print(json.dumps(dict(files=files,ninja=ninja)))
'''


def sample(node):
    command = ['python3', '-c', READ]
    if node != 2:
        command = ['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                   f'choiceoh@10.10.10.{node}', shlex.join(command)]
    value = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    return f'srv{node}', json.loads(value.stdout)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = dict(pool.map(sample, (1,2,3,4)))
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
