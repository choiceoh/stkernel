"""Read the native modules actually mapped by the running TP4 workers; no GPU calls.

Run on srv2 during each reserved consumer arm. JSON includes process maps,
module/source hashes and mtimes. It does not assert that two boots match.
"""
import concurrent.futures
import json
import shlex
import subprocess

CODE = r'''
import hashlib,json,os
from pathlib import Path
modules={}
for maps in Path('/proc').glob('[0-9]*/maps'):
 try: lines=maps.read_text().splitlines()
 except (FileNotFoundError,PermissionError,ProcessLookupError): continue
 for line in lines:
  path=line.split()[-1]
  if '/st_dense_' not in path and '/st_mla_' not in path: continue
  if path.endswith('.so'): modules.setdefault(path,set()).add(int(maps.parent.name))
result=[]
for filename,pids in sorted(modules.items()):
 p=Path(filename)
 files=[]
 for f in [p,*sorted((p.parent/'src').glob('*.cu'))]:
  files.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                    bytes=f.stat().st_size,mtime=f.stat().st_mtime))
 result.append(dict(pids=sorted(pids),files=files))
print(json.dumps(dict(release=os.environ.get('ST_RELEASE'),modules=result)))
'''


def read(item):
    rank, host = item
    local = ['docker', 'exec', 'st-glm53', 'python3', '-c', CODE]
    cmd = local if rank == 0 else ['ssh', '-n', '-o', 'BatchMode=yes', 'choiceoh@' + host, shlex.join(local)]
    return dict(rank=rank, node=host, **json.loads(subprocess.check_output(cmd, text=True)))


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(read, enumerate(('10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4'))))
    if any(not row['modules'] for row in rows):
        raise RuntimeError('a rank has no mapped dense/MLA module; wait for boot to load its kernels')
    print(json.dumps(dict(scope='read-only process map and file identity, no GPU calls', ranks=rows), indent=2))
