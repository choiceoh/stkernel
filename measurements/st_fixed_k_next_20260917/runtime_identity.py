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
  if not any(name in path for name in ('/st_dense_', '/st_mla_', '/st_router_fused_')): continue
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
    identity = json.loads(subprocess.check_output(cmd, text=True))
    log_cmd = ['docker', 'logs', 'st-glm53']
    if rank != 0:
        log_cmd = ['ssh', '-n', '-o', 'BatchMode=yes', 'choiceoh@' + host, shlex.join(log_cmd)]
    lines = subprocess.check_output(log_cmd, text=True, stderr=subprocess.STDOUT).splitlines()
    proof = [json.loads(line.split('ST_NATIVE_EXECUTION ', 1)[1])
             for line in lines if line.startswith('ST_NATIVE_EXECUTION ')]
    if len(proof) != 1 or proof[0]['rank'] != rank:
        raise RuntimeError(f'rank {rank}: missing or ambiguous execution proof')
    return dict(rank=rank, node=host, **identity, native_execution=proof[0],
                moe_served=[line for line in lines if line.startswith('[b12x static v2] lane serving:')])


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(read, enumerate(('10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4'))))
    for row in rows:
        paths = [m['files'][0]['path'] for m in row['modules']]
        if any(sum(name in path for path in paths) != 1
               for name in ('/st_dense_', '/st_mla_', '/st_router_fused_')):
            raise RuntimeError(f"rank {row['rank']}: require exactly one mapped dense, MLA and fused-router native")
    print(json.dumps(dict(scope='read-only process map and file identity, no GPU calls', ranks=rows), indent=2))
