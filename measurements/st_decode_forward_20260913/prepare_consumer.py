"""Run on srv2 before measurement; copy private RTN caches without calibration."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

SOURCE = '/home/choiceoh/glm53-cache-st-decode22-consumer-v6-A'
DEST = '/home/choiceoh/glm53-cache-st-decode-forward-consumer-A'
ENV = '/home/choiceoh/glm53-logs/st-decode-forward-consumer.env'
RANKS = Path('/home/choiceoh/models/st-glm53-9391-up-gate-full')
META_SOURCE = Path('/home/choiceoh/st-engine/st-glm53-meta')
META = Path('/home/choiceoh/glm53-logs/st-decode-forward-consumer-b12x-meta')


def prepare_metadata():
    """Bind the actual B12x ranks to matching metadata before any GPU boot."""
    from engine.profiles.glm53 import facts
    F = facts.load(META_SOURCE)
    for rank in range(4):
        with (RANKS / f'rank{rank}of4.safetensors').open('rb') as stream:
            header = json.loads(stream.read(int.from_bytes(stream.read(8), 'little')))
        assert header['__metadata__']['weight_layout'] == F.weight_layout
    META.mkdir(exist_ok=False)
    names = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
             'generation_config.json', 'processor_config.json')
    hashes = {}
    for name in names:
        shutil.copy2(META_SOURCE / name, META / name)
        hashes[name] = hashlib.sha256((META / name).read_bytes()).hexdigest()
    receipt = dict(source=str(META_SOURCE), destination=str(META),
                   weight_layout=F.weight_layout, files=hashes)
    (META / 'source-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt

NODE = '''
import json, shutil, subprocess, time
from pathlib import Path
source, dest = Path(SOURCE), Path(DEST)
partial = dest.with_name(dest.name + '.partial')
assert source.is_dir() and not dest.exists() and not partial.exists()
files = [p for p in source.rglob('*') if p.is_file() and 'mkcalib' not in p.relative_to(source).parts]
size = sum(p.stat().st_size for p in files)
assert shutil.disk_usage(dest.parent).free > size + 5*1024**3
excluded = [{'path': str(p.relative_to(source)), 'bytes': p.stat().st_size}
            for p in (source/'mkcalib').rglob('*') if p.is_file()]
partial.mkdir()
started = time.time()
subprocess.run(['rsync', '-a', '--exclude=/mkcalib', str(source)+'/', str(partial)+'/'], check=True)
assert not (partial/'mkcalib').exists()
for p in files:
    got = partial/p.relative_to(source)
    assert got.stat().st_size == p.stat().st_size, str(p)
    assert (got.stat().st_dev, got.stat().st_ino) != (p.stat().st_dev, p.stat().st_ino)
partial.rename(dest)
print(json.dumps({'source':str(source), 'destination':str(dest), 'files':len(files), 'bytes':size,
                  'excluded_calibration':excluded, 'seconds':time.time()-started,
                  'policy':'independent copy; exclude mkcalib; preserve original; content-addressed RTN packs'}))
'''


def prepare(host):
    script = f'SOURCE={SOURCE!r}\nDEST={DEST!r}\n' + NODE
    cmd = ['python3', '-c', script] if host is None else ['ssh', host, 'python3 -c ' + shlex.quote(script)]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return {'host':host or 'head', 'receipt':json.loads(result.stdout)}


if __name__ == '__main__':
    metadata = prepare_metadata()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(prepare, (None, 'choiceoh@10.10.10.1', 'choiceoh@10.10.10.3', 'choiceoh@10.10.10.4')))
    values = dict(RANKS_DIR=str(RANKS), CKPT=str(META),
                  DRAFTER='/home/choiceoh/models/GLM-5.3-Flash-DFlash2', ST_KV_GIB='6', CACHE_DIR=DEST)
    with Path(ENV).open('x') as stream:
        stream.write(''.join(f'{k}={shlex.quote(v)}\n' for k,v in values.items()))
    print(json.dumps({'nodes':receipts, 'metadata':metadata, 'production_env':ENV, 'values':values}, indent=2))
