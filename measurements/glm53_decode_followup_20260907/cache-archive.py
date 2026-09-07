import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path('/home/choiceoh/glm53-cache/glm53-fp8')
PLAN = Path('/tmp/mhcbf16-fp8-cache-plan.json')
LIST = Path('/tmp/mhcbf16-fp8-cache-files.list')
CUTOFF = time.mktime(time.strptime('2026-09-07 08:00:00', '%Y-%m-%d %H:%M:%S'))

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def safe_name(name):
    if not re.fullmatch(r'[0-9a-f]{64}\.pt', name):
        raise ValueError('unsafe cache filename')

if sys.argv[1] == 'plan':
    if PLAN.exists() or LIST.exists():
        raise RuntimeError('archive plan already exists; review before changing it')
    files = []
    for p in ROOT.iterdir():
        if p.is_symlink() or not p.is_file() or not re.fullmatch(r'[0-9a-f]{64}\.pt', p.name):
            continue
        st = p.stat()
        if st.st_mtime < CUTOFF and st.st_nlink == 1:
            files.append((st.st_mtime_ns, p.name, st.st_size))
    entries, total = [], 0
    for mtime_ns, name, size in sorted(files):
        p = ROOT / name
        entries.append(dict(name=name, size=size, mtime_ns=mtime_ns, sha256=sha(p)))
        total += size
        if total >= 8 * 2**30:
            break
    if total < 8 * 2**30:
        raise RuntimeError('insufficient eligible old cache files')
    data = dict(source=str(ROOT), cutoff='2026-09-07 08:00:00 KST', total_bytes=total, entries=entries)
    PLAN.write_text(json.dumps(data, indent=2) + '\n')
    LIST.write_bytes(b''.join(e['name'].encode() + b'\0' for e in entries))
    print(PLAN.read_text(), end='')
elif sys.argv[1] == 'verify-backup':
    data = json.loads(Path(sys.argv[2]).read_text())
    backup = Path(sys.argv[3])
    for e in data['entries']:
        safe_name(e['name'])
        p = backup / e['name']
        assert not p.is_symlink() and p.stat().st_size == e['size'] and sha(p) == e['sha256'], e['name']
    print(json.dumps(dict(verified_files=len(data['entries']), verified_bytes=data['total_bytes'])))
elif sys.argv[1] == 'remove-archived':
    data = json.loads(PLAN.read_text())
    # Verify every original before removing any. Caller first verifies the backup.
    for e in data['entries']:
        safe_name(e['name'])
        p = ROOT / e['name']
        st = p.lstat()
        assert not p.is_symlink() and st.st_nlink == 1 and st.st_size == e['size'] and st.st_mtime_ns == e['mtime_ns'], e['name']
        assert sha(p) == e['sha256'], e['name']
    for e in data['entries']:
        p = ROOT / e['name']
        st = p.lstat()
        assert st.st_size == e['size'] and st.st_mtime_ns == e['mtime_ns'], e['name']
        p.unlink()
    print(json.dumps(dict(removed_files=len(data['entries']), archived_bytes=data['total_bytes'])))
else:
    raise ValueError('unknown action')
