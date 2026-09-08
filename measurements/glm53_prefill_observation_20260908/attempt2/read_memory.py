"""Read-only host memory census; does not collect arguments or environment."""
import datetime
import json
from pathlib import Path
import socket
import subprocess


def fields(path):
    result = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            result[parts[0].rstrip(':')] = int(parts[1])
    return result


result = dict(host=socket.gethostname(), time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              meminfo_kib=fields(Path('/proc/meminfo')), processes=[], process_errors=0, containers=[])
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        state = fields(path/'status')
        if state.get('VmRSS', 0) < 10240:
            continue
        record = dict(pid=int(path.name), name=(path/'comm').read_text().strip(),
                      cgroup=(path/'cgroup').read_text().strip(),
                      status_kib={k:v for k,v in state.items() if k in
                        ('VmRSS','RssAnon','RssFile','RssShmem','VmLck','VmPin','VmSwap')})
        try:
            record['rollup_kib'] = fields(path/'smaps_rollup')
        except (OSError, ValueError) as exc:
            record['rollup_error'] = type(exc).__name__
        result['processes'].append(record)
    except (OSError, ValueError):
        result['process_errors'] += 1
result['processes'].sort(key=lambda x:x['status_kib']['VmRSS'], reverse=True)
ids = subprocess.check_output(['docker','ps','-q'],text=True).split()
if ids:
    for c in json.loads(subprocess.check_output(['docker','inspect',*ids],text=True)):
        row = dict(id=c['Id'], name=c['Name'], pid=c['State']['Pid'],
                   image=c['Image'], started=c['State']['StartedAt'],
                   limits={k:c['HostConfig'].get(k) for k in ('Memory','MemorySwap','ShmSize')})
        try:
            cg = Path('/proc',str(row['pid']),'cgroup').read_text().strip().split('::',1)[1]
            root = Path('/sys/fs/cgroup')/cg.lstrip('/')
            row['cgroup'] = cg
            row['memory_current_bytes'] = int((root/'memory.current').read_text())
            row['memory_stat_bytes'] = fields(root/'memory.stat')
            row['memory_events'] = fields(root/'memory.events')
        except (OSError, ValueError, IndexError) as exc:
            row['cgroup_error'] = type(exc).__name__
        result['containers'].append(row)
print(json.dumps(result,indent=2))
