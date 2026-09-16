"""Read candidate boot evidence on all four hosts from srv1; no GPU work."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess

SESSION = 'tool-workflow-c2-0916'
SHA = 'b7d577cf131fe0c59ff90b9a23a0f9d0f28a3534'
OUT = Path('/home/choiceoh/expert-capture') / SESSION / 'boot-evidence'
OUT.mkdir(parents=True, exist_ok=True)


def capture(host):
    script = '''
from pathlib import Path
import json
p=Path('/home/choiceoh/glm53-logs/st-bracket-dumps/SESSION-hold-SHORT')
files={f.name:json.loads(f.read_text()) for f in p.glob('*rank*.json')
       if f.name.startswith(('boot-', 'memory-'))}
print(json.dumps(files))
'''.replace('SESSION', SESSION).replace('SHORT', SHA[:12])
    command = ['python3', '-'] if host == 1 else [
        'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        f'10.10.10.{host}', 'python3 -']
    result = subprocess.run(command, input=script, text=True, capture_output=True,
                            check=True, timeout=30)
    files = json.loads(result.stdout)
    (OUT / f'srv{host}.json').write_text(json.dumps(files, indent=2) + '\n')
    memory = next(v for k, v in files.items() if k.startswith('memory-'))
    boot = next(v for k, v in files.items() if k.startswith('boot-'))
    phase = next(p for p in memory['phases'] if p['phase'] == 'decode experts')
    return f'srv{host}', {
        'ready': memory['ready'], 'when': boot['when'],
        'layout': boot['root']['counters']['weight_layout'],
        'decode_experts': phase,
        'failed_phases': [p['phase'] for p in memory['phases'] if p.get('passed') is False],
    }


with ThreadPoolExecutor(max_workers=4) as pool:
    ranks = dict(pool.map(capture, range(1, 5)))
report = {'sha': SHA, 'at': datetime.now().astimezone().isoformat(), 'ranks': ranks}
(OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
assert all(r['ready'] and not r['failed_phases'] for r in ranks.values())
print(json.dumps({k: {'ready': v['ready'], 'expert_seconds': v['decode_experts']['seconds'],
                     'layout': v['layout']} for k, v in ranks.items()}))
