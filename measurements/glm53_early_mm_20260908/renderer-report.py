#!/usr/bin/env python3
"""Summarize the matched renderer boot bracket from retained logs only."""
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

root = Path(sys.argv[1])
subprocess.run([sys.executable, str(Path(__file__).with_name('startup-cache-report.py')), str(root)],
               check=True, stdout=subprocess.DEVNULL)
data = json.loads((root / 'report.json').read_text())
data['runtime_source_commit'] = (root / 'runtime-source-commit.txt').read_text().strip().removeprefix('# source_commit=')
ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
for arm, detail in data['arms'].items():
    boot = root / f'{arm}-boot.out'
    detail['compile_cache'] = [json.loads(row) for row in re.findall(r'\[compile-cache\] (\{[^\n]+\})', boot.read_text())] if boot.exists() else []
    env = root / f'{arm}-cache-env.json'
    if env.exists():
        detail['environment'] = dict(v.split('=', 1) for v in json.loads(env.read_text()))
    for node, row in detail.get('nodes', {}).items():
        text = ansi.sub('', (root / f'{arm}-{node}.log').read_text(errors='replace'))
        if node == 'srv2':
            first_health = text.find('GET /health HTTP/1.1" 200')
            posts = list(re.finditer(r'(\d+\.\d+\.\d+\.\d+):\d+ - "POST /', text))
            detail['serving_requests'] = {
                'loopback': sum(m[1].startswith('127.') for m in posts),
                'non_loopback': sum(not m[1].startswith('127.') for m in posts),
                'completed_before_first_health': sum(m.start() < first_health for m in posts),
            }
        row['pack_io'] = []
        for model, fields in re.findall(r'\[mk-pack-io\] (\S+) ([^\n]*)', text):
            row['pack_io'].append({'model': model, **{k: float(v) if k.endswith('_s') else int(v)
                for k, v in re.findall(r'(\w+)=([\d.]+)', fields)}})
        row['renderer'] = {
            'stock_warmup_s': [float(v) for v in re.findall(r'(?:Multi-modal|Readonly multi-modal) warmup completed in ([\d.]+)s', text)],
            'early_elapsed_s': [float(v) for v in re.findall(r'\[early-mm-warmup\] completed processors=2/2 elapsed_s=([\d.]+)', text)],
            'reuse_join_s': [float(v) for v in re.findall(r'\[early-mm-warmup\] reused .*? join_s=([\d.]+)', text)],
            'completed_before_model': '[early-mm-warmup] completed' in text and
                text.index('[early-mm-warmup] completed') < text.index('[boot-stamp] load-model took'),
        }
base = next((a for a in data['arms'] if a.endswith('BASE1')), None)
if base and (root / f'{base}-responses.jsonl').exists():
    reference = {r['prompt_sha256']: r for r in map(json.loads, (root / f'{base}-responses.jsonl').read_text().splitlines())}
    for arm, row in data['arms'].items():
        p = root / f'{arm}-responses.jsonl'
        if p.exists():
            responses = list(map(json.loads, p.read_text().splitlines()))
            row['response_comparison'] = {'baseline': base, 'n': len(responses),
                'matching_prompts': sum(r['prompt_sha256'] in reference for r in responses),
                'exact_responses': sum(r['prompt_sha256'] in reference and r['response_sha256'] == reference[r['prompt_sha256']]['response_sha256'] for r in responses)}
summary = {}
for kind in ('BASE', 'FAST'):
    rows = [r for a, r in data['arms'].items() if re.search(kind + r'[12]$', a) and 'srv2' in r.get('nodes', {})]
    values = {k: [] for k in ('health_s', 'model_s', 'profile_s', 'mm_warmup_s')}
    for r in rows:
        head = r['nodes']['srv2']
        values['health_s'].append(r.get('health_wall_s'))
        for key, label in (('model_s', 'load-model'), ('profile_s', 'profile/determine-memory')):
            values[key].extend(head['phase_s'].get(label, []))
        values['mm_warmup_s'].append(sum(head['renderer']['stock_warmup_s']))
    summary[kind] = {k: {'n': len(v), 'mean': statistics.mean(v), 'min': min(v), 'max': max(v), 'samples': v}
                    for k, raw in values.items() if (v := [x for x in raw if x is not None])}
data['head_comparison'] = summary
(root / 'report.json').write_text(json.dumps(data, indent=2)+'\n')
lines = ['# GLM CPU renderer warmup overlap', '',
         f"Runtime source: `{data['runtime_source_commit']}`; benchmark source: `{data['source_commit']}`.", '',
         'PRIME refreshes artifacts and compilation. Timed order is BASE1, FAST1, FAST2, BASE2 with the same code/image/profile. Only EARLY_MM_WARMUP changes (BASE=0, FAST=1). W4 SHA256/fast IO and warm rank/FP8 caches remain enabled. All boots use PREFILL_WARMUP=0 and the canonical Korean onepass workload at 2K/32K.', '',
         '| Arm | Health s | Head model s | Profile s | Both MM warmups s | Early complete / reuse | Quality / corruption |',
         '|---|---:|---:|---:|---:|---|---|']
for arm, row in data['arms'].items():
    head = row.get('nodes', {}).get('srv2', {})
    phase, mm, op = head.get('phase_s', {}), head.get('renderer', {}), row.get('onepass', {})
    q, k = op.get('quality', {}), op.get('korean', {})
    quality = f"{q.get('ok', '?')}/{q.get('total', '?')}; {k.get('dirty', '?')}/{k.get('n', '?')}"
    lines.append(f"| {arm} | {row.get('health_wall_s')} | {phase.get('load-model', ['?'])[0]} | {phase.get('profile/determine-memory', ['?'])[0]} | {sum(mm.get('stock_warmup_s', [])):.3f} | {mm.get('early_elapsed_s')} / {mm.get('reuse_join_s')} | {quality} |")
lines += ['', 'Two warm samples per arm; PRIME excluded:', '',
          '| Metric | BASE mean [range] | FAST mean [range] | FAST minus BASE |', '|---|---:|---:|---:|']
for key in ('health_s', 'model_s', 'profile_s', 'mm_warmup_s'):
    if key not in summary['BASE'] or key not in summary['FAST']:
        continue
    b, f = summary['BASE'][key], summary['FAST'][key]
    lines.append(f"| {key} | {b['mean']:.3f} [{b['min']:.3f}, {b['max']:.3f}] | {f['mean']:.3f} [{f['min']:.3f}, {f['max']:.3f}] | {f['mean']-b['mean']:+.3f} |")
lines += ['', 'All-rank cache receipts:', '',
          '| Arm | Node | Rank artifact | FP8 hit/miss/error | W4 SHA/fast/legacy |', '|---|---|---|---|---|']
for arm, row in data['arms'].items():
    for node, r in row.get('nodes', {}).items():
        io = r['pack_io'][-1] if r['pack_io'] else {}
        rank = ', '.join(f"{x['kind']} {x['seconds']:.3f}s" for x in r['rank'])
        fp8 = '/'.join(str(sum(x[k] for x in r['fp8'])) for k in ('hit', 'miss', 'errors'))
        lines.append(f"| {arm} | {node} | {rank} | {fp8} | {io.get('sha_hits')}/{io.get('fast_hits')}/{io.get('legacy_hits')} |")
lines += ['', 'Generated response matches against BASE1 (separate from exact CPU preprocessing checks):', '']
for arm, row in data['arms'].items():
    if 'response_comparison' in row:
        lines.append(f"- {arm}: `{row['response_comparison']}`")
lines += ['', 'Serving POST completions in the preserved logs:', '']
for arm, row in data['arms'].items():
    lines.append(f"- {arm}: `{row.get('serving_requests')}`")
lines += ['', 'Non-loopback serving traffic overlapped the quality workload. Onepass throughput, TTFT and acceptance counters are therefore not a matched performance comparison. The startup endpoint is the first HTTP health 200 on a new container; logs place that health response before the recorded POST completions.', '']
lines += ['', 'Host memory and disk sampled every 10 seconds; these are OS samples, not CUDA peak allocations:', '',
          '| Node | Samples | Minimum available RAM GiB | Minimum disk GiB | Net swap growth MiB |', '|---|---:|---:|---:|---:|']
for node, r in data['host_resource_samples'].items():
    lines.append(f"| {node} | {r['n']} | {r['min_available_gib']:.2f} | {r['min_disk_available_gib']:.2f} | {r['swap_used_increase_mib']:.1f} |")
lines += ['', f"Trial exit: `{data['exit_code']}` (null means ongoing).", '',
          'Raw logs, actual environment snapshots, exact response files and source hashes are retained beside this report. Two warm samples per arm do not establish broad throughput or full-context quality. Warmup still runs both stock processors and clears their caches; the candidate moves those CPU operations before engine readiness.', '']
(root / 'report.md').write_text('\n'.join(lines))
print('\n'.join(lines[:17]))
