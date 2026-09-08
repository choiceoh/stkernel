#!/usr/bin/env python3
"""Read preserved startup logs; no GPU or serving requests."""
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

root = Path(sys.argv[1])
parser = Path(__file__).with_name('startup-cache-report.py')
subprocess.run([sys.executable, str(parser), str(root)], check=True, stdout=subprocess.DEVNULL)
data = json.loads((root / 'report.json').read_text())
runtime_file = root / 'runtime-source-commit.txt'
data['runtime_source_commit'] = runtime_file.read_text().strip().removeprefix('# source_commit=') if runtime_file.exists() else data['source_commit']
ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
for arm, detail in data['arms'].items():
    boot = root / f'{arm}-boot.out'
    detail['compile_cache'] = [json.loads(row) for row in re.findall(r'\[compile-cache\] (\{[^\n]+\})', boot.read_text())] if boot.exists() else []
    for node, row in detail.get('nodes', {}).items():
        text = ansi.sub('', (root / f'{arm}-{node}.log').read_text(errors='replace'))
        row['pack_io'] = []
        for model, fields in re.findall(r'\[mk-pack-io\] (\S+) ([^\n]*)', text):
            values = {k: float(v) if k.endswith('_s') else int(v) for k, v in re.findall(r'(\w+)=([\d.]+)', fields)}
            row['pack_io'].append({'model': model, **values})
        row['fold_phases'] = [dict(model=model, **{k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', fields)})
                              for model, fields in re.findall(r'\[fp8-dense\] (\S+) host-seconds=([^\n]*)', text)]
base = next((a for a in data['arms'] if a.endswith('BASE1')), None)
if base:
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
    values = {key: [] for key in ('health_s', 'model_s', 'mk_attach_s', 'key_s', 'read_s', 'copy_s', 'profile_s')}
    for r in rows:
        head = r['nodes']['srv2']
        values['health_s'].append(r.get('health_wall_s'))
        values['model_s'].extend(head['phase_s'].get('load-model', []))
        values['profile_s'].extend(head['phase_s'].get('profile/determine-memory', []))
        values['mk_attach_s'].append(sum(f['mk'] for f in head['fold_phases']))
        if head['pack_io']:
            for key in ('key_s', 'read_s', 'copy_s'):
                values[key].append(head['pack_io'][-1][key])
    summary[kind] = {k: {'n': len(v), 'mean': statistics.mean(v), 'min': min(v), 'max': max(v), 'samples': v}
                     for k, raw in values.items() if (v := [x for x in raw if x is not None])}
data['head_comparison'] = summary
(root / 'report.json').write_text(json.dumps(data, indent=2) + '\n')
lines = ['# GLM W4 SHA256 cache key fleet validation', '', f"Runtime source: `{data['runtime_source_commit']}`; benchmark source: `{data['source_commit']}`", '',
         'PRIME creates startup artifacts and warms compilation. Timed order: BASE1 → FAST1 → FAST2 → BASE2. Timed boots use the same runtime/profile, warm rank and FP8 artifacts, PREFILL_WARMUP=0 and Korean onepass at 2K/32K. FAST enables SHA256 W4 keys; all arms retain FAST_IO=1. PRIME aliases historical MD5 packs before timing.', '',
         '| Arm | Health (s) | Head model (s) | W4 attach (s) | W4 key / read / copy (s) | Profile (s) | Quality / corrupt | Raw acceptance |',
         '|---|---:|---:|---:|---|---:|---|---:|']
for arm, r in data['arms'].items():
    h = r.get('nodes', {}).get('srv2', {})
    phase = h.get('phase_s', {})
    p = h.get('pack_io', [{}])[-1] if h.get('pack_io') else {}
    op = r.get('onepass', {})
    q, k = op.get('quality', {}), op.get('korean', {})
    quality = f"{q.get('ok', '?')}/{q.get('total', '?')}; {k.get('dirty', '?')}/{k.get('n', '?')}"
    mk = sum(f['mk'] for f in h.get('fold_phases', []))
    acc = op.get('decode', {}).get('acc_raw')
    accept = f'{100 * acc:.2f}%' if acc is not None else 'pending'
    io = ' / '.join(str(p.get(key, '?')) for key in ('key_s', 'read_s', 'copy_s'))
    lines.append(f"| {arm} | {r.get('health_wall_s', '?')} | {phase.get('load-model', ['?'])[0]} | {mk:.3f} | {io} | {phase.get('profile/determine-memory', ['?'])[0]} | {quality} | {accept} |")
if summary.get('BASE') and summary.get('FAST'):
    lines.extend(['', 'Two-sample head comparisons (PRIME excluded):', '', '| Metric | BASE mean [range] | FAST mean [range] | Difference |', '|---|---:|---:|---:|'])
    for key in ('health_s', 'model_s', 'mk_attach_s', 'key_s', 'read_s', 'copy_s', 'profile_s'):
        if key not in summary['BASE'] or key not in summary['FAST']:
            continue
        b, f = summary['BASE'][key], summary['FAST'][key]
        lines.append(f"| {key} | {b['mean']:.3f} [{b['min']:.3f}, {b['max']:.3f}] | {f['mean']:.3f} [{f['min']:.3f}, {f['max']:.3f}] | {f['mean'] - b['mean']:+.3f} |")
lines.extend(['', 'All-node warm receipts:', '', '| Arm | Node | Rank cache | FP8 hit/miss/error | W4 fast/legacy hits | SHA hits / MD5 fallbacks / aliases / errors | Copy disarms |', '|---|---|---|---|---|---|---|'])
for arm, r in data['arms'].items():
    for node, row in r.get('nodes', {}).items():
        p = row['pack_io'][-1] if row.get('pack_io') else {}
        rank = ', '.join(f"{x['kind']} {x['seconds']:.3f}s" for x in row['rank'])
        fp8 = '/'.join(str(sum(x[k] for x in row['fp8'])) for k in ('hit', 'miss', 'errors'))
        lines.append(f"| {arm} | {node} | {rank} | {fp8} | {p.get('fast_hits', '?')}/{p.get('legacy_hits', '?')} | {p.get('sha_hits', '?')}/{p.get('md5_fallback', '?')}/{p.get('aliases', '?')}/{p.get('alias_errors', '?')} | {row['copy_disarmed']} |")
lines.extend(['', 'Prompt/output comparison with BASE1 (artifact bytes and generated responses are separate checks):', ''])
for arm, r in data['arms'].items():
    if 'response_comparison' in r:
        lines.append(f"- {arm}: `{r['response_comparison']}`")
lines.extend(['', 'Host samples every 10 seconds; sampled OS minima, not CUDA peak memory:', '', '| Node | Samples | Min available RAM (GiB) | Min free disk (GiB) | Net swap growth (MiB) |', '|---|---:|---:|---:|---:|'])
for node, r in data['host_resource_samples'].items():
    lines.append(f"| {node} | {r['n']} | {r['min_available_gib']:.2f} | {r['min_disk_available_gib']:.2f} | {r['swap_used_increase_mib']:.1f} |")
lines.extend(['', f"Trial exit: `{data['exit_code']}` (null means ongoing).", '', 'See report.json, pack-key-gpu.json and the adjacent raw logs, environment snapshots and exact response files. Host stage times include synchronization. Two warm samples per arm do not establish full-context quality or broad throughput performance.', ''])
(root / 'report.md').write_text('\n'.join(lines))
print('\n'.join(lines[:16]))
