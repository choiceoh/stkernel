import json
import math
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (root / 'onepass.jsonl').read_text().splitlines() if line.strip()]
names = ['SPFR30907B1', 'SPFR30907A', 'SPFR30907B2']
records = {name: next(r for r in reversed(rows) if r['name'] == name) for name in names}
attests = {name: json.loads((root / (name + '.attest.json')).read_text()) for name in names}
issues = []
reference = records[names[0]]
for name, record in records.items():
    if attests[name]['issues']:
        issues.append(name + ': attestation failed')
    if record['quality'] != {'ok': 15, 'total': 15} or record['korean']['dirty']:
        issues.append(name + ': quality failed')
    if record.get('evidence_issues') or record['traffic'].get('issues'):
        issues.append(name + ': traffic evidence failed')
    for key in ['git', 'overlay', 'harness', 'doc_lang', 'thinking', 'workload', 'endpoint']:
        if record.get(key) != reference.get(key):
            issues.append(name + ': mismatched ' + key)
    request_keys = lambda r: [(v['ctx'], v['question'], v['request_sha256'], v['prompt_tokens']) for v in r['requests']]
    if request_keys(record) != request_keys(reference):
        issues.append(name + ': request bodies or token counts differ')
    for ip, node in attests[name]['nodes'].items():
        base = attests[names[0]]['nodes'][ip]
        for key in ['image', 'args', 'manifest_sha', 'mounts']:
            if node[key] != base[key]:
                issues.append(name + ': node ' + ip + ' differs in ' + key)

table = []
for ctx in reference['workload']['ctx']:
    prefill = {name: next(v for v in record['prefill'] if v['ctx'] == ctx) for name, record in records.items()}
    baselines = [prefill[name]['cold_s'] for name in [names[0], names[2]]]
    candidate = prefill[names[1]]['cold_s']
    assert all(math.isfinite(x) and x > 0 for x in baselines + [candidate])
    row = {'ctx': ctx, 'first_ttft_s': {name: prefill[name]['cold_s'] for name in names},
           'baseline_mean_s': mean(baselines),
           'latency_reduction_pct': 100 * (1 - candidate / mean(baselines)),
           'throughput_gain_pct': 100 * (mean(baselines) / candidate - 1),
           'throughput_gain_vs_each_baseline_pct': [100 * (b / candidate - 1) for b in baselines],
           'baseline_spread_pct': 100 * (max(baselines) / min(baselines) - 1),
           'all_ttft_samples_s': {name: prefill[name]['ttft_samples_s'] for name in names}}
    if not prefill[names[0]]['combined']:
        warm_base = [prefill[name]['warm_s'] for name in [names[0], names[2]]]
        warm_cand = prefill[names[1]]['warm_s']
        row['warm_ttft_s'] = {name: prefill[name]['warm_s'] for name in names}
        row['warm_latency_reduction_pct'] = 100 * (1 - warm_cand / mean(warm_base))
        row['warm_throughput_gain_pct'] = 100 * (mean(warm_base) / warm_cand - 1)
    table.append(row)

out = {'issues': issues, 'arms': names, 'source': reference['git'], 'overlay': reference['overlay'],
       'requests_matched': not any('request bodies' in issue for issue in issues),
       'cold_compile': {name: bool(r.get('cold_compile')) for name, r in records.items()},
       'quality': {name: r['quality'] for name, r in records.items()},
       'korean': {name: r['korean'] for name, r in records.items()},
       'decode': {name: r['decode'] for name, r in records.items()},
       'prefill_comparison': table,
       'limitations': ['Two independent baseline boots and one candidate boot; no confidence interval.',
                      'First-content TTFT includes request processing and first decode, not isolated kernel time.',
                      'Warm columns are minima of later short-context requests with prefix caching enabled.',
                      'Reduced KV capacity differs from production-capacity acceptance.']}
(root / 'comparison.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'issues': issues, 'prefill_comparison': table}, ensure_ascii=False, indent=2))
raise SystemExit(3 if issues else 0)
