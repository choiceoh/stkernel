"""Derive paired means from every retained final holdout sample."""
import json
from pathlib import Path
root = Path(__file__).parent / 'rtx5050'
fc = json.loads((root/'split-fc.json').read_text())
head = json.loads((root/'split-head.json').read_text())
assert fc['status'] == head['status'] == 'PASS'
rows = []
for cell in fc['cells']:
    row = dict(shape=cell['shape'], choice=cell['choice'], comparisons={})
    for label in ('direct_cublas', 'deep_gemm'):
        brackets = [h['samples'] for h in cell['holdout'] if h['baseline'] == label]
        assert len(brackets) == 2
        b = sum(t[0]+t[3] for t in brackets)/4
        a = sum(t[1]+t[2] for t in brackets)/4
        row['comparisons'][label] = dict(baseline_ms=b, split_ms=a, latency_change_percent=100*(a/b-1),
                                        all_four_pairs_win_2pct=all(t[1]<.98*t[0] and t[2]<.98*t[3] for t in brackets))
    rows.append(row)
heads = []
for cell in head['cells']:
    choice = cell['choice']
    bracket = next(b for b in cell['brackets'] if b[:3] == [choice['index'], choice['workspace'], choice['producer_warps']])
    b, a = (bracket[3]+bracket[6])/2, (bracket[4]+bracket[5])/2
    heads.append(dict(shape=cell['shape'], baseline_ms=b, cublas_ms=a, latency_change_percent=100*(a/b-1)))
summary = dict(status='PASS', FC=rows, head=heads)
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
