"""Recompute consumer observations, retaining failed gates and output differences.

Usage: python3 summarize.py consumer.jsonl > consumer-summary.json
The canonical adoption verdict remains bench/st_judge.py; this separates
normal quality requests from bounded, fixed-token requests without regrading.
"""
import json
from pathlib import Path
import runpy
import sys

common = runpy.run_path(str(Path(__file__).resolve().parents[1]
                           / 'st_fixed_k_cost_20260917/summarize.py'))
pooled = common['pooled']


def quality(requests):
    rows = [q for r in requests for q in r.get('quality', [])]
    return dict(ok=sum(q['passed'] for q in rows), total=len(rows),
                failures=[dict(ctx=r['ctx'], question=r['question'], client=r.get('client'),
                    case=q['case'], dimensions=[name for name, ok in q['checks'].items() if not ok])
                    for r in requests for q in r.get('quality', []) if not q['passed']])


def groups(record):
    requests = record.get('requests', [])
    fixed = record.get('concurrency_fixed') or {}
    return dict(normal_c1=[r for r in requests if not r.get('fixed_decode')],
                fixed_decode_c1=[r for r in requests if r.get('fixed_decode')],
                fixed_concurrency_c1=fixed.get('c1_requests', []),
                fixed_concurrency_many=fixed.get('many_requests', []),
                normal_many=[r for g in record.get('c4', []) for r in g['requests']])


def key(r):
    return r['ctx'], str(r['question']), r.get('rep', 0), r.get('client')


def compare_group(base, candidate):
    b, a = ({key(r): r for r in rows} for rows in (base, candidate))
    if len(b) != len(base) or len(a) != len(candidate) or b.keys() != a.keys():
        raise ValueError('different or duplicate request cases')
    result = []
    for k, br in b.items():
        ar = a[k]
        if br['workload_sha256'] != ar['workload_sha256']:
            raise ValueError(f'unsalted request changed: {k}')
        result.append(dict(case=k, output_equal=br['output_sha256'] == ar['output_sha256'],
            tokens_equal=br['completion_tokens'] == ar['completion_tokens'],
            base_tokens=br['completion_tokens'], candidate_tokens=ar['completion_tokens'],
            base_tok_s=br['decode_tok_s'], candidate_tok_s=ar['decode_tok_s'],
            base_ttft_s=br['ttft_s'], candidate_ttft_s=ar['ttft_s']))
    equal = [key(r) for r in base if r['output_sha256'] == a[key(r)]['output_sha256']
             and r['completion_tokens'] == a[key(r)]['completion_tokens']]
    bs, cs = pooled(base), pooled(candidate)
    return dict(requests=result, base_pooled_tok_s=bs, candidate_pooled_tok_s=cs,
                observed_change_pct=100 * (cs / bs - 1) if bs and cs else None,
                equal_output_count=len(equal),
                equal_output_base_tok_s=pooled([b[k] for k in equal]),
                equal_output_candidate_tok_s=pooled([a[k] for k in equal]),
                rate_scope='sum(tokens - 1) / sum(request decode seconds); not aggregate concurrency throughput')


def main(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    indexed = {(r['name'], r['run_index']): r for r in records}
    if len(indexed) != len(records):
        raise ValueError('duplicate arm/run')
    runs, pairs = [], []
    for r in records:
        row = common['summarize'](r)
        intervals = (r.get('decode') or {}).get('fixed_intervals', [])
        row['fixed_pooled_step_s'] = (sum(w['steps'] for w in intervals)
            / sum(w['seconds'] for w in intervals)) if intervals else None
        row['fixed_window_count'] = len(intervals)
        row['request_groups'] = {name: dict(count=len(req), quality=quality(req),
            pooled_decode_tok_s=pooled(req)) for name, req in groups(r).items()}
        runs.append(row)
    for run in (1, 2):
        if ('B', run) not in indexed or ('A', run) not in indexed:
            continue
        b, a = indexed['B', run], indexed['A', run]
        row = common['compare'](b, a)
        bg, ag = groups(b), groups(a)
        row['request_groups'] = {name: compare_group(bg[name], ag[name]) for name in bg}
        pairs.append(row)
    return dict(runs=runs, pairs=pairs, complete=len(pairs) == 2,
                scope='Observations only; failed quality/preparation gates are retained, not waived')


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1]), ensure_ascii=False, indent=2))
