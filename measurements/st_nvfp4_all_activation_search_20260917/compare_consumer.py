#!/usr/bin/env python3
"""Compare completed raw receipts without relaxing the canonical quality gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench'))
import st_judge


def load(path):
    raw = path.read_bytes()
    record = json.loads(raw)
    if record.get('recording', {}).get('status') != 'complete':
        raise ValueError(f'{path}: not complete')
    record['_receipt_sha256'] = hashlib.sha256(raw).hexdigest()
    return record


def quality(records, key):
    items = [r[key] for r in records if r.get(key)]
    dims = sorted({name for q in items for name in q.get('dimensions', {})})
    return {
        **{k: sum(q.get(k, 0) for q in items) for k in ('ok', 'total', 'score', 'max_score')},
        'dimensions': {d: {k: sum(q.get('dimensions', {}).get(d, {}).get(k, 0) for q in items)
                           for k in ('ok', 'total')} for d in dims},
    }


def arm(records):
    fixed = [r['concurrency_fixed'] for r in records if r.get('concurrency_fixed')]
    if len(fixed) != 1:
        raise ValueError('Expected one fixed-length sample per arm')
    return dict(
        runs=[dict(run_id=r['run_id'], run_index=r['run_index'], arm_sha=r['arm_sha'],
                   boot_id=r['boot_id'], source_record_sha256=r['_receipt_sha256'],
                   canonical_errors=st_judge.errors(r),
                   steady_state_valid=r['steady_state']['valid'],
                   natural_acceptance=r['decode']['acc_raw'],
                   tokens_per_step=r['decode']['tokens_per_step'],
                   c1_output_tokens=r['decode']['gen_tokens'],
                   c1=[{k:q.get(k) for k in ('ctx', 'question', 'completion_tokens', 'elapsed_s',
                                            'ttft_s', 'decode_tok_s', 'output_sha256')}
                       for q in r['requests']],
                   prefill=r['prefill']) for r in records],
        quality_c1=quality(records, 'quality'), quality_c2=quality(records, 'quality_c4'),
        fixed={k:fixed[0][k] for k in ('valid', 'issues', 'c1_tok_s', 'many_tok_s', 'multiplier',
                                     'c1_decode_tok_s', 'many_decode_tok_s_sum', 'decode_multiplier')},
    )


def main(args):
    base, cand = [[load(p) for p in paths] for paths in (args.base, args.candidate)]
    if len(base) != 2 or len(cand) != 2:
        raise ValueError('Expected two C=1 runs on each arm')
    if any(not st_judge.comparable(a, b) for a in base for b in cand):
        raise ValueError('Harness, quality protocol, workload or budgets differ')
    for records in (base, cand):
        if sorted(r['run_index'] for r in records) != [1, 2] or len({r['boot_id'] for r in records}) != 1:
            raise ValueError('Expected run 1 and 2 on one boot')
    b, a = arm(base), arm(cand)
    ratio = a['fixed']['c1_tok_s'] / b['fixed']['c1_tok_s']
    result = dict(
        scope='One boot per arm; C=1 natural EOS twice, C=2 and fixed 1024 tokens once. '
              'Repeated cases are not independent tasks. Preserve all canonical quality failures.',
        comparable=True, baseline=b, candidate=a,
        fixed_relative_change_pct={k:100*(a['fixed'][k]/b['fixed'][k]-1)
                                   for k in ('c1_tok_s', 'many_tok_s', 'multiplier')},
        c1_speed_guard_pass=all(x['fixed']['valid'] for x in (a, b)) and ratio >= .95,
        canonical_verdict=st_judge.judge(base+cand, cand[0]['arm_sha'], base[0]['arm_sha']),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, nargs='+', required=True)
    p.add_argument('--candidate', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
