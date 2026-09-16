"""Summarize retained onepass records using measured request times, not step/s * acceptance.

Usage: python3 summarize.py consumer.jsonl > consumer-summary.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def pooled(requests):
    usable = [r for r in requests if r.get('completion_tokens', 0) > 1 and r.get('decode_s', 0) > 0]
    return (sum(r['completion_tokens'] - 1 for r in usable) / sum(r['decode_s'] for r in usable)
            if usable else None)


def summarize(record):
    by_context = defaultdict(list)
    for request in record.get('requests', []):
        by_context[request['ctx']].append(request)
    decode = record.get('decode', {})
    return dict(arm=record['name'], run=record['run_index'], run_id=record['run_id'],
        sha=record['arm_sha'], engine_source_sha256=record.get('engine_source_sha256'),
        image=record.get('image'), shape=record.get('engine_shape'),
        profile=record.get('workload_profile'), evidence_issues=record.get('evidence_issues', []),
        quality=record.get('quality'), korean=record.get('korean'),
        k=decode.get('num_spec'), acceptance=decode.get('acc_raw'),
        step_s=decode.get('windows_med'), observed_step_s=decode.get('raw_windows_med', decode.get('windows_med')),
        request_decode_tok_s=pooled(record.get('requests', [])),
        contexts={str(ctx): dict(decode_tok_s=pooled(requests),
            ttft_s=[r['ttft_s'] for r in requests],
            completion_tokens=[r['completion_tokens'] for r in requests],
            output_sha256=[r['output_sha256'] for r in requests]) for ctx, requests in by_context.items()},
        concurrency_coverage=record.get('concurrency_coverage'), quality_concurrent=record.get('quality_c4'),
        concurrent=[{k: r.get(k) for k in ('concurrency', 'ctx', 'aggregate_output_tok_s', 'valid', 'issues')}
                    for r in record.get('c4', [])])


def compare(base, candidate):
    def keyed(record):
        return {(r['ctx'], str(r['question']), r.get('rep', 0)): r for r in record.get('requests', [])}
    b, a = keyed(base), keyed(candidate)
    if not b or b.keys() != a.keys():
        raise ValueError('the two arms do not contain the same request cases')
    if base['workload'] != candidate['workload'] or base['engine_shape'] != candidate['engine_shape']:
        raise ValueError('workload or serving shape differs')
    requests = []
    for key in b:
        br, ar = b[key], a[key]
        if br['workload_sha256'] != ar['workload_sha256']:
            raise ValueError(f'unsalted workload differs: {key}')
        requests.append(dict(case=key, output_equal=br['output_sha256'] == ar['output_sha256'],
            tokens_equal=br['completion_tokens'] == ar['completion_tokens'],
            base_tok_s=br['decode_tok_s'], candidate_tok_s=ar['decode_tok_s'],
            base_ttft_s=br['ttft_s'], candidate_ttft_s=ar['ttft_s']))
    bs, cs = pooled(base['requests']), pooled(candidate['requests'])
    equal = all(r['output_equal'] and r['tokens_equal'] for r in requests)
    eligible = equal and not base.get('evidence_issues') and not candidate.get('evidence_issues')
    return dict(run=base['run_index'], all_outputs_equal=equal, eligible_for_speed_claim=eligible,
        base_request_tok_s=bs, candidate_request_tok_s=cs,
        observed_request_tok_s_change_pct=100 * (cs / bs - 1), requests=requests,
        evidence_issues=dict(base=base.get('evidence_issues', []), candidate=candidate.get('evidence_issues', [])))


if __name__ == '__main__':
    records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    indexed = {(r['name'], r['run_index']): r for r in records}
    if len(indexed) != len(records):
        raise ValueError('duplicate arm/run: select one matched B/A bracket')
    pairs = [compare(indexed['B', run], indexed['A', run]) for run in (1, 2)
             if ('B', run) in indexed and ('A', run) in indexed]
    print(json.dumps(dict(runs=[summarize(r) for r in records], pairs=pairs,
        complete=len(pairs) == 2, scope='measured consumer intervals; adoption also requires quality and valid evidence'),
        indent=2, ensure_ascii=False))
