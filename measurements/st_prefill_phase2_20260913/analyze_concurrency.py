"""Read-only client scaling and recorded bounded-device iteration accounting.

No inference/profiler is started. Device body rates exclude prefill, host gaps
and unrecorded ordinary iterations; they are not consumer throughput scores.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys


def body(rows):
    if not rows:
        return None
    active = sum(sum(n > 0 for n in r['committed']) for r in rows)
    committed = sum(sum(r['committed']) for r in rows)
    accepted = sum(sum(r['accepted']) for r in rows)
    seconds = sum(r['duration_us'] for r in rows) / 1e6
    return dict(iterations=len(rows), active_row_iterations=active,
                committed=committed, accepted=accepted,
                bounded_acceptance=accepted/(6*active),
                median_ms=median(r['duration_us'] for r in rows)/1000,
                mean_ms=seconds*1000/len(rows), device_seconds=seconds,
                body_output_tok_s=committed/seconds,
                stage_mean_ms={k: mean(r['stages_us'][k] for r in rows)/1000
                               for k in rows[0]['stages_us']})


def summarize(base):
    record = json.loads((base / 'record-before-user-cancellation.json').read_text())
    raw = [json.loads(line) for line in (base/'requests.jsonl').read_text().splitlines()]
    phases = ['measure-c1'] + [g['latency_artifacts'] for g in record['c4']]
    result = dict(candidate=record['arm_sha'], run_id=record['run_id'], spec_k=6,
                  scope='retained profiler-off device bodies and client requests; not an accepted performance result',
                  phases={}, client_ratios=[])
    for phase in phases:
        path = base/phase/'server.json'
        data = path.read_bytes()
        ranks = json.loads(data)['ranks']
        assert {r['rank'] for r in ranks} == {0, 1, 2, 3}
        assert all(r['complete'] and not r['diagnostic'] and not r['preparation_changed'] for r in ranks)
        key = lambda rank: [(r['rows'], r['positions'], r['committed'], r['accepted'])
                            for r in rank['rows'] if r['kind']=='gpu_iteration']
        zero = next(r for r in ranks if r['rank']==0)
        assert all(key(r)==key(zero) for r in ranks), 'rank token records differ'
        active, per_request = {}, defaultdict(list)
        iterations = []
        for row in zero['rows']:
            if row['kind']=='request' and row['operation']=='admit':
                active[row['row']] = row['request_id']
            if row['kind']=='gpu_iteration':
                iterations.append(row)
                if phase=='measure-c1':
                    assert len(row['rows'])==1
                    per_request[active[row['rows'][0]]].append(row)
        ordinary = [r for r in zero['rows'] if r['kind']=='host_step'
                    and r.get('phase')=='decode' and r.get('dispatch')=='_run']
        scored = [r for r in raw if r['phase']==phase]
        expected = sum(r['completion_tokens']-1 for r in scored if not r.get('fixed_decode'))
        b = body(iterations)
        result['phases'][phase] = dict(server_sha256=hashlib.sha256(data).hexdigest(),
            source=str(path), rank_token_records_identical=True, bounded=b,
            by_width={str(w): body([r for r in iterations if len(r['rows'])==w])
                      for w in sorted({len(r['rows']) for r in iterations})},
            c1_by_request={str(k): body(v) for k,v in per_request.items()},
            ordinary_host_iterations=len(ordinary), consumer_decode_tokens=expected,
            bounded_commit_coverage=b['committed']/expected)
    natural = [r for r in raw if r['phase']=='measure-c1' and not r.get('fixed_decode')]
    for group in record['c4']:
        rows = group['requests']
        one = next(r for r in natural if (r['ctx'],r['question'])==(rows[0]['ctx'],rows[0]['question']))
        start = min(r['started_monotonic'] for r in rows)
        firsts = [r['started_monotonic']+r['ttft_s']-start for r in rows]
        ends = [r['ended_monotonic']-start for r in rows]
        c1_rate = one['completion_tokens']/one['elapsed_s']
        result['client_ratios'].append(dict(ctx=one['ctx'],question=one['question'],
            c1_output_tok_s_including_prefill=c1_rate,
            c4_output_tok_s_including_prefill=group['aggregate_output_tok_s'],
            matched_metric_ratio=group['aggregate_output_tok_s']/c1_rate,
            all_four_after_first_token_s=min(ends)-max(firsts),
            tail_after_first_completion_s=max(ends)-min(ends), elapsed_s=group['elapsed_s'],
            first_token_arrivals_s=firsts,completion_arrivals_s=ends))
    return result


if __name__=='__main__':
    print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
