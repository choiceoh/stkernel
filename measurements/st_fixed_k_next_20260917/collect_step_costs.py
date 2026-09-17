"""Read measured device iteration costs and acceptance; never start a profiler.

Run on the artifact host with the completed consumer JSONL. Rank 0 is counted
once; C2's single-request startup/tail iterations are separate from width two.
These durations cover the device body and TP4 stop agreement, not HTTP time.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys


def summarize(rows, k):
    durations = [row['duration_us'] for row in rows]
    accepted = sum(sum(row['accepted']) for row in rows)
    slots = sum(len(row['rows']) for row in rows)
    committed = sum(sum(row['committed']) for row in rows)
    stages = sorted({key for row in rows for key in row.get('stages_us', {})})
    return dict(
        steps=len(rows), request_steps=slots,
        sum_us=sum(durations), mean_ms=statistics.mean(durations) / 1000,
        median_ms=statistics.median(durations) / 1000,
        body_step_s=1e6 * len(rows) / sum(durations),
        accepted=accepted, drafted=k * slots, acceptance=accepted / (k * slots),
        committed=committed, committed_per_request_step=committed / slots,
        mean_stages_ms={key: sum(row.get('stages_us', {}).get(key, 0) for row in rows)
                       / len(rows) / 1000 for key in stages})


def collect(path):
    runs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        root = Path(record['artifacts'])
        k = record['decode']['num_spec']
        groups, sources = defaultdict(list), []
        phases = [('measure-c1', record['requests'], True)]
        phases += [(row['latency_artifacts'], row['requests'], False)
                   for row in record.get('c4', [])]
        for phase, requests, c1 in phases:
            source = root / phase / 'latency.jsonl'
            raw = source.read_bytes()
            metadata = json.loads(source.with_name('server.json').read_text())
            ranks = metadata['ranks']
            if len(ranks) != 4 or any(row['diagnostic'] for row in ranks):
                raise ValueError(f'{phase}: require four unprofiled measurement ranks')
            if any(row['preparation_changed'] for row in ranks):
                raise ValueError(f'{phase}: preparation changed during measurement')
            admissions, request_index = [], -1
            counts = defaultdict(int)
            committed_by_request = defaultdict(int)
            for entry in raw.splitlines():
                row = json.loads(entry)
                if row.get('rank') != 0:
                    continue
                if row['kind'] == 'request' and row['operation'] == 'admit':
                    admissions.append(row['request_id'])
                    if c1:
                        request_index += 1
                if row['kind'] != 'gpu_iteration' or row['phase'] != 'decode':
                    continue
                if row['operation'] != 'bounded_decode' or row['duration_us'] <= 0:
                    raise ValueError(f'{phase}: unexpected iteration')
                width = len(row['rows'])
                if len(row['accepted']) != width or len(row['committed']) != width:
                    raise ValueError(f'{phase}: incomplete iteration counters')
                if c1:
                    if width != 1 or not 0 <= request_index < len(requests):
                        raise ValueError(f'{phase}: request ownership is ambiguous')
                    request = requests[request_index]
                    kind = 'fixed_c1' if request.get('fixed_decode') else 'ordinary_c1'
                    ctx = request['ctx']
                    if not counts[request_index] and row['positions'] != [request['prompt_tokens']]:
                        raise ValueError(f'{phase}: first iteration does not match prompt length')
                    counts[request_index] += 1
                    committed_by_request[request_index] += sum(row['committed'])
                else:
                    kind, ctx = 'ordinary_c2', requests[0]['ctx']
                groups[kind, ctx, width].append(row)
            if len(admissions) != len(requests) or len(set(admissions)) != len(requests):
                raise ValueError(f'{phase}: admission count does not match requests')
            required = {i for i, request in enumerate(requests) if not request.get('fixed_decode')}
            if c1 and not required.issubset(counts):
                raise ValueError(f'{phase}: missing request iterations')
            if c1 and any(committed_by_request[i] != requests[i]['completion_tokens'] - 1
                          for i in counts):
                raise ValueError(f'{phase}: iteration tokens do not match completed requests')
            sources.append(dict(phase=phase, path=str(source),
                sha256=hashlib.sha256(raw).hexdigest(), admissions=admissions,
                c1_steps_by_request=dict(counts),
                c1_committed_by_request=dict(committed_by_request),
                missing_device_iterations=[i for i in range(len(requests)) if i not in counts]
                    if c1 else [], preparation_unchanged=True))
        runs.append(dict(arm=record['name'], run=record['run_index'],
            run_id=record['run_id'], sha=record['arm_sha'], rank=0, k=k,
            sources=sources, groups=[dict(kind=key[0], ctx=key[1], width=key[2],
                                         **summarize(rows, k))
                                    for key, rows in sorted(groups.items())]))
    indexed = {(row['arm'], row['run']): row for row in runs}
    pairs = []
    for run in (1, 2):
        if ('A', run) not in indexed or ('B', run) not in indexed:
            continue
        bg, ag = ({(g['kind'], g['ctx'], g['width']): g for g in indexed[arm, run]['groups']}
                  for arm in ('B', 'A'))
        if bg.keys() != ag.keys():
            raise ValueError('device iteration group coverage differs')
        pairs.append(dict(run=run, groups=[dict(kind=key[0], ctx=key[1], width=key[2],
            base=bg[key], candidate=ag[key],
            mean_cost_reduction_pct=100 * (1 - ag[key]['mean_ms'] / bg[key]['mean_ms']),
            acceptance_change_pp=100 * (ag[key]['acceptance'] - bg[key]['acceptance']))
            for key in bg]))
    return dict(scope='Rank 0 measured device body and TP4 stop agreement, globaltimer; '
                'not HTTP latency or profiler replay. C2 width one is excluded from width two. '
                'Fixed-output requests without bounded iteration records use the separate wall-time metric. '
                'Outputs differ and quality failures remain; observations do not establish causality.',
                runs=runs, pairs=pairs)


if __name__ == '__main__':
    print(json.dumps(collect(sys.argv[1]), indent=2))
