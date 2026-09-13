"""Summarize exact kernel names on each profiled rank/step, preserving clocks."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def union(intervals):
    total, end = 0., float('-inf')
    for start, stop in sorted(intervals):
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    record = json.loads((args.run / 'record.json').read_text())
    report = dict(run_id=record['run_id'], source=record['arm_sha'], boot_id=record['boot_id'],
        scope='diagnostic GPU activities; exact kernel names, no inferred operator attribution; '
              'durations include dependency waits and may overlap; not unprofiled step latency',
        phases=[])
    for ctx in (2000, 32000, 128000):
        path = args.run / f'diagnostic-c1-{ctx}' / 'latency.jsonl'
        if not path.exists():
            continue
        spans, kernels, seen = collections.defaultdict(list), {}, collections.defaultdict(set)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                if row.get('kind') != 'gpu_activity' or row.get('phase') != 'decode':
                    continue
                rank, step = row['rank'], row['step']
                seen[rank].add(step)
                spans[rank, step].append((row['start_us'], row['start_us'] + row['duration_us']))
                if row.get('category') == 'kernel':
                    key = (rank, row['kernel'])
                    value = kernels.setdefault(key, dict(us=0., calls=0))
                    value['us'] += row['duration_us']; value['calls'] += 1
        result = dict(ctx=ctx, file_sha256=digest.hexdigest(), ranks=[])
        for rank, steps in sorted(seen.items()):
            top = sorted(((name, v) for (r, name), v in kernels.items() if r == rank),
                         key=lambda item: item[1]['us'], reverse=True)
            result['ranks'].append(dict(rank=rank, steps=sorted(steps),
                gpu_interval_union_ms_by_step={str(step): union(spans[rank, step]) / 1000 for step in sorted(steps)},
                top_kernels=[dict(kernel=name, calls=v['calls'],
                    average_ms_per_profiled_step=v['us'] / len(steps) / 1000) for name, v in top[:16]]))
        report['phases'].append(result)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(phases=len(report['phases']), output=str(args.output))))


if __name__ == '__main__':
    main()
