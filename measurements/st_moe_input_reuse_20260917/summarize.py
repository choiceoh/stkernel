"""Retain correctness gates and each B/A/A/B block; do not promote component timing."""
import json
from pathlib import Path
import statistics
import sys


def summarize(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    final = [r for r in records if r.get('event') == 'complete']
    frontend = [r for r in records if r.get('event') == 'frontend_bytes']
    exact = [r for r in records if r.get('event') == 'exact']
    timing = []
    for record in records:
        if record.get('event') != 'timing':
            continue
        samples, blocks = record['samples'], []
        if len(samples) % 4:
            raise ValueError('incomplete B/A/A/B block')
        for offset in range(0, len(samples), 4):
            b1, a1, a2, b2 = samples[offset:offset+4]
            if [r['arm'] for r in (b1, a1, a2, b2)] != [
                    record['control'], record['candidate'], record['candidate'], record['control']]:
                raise ValueError('unexpected bracket order')
            b, a = (b1['us'] + b2['us']) / 2, (a1['us'] + a2['us']) / 2
            blocks.append(dict(control_us=b, candidate_us=a, change_pct=100*(a/b-1)))
        changes = [b['change_pct'] for b in blocks]
        timing.append(dict(rows=record['rows'], scope=record['scope'], cache=record['cache'],
            candidate=record['candidate'], control_mean_us=record['control_us']['mean'],
            candidate_mean_us=record['candidate_us']['mean'], mean_change_pct=record['mean_change_pct'],
            median_block_change_pct=statistics.median(changes),
            block_range_pct=[min(changes), max(changes)], faster_blocks=sum(c < 0 for c in changes),
            blocks=blocks))
    return dict(source=str(path), identity=[r for r in records if r.get('event') in ('identity', 'device')],
        complete=len(final) == 1 and final[0].get('passed') is True,
        frontend_cells=len(frontend), frontend_failures=[r for r in frontend if r['mismatched_routes']],
        output_cells=len(exact), output_failures=[r for r in exact if not r['passed']],
        comparisons=timing, stamps=[r for r in records if r.get('event') == 'stamps'],
        scope='Same-build component evidence only; full consumer qualification is separate')


if __name__ == '__main__':
    print(json.dumps(summarize(sys.argv[1]), indent=2))
