"""Recompute component observations; failure rows never become speed verdicts."""
import collections
import json
from pathlib import Path
import statistics
import sys


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    failed = [r for r in rows if r['event'] == 'component_failed']
    out = dict(file=path.name, identity=rows[0], counts=dict(collections.Counter(r['event'] for r in rows)),
               complete=next((r for r in reversed(rows) if r['event'] == 'complete'), None),
               failed=failed, latency=[])
    for r in rows:
        if r['event'] != 'timing':
            continue
        control, candidate = r.get('control', 0), r.get('candidate', 1)
        samples = r['samples']
        b = [s['us'] for s in samples if s['arm'] == control]
        a = [s['us'] for s in samples if s['arm'] == candidate]
        component = r.get('component', 'moe_' + str(candidate))
        if path.name == 'gpu-v1.jsonl' and component == 'mla_tile32':
            component = 'mla_sync_cleanup'  # v1 inherited the previous probe's label
        out['latency'].append(dict(component=component,
            **{k: r[k] for k in ('rows', 'width', 'case', 'cache', 'packets', 'scope', 'layers') if k in r},
            control_median_us=statistics.median(b), candidate_median_us=statistics.median(a),
            median_latency_change_pct=100*(statistics.median(a)/statistics.median(b)-1),
            mean_latency_change_pct=100*(statistics.mean(a)/statistics.mean(b)-1)))
    return out


if __name__ == '__main__':
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path(__file__).parent.glob('gpu-v*.jsonl'))
    print(json.dumps([summarize(path) for path in paths], indent=2))
