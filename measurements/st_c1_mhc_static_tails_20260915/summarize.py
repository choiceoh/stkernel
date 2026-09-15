"""Summarize a complete same-build native MHC gate; never serving throughput."""
import collections
import json
from pathlib import Path
import statistics
import sys


def main(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or rows[-1].get('event') != 'complete' or rows[-1].get('status') != 'PASS':
        raise ValueError('a complete passing probe is required')
    for event in ('exact', 'transitions', 'fp32_fallback'):
        checks = [r for r in rows if r.get('event') == event]
        if len(checks) != 2 or {r['packets'] for r in checks} != {False, True} or not all(r['bitwise'] for r in checks):
            raise ValueError(f'missing bitwise {event} checks for both input families')
    groups = collections.defaultdict(list)
    for row in rows:
        if row['event'] == 'timing':
            groups[row['packets'], row['scope'], row['cache']].append(row)
    expected = {(p, s, c) for p in (False, True) for s in ('single', 'chain') for c in ('warm', 'evicted')}
    if set(groups) != expected:
        raise ValueError('both input families, interval scopes and cache states are required')
    print('# Native C=1 MHC timing\n')
    print('Negative time change means faster. These are local native intervals; TP4 transport and serving are outside their scope.\n')
    print('| Input | Interval | Cache | Dynamic us | Static us | Time change | Capture range |')
    print('| --- | --- | --- | ---: | ---: | ---: | ---: |')
    for (packets, scope, cache), values in sorted(groups.items()):
        if len(values) != 2 or {v['capture'] for v in values} != {0, 1}:
            raise ValueError('two independent, opposite-order captures are required')
        control = statistics.mean(v['control_us'] for v in values)
        candidate = statistics.mean(v['candidate_us'] for v in values)
        changes = [v['change_pct'] for v in values]
        layers = {v['layers'] for v in values}
        if len(layers) != 1:
            raise ValueError('capture shapes differ')
        name = 'local packets' if packets else 'ordinary AR'
        print(f'| {name} | {scope} ({layers.pop()}) | {cache} | {control:.3f} | {candidate:.3f} | '
              f'{100*(candidate/control-1):+.2f}% | {min(changes):+.2f}% to {max(changes):+.2f}% |')


if __name__ == '__main__':
    main(sys.argv[1])
