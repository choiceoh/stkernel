"""Summarize probes/engine_mhc_c2_packed.py output: exact groups, diagnosis and B/A/A/B timing tables.

usage: summarize.py REPORT.json [--markdown]
"""
import json
import statistics
import sys
from collections import defaultdict


def main():
    rows = json.load(open(sys.argv[1]))
    markdown = '--markdown' in sys.argv
    exact = [r for r in rows if r['event'] == 'mhc_packed_exact']
    print(f'exact groups: {len(exact)}')
    for r in exact:
        print(f"  {r['label']:>11} {r['family']:>8} rows {r['rows']:>2} calls {r['calls']:>2} arms {','.join(r['arms'])}")
    for r in rows:
        if r['event'] in ('m14_diagnosis', 'm14_scalar_layout'):
            fields = {k: (v['mismatched'], v['total'], v.get('max_absolute'), v.get('max_relative'))
                      for k, v in r['fields'].items()}
            print(r['event'], r['rows'], r['input_scale'], r.get('layout', ''), fields)
        if r['event'] in ('mhc_packed_weights', 'direct_mhc_dispatch', 'identity', 'complete'):
            print(r['event'], {k: v for k, v in r.items() if k not in ('event', 'source_sha256')})
    samples = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['event'] != 'timing':
            continue
        key = (r['candidate'], r['rows'], r['cache'])
        for s in r['samples']:
            samples[key][s['arm']].append(s['ms'] * 1000.)
    if markdown:
        print('| Comparison | Rows | Cache | B mean µs | A mean µs | Change (mean) | B min µs | A min µs | Change (min) | n |')
        print('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for (name, width, cache), arms in sorted(samples.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        b, a = arms['B'], arms['A']
        bm, am, bn, an = statistics.mean(b), statistics.mean(a), min(b), min(a)
        if markdown:
            print(f'| {name} | {width} | {cache} | {bm:.1f} | {am:.1f} | {100 * (am / bm - 1):+.2f}% | {bn:.1f} | {an:.1f} '
                  f'| {100 * (an / bn - 1):+.2f}% | {len(b)}+{len(a)} |')
        else:
            print(f'{name:40s} rows {width:>2} {cache:>7}  B {bm:9.1f} A {am:9.1f} ({100 * (am / bm - 1):+6.2f}%)  '
                  f'min B {bn:9.1f} A {an:9.1f} ({100 * (an / bn - 1):+6.2f}%)  samples {[round(x, 1) for x in b]} {[round(x, 1) for x in a]}')


if __name__ == '__main__':
    main()
