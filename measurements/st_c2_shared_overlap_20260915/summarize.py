"""Tables for the C=2 shared-overlap gate (probes/engine_moe_shared_overlap.py). Stdlib only.

usage: python3 summarize.py RAW.json [RAW.json ...]
"""
import json
import sys
from statistics import mean


def pct(value):
    return f'{100 * value:+.2f}%'


def us(ms):
    return f'{1e3 * ms:.1f}'


def bracket_ratios(record, arm, base):
    ratios = []
    for bracket in range(record['samples']):
        entries = [e for e in record['entries'] if e['bracket'] == bracket]
        b = mean(e['median_ms'] for e in entries if e['arm'] == base)
        a = mean(e['median_ms'] for e in entries if e['arm'] == arm)
        ratios.append(a / b - 1)
    return ratios


def main(paths):
    for path in paths:
        data = json.load(open(path))
        rows = data['records']
        print(f'## {path}: passed={data.get("passed")} exact={data.get("exact")} error={data.get("error")}')
        identity = next((r for r in rows if r['lane'] == 'identity'), {})
        print(f'rank file: {identity.get("rank_file")}; recipe {identity.get("moe_static")}; '
              f'torch {identity.get("torch")} cuda {identity.get("cuda")}; {identity.get("gpu")}')
        print()
        print('| Rows | Draws (eager) | Output elements | Output bytes / values differing | Activation elements | '
              'Activation bytes / values differing | Captured replays, bytes / values differing | Distinct BF16 gate / up |')
        print('|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in (r for r in rows if r['lane'] == 'shared_audit'):
            print(f"| {r['rows']} | {r['draws']} | {r['output_elements']:,} | {r['output_mismatched']} / "
                  f"{r['output_value_mismatched']} | {r['activation_elements']:,} | {r['activation_mismatched']} / "
                  f"{r['activation_value_mismatched']} | {r['captured_replays']}, {r['captured_mismatched']} / "
                  f"{r['captured_value_mismatched']} | {r['distinct_bf16']['gate']:,} / {r['distinct_bf16']['up']:,} |")
        print()
        print('| Rows | Cells | A cells differing (bytes / values) | Third arm cells differing | Repeat cells differing | '
              'Dispatch: overlap calls / serial linears |')
        print('|---:|---:|---|---|---:|---|')
        dispatch = {r['rows']: r['dispatch'] for r in rows if r['lane'] == 'consumer_dispatch'}
        for r in (r for r in rows if r['lane'] == 'consumer_numerics'):
            d = ', '.join(f"{arm} {v['overlap']}/{v['serial_linears']}" for arm, v in dispatch[r['rows']].items())
            third = ', '.join(f"{k.split('_')[0]} {v}" for k, v in r.items() if k.endswith('_mismatched_cells')
                              and k not in ('repeat_mismatched_cells',)) or '—'
            print(f"| {r['rows']} | {r['cells']} | {r['mismatched_cells']} ({r['mismatched_elements']} / "
                  f"{r['value_mismatched_elements']}) | {third} | {r['repeat_mismatched_cells']} | {d} |")
        cells = [r for r in rows if r['lane'] == 'consumer_cell']
        uniques = sorted({(r['rows'], r.get('request_groups'), r['unique_experts']) for r in cells if r['cell'] == 'real_router'})
        print(f'\nReal-router unique experts (rows, groups, U): {uniques}')
        print()
        print('| Rows | Fixture | U | Cache | B µs mean (min) | A µs mean (min) | A vs B mean / min | Third arm vs B mean / min |')
        print('|---:|---|---:|---|---:|---:|---:|---:|')
        timings = [r for r in rows if r['lane'] == 'timing']
        for r in timings:
            s, c = r['summary'], r['change']
            third = [arm for arm in c if arm != 'A']
            t = (f"{third[0]} {pct(c[third[0]]['mean'])} / {pct(c[third[0]]['min'])}" if third else '—')
            print(f"| {r['rows']} | {r['label']} | {r['unique_experts']} | {r['cache']} | "
                  f"{us(s['B']['mean_ms'])} ({us(s['B']['min_ms'])}) | {us(s['A']['mean_ms'])} ({us(s['A']['min_ms'])}) | "
                  f"{pct(c['A']['mean'])} / {pct(c['A']['min'])} | {t} |")
        print()
        print('Per-bracket A vs B:')
        for r in timings:
            print(f"  M{r['rows']} {r['label']} U{r['unique_experts']} {r['cache']}: "
                  + ' '.join(pct(v) for v in bracket_ratios(r, 'A', 'B')))
        components = [r for r in rows if r['lane'] == 'component_timing']
        if components:
            print()
            print('| Rows | Fixture | U | Cache | R: routed + cast/add µs | H: fused shared µs | C: serial chain µs | '
                  'B − R µs | A − R µs | A saving of min(H, C) |')
            print('|---:|---|---:|---|---:|---:|---:|---:|---:|---:|')
            for r in components:
                s = r['summary']
                main_row = next(t for t in timings if t['rows'] == r['rows'] and t['label'] == r['label']
                                and t['cache'] == r['cache'])
                b, a = main_row['summary']['B']['mean_ms'], main_row['summary']['A']['mean_ms']
                shared = min(s['H']['mean_ms'], s['C']['mean_ms'])
                print(f"| {r['rows']} | {r['label']} | {r['unique_experts']} | {r['cache']} | {us(s['R']['mean_ms'])} | "
                      f"{us(s['H']['mean_ms'])} | {us(s['C']['mean_ms'])} | {us(b - s['R']['mean_ms'])} | "
                      f"{us(a - s['R']['mean_ms'])} | {100 * (b - a) / shared:+.0f}% |")
        complete = next((r for r in rows if r['lane'] == 'complete'), None)
        if complete:
            print(f"\npeak allocated {complete['max_allocated_bytes'] / 2**30:.2f} GiB")


if __name__ == '__main__':
    main(sys.argv[1:])
