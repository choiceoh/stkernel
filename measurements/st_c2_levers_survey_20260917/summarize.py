#!/usr/bin/env python3
"""Tables from a router_cells ticket's event stream (jsonl or a run log). stdlib only.

    python3 summarize.py router-373dc770.jsonl     # the final cut (v6): the number this record reports
    python3 summarize.py router-79a41b03.jsonl     # the first sized cut (48 CTAs, plain loads)
"""
import json
import sys


def events(path):
    for line in open(path, errors='replace'):
        line = line.strip()
        if line.startswith('{"event"'):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main(path):
    ev = list(events(path))
    floors = {(e['fixture'], e['scope'], e['cache']): e['mean_change_pct']
              for e in ev if e['event'] == 'timing' and e['candidate'] == 'served_b'}
    print('| fixture | scope | cache | served us | fused us | change % | floor % | saved us a layer |')
    print('|---|---|---|---:|---:|---:|---:|---:|')
    for e in ev:
        if e['event'] == 'timing' and e['candidate'] == 'fused':
            key, n = (e['fixture'], e['scope'], e['cache']), e['layers']
            print(f"| {e['fixture']} | {e['scope']} ({n}) | {e['cache']} | {e['control_us']['mean']:.1f} | "
                  f"{e['candidate_us']['mean']:.1f} | {e['mean_change_pct']:+.1f} | {floors.get(key, float('nan')):+.2f} | "
                  f"{(e['control_us']['mean'] - e['candidate_us']['mean']) / n:.1f} |")
    print()
    print('| fixture | scope | rows judged | set flips | order flips | own set | own order | logits max ulps | '
          'weights max ulps | own weights ulps | control self diff |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for e in ev:
        if e['event'] == 'exact':
            print(f"| {e['fixture']} | {e['scope']} | {e['rows_judged']} | {e['set_mismatch_rows']} ({e['set_flip_pct']:.2f}%) | "
                  f"{e['order_mismatch_rows']} ({e['order_flip_pct']:.2f}%) | {e['own_set_mismatch_rows']} | "
                  f"{e['own_order_mismatch_rows']} | {e['logits_max_ulps']:.0f} | {e['weights_max_ulps']:.1f} | "
                  f"{e['own_weights_max_ulps']:.1f} | {e['control_self_diff']} |")
    for e in ev:
        if e['event'] == 'identity':
            print()
            print('gpu', e['gpu'], 'torch', e['torch'], 'cuda', e['cuda'], 'kernel',
                  e['source_sha256'].get('engine/kernels/router_fused.cu', '')[:12])
        elif e['event'] == 'router_verdict':
            print('step_ms_saved_chain_evicted', e['step_ms_saved_chain_evicted'])
        elif e['event'] == 'complete':
            print('complete', e['status'], e['failed'], 'max_alloc_MiB', round(e['max_allocated_bytes'] / 2 ** 20, 1))


if __name__ == '__main__':
    main(sys.argv[1])
