#!/usr/bin/env python3
"""Summarize moe_c2_cells events (jsonl, or a launch log containing event lines). stdlib only."""
import json
import statistics
import sys


def events(path):
    for line in open(path, errors='replace'):
        line = line.strip()
        if not line.startswith('{"event"'):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def per_bracket(e):
    """Candidate/control change of each B/A/A/B bracket (pairs the bracket's two B and two A samples)."""
    s = e['samples']
    out = []
    for i in range(0, len(s), 4):
        b = [x['us'] for x in s[i:i + 4] if x['arm'] == e['control']]
        a = [x['us'] for x in s[i:i + 4] if x['arm'] == e['candidate']]
        if b and a:
            out.append(100 * (statistics.mean(a) / statistics.mean(b) - 1))
    return out


def main():
    for e in events(sys.argv[1]):
        ev = e['event']
        if ev == 'timing':
            u = e.get('unique_experts')
            pb = ' '.join(f'{v:+.2f}' for v in per_bracket(e))
            print(f"timing {str(e.get('fixture')):28s} rows={e['rows']:5d} {e['scope']:6s} {e['cache']:7s} "
                  f"{e['control']}->{e['candidate']} ctl={e['control_us']['mean']:9.1f}/{e['control_us']['min']:9.1f} "
                  f"cand={e['candidate_us']['mean']:9.1f}/{e['candidate_us']['min']:9.1f} "
                  f"mean%={e['mean_change_pct']:+6.2f} min%={e['min_change_pct']:+6.2f} U={u} brackets[{pb}]")
        elif ev == 'exact':
            keys = [k for k in ('fp32_diff', 'fp32_max_abs', 'fp32_max_ulps', 'bf16_diff', 'bf16_max_abs', 'bf16_max_rel')
                    if k in e]
            stats = ' '.join(f"{k}={e[k]:.3g}" if isinstance(e[k], float) else f"{k}={e[k]}" for k in keys)
            print(f"exact {e['fixture']:24s} rows={e['rows']:5d} {e['scope']:6s} arm={e.get('arm')} "
                  f"noise_ctl={e.get('noise_control')} cells={e.get('cells')} elements={e.get('elements')} "
                  f"passed={e.get('passed')} U={e.get('unique_experts')} {stats}")
        elif ev in ('calibration', 'complete', 'component_failed'):
            print(ev, json.dumps({k: v for k, v in e.items() if k != 'event'})[:800])
        elif ev == 'rate':
            print(f"rate {e['fixture']:24s} rows={e['rows']} {e['scope']:6s} cand={e['candidate']} "
                  f"ctl={e['control_gbps_evicted']:.1f} cand={e['candidate_gbps_evicted']:.1f} GB/s "
                  f"(expert bytes {e['expert_bytes']/1e6:.1f} MB)")
        elif ev == 'stamps':
            m = e['medians']
            extra = ''
            if 'phase0_barrier1_us' in m:
                extra = (f" ph0+b1={m['phase0_barrier1_us']:.1f} route/quant+b2={m['route_quant_barrier2_us']:.1f}"
                         f" skew={m['start_skew_us']:.1f}")
            print(f"stamps arm={e.get('arm', e.get('chunk'))} rows={e['rows']} U={e['unique_experts']} "
                  f"span={m['span_us']:.1f} front={m['frontend_us']:.1f}{extra} fc1={m['fc1_quant_us']:.1f} "
                  f"pub={m['publish_us']:.2f} fc2={m['fc2_us']:.1f} tail={m['idle_tail_us']:.1f} "
                  f"dma_fc1={m['dma_fc1_us']:.1f} dma_fc2={m['dma_fc2_us']:.1f} "
                  f"items/cta={min(e['items_per_cta'])}..{max(e['items_per_cta'])} "
                  f"fc1 {e['fc1_gbps_per_cta']:.2f} fc2 {e['fc2_gbps_per_cta']:.2f} GB/s per CTA")


if __name__ == '__main__':
    main()
