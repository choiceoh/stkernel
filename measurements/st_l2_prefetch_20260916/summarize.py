#!/usr/bin/env python3
"""Tables from the L2-prefetch tickets' event streams (jsonl or a run log). stdlib only.

    python3 summarize.py moe-l-0bb8fbf0.jsonl          # engine_moe_c2_cells prefetch section
    python3 summarize.py dense-l2-7b57c7af.jsonl       # engine_dense_cells with the *_l2 arms
    python3 summarize.py dense-ksr-5ec0d03e.jsonl      # engine_dense_cells with the *_ksr<n> arms (1-ulp gate)
    python3 summarize.py moe-z-95588042.jsonl          # engine_moe_c2_cells bulk section (z / zn / zp storage orders)
"""
import json
import statistics
import sys


def events(path):
    for line in open(path, errors='replace'):
        line = line.strip()
        if line.startswith('{"event"'):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def per_bracket(e):
    s = e['samples']
    out = []
    for i in range(0, len(s), 4):
        b = [x['us'] for x in s[i:i + 4] if x['arm'] == e['control']]
        a = [x['us'] for x in s[i:i + 4] if x['arm'] == e['candidate']]
        if b and a:
            out.append(100 * (statistics.mean(a) / statistics.mean(b) - 1))
    return out


def main(path):
    for e in events(path):
        ev = e['event']
        if ev == 'exact':
            if 'arm' in e:      # the MoE cells: one arm against the control, ulps
                print(f"exact  {e.get('fixture', ''):18s} rows={e['rows']:3d} {e['scope']:6s} arm={e['arm']:9s} "
                      f"noise_control={e['noise_control']} passed={e['passed']} fp32_max_ulps={e.get('fp32_max_ulps')} "
                      f"bf16_differing={e.get('bf16_differing')}")
            else:               # the dense cells: every arm reproduced the reference's bytes (or the section failed)
                print(f"exact  {e['cell']:16s} rows={e['rows']:3d} {e['scope']:6s} arms={e['arms']} reference={e['reference']} "
                      f"plan={e.get('plan')}")
        elif ev == 'timing':
            pb = ' '.join(f'{v:+.2f}' for v in per_bracket(e))
            what = e.get('fixture') or e.get('cell')
            print(f"timing {str(what):26s} rows={e['rows']:3d} {e['scope']:6s} {e['cache']:7s} {e['control']}->{e['candidate']:11s} "
                  f"ctl={e['control_us']['mean']:9.1f}/{e['control_us']['min']:9.1f} cand={e['candidate_us']['mean']:9.1f}/"
                  f"{e['candidate_us']['min']:9.1f} mean%={e['mean_change_pct']:+6.2f} min%={e['min_change_pct']:+6.2f} "
                  f"brackets[{pb}]")
        elif ev == 'rate':
            print(f"rate   {e['fixture']:26s} rows={e['rows']:3d} {e['scope']:6s} {e['candidate']:4s} "
                  f"{e['control_gbps_evicted']:6.1f} -> {e['candidate_gbps_evicted']:6.1f} GB/s evicted (U={e['unique_experts']})")
        elif ev == 'stamps':
            m = e['medians']
            print(f"stamps {e['arm']:8s} span {m['span_us']:7.1f} us  fc1 {m['fc1_quant_us']:6.1f} ({e['fc1_gbps_per_cta']:.2f} GB/s/cta)  "
                  f"fc2 {m['fc2_us']:5.1f} ({e['fc2_gbps_per_cta']:.2f})  idle tail {m['idle_tail_us']:5.1f}  U={e['unique_experts']}")
        elif ev == 'bulk_verdict':
            print(f"bulk   exact_kinds={e['exact_kinds']} failed_kinds={e['failed_kinds']}")
        elif ev == 'component_failed':
            print('FAILED', e.get('cell') or e.get('section'), e.get('rows'), e.get('error', '')[:300])
        elif ev == 'complete':
            print('complete', e['status'], e.get('failed'))


if __name__ == '__main__':
    main(sys.argv[1])
