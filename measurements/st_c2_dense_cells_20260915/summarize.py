"""Tables from a `probes/engine_dense_cells.py` event log (stdlib only).

    python3 summarize.py gpu.jsonl [more.jsonl ...]

One row per comparison: cell, rows, scope, cache, control/candidate mean and min
µs over their B/A/A/B samples, and the candidate's change. `exact` lines state
which arms matched bit for bit; `component_failed` lines are printed as they are.
"""
import json
import sys

# Calls per target forward on this rank (34 KDA, 11 DSA, 3 dense-MLP layers); query pairs count once.
CALLS = {'kda.in_proj': 34, 'kda.o_proj': 34, 'mla.o_proj': 11, 'mla.qkv_a': 11, 'mla.query': 11,
         'mlp.gate_up': 3, 'mlp.down': 3}


def main(paths):
    events = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('{'):
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    for e in events:
        if e.get('event') == 'identity':
            print(f"identity: torch {e.get('torch')} cuda {e.get('cuda')} rows {e.get('rows')} brackets {e.get('brackets')}")
            print('  kernels.cu', e.get('source_sha256', {}).get('engine/kernels/dense/kernels.cu'))
        elif e.get('event') == 'rows16_resources':
            for name, r in e['kernels'].items():
                print(f"  {name}: {r['registers']} registers, {r['blocks_per_sm']} blocks/SM, {r['smem']} B smem, local {r['local_bytes']}")
        elif e.get('event') == 'component_failed':
            print('FAILED', e.get('cell'), e.get('rows'), e.get('phase', 'baseline'), e.get('error'))
        elif e.get('event') == 'complete':
            print('complete:', e.get('status'), e.get('failed'))
    print()
    print('| cell | rows | scope | cache | control | candidate | control mean / min µs | candidate mean / min µs | mean Δ | min Δ |')
    print('|---|---:|---|---|---|---|---:|---:|---:|---:|')
    for e in events:
        if e.get('event') != 'timing':
            continue
        c, a = e['control_us'], e['candidate_us']
        print(f"| {e['cell']} | {e['rows']} | {e['scope']}{'' if e.get('layers', 1) == 1 else ' ×' + str(e['layers'])} | {e['cache']} "
              f"| {e['control']} | {e['candidate']} | {c['mean']:.2f} / {c['min']:.2f} | {a['mean']:.2f} / {a['min']:.2f} "
              f"| {e['mean_change_pct']:+.1f}% | {e['min_change_pct']:+.1f}% |")
    print()
    print('Per forward, from the chain scope (µs per layer x calls; component time only, not a step verdict):')
    print('| cell | rows | cache | control | candidate | control ms/forward | candidate ms/forward | Δ ms (mean) |')
    print('|---|---:|---|---|---|---:|---:|---:|')
    for e in events:
        if e.get('event') != 'timing' or e.get('scope') != 'chain' or e['cell'] not in CALLS:
            continue
        per = CALLS[e['cell']] / e.get('layers', 1) / 1000
        c, a = e['control_us']['mean'] * per, e['candidate_us']['mean'] * per
        print(f"| {e['cell']} | {e['rows']} | {e['cache']} | {e['control']} | {e['candidate']} | {c:.3f} | {a:.3f} | {a - c:+.3f} |")
    print()
    for e in events:
        if e.get('event') == 'exact':
            print(f"exact {e['cell']} rows {e['rows']} {e['scope']} {e.get('phase', 'baseline')}: arms {e['arms']} "
                  f"(reference {e['reference']}, not projections {e.get('not_projections', [])}), plan {e.get('plan')}")


if __name__ == '__main__':
    main(sys.argv[1:])
