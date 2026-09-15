"""Summarize the recorded GPU intervals, without inferring engine throughput."""
import json
from pathlib import Path
import sys


def summarize(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records or records[-1].get('event') != 'complete' or records[-1].get('status') != 'PASS':
        raise ValueError(f'{path}: the GPU run did not complete successfully')
    exact = [r for r in records if r['event'] in ('exact_pack', 'exact')]
    print(f'{path}: {len(exact)} exact groups; GPU component intervals only.\n')
    print('| Component | Scope | Cache | Warps/CTA | Control us | Candidate us | Change |')
    print('|---|---|---|---:|---:|---:|---:|')
    for row in records:
        if row['event'] != 'timing':
            continue
        name = row.get('cell', f"pack K={row.get('width')}")
        scope = row.get('scope', 'single')
        layers = row.get('layers', 1)
        print(f"| {name} | {scope} x{layers} | {row['cache']} | {row['candidate']} | "
              f"{row['control_us']:.3f} | {row['candidate_us']:.3f} | {row['change_pct']:+.2f}% |")


if __name__ == '__main__':
    for path in sys.argv[1:]:
        summarize(path)
