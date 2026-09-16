"""Print the lane's timing table and exactness failures: python3 summarize.py topk-hpcops-0917.jsonl"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
device = next(r for r in rows if r['event'] == 'device')
print(f"{device['name']}  torch {device['torch']}  cuda {device['cuda']}  smem/block {device['smem_per_block']} B")
t = {(r['kind'], r['rows'], r['n'], r['arm']): r['median_us'] for r in rows if r['event'] == 'timing'}
print(f"{'kind':>7} {'rows':>5} {'n_cand':>7} {'ours us':>9} {'hpcops us':>10} {'floor us':>9} {'ours/floor':>10} {'hpc/floor':>9} {'ours/hpc':>8}")
for kind, r, n in sorted({k[:3] for k in t}):
    own = t[(kind, r, n, 'st_dsa_select' if kind == 'decode' else 'prefill_topk')]
    hpc, floor = t.get((kind, r, n, 'hpcops')), t[(kind, r, n, 'read_floor')]
    cells = (f"{hpc:>10.1f}", f"{hpc / floor:>9.2f}", f"{own / hpc:>8.2f}") if hpc else (f"{'inexact':>10}", f"{'-':>9}", f"{'-':>8}")
    print(f"{kind:>7} {r:>5} {n:>7} {own:>9.1f} {cells[0]} {floor:>9.1f} {own / floor:>10.2f} {cells[1]} {cells[2]}")
print("set mismatches against masked torch.topk:")
for r in rows:
    if r['event'] == 'exactness' and r['mismatched_rows']:
        print(f"  {r['kind']} {r['rows']}x{r['n']} {r['dist']}: {r['arm']} {r['mismatched_rows']} rows")
