"""Empirical: what set does torch.topk(sorted=False) pick when the k-th value ties?

Hypothesis (what the prefill radix kernel assumes): the set is
    {i : v[i] > kth}  U  {the lowest indices among {i : v[i] == kth}}
i.e. exact ties choose the LOWER index.
"""
import torch

dev = "cuda"
torch.manual_seed(0)


def expected_lower_index(v, k):
    """Reference set under the 'ties -> lower index' rule, per row."""
    out = []
    for row in v:
        order = sorted(range(row.numel()), key=lambda i: (-row[i].item(), i))
        out.append(set(order[:k]))
    return out


def check(rows, cols, k, tie_pop, label):
    # base values with a large tied plateau straddling the k-th boundary
    v = torch.randn(rows, cols, device=dev, dtype=torch.float32)
    # force a plateau: pick `tie_pop` random columns per row, set them all to the same value,
    # chosen so the plateau straddles k
    for r in range(rows):
        cols_sel = torch.randperm(cols, device=dev)[:tie_pop]
        v[r] = v[r].sort(descending=True).values[torch.randperm(cols, device=dev)]
        hi = v[r].topk(max(1, k - tie_pop // 2)).values[-1]
        v[r, cols_sel] = hi
    idx = torch.topk(v, k, dim=-1, sorted=False).indices
    got = [set(r.tolist()) for r in idx.cpu()]
    want = expected_lower_index(v.cpu(), k)
    bad = [r for r in range(rows) if got[r] != want[r]]
    # how many rows actually had a tie at the boundary?
    kth = torch.topk(v, k, dim=-1).values[:, -1]
    straddle = int(((v == kth[:, None]).sum(-1) > 1).sum())
    print(f"{label:38s} rows={rows:3d} cols={cols:6d} k={k:4d} "
          f"boundary-ties={straddle:3d}/{rows}  mismatched rows={len(bad)}")
    return len(bad), straddle


print("torch", torch.__version__, torch.cuda.get_device_name(0))
tot_bad = tot_tie = 0
for rows, cols, k, tie in [
    (16, 1024, 512, 64),        # small: single-block topk
    (16, 4096, 512, 128),
    (8, 8192, 512, 200),
    (16, 16384, 512, 300),      # near the 20000-col multiblock threshold
    (16, 32768, 512, 400),      # multiblock
    (16, 50688, 512, 512),      # 202752 ctx bucket
    (4, 65536, 512, 900),
]:
    b, s = check(rows, cols, k, tie, f"ties={tie}")
    tot_bad += b; tot_tie += s

# and the degenerate case the ledger's stage 3 uses: lots of exact 0.0 (relu output)
v = torch.zeros(16, 32768, device=dev)
v[:, ::7] = torch.randn(16, (32768 + 6) // 7, device=dev).abs()
idx = torch.topk(v, 512, dim=-1, sorted=False).indices
got = [set(r.tolist()) for r in idx.cpu()]
want = expected_lower_index(v.cpu(), 512)
bad = sum(1 for r in range(16) if got[r] != want[r])
print(f"{'relu zeros plateau':38s} rows= 16 cols= 32768 k= 512 mismatched rows={bad}")
tot_bad += bad

print(f"\nTOTAL mismatched rows: {tot_bad}   (rows that actually had a boundary tie: {tot_tie})")
