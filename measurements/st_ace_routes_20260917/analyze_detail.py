"""Detail for the ACE gate proxy on real routes: skipped routing mass, anchor vs draft rows, layer spread, per-request spread."""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
from analyze_routes import load, distinct, top1_mask

rows_path, bin_path = sys.argv[1], sys.argv[2]
steps = [s for s in load(rows_path, bin_path) if s[0]['rows'] == 8]
ids = np.stack([s[1] for s in steps])            # [S, L, R, K]
w = np.stack([s[2] for s in steps]).astype(np.float64)
p = w / w.sum(-1, keepdims=True)
top = np.zeros(p.shape, dtype=bool)
np.put_along_axis(top, p.argmax(-1)[..., None], True, axis=-1)
S, L, R, K = ids.shape
print(f"steps {S}; routed_scale sum of weights per token: mean {w.sum(-1).mean():.4f}")
sorted_p = -np.sort(-p, axis=-1)
print("mean normalised gate by rank 1..8: " + " ".join(f"{x:.3f}" for x in sorted_p.reshape(-1, K).mean(0)))
pool = p[~top]
out = {}
for target in (0.05, 0.10, 0.15, 0.20, 0.25):
    q = target * K / (K - 1)
    tau = np.quantile(pool, q)
    keep = (p >= tau) | top
    skipped_mass = (p * ~keep).sum(-1)                   # [S, L, R]
    per_tok_skips = (~keep).sum(-1)
    # distinct per (step, layer)
    base = np.array([[np.unique(ids[s, l]).size for l in range(L)] for s in range(S)])
    after = np.array([[np.unique(ids[s, l][keep[s, l]]).size for l in range(L)] for s in range(S)])
    red_layer = 1 - after.sum(0) / base.sum(0)
    # anchor row (0) vs draft rows
    anchor_skip = (~keep[:, :, 0]).mean()
    draft_skip = (~keep[:, :, 1:]).mean()
    # per-request (seq resets are not recorded; use contiguous context runs)
    ctx = np.array([s[0]['context'] for s in steps])
    breaks = np.where(np.diff(ctx) < 0)[0] + 1
    req_red = [1 - after[a:b].sum() / base[a:b].sum() for a, b in zip(np.r_[0, breaks], np.r_[breaks, S])]
    out[target] = dict(tau=float(tau), skipped_mass_mean=float(skipped_mass.mean()),
                       skipped_mass_p95=float(np.quantile(skipped_mass, .95)),
                       tokens_with_any_skip=float((per_tok_skips > 0).mean()),
                       max_skips_per_token=int(per_tok_skips.max()),
                       anchor_slot_skip=float(anchor_skip), draft_slot_skip=float(draft_skip),
                       read_reduction=float(1 - after.sum() / base.sum()),
                       layer_reduction_min=float(red_layer.min()), layer_reduction_max=float(red_layer.max()),
                       request_reduction=[round(float(x), 4) for x in req_red])
    o = out[target]
    print(f"skip {target:.0%}: tau {tau:.4f}; skipped routing mass mean {o['skipped_mass_mean']:.2%} (p95 {o['skipped_mass_p95']:.2%}); "
          f"tokens touched {o['tokens_with_any_skip']:.1%}, max {o['max_skips_per_token']}/token; "
          f"anchor slots {anchor_skip:.1%} vs draft {draft_skip:.1%}; reads -{o['read_reduction']:.1%} "
          f"(layers {o['layer_reduction_min']:.1%}..{o['layer_reduction_max']:.1%}; requests {min(req_red):.1%}..{max(req_red):.1%}, n={len(req_red)})")
json.dump(out, open('summary-detail.json', 'w'), indent=1)
