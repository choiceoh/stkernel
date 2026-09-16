"""How many expert reads an ACE-style slot skip removes from real C=1 verify steps.

Input: the `route_dump` rows (ace_route_probe.py) and rank 0's binary dump (debug-ace-route-dump f19573f0):
per step int16 ids then float32 weights, both [layers, rows, top-k]. Weights are GLM's renormalised sigmoid scores
times routed_scaling_factor, so the normalised gate of a slot is w / sum(w).

A skipped slot saves bytes only if no other slot of the same step and layer reads that expert: the 8 verify rows
share experts (41.9 distinct of 64 slots a layer in PR #966). The rule is ACE's: always keep each token's top-1,
skip a slot whose score is below one global threshold chosen as a quantile of the non-top-1 scores. The score is the
normalised gate, optionally multiplied by a per-expert table [layers, experts] (ACE's GSP/RCR importance, when one
is supplied with --table).

    python3 analyze_routes.py rows.jsonl routes.bin [--table gsp.npy] [--moe-share 0.458]
"""
import argparse
import json

import numpy as np

SKIPS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def load(rows_path, bin_path):
    rows = [json.loads(line) for line in open(rows_path)]
    rows = [r for r in rows if r.get("kind") == "route_dump"]
    blob = np.memmap(bin_path, dtype=np.uint8, mode="r")
    steps = []
    for r in rows:
        L, R, K = r["layers"], r["rows"], r["topk"]
        n_ids = L * R * K * 2
        raw = np.asarray(blob[r["offset"]: r["offset"] + r["nbytes"]])
        if raw.size != n_ids + L * R * K * 4:
            raise ValueError(f"row at offset {r['offset']} has {raw.size} bytes, expected {n_ids + L * R * K * 4}")
        ids = raw[:n_ids].view(np.int16).reshape(L, R, K).astype(np.int32)
        w = raw[n_ids:].view(np.float32).reshape(L, R, K)
        steps.append((r, ids, w))
    return steps


def distinct(ids, keep):
    """Distinct experts per layer over the kept slots: [layers]."""
    L = ids.shape[0]
    out = np.empty(L, dtype=np.int64)
    for layer in range(L):
        out[layer] = np.unique(ids[layer][keep[layer]]).size
    return out


def scores(ids, w, table):
    p = w / np.maximum(w.sum(-1, keepdims=True), 1e-20)
    if table is not None:
        layer_idx = np.arange(ids.shape[0])[:, None, None]
        p = p * table[layer_idx, ids]
    return p


def top1_mask(w):
    top = np.zeros(w.shape, dtype=bool)
    arg = w.argmax(-1)
    np.put_along_axis(top, arg[..., None], True, axis=-1)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("bin")
    ap.add_argument("--table", help="npy [layers, experts] multiplicative expert importance (ACE GSP/RCR)")
    ap.add_argument("--moe-share", type=float, default=0.458,
                    help="MoE kernel share of the C=1 K=7 iteration (PR #963: 25.13 of 54.89 ms)")
    ap.add_argument("--json", help="write the summary here")
    a = ap.parse_args()
    table = np.load(a.table) if a.table else None
    steps = [s for s in load(a.rows, a.bin) if s[0]["rows"] == 8]
    if not steps:
        raise SystemExit("no 8-row steps")
    L, R, K = steps[0][1].shape
    # thresholds from the pooled non-top-1 scores, so one global tau per operating point (ACE's quantile mapping)
    pool = np.concatenate([scores(ids, w, table)[~top1_mask(w)] for _, ids, w in steps])
    base = np.stack([distinct(ids, np.ones_like(ids, dtype=bool)) for _, ids, _ in steps])   # [steps, layers]
    slots = L * R * K
    summary = dict(steps=len(steps), layers=int(L), rows=int(R), topk=int(K),
                   distinct_per_layer=float(base.mean()), uniform_random=288 * (1 - (280 / 288) ** 8),
                   tokens_per_step=float(np.mean([s[0]["committed"] for s in steps])), table=a.table, points=[])
    print(f"steps {len(steps)} ({R} rows, {L} MoE layers, top-{K}); tokens/step {summary['tokens_per_step']:.3f}")
    print(f"distinct experts a layer, all slots kept: {base.mean():.2f} of {R * K} slots "
          f"(uniform random would be {summary['uniform_random']:.1f})")
    print(f"{'target skip':>11} {'tau':>9} {'skipped':>8} {'distinct':>9} {'reads':>7} {'reads/slot':>10} {'step est':>9}")
    for target in SKIPS:
        # fraction of ALL slots; top-1 slots (1/K of them) are never eligible
        q = target * K / (K - 1)
        if q >= 1:
            continue
        tau = float(np.quantile(pool, q))
        skipped, after = 0, []
        for _, ids, w in steps:
            keep = (scores(ids, w, table) >= tau) | top1_mask(w)
            skipped += int((~keep).sum())
            after.append(distinct(ids, keep))
        after = np.stack(after)
        skip_frac = skipped / (slots * len(steps))
        reads = 1 - after.sum() / base.sum()
        point = dict(target=target, tau=tau, skipped=skip_frac, distinct_per_layer=float(after.mean()),
                     read_reduction=float(reads), reads_per_skipped_slot=float(reads / skip_frac) if skip_frac else 0.0,
                     step_time_upper=float(reads * a.moe_share))
        summary["points"].append(point)
        print(f"{target:>10.0%} {tau:>9.4f} {skip_frac:>8.1%} {after.mean():>9.2f} {-reads:>+7.1%} "
              f"{point['reads_per_skipped_slot']:>10.2f} {-point['step_time_upper']:>+9.1%}")
    print("step est = read reduction x MoE share: an upper bound (MoE time taken as proportional to distinct experts, "
          "tokens/step taken as unchanged -- skipping changes the target, so acceptance must be measured)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
