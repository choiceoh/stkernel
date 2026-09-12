#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Judge one arm against its baseline, with the noise floor beside the delta.

    python3 bench/judge.py FUS7                # base = the latest baseline on FUS7's build
    python3 bench/judge.py FUS7 --base FUS7BASE
    python3 bench/judge.py FUS7 --write        # also append the verdict to verdicts.jsonl

The delta alone misled twice on 2026-09-06 (+0.0% then +4.7% for the same
arm): the floor is what the SAME configuration does run to run. It is taken
from the baseline records of the same build (the spread of their decode
window medians), falling back to the same harness across builds, and the
verdict says whether the delta clears it. Gates: Korean corruption 0,
quality all, and the lane proof complete -- an unproved arm gets no verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline import (JSONL, deployed_git, for_build, is_baseline, load,
                      comparison_scope, comparison_baseline, required_proofs, proof_complete)  # noqa: E402

VERDICTS = os.environ.get("ONEPASS_VERDICTS",
                          os.path.join(os.path.dirname(JSONL), "verdicts.jsonl"))


def last(rows, name):
    r = [x for x in rows if x.get("name") == name and not x.get("rehearsal")]
    return r[-1] if r else None


def build_of(rec):
    return rec.get("overlay") or (rec.get("git") or "")[:7] or None


def baselines_on(rows, rec):
    """Baseline records of the arm's build (stamp first, git for older rows)."""
    b = build_of(rec)
    mine = [r for r in for_build(rows, rec.get("overlay"), rec.get("git")) if not r.get("rehearsal")
            and compatible(r, rec)]
    scope = comparison_scope()
    # Include a matching but unproved record so judge reports its invalidity;
    # counts, reuse and noise-floor consumers must still check its proof.
    return [r for r in mine if comparison_baseline(r, rec, scope=scope, require_proof=False)], b


def compatible(a, b):
    try:
        if required_proofs(a) != required_proofs(b):
            return False
    except ValueError:
        return False
    if b.get("overlay") and a.get("overlay") != b["overlay"]:
        return False
    return all(a.get(k) == b.get(k) for k in
               ("harness", "doc_lang", "thinking", "workload", "generation_budget", "runtime", "measurement_policy", "engine_shape", "quality_protocol"))


def record_errors(rec):
    """Missing gates and unknown markers cannot become reusable evidence."""
    errors = []
    errors.extend(rec.get('evidence_issues') or [])
    if rec.get('harness', 0) >= 41 and (rec.get('steady_state') or {}).get('valid') is not True:
        errors.append('steady preparation state not proven')
    q, k, d = (rec.get(key) or {} for key in ("quality", "korean", "decode"))
    if not isinstance(q.get("total"), int) or q["total"] <= 0 or q.get("ok") != q["total"]:
        errors.append(f"quality {q.get('ok')}/{q.get('total')}")
    if not isinstance(k.get("n"), int) or k["n"] <= 0 or k.get("dirty") != 0:
        errors.append(f"korean {k.get('dirty')}/{k.get('n')}")
    value = d.get("windows_med")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        errors.append("no finite decode window")
    if "knobs" not in rec:
        errors.append("knobs not attested")
    if unproved(rec):
        errors.append("unproved active knobs")
    return errors


def unproved(rec):
    return not proof_complete(rec)


def floor_of(rows, rec, objective=None):
    """(rel spread, n, scope): spread of windows_med over the baselines of the
    build; fewer than 2 -> the harness across builds; fewer than 2 -> None."""
    from measurement_contract import metric_value, metric_compatible
    objective = objective or {'metric': 'decode_steps'}
    bases, _ = baselines_on(rows, rec)
    def windows(records):
        # Multiple onepass records on the same attested boot are one noise
        # sample. Older untagged rows retain their legacy interpretation.
        boots = {}
        for index, x in enumerate(records):
            value = metric_value(x, objective)
            if not record_errors(x) and value is not None and metric_compatible(x, rec, objective):
                boots[x.get("boot_id") or ("legacy", index)] = value
        return list(boots.values())
    wins = windows(bases)
    scope = "same build"
    if len(wins) < 2 and comparison_scope() is None:
        wins = windows([x for x in rows
                if is_baseline(x)[0] and not x.get("rehearsal")
                and x.get("harness") == rec.get("harness")
                and all(x.get(k) == rec.get(k) for k in
                        ("doc_lang", "thinking", "workload", "generation_budget", "runtime"))
                and not record_errors(x) and required_proofs(x) == required_proofs(rec)])
        scope = "same harness, across builds"
    if len(wins) < 2:
        return None, len(wins), scope
    med = statistics.median(wins)
    return (max(wins) - min(wins)) / med if med else None, len(wins), scope


def fmt(rec):
    d = rec.get("decode") or {}
    q = rec.get("quality") or {}
    k = rec.get("korean") or {}
    p = {int(x["ctx"]): x for x in rec.get("prefill", [])}
    w32 = (p.get(32000) or {}).get("warm_tok_s") or 0
    return (f"{rec.get('name','?'):<14}{d.get('windows_med') or 0:7.2f}{d.get('tokens_per_step') or 0:8.2f}"
            f"{100 * (d.get('acc_raw') or 0):7.1f}%  {q.get('ok','?')}/{q.get('total','?'):<4}"
            f"{k.get('dirty','?')}/{k.get('n','?'):<4}{w32:7.0f}  {rec.get('proof_ok') or '-':<6}"
            f"{'cold' if rec.get('cold_compile') else '':<5}{rec.get('t','')[5:16]}")


def judge(cand, base, rows, objective=None):
    from measurement_contract import objective as normalize_objective, metric_value, metric_compatible
    objective = normalize_objective(objective)
    out = {"cand": cand.get("name"), "base": base.get("name") if base else None,
           "build": build_of(cand), "harness": cand.get("harness"), "session": cand.get("session"),
           "status": "incomplete"}
    gates = record_errors(cand)
    proof = cand.get("proof_ok")
    out["gates"] = gates
    out["proof_ok"] = proof
    out['objective'] = objective
    scope = comparison_scope()
    if scope is not None:
        out['baseline_scope'] = scope
        out['runtime_attestation'] = ('recorded' if cand.get('runtime') is not None
                                      else 'not recorded; launch identity requires external evidence')
    if unproved(cand):
        out.update(status="invalid", verdict=f"UNPROVED (proof {proof or 'none'}): the delta is not evidence")
        return out
    if gates:
        out.update(status="invalid", verdict="GATE FAIL: " + ", ".join(gates))
        return out
    if base is None:
        out["verdict"] = "no baseline on this build"
        return out
    if unproved(base):
        out.update(status="invalid", verdict="UNPROVED BASELINE: active knob proof is missing or false")
        return out
    if (not compatible(base, cand) or not comparison_baseline(base, cand, scope=scope) or record_errors(base)
            or not metric_compatible(base, cand, objective)):
        out["verdict"] = "baseline is incompatible or failed its gates"
        return out
    if objective['metric'] == 'quality':
        out.update(status='valid', decision='quality_pass', verdict='all declared onepass quality/proof gates passed')
        return out
    cw, bw = metric_value(cand, objective), metric_value(base, objective)
    if cw is None or bw is None:
        out['verdict'] = 'missing finite objective measurement'
        return out
    out['candidate_value'], out['baseline_value'] = cw, bw
    out["delta"] = (1 - cw / bw) if objective['metric'] == 'prefill_ttft' else (cw / bw - 1)
    fl, n, scope = floor_of(rows, cand, objective)
    out["floor"], out["floor_n"], out["floor_scope"] = fl, n, scope
    if out["delta"] is None:
        out["verdict"] = "no decode window"
    elif fl is None or (objective['metric'] != 'decode_steps' and n < 3):
        out["verdict"] = f"{100 * out['delta']:+.1f}% with no floor yet (n={n}): one more baseline sample"
    elif abs(out["delta"]) <= fl:
        out["status"] = "inconclusive"
        out["verdict"] = f"{100 * out['delta']:+.1f}% WITHIN the floor ±{100 * fl:.1f}% (n={n}, {scope}): one more sample"
    else:
        out["status"] = "valid" if scope == "same build" else "inconclusive"
        out["verdict"] = f"{100 * out['delta']:+.1f}% BEYOND the floor ±{100 * fl:.1f}% (n={n}, {scope})"
    out['decision'] = ('improved' if out['delta'] > 0 else 'regressed') if out['status'] == 'valid' else 'unresolved'
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--base", default=None)
    ap.add_argument("--jsonl", default=JSONL)
    ap.add_argument("--write", action="store_true", help="append the verdict to verdicts.jsonl")
    ap.add_argument("--allow-rehearsal", action="store_true")
    ap.add_argument("--fail-invalid", action="store_true", help="stop dependent arms after a gate/proof failure")
    a = ap.parse_args()
    try:
        comparison_scope()
    except ValueError as exc:
        ap.error(str(exc))
    rows = load(a.jsonl)
    if a.allow_rehearsal:
        for r in rows:
            r.pop("rehearsal", None)
    cand = last(rows, a.cand)
    if not cand:
        print(f"judge> no record named {a.cand}")
        return 2
    if a.base:
        base = last(rows, a.base)
    else:
        bases, _ = baselines_on(rows, cand)
        base = bases[-1] if bases else None
    print(f"judge> {'arm':<14}{'step/s':>7}{'tok/st':>8}{'acc':>8}  {'qual':<5}{'ko':<5}{'32Kw':>7}  {'proof':<6}{'':<5}when")
    print("judge> " + fmt(cand))
    if base:
        print("judge> " + fmt(base))
    v = judge(cand, base, rows, json.loads(os.environ['FLEET_OBJECTIVE']) if os.environ.get('FLEET_OBJECTIVE') else None)
    print(f"judge> verdict: {v['verdict']}")
    if a.write:
        v["t"] = time.strftime("%F %T")
        os.makedirs(os.path.dirname(VERDICTS), exist_ok=True)
        with open(VERDICTS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
        print(f"judge> written to {VERDICTS}")
        if os.environ.get("FLEET_EXPERIMENT_ID") and os.environ.get("FLEET_EXPERIMENT_ROOT"):
            from experiments import Store
            store = Store(os.environ["FLEET_EXPERIMENT_ROOT"])
            with store.db:
                store.event(os.environ["FLEET_EXPERIMENT_ID"], "arm_result", v)
    return 4 if a.fail_invalid and v["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
