#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Judge one ST commit against another from onepass records: warm against warm, cold beside it.

    python3 bench/st_judge.py samples --sha <sha> [--allow-rehearsal]           # warm samples of that sha: a count
    python3 bench/st_judge.py boots --sha <sha> [--allow-rehearsal]             # ... and the boot each came from
    python3 bench/st_judge.py judge --cand <sha> --base <sha> [--write] [--allow-rehearsal]

D17 (CHARTER, PR #742) says a change that claims speed is not finished until the fleet has
measured it, and 45차 §93 says how: two onepass runs on ONE boot, the record carrying the served
shape, a bracket that hit the prefix cache excluded. bench/st_bracket.sh produces exactly that --
run 1 is the cold column (TTFT, the compile tail), run 2 the warm column (decode step/s, warm
prefill) -- and this judges the warm columns of two commits against each other, with the base's
own run-to-run spread as the floor when it has more than one boot. bench/judge.py stays the vLLM
bracket's judge: it matches records by overlay stamp and lane proofs, neither of which an ST
record has. Here the identity of a record is the commit the bracket named (`arm_sha`), or the
release the launcher stamped into the container (`release`), twelve characters of either.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time

JSONL = os.environ.get("ONEPASS_JSONL", os.path.expanduser("~/glm53-logs/bracket-onepass.jsonl"))
VERDICTS = os.environ.get("ONEPASS_VERDICTS", os.path.join(os.path.dirname(JSONL), "verdicts.jsonl"))
HEX = re.compile(r"[0-9a-f]{7,40}")


def load(path=JSONL):
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []


def identity(rec) -> str:
    """Twelve characters of the commit this record measured, or ''."""
    for key in ("arm_sha", "release"):
        value = str(rec.get(key) or "")
        if HEX.fullmatch(value):
            return value[:12]
    return ""


def same(sha, rec, tree=None) -> bool:
    """The record measured this commit -- or the same engine tree under another commit (`tree`)."""
    if tree and rec.get("arm_tree") and str(rec["arm_tree"])[:12] == str(tree)[:12]:
        return True
    mine = identity(rec)
    return bool(mine) and (sha[:12] == mine or mine.startswith(sha[:12]) or sha[:12].startswith(mine))


def errors(rec) -> list:
    """Why this record cannot be evidence (empty when it can)."""
    out = list(rec.get("evidence_issues") or [])
    if not full_evidence(rec):
        out.append('screen only; full validation pending')
    if rec.get("engine") != "st":
        out.append("not an ST record")
    q, k, d = (rec.get(key) or {} for key in ("quality", "korean", "decode"))
    if not isinstance(q.get("total"), int) or q["total"] <= 0 or q.get("ok") != q["total"]:
        out.append(f"quality {q.get('ok')}/{q.get('total')}")
    if not isinstance(k.get("n"), int) or k["n"] <= 0 or k.get("dirty") != 0:
        out.append(f"korean {k.get('dirty')}/{k.get('n')}")
    value = d.get("windows_med")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        out.append("no finite decode window")
    if (rec.get("traffic") or {}).get("issues"):
        out.append("traffic during the run")
    return out


def full_evidence(rec) -> bool:
    """Legacy onepass rows were full; explicitly scoped screening can never become a baseline."""
    return rec.get('evidence_scope', 'full') == 'full' and rec.get('adoption_eligible') is not False


def warm(rec) -> bool:
    """The judged column. Run 1 of a bracket is the cold one (a boot's compile tail); a record that
    names no run, or whose run 1 followed a prefix reset on a live door (cold=reset: a D17 probe),
    is warm by construction -- the engine had been serving."""
    return rec.get("run_index") != 1 or rec.get("cold") == "reset"


def samples(rows, sha, *, allow_rehearsal=False, tree=None):
    """The warm, valid records of `sha` -- or of its engine tree under another commit -- one per boot
    (two runs on one boot are one sample). This is where an adopted candidate's measurement becomes
    the next baseline: the deployed commit's tree is the candidate's, so its records are the base's."""
    picked = {}
    for index, rec in enumerate(rows):
        if not same(sha, rec, tree) or not warm(rec) or errors(rec):
            continue
        if rec.get("rehearsal") and not allow_rehearsal:
            continue
        picked[rec.get("boot_id") or rec.get("run_id") or ("row", index)] = rec
    return list(picked.values())


def colds(rows, sha, *, allow_rehearsal=False, tree=None):
    """The cold column is a BOOT's run 1 (TTFT with the compile tail). A probe on the live door
    marks its run 1 cold=reset -- after a prefix reset, not a boot -- and stays out of it."""
    return [rec for rec in rows if same(sha, rec, tree) and rec.get("run_index") == 1
            and full_evidence(rec)
            and rec.get("cold", "boot") == "boot" and (allow_rehearsal or not rec.get("rehearsal"))]


def prefill(rec, ctx, key):
    for row in rec.get("prefill") or []:
        if int(row.get("ctx") or 0) == ctx:
            return row.get(key)
    return None


def med(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return statistics.median(values) if values else None


def summary(warm_recs, cold_recs) -> dict:
    d = [r.get("decode") or {} for r in warm_recs]
    steps = med([x.get("windows_med") for x in d])
    tps = med([x.get("tokens_per_step") for x in d])
    return {
        "n": len(warm_recs),
        "decode_steps_s": steps,
        "tokens_per_step": tps,
        "decode_tok_s": (steps * tps) if steps and tps else None,
        "warm_prefill_2k_tok_s": med([prefill(r, 2000, "warm_tok_s") for r in warm_recs]),
        "warm_prefill_32k_tok_s": med([prefill(r, 32000, "warm_tok_s") for r in warm_recs]),
        "cold_ttft_2k_s": med([prefill(r, 2000, "cold_s") for r in cold_recs]),
        "cold_ttft_32k_s": med([prefill(r, 32000, "cold_s") for r in cold_recs]),
        "quality": [f"{(r.get('quality') or {}).get('ok')}/{(r.get('quality') or {}).get('total')}" for r in warm_recs],
        "korean": [f"{(r.get('korean') or {}).get('dirty')}/{(r.get('korean') or {}).get('n')}" for r in warm_recs],
        "names": [r.get("name") for r in warm_recs],
        "shape": (warm_recs[-1].get("engine_shape") if warm_recs else None),
    }


def floor(warm_recs):
    """(relative spread, n) of the base's decode step/s over its boots; None with fewer than two."""
    values = [(r.get("decode") or {}).get("windows_med") for r in warm_recs]
    values = [v for v in values if isinstance(v, (int, float)) and v > 0]
    if len(values) < 2:
        return None, len(values)
    m = statistics.median(values)
    return (max(values) - min(values)) / m if m else None, len(values)


def pooled_floor(rows, *, exclude=(), allow_rehearsal=False):
    """(relative spread, shas) -- the median run-to-run spread over every commit that has two or more
    boots in the records, for a base that has only one. Booting the base again just to learn what
    noise is costs a boot and the fleet; the noise of this engine on these four boxes is a property
    of the boxes far more than of the commit, and the records already hold it."""
    spreads = []
    seen = set()
    for rec in rows:
        sha = identity(rec)
        if not sha or sha in seen or any(same(x, rec) for x in exclude):
            continue
        seen.add(sha)
        spread, n = floor(samples(rows, sha, allow_rehearsal=allow_rehearsal))
        if spread is not None:
            spreads.append(spread)
    if not spreads:
        return None, 0
    return statistics.median(spreads), len(spreads)


def judge(rows, cand, base, *, allow_rehearsal=False, cand_tree=None, base_tree=None) -> dict:
    cw = samples(rows, cand, allow_rehearsal=allow_rehearsal, tree=cand_tree)
    bw = samples(rows, base, allow_rehearsal=allow_rehearsal, tree=base_tree)
    cc = colds(rows, cand, allow_rehearsal=allow_rehearsal, tree=cand_tree)
    bc = colds(rows, base, allow_rehearsal=allow_rehearsal, tree=base_tree)
    cs, bs = summary(cw, cc), summary(bw, bc)
    invalid = [f"{r.get('name')}: {', '.join(errors(r))}" for r in rows if same(cand, r, cand_tree) and warm(r) and errors(r)
               and (allow_rehearsal or not r.get("rehearsal"))]
    out = dict(t=time.strftime("%F %T"), engine="st", cand=cand[:12], base=base[:12], cand_summary=cs, base_summary=bs,
               invalid_candidates=invalid, cand_tree=cand_tree, base_tree=base_tree)
    if not cw:
        out["verdict"] = "NO EVIDENCE: the candidate has no valid warm sample" + (" (its records failed gates)" if invalid else "")
        return out
    if not bw:
        out["verdict"] = "NO BASE: the base has no valid warm sample -- boot it (st-pair does when none exists)"
        return out
    delta = (cs["decode_steps_s"] - bs["decode_steps_s"]) / bs["decode_steps_s"] * 100
    spread, n = floor(bw)
    source, where = "base", f"n={n}"
    if spread is None:
        spread, k = pooled_floor(rows, exclude=(cand, base), allow_rehearsal=allow_rehearsal)
        source, where = ("pooled", f"pooled from {k} commits' boots; the base has {n}") if spread is not None else (None, "")
    out.update(delta_pct=delta, floor_pct=(spread * 100) if spread is not None else None, n_base=n, n_cand=len(cw),
               floor_source=source)
    if spread is None:
        out["verdict"] = f"{delta:+.1f}% decode step/s, NO FLOOR (the base has {n} boot and no commit in the records has two)"
    elif abs(delta) > spread * 100:
        out["verdict"] = f"{delta:+.1f}% decode step/s, BEYOND the {source} floor ±{spread * 100:.1f}% ({where})"
    else:
        out["verdict"] = f"{delta:+.1f}% decode step/s, WITHIN the {source} floor ±{spread * 100:.1f}% ({where})"
    return out


def table(out) -> str:
    cs, bs = out["cand_summary"], out["base_summary"]
    def cell(value, fmt):
        return (fmt % value) if isinstance(value, (int, float)) else "-"
    rows = [("decode step/s", "%.2f", "decode_steps_s"), ("tokens/step", "%.3f", "tokens_per_step"),
            ("decode tok/s", "%.1f", "decode_tok_s"), ("warm prefill 2K tok/s", "%.0f", "warm_prefill_2k_tok_s"),
            ("warm prefill 32K tok/s", "%.0f", "warm_prefill_32k_tok_s"), ("cold TTFT 2K s", "%.2f", "cold_ttft_2k_s"),
            ("cold TTFT 32K s", "%.2f", "cold_ttft_32k_s")]
    lines = [f"{'':<24}{'cand ' + out['cand']:>18}{'base ' + out['base']:>18}",
             f"{'boots (warm samples)':<24}{cs['n']:>18}{bs['n']:>18}"]
    for label, fmt, key in rows:
        lines.append(f"{label:<24}{cell(cs.get(key), fmt):>18}{cell(bs.get(key), fmt):>18}")
    lines.append(f"{'quality':<24}{' '.join(cs['quality']) or '-':>18}{' '.join(bs['quality']) or '-':>18}")
    lines.append(f"{'korean dirty':<24}{' '.join(cs['korean']) or '-':>18}{' '.join(bs['korean']) or '-':>18}")
    if cs.get("shape") and bs.get("shape") and cs["shape"] != bs["shape"]:
        lines.append("SHAPE DIFFERS between the two engines: " + json.dumps(cs["shape"], sort_keys=True)
                     + " vs " + json.dumps(bs["shape"], sort_keys=True) + " -- the numbers are not comparable (§93)")
    for line in out.get("invalid_candidates") or []:
        lines.append("invalid: " + line)
    lines.append("verdict: " + out["verdict"])
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("samples", "boots", "judge"))
    ap.add_argument("--sha", default="")
    ap.add_argument("--tree", default="", help="the engine/ tree of --sha: records of the same tree under another commit count too")
    ap.add_argument("--cand", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--cand-tree", default="")
    ap.add_argument("--base-tree", default="")
    ap.add_argument("--write", action="store_true", help="append the verdict to verdicts.jsonl")
    ap.add_argument("--allow-rehearsal", action="store_true")
    ap.add_argument("--jsonl", default=JSONL)
    a = ap.parse_args(argv)
    rows = load(a.jsonl)
    if a.action in ("samples", "boots"):
        if not HEX.fullmatch(a.sha):
            ap.error("--sha must be a commit id")
        picked = samples(rows, a.sha, allow_rehearsal=a.allow_rehearsal, tree=a.tree or None)
        if a.action == "samples":
            print(len(picked))
        else:                                        # one line per sample: the boot it came from (deploy-watch asks)
            for rec in picked:
                print(rec.get("boot_id") or rec.get("run_id") or "?")
        return 0
    if not (HEX.fullmatch(a.cand) and HEX.fullmatch(a.base)):
        ap.error("--cand and --base must be commit ids")
    out = judge(rows, a.cand, a.base, allow_rehearsal=a.allow_rehearsal, cand_tree=a.cand_tree or None, base_tree=a.base_tree or None)
    print(table(out))
    if a.write:
        path = os.environ.get("ONEPASS_VERDICTS", os.path.join(os.path.dirname(a.jsonl), "verdicts.jsonl"))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"verdict written to {path}")
    return 0 if "NO EVIDENCE" not in out["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
