#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Armed is not serving: prove the arm's lanes from the head log.

For every knob the serving container carries at a non-default value (or every
knob you name), look for its serving marker (bench/proof-markers.tsv) in the
head log with a FIXED-STRING search, and say PASS / ---- / no-marker. The
result rides in the onepass record (`proof`, `proof_ok`) so a measurement can
never again be filed without knowing whether its lanes ran.

    python3 bench/proof.py                       # knobs from the container, log = head log
    python3 bench/proof.py --log boot-FUS7.log   # a preserved per-arm log
    python3 bench/proof.py --knobs VLLM_GLM53_KDA_ONEPASS,VLLM_GLM53_MK_SMLP2
    python3 bench/proof.py --json                # machine form

The head log is /glmlogs/glm53.log inside the head container -- the server's
stdout, rewritten by every boot (bench/ab-lever.sh keeps a per-arm copy as
boot-<NAME>.log). Serving markers appear only AFTER traffic, so check after
the leg, not after health.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS = os.path.join(HERE, "proof-markers.tsv")
HEAD_LOG = os.environ.get("MK_HEAD_LOG", "/home/choiceoh/glm53-logs/glm53.log")


def markers(path: str = MARKERS) -> dict[str, tuple[str, str]]:
    """{knob: (marker, src)} -- comments and blank lines skipped."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else parts[1])
    return out


def served_knobs(repo: str | None = None) -> dict[str, str]:
    """The container's non-default VLLM_GLM53_* knobs (same rule as onepass)."""
    sys.path.insert(0, HERE)
    try:
        from onepass import _served_build  # type: ignore
    except Exception:
        return {}
    repo = repo or os.path.dirname(HERE)
    return (_served_build(repo) or {}).get("knobs") or {}


def check(knobs: list[str], log_path: str, table: dict[str, tuple[str, str]] | None = None) -> dict:
    table = table or markers()
    try:
        with open(log_path, "rb") as fh:
            log = fh.read().decode("utf-8", "replace")
    except Exception:
        log = ""
    res = {}
    for k in knobs:
        if k not in table:
            res[k] = None                      # no marker known: cannot judge
            continue
        res[k] = table[k][0] in log            # fixed string, never a regex
    judged = [v for v in res.values() if v is not None]
    return {"proof": res, "proof_ok": f"{sum(judged)}/{len(judged)}",
            "log": log_path, "log_bytes": len(log)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=HEAD_LOG)
    ap.add_argument("--knobs", default=None, help="comma list; default: the container's non-default knobs")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.knobs:
        knobs = [k.strip() for k in a.knobs.split(",") if k.strip()]
    else:
        served = served_knobs()
        knobs = [k for k, v in served.items() if v not in ("0", "", "off")]
    r = check(knobs, a.log)
    if a.json:
        print(json.dumps(r))
        return 0
    if not knobs:
        print("proof: no non-default knob to prove (defaults arm)")
        return 0
    for k, v in r["proof"].items():
        tag = "PASS" if v else ("----" if v is False else "no-marker")
        print(f"  {tag:<9} {k}")
    print(f"  -> {r['proof_ok']} lanes proved serving ({r['log_bytes']} bytes of {r['log']})")
    return 0 if r["proof_ok"].split("/")[0] == r["proof_ok"].split("/")[1] else 1


if __name__ == "__main__":
    raise SystemExit(main())
