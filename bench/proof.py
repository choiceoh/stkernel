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
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS = os.path.join(HERE, "proof-markers.tsv")
HEAD_LOG = os.environ.get("MK_HEAD_LOG", "/home/choiceoh/glm53-logs/glm53.log")


def _json_marker_receipts(log: str, prefix: str) -> list[dict] | None:
    """Decode complete literal JSON records; duplicates/nonfinite values fail."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")

    def finite_float(value):
        number = float(value)
        if not -float("inf") < number < float("inf"):
            raise ValueError("nonfinite JSON number")
        return number

    records = []
    for line in log.splitlines():
        if prefix not in line:
            continue
        try:
            record = json.loads(line.split(prefix, 1)[1].strip(),
                                object_pairs_hook=pairs, parse_constant=invalid_constant,
                                parse_float=finite_float)
            if not isinstance(record, dict):
                return None
        except (ValueError, TypeError):
            return None
        records.append(record)
    return records


def _ep_tiled_case_proof(case):
    if (not isinstance(case, dict) or case.get("verdict") != "PASS"
            or case.get("phase") != "complete"
            or any(key in case for key in ("error", "cleanup_error", "diagnostic_error", "first_failure_rows"))):
        return False
    graph = case.get("case") in {"mixed6", "balanced12", "concentrated24", "zeros32", "remote33", "balanced8192"}
    if case.get("graph_replay") is not graph:
        return False
    labels = ("C1-eager", "C2-graph-current" if graph else "C2-eager",
              "C3-graph-side" if graph else "C3-side")
    phases = [phase + "-" + label for phase in ("initial", "changed") for label in labels]
    items = case.get("candidate")
    if not isinstance(items, list) or len(items) != 6:
        return False
    metrics = ("max_row_relative_l2", "max_row_relative_abs",
               "stock_max_row_relative_l2", "stock_max_row_relative_abs")
    return all(isinstance(item, dict) and item.get("phase") == phase
               and type(item.get("bad_rows")) is int and item["bad_rows"] == 0
               and all(type(item.get(key)) in (int, float) and item[key] >= 0 for key in metrics)
               and not any(key in item for key in ("error", "cleanup_error", "diagnostic_error"))
               for item, phase in zip(items, phases))


def _startup_proof(knob: str, log: str) -> bool | None:
    """Composite execution evidence; armed or partial progress is insufficient."""
    if knob == "VLLM_GLM53_EP_TILED":
        if "[ep-tiled-selftest] FAIL" in log:
            return False
        for lane, low, high in (("decode", 1, 32), ("prefill", 33, 16384)):
            prefix = "[ep-tiled] LAUNCHED " + lane + " E72/H4096/I2048/top8 T="
            values = [line.split(prefix, 1)[1].strip() for line in log.splitlines()
                      if prefix in line]
            if not values or any(re.fullmatch(r"[0-9]+", value) is None
                                 or not low <= int(value) <= high for value in values):
                return False
        records = _json_marker_receipts(log, "[ep-tiled-selftest] PASS ")
        expected = {"mixed6", "balanced12", "concentrated24", "zeros32", "remote33",
                    "balanced2128", "balanced4096", "concentrated6912", "balanced8192"}
        return bool(records) and all(
            record.get("verdict") == "PASS" and record.get("phase") == "complete"
            and record.get("caller_preserved") is True
            and not any(key in record for key in ("error", "cleanup_error"))
            and isinstance(record.get("cases"), list)
            and len(record["cases"]) == len(expected)
            and all(isinstance(case, dict) and isinstance(case.get("case"), str)
                    for case in record["cases"])
            and {case.get("case") for case in record["cases"] if isinstance(case, dict)} == expected
            and all(_ep_tiled_case_proof(case) for case in record["cases"])
            for record in records)
    if knob == "VLLM_GLM53_STARTUP_TRIM":
        records = _json_marker_receipts(log, "[glm53-startup-trim] ")
        if records is None or len(records) != 1:
            return False
        record = records[0]
        try:
            if (type(record.get("schema")) is not int or record["schema"] != 1
                    or record.get("verdict") != "COMPLETE"
                    or type(record.get("rank")) is not int or record["rank"] != 0
                    or record.get("measurement_errors") != []
                    or any(key in ("error", "failed_stage") or key.endswith("_error") for key in record)):
                return False
            stages = record["stages"]
            if ([item["stage"] for item in stages] !=
                    ["synchronize", "gc_collect", "empty_cache", "malloc_trim"]
                    or any(item["status"] != "COMPLETE"
                           or any(key == "error" or key.endswith("_error") for key in item)
                           for item in stages)
                    or type(stages[1]["collected"]) is not int or stages[1]["collected"] < 0
                    or type(stages[3]["returned"]) is not int or stages[3]["returned"] not in (0, 1)):
                return False
            for phase in ("before", "after"):
                if set(record[phase]) != {"allocated", "reserved", "mem_available", "vm_rss"}:
                    return False
                if any(type(value) is not int or value < 0 for value in record[phase].values()):
                    return False
            started, completed = record["started_at"], record["completed_at"]
            return (type(started) in (int, float) and type(completed) in (int, float)
                    and 0 <= started <= completed < float("inf"))
        except (KeyError, TypeError, ValueError):
            return False
    if knob == "VLLM_GLM53_TP_SF6_Q0":
        prefix = "[tp-sf6-q0] LAUNCHED E288/H4096/I512/top8 T="
        if "[tp-sf6-q0-selftest] FAIL" in log:
            return False
        launches = [line.split(prefix, 1)[1].strip() for line in log.splitlines()
                    if prefix in line]
        if not launches or any(re.fullmatch(r"[0-9]+", value) is None
                               or not 4096 <= int(value) <= 8192 for value in launches):
            return False
        records = _json_marker_receipts(log, "[tp-sf6-q0-selftest] PASS ")
        return bool(records) and all(record.get("verdict") == "PASS"
                                     and record.get("phase") == "complete"
                                     and not any(key in record for key in ("error", "cleanup_error"))
                                     for record in records)
    if knob == "VLLM_B12X_EP_ZERO_WEIGHT_MICRO":
        prefix = "b12x EP zero-weight micro: "
        lines = [line.split(prefix, 1)[1].strip() for line in log.splitlines()
                 if prefix in line]
        if not lines:
            return False
        for line in lines:
            match = re.fullmatch(
                r"([0-9]+) tokens -> ([0-9]+) top-k=8 calls "
                r"\(8 tokens / 64 routed pairs each; padded tail=([01])\)", line)
            if match is None:
                return False
            tokens, calls, padded = map(int, match.groups())
            if not (1 <= tokens <= 80 and calls == (tokens + 7) // 8
                    and padded == int(tokens % 8 != 0)):
                return False
        return True
    if knob == "VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE":
        skipped = ("[glm53-graph-profile] skipped unused estimate rank=0; "
                   "model/MM profile and real graph warmup retained")
        position = log.find(skipped)
        return (position >= 0 and "Profiling CUDA graph memory" not in log
                and re.search(r"Graph capturing finished in [0-9]+ secs, took "
                              r"[0-9]+(?:\.[0-9]+)? GiB", log[position + len(skipped):]) is not None)
    if knob == "VLLM_B12X_EP_WARM_COMPACT":
        prefix = "[b12x EP compact warmup]"
        if any(prefix + " " + state in log for state in ("FAILED", "INCOMPLETE")):
            return False
        lines = [line.split(prefix, 1)[1].strip() for line in log.splitlines()
                 if prefix + " COMPLETE" in line]
        if not lines:
            return False
        for line in lines:
            match = re.fullmatch(
                r"COMPLETE device=cuda:[0-9]+ launch_rows=([0-9]+) specializations=([0-9]+) "
                r"static=([0-9]+) dynamic=([0-9]+) required=([0-9]+) ready=([0-9]+) "
                r"representatives=([0-9]+(?:,[0-9]+)*)", line)
            if match is None:
                return False
            rows, total, static, dynamic, required, ready = map(int, match.groups()[:6])
            representatives = list(map(int, match[7].split(",")))
            if not (rows >= total > 0 and total == static + dynamic == required == ready
                    and len(representatives) == len(set(representatives)) == total
                    and min(representatives) > 0):
                return False
        return True
    return None


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
        startup = _startup_proof(k, log)
        if startup is not None:
            res[k] = startup
            continue
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
    return 0 if all(value is True for value in r["proof"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
