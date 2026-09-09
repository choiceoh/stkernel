#!/usr/bin/env python3
"""Offline diagnostic of two independent SF6 unpack reservations, never an A/B verdict.

Usage: python3 analyze_baseline.py CANDIDATE_FINAL BASELINE_FINAL
Only input files are read. A successful exit means the diagnostic's evidence
checks passed; it does not mean output quality passed or authorize adoption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "probes"))
import analyze_decode_next_onepass as audit

proof = audit.runtime_proof
REVISIONS = dict(candidate="c85a1e75a7187ad69baefb30ef143f92f13430e8",
                 baseline="c03da265a893c61e759eb56e2e43f77bac74a2cc")
VARIANT = dict(sf6_direct=False, sf6_unpack=True)
BASELINE_SCHEMA = "sf6-baseline-observation-v1"
QUALITY_ERRORS = {"quality must pass 18/18", "Korean corruption/coverage gate failed"}
require, close, number = audit.require, audit.close, audit.number


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local_path(root, relative):
    path = root / relative
    require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root.resolve()),
            "artifact path escapes evidence directory")
    return path


def raw_timing(record):
    """Recompute timing independently: validate_record stops at failed quality."""
    decode, requests = record["decode"], record["requests"]
    require(decode["primary"] == "fixed-2K" and decode["num_spec"] == 5,
            "fixed-2K SPEC_K=5 primary required")
    fixed = requests[-3:]
    require(len(requests) == 8 and all(q.get("fixed_decode") is True and q["rep"] == i
            and q["completion_tokens"] == q["min_tokens"] == q["max_tokens"] == 2048
            and q["seed"] == 7 + i and q["finish_reason"] == "length"
            and number(q["decode_s"]) for i, q in enumerate(fixed)), "three complete fixed requests required")
    windows = decode["fixed_intervals"]
    require(len(windows) >= 20 and {w["request"] for w in windows} == {0, 1, 2},
            "all three fixed requests need decode windows")
    previous_end, previous_request = -math.inf, -1
    for window in windows:
        require(type(window["request"]) is int and window["request"] >= previous_request
                and number(window["steps"], zero=True) and number(window["seconds"])
                and number(window["start"], zero=True) and number(window["end"])
                and window["start"] >= previous_end
                and close(window["end"] - window["start"], window["seconds"]),
                "unordered, overlapping or invalid decode windows")
        previous_end, previous_request = window["end"], window["request"]
    require(all(sum(w["seconds"] for w in windows if w["request"] == i) <= q["decode_s"]
                for i, q in enumerate(fixed)), "windows exceed request decode time")
    steps, seconds = sum(w["steps"] for w in windows), sum(w["seconds"] for w in windows)
    rates = [w["steps"] / w["seconds"] for w in windows]
    pooled = steps / seconds
    require(number(pooled) and close(pooled, decode["fixed_pooled_step_s"]), "pooled step arithmetic differs")
    require(len(rates) == len(decode["windows"]) and all(close(a, b) for a, b in zip(rates, decode["windows"]))
            and close(median(rates), decode["windows_med"]), "window median arithmetic differs")
    return dict(steps=steps, seconds=seconds, windows=len(windows), fixed_pooled_step_s=pooled,
                ms_per_step=1000 / pooled, windows_med=median(rates),
                output_tok_s=sum(q["completion_tokens"] - 1 for q in fixed) / sum(q["decode_s"] for q in fixed))


def one_record(root):
    raw = (root / "records.raw.jsonl").read_bytes()
    lines = [line for line in raw.splitlines(keepends=True) if line.strip()]
    require(len(lines) == 1 and lines[0].endswith(b"\n"), "exactly one complete named record required")
    line = lines[0].rstrip(b"\r\n")
    record = audit.transport_evidence._record(line)
    require(isinstance(record.get("name"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", record["name"]),
            "safe record name required")
    return record, line


def phase(root, directory, receipt, mode, name, revision, record, line, prefix):
    """Verify sealed raw bytes before accepting any reported marker or identity."""
    require(receipt.get("phase") == prefix and receipt.get("mode") == mode
            and receipt.get("source_commit") == revision
            and all(receipt.get(k) is v for k, v in VARIANT.items()), "phase source/variant identity differs")
    before, after = receipt["head_before"], receipt["head_after"]
    require(before == after and after.get("boot_id") == record["boot_id"]
            and after.get("image") == proof.IMAGE and after.get("running") is True,
            "phase head boot/image changed or differs from record")
    if prefix == "prepared":
        require(receipt.get("owned_before") is True and receipt.get("owned_after") is True
                and receipt.get("record_present_before") is False and receipt.get("record_present_after") is False,
                "prepared receipt did not precede the completed record under owned hold")
    else:
        require(receipt.get("record_present_before") is True and receipt.get("record_present_after") is True
                and receipt.get("record_sha256") == sha(line), "runtime receipt record SHA differs")
    if mode == "candidate":
        require(receipt.get("schema") == 1 and receipt.get("arm") == name
                and receipt.get("status") == "PASS" and not receipt.get("errors"), "candidate phase failed")
        stems = {host: prefix + "-" + name + "-" + host for host in audit.HOSTS}
    else:
        require(receipt.get("schema") == BASELINE_SCHEMA and receipt.get("name") == name
                and receipt.get("collection_status") == "COMPLETE" and not receipt.get("collection_errors"),
                "baseline phase collection failed")
        stems = {host: host for host in audit.HOSTS}
    required = {stem + suffix for stem in stems.values() for suffix in (".json", ".log")}
    if mode == "baseline" and prefix == "runtime":
        required.add("record.raw.json")
    hashes = receipt.get("artifacts_sha256", {})
    require(isinstance(hashes, dict) and required <= set(hashes), "phase artifact seal is incomplete")
    audit.sealed(directory, receipt, hashes)
    if mode == "baseline" and prefix == "runtime":
        require((directory / "record.raw.json").read_bytes() == line, "sealed baseline record differs from ledger")
    return {host: (directory / (stem + ".json"), directory / (stem + ".log")) for host, stem in stems.items()}


def runtime(root, mode, record, line):
    name, revision = record["name"], REVISIONS[mode]
    source_file = "source.commit" if mode == "candidate" else "baseline-source.commit"
    expected_file = "expected-" + name + ".json" if mode == "candidate" else "baseline-expected.json"
    require((root / source_file).read_text().strip() == revision, "unexpected source revision")
    require(isinstance(record.get("git"), str) and re.fullmatch(r"[0-9a-f]{7,40}", record["git"])
            and revision.startswith(record["git"]), "record/source revision differs")
    expected = proof.validate_manifest(audit.read_json(root / expected_file))
    require(len(expected) == 63, "complete 63-file serving manifest required")
    observer = audit.read_json(root / ("observer.json" if mode == "candidate" else "baseline-observer.json"))
    require(observer.get("source_commit") == revision
            and all(observer.get(k) is v for k, v in VARIANT.items()), "observer source/variant differs")
    if mode == "candidate":
        require(observer.get("schema") == 1 and observer.get("candidate") == name, "candidate observer identity differs")
        arm = observer["arms"][name]
        require(arm.get("status") == "PASS" and arm.get("mode") == mode and not arm.get("errors")
                and arm.get("head_boot_id") == record["boot_id"] and arm.get("record_sha256") == sha(line),
                "candidate arm snapshot/record binding failed")
    else:
        require(observer.get("schema") == BASELINE_SCHEMA and observer.get("name") == name
                and observer.get("mode") == mode and observer.get("session") == record.get("session")
                and observer.get("collection_status") == "COMPLETE" and not observer.get("errors"),
                "baseline observer collection/identity failed")
    reports, receipts = {}, {}
    for prefix, field in (("prepared", "before_receipt"), ("runtime", "after_receipt")):
        if mode == "candidate":
            receipt_name = "observed-" + prefix + "-" + name + ".json"
            require(arm.get(field) == receipt_name, "candidate phase filename differs")
            directory, receipt = root, audit.read_json(root / receipt_name)
        else:
            entry = observer["phases"][prefix]
            directory = local_path(root, entry["path"])
            raw_receipt = (directory / "receipt.json").read_bytes()
            require(entry.get("collection_status") == "COMPLETE" and entry.get("receipt_sha256") == sha(raw_receipt),
                    "baseline phase receipt seal differs")
            receipt = audit.transport_evidence._record(raw_receipt)
        paths = phase(root, directory, receipt, mode, name, revision, record, line, prefix)
        receipts[prefix], reports[prefix] = receipt, {}
        for host, (report_path, log_path) in paths.items():
            report, raw_log = audit.read_json(report_path), log_path.read_bytes()
            require(report.get("log_sha256") == sha(raw_log) and report.get("markers") == proof.parse_markers(
                raw_log.decode(errors="replace"), sf6_unpack=True), host + ": log SHA/parser differs")
            require(report.get("host") == host and report.get("mode") == mode, "rank/arm identity differs")
            issues = proof.validate_report(report, expected, **VARIANT)
            require(not issues, host + ": " + "; ".join(issues))
            mm = json.loads(proof.cli_option(report["serving_argv"], "--limit-mm-per-prompt"))
            require(mm == {"image": 4, "video": 1}, host + ": actual multimedia CLI differs from candidate")
            if host == "srv2":
                require(all(report[k] == receipt["head_after"][k] for k in ("boot_id", "image", "knobs")),
                        "head proof differs from phase receipt")
            reports[prefix][host] = report
    require(receipts["prepared"]["head_after"] == receipts["runtime"]["head_before"], "head changed between phases")
    for host in audit.HOSTS:
        issues = proof.compare_snapshots(reports["prepared"][host], reports["runtime"][host], **VARIANT)
        require(not issues, host + ": " + "; ".join(issues))
    if mode == "baseline":
        require(observer.get("runtime_validation") == "PASS" and all(
            receipt.get("runtime_validation") == "PASS" and not receipt.get("validation_errors")
            for receipt in receipts.values()), "retained baseline runtime validation failed")
    return dict(expected=expected, reports=reports["runtime"], source_commit=revision)


def summarize(candidate_dir, baseline_dir):
    result = dict(schema="sf6-unpack-independent-baseline-diagnostic-v1", valid=False, comparison=None,
        raw_speed_delta=None, errors=[], arms={},
        limitations=["Different reservations and source commits; this is not a valid A/B comparison.",
                     "Candidate output quality failed. Baseline quality is reported independently, even if its supervisor exited 0.",
                     "No adoption or causal performance conclusion. Cold-prefill gains are not calculated."])

    def check(stage, action):
        try:
            return action()
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, ZeroDivisionError) as exc:
            result["errors"].append(dict(stage=stage, error=str(exc)))

    runtimes, records = {}, {}
    for mode, directory in (("candidate", candidate_dir), ("baseline", baseline_dir)):
        root = Path(directory)
        entry = check(mode + " record", lambda: one_record(root))
        if entry is None:
            continue
        record, line = entry
        records[mode] = record
        arm = result["arms"][mode] = dict(name=record["name"], record_sha256=sha(line),
            source_commit=REVISIONS[mode], record_git=record.get("git"), session=record.get("session"),
            boot_id=record.get("boot_id"), cold_compile=record.get("cold_compile", False),
            quality=record.get("quality"), korean=record.get("korean"), raw_decode=record.get("decode"),
            raw_prefill=record.get("prefill"), record_validation_errors=[])
        campaign_log = root / "campaign.log"
        if campaign_log.exists():
            raw_log = campaign_log.read_bytes()
            lines = raw_log.decode(errors="replace").splitlines()
            selected = {j for i, text in enumerate(lines) if "cjk_mixed" in text or "Halvorsen博士" in text
                        for j in (i, i + 1) if j < len(lines)}
            arm["campaign_log_sha256"] = sha(raw_log)
            arm["korean_scanner_excerpt"] = [dict(line=i + 1, text=lines[i]) for i in sorted(selected)]
        try:
            audit.validate_record(record, canonical=True)
        except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as exc:
            arm["record_validation_errors"].append(str(exc))
            if str(exc) not in QUALITY_ERRORS:
                result["errors"].append(dict(stage=mode + " onepass", error=str(exc)))
        arm["raw_timing"] = check(mode + " timing", lambda: raw_timing(record))
        arm["memory"] = check(mode + " memory", lambda: audit.validate_memory(audit.read_jsonl(root / (record["name"] + ".memory.jsonl"))))
        arm["campaign_returncode"] = check(mode + " supervisor receipt", lambda: int((root / "campaign.exit").read_text().strip()))
        runtimes[mode] = check(mode + " runtime", lambda: runtime(root, mode, record, line))
        if runtimes[mode]:
            arm["mhc_runtime"] = {host: proof.mhc_runtime_state(report["markers"])
                                  for host, report in runtimes[mode]["reports"].items()}
    if set(records) == {"candidate", "baseline"}:
        a, b = records["candidate"], records["baseline"]
        identity = ("request_sha256", "prompt_tokens", "seed", "min_tokens", "max_tokens")
        check("ordered requests", lambda: require(
            [[q[k] for k in identity] for q in a["requests"]] == [[q[k] for k in identity] for q in b["requests"]],
            "ordered request identities differ"))
        def output_hashes():
            require(len(a["requests"]) == len(b["requests"]) == 8, "eight output hashes per arm required")
            pairs = [dict(index=i, candidate=q["output_sha256"], baseline=r["output_sha256"],
                          equal=q["output_sha256"] == r["output_sha256"])
                     for i, (q, r) in enumerate(zip(a["requests"], b["requests"]))]
            require(all(audit.SHA.fullmatch(row[key]) for row in pairs for key in ("candidate", "baseline")),
                    "invalid request output hash")
            return dict(equal_count=sum(row["equal"] for row in pairs), total=8,
                        fixed2k_rep2_same=pairs[-1]["equal"], per_request=pairs)
        result["output_hash_matches"] = check("output hashes", output_hashes)
        check("workload/source stamp", lambda: require(a.get("workload") == b.get("workload")
            and isinstance(a.get("overlay"), str) and re.fullmatch(r"[0-9a-f]{12}", a["overlay"])
            and a["overlay"] == b.get("overlay"), "workload or served overlay stamp differs"))
        check("independent reservations", lambda: require(isinstance(a.get("session"), str) and a["session"]
            and isinstance(b.get("session"), str) and b["session"] and a["session"] != b["session"],
            "distinct named reservations required"))
    if all(runtimes.get(mode) for mode in ("candidate", "baseline")):
        a, b = runtimes["candidate"], runtimes["baseline"]
        check("source equivalence", lambda: require(a["expected"] == b["expected"], "63 serving source hashes differ"))
        check("across-arm runtime", lambda: require(not (issues := proof.compare_arms(
            b["reports"], a["reports"], **VARIANT)), "; ".join(issues)))
    if all(result["arms"].get(mode, {}).get("raw_timing") for mode in ("candidate", "baseline")):
        a, b = (result["arms"][mode]["raw_timing"] for mode in ("candidate", "baseline"))
        result["raw_speed_delta"] = dict(primary="fixed_pooled_step_s", candidate_step_s=a["fixed_pooled_step_s"],
            baseline_step_s=b["fixed_pooled_step_s"],
            step_s_change_pct=100 * (a["fixed_pooled_step_s"] / b["fixed_pooled_step_s"] - 1),
            candidate_ms_per_step=a["ms_per_step"], baseline_ms_per_step=b["ms_per_step"],
            saved_ms_per_step=b["ms_per_step"] - a["ms_per_step"],
            ms_per_step_reduction_pct=100 * (1 - a["ms_per_step"] / b["ms_per_step"]),
            evidence_checks_passed=not result["errors"], interpretation="Raw arithmetic only; quality failures and runtime differences remain applicable.")
    result["diagnostic_checks_passed"] = not result["errors"]
    result["status"] = "DIAGNOSTIC" if not result["errors"] else "DIAGNOSTIC_WITH_ERRORS"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_final", type=Path)
    parser.add_argument("baseline_final", type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.candidate_final, args.baseline_final)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(not result["diagnostic_checks_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
