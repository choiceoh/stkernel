#!/usr/bin/env python3
"""Portable, read-only audit of the one-boot-per-arm decode-next campaign.

The default checks independent stream and GPU receipts. --canonical audits the
official onepass-only path, which has neither. Both modes retain complete
workload, runtime identity and memory checks without consulting git, Docker,
CUDA, a remote machine or the current source checkout.
--sf6-direct selects the packed-only SF6 variant of the canonical campaign.
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

import ar_consumer_gpu_identity as gpu_identity
import decode_next_runtime_proof as runtime_proof
from decode_transport_gpu_probe import FLAGS, SCHEMA as TRANSPORT_SCHEMA, validate_transport_proof
import reuse_ar_consumer_gpu_evidence as transport_evidence
import run_decode_sf6_gpu as sf6

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from measurement_contract import metadata, workload
from window_metrics import exclusive_errors

SHA = re.compile(r"[0-9a-f]{64}\Z")
HOSTS = ("srv2", "srv1", "srv3", "srv4")
MEMORY_HOSTS = {"local", "choiceoh@10.10.10.1", "choiceoh@10.10.10.3", "choiceoh@10.10.10.4"}
WORKLOAD = workload(dict(fixed_decode_tokens=2048, fixed_decode_reps=3, require_exclusive=True))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return transport_evidence._record(Path(path).read_bytes())


def read_jsonl(path):
    return [transport_evidence._record(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def number(value, *, zero=False):
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0)


def close(a, b):
    return number(a, zero=True) and number(b, zero=True) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def sealed(out, receipt, names):
    hashes = receipt.get("artifacts_sha256", {})
    require(isinstance(hashes, dict), "artifact integrity manifest missing")
    for name in names:
        require(Path(name).name == name, "artifact must be a local filename")
        require(hashes.get(name) == hashlib.sha256((out / name).read_bytes()).hexdigest(),
                "sealed artifact missing or changed: " + name)


def bind_sources(hashes, expected):
    """GPU module basenames are unique in the retained serving manifest."""
    require(isinstance(hashes, dict) and hashes, "GPU source manifest missing")
    for path, digest in hashes.items():
        require(isinstance(digest, str) and SHA.fullmatch(digest), "invalid GPU source digest")
        matches = [value for target, value in expected.items() if Path(target).name == Path(path).name]
        require(matches == [digest], "GPU/serving source mismatch: " + path)


def retained_transport(out, expected, revision):
    receipt = read_json(out / "admission.json")
    require(receipt.get("schema") == TRANSPORT_SCHEMA and receipt.get("status") == "PASS"
            and receipt.get("flags") == FLAGS and receipt.get("image") == runtime_proof.IMAGE,
            "compact/inline GPU admission not PASS: " + str(receipt.get("error", receipt.get("status"))))
    selected = receipt.get("selected_stages", [])
    require(isinstance(selected, list) and len(selected) == len(set(selected))
            and "probe" in selected and set(selected) <= {"probe", "memcheck", "racecheck"},
            "plain distributed transport cohort required")
    sealed(out, receipt, ("source.json", "runtime.json"))
    source, runtime = read_json(out / "source.json"), read_json(out / "runtime.json")
    require(source["revision"] == revision, "transport/campaign revision differs")
    gpu_identity.validate_runtime(runtime)
    # The synthetic delay helper is a probe input, not a serving mount.
    hashes = source["source_sha256"]
    require(set(hashes) == set(transport_evidence.SOURCES), "transport source coverage incomplete")
    bind_sources({name: value for name, value in hashes.items() if name.startswith("overlay/")}, expected)
    wrapper = source["harness_sha256"]["probes/decode_transport_gpu_probe.py"]
    require(isinstance(wrapper, str) and SHA.fullmatch(wrapper), "missing transport wrapper identity")
    completed = receipt.get("completed", [])
    names = {stage + "-rank" + str(rank) for stage in selected for rank in range(4)}
    require(isinstance(completed, list) and len(completed) == len(names)
            and {entry["stage"] for entry in completed} == names, "incomplete/duplicate transport ranks")
    for entry in completed:
        name = entry["stage"]
        rank = int(name[-1])
        sealed(out, receipt, (name + ".json", name + ".container.json", name + ".log"))
        report, container = read_json(out / (name + ".json")), read_json(out / (name + ".container.json"))
        require(entry["node"] == gpu_identity.NODES[rank] and type(report["rank"]) is int
                and report["rank"] == rank, "transport rank/node mismatch: " + name)
        require(report.get("status") == "PASS" and report.get("mode") == "distributed",
                "transport rank did not pass: " + name)
        require((report.get("torch"), report.get("cuda"), report.get("device"))
                == ("2.13.0+cu130", "13.0", "NVIDIA GB10"), "transport GPU runtime differs")
        require(entry["source_sha256"] == report["source_sha256"] == hashes
                and report.get("wrapper_sha256") == wrapper, "transport executed-source mismatch")
        transport_evidence._validate_cases(report, True)
        validate_transport_proof(report)
        state = container["state"]
        require(type(state["ExitCode"]) is int and state["ExitCode"] == 0
                and state["OOMKilled"] is False and container["image"] == runtime_proof.IMAGE,
                "transport container failed or OOM: " + name)
        limit = (24 if "racecheck" in name else 8) * 1024**3
        require(container["memory_limit"] == container["memory_swap_limit"] == limit
                and container["cpus"] == "14-17", "transport isolation changed")
        transport_evidence._validate_sanitizer(name, entry, (out / (name + ".log")).read_text())
    return dict(status="PASS", stages=selected, ranks_per_stage=4, source_commit=revision,
                verification="retained artifacts; current-source admission is performed by the campaign")


def retained_sf6(out, expected, revision):
    receipt = read_json(out / "admission.json")
    require(receipt.get("schema") == sf6.SCHEMA and receipt.get("status") == "PASS"
            and receipt.get("image") == runtime_proof.IMAGE and receipt.get("returncode") == 0
            and not receipt.get("issues"), "SF6 GPU admission not PASS: " + str(receipt.get("issues", receipt.get("status"))))
    sealed(out, receipt, ("source.json", "result.json", "probe.log", "container.json", "samples.json"))
    source = read_json(out / "source.json")
    require(source["revision"] == revision == receipt["source_commit"], "SF6/campaign revision differs")
    bind_sources(source["kernels_sha256"], expected)
    sf6.validate_container(read_json(out / "container.json"), receipt["container_id"], receipt["owner_token"])
    sf6.validate_report(read_json(out / "result.json"), source)
    require("REFORM_SF6_CORRECTNESS_PASS" in (out / "probe.log").read_text(), "SF6 PASS log marker missing")
    samples = read_json(out / "samples.json")
    require(samples and number(samples[0]["available_bytes"]) and samples[0]["available_bytes"] >= 24 * 1024**3
            and all(number(row["available_bytes"]) and row["available_bytes"] >= 12 * 1024**3 for row in samples),
            "SF6 memory guard failed")
    return dict(status="PASS", cases=12, source_commit=revision, verification="retained artifacts")


def validate_memory(rows):
    require(bool(rows), "no serving memory samples")
    previous = -1
    minima = {host: math.inf for host in MEMORY_HOSTS}
    for row in rows:
        require(not row["issues"] and number(row["elapsed_s"], zero=True) and row["elapsed_s"] >= previous,
                "serving memory guard failed or timestamps regressed")
        previous = row["elapsed_s"]
        require(type(row["minimum_kib"]) is int and row["minimum_kib"] >= 10 * 1024**2
                and set(row["nodes"]) == MEMORY_HOSTS, "four-host 10 GiB guard required")
        for host, state in row["nodes"].items():
            require(not state.get("error") and type(state.get("available_kib")) is int
                    and type(state.get("total_kib")) is int
                    and row["minimum_kib"] <= state["available_kib"] <= state["total_kib"],
                    "low/unavailable host memory: " + host)
            minima[host] = min(minima[host], state["available_kib"])
    return dict(samples=len(rows), minimum_available_gib={host: value / 1024**2 for host, value in minima.items()})


def validate_record(record, channels=None, *, canonical=False):
    require(not record.get("evidence_issues"), "onepass evidence issues: " + str(record.get("evidence_issues")))
    require(all(record.get(key) == value for key, value in metadata(WORKLOAD).items()),
            "unchanged Korean/thinking/exclusive fixed 3x2048 workload required")
    require(type(record.get("cold_compile", False)) is bool, "invalid cold_compile state")
    requests = record["requests"]
    wanted_order = [(2000, q, None) for q in range(3)] + [(32000, "all", None), (128000, "all", None)]
    wanted_order += [(2000, "fixed-all", rep) for rep in range(3)]
    require([(q["ctx"], q["question"], q.get("rep")) for q in requests] == wanted_order,
            "complete ordered quality ladder and three fixed requests required")
    if canonical:
        require(channels is None, "canonical mode does not admit injected stream evidence")
    else:
        require(isinstance(channels, list) and len(channels) == len(requests), "stream/request count differs")
    for index, q in enumerate(requests):
        fixed = index >= 5
        limit = 2048 if fixed else (400 if index < 3 else 1200)
        require(q.get("fixed_decode", False) is fixed and q["min_tokens"] == (2048 if fixed else 0)
                and q["max_tokens"] == limit and q["seed"] == (7 + index - 5 if fixed else None),
                "request token limits/seed differ: " + str(index))
        require(type(q["prompt_tokens"]) is int and q["prompt_tokens"] > 0
                and type(q["completion_tokens"]) is int and 1 < q["completion_tokens"] <= limit
                and (not fixed or q["completion_tokens"] == 2048), "partial fixed or empty request")
        require(q["finish_reason"] in ("length", "stop") and (not fixed or q["finish_reason"] == "length"),
                "missing/unexpected stream finish")
        require(number(q["ttft_s"]) and number(q["decode_s"])
                and close(q["ttft_s"] + q["decode_s"], q["elapsed_s"])
                and close(q["decode_tok_s"], (q["completion_tokens"] - 1) / q["decode_s"])
                and close(q["tpot_ms"], 1000 * q["decode_s"] / (q["completion_tokens"] - 1)),
                "request timing arithmetic differs")
        require(all(isinstance(q[key], str) and SHA.fullmatch(q[key]) for key in ("request_sha256", "output_sha256")),
                "request/output digest missing")
        if not canonical:
            channel = channels[index]
            require(channel["request_sha256"] == q["request_sha256"] and channel["output_sha256"] == q["output_sha256"]
                    and channel["timing"] == q and channel["finish_reason"] == q["finish_reason"],
                    "ordered stream request/output hash or timing mismatch: " + str(index))
            parts = channel["channels"]
            require(set(parts) == {"content", "reasoning_content", "reasoning"}
                    and all(isinstance(text, str) for text in parts.values()) and any(parts.values()),
                    "stream channels incomplete")
    traffic = record["traffic"]
    require(traffic["samples"] and not traffic["issues"], "exclusive traffic receipt missing/failed")
    for state in [traffic["before"], *traffic["samples"], traffic["after"]]:
        require(set(state) == {"finished", "running", "waiting"}
                and all(number(value, zero=True) for value in state.values()), "traffic counters unavailable")
    require(not exclusive_errors(traffic["before"], traffic["after"], traffic["samples"], len(requests)),
            "independent exclusive traffic check failed")
    require(record["quality"] == {"ok": 18, "total": 18}, "quality must pass 18/18")
    korean = record["korean"]
    require(korean["dirty"] == 0 and korean["n"] == 8 and korean["hits"] == []
            and {"replacement", "lone_jamo", "cjk_mixed", "control"} <= set(korean["kinds"])
            and all(value == 0 for value in korean["kinds"].values()), "Korean corruption/coverage gate failed")
    decode = record["decode"]
    require(decode["primary"] == "fixed-2K" and decode["num_spec"] == 5, "fixed-2K SPEC_K=5 primary required")
    windows = decode["fixed_intervals"]
    require(len(windows) >= 20 and {w["request"] for w in windows} == {0, 1, 2}, "incomplete fixed decode windows")
    previous_end = -math.inf
    previous_request = -1
    for w in windows:
        require(type(w["request"]) is int and 0 <= w["request"] <= 2 and w["request"] >= previous_request
                and number(w["steps"], zero=True) and number(w["seconds"])
                and number(w["start"], zero=True) and number(w["end"])
                and w["start"] >= previous_end and close(w["end"] - w["start"], w["seconds"]),
                "overlapping, unordered or invalid fixed window")
        previous_end, previous_request = w["end"], w["request"]
    fixed = requests[-3:]
    require(all(sum(w["seconds"] for w in windows if w["request"] == rep) <= q["decode_s"]
                for rep, q in enumerate(fixed)), "windows exceed request decode time")
    rates = [w["steps"] / w["seconds"] for w in windows]
    rate = sum(w["steps"] for w in windows) / sum(w["seconds"] for w in windows)
    require(number(rate) and close(rate, decode["fixed_pooled_step_s"]), "pooled rate arithmetic mismatch")
    require(len(decode["windows"]) == len(rates) and all(close(a, b) for a, b in zip(rates, decode["windows"]))
            and close(median(rates), decode["windows_med"]), "window median/rates arithmetic mismatch")
    prefill = record["prefill"]
    require([row["ctx"] for row in prefill] == [2000, 32000, 128000], "prefill context coverage incomplete")
    for row in prefill:
        qs = [q for q in requests[:5] if q["ctx"] == row["ctx"]]
        samples = [q["ttft_s"] for q in qs]
        warm = min(samples[1:]) if len(samples) > 1 else samples[0]
        require(row["ttft_samples_s"] == samples and row["combined"] is (len(samples) == 1)
                and row["tok"] == qs[-1]["prompt_tokens"]
                and close(row["cold_s"], samples[0]) and close(row["warm_s"], warm)
                and close(row["cold_tok_s"], row["tok"] / samples[0])
                and close(row["warm_tok_s"], row["tok"] / warm), "prefill timing arithmetic mismatch")
    return dict(name=record["name"], boot_id=record["boot_id"], cold_compile=record.get("cold_compile", False),
                fixed_pooled_step_s=rate, ms_per_step=1000 / rate, windows=len(windows),
                windows_med=decode["windows_med"], median_ms_per_step=1000 / decode["windows_med"],
                output_tok_s=sum(q["completion_tokens"] - 1 for q in fixed) / sum(q["decode_s"] for q in fixed),
                acceptance_all_requests=decode.get("acc_raw"), prefill=prefill, quality=record["quality"],
                korean=korean, exclusive=True, recorded_request_output_hashes_validated=True,
                ordered_stream_hashes_verified=not canonical, independent_stream_hashes_verified=not canonical)


def comparison(candidate, baseline):
    cold_comparable = candidate["cold_compile"] == baseline["cold_compile"]
    prefill = []
    for a, b in zip(candidate["prefill"], baseline["prefill"]):
        row = dict(ctx=a["ctx"], cold_comparable=cold_comparable,
                   candidate_cold_s=a["cold_s"], baseline_cold_s=b["cold_s"],
                   candidate_cold_tok_s=a["cold_tok_s"], baseline_cold_tok_s=b["cold_tok_s"],
                   cold_ttft_change_pct=100 * (a["cold_s"] / b["cold_s"] - 1) if cold_comparable else None,
                   cold_note="same compile state" if cold_comparable else "cold incomparable: compile-cold states differ",
                   warm_independent=not a["combined"])
        if not a["combined"]:
            row.update(candidate_warm_s=a["warm_s"], baseline_warm_s=b["warm_s"],
                       candidate_warm_tok_s=a["warm_tok_s"], baseline_warm_tok_s=b["warm_tok_s"],
                       warm_ttft_change_pct=100 * (a["warm_s"] / b["warm_s"] - 1),
                       warm_kind="minimum of two repeated TTFT requests")
        else:
            row["warm_note"] = "one combined request; stored warm equals cold and is not a separate warm sample"
        prefill.append(row)
    return dict(primary="fixed_pooled_step_s", candidate_step_s=candidate["fixed_pooled_step_s"],
                baseline_step_s=baseline["fixed_pooled_step_s"],
                step_s_change_pct=100 * (candidate["fixed_pooled_step_s"] / baseline["fixed_pooled_step_s"] - 1),
                candidate_ms_per_step=candidate["ms_per_step"], baseline_ms_per_step=baseline["ms_per_step"],
                saved_ms_per_step=baseline["ms_per_step"] - candidate["ms_per_step"],
                ms_per_step_reduction_pct=100 * (1 - candidate["ms_per_step"] / baseline["ms_per_step"]),
                windows_med_change_pct=100 * (candidate["windows_med"] / baseline["windows_med"] - 1),
                output_tok_s_change_pct=100 * (candidate["output_tok_s"] / baseline["output_tok_s"] - 1),
                prefill=prefill)


def validate_observer(root, candidate, baseline, revision, records, *, sf6_direct=False):
    """A passive observer's completion is separate from supervisor completion."""
    report = read_json(root / "observer.json")
    require(report.get("schema") == 1 and report.get("status") == "PASS" and not report.get("errors"),
            "passive observer is not complete: " + str(report.get("status")) + " " + str(report.get("errors", [])))
    require(report.get("source_commit") == revision and report.get("candidate") == candidate
            and report.get("baseline") == baseline, "observer source/arm identity differs")
    require(report.get("sf6_direct", False) is sf6_direct, "observer SF6 direct variant differs")
    arms = report.get("arms", {})
    require(set(arms) == {candidate, baseline}, "both observed arms required")
    record_hashes = {transport_evidence._record(line)["name"]: hashlib.sha256(line).hexdigest()
                     for line in (root / "records.raw.jsonl").read_bytes().splitlines() if line.strip()}
    for mode, name in (("candidate", candidate), ("baseline", baseline)):
        arm = arms[name]
        require(arm.get("mode") == mode and arm.get("status") == "PASS" and not arm.get("errors")
                and arm.get("head_boot_id") == records[name]["boot_id"], "observer arm incomplete or boot differs: " + name)
        require(arm.get("record_sha256") == record_hashes[name], "observed onepass record changed: " + name)
        for field, prefix in (("before_receipt", "prepared"), ("after_receipt", "runtime")):
            filename = "observed-" + prefix + "-" + name + ".json"
            require(arm.get(field) == filename, "observer phase receipt identity differs")
            receipt = read_json(root / filename)
            require(receipt.get("status") == "PASS" and not receipt.get("errors"), "observer phase did not pass: " + filename)
            require(receipt.get("sf6_direct", False) is sf6_direct,
                    "observer phase SF6 direct variant differs: " + filename)
            if sf6_direct:
                require(receipt.get("schema") == 1 and receipt.get("phase") == prefix
                        and receipt.get("arm") == name and receipt.get("mode") == mode
                        and receipt.get("source_commit") == revision,
                        "observer phase source/arm identity differs: " + filename)
            required = {prefix + "-" + name + "-" + host + suffix for host in HOSTS for suffix in (".json", ".log")}
            hashes = receipt.get("artifacts_sha256", {})
            require(isinstance(hashes, dict) and required <= set(hashes), "observer phase artifacts incomplete")
            sealed(root, receipt, hashes)
    return dict(status="PASS", source_commit=revision, arms=[candidate, baseline], sf6_direct=sf6_direct,
                record_and_snapshot_artifacts_verified=True,
                first_snapshot="before completed record; not necessarily before traffic")


def summarize(root, candidate, baseline, *, canonical=False, sf6_direct=False):
    root = Path(root)
    result = dict(schema="decode-next-onepass-v1", status="INVALID", valid=False, errors=[], per_boot=[],
                  candidate=candidate, baseline=baseline, comparison=None, gpu={}, sf6_direct=sf6_direct,
                  mode="canonical" if canonical else "independent-gates", pending=[],
                  note="One boot per arm. Observed differences only; no statistical significance claim. "
                       "Within-boot windows are correlated and do not estimate boot drift. "
                       "Output hashes match each arm's recorded stream; outputs need not match across arms.")
    if canonical:
        result.update(independent_stream_hashes_verified=False,
                      dedicated_gpu_correctness="not run under onepass-only policy",
                      coverage=dict(measured_onepass_valid=False, startup_selftests_verified=False,
                                    independent_stream_hashes_verified=False, dedicated_gpu_correctness_verified=False))
        result["note"] = ("One boot per arm. Observed differences only; no statistical significance claim. "
                          "Within-boot windows are correlated and do not estimate boot drift. "
                          "Request/output hashes come from onepass itself; there is no independent SSE recording. "
                          "Startup self-tests and onepass quality do not replace dedicated GPU numerical or race testing.")

    def check(stage, action):
        try:
            return action()
        except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as exc:
            result["errors"].append(dict(stage=stage, error=str(exc)))
            return None

    def load_campaign():
        require(type(sf6_direct) is bool and (not sf6_direct or canonical),
                "SF6 direct evidence requires canonical mode")
        require(candidate != baseline and all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in (candidate, baseline)),
                "distinct safe arm names required")
        revision = (root / "source.commit").read_text().strip()
        require(re.fullmatch(r"[0-9a-f]{40}", revision), "full source commit required")
        records = read_jsonl(root / "records.raw.jsonl")
        require(len(records) == 2 and {r["name"] for r in records} == {candidate, baseline}, "exactly two named onepass records required")
        require(len({r["boot_id"] for r in records}) == 2, "two distinct serving boots required")
        require(len({r["overlay"] for r in records}) == 1 and all(re.fullmatch(r"[0-9a-f]{12}", r["overlay"]) for r in records),
                "matched nonempty served overlay stamp required")
        require(len({r["git"] for r in records}) == 1 and all(re.fullmatch(r"[0-9a-f]{7,40}", r["git"])
                and revision.startswith(r["git"]) for r in records), "onepass/campaign source differs")
        return revision, {r["name"]: r for r in records}

    campaign = check("campaign records", load_campaign)
    exit_path = root / "campaign.exit"
    result["campaign_cleanup"] = "pending" if not exit_path.exists() else "complete"
    if exit_path.exists():
        check("campaign exit/cleanup", lambda: require(exit_path.read_text().strip() == "0", "campaign exited " + exit_path.read_text().strip()))
    elif canonical:
        result["pending"].append("supervisor final exit receipt is not available")
    if campaign is None:
        return result
    revision, records = campaign
    result["source_commit"] = revision
    if canonical:
        result["observer"] = check("passive observer completion", lambda: validate_observer(
            root, candidate, baseline, revision, records, sf6_direct=sf6_direct))
    snapshots, manifests, rows = {}, {}, {}
    for mode, name in (("candidate", candidate), ("baseline", baseline)):
        if not canonical:
            check(name + " arm exit", lambda name=name: require((root / ("arm-" + name + ".exit")).read_text().strip() == "0", "serving arm failed"))
        record = records[name]
        row = check(name + (" onepass" if canonical else " onepass/channels"), lambda: validate_record(
            record, None if canonical else read_jsonl(root / ("channels-" + name + ".jsonl")), canonical=canonical))
        if row is not None:
            rows[name] = row
            result["per_boot"].append(row)
        else:
            result["per_boot"].append(dict(name=name, valid=False, unvalidated_decode=record.get("decode", {})))
        memory_path = name + ".memory.jsonl" if canonical else "memory-" + name + ".jsonl"
        memory = check(name + " memory", lambda: validate_memory(read_jsonl(root / memory_path)))
        if row is not None:
            row["memory"] = memory
        expected = check(name + " expected source", lambda: runtime_proof.validate_manifest(read_json(root / ("expected-" + name + ".json"))))
        if expected is None:
            continue
        manifests[name] = expected
        snapshots[mode] = {}
        for host in HOSTS:
            def verify_host():
                before = read_json(root / ("prepared-" + name + "-" + host + ".json"))
                after = read_json(root / ("runtime-" + name + "-" + host + ".json"))
                issues = runtime_proof.validate_report(before, expected, sf6_direct=sf6_direct)
                issues += runtime_proof.validate_report(after, expected, sf6_direct=sf6_direct)
                issues += runtime_proof.compare_snapshots(before, after, sf6_direct=sf6_direct)
                require(not issues, "; ".join(issues))
                if sf6_direct:
                    for prefix, proof in (("prepared", before), ("runtime", after)):
                        filename = prefix + "-" + name + "-" + host + ".log"
                        raw = (root / filename).read_bytes()
                        require(proof.get("log_sha256") == hashlib.sha256(raw).hexdigest(),
                                "runtime log SHA differs: " + filename)
                        require(proof.get("markers") == runtime_proof.parse_markers(raw.decode(errors="replace")),
                                "runtime markers differ from retained log: " + filename)
                require(before["mode"] == after["mode"] == mode and after["host"] == host, "rank host/arm identity mismatch")
                require(after["knobs"].get("VLLM_GLM53_SPEC_K") == "5", "actual SPEC_K differs")
                if host == "srv2":
                    require(after["boot_id"] == record["boot_id"], "onepass and runtime boot differ")
                return after
            after = check(name + " " + host + " runtime", verify_host)
            if after is not None:
                snapshots[mode][host] = after
    check("matched runtime arms", lambda: require(not (issues := runtime_proof.compare_arms(
        snapshots.get("baseline"), snapshots.get("candidate"), sf6_direct=sf6_direct)), "; ".join(issues)))
    identity_keys = ("request_sha256", "prompt_tokens", "seed", "min_tokens", "max_tokens")
    check("matched ordered requests", lambda: require(
        [[q[key] for key in identity_keys] for q in records[candidate]["requests"]]
        == [[q[key] for key in identity_keys] for q in records[baseline]["requests"]], "ordered requests differ between arms"))
    if candidate in manifests and baseline in manifests:
        check("matched source manifests", lambda: require(manifests[candidate] == manifests[baseline], "candidate/baseline source manifests differ"))
        if not canonical:
            result["gpu"]["transport"] = check("transport GPU gate", lambda: retained_transport(root / "transport-gpu", manifests[candidate], revision))
            result["gpu"]["sf6"] = check("SF6 GPU gate", lambda: retained_sf6(root / "sf6-gpu", manifests[candidate], revision))
    if canonical:
        result["coverage"]["measured_onepass_valid"] = len(rows) == 2 and not any(
            error["stage"].endswith(("onepass", "memory")) or error["stage"] == "matched ordered requests"
            for error in result["errors"])
        result["coverage"]["startup_selftests_verified"] = all(len(snapshots.get(mode, {})) == 4 for mode in ("baseline", "candidate"))
    result["valid"] = not result["errors"] and not result["pending"]
    result["status"] = "INVALID" if result["errors"] else "PENDING" if result["pending"] else "PASS"
    if result["valid"]:
        result["comparison"] = comparison(rows[candidate], rows[baseline])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--canonical", action="store_true", help="official onepass-only evidence; no independent SSE or GPU gates")
    parser.add_argument("--sf6-direct", action="store_true", help="packed-only SF6 variant; requires --canonical")
    args = parser.parse_args(argv)
    if args.sf6_direct and not args.canonical:
        parser.error("--sf6-direct requires --canonical")
    result = summarize(args.root, args.candidate, args.baseline, canonical=args.canonical, sf6_direct=args.sf6_direct)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
