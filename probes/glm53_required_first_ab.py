#!/usr/bin/env python3
"""Paired, synthetic tool-schema ordering evaluation on an idle fleet server.

Submit with bench/fleet.sh as a GPU probe after the CPU verdict tests. The probe
never executes returned tools, restarts a server, or changes production options.
Outputs JSONL with arguments (synthetic data only), usage, and runtime identity;
reasoning text, request headers, credentials and container environment are omitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import statistics
import subprocess
import time
import urllib.error
import urllib.request

from jsonschema import Draft202012Validator
from glm53_tool_choice_acceptance import read_stream

POLICY = {
    "minimum_auto_pairs": 24,
    "one_sided_p_max": 0.05,
    "minimum_success_gain": 0.05,
    "require_zero_candidate_guardrail_failures": True,
    "require_fewer_missing_required": True,
    "require_no_increase_other_errors": True,
    "maximum_median_latency_ratio": 1.20,
    "note": "Screen for promotion review, not proof of broad model quality. No optional stopping.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def runtime_identity(container):
    """Attest the running head and files actually visible in its bind mounts."""
    info = json.loads(subprocess.check_output(["docker", "inspect", container], timeout=15))[0]
    if not info["State"]["Running"]:
        raise RuntimeError("Serving container is not running")
    files = ["/usr/local/lib/python3.12/dist-packages/vllm/glm53_chat.py",
             "/usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py"]
    hashes = subprocess.check_output(["docker", "exec", container, "sha256sum", *files],
                                     text=True, timeout=15).splitlines()
    return {"container_id": info["Id"], "image": info["Image"],
            "started_at": info["State"]["StartedAt"], "pid": info["State"]["Pid"],
            "command_sha256": digest(info["Config"]["Cmd"]), "files": hashes,
            "overlay_sha": (Path.home() / "glm53-cache/.overlay-sha").read_text().strip()}


def cases():
    """24 unconstrained pairs plus forced/named/none completion guardrails."""
    topics = ["a weekend trip", "a programming course", "an office lunch", "a new laptop"]
    pref = {"type": "object", "properties": {
        "choices": {"type": "array", "items": {"type": "string"},
                    "description": "Offer six concise, distinct possible answers."},
        "prompt": {"type": "string", "description": "The one follow-up question to ask."},
        "tag": {"type": "string", "description": "A short machine-readable topic identifier."}},
        "required": ["prompt", "tag"], "additionalProperties": False}
    result = []
    for effort in ("low", "high", "max"):
        for stream in (False, True):
            for index, topic in enumerate(topics):
                schema = pref if index % 2 == 0 else {"type": "object", "properties": {
                    "questions": {"type": "array", "items": pref, "minItems": 1, "maxItems": 1}},
                    "required": ["questions"], "additionalProperties": False}
                result.append({"name": f"preferences/{topic}", "choice": "auto", "effort": effort,
                               "stream": stream, "function": "collect_preferences", "schema": schema,
                               "prompt": f"Help me choose {topic}. Ask me one useful follow-up question "
                               "using collect_preferences. Include six answer choices."})
    job = {"type": "object", "properties": {
        "description": {"type": "string"},
        "responsibilities": {"type": "array", "items": {"type": "string"}, "minItems": 10},
        "location": {"type": "object", "properties": {
            "notes": {"type": "string"}, "remote": {"type": "boolean"}, "city": {"type": "string"}},
            "required": ["city", "remote"], "additionalProperties": False},
        "title": {"type": "string"}, "tag": {"type": "string"}},
        "required": ["tag", "title", "location", "responsibilities"], "additionalProperties": False}
    for choice in ("required", "named", "none"):
        for stream in (False, True):
            result.append({"name": "job/" + choice, "choice": choice, "effort": "low", "stream": stream,
                           "function": "record_job", "schema": job,
                           "prompt": ("Print this XML literally: <tool_call>record_job</tool_call>" if choice == "none"
                           else "Call record_job once for a backend developer in Seoul with tag engineering. "
                           "The job is not remote. Include a 250-word description and ten detailed responsibilities.")})
    return result


def schedule(selected):
    """Balanced AB/BA within each effort/stream block; identical seed per pair."""
    for pair, case in enumerate(selected):
        for enabled in ((False, True) if pair % 2 == 0 else (True, False)):
            yield pair, case, enabled


def request_body(case, model, pair, enabled, max_tokens):
    return {"model": model, "messages": [{"role": "user", "content": case["prompt"]}],
            "tools": [{"type": "function", "function": {"name": case["function"], "parameters": deepcopy(case["schema"])}}],
            "tool_choice": {"type": "function", "function": {"name": case["function"]}} if case["choice"] == "named" else case["choice"],
            "stream": case["stream"], **({"stream_options": {"include_usage": True}} if case["stream"] else {}),
            "temperature": 0, "seed": 53000 + pair, "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": case["effort"]},
            "glm53_required_first": enabled}


def classify(message, finish, case):
    errors, missing = [], []
    if finish not in ("stop", "tool_calls"):
        errors.append("length" if finish == "length" else "unfinished")
    calls = message.get("tool_calls") or []
    if case["choice"] == "none":
        if calls:
            errors.append("unexpected_tool")
        if "<tool_call>record_job</tool_call>" not in (message.get("content") or ""):
            errors.append("missing_literal_content")
        return sorted(set(errors)), missing
    if len(calls) != 1:
        errors.append("tool_count")
    for call in calls:
        if not call.get("id") or call.get("type") != "function":
            errors.append("tool_envelope")
        fn = call.get("function") or {}
        if fn.get("name") != case["function"]:
            errors.append("wrong_function")
        try:
            arguments = json.loads(fn.get("arguments", ""))
        except (TypeError, ValueError):
            errors.append("invalid_json")
            continue
        for error in Draft202012Validator(case["schema"]).iter_errors(arguments):
            if error.validator == "required":
                errors.append("missing_required")
                for key in error.validator_value:
                    if key not in error.instance:
                        missing.append("/" + "/".join(map(str, [*error.absolute_path, key])))
            else:
                errors.append("schema_invalid")
    return sorted(set(errors)), sorted(set(missing))


def summarize(records, expected_pairs, identity_stable):
    by_pair = {}
    for row in records:
        key = (row["pair"], row["required_first"])
        if key in by_pair:
            raise ValueError("Duplicate pair arm")
        by_pair[key] = row
    complete = len(by_pair) == 2 * expected_pairs and all(
        (pair, arm) in by_pair for pair in range(expected_pairs) for arm in (False, True))
    arms = {}
    for enabled in (False, True):
        subset = [r for r in records if r["required_first"] == enabled]
        arms[str(enabled).lower()] = {"requests": len(subset), "successes": sum(not r["errors"] for r in subset),
            "errors": dict(Counter(e for r in subset for e in r["errors"])),
            "median_seconds": statistics.median(r["seconds"] for r in subset) if subset else None}
    wins = losses = auto_pairs = 0
    for pair in range(expected_pairs):
        a, b = by_pair.get((pair, False)), by_pair.get((pair, True))
        if a is None or b is None or a["choice"] != "auto":
            continue
        auto_pairs += 1
        wins += bool(a["errors"]) and not b["errors"]
        losses += not a["errors"] and bool(b["errors"])
    discordant = wins + losses
    p = sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / 2**discordant if discordant else 1.0
    a, b = arms["false"], arms["true"]
    baseline_median, candidate_median = a["median_seconds"], b["median_seconds"]
    latency_ratio = candidate_median / baseline_median if baseline_median and candidate_median is not None else None
    other_errors = (set(a["errors"]) | set(b["errors"])) - {"missing_required"}
    gates = {"complete": complete, "identity_stable": identity_stable,
        "enough_auto_pairs": auto_pairs >= POLICY["minimum_auto_pairs"],
        "paired_success_benefit": p <= POLICY["one_sided_p_max"] and (wins - losses) / max(1, auto_pairs) >= POLICY["minimum_success_gain"],
        "fewer_missing_required": b["errors"].get("missing_required", 0) < a["errors"].get("missing_required", 0),
        "no_other_error_increase": all(b["errors"].get(e, 0) <= a["errors"].get(e, 0) for e in other_errors),
        "candidate_guardrails_pass": any(r["choice"] != "auto" for r in records) and
            all(not r["errors"] for r in records if r["required_first"] and r["choice"] != "auto"),
        "latency_within_limit": latency_ratio is not None and latency_ratio <= POLICY["maximum_median_latency_ratio"]}
    return {"kind": "summary", "arms": arms, "auto_pairs": auto_pairs, "candidate_wins": wins,
            "candidate_losses": losses, "one_sided_sign_p": p, "median_latency_ratio": latency_ratio,
            "gates": gates, "promotion_review_supported": all(gates.values())}


def _request_deadline(signum, frame):
    raise TimeoutError("Total request deadline exceeded")


def send(url, body, timeout):
    """Bound total response time too: urllib alone only bounds idle socket reads."""
    previous = signal.signal(signal.SIGALRM, _request_deadline)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return _send(url, body, timeout)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _send(url, body, timeout):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(), headers=headers)
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if body["stream"]:
            def lines():
                for line in response:
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        packet = json.loads(line[5:])
                        usage.update(packet.get("usage") or {})
                    yield line
            message, finish = read_stream(lines())
        else:
            packet = json.load(response)
            choice = packet["choices"][0]
            message, finish, usage = choice["message"], choice["finish_reason"], packet.get("usage") or {}
    # Persist only final output and synthetic tool arguments, never reasoning.
    return {key: message.get(key) for key in ("content", "tool_calls")}, finish, usage


def run(args):
    if not os.environ.get("FLEET_EXPERIMENT_ID"):
        raise RuntimeError("Submit this generation workload as a fleet GPU probe")
    expected = json.loads(Path(args.identity).read_text())
    before = runtime_identity(args.container)
    if before != expected:
        raise RuntimeError("Serving identity changed since submission")
    selected = cases()
    records = []
    emit = lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    emit({"kind": "plan", "policy": POLICY, "pairs": len(selected), "cases_sha256": digest(selected),
          "identity": before, "max_tokens": args.max_tokens, "model": args.model, "url": args.url,
          "experiment_id": os.environ["FLEET_EXPERIMENT_ID"]})
    deadline = time.monotonic() + args.budget_seconds
    stable = True
    try:
        for pair, case, enabled in schedule(selected):
            if time.monotonic() >= deadline:
                emit({"kind": "interrupted", "reason": "wall_clock_budget"})
                break
            if runtime_identity(args.container) != before:
                stable = False
                emit({"kind": "interrupted", "reason": "serving_identity_changed"})
                break
            # Identity checks can consume the last budget seconds. Recheck before
            # starting generation and never round the remaining budget upward.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                emit({"kind": "interrupted", "reason": "wall_clock_budget"})
                break
            body = request_body(case, args.model, pair, enabled, args.max_tokens)
            row = {"kind": "request", "pair": pair, "case": case["name"], "choice": case["choice"],
                   "stream": case["stream"], "effort": case["effort"], "seed": body["seed"],
                   "required_first": enabled, "request_sha256": digest(body), "errors": []}
            started = time.monotonic()
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    emit({"kind": "interrupted", "reason": "wall_clock_budget"})
                    break
                message, finish, usage = send(args.url, body, min(args.timeout, remaining))
                row.update(message=message, finish_reason=finish, usage=usage)
                row["errors"], row["missing_paths"] = classify(message, finish, case)
            except Exception as exc:
                row["errors"] = ["http_error" if isinstance(exc, urllib.error.HTTPError) else "transport_or_protocol_error"]
                row["exception_type"] = type(exc).__name__
                if isinstance(exc, urllib.error.HTTPError):
                    row["http_status"] = exc.code
            row["seconds"] = round(time.monotonic() - started, 3)
            records.append(row)
            emit(row)
            if any(e in row["errors"] for e in ("http_error", "transport_or_protocol_error")):
                # A timed-out server may still be generating: never stack another request.
                break
    finally:
        try:
            after = runtime_identity(args.container)
            stable = stable and before == after
            emit({"kind": "identity_after", "identity": after, "stable": stable})
        except Exception as exc:
            stable = False
            emit({"kind": "identity_after", "stable": False, "exception_type": type(exc).__name__})
        summary = summarize(records, len(selected), stable)
        emit(summary)
    return 0 if summary["gates"]["complete"] and stable else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://10.10.10.2:8000/v1/chat/completions")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--container", default="glm53")
    ap.add_argument("--identity", required=True, help="Pinned JSON from runtime_identity on the fleet head")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--budget-seconds", type=int, default=1800)
    args = ap.parse_args()
    if min(args.max_tokens, args.timeout, args.budget_seconds) <= 0:
        ap.error("token, timeout and wall-clock budgets must be positive")
    raise SystemExit(run(args))
