#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Structured numerical probe evidence; a pass is not a serving speed verdict."""
import json
import math
import os
from pathlib import Path


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def contract(value):
    if not isinstance(value, dict) or set(value) != {"checks", "proof", "min_samples"}:
        raise ValueError("probe_contract requires checks, proof and min_samples")
    if type(value["min_samples"]) is not int or value["min_samples"] < 1:
        raise ValueError("probe min_samples must be positive")
    if (not isinstance(value["proof"], list) or not value["proof"]
            or any(not isinstance(k, str) or not k for k in value["proof"])
            or len(set(value["proof"])) != len(value["proof"])):
        raise ValueError("probe requires distinct nonempty proof markers")
    if not isinstance(value["checks"], dict) or not value["checks"]:
        raise ValueError("probe requires numerical checks")
    for name, rule in value["checks"].items():
        if (not name or not isinstance(rule, dict) or set(rule) != {"op", "value"}
                or rule["op"] not in {"le", "ge", "eq"} or not number(rule["value"])):
            raise ValueError("probe checks require a finite threshold and le/ge/eq")
    return value


def write_report(metrics, proof, samples, device):
    """Call only after the real probe finishes all cases and guard checks.

    The runner supplies a fresh nonce and a private output directory. Container
    wrappers must explicitly forward those fields and mount that directory.
    """
    if not os.environ.get("FLEET_PROBE_REPORT"):
        return
    report = dict(schema=1, nonce=os.environ["FLEET_PROBE_NONCE"],
                  experiment_id=os.environ["FLEET_EXPERIMENT_ID"],
                  binding=json.loads(os.environ["FLEET_PROBE_BINDING"]),
                  metrics=metrics, proof=proof, samples=samples, device=device)
    path = Path(os.environ["FLEET_PROBE_REPORT"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, allow_nan=False) + "\n")
    tmp.replace(path)


def adjudicate(path, expected, job, nonce, binding):
    if not expected or not Path(path).exists():
        return "incomplete", {"evidence": "probe-log", "reason": "no structured probe contract/report"}
    try:
        report = json.loads(Path(path).read_text())
        if not isinstance(report, dict) or any(report.get(k) != v for k, v in {
                "schema": 1, "nonce": nonce, "experiment_id": job, "binding": binding}.items()):
            raise ValueError("stale or mismatched probe report")
        if (type(report.get("samples")) is not int or report["samples"] < expected["min_samples"]
                or not isinstance(report.get("device"), str) or not report["device"].strip()):
            raise ValueError("probe coverage or device evidence missing")
        if not isinstance(report.get("metrics"), dict) or not isinstance(report.get("proof"), dict):
            raise ValueError("probe metrics/proof missing")
        failures = []
        for name, rule in expected["checks"].items():
            value = report["metrics"].get(name)
            limit = rule["value"]
            if not number(value) or not {"le": lambda: value <= limit, "ge": lambda: value >= limit,
                                        "eq": lambda: value == limit}[rule["op"]]():
                failures.append(name)
        failures += [name for name in expected["proof"] if report["proof"].get(name) is not True]
        return ("failed" if failures else "succeeded"), dict(
            evidence="gpu-probe", scope="declared numerical contract only", report=report, failures=failures)
    except (ValueError, TypeError, KeyError) as exc:
        return "incomplete", {"evidence": "probe-log", "reason": str(exc)}
