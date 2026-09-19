#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What a check measured, left where the queue finds it again.

A check run through the queue answers with an exit code, and its numbers scroll past in a
log. `write_report` leaves them as one JSON file instead -- metrics, proof markers, how many
samples, which device -- bound to the ticket that ran it.

It lives under probes/ because that is what reaches a check. The single-GPU lane's runner
ships engine/, probes/ and tests/ to the lane's host and nothing else (run_engine_probe.sh),
and only ST_* and STK_* variables cross into the container: a writer under bench/ cannot be
imported there, and one that waits for FLEET_* variables is never told where to write. The
first probe to use the old one (engine_qwen38_hc_mix_fused.py) would have died on its import
line on the lane its docstring names.

For a single-GPU ticket the supervisor (bench/fleet_boot.py) sets

    ST_PROBE_REPORT    a file under the container's /cache -- the host's ~/.cache/st, which the
                       lane copies back to results/<session>/ when the ticket releases
    ST_PROBE_SESSION   the ticket

Without ST_PROBE_REPORT this writes nothing: a bare run prints, as it always did.

A report is what the check saw on that device. `passed` is its own proof markers, all true;
it is never a speed verdict, and nothing that judges serving speed reads it.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

SCHEMA = 2                      # 1 was bench/probe_report.py's nonce-bound form, for a lane that is gone
SCOPE = 'what this check measured on this device; not a speed verdict'


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build(metrics, proof, samples, device, *, session='', probe='') -> dict:
    """The report, or ValueError: a malformed report is refused here, not discovered by its reader."""
    if not isinstance(metrics, dict) or not metrics or any(not isinstance(k, str) or not k or not _number(v)
                                                           for k, v in metrics.items()):
        raise ValueError('probe report metrics must map names to finite numbers')
    if not isinstance(proof, dict) or not proof or any(not isinstance(k, str) or not k or type(v) is not bool
                                                       for k, v in proof.items()):
        raise ValueError('probe report proof must map marker names to booleans')
    if type(samples) is not int or samples < 1:
        raise ValueError('probe report samples must be a positive integer')
    if not isinstance(device, str) or not device.strip():
        raise ValueError('probe report must name the device it ran on')
    return dict(schema=SCHEMA, session=session, probe=probe, device=device.strip(), samples=samples,
                passed=all(proof.values()), failed=sorted(k for k, v in proof.items() if not v),
                metrics=dict(metrics), proof=dict(proof), scope=SCOPE, written_at=round(time.time(), 3))


def write_report(metrics, proof, samples, device, environ=None):
    """Call once, after every case and guard has run. Returns the path written, or None when nobody asked."""
    env = os.environ if environ is None else environ
    target = env.get('ST_PROBE_REPORT', '').strip()
    if not target:
        return None
    probe = '/'.join(Path(sys.argv[0]).parts[-2:]) if sys.argv and sys.argv[0] else ''
    report = build(metrics, proof, samples, device, session=env.get('ST_PROBE_SESSION', ''), probe=probe)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, allow_nan=False, sort_keys=True) + '\n')
    temporary.replace(path)
    return path

