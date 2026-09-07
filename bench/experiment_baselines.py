#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One shared defaults reservation per immutable pair context.

Build the independent noise sample set once, before admitting candidate pairs.
The reservation goes through normal fleet preflight/admission and never holds
the GPU while waiting for a CPU/probe prerequisite.
"""
import copy
import subprocess


def reference(payload, index=0):
    from measurement_contract import metadata, evaluations
    return dict(overlay=payload["snapshot"]["build"][:12], git=payload["spec"]["revision"],
                runtime=payload["spec"]["context"], **metadata(evaluations(payload['spec'])[index]['workload']))


def samples(payload, index=0):
    from measurement_contract import evaluations, metric_value
    from baseline import load
    from judge import baselines_on, record_errors
    rows, _ = baselines_on(load(payload["paths"]["ONEPASS_JSONL"]), reference(payload, index))
    # Actual container ID + StartedAt distinguishes independent boots. Replays
    # of one boot and historical records without this evidence cannot pad n.
    distinct = {}
    obj = evaluations(payload['spec'])[index]['objective']
    for r in rows:
        usable = obj['metric'] == 'quality' or metric_value(r, obj) is not None
        if obj['metric'] == 'prefill_ttft' and r.get('cold_compile'):
            usable = False
        if (isinstance(r.get("boot_id"), str) and r["boot_id"] and not record_errors(r) and usable
                and payload['spec']['revision'].startswith(r.get('git') or 'MISSING')):
            distinct[r["boot_id"]] = r
    return list(distinct.values())


def reserve(store, job):
    existing = store.db.execute("SELECT dependency FROM dependencies WHERE job=? AND kind='baseline'", (job,)).fetchone()
    if existing:
        return existing[0]
    row = store.get(job)
    payload = copy.deepcopy(row["payload"])
    payload["spec"].update(kind="baseline", knobs={}, depends_on=[], command=[],
                           hypothesis="Prepare shared independent defaults samples", estimate_min=30)
    # CPU/probe contracts gate their own candidate. This reservation is only
    # created AFTER those gates pass, and is otherwise candidate-independent.
    payload["spec"].pop("probe_contract", None)
    payload["baseline_samples"] = 3
    request = store.submit("baseline-" + job, payload, repeat=row["repeat_reason"])
    with store.db:
        store.db.execute("INSERT OR IGNORE INTO dependencies VALUES(?,?,?)", (job, request["id"], "baseline"))
        store.event(job, "baseline_reserved", request)
    return request["id"]


def ready(payload):
    return not missing_workloads(payload)


def missing_workloads(payload):
    from measurement_contract import evaluations
    from serving_group import workloads
    values = workloads(payload['spec'])
    return sorted({values.index(e['workload']) for i, e in enumerate(evaluations(payload['spec']))
                   if len(samples(payload, i)) < (1 if e['objective']['metric'] == 'quality' else 3)})


def run(store, job, payload):
    from serving_group import measure
    target = payload['baseline_samples']
    try:
        # A first compile-cold record cannot supply a steady-compile TTFT sample.
        from measurement_contract import evaluations
        extra = int(any(e['objective']['metric'] == 'prefill_ttft' for e in evaluations(payload['spec'])))
        for index in range(target + extra):
            if ready(payload):
                break
            before = {r['boot_id'] for i, _ in enumerate(evaluations(payload['spec'])) for r in samples(payload, i)}
            missing = missing_workloads(payload)
            records = measure(store, job, payload, f'EXP-{job}-BASE-{index + 1}', {}, work_indices=missing)
            if records[0]['boot_id'] in before:
                raise ValueError('defaults sample reused an earlier boot')
            with store.db:
                store.event(job, 'baseline_sample', {'records': records, 'measured_workload_indices': missing})
        state = 'succeeded' if ready(payload) else 'incomplete'
        return state, dict(evidence='gpu-baseline', samples=len(samples(payload)),
                           baseline=samples(payload), scope='same build/workload/runtime')
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return 'failed', dict(evidence='gpu-baseline', reason=str(exc))
