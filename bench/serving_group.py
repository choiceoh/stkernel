# SPDX-License-Identifier: Apache-2.0
"""Several onepass workloads on one attested boot, within one fleet hold."""
import json
import os
from pathlib import Path
import subprocess
import re
from measurement_contract import evaluations, environment, metadata


def workloads(spec):
    result = []
    for item in evaluations(spec):
        if item['workload'] not in result:
            result.append(item['workload'])
    return result


def name_for(name, spec, index):
    index = workloads(spec).index(evaluations(spec)[index]['workload'])
    return name if index == 0 else f'{name}-E{index + 1}'


def measure(store, job, payload, name, knobs, work_indices=None):
    from baseline import load, is_baseline
    from experiments import verify, knob_mismatch
    from judge import compatible, record_errors, unproved
    boot = None
    records = []
    profile_text = (Path(payload['repo']) / 'profiles/glm53.env').read_text()
    values = workloads(payload['spec'])
    indices = list(range(len(values))) if work_indices is None else work_indices
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(values) for i in indices):
        raise ValueError('invalid workload selection')
    for index in indices:
        work = values[index]
        verify(payload)
        arm = name if index == 0 else f'{name}-E{index + 1}'
        env = dict(os.environ, **environment(dict(workload=work, objective={'metric':'decode_steps'})))
        if not records:
            command = [payload['bash'], str(Path(payload['repo']) / 'bench/ab-lever.sh'), arm,
                       ' '.join(k + '=' + v for k, v in sorted(knobs.items()))]
            env.update(SKIP_BOOT='0', LEGS='onepass')
        else:
            from onepass import _served_build
            actual = _served_build(payload['repo'])
            if actual.get('boot_id') != boot:
                raise ValueError('serving changed between grouped workloads')
            command = ['python3', 'bench/onepass.py', '--name', arm]
            spec_k = re.findall(r'^SPEC_K=([0-9]+)', profile_text, re.M)
            env.update(SPEC_K=spec_k[-1] if spec_k else '6', BENCH_MODEL='glm-5.3-flash')
            if records[0].get('cold_compile'):
                env['MK_COLD_COMPILE'] = '1'
        if records:
            from experiment_metrics import timed
            with timed(store,job,'measure'):
                rc = subprocess.call(command, cwd=payload['repo'], env=env)
        else:
            rc = subprocess.call(command, cwd=payload['repo'], env=env)
        if rc:
            raise ValueError(f'onepass arm {arm} exited {rc}')
        verify(payload)
        fresh = [r for r in load(payload['paths']['ONEPASS_JSONL'])
                 if r.get('name') == arm and r.get('experiment_id') == job and not r.get('rehearsal')]
        ref = dict(overlay=payload['snapshot']['build'][:12], runtime=payload['spec']['context'], **metadata(work))
        if (len(fresh) != 1 or not compatible(fresh[0], ref) or record_errors(fresh[0])
                or unproved(fresh[0]) or not fresh[0].get('boot_id')):
            raise ValueError('grouped arm lacks fresh matching workload, boot or quality/proof evidence')
        record = fresh[0]
        if not payload['spec']['revision'].startswith(record.get('git') or 'MISSING'):
            raise ValueError('grouped arm revision is not the submitted source')
        if not knobs and not is_baseline(record)[0]:
            raise ValueError('defaults boot attested unexpected knobs')
        if knob_mismatch(record, knobs, payload['repo']):
            raise ValueError('serving knobs do not exactly match the requested configuration')
        if boot is not None and record['boot_id'] != boot:
            raise ValueError('grouped arm did not run on the same boot')
        boot = record['boot_id']
        records.append(record)
        with store.db:
            store.event(job, 'arm_result', dict(name=arm, record=record))
    return records


def run_pair(store, job, payload):
    from experiments import pair_result
    try:
        measure(store, job, payload, 'EXP-' + job, payload['spec']['knobs'])
        state, result = pair_result(store.get(job)['payload'], job)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        state, result = 'failed', dict(evidence='gpu-pair', reason=str(exc))
    return state, result
