#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One fixed build, one PRIME, and two shared controls around repeated candidates.

No cross-build/cross-run baseline reuse. Every timed arm still needs four-node
warm-cache receipts and the canonical onepass quality workload. This produces
exploration evidence; drift or missing proof prevents a performance verdict.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(spec, profile):
    if not isinstance(spec, dict) or spec.get('schema') != 1 or set(spec) - {'schema', 'baseline', 'prime', 'candidates', 'max_drift_fraction'}:
        raise ValueError('campaign schema must be 1 with baseline, prime and candidates')
    candidates = spec.get('candidates')
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 8:
        raise ValueError('campaign needs 1..8 candidates')
    declared = dict(re.findall(r'^(VLLM_[A-Z0-9_]+)=["\']?([^\n"\']*)', profile, re.M))
    def knobs(value):
        if not isinstance(value, dict) or not value:
            raise ValueError('each arm needs an explicit knob map')
        for key, val in value.items():
            if key not in declared or not isinstance(val, str) or not re.fullmatch(r'[A-Za-z0-9_.,:/+-]+', val):
                raise ValueError('undeclared or invalid knob: ' + str(key))
        effective = dict(declared, **value)
        if any(effective.get(k, '').lower() in ('', '0', 'off', 'false') for k in ('VLLM_GLM53_RANK_CACHE', 'VLLM_GLM53_FP8_CACHE')):
            raise ValueError('campaign sharing requires both artifact caches enabled')
        return value
    base, prime = knobs(spec.get('baseline')), knobs(spec.get('prime'))
    names, configs = set(), set()
    arms = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {'name', 'knobs'} or not isinstance(candidate['name'], str) or not re.fullmatch(r'[A-Z][A-Z0-9]{0,15}', candidate['name']):
            raise ValueError('candidate requires an uppercase name and knobs')
        name, values = candidate['name'], knobs(candidate['knobs'])
        config = json.dumps(values, sort_keys=True)
        if name in names or name in {'BASE', 'PRIME'} or config in configs or values == base:
            raise ValueError('duplicate/reserved candidate or unchanged baseline')
        names.add(name); configs.add(config)
        arms.append(dict(candidate=name, knobs=values))
    if any(set(row['knobs']) != set(base) for row in arms) or set(prime) != set(base):
        raise ValueError('all arms must explicitly set the same knobs to prevent inherited carryover')
    for key in ('VLLM_GLM53_RANK_CACHE', 'VLLM_GLM53_FP8_CACHE'):
        if any(values.get(key, declared.get(key)) != base.get(key, declared.get(key))
               for values in [prime, *(row['knobs'] for row in arms)]):
            raise ValueError('all campaign arms must use the same cache directories')
    drift = spec.get('max_drift_fraction', .1)
    if type(drift) not in (float, int) or not math.isfinite(drift) or not 0 < drift <= .25:
        raise ValueError('max_drift_fraction must be in (0, .25]')
    return [dict(stage='PRIME', role='prime', knobs=prime), dict(stage='BASE1', role='baseline', knobs=base)] + [
        dict(stage=row['candidate']+'R1', role='candidate', **row) for row in arms] + [
        dict(stage=row['candidate']+'R2', role='candidate', **row) for row in reversed(arms)] + [
        dict(stage='BASE2', role='baseline', knobs=base)]


def identity(repo):
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('campaign source must be committed and clean')
    stamp = Path(os.environ.get('MK_OVERLAY_STAMP', '/home/choiceoh/glm53-cache/.overlay-sha'))
    profile = (repo/'profiles/glm53.env').read_text()
    overlay = re.search(r'^PROFILE_OVERLAY_DIR=["\']?([^"\'\n]+)', profile, re.M)
    manifest = Path(overlay[1])/'manifest.tsv' if overlay else None
    if not manifest or f'# source_commit={revision}' not in manifest.read_text().splitlines():
        raise ValueError('deploy this campaign revision before starting it')
    # PRIME will publish the new manifest stamp at boot. Pin the deployed
    # manifest now instead of treating the previous boot's cache stamp as it.
    return dict(revision=revision, profile=digest(repo/'profiles/glm53.env'), overlay=stamp.read_text().strip(),
                deployed=digest(manifest), ctx=os.environ.get('QUALITY_CTX', '2000,32000'))


def check(root, stage, record, current):
    saved = json.loads((root/'campaign.json').read_text())
    expected = dict(saved['identity'], overlay=saved['identity'].get('deployed', saved['identity']['overlay']))
    if current != expected:
        raise ValueError('campaign build/profile/cache stamp/workload changed')
    arm = next(row for row in saved['arms'] if row['stage'] == stage)
    revision = record.get('git') or ''
    if (record.get('rehearsal') or len(revision) < 7 or not current['revision'].startswith(revision)
            or record.get('overlay') != current['overlay'][:12] or not record.get('boot_id')):
        raise ValueError('arm lacks a fresh matching build and boot receipt')
    quality, korean = record.get('quality', {}), record.get('korean', {})
    if not quality.get('total') or quality.get('ok') != quality['total'] or korean.get('dirty') != 0 or not korean.get('n'):
        raise ValueError('arm quality gate failed')
    # Compare the inspected container settings, including values equal to defaults.
    env = dict(v.split('=', 1) for v in json.loads((root/(record['name']+'-cache-env.json')).read_text()))
    if any(env.get(k) != v for k, v in arm['knobs'].items()):
        raise ValueError('serving knobs differ from campaign plan')
    path = root/'campaign-receipts.jsonl'
    previous = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    runtime = []
    for node in (1, 2, 3, 4):
        state = (root/f'{record["name"]}-srv{node}.state').read_text()
        image = re.search(r'\bsha256:[a-f0-9]{64}\b', state.splitlines()[0])
        hashes = (root/f'{record["name"]}-srv2.sha256').read_text() if node == 2 else state
        modules = re.findall(r'^([a-f0-9]{64})\s+(/usr/\S+)', hashes, re.M)
        if not image or len(modules) != 3:
            raise ValueError(f'srv{node}: missing image/module runtime receipt')
        runtime.append(dict(image=image[0], modules=modules))
    if previous and json.dumps(runtime, sort_keys=True) != json.dumps(previous[0]['runtime'], sort_keys=True):
        raise ValueError('campaign container image or node module changed')
    if any(row['boot_id'] == record['boot_id'] or row['stage'] == stage for row in previous):
        raise ValueError('campaign requires a distinct boot per arm')
    if stage != saved['arms'][len(previous)]['stage']:
        raise ValueError('campaign arm order changed')
    with path.open('a') as stream:
        stream.write(json.dumps(dict(stage=stage, boot_id=record['boot_id'], name=record['name'], runtime=runtime))+'\n')


def summarize(root):
    saved = json.loads((root/'campaign.json').read_text())
    receipts = [json.loads(line) for line in (root/'campaign-receipts.jsonl').read_text().splitlines()]
    if [r['stage'] for r in receipts] != [r['stage'] for r in saved['arms']]:
        raise ValueError('campaign incomplete')
    times = {}
    for line in (root/'health-wall-seconds.tsv').read_text().splitlines():
        name, seconds = line.split('\t')
        value = float(seconds)
        if name in times or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid or duplicate health timing')
        times[name] = value
    timings = {r['stage']: times[r['name']] for r in receipts}
    bases = [timings['BASE1'], timings['BASE2']]
    drift = abs(bases[1]-bases[0])/statistics.mean(bases)
    result = dict(status='exploration' if drift <= saved['spec'].get('max_drift_fraction', .1) else 'incomplete-drift',
                  promotion_ready=False, baseline_health_seconds=bases, baseline_drift_fraction=drift,
                  boots=len(receipts), independent_boots=5*len(saved['spec']['candidates']), candidates=[])
    for candidate in saved['spec']['candidates']:
        name = candidate['name']
        values = [timings[name+'R1'], timings[name+'R2']]
        result['candidates'].append(dict(name=name, health_seconds=values,
            reduction_fraction=1-statistics.mean(values)/statistics.mean(bases)))
    (root/'campaign-result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['validate', 'plan', 'check', 'summarize'])
    ap.add_argument('--spec', type=Path)
    ap.add_argument('--evidence', type=Path)
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--stage')
    args = ap.parse_args()
    if args.action in ('plan', 'validate'):
        spec = json.loads(args.spec.read_text())
        arms = plan(spec, (args.repo/'profiles/glm53.env').read_text())
        if args.action == 'validate':
            print(json.dumps(dict(boots=len(arms), independent_boots=5*len(spec['candidates']), arms=arms), indent=2))
            return
        with (args.evidence/'campaign.json').open('x') as stream:
            json.dump(dict(schema=1, spec=spec, arms=arms, identity=identity(args.repo)), stream, indent=2)
        for arm in arms:
            print(arm['stage']+'\t'+' '.join(k+'='+v for k,v in sorted(arm['knobs'].items())))
    elif args.action == 'check':
        record = json.loads((args.evidence/'onepass.jsonl').read_text().splitlines()[-1])
        check(args.evidence, args.stage, record, identity(args.repo))
    else:
        print(json.dumps(summarize(args.evidence), indent=2))


if __name__ == '__main__':
    main()
