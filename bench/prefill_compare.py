#!/usr/bin/env python3
"""Validate and compare three fresh-cache onepass arms (B1, A, B2).

Each arm JSON contains revision, knob, enabled, before/after node snapshots,
and priming/measured phases containing canonical `record` and `fresh` JSON.
Node fields: id, started_at, image, args, env, mounts, manifest_sha, model,
hardware. The caller collects these outside the measured request interval.
Invalid evidence produces issues and no performance comparison.
"""
import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean

NODES = {'10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4'}
NODE_FIELDS = ('id', 'started_at', 'image', 'args', 'env', 'mounts',
               'manifest_sha', 'model', 'hardware')
CONTEXTS = [2000, 32000, 128000]
REQUESTS = [(2000, 0), (2000, 1), (2000, 2), (32000, 'all'), (128000, 'all')]


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def request_identity(request):
    return tuple(request.get(k) for k in ('ctx', 'question', 'request_sha256',
                 'prompt_tokens', 'min_tokens', 'max_tokens', 'seed'))


def validate_phase(phase, arm, salts, issues, label):
    record, fresh = phase['record'], phase['fresh']
    head = arm['before']['10.10.10.2']
    if record.get('boot_id') != head['id'] + '|' + head['started_at']:
        issues.append(label + ': boot identity mismatch')
    if record.get('git') != arm['revision'][:7] or record.get('overlay') != head['manifest_sha'][:12]:
        issues.append(label + ': source or overlay mismatch')
    if record.get('quality') != {'ok': 9, 'total': 9} or record.get('korean', {}).get('dirty') != 0 or record.get('korean', {}).get('n') != 5:
        issues.append(label + ': quality or Korean gate failed')
    if record.get('evidence_issues') or record.get('traffic', {}).get('issues') or 'traffic' not in record:
        issues.append(label + ': traffic evidence failed or missing')
    work = record.get('workload', {})
    if work.get('ctx') != CONTEXTS or work.get('require_exclusive') is not True or work.get('fixed_decode_tokens') != 0:
        issues.append(label + ': workload contract mismatch')
    requests, evidence = record.get('requests', []), fresh.get('requests', [])
    if fresh.get('name') != record.get('name') or fresh.get('schema') != 1:
        issues.append(label + ': fresh evidence identity mismatch')
    if [(r.get('ctx'), r.get('question')) for r in requests] != REQUESTS or len(evidence) != 5:
        issues.append(label + ': incomplete request evidence')
        return
    for index, (r, f) in enumerate(zip(requests, evidence)):
        where = f'{label} request {index}'
        salt = f.get('cache_salt')
        if not isinstance(salt, str) or not salt or salt in salts:
            issues.append(where + ': reused or missing cache identity')
        if isinstance(salt, str):
            salts.add(salt)
        if f.get('issues') or f.get('error'):
            issues.append(where + ': request failed')
        if not re.fullmatch('[a-f0-9]{64}', str(r.get('request_sha256'))) or f.get('unsalted_sha256') != r.get('request_sha256'):
            issues.append(where + ': input hash mismatch')
        if not re.fullmatch('[a-f0-9]{64}', str(f.get('wire_sha256'))) or f.get('wire_sha256') == f.get('unsalted_sha256'):
            issues.append(where + ': salted wire hash missing')
        if not positive(r.get('prompt_tokens')) or f.get('prompt_tokens') != r.get('prompt_tokens'):
            issues.append(where + ': token count mismatch')
        if not positive(r.get('ttft_s')) or f.get('ttft_s') != r.get('ttft_s'):
            issues.append(where + ': TTFT missing or inconsistent')
        hits = [f.get(k, {}).get('prefix_hits') for k in ('before', 'after')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in hits) or hits[0] != hits[1]:
            issues.append(where + ': prefix hit or invalid counter')
        for when in ('before', 'after'):
            traffic = f.get(when, {}).get('traffic', {})
            if traffic.get('running') != 0 or traffic.get('waiting') != 0:
                issues.append(where + ': non-idle boundary')


def compare(arms):
    issues, salts = [], set()
    out = dict(schema=1, issues=issues, comparison=[],
        limitations=['Two independent baseline boots and one candidate boot; no confidence interval.',
                     'TTFT includes request processing and first content generation, not isolated kernel time.',
                     'Throughput denotes prompt tokens divided by TTFT, not full-request throughput.',
                     'Priming is excluded. Every measured request has a fresh prefix-cache identity.',
                     'Reduced measurement KV capacity does not establish production-capacity acceptance.',
                     'This bracket does not establish cumulative improvement from the original campaign baseline.'])
    try:
        if len(arms) != 3 or [a['enabled'] for a in arms] != [False, True, False]:
            raise ValueError('require exactly B1, candidate, B2 in order')
        baseline = arms[0]
        knob = baseline['knob']
        if not re.fullmatch(r'VLLM_GLM53_[A-Z0-9_]+', knob):
            raise ValueError('candidate knob is missing')
        if not re.fullmatch('[a-f0-9]{40}', baseline['revision']):
            raise ValueError('full source revision required')
        boot_ids = []
        for index, arm in enumerate(arms):
            label = ('B1', 'A', 'B2')[index]
            if arm['revision'] != baseline['revision'] or arm['knob'] != knob:
                issues.append(label + ': different source or knob')
            if set(arm['before']) != NODES or set(arm['after']) != NODES:
                raise ValueError(label + ': exactly four nodes required')
            for node, before in arm['before'].items():
                after = arm['after'][node]
                reference = baseline['before'][node]
                head = arm['before']['10.10.10.2']
                if any(not before.get(k) for k in NODE_FIELDS):
                    issues.append(label + ': missing node identity ' + node)
                if not re.fullmatch(r'sha256:[a-f0-9]{64}', str(before.get('image'))):
                    issues.append(label + ': immutable image required ' + node)
                for key in ('image', 'mounts', 'manifest_sha'):
                    if before.get(key) != head.get(key):
                        issues.append(label + ': ranks disagree on ' + key + ' at ' + node)
                if any(before.get(k) != after.get(k) for k in NODE_FIELDS):
                    issues.append(label + ': node changed during traffic ' + node)
                for key in ('image', 'args', 'mounts', 'manifest_sha', 'model', 'hardware'):
                    if before.get(key) != reference.get(key):
                        issues.append(label + ': mismatched ' + key + ' on ' + node)
                expected = '1' if arm['enabled'] else '0'
                if before['env'].get(knob) != expected:
                    issues.append(label + ': candidate setting mismatch ' + node)
                other = lambda env: {k:v for k,v in env.items() if k != knob}
                if other(before['env']) != other(reference['env']):
                    issues.append(label + ': another setting changed ' + node)
                if before['args'].get('host') != '127.0.0.1' or before['args'].get('port') != '18000':
                    issues.append(label + ': measurement endpoint is not private ' + node)
                if before['args'].get('max-model-len') != '262144' or before['args'].get('num-gpu-blocks-override') != '415':
                    issues.append(label + ': measurement capacity mismatch ' + node)
            head = arm['before']['10.10.10.2']
            boot_ids.append(head['id'] + '|' + head['started_at'])
            if arm['enabled'] and (set(arm.get('launch_proof', {})) != NODES or
                                   not all(v is True for v in arm['launch_proof'].values())):
                issues.append(label + ': actual candidate launch proof missing on a rank')
            for phase in ('priming', 'measured'):
                validate_phase(arm[phase], arm, salts, issues, label + '/' + phase)
                record = arm[phase]['record']
                reference = baseline['measured']['record']
                for key in ('harness', 'workload', 'endpoint', 'doc_lang', 'thinking'):
                    if record.get(key) is None or record.get(key) != reference.get(key):
                        issues.append(label + '/' + phase + ': mismatched ' + key)
                if [request_identity(r) for r in record['requests']] != [request_identity(r) for r in reference['requests']]:
                    issues.append(label + '/' + phase + ': request bodies or token counts differ')
            if arm['measured']['record'].get('cold_compile'):
                issues.append(label + ': measured phase still marked compile-cold')
        if len(set(boot_ids)) != 3:
            issues.append('bracket does not contain three independent boots')
    except (KeyError, TypeError, ValueError) as exc:
        issues.append('incomplete or malformed evidence: ' + str(exc))
    if issues:
        return out
    for ctx in CONTEXTS:
        grouped = [[r for r in a['measured']['record']['requests'] if r['ctx'] == ctx] for a in arms]
        values = [mean(r['ttft_s'] for r in group) for group in grouped]
        b1, candidate, b2 = values
        base = mean((b1, b2))
        out['comparison'].append(dict(ctx=ctx, mean_ttft_s=dict(B1=b1, A=candidate, B2=b2),
            latency_reduction_pct=100*(1-candidate/base),
            throughput_gain_pct=100*(base/candidate-1),
            throughput_gain_vs_each_baseline_pct=[100*(b/candidate-1) for b in (b1,b2)],
            baseline_spread_pct=100*(max(b1,b2)/min(b1,b2)-1),
            requests=[dict(question=rows[0]['question'], prompt_tokens=rows[0]['prompt_tokens'],
                           ttft_s=[r['ttft_s'] for r in rows]) for rows in zip(*grouped)]))
    out.update(revision=arms[0]['revision'], knob=arms[0]['knob'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('arms', nargs=3, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    result = compare([json.loads(p.read_text()) for p in args.arms])
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 3 if result['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
