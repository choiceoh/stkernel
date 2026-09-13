"""Read-only phase2 consumer assessment using each request's actual token count.

The canonical context summary may pair its last token count with its first
TTFT when questions have different lengths. Never use that aggregate for the
phase2 target. This report keeps request, fixed-decode and C4 rates separate.
Runtime source/device attestation still needs the four ready/final manifests.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench'))
from onepass_recording import steady_errors

TARGETS = {2000: 3300, 128000: 4000}
EXPECTED = Counter({2000: 3, 32000: 1, 128000: 1})
EXPECTED_C4 = Counter({2000: 3, 32000: 1})


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def request_row(value):
    tokens, ttft = value.get('prompt_tokens'), value.get('ttft_s')
    if type(tokens) is not int or tokens <= 0 or type(ttft) not in (int, float) or not math.isfinite(ttft) or ttft <= 0:
        raise ValueError('positive actual prompt tokens and finite positive TTFT are required')
    target = TARGETS.get(value.get('ctx'))
    return dict(ctx=value['ctx'], question=value.get('question'), client=value.get('client'),
                prompt_tokens=tokens, ttft_s=ttft, prefill_tok_s=tokens / ttft,
                target_tok_s=target, target_met=tokens / ttft >= target if target else None,
                completion_tokens=value.get('completion_tokens'),
                decode_tok_s=value.get('decode_tok_s'), output_sha256=value.get('output_sha256'),
                request_sha256=value.get('request_sha256'))


def inspect_run(directory, candidate, index):
    results = jsonl(directory / 'result.jsonl')
    if len(results) != 1:
        raise ValueError('one fresh result is required per invocation')
    rec = results[0]
    run_id = rec.get('run_id', '')
    if not re.fullmatch(r'[A-Za-z0-9-]+', run_id):
        raise ValueError('invalid run id')
    artifacts = directory / 'onepass-runs' / run_id
    issues = []
    def require(ok, text):
        if not ok:
            issues.append(text)
    require(rec.get('arm_sha') == candidate and rec.get('run_index') == index, 'candidate or run-index mismatch')
    require(rec.get('harness') == 44, 'unexpected workload harness')
    require(rec.get('recording', {}).get('status') == 'complete', 'recording incomplete')
    require(rec.get('engine_shape', {}).get('engine') == 'ST' and rec.get('engine_shape', {}).get('model') == 'glm-5.3-flash', 'engine/model mismatch')
    require(rec.get('engine_shape', {}).get('speculative_tokens') == 6, 'speculative width mismatch')
    require(rec.get('kda_state_dtype') == 'fp32', 'FP32 KDA state not established')
    require(rec.get('steady_state', {}).get('valid') is True and rec['steady_state'].get('profile') == 'off', 'steady profiler-off C1 evidence missing')
    issues.extend(rec.get('evidence_issues', ['evidence issue list missing']))
    raw = jsonl(artifacts / 'requests.jsonl')
    require(not any(r.get('concurrency') == 4 and r.get('ctx') in (128000, 131072) for r in raw),
            'excluded C4 128K traffic was submitted')
    raw_by_hash = {r.get('request_sha256'): r for r in raw}
    require(len(raw_by_hash) == len(raw) and None not in raw_by_hash, 'raw request hashes missing or repeated')
    scored, fixed, c4, used = [], [], [], set()
    def check_request(request, phase):
        key = request.get('request_sha256')
        source = raw_by_hash.get(key, {})
        require(key and key not in used, 'scored request missing or duplicated')
        used.add(key)
        require(source.get('phase') == phase, 'request is not from its scored phase')
        require(source.get('output_sha256') == request.get('output_sha256') == hashlib.sha256(source.get('text', '').encode()).hexdigest(), 'raw output hash mismatch')
        require('channels' in source and ''.join(d.get('content') or d.get('reasoning_content') or d.get('reasoning') or '' for d in source.get('channels', [])) == source.get('text'), 'raw channel reconstruction mismatch')
        require(all(source.get(k) == request.get(k) for k in ('prompt_tokens', 'ttft_s', 'completion_tokens', 'cached_tokens')), 'timing/usage differs from raw response')
        require(request.get('cached_tokens') == 0 and request.get('prefix_policy') == 'unique salt', 'cache reuse or fresh-prefix evidence missing')
        require(bool(request.get('first_channels_s')), 'no first-token arrival evidence')
        require(bool(request.get('quality')) and all(q.get('passed') is True for q in request.get('quality', [])), 'request quality failed or missing')
        require(not request.get('corruption'), 'C4 Korean corruption')
        return request_row(request)
    def check_phase(phase, requests, concurrency):
        report = read(artifacts / phase / 'server.json')
        require({r.get('rank') for r in report.get('ranks', [])} == {0, 1, 2, 3}, 'four distinct rank records required')
        require(report.get('boot_id') == rec.get('recording', {}).get('server', {}).get('boot_id'), 'latency boot identity mismatch')
        issues.extend(steady_errors(report, requests, concurrency))
    requests = rec.get('requests', [])
    for request in requests:
        row = check_request(request, 'measure-c1')
        if request.get('fixed_decode'):
            require(request.get('completion_tokens') == request.get('min_tokens') == request.get('max_tokens') == 1024, 'fixed decode did not complete exactly 1024 tokens')
            fixed.append(row)
        else:
            scored.append(row)
    require(Counter(r['ctx'] for r in scored) == EXPECTED, 'C1 context/question coverage mismatch')
    require(len(fixed) == 3, 'three fixed-decode requests required')
    check_phase('measure-c1', requests, 1)
    require(rec.get('korean', {}).get('dirty') == 0 and rec['korean'].get('n') == len(requests), 'C1 Korean coverage failed or missing')
    for group in rec.get('c4', []):
        require(group.get('valid') is True and len(group.get('requests', [])) == 4, 'C4 group invalid')
        phase = group['latency_artifacts']
        require(re.fullmatch(r'measure-c4-[A-Za-z0-9-]+', phase), 'invalid C4 phase')
        rows = [check_request(r, phase) for r in group['requests']]
        check_phase(phase, group['requests'], 4)
        c4.append(dict(ctx=group['ctx'], requests=rows,
                       aggregate_output_tok_s=group.get('aggregate_output_tok_s'), elapsed_s=group.get('elapsed_s')))
    require(Counter(g['ctx'] for g in c4) == (EXPECTED_C4 if index == 1 else Counter()), 'C4 once-per-boot coverage mismatch')
    summaries = []
    for ctx in sorted(EXPECTED):
        rows = [r for r in scored if r['ctx'] == ctx]
        rates = [r['prefill_tok_s'] for r in rows]
        if rows:
            summaries.append(dict(ctx=ctx, samples=len(rows), min_tok_s=min(rates), median_tok_s=median(rates),
                                  first_tok_s=rates[0], all_target_met=all(r['target_met'] for r in rows) if ctx in TARGETS else None))
    return dict(run_index=index, run_id=run_id, candidate=rec.get('arm_sha'), boot_id=rec.get('boot_id'),
                latency_boot_id=rec.get('recording', {}).get('server', {}).get('boot_id'),
                consumer_gates_passed=not issues, issues=issues, prefill=scored,
                context_summary=summaries, fixed_decode=fixed, c4=c4,
                throughput_targets_met=all(any(r['ctx'] == ctx for r in scored) and all(r['target_met'] for r in scored if r['ctx'] == ctx) for ctx in TARGETS))


def assess(arm, candidate):
    runs, issues = [], []
    for index in (1, 2):
        try:
            runs.append(inspect_run(arm / str(index), candidate, index))
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f'run {index}: {error}')
    if len(runs) == 2:
        for key in ('boot_id', 'latency_boot_id'):
            if not runs[0][key] or runs[0][key] != runs[1][key]:
                issues.append('C1 repeats do not share the same ' + key)
    return dict(candidate=candidate, target_tok_s=TARGETS, runs=runs, issues=issues,
                consumer_target_passed=len(runs) == 2 and not issues and all(r['consumer_gates_passed'] and r['throughput_targets_met'] for r in runs),
                runtime_attestation='separately verify all four ready/final source manifests and GPU identity',
                comparison='absolute per-request tokens / TTFT; no baseline speedup or decode nonregression claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', type=Path)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.candidate):
        parser.error('candidate must be the exact 40-character admitted commit')
    print(json.dumps(assess(args.arm, args.candidate), indent=2, ensure_ascii=False))
