#!/usr/bin/env python3
"""Read only saved canonical onepass prefixes; no service or device access.

Print Markdown by default, or the same data as JSON with --json. Input records
remain unchanged. Source identity here is the record's short SHA/stamp, not a
replacement for the separate frozen-source and four-node container receipts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

REV = 'f6b0934eb3d14b46cc58c29f6c9983f776eed250'
SESSION = 'eptiledsf60909v1'
OVERLAY = '42e916027e1c'
ARMS = ('EPTILEDSF6B0', 'EPTILEDSF6B1', 'EPTILEDSF6A', 'EPTILEDSF6ABASE')
EXPECTED = {'VLLM_GLM53_EP_TILED': '1', 'VLLM_GLM53_TP_SF6_Q0': '0'}
WORK = dict(ctx=[2000, 32000, 128000], seed=7, max_tokens=400,
            combine_min_ctx=32000, fixed_decode_tokens=1024,
            fixed_decode_reps=3, require_exclusive=True)
ENDPOINT = dict(completion='http://127.0.0.1:18000/v1/chat/completions',
                metrics='http://127.0.0.1:18000/metrics')


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('missing/nonpositive timing or count')
    return value


def load_prefix(path, expected_sha256=None):
    if expected_sha256 is not None and (path.is_dir() or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None):
        raise ValueError('--sha256 requires one explicit file and a full SHA256')
    paths = sorted(path.glob('records-*.jsonl')) if path.is_dir() else [path]
    captured = []
    for item in paths:
        raw = item.read_bytes()
        if not raw or len(raw) > 32 * 1024 * 1024 or not raw.endswith(b'\n'):
            raise ValueError('not a bounded complete record prefix: ' + str(item))
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None:
            if digest != expected_sha256:
                raise ValueError('explicit original file SHA mismatch: ' + str(item))
        elif item.name != 'records-' + digest + '.jsonl':
            raise ValueError('observer filename SHA mismatch: ' + str(item))
        captured.append((raw, item, digest))
    if not captured:
        raise ValueError('no saved canonical record prefix yet')
    raw, selected, digest = max(captured, key=lambda entry: len(entry[0]))
    if any(not raw.startswith(older) for older, _, _ in captured):
        raise ValueError('saved prefixes diverge; no automatic selection')
    rows = [json.loads(line, object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            for line in raw.splitlines()]
    return rows, dict(path=str(selected), bytes=len(raw), sha256=digest)


def summarize(rows, source):
    output, request_identity, common, seen, boots = [], None, None, set(), set()
    for rec in rows:
        name = rec.get('name')
        if name not in ARMS or name in seen:
            raise ValueError('unknown or repeated arm: ' + str(name))
        seen.add(name)
        if (rec.get('session') != SESSION or rec.get('overlay') != OVERLAY
                or rec.get('git') not in (REV, REV[:8]) or rec.get('rehearsal')
                or rec.get('workload') != WORK or rec.get('endpoint') != ENDPOINT
                or rec.get('knobs') != (EXPECTED if name == ARMS[2] else {})):
            raise ValueError('record source/session/workload/knob identity differs: ' + name)
        boot = rec.get('boot_id')
        if not isinstance(boot, str) or re.fullmatch(r'[0-9a-f]{64}\|[^|]+', boot) is None or boot in boots:
            raise ValueError('missing or reused boot identity')
        boots.add(boot)
        identity = {k: rec.get(k) for k in ('harness', 'doc_lang', 'thinking', 'runtime')}
        if identity['harness'] != 40 or identity['doc_lang'] != 'ko' or identity['thinking'] is not True:
            raise ValueError('unexpected harness/language/thinking')
        if common is not None and common != identity:
            raise ValueError('cross-arm runtime or harness differs')
        common = identity
        requests = rec.get('requests', [])
        expected_order = [(2000, 0, False, None), (2000, 1, False, None), (2000, 2, False, None),
                          (32000, 'all', False, None), (128000, 'all', False, None)]
        expected_order += [(2000, 'fixed-all', True, rep) for rep in range(3)]
        order = [(r.get('ctx'), r.get('question'), bool(r.get('fixed_decode')), r.get('rep')) for r in requests]
        if order != expected_order:
            raise ValueError('request order/count differs: ' + name)
        signatures = []
        for request in requests:
            for field in ('request_sha256', 'output_sha256'):
                if re.fullmatch(r'[0-9a-f]{64}', str(request.get(field))) is None:
                    raise ValueError('missing request/output hash')
            tokens = request.get('prompt_tokens')
            if type(tokens) is not int or tokens <= 0:
                raise ValueError('missing actual prompt tokens')
            signatures.append((request['request_sha256'], tokens))
        if request_identity is not None and signatures != request_identity:
            raise ValueError('cross-arm request bytes or actual prompt tokens differ')
        request_identity = signatures
        fixed = [r for r in requests if r.get('fixed_decode')]
        if any(r.get('completion_tokens') != 1024 for r in fixed):
            raise ValueError('fixed request failed its actual 1024-token contract')
        duration = sum(number(r.get('decode_s')) for r in fixed)
        count = sum(r['completion_tokens'] - 1 for r in fixed)
        errors = []
        quality, korean = rec.get('quality', {}), rec.get('korean', {})
        if quality != dict(ok=18, total=18): errors.append('quality')
        if korean.get('dirty') != 0 or korean.get('n') != 8: errors.append('Korean')
        active = [k for k, v in rec['knobs'].items() if v not in ('0', '', 'off')]
        if any((rec.get('proof') or {}).get(k) is not True for k in active): errors.append('proof')
        if rec.get('traffic', {}).get('issues') != [] or rec.get('evidence_issues'): errors.append('traffic/evidence')
        try: number(rec.get('decode', {}).get('windows_med'))
        except ValueError: errors.append('decode windows')
        prefill = []
        for ctx in WORK['ctx']:
            samples = [r for r in requests if not r.get('fixed_decode') and r['ctx'] == ctx]
            prefill.append(dict(ctx=ctx, samples=[dict(prompt_tokens=r['prompt_tokens'],
                ttft_s=number(r.get('ttft_s')), tok_s=r['prompt_tokens'] / number(r.get('ttft_s')),
                request_sha256=r['request_sha256']) for r in samples]))
        output.append(dict(name=name, cold_compile=bool(rec.get('cold_compile')), knobs=rec['knobs'],
            quality=quality, korean=korean, proof=rec.get('proof'), gate_errors=errors,
            valid_for_comparison=not errors, fixed_decode_tok_s=count / duration,
            fixed_decode_numerator=count, fixed_decode_seconds=duration,
            fixed_decode_reps=[(r['completion_tokens']-1)/number(r['decode_s']) for r in fixed],
            prefill=prefill, recorded_prefill=rec.get('prefill'),
            offending_requests=[dict(rep=r.get('rep'), ctx=r['ctx'],
                channel_diagnostics=r['channel_diagnostics']) for r in requests
                if any(r.get('channel_diagnostics', {}).get('combined_gated_counts', {}).values())]))
    candidate = next((r for r in output if r['name'] == ARMS[2]), None)
    baselines = [r for r in output if r['name'] != ARMS[2] and r['valid_for_comparison']]
    comparison = dict(status='pending candidate' if candidate is None else 'unresolved', acceptance=False,
                      valid_baselines=[r['name'] for r in baselines], excluded_baselines=[r['name'] for r in output
                      if r['name'] != ARMS[2] and not r['valid_for_comparison']])
    if candidate and candidate['valid_for_comparison'] and baselines:
        pooled = sum(r['fixed_decode_numerator'] for r in baselines) / sum(r['fixed_decode_seconds'] for r in baselines)
        comparison.update(baseline_pooled_decode_tok_s=pooled,
                          decode_delta_pct=100*(candidate['fixed_decode_tok_s']/pooled-1))
        comparison['prefill'] = [dict(baseline=b['name'], ctx=ctx,
            ttft_reduction_pct=100*(1-c['samples'][0]['ttft_s']/p['samples'][0]['ttft_s']))
            for b in baselines if b['cold_compile'] == candidate['cold_compile']
            for ctx, p, c in zip(WORK['ctx'], b['prefill'], candidate['prefill'])]
    return dict(source=source, expected_revision=REV, rows=output, comparison=comparison,
                limits=['Record source SHA/stamp require separate full-source/container attestation.',
                        'Invalid baselines never enter ratios. Cold B0 remains eligible for decode if valid.',
                        '32K/128K are one HTTP request per arm, not independent warm-request samples.',
                        '2K per-request tok/s uses actual prompt tokens; canonical recorded_prefill is retained unchanged.',
                        'No statistical acceptance or gate rewrite is performed.'])


def markdown(result):
    print('원본: `' + result['source']['path'] + '` / SHA256 `' + result['source']['sha256'] + '`')
    print('\n| Arm | Compile cold | Quality | Korean | Gate | Fixed rep0/1/2 tok/s | Pooled tok/s |')
    print('|---|---|---|---|---|---|---|')
    for row in result['rows']:
        q, k = row['quality'], row['korean']
        print(f"| {row['name']} | {row['cold_compile']} | {q.get('ok')}/{q.get('total')} | {k.get('dirty')}/{k.get('n')} | "
              + (','.join(row['gate_errors'])+' FAIL (참고값)' if row['gate_errors'] else 'PASS')
              + ' | ' + ' / '.join(f'{v:.4f}' for v in row['fixed_decode_reps']) + f" | {row['fixed_decode_tok_s']:.6f} |")
    print('\n| Arm | Context | 실제 prompt tokens | TTFT s (각 요청) | 실제 tokens / TTFT |')
    print('|---|---|---|---|---|')
    for row in result['rows']:
        for p in row['prefill']:
            print(f"| {row['name']} | {p['ctx']} | " + ' / '.join(str(s['prompt_tokens']) for s in p['samples'])
                  + ' | ' + ' / '.join(f"{s['ttft_s']:.6f}" for s in p['samples'])
                  + ' | ' + ' / '.join(f"{s['tok_s']:.2f}" for s in p['samples']) + ' |')
    print('\n비교: `' + json.dumps(result['comparison'], ensure_ascii=False, sort_keys=True) + '`')
    print('\n판정 실패 기준선은 위 표의 참고 수치만 보존하며 비교 기준으로 사용하지 않습니다. 정식 judge 원본을 별도로 확인해야 합니다.')
    for limit in result['limits']: print('- ' + limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path, default=Path('/tmp/glm53-ep-tiled-sf6-onepass1-streams'))
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--sha256', help='original SHA256 for an explicitly named final record copy')
    args = parser.parse_args()
    rows, source = load_prefix(args.input, args.sha256)
    result = summarize(rows, source)
    if args.json: print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else: markdown(result)


if __name__ == '__main__':
    main()
