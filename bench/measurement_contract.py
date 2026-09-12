# SPDX-License-Identifier: Apache-2.0
"""Shared onepass workload identity and objective selection. No device imports."""
import json
import math

HARNESS = 43
# Harness 42 exhausted all three individual reasoning streams at 800 tokens,
# before their calculations finished. Keep half the larger budget for answers.
MAX_TOKENS = 8192
COMBINED_MAX_TOKENS = 24576
COMBINED_REASONING_BUDGET = 12288
DEFAULTS = dict(ctx=[2000, 32000, 128000], seed=7, max_tokens=MAX_TOKENS, combine_min_ctx=32000,
                fixed_decode_tokens=0, fixed_decode_reps=0, require_exclusive=False)


def workload(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("unknown workload fields")
    value = dict(DEFAULTS, **raw)
    if (not isinstance(value['ctx'], list) or not value['ctx'] or len(value['ctx']) > 8
            or any(type(c) is not int or not 1 <= c <= 1000000 for c in value['ctx'])
            or len(set(value['ctx'])) != len(value['ctx'])):
        raise ValueError("ctx must list distinct positive context sizes")
    for key in ('seed', 'max_tokens', 'combine_min_ctx', 'fixed_decode_tokens', 'fixed_decode_reps'):
        if type(value[key]) is not int or not 0 <= value[key] <= 1000000:
            raise ValueError("invalid workload " + key)
    if value['max_tokens'] < 1 or type(value['require_exclusive']) is not bool:
        raise ValueError("invalid token budget or exclusivity")
    if value['fixed_decode_tokens']:
        if not 1 <= value['fixed_decode_reps'] <= 20:
            raise ValueError("fixed decode needs 1..20 repetitions")
    else:
        value['fixed_decode_reps'] = 0
    value['ctx'] = list(value['ctx'])
    return value


def from_args(args):
    return workload(dict(ctx=[int(c) for c in args.ctx.split(',')], seed=args.seed,
                         max_tokens=args.max_tokens, combine_min_ctx=args.combine_min_ctx,
                         fixed_decode_tokens=args.fixed_decode_tokens,
                         fixed_decode_reps=args.fixed_decode_reps, require_exclusive=args.require_exclusive))


def metadata(value=None):
    return dict(harness=HARNESS, doc_lang='ko', thinking=True, window_s=1.0, workload=workload(value))


def objective(raw=None):
    raw = {'metric': 'decode_steps'} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - {'metric', 'ctx'}:
        raise ValueError("objective requires metric and optional ctx")
    metric = raw.get('metric')
    if metric not in {'decode_steps', 'decode_tokens', 'prefill_ttft', 'quality'}:
        raise ValueError("objective metric must be decode_steps, decode_tokens, prefill_ttft or quality")
    if metric == 'prefill_ttft':
        if type(raw.get('ctx')) is not int or raw['ctx'] < 1:
            raise ValueError("prefill_ttft requires a positive ctx")
    elif 'ctx' in raw:
        raise ValueError("ctx only applies to prefill_ttft")
    return dict(raw)


def evaluations(spec):
    values = spec.get('evaluations')
    if values is None:
        return [dict(objective=objective(), workload=workload({'ctx': [int(c) for c in
                    spec.get('env', {}).get('QUALITY_CTX', '2000,32000,128000').split(',')]}))]
    if not isinstance(values, list) or not 1 <= len(values) <= 6:
        raise ValueError("evaluations must contain 1..6 workloads on the same serving configuration")
    result = []
    for item in values:
        if not isinstance(item, dict) or set(item) - {'objective', 'workload'}:
            raise ValueError("evaluation requires objective/workload")
        obj, work = objective(item.get('objective')), workload(item.get('workload'))
        if obj['metric'] == 'prefill_ttft' and obj['ctx'] not in work['ctx']:
            raise ValueError("prefill objective context is not in the workload")
        if obj['metric'] == 'decode_tokens' and (not work['fixed_decode_tokens'] or not work['require_exclusive']):
            raise ValueError("decode_tokens requires fixed length requests and exclusive traffic")
        result.append(dict(objective=obj, workload=work))
    if len({json.dumps(v, sort_keys=True) for v in result}) != len(result):
        raise ValueError("duplicate evaluations")
    return result


def environment(evaluation):
    return {'FLEET_WORKLOAD': json.dumps(evaluation['workload']),
            'FLEET_OBJECTIVE': json.dumps(evaluation['objective'])}


def metric_value(record, obj):
    metric = obj['metric']
    if metric == 'decode_steps':
        value = (record.get('decode') or {}).get('windows_med')
    elif metric == 'decode_tokens':
        requests = [r for r in record.get('requests', []) if r.get('fixed_decode')]
        work = record.get('workload') or {}
        if len(requests) != work.get('fixed_decode_reps') or not requests:
            return None
        if any(r.get('completion_tokens') != work.get('fixed_decode_tokens') or
               type(r.get('decode_s')) not in (int, float) or not math.isfinite(r['decode_s']) or
               r['decode_s'] <= 0 for r in requests):
            return None
        value = sum(r['completion_tokens'] - 1 for r in requests) / sum(r['decode_s'] for r in requests)
    elif metric == 'prefill_ttft':
        rows = [r for r in record.get('prefill', []) if r.get('ctx') == obj['ctx']]
        value = rows[0].get('cold_s') if len(rows) == 1 else None
    else:
        return None
    return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def metric_compatible(a, b, obj):
    # Compile-cold TTFT and an already compiled serving process answer different
    # questions. The planned prefill gate collects steady-compile samples.
    return obj['metric'] != 'prefill_ttft' or bool(a.get('cold_compile')) == bool(b.get('cold_compile'))
