# SPDX-License-Identifier: Apache-2.0
"""Shared onepass workload identity and objective selection. No device imports."""
import json
import math

HARNESS = 46
# Harness 46: onepass has two named workloads (`PROFILES`) and `default` is the cheap one -- two
# contexts and no C=N arm. The full set is `extended`, asked for by name. Two things made this
# necessary at once: a D17 probe reserves the live door for its whole run and answers 409 to every
# other request (2026-09-16, it reached a user as `API error 409`), and `fixed_concurrency_tokens`
# was NOT part of the recorded workload, so two records could differ in whether they ran the C=N arm
# and nothing said so. It is in the identity now, and `st_judge` compares like with like.
#
# What this costs, plainly: no record written before harness 46 carries a profile, and the judge will
# not read one against the other. The first bracket of each commit after this boots its own base
# instead of reusing a probe's sample, until samples accumulate in the new profile. That is what a
# harness number is for -- a measurement that changed is a new generation, not a continuation.
#
# Harness 45: onepass requests say `retain: false`, so the server no longer parks each finished
# request to the NVMe tier (the park overlapped the next request; on the live door a D17 probe filled
# production's tier). The questions and budgets are harness 44's.
# Harness 43 (8192/4096 individual, 24576/12288 combined) ended every measured
# reasoning stream at its cap mid-sentence; the 128K combined request spent the
# whole cap on the first of its three cases. Keep half the doubled budget for answers.
MAX_TOKENS = 16384
COMBINED_MAX_TOKENS = 49152
COMBINED_REASONING_BUDGET = 24576
DEFAULTS = dict(ctx=[2000, 32000, 128000], seed=7, max_tokens=MAX_TOKENS, combine_min_ctx=32000,
                fixed_decode_tokens=0, fixed_decode_reps=0, fixed_concurrency_tokens=1024,
                require_exclusive=False)

# The two workloads a run can be. `default` is what every routine measurement uses -- the D17 probe
# after a deploy and both arms of a bracket -- so base and candidate always measure the same thing.
# `extended` is the full set, asked for by name when the question needs 128K or the C=N multiplier.
PROFILES = {
    "default": dict(ctx=[2000, 32000], fixed_concurrency_tokens=0),
    "extended": dict(ctx=[2000, 32000, 128000], fixed_concurrency_tokens=1024),
}
DEFAULT_PROFILE = "default"


def profile(name=None):
    """A named workload, complete. An unknown name is a refusal, not a silent default."""
    name = DEFAULT_PROFILE if name in (None, "") else str(name)
    if name not in PROFILES:
        raise ValueError(f"unknown workload profile {name!r}: " + ", ".join(sorted(PROFILES)))
    return workload(dict(PROFILES[name]))


def profile_of(value):
    """The profile this workload IS, or 'custom' -- what a record is compared within."""
    value = workload(value)
    for name in PROFILES:
        if profile(name) == value:
            return name
    return "custom"


def workload(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("unknown workload fields")
    value = dict(DEFAULTS, **raw)
    if (not isinstance(value['ctx'], list) or not value['ctx'] or len(value['ctx']) > 8
            or any(type(c) is not int or not 1 <= c <= 1000000 for c in value['ctx'])
            or len(set(value['ctx'])) != len(value['ctx'])):
        raise ValueError("ctx must list distinct positive context sizes")
    for key in ('seed', 'max_tokens', 'combine_min_ctx', 'fixed_decode_tokens', 'fixed_decode_reps',
                'fixed_concurrency_tokens'):
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
                         fixed_decode_reps=args.fixed_decode_reps,
                         fixed_concurrency_tokens=getattr(args, 'fixed_concurrency_tokens', 0),
                         require_exclusive=args.require_exclusive))


def metadata(value=None):
    value = workload(value)
    return dict(harness=HARNESS, doc_lang='ko', thinking=True, window_s=1.0,
                workload=value, workload_profile=profile_of(value))


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
