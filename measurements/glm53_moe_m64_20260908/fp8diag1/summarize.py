"""Summarize the frozen diagnostic; descriptive counts never approve serving."""
import gzip
import json
from pathlib import Path
import statistics
import struct


def float32_steps(a, b):
    """Positive finite metric values only; descriptive, never a changed limit."""
    return struct.unpack('I', struct.pack('f', a))[0] - struct.unpack('I', struct.pack('f', b))[0]


def summarize(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        records = [json.loads(line) for line in stream if line.startswith('{')]
    trials = [r for r in records if r.get('kind') == 'MOE_M64_FP8_DIAGNOSTIC_TRIAL']
    finals = [r for r in records if r.get('verdict') == 'MOE_M64_FP8_DIAGNOSTIC_COMPLETE']
    assert len(finals) == 1 and len(trials) == finals[0]['trials'] == 72
    assert finals[0]['serving_gate'] is False and finals[0]['numerical_acceptance'] is False
    groups = []
    keys = list(dict.fromkeys((r['rows'], r['skew'], r['seed']) for r in trials))
    for rows, skew, seed in keys:
        selected = [r for r in trials if (r['rows'], r['skew'], r['seed']) == (rows, skew, seed)]
        assert [r['trial'] for r in selected] == list(range(8))
        for phase in ('transport', 'local'):
            counts = []
            for record in selected:
                pairs = record[phase]
                assert [p['rank'] for p in pairs] == list(range(4))
                counts.append(dict(trial=record['trial'], order=record['order'],
                    **{k: sum(p[k] for p in pairs) for k in (
                        'candidate_bad', 'control_bad', 'candidate_only', 'control_only', 'both_bad')},
                    finite=all(p['finite'] for p in pairs)))
            groups.append(dict(rows=rows, skew=skew, seed=seed, phase=phase, trials=counts,
                candidate_bad=sum(c['candidate_bad'] for c in counts),
                control_bad=sum(c['control_bad'] for c in counts),
                candidate_failed_trials=sum(c['candidate_bad'] > 0 for c in counts),
                control_failed_trials=sum(c['control_bad'] > 0 for c in counts),
                excess_by_trial=[c['candidate_bad']-c['control_bad'] for c in counts],
                median_excess=statistics.median(c['candidate_bad']-c['control_bad'] for c in counts),
                error_details={arm: dict(
                    l2_failures=sum(v['error_l2'] > v['limit_l2'] for v in values),
                    peak_failures=sum(v['error_peak'] > v['limit_peak'] for v in values),
                    peak_one_float32_step_above_limit=sum(
                        float32_steps(v['error_peak'], v['limit_peak']) == 1 for v in values),
                    max_l2=max((v['error_l2'] for v in values), default=0),
                    max_peak=max((v['error_peak'] for v in values), default=0),
                ) for arm in ('candidate', 'control')
                    for values in [[failure[arm] for r in selected for pair in r[phase]
                        for failure in pair['failures'] if failure[arm+'_bad']]]},
                finite=all(c['finite'] for c in counts)))
    return dict(source=path.name, serving_gate=False, numerical_acceptance=False,
        note='Descriptive row-trial counts only; rows and repeated trials are not independent experiments.',
        diagnostic=finals[0], groups=groups)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    output = summarize(root/'fp8-v3-rank-0.log.gz')
    (root/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    for group in output['groups']:
        print(group['rows'], 'skew' if group['skew'] else 'balanced', group['seed'], group['phase'],
              'candidate/control', group['candidate_bad'], group['control_bad'],
              'failed trials', group['candidate_failed_trials'], group['control_failed_trials'],
              'excess', group['excess_by_trial'])
