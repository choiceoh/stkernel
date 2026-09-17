"""Preserve the consumer gates and show ordinary and fixed decode separately."""
import json
from pathlib import Path
import runpy
import statistics
import sys

prior = runpy.run_path(str(Path(__file__).resolve().parents[1]
    / 'st_fixed_k_cost_followup_20260917/summarize.py'))


def normal_windows(record):
    decode = record.get('decode') or {}
    by_context = {str(ctx): list(rates)
                  for ctx, rates in decode.get('windows_by_ctx', {}).items()}
    fixed = decode.get('fixed_intervals', [])
    if fixed:
        # onepass measures all ordinary contexts, then appends the fixed 2K
        # requests. Those windows also occur at the end of the 2K bucket.
        # Refuse an unfamiliar ordering instead of silently mixing workloads.
        rates = [item['steps'] / item['seconds'] for item in fixed]
        if by_context.get('2000', [])[-len(rates):] != rates:
            raise ValueError('fixed windows are not the trailing 2K windows')
        by_context['2000'] = by_context['2000'][:-len(rates)]
    return {ctx: dict(windows=len(rates), mean_step_s=statistics.mean(rates),
                      median_step_s=statistics.median(rates),
                      equivalent_ms_per_step=1000 / statistics.mean(rates))
            for ctx, rates in by_context.items() if rates}


def fixed_steps(record):
    intervals = (record.get('decode') or {}).get('fixed_intervals', [])
    steps = sum(item['steps'] for item in intervals)
    seconds = sum(item['seconds'] for item in intervals)
    return dict(windows=len(intervals), steps=steps, seconds=seconds,
                pooled_step_s=steps / seconds if seconds else None,
                pooled_ms_per_step=1000 * seconds / steps if steps else None)


def main(path):
    report = prior['main'](path)
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    indexed = {(record['name'], record['run_index']): record for record in records}
    for row in report['runs']:
        record = indexed[row['arm'], row['run']]
        row['normal_step_windows'] = normal_windows(record)
        row['fixed_step_cost'] = fixed_steps(record)
    for pair in report['pairs']:
        br, ar = indexed['B', pair['run']], indexed['A', pair['run']]
        base = normal_windows(br)
        candidate = normal_windows(ar)
        if base.keys() != candidate.keys():
            raise ValueError('ordinary decode context coverage differs')
        pair['normal_step_windows'] = {
            ctx: dict(base=base[ctx], candidate=candidate[ctx],
                      mean_change_pct=100 * (candidate[ctx]['mean_step_s']
                                             / base[ctx]['mean_step_s'] - 1),
                      equivalent_cost_reduction_pct=100 * (1 - base[ctx]['mean_step_s']
                                                          / candidate[ctx]['mean_step_s']))
            for ctx in base}
        bf, af = fixed_steps(br), fixed_steps(ar)
        pair['fixed_step_cost'] = dict(
            base=bf, candidate=af,
            reduction_ms=(bf['pooled_ms_per_step'] - af['pooled_ms_per_step'])
                if bf['steps'] and af['steps'] else None,
            reduction_pct=100 * (1 - af['pooled_ms_per_step'] / bf['pooled_ms_per_step'])
                if bf['steps'] and af['steps'] else None)
        ba, aa = br['decode'].get('acc_raw'), ar['decode'].get('acc_raw')
        pair['c1_acceptance'] = dict(
            base=ba, candidate=aa,
            change_pp=100 * (aa - ba) if ba is not None and aa is not None else None,
            scope='C1 ordinary and fixed requests together; not C2 acceptance')
    report['normal_step_scope'] = (
        'Arithmetic mean and median of the existing positive one-second decode windows, '
        'excluding the fixed-output windows; output lengths and hashes may differ. '
        'The legacy ordinary window series omits zero-step stalls. '
        'Equivalent ms/step is the reciprocal of this window mean, not mean measured latency. '
        'This is not a pooled rate or an isolated kernel speedup.')
    return report


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1]), ensure_ascii=False, indent=2))
