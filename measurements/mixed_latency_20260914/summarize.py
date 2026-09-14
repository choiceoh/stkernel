"""Summarize each frozen native/mixed run separately; never splice baselines."""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
manifest = json.loads((root/'sources.json').read_text())
summary = dict(target='32K fresh complete FFN <= 52 ms; decode must also beat native for adoption', runs=[])
for record in manifest['gpu_runs']:
    report = json.loads((root/record['file']).read_text())
    row = dict(reservation=record['reservation'], revision=record['revision'], status=report['status'], cells=[])
    if report['status'] != 'PASS':
        row['error'] = report.get('error')
        summary['runs'].append(row)
        continue
    assert report['value_check_gate']['cases'] == 260
    assert len(report['cases']) == 4
    row.update(peak_allocated_gib=report['scratch_peak_bytes']/(1 << 30),
               mixed_samples=sum(len(c['samples']) for c in report['cases']),
               native_samples=sum(len(c['native_samples']) for c in report['cases']))
    for case in report['cases']:
        native = [s for s in case['native_samples'] if not s['includes_first_use_compile']]
        for quota in (0, 128):
            samples = [s for s in case['samples'] if s['hot_quota']==quota]
            warm = [s for s in samples if not s['includes_first_use_compile']]
            assert len(warm) == len(native)
            cell = dict(decode_rows=case['decode_rows'], prefill_rows=case['prefill_rows'],
                hot_quota=quota, warm_samples=len(warm),
                native_decode_ms=median(s['decode_ready_wall_ms'] for s in native),
                native_complete_ms=median(s['prefill_complete_wall_ms'] for s in native),
                native_complete_range_ms=[min(s['prefill_complete_wall_ms'] for s in native),
                                          max(s['prefill_complete_wall_ms'] for s in native)],
                prepare_ms=median(s['prepare_admit_wall_ms'] for s in warm),
                decode_ms=median(s['decode_ready_wall_ms'] for s in warm),
                complete_ms=median(s['prefill_complete_wall_ms'] for s in warm),
                complete_range_ms=[min(s['prefill_complete_wall_ms'] for s in warm),
                                   max(s['prefill_complete_wall_ms'] for s in warm)],
                paired_complete_ratio_median=median(s['prefill_complete_wall_ms']/b['prefill_complete_wall_ms']
                                                    for s, b in zip(warm, native)),
                stages_ms={k: median(s['planning_stages_ms'][k] for s in warm)
                           for k in warm[0]['planning_stages_ms']},
                maximum_errors={output: {field: max(s['errors'][output][field] for s in samples)
                    for field in ('relative_max', 'relative_rms')} for output in ('decode', 'prefill')})
            cell['target_52_ms_met'] = cell['complete_ms'] <= 52
            cell['adoption_latency_gate_met'] = (cell['target_52_ms_met']
                and cell['decode_ms'] <= cell['native_decode_ms']
                and cell['complete_ms'] < cell['native_complete_ms'])
            row['cells'].append(cell)
    summary['runs'].append(row)
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
