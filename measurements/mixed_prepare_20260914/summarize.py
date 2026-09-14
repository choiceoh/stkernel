"""Rebuild the same-build preparation table from complete unprofiled samples."""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
report = json.loads((root / 'gpu.json').read_text())
assert report['status'] == 'PASS' and report['compare_preparation']
assert report['value_check_gate']['status'] == 'PASS'
assert report['value_check_gate']['cases'] == 260
assert len(report['cases']) == 4
summary = dict(source_revision='153b6f8bdbb9ebf18af78c79bb9173f96a6769c2',
    status=report['status'], total_samples=sum(len(c['samples']) for c in report['cases']),
    peak_torch_allocation_gib=report['scratch_peak_bytes'] / (1 << 30),
    value_check_cases=report['value_check_gate']['cases'], cells=[])
for case in report['cases']:
    for quota in (0, 128):
        cell = dict(decode_rows=case['decode_rows'], prefill_rows=case['prefill_rows'],
                    hot_quota=quota, arms={})
        for arm in ('packed_v1', 'packed_v2'):
            samples = [s for s in case['samples'] if s['planning_arm'] == arm and s['hot_quota'] == quota]
            warm = [s for s in samples if not s['includes_first_use_compile']]
            assert len(samples) == 4 and len(warm) == 3
            cell['arms'][arm] = dict(warm_samples=len(warm),
                median_prepare_admit_ms=median(s['prepare_admit_wall_ms'] for s in warm),
                median_decode_ready_ms=median(s['decode_ready_wall_ms'] for s in warm),
                median_complete_ms=median(s['prefill_complete_wall_ms'] for s in warm),
                median_decode_after_prepare_ms=median(s['decode_ready_wall_ms'] - s['prepare_admit_wall_ms'] for s in warm),
                max_decode_relative_max=max(s['errors']['decode']['relative_max'] for s in samples),
                max_decode_relative_rms=max(s['errors']['decode']['relative_rms'] for s in samples),
                max_prefill_relative_max=max(s['errors']['prefill']['relative_max'] for s in samples),
                max_prefill_relative_rms=max(s['errors']['prefill']['relative_rms'] for s in samples),
                median_stages_ms={field: median(s['planning_stages_ms'][field] for s in warm)
                                 for field in sorted(warm[0]['planning_stages_ms'])})
        old, new = cell['arms']['packed_v1'], cell['arms']['packed_v2']
        for field in ('prepare_admit', 'decode_ready', 'complete'):
            key = 'median_' + field + '_ms'
            cell[field + '_reduction_percent'] = 100 * (1 - new[key] / old[key])
        summary['cells'].append(cell)
assert summary['total_samples'] == 64
(root / 'gpu_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
