"""Rebuild the warm, per-quota GPU comparison directly from its raw samples."""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
report = json.loads((root/'gpu.json').read_text())
assert report['status'] == 'PASS' and report['compare_planning']
summary = dict(source_revision='c33370f9008f13c880f488e50a55f83f05091b37', status=report['status'],
    total_samples=sum(len(c['samples']) for c in report['cases']),
    peak_torch_allocation_gib=report['scratch_peak_bytes']/(1 << 30), cells=[])
for case in report['cases']:
    for quota in (0, 128):
        cell = dict(decode_rows=case['decode_rows'], prefill_rows=case['prefill_rows'], hot_quota=quota, arms={})
        for arm in ('legacy', 'packed'):
            all_samples = [s for s in case['samples'] if s['planning_arm'] == arm and s['hot_quota'] == quota]
            warm = [s for s in all_samples if not s['includes_first_use_compile']]
            assert len(all_samples) == 4 and len(warm) == 3
            cell['arms'][arm] = dict(warm_samples=len(warm),
                median_prepare_admit_ms=median(s['prepare_admit_wall_ms'] for s in warm),
                median_decode_ready_ms=median(s['decode_ready_wall_ms'] for s in warm),
                median_complete_ms=median(s['prefill_complete_wall_ms'] for s in warm),
                median_decode_after_prepare_ms=median(s['decode_ready_wall_ms']-s['prepare_admit_wall_ms'] for s in warm),
                max_decode_relative_max=max(s['errors']['decode']['relative_max'] for s in all_samples),
                max_prefill_relative_max=max(s['errors']['prefill']['relative_max'] for s in all_samples),
                max_prefill_relative_rms=max(s['errors']['prefill']['relative_rms'] for s in all_samples),
                median_stages_ms={field: median(s['planning_stages_ms'][field] for s in warm)
                                 for field in ('hot_plan_ms', 'cold_plan_ms', 'agreement_ms')})
        old, new = cell['arms']['legacy'], cell['arms']['packed']
        cell['prepare_reduction_percent'] = 100 * (1-new['median_prepare_admit_ms']/old['median_prepare_admit_ms'])
        summary['cells'].append(cell)
(root/'gpu_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
