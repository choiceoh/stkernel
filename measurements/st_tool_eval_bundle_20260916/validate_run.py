"""Validate the complete official reference-version run without changing grades."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument('raw', type=Path)
a = ap.parse_args()
result = json.loads((a.raw / 'result.json').read_text())
manifest = json.loads((a.raw / 'status.json').read_text())
scores, config = result['scores'], result['config']
rows = scores['scenario_results']
expected = {f'TC-{i:02}' for i in range(1, 89)}
assert result['tool_eval_bench_version'] == '2.6.1.dev65+g6be685f0e'
assert manifest['cli_commit'] == '6be685f0e6b9e0df05ed024848cf7fe1eca48752'
assert manifest['status'] == 'complete' and manifest['exit_code'] == 0
assert manifest['protocol'] == 'reference_t1'
assert len(rows) == 88 and {r['scenario_id'] for r in rows} == expected
assert config['scenario_count'] == 88 and set(config['scenario_ids']) == expected
assert config['temperature'] == 1 and config['seed'] == 42 and config['concurrency'] == 1
assert config['timeout_seconds'] == 120 and config['max_turns'] == 8
assert config['extra_params']['top_p'] == .95
assert config['extra_params']['chat_template_kwargs'] == {'thinking': True}
assert config['extra_params']['retain'] is False
assert config['error_rate'] == 0 and not config['weight_by_difficulty']
assert manifest['trials'] == 1
assert not scores.get('excluded_scenarios')
assert scores.get('completion_rate', 100) == 100
assert not [r for r in rows if r.get('failure_kind') in ('timeout', 'connection_error', 'server_error')]
assert all(r['points'] == {'pass': 2, 'partial': 1, 'fail': 0}[r['status']] for r in rows)
points = sum(r['points'] for r in rows)
assert points == scores['total_points'] and scores['max_points'] == 176
assert round(100 * points / 176) == result['final_score'] == scores['final_score']
db = sqlite3.connect('file:' + str(a.raw / 'data/benchmarks.sqlite') + '?mode=ro', uri=True)
assert db.execute('SELECT status FROM scenario_runs WHERE run_id=?', (result['run_id'],)).fetchone() == ('completed',)
assert list((a.raw / 'runs').rglob('*.md'))
identity = json.loads((a.raw / 'candidate-evidence/identity.json').read_text())
candidate = json.loads((a.raw.parent / 'candidate.json').read_text())
assert identity['sha'] == manifest['engine_sha']
assert identity['sha'] == candidate['sha']
assert len(identity['ranks']) == 4
assert all(r.get('state', {}).get('Running') and r['code_files'] == 337 for r in identity['ranks'].values())
assert len({r['code_sha256'] for r in identity['ranks'].values()}) == 1
assert all(r['code_sha256'] == candidate['engine_sha256'] for r in identity['ranks'].values())

def aggregate(items):
    earned = sum(r['points'] for r in items)
    return {'scenarios': len(items), 'points': earned, 'maximum': 2 * len(items),
            'percent': 100 * earned / (2 * len(items)), 'counts': dict(Counter(r['status'] for r in items))}

def contaminated(r):
    return any('</arg_key>' in line or '<arg_value>' in line for line in r['raw_log'].splitlines()
               if line.startswith('tool_call='))

report = {
    'engine_sha': manifest['engine_sha'], 'run_id': result['run_id'],
    'official_cli_version': result['tool_eval_bench_version'], 'official_score': result['final_score'],
    'protocol': {k: config[k] for k in ('temperature', 'seed', 'concurrency', 'extra_params')},
    'overall': aggregate(rows),
    'standard': aggregate([r for r in rows if int(r['scenario_id'][3:]) <= 69]),
    'hardmode': aggregate([r for r in rows if int(r['scenario_id'][3:]) >= 70]),
    'target_at_least_95_unrounded': points >= 168,
    'non_pass': [{k: r.get(k) for k in ('scenario_id', 'status', 'points', 'summary', 'failure_kind',
                                      'turn_budget_exceeded', 'turn_count', 'duration_seconds')}
                 for r in rows if r['status'] != 'pass'],
    'tag_contamination': [r['scenario_id'] for r in rows if contaminated(r)],
    'evaluator_errors': [r['scenario_id'] for r in rows if r.get('failure_kind') == 'evaluator_error'],
    'turn_budget_exceeded': [r['scenario_id'] for r in rows if r.get('turn_budget_exceeded')],
    'safety_warnings': result.get('safety_warnings', []),
    'median_turn_ms': scores.get('median_turn_ms'), 'total_tokens': scores.get('total_tokens'),
    'elapsed_seconds': sum(r['duration_seconds'] for r in rows),
    'notes': ['All 88 official grades are preserved; no infrastructure exclusions.',
              'Baseline B used the identical fixed CLI version and sampling protocol.',
              'The user reference used default T=0; this run explicitly retains T=1.',
              'One trial does not establish repeatability or production workload quality.'],
}
baseline = json.loads((Path(__file__).resolve().parents[1] / 'st_tool_eval_fix_20260916/candidate-b/raw/result.json').read_text())
keys = ('model', 'backend', 'temperature', 'timeout_seconds', 'max_turns', 'seed', 'reference_date',
        'scenario_count', 'scenario_ids', 'concurrency', 'error_rate', 'alpha', 'extra_params', 'weight_by_difficulty')
assert baseline['tool_eval_bench_version'] == result['tool_eval_bench_version']
assert all(baseline['config'].get(k) == config.get(k) for k in keys)
old = {r['scenario_id']: r for r in baseline['scores']['scenario_results']}
report['points_change'] = points - baseline['scores']['total_points']
report['improved'] = [r['scenario_id'] for r in rows if r['points'] > old[r['scenario_id']]['points']]
report['regressed'] = [r['scenario_id'] for r in rows if r['points'] < old[r['scenario_id']]['points']]
report['baseline_run_id'] = baseline['run_id']
report['same_cli_and_sampling_as_baseline'] = True
(a.raw.parent / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
