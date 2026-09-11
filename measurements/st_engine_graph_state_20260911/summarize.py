"""Validate evidence completeness and summarize independent runs without filtering."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
from functools import cache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MEASURED = '7068978c13f064c23f1ee2de3c0a1e0356c3b912'


@cache
def source_digest(path):
    # This proposal was superseded by the direct-state implementation in #549.
    # Validate historical measurements against their immutable source revision.
    data = subprocess.check_output(['git', 'show', f'{MEASURED}:{path}'], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def paired(names):
    by_case = {}
    for name in names:
        report = read(name)
        assert len(report['results']) == 4, name
        for path, digest in report['source_sha256'].items():
            assert source_digest(path) == digest, (name, path)
        for case in report['results']:
            assert all(case['hidden_logits_state_paged_exact']), (name, case)
            a,b = [case['timing'][k]['median_us'] for k in ('baseline', 'optimized')]
            by_case.setdefault((case['context'], case['tokens']), []).append(
                dict(file=name, baseline_us=a, optimized_us=b, reduction_percent=100*(1-b/a)))
    return [dict(context=ctx, tokens=tokens, runs=values,
                 reduction_range_percent=[min(v['reduction_percent'] for v in values),
                                          max(v['reduction_percent'] for v in values)])
            for (ctx,tokens),values in sorted(by_case.items())]


def transfers():
    grouped = {}
    for name in ('transfer.json', 'transfer-2.json', 'transfer-3.json'):
        r = read(name)
        assert len(r['results']) == 6
        for x in r['results']:
            for tokens in (1, 6):
                a,b = [x['gather'][k]['median_us'] + x['commit'][str(tokens)][k]['median_us']
                       for k in ('baseline','optimized')]
                grouped.setdefault((x['kda_layers'],x['active_sequences'],tokens), []).append(
                    dict(file=name, baseline_us=a, optimized_us=b, reduction_percent=100*(1-b/a)))
    return [dict(kda_layers=layers, active_sequences=n, tokens=t, runs=v,
                 # This is the sum of separately measured gather/commit medians.
                 baseline_median_us=statistics.median(x['baseline_us'] for x in v),
                 optimized_median_us=statistics.median(x['optimized_us'] for x in v),
                 reduction_range_percent=[min(x['reduction_percent'] for x in v),
                                          max(x['reduction_percent'] for x in v)])
            for (layers,n,t),v in sorted(grouped.items())]


gpu_log = (HERE / 'gpu-tests.log').read_text()
assert re.search(r'Ran 145 tests in .*\n\nOK\n', gpu_log), 'GPU suite incomplete or skipped'
graph_checks = {}
for rank in range(4):
    name = f'st-state-graph-check-f4d7-rank{rank}.log'
    lines = (HERE / name).read_text().splitlines()
    cases = [s for s in lines if s.startswith(('case ', 'rejected-future '))]
    assert len(cases) == 17 and 'PASS' in lines
    assert all('relative 0.0 state_exact True paged_exact True' in s for s in cases)
    graph_checks[rank] = len(cases)
runtime = read('runtime-final.json')
assert runtime['passed'] and not runtime['vllm_present']
for path, digest in runtime['source_files'].items():
    assert source_digest('engine/' + path) == digest, path
summary = dict(
    scope='recurrent transfer improvement; no qualified full-model or TP4 speedup claim',
    gpu_tests=145, graph_checks_per_rank=graph_checks,
    transfers=transfers(),
    isolated=paired([f'isolated-paired-{i}.json' for i in (1,2,3)]),
    tp4_initial_2cpu=paired([f'st-state-paired-{i}-f4d7-rank{rank}.json'
                            for i in (1,2,3) for rank in range(4)]),
    tp4_4cpu=paired([f'st-state-paired-{i}-f4d7-rank{rank}.json'
                    for i in (4,5,6) for rank in range(4)]))
(HERE / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
for x in summary['transfers']:
    if x['kda_layers'] == 34:
        print(x['active_sequences'], x['tokens'],
              round(x['baseline_median_us'], 1), round(x['optimized_median_us'], 1),
              [round(v, 2) for v in x['reduction_range_percent']])
for group in ('isolated', 'tp4_4cpu'):
    print(group, [(x['context'], x['tokens'], [round(v,2) for v in x['reduction_range_percent']])
                  for x in summary[group]])
