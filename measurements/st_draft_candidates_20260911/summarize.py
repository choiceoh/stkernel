"""Verify each measured revision separately; never transfer TP4 results to unrun code."""
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UNFUSED = '9de1a199'
FUSED = '28fcdfbc'
BASELINE = '2ead1f09'


@lru_cache(None)
def source_sha(revision, path):
    return hashlib.sha256(subprocess.check_output(['git', 'show', revision+':'+path], cwd=ROOT)).hexdigest()


def load(name, revision, old_rtx=False):
    report = json.loads((HERE/name).read_text())
    assert report['passed'], name
    assert report['torch_git'] == 'cf30153c4c131c8164ee7798e5022d810682e2cb', name
    for path, sha in report['source_sha256'].items():
        expected = (hashlib.sha256((HERE/'rtx-probe.py').read_bytes()).hexdigest()
                    if old_rtx and path == 'probes/engine_draft_candidate_check.py'
                    else source_sha(revision, path))
        assert sha == expected, (name, path)
    if 'baseline_sha256' in report:
        assert report['baseline_sha256'] == source_sha(BASELINE, 'engine/profiles/glm53/drafter.py')
    if 'unfused_sha256' in report:
        assert report['unfused_sha256'] == source_sha(UNFUSED, 'engine/modules/vocab.py')
    for case in report['cases']:
        assert case['exact'], (name, case)
        for timing in case.get('timing', {}).values():
            samples = timing['samples_us']
            rounds, count = (5, 50) if name == 'pack-rtx.json' else ((5, 30) if name.startswith('tp4/') else (3, 30))
            assert len(samples) == rounds and all(len(r) == count for r in samples), name
            flat = sum(samples, [])
            assert all(math.isfinite(v) and v > 0 for v in flat), name
            assert statistics.median(flat) == timing['median_us'], name
    return report


def timing_range(reports, index):
    a = [r['cases'][index]['timing']['legacy']['median_us'] for r in reports]
    b = [r['cases'][index]['timing']['candidate']['median_us'] for r in reports]
    gain = [100*(1-y/x) for x,y in zip(a,b)]
    return dict(legacy_us=[min(a), max(a)], candidate_us=[min(b), max(b)],
                reduction_percent=[min(gain), max(gain)])


def fleet(directory, expected_cases):
    reports = [load(f'{directory}/rank{rank}.json', UNFUSED) for rank in range(4)]
    runs = json.loads((HERE/directory/'runs.json').read_text())
    assert len(runs) == 4 and {r['rank'] for r in runs} == set(range(4))
    assert all(r['exit'] == 0 and r['state']['ExitCode'] == 0 for r in runs)
    for rank, report in enumerate(reports):
        assert report['rank'] == rank and len(report['cases']) == expected_cases
        cap = (512 << 20) if directory == 'tp4' else 5*2**30
        assert report['peak_reserved_bytes'] <= cap
        log = (HERE/directory/f'rank{rank}.log').read_text()
        assert 'Using network IB' in log and 'Destroy COMPLETE' in log
    before = json.loads((HERE/directory/'before.json').read_text())
    after = json.loads((HERE/directory/'after.json').read_text())
    assert [n['node'] for n in before] == [n['node'] for n in after]
    assert not any(s['name'].startswith('st-draft-topk-f4d7-') for n in after for s in n['services'])
    changes = []
    for a,b in zip(before, after):
        old, new = ({s['name'] for s in n['services']} for n in (a,b))
        if old != new:
            changes.append(dict(node=a['node'], started=sorted(new-old), stopped=sorted(old-new)))
    return reports, changes


def main():
    manifest = json.loads((HERE/'final-source-manifest.json').read_text())
    data = subprocess.check_output(['git', 'archive', manifest['final_source_revision'],
                                    'engine', 'tests', 'probes', 'launchers'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        sources = {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive if m.isfile()}
    for path, sha in manifest['source_sha256'].items():
        assert sources[path] == sha, path
    tp, micro_changes = fleet('tp4', 27)
    real, real_changes = fleet('real-tp4', 6)
    expected = [(rows, limit, mode) for rows in (1,5,20) for limit in (153880,38710,7)
                for mode in ('random','ties','nonfinite')]
    for report in tp:
        assert [(c['rows'], c['decodable'], c['mode']) for c in report['cases']] == expected
        for c in report['cases']:
            assert c['local_candidate_bytes'] == c['rows']*16*8
            assert c['gathered_candidate_bytes'] == c['local_candidate_bytes']*4
            assert c['local_legacy_bytes'] == c['rows']*38720*2
    for report in real:
        assert report['world_size'] == 4
        assert [c['context'] for c in report['cases']] == [0,1,17,2047,2048,2057]
        assert all(c['rank_agreement'] and len(c['ids']) == 5 for c in report['cases'])
        assert [c['ids'] for c in report['cases']] == [c['ids'] for c in real[0]['cases']]
    original = load('real-rtx.json', UNFUSED, old_rtx=True)
    fused = load('real-rtx-fused.json', FUSED)
    packed = load('pack-rtx.json', FUSED)
    assert len(original['cases']) == len(fused['cases']) == 6
    assert [c['rows'] for c in packed['cases']] == [1,5,20]
    for log, count, suffix in (('cpu-native-final.log', 231, ' (skipped=57)'),
                               ('rtx-fused-tests.log', 10, '')):
        assert re.search(rf'Ran {count} tests in .*\n\nOK{re.escape(suffix)}\n', (HERE/log).read_text()), log
    integration = json.loads((HERE/'merge-validation/provenance.json').read_text())
    for path, sha in integration['source_sha256'].items():
        assert source_sha(integration['revision'], path) == sha, path
    for log, count, suffix in (('native-cpu-tests.log', 246, ' (skipped=64)'), ('rtx-tests.log', 20, '')):
        assert re.search(rf'Ran {count} tests in .*\n\nOK{re.escape(suffix)}\n',
                         (HERE/'merge-validation'/log).read_text()), log
    old_bytes, new_bytes = 5*154880*2, 5*4*16*8
    result = dict(
        status='Implemented; fused revision still needs GB10/TP4 qualification. No full-model speedup claim.',
        unfused_revision=UNFUSED, fused_revision=FUSED,
        merge_validation=integration,
        payload=dict(legacy_gathered_bytes=old_bytes, candidate_gathered_bytes=new_bytes,
                     reduction_percent=100*(1-new_bytes/old_bytes)),
        tests=dict(final_native_cpu_passed=174, final_native_cpu_skipped=57, final_rtx_passed=10,
                   unfused_tp4_micro_rank_cases=108, unfused_real_tp4_rank_contexts=24,
                   fused_real_rtx_contexts=6),
        unfused_tp4_selection=[dict(rows=tp[0]['cases'][i]['rows'], **timing_range(tp,i)) for i in (0,9,18)],
        unfused_real_tp4=[dict(context=real[0]['cases'][i]['context'], **timing_range(real,i)) for i in range(6)],
        fleet_lifecycle_changes=dict(selection=micro_changes, real_weights=real_changes),
        fused_rtx_local_selection=[dict(rows=c['rows'], **{k:v['median_us'] for k,v in c['timing'].items()})
                                   for c in packed['cases']],
        fused_real_rtx=[dict(context=c['context'], **{k:v['median_us'] for k,v in c['timing'].items()})
                        for c in fused['cases']])
    (HERE/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
