"""Revalidate the frozen diagnostic and summarize its complete row population."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from glm53_moe_m64_int8_diagnostic import completion,MARKER


def analyze():
    rows=[json.loads(l) for l in gzip.open(ROOT/'tp4-raw/fp8-v3-rank-0.log.gz','rt').read().splitlines() if l.startswith('{')]
    trials=[r for r in rows if r.get('kind')=='MOE_M64_INT8_DIAGNOSTIC_TRIAL']
    final=[r for r in rows if r.get('verdict')==MARKER]
    assert len(final)==1
    report=final[0]
    assert completion(trials,report['preflight'],report['provenance'])==report
    assert hashlib.sha256((ROOT/'source/glm53_prefill_collectives.py').read_bytes()).hexdigest()==report['provenance']['glm53_prefill_collectives.py']
    for rank in range(4):
        log=gzip.open(ROOT/f'tp4-raw/fp8-v3-rank-{rank}.log.gz','rt').read()
        assert '[prefill-sp] packed INT8 reduce-scatter engaged' in log
        assert 'Traceback (most recent call last)' not in log
    before=json.loads((ROOT/'evidence/before.json').read_text())
    after=json.loads((ROOT/'evidence/restored.json').read_text())
    recovery={n:all(before[n][k]==after[n][k] for k in
        ('id','image','config','host_config','mounts','overlays','manifest','port')) and after[n]['running'] for n in before}
    assert len(recovery)==4 and all(recovery.values())
    lifecycle=json.loads((ROOT/'evidence/completion.json').read_text())
    worker=json.loads((ROOT/'completion.json').read_text())
    assert worker['exit_code']==lifecycle['exit_code']==0 and lifecycle['restored_original']
    cases=[]
    for n in (4096,6144,8192):
        selected=[r for r in trials if r['rows']==n]
        counts={phase:{k:sum(a[k] for r in selected for a in r[phase]) for k in ('candidate_bad','control_bad')}
                for phase in ('fp8','int8','partial')}
        quantization={}
        for mode in ('fp8','int8'):
            values=[v['arms']['candidate'][mode] for r in selected for v in r['quality']]
            quantization[mode]=dict(median_of_rank_trial_median_l2=statistics.median(v['median_l2'] for v in values),
                worst_row_l2=max(v['max_l2'] for v in values),worst_row_peak=max(v['max_peak'] for v in values))
        cases.append(dict(rows=n,skew=selected[0]['skew'],trials=len(selected),row_trials=n*len(selected),
                          failures=counts,candidate_quantization=quantization))
    return dict(source_revision=json.loads((ROOT/'request.json').read_text())['revision'],
        diagnostic_complete=True,numerical_acceptance=False,serving_acceptance=False,speed_measured=False,
        trials=len(trials),row_trials_per_arm=sum(r['rows'] for r in trials),
        packet_checks=sum(len(v['arms']) for r in trials for v in r['checks']),
        cpu_packet_checks=sum(a['cpu_reference'] for r in trials for v in r['checks'] for a in v['arms'].values()),
        codec_cases=128,short_identity_cases=8,thresholds=report['thresholds'],cases=cases,
        exact_incoming_recovery=recovery,worker_completion=worker,
        limitations=['Synthetic MoE and transport cases; no full-model quality or TTFT measurement.',
                     'Row-trials are repeated observations, not independent samples.',
                     'The diagnostic completion marker never substitutes for the full acceptance gate.'])


if __name__=='__main__':
    print(json.dumps(analyze(),indent=2))
