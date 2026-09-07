#!/usr/bin/env python3
"""Matched fixed-window B/A/A/B. Independent boots, not windows, are replicates."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parent/'serving'
NAMES=('IREUSEB1','IREUSEA1','IREUSEA2','IREUSEB2')
KNOBS={'VLLM_GLM53_MK_INPUT_REUSE':'1'}
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'


def summarize(rows, *, incomplete=False):
    expected=NAMES[:2] if incomplete else NAMES
    assert [r['name'] for r in rows]==list(expected), 'four complete independent B/A/A/B boots required unless explicitly reporting the failed first pair'
    assert len({r['boot_id'] for r in rows})==len(expected)
    if incomplete:
        assert len(rows)==2, 'partial report must preserve the completed first pair'
        verdicts=[json.loads(line) for line in (ROOT/'verdicts.jsonl').read_text().splitlines() if line.strip()]
        assert verdicts, 'partial report requires the recorded verdict'
    assert len({r['overlay'] for r in rows})==1
    first_requests=None
    receipts=[]
    missing_receipts=[]
    out=[]
    source=(ROOT/'source.commit').read_text().strip()
    production=json.loads((ROOT/'production-gate.json').read_text())
    assert production['status']=='PASS'
    for tool in ('racecheck','memcheck'):
        assert json.loads((ROOT/f'{tool}.json').read_text())['status']=='PASS'
        assert 'ERROR SUMMARY: 0 errors' in (ROOT/f'{tool}.log').read_text()
    for record in rows:
        name=record['name']; arm='A' if name in NAMES[1:3] else 'B'
        assert not record.get('rehearsal') and not record.get('evidence_issues')
        assert all(record.get(k)==rows[0].get(k) for k in ('workload','runtime','harness','thinking','doc_lang'))
        assert record['endpoint']=={'completion':'http://127.0.0.1:18000/v1/chat/completions',
                                   'metrics':'http://127.0.0.1:18000/metrics'}
        assert record['knobs']==(KNOBS if arm=='A' else {})
        if arm=='A':
            assert record['proof_ok']=='1/1' and all(record['proof'][k] for k in KNOBS)
        assert not record['traffic']['issues']
        assert record['traffic']['after']['finished']-record['traffic']['before']['finished']==len(record['requests'])
        for node in (1,2,3,4):
            path=ROOT/f'runtime-{name}-srv{node}.json'
            if incomplete and arm=='A' and not path.exists():
                missing_receipts.append(path.name)
                continue
            proof=json.loads(path.read_text())
            assert proof['image']==IMAGE and proof['running']
            assert proof['knobs']['VLLM_GLM53_MK_INPUT_REUSE']==('1' if arm=='A' else '0')
            assert proof['source_sha256']['glm53_megakernel.cu']==production['source_sha256']
            assert proof['arm']==('candidate' if arm=='A' else 'baseline')
            assert bool(proof['markers'])==(arm=='A')
            if arm=='A': assert any('M=6 N=6528 K=4096 split=8' in s for s in proof['markers'])
            if node==2:
                assert proof['boot_id']==record['boot_id']
            receipts.append(proof)
        boot_log=(ROOT/f'boot-{name}.log').read_text(errors='replace')
        for marker in ('[megakernel] input-reuse CAPTURED M=6 N=6528 K=4096 split=8',):
            assert (marker in boot_log)==(arm=='A')
        decode=record['decode']
        assert decode['primary']=='fixed-2K' and decode['num_spec']==5
        windows=decode['fixed_intervals']
        assert len(windows)>=20 and all(w['seconds']>0 and w['steps']>=0 for w in windows)
        rate=sum(w['steps'] for w in windows)/sum(w['seconds'] for w in windows)
        assert math.isclose(rate,decode['fixed_pooled_step_s'])
        assert math.isclose(median(w['steps']/w['seconds'] for w in windows),decode['windows_med'])
        requests=[q for q in record['requests'] if q.get('fixed_decode')]
        assert len(requests)==5 and all(q['completion_tokens']==2048 for q in requests)
        identity=[(q['request_sha256'],q['prompt_tokens'],q['seed']) for q in requests]
        if first_requests is None: first_requests=identity
        assert identity==first_requests
        tokens=sum(q['completion_tokens']-1 for q in requests)
        seconds=sum(q['decode_s'] for q in requests)
        assert seconds>0
        q,k=record['quality'],record['korean']
        out.append({'name':name,'arm':arm,'windows':len(windows),'step_s':rate,
                    'window_step_counts':dict(sorted(Counter(str(int(w['steps'])) for w in windows).items())),
                    'ms_per_step':1000/rate,'window_median_step_s':decode['windows_med'],
                    'pooled_output_tok_s':tokens/seconds,'pooled_tpot_ms':1000*seconds/tokens,
                    'median_request_tok_s':median(q['decode_tok_s'] for q in requests),
                    'acceptance_all_requests':decode['acc_raw'],
                    'tokens_per_step_all_requests':decode['tokens_per_step'],
                    'quality':q,'korean':k,'quality_pass':q['ok']==q['total'] and k['dirty']==0})
    assert all(p['source_sha256']==receipts[0]['source_sha256'] for p in receipts)
    metrics=('window_median_step_s','step_s','pooled_output_tok_s','pooled_tpot_ms','ms_per_step')
    stats={a:{k:median(r[k] for r in out if r['arm']==a) for k in metrics} for a in ('B','A')}
    paired=[{'baseline':out[b]['name'],'candidate':out[a]['name'],
             **{k+'_change_pct':100*(out[a][k]/out[b][k]-1) for k in ('window_median_step_s','step_s','pooled_output_tok_s')}}
            for b,a in (((0,1),) if incomplete else ((0,1),(3,2)))]
    spread={a:{k:100*(max(r[k] for r in out if r['arm']==a)-min(r[k] for r in out if r['arm']==a))/stats[a][k]
               for k in ('window_median_step_s','step_s','pooled_output_tok_s')} for a in ('B','A')} if not incomplete else None
    return {'source_commit':source,'per_boot':out,'arms':stats,'paired':paired,
            'change_pct':{k:100*(stats['A'][k]/stats['B'][k]-1) for k in metrics},
            'within_arm_boot_spread_pct':spread,'independent_boots_per_arm':1 if incomplete else 2,
            'complete_abba':not incomplete,'missing_runtime_receipts':missing_receipts,
            'all_quality_gates_passed':all(r['quality_pass'] for r in out),
            'new_defaults_promoted':False,
            'note':('First B/A pair only: the chain stopped before A2/B2. '
                    'See the recorded verdict and receipts; no stable speedup can be concluded from this partial bracket.'
                    if incomplete else
                    'Each boot has equal weight. Windows within a boot are correlated; four boots do not give a narrow confidence interval.')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incomplete',action='store_true',help='Report the completed first pair explicitly; never writes summary.json')
    args=parser.parse_args()
    rows=[json.loads(line) for line in (ROOT/'records.raw.jsonl').read_text().splitlines() if line.strip()]
    result=summarize(rows,incomplete=args.incomplete)
    (ROOT/('partial-summary.json' if args.incomplete else 'summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
