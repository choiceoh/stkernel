#!/usr/bin/env python3
"""Summarize the actual B-A-B onepass evidence; windows are not independent boots."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def summarize(root):
    records = [json.loads(s) for s in (root/'records.raw.jsonl').read_text().splitlines() if s]
    assert [r['name'] for r in records] == ['MOEWSB1', 'MOEWSA1', 'MOEWSB2']
    assert len({r['boot_id'] for r in records}) == 3
    assert len({r['overlay'] for r in records}) == 1
    assert len({r['git'] for r in records}) == 1
    reference = None
    source = None
    rows = []
    for record in records:
        name = record['name']
        mode = 't,ws' if name == 'MOEWSA1' else 't'
        assert not record.get('evidence_issues'), record.get('evidence_issues')
        assert record['workload']['require_exclusive']
        assert record['decode']['primary'] == 'fixed-2K' and record['decode']['num_spec'] == 5
        for node in (2,1,3,4):
            before = json.loads((root/f'prepared-{name}-srv{node}.json').read_text())
            after = json.loads((root/f'runtime-{name}-srv{node}.json').read_text())
            assert {k:v for k,v in before.items() if k != 'markers'} == {k:v for k,v in after.items() if k != 'markers'}, (name,node,'runtime changed during traffic')
            suffix = 'tm32f2g2a32wut' + ('ws' if mode == 't,ws' else '') + ' ('
            for proof in (before,after):
                assert any('static2_m6_k4096_n512_t8_r' in line and suffix in line for line in proof['markers'])
            assert after['mode'] == mode and after['knobs']['VLLM_GLM53_MK_INPUT_CTA'] == '2'
            if node == 2:
                assert after['boot_id'] == record['boot_id']
            source = source or after['source_sha256']
            assert source == after['source_sha256']
        identity = [(q['request_sha256'],q['prompt_tokens'],q['seed']) for q in record['requests']]
        reference = reference or identity
        assert identity == reference, 'requests differ between arms'
        windows = record['decode']['fixed_intervals']
        assert len(windows) >= 20
        rate = sum(w['steps'] for w in windows)/sum(w['seconds'] for w in windows)
        assert math.isclose(rate,record['decode']['fixed_pooled_step_s'])
        fixed = [q for q in record['requests'] if q.get('fixed_decode')]
        assert len(fixed) == 3 and all(q['completion_tokens'] == 2048 for q in fixed)
        channels = [json.loads(s) for s in (root/f'channels-{name}.jsonl').read_text().splitlines()]
        assert len(channels) == len(record['requests'])
        assert [q['request_sha256'] for q in channels] == [q['request_sha256'] for q in record['requests']]
        assert [q['output_sha256'] for q in channels] == [q['output_sha256'] for q in record['requests']]
        rows.append(dict(name=name,mode=mode,windows=len(windows),step_s=rate,ms_per_step=1000/rate,
                         median_step_s=median(w['steps']/w['seconds'] for w in windows),
                         output_tok_s=sum(q['completion_tokens']-1 for q in fixed)/sum(q['decode_s'] for q in fixed),
                         acceptance_all_requests=record['decode']['acc_raw'],
                         quality=record['quality'],korean=record['korean'],
                         prefill=record['prefill']))
    b1,a,b2 = rows
    base = {key:(b1[key]+b2[key])/2 for key in ('step_s','output_tok_s','median_step_s')}
    prefill = []
    for index, ctx in enumerate((2000,32000,128000)):
        key = 'warm_tok_s' if ctx == 2000 else 'cold_tok_s'
        values = [r['prefill'][index][key] for r in rows]
        baseline = (values[0]+values[2])/2
        prefill.append(dict(ctx=ctx,kind='repeat-min-TTFT' if ctx == 2000 else 'single-request',
                            per_boot_tok_s=values,baseline_mean_tok_s=baseline,
                            candidate_change_pct=100*(values[1]/baseline-1)))
    return dict(per_boot=rows,baseline_equal_boot_mean=base,
                change_pct={key:100*(a[key]/value-1) for key,value in base.items()},
                baseline_boot_spread_pct={key:100*abs(b2[key]-b1[key])/value for key,value in base.items()},
                matched_requests=True,matched_source_sha256=source,
                source_commit=records[0]['git'],prefill_comparison=prefill,
                note='B-A-B bracket, one candidate boot. Within-boot windows are correlated. Acceptance covers all requests; output tok/s covers fixed 2K decode only.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    args = ap.parse_args()
    result = summarize(args.root)
    (args.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
