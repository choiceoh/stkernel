"""Reconcile canonical request evidence, profile ranges and all-rank guards."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def main():
    rows=[json.loads(line) for line in (ROOT/'onepass.jsonl').read_text().splitlines() if line.strip()]
    by_name={r['name']:r for r in rows}
    issues=[]
    result={'issues':issues,'contexts':{},'memory_min_gib':{},
            'scope':'Current observed production model settings, reduced KV capacity; diagnostic only'}
    for ctx in [32,128]:
        records={tag:by_name[f'PATTR{ctx}K{tag}'] for tag in ['B1','P','B2']}
        requests={tag:r['requests'][0] for tag,r in records.items()}
        if len({r['request_sha256'] for r in requests.values()})!=1:
            issues.append(f'{ctx}K: model request bodies differ')
        if len({r['prompt_tokens'] for r in requests.values()})!=1:
            issues.append(f'{ctx}K: actual token counts differ')
        for tag,r in records.items():
            if r['quality']!={'ok':3,'total':3} or r['korean']['dirty']!=0:
                issues.append(f'{ctx}K {tag}: quality failure')
            if r.get('evidence_issues') or r['traffic']['issues'] or len(r['requests'])!=1:
                issues.append(f'{ctx}K {tag}: invalid onepass evidence')
            capture=json.loads((ROOT/(r['name']+'.capture.json')).read_text())
            if capture['issues'] or capture['unsalted_request_sha256']!=requests[tag]['request_sha256']:
                issues.append(f'{ctx}K {tag}: invalid capture evidence')
        traces={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'analysis'/f'PATTR{ctx}KP').glob('*.json'))}
        if len(traces)!=4:
            issues.append(f'{ctx}K: expected four rank analyses')
        for node,trace in traces.items():
            if trace['prefill_tokens']!=requests['P']['prompt_tokens']:
                issues.append(f'{ctx}K {node}: traced tokens differ from prompt tokens')
            if trace['graph_events']:
                issues.append(f'{ctx}K {node}: captured graph events in pure prefill')
        keys={c for t in traces.values() for c in t['categories']}
        cats={}
        for cat in keys:
            vals=[t['categories'].get(cat,{}).get('occupied_pct_span',0) for t in traces.values()]
            cats[cat]={'rank_mean_pct':statistics.mean(vals),'rank_min_pct':min(vals),
                       'rank_max_pct':max(vals)}
        b1,p,b2=(requests[tag]['ttft_s'] for tag in ['B1','P','B2'])
        result['contexts'][str(ctx)]={
            'tokens':requests['P']['prompt_tokens'],'ttft_s':{'before':b1,'profiled':p,'after':b2},
            'instrumented_vs_after_ratio':p/b2,'categories':dict(sorted(cats.items(),key=lambda kv:-kv[1]['rank_mean_pct'])),
            'ranks':{node:{k:t[k] for k in ['span_ms','busy_ms','idle_ms','cross_category_overlap_ms',
                       'communication_compute_overlap_ms','prefill_chunks','selected_events','excluded_events']}
                     for node,t in traces.items()}}
    for path in ROOT.glob('PATTR*.memory.jsonl'):
        for line in path.read_text().splitlines():
            sample=json.loads(line)
            if sample['issues']:
                issues.append(path.name+': memory guard issue')
            for node,state in sample['nodes'].items():
                if 'available_kib' not in state:
                    issues.append(node+': missing memory')
                    continue
                value=state['available_kib']/1048576
                old=result['memory_min_gib'].get(node,value)
                result['memory_min_gib'][node]=min(old,value)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if issues:
        raise SystemExit(1)


if __name__=='__main__':
    main()
