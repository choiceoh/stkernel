"""Export an explicit allowlist of numerical aggregates; never copy private reports."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--private',type=Path,required=True)
ap.add_argument('--conversation',type=Path,required=True)
ap.add_argument('--work',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
p,root=args.private,args.work
load=lambda name:json.loads((p/name).read_text())
audit=load('token-audit.json'); capture=load('capture.json'); final=load('final-capture.json')
frozen=load('frozen.json')
mounts={'/private':p,'/conversation':args.conversation,'/work':root}
for key,expected in frozen['artifact_sha256'].items():
 prefix='/'+key.split('/')[1]
 path=mounts[prefix]/key.removeprefix(prefix+'/')
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
assert frozen['frozen_before_final_capture']
assert (p/'frozen.json').stat().st_mtime_ns < (p/'final-capture.pt').stat().st_mtime_ns
assert not ({tuple(r['token_ids']) for r in final['prompts']} & {tuple(r['token_ids']) for r in capture['prompts']})
assert all(r['split']=='test' for r in final['prompts'])
allowed_metrics={'relative_l2','relative_max','row_relative_l2_p50','row_relative_l2_p95','mean_cosine','finite'}
def metric(value):
 return {k:v for k,v in value.items() if k in allowed_metrics}
def summary(meta):
 return dict(prompts=len(meta['prompts']),tokens=meta['total_kept_tokens'],peak_torch_allocated_bytes=meta['peak_torch_allocated_bytes'],all_tp_replicas_exact=all(b['tp_replicas_exact'] for b in meta['batches']),counts=dict(collections.Counter(r.get('category','conversation')+'/'+r['split'] for r in meta['prompts'])),split_tokens={s:sum(r['kept_tokens'] for r in meta['prompts'] if r['split']==s) for s in ['train','validation','test']})
report=dict(scope='Private Deneb workload calibration, L3 rank0 experts 10/4/119/178; local numerical errors against existing ST quantized checkpoint, not full-model accuracy',production_adopted=False,private_inputs_exported=False,balanced_diagnostic_sample_not_traffic_frequency=True,
 calibration=summary(capture),evaluation=summary(final),tokenized_splits_disjoint=True,baseline_training_prefixes_disjoint=True,
 frozen_artifacts_verified=True,frozen_before_final_capture=True,variants={})
for name in ['public-small','public-large','conversation','workloads-independent','workloads-chain']:
 data=load('final-'+name+'.json')
 report['variants'][name]=dict(selected_route_fraction=data['selected_route_fraction'],tokens_with_selected_route=data['tokens_with_selected_route'],peak_torch_allocated_bytes=data['peak_torch_allocated_bytes'],weighted_selected_expert_sum={k:metric(v) for k,v in data['weighted_selected_expert_sum'].items()},per_category={k:dict(tokens_with_selected_route=v['tokens_with_selected_route'],variants={m:metric(n) for m,n in v['variants'].items()}) for k,v in data['per_category'].items()},whole_experts=[dict(expert=c['expert'],rows=c['rows'],expert_chain={k:metric(v) for k,v in c.get('expert_chain',{}).items()}) for c in data['cases']])
recovery=load('recovery.json');chain=load('chain.json')
checks=[projection['kernel_correctness'] for c in recovery['cases'] for projection in c['projections'].values()]
assert len(checks)==8 and all(x['sparse_vs_dense']['bits_equal'] and all(r['bits_equal'] for r in x['reference']) for x in checks)
report['native_projection_checks']=dict(count=len(checks),all_bits_equal=True)
report['factor_bytes']=chain['factor_bytes']
report['factor_budget_bytes']=4161536
assert report['factor_bytes']<=report['factor_budget_bytes']
report['training_route_counts']={str(c['expert']):c['counts'] for c in recovery['cases']}
args.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(evaluation=report['evaluation'],factor_bytes=report['factor_bytes'],results={k:dict(overall=v['weighted_selected_expert_sum']['residual']['relative_l2'],categories={a:b['variants']['residual']['relative_l2'] for a,b in v['per_category'].items()}) for k,v in report['variants'].items()})))
