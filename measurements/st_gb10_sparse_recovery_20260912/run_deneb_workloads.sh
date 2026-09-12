set -euo pipefail
umask 077
trap 'chown -R 1000:1000 /private' EXIT
cd /repo
python3 - <<'PY'
import json,hashlib,collections
from pathlib import Path
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from probes.engine_sparse_capture import prompt_messages
p=Path('/private');rows=json.loads((p/'prompts.json').read_text())
old=json.loads(Path('/conversation/calibration-prompts.json').read_text())
rows += [dict(r,category='public_diagnostic') for r in old if r['split']=='test']
t=AutoTokenizer.from_pretrained('/meta',local_files_only=True)
rt=Tokenizer.from_file('/meta/tokenizer.json')
template=Path('/repo/launchers/chat_template_mm_v2.jinja').read_text()
prior=set()
for path in ['/work/capture.json','/work/expanded-capture.json','/conversation/capture.json']:
 prior.update(tuple(r['token_ids']) for r in json.loads(Path(path).read_text())['prompts'])
seen=set();kept=[];excluded=collections.Counter()
for row in sorted(rows,key=lambda r:({'train':0,'validation':1,'test':2}[r['split']],r['id'])):
 text=t.apply_chat_template(prompt_messages(row),chat_template=template,add_generation_prompt=True,tokenize=False,thinking=False)
 ids=tuple(rt.encode(text,add_special_tokens=False).ids[:512])
 if ids in seen or (row['split']=='test' and row['category']!='public_diagnostic' and ids in prior):
  excluded[row['category']+'/'+row['split']]+=1;continue
 seen.add(ids);kept.append(row)
calibration=[r for r in kept if r['split']!='test' or r['category']=='public_diagnostic']
evaluation=[r for r in kept if r['split']=='test' and r['category']!='public_diagnostic']
assert len(calibration)<=256 and len(evaluation)>=30
for category in ('conversation','mail','notification'):
 assert sum(r['category']==category for r in evaluation)>=8
for name,data in [('calibration-prompts.json',calibration),('evaluation-prompts.json',evaluation)]:
 (p/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
report=dict(counts=dict(collections.Counter(r['category']+'/'+r['split'] for r in kept)),prefix_duplicates_excluded=dict(excluded),final_prompt_count=len(evaluation),calibration_prompt_count=len(calibration),max_tokens=512,tokenized_splits_disjoint=True,baseline_training_prefixes_disjoint=True)
(p/'token-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
PY
python3 probes/engine_sparse_capture.py --ranks /ranks --metadata /meta --prompts /private/calibration-prompts.json --max-tokens 512 --out /private/capture.pt > /private/capture.log 2>&1
python3 probes/engine_sparse_calibrate.py --capture /private/capture.pt --rank /ranks/rank0of4.safetensors --library /native/sparse.so --expert-ids 10 4 119 178 --train-cap 8192 --out /private/recovery.json > /private/recovery.log 2>&1
python3 probes/engine_sparse_residual.py --capture /private/capture.pt --recovery /private/recovery.json --rank /ranks/rank0of4.safetensors --rank-budget-from /work/residual.json --out /private/residual.json > /private/residual.log 2>&1
python3 probes/engine_sparse_chain_residual.py --capture /private/capture.pt --recovery /private/recovery.json --residual /private/residual.json --rank /ranks/rank0of4.safetensors --out /private/chain.json > /private/chain.log 2>&1
python3 - <<'PY'
import json,hashlib,datetime
from pathlib import Path
names=['recovery.json','recovery.weights.pt','residual.json','residual.weights.pt','chain.json','chain.weights.pt']
items={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in ['/private','/conversation'] for name in names if (p:=Path(base)/name).exists()}
for name in ['recovery.json','recovery.weights.pt','residual.json','residual.weights.pt','expanded-recovery.json','expanded-recovery.weights.pt','joint-residual.json','joint-residual.weights.pt']:
 p=Path('/work')/name;items[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
Path('/private/frozen.json').write_text(json.dumps(dict(frozen_before_final_capture=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_sha256=items),indent=2)+'\n')
PY
python3 probes/engine_sparse_capture.py --ranks /ranks --metadata /meta --prompts /private/evaluation-prompts.json --max-tokens 512 --out /private/final-capture.pt > /private/final-capture.log 2>&1
for variant in public-small public-large conversation workloads-independent workloads-chain; do
 case "$variant" in
 public-small) source=/work/capture.pt; recovery=/work/recovery.json; residual=/work/residual.json;;
 public-large) source=/work/expanded-capture.pt; recovery=/work/expanded-recovery.json; residual=/work/joint-residual.json;;
 conversation) source=/conversation/capture.pt; recovery=/conversation/recovery.json; residual=/conversation/chain.json;;
 workloads-independent) source=/private/capture.pt; recovery=/private/recovery.json; residual=/private/residual.json;;
 workloads-chain) source=/private/capture.pt; recovery=/private/recovery.json; residual=/private/chain.json;;
 esac
 python3 probes/engine_sparse_holdout.py --capture /private/final-capture.pt --training-capture "$source" --recovery "$recovery" --residual "$residual" --rank /ranks/rank0of4.safetensors --out "/private/final-$variant.json" > "/private/final-$variant.log" 2>&1
done
