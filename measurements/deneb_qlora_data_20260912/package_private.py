"""Run on srv4; package already validated private outputs without modifying state."""
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, '/tmp/deneb-qlora-9391')
from probes.deneb_qlora_dataset import dump, jsonl, load_jsonl, private_destination, seal, sha
from probes.deneb_qlora_audit import audit

os.umask(0o077)
base=Path('/home/choiceoh/datasets/deneb')
final=base/'qlora-2026-09-12-v2'
if final.exists():raise ValueError('immutable published bundle exists')
root=private_destination(final.with_name(final.name+'.incomplete'))
for source,target in [('qlora-pool-2026-09-12-v5','pool'),
                      ('qlora-sft-2026-09-12-v3','sft'),
                      ('qlora-tokens-glm53-2026-09-12-v4','tokens-glm53')]:
    shutil.copytree(base/source,root/target)
tools=root/'tools/probes';tools.mkdir(parents=True,mode=0o700)
for name in ('deneb_qlora_dataset.py','deneb_qlora_tokenize.py','deneb_qlora_audit.py',
             'engine_sparse_deneb_corpus.py','engine_sparse_deneb_workloads.py'):
    shutil.copyfile(Path('/tmp/deneb-qlora-9391/probes')/name,tools/name)
meta=root/'model-metadata';meta.mkdir(mode=0o700)
for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
    shutil.copyfile(Path('/home/choiceoh/models/glm53-nvidia-nvfp4')/name,meta/name)
shutil.copyfile('/tmp/deneb-qlora-9391/PRIVATE_README.md',root/'README.md')
shutil.copyfile('/tmp/deneb-qlora-9391/audit.ipynb',root/'audit.ipynb')
rows=load_jsonl(root/'pool/pool.jsonl')
annotated={r['id'] for r in load_jsonl(root/'sft/annotations.jsonl')}
queue=[]
for r in rows:
    if r['split']=='quarantine' or r['id'] in annotated:continue
    queue.append(dict(id=r['id'],category=r['category'],split=r['split'],group_id=r['group_id'],
        payload_sha256=r['payload_sha256'],label_status=r['label_status'],quality_flags=r['quality_flags'],
        next_step='source_and_answer_review' if r['candidate_response'] else 'write_grounded_target'))
queue.sort(key=lambda r:(bool(r['quality_flags']),r['label_status']=='unlabeled',r['category'],r['id']))
jsonl(root/'review-queue.jsonl',queue)
report=audit(root);dump(root/'quality-report.json',report)
dump(root/'bundle-manifest.json',dict(version=1,private=True,canonical=True,
    source_datasets=['qlora-pool-2026-09-12-v5','qlora-sft-2026-09-12-v3','qlora-tokens-glm53-2026-09-12-v4'],
    pool_rows=len(rows),review_queue_rows=len(queue),reviewed_sft_rows=len(annotated),
    human_verified_labels=0,training_performed=False,
    code_sha256={p.name:sha(p.read_bytes()) for p in tools.glob('*.py')}))
for p in root.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
root.rename(final)
seal(final)
print(json.dumps(report,ensure_ascii=False))
