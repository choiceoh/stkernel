"""Audit the private portable bundle; print aggregate evidence only."""
import argparse
import collections
import datetime
import email.utils
import json
from pathlib import Path
import re

from probes.deneb_qlora_dataset import SPLITS, dump, load_jsonl, norm, sha, validate_annotation


def timestamp(value):
    try:
        if isinstance(value,(int,float)):
            return datetime.datetime.fromtimestamp(value/(1000 if value>1e11 else 1),datetime.timezone.utc)
        try:
            result=datetime.datetime.fromisoformat(str(value).replace('Z','+00:00'))
        except ValueError:
            result=email.utils.parsedate_to_datetime(str(value))
        return result.replace(tzinfo=datetime.timezone.utc) if result.tzinfo is None else result.astimezone(datetime.timezone.utc)
    except (ValueError,TypeError,OverflowError):
        return None


def audit(root):
    rows=load_jsonl(root/'pool/pool.jsonl')
    inventory=json.loads((root/'pool/inventory-report.json').read_text())
    lookup={r['id']:r for r in rows}
    assert len(lookup)==len(rows)==inventory['pool_rows']
    assert len({sha(norm(r['payload'])) for r in rows})==len(rows)
    assert all(sha(r['payload'])==r['payload_sha256'] for r in rows)
    groups={s:{r['group_id'] for r in rows if r['split']==s} for s in SPLITS}
    pairs=[('train','validation'),('train','test'),('validation','test')]
    assert all(not groups[a]&groups[b] for a,b in pairs)
    assert inventory['rows_before_dedup']-len(rows)==len(load_jsonl(root/'pool/duplicates.jsonl'))
    email_pattern=re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
    residual_emails=sum(bool(email_pattern.search(r['payload']+str(r.get('candidate_response') or ''))) for r in rows)
    assert residual_emails==0, 'email-pattern redaction incomplete'
    annotations={r['id']:r for r in load_jsonl(root/'sft/annotations.jsonl')}
    sft_ids=set();counts={};decision_counts={};evidence=0
    token_manifest=json.loads((root/'tokens-glm53/manifest.json').read_text())
    for split in SPLITS:
        data=load_jsonl(root/'sft'/(split+'.jsonl'))
        token_data=load_jsonl(root/'tokens-glm53'/(split+'.jsonl'))
        assert {r['id'] for r in data}=={r['id'] for r in token_data}
        counts[split]=len(data)
        decisions=collections.Counter()
        for r in data:
            assert r['id'] not in sft_ids
            sft_ids.add(r['id'])
            source=lookup[r['id']]
            assert source['split']==split and r['human_verified'] is False
            answer=validate_annotation(source,annotations[r['id']])
            assert r['prompt'][-1]['role']=='user' and r['completion'][0]['role']=='assistant'
            assert len(r['completion'])==1 and json.loads(r['completion'][0]['content'])==answer
            decisions[answer['decision']]+=1;evidence+=len(answer['evidence'])
        decision_counts[split]=dict(decisions)
        for r in token_data:
            ids,labels=r['input_ids'],r['labels']
            assert len(ids)==len(labels)==len(r['attention_mask'])<=token_manifest['max_length']
            first=next(i for i,x in enumerate(labels) if x!=-100)
            assert first>0 and labels[:first]==[-100]*first and labels[first:]==ids[first:]
            assert labels[-1]==token_manifest['eos_id']
    assert sft_ids==set(annotations)
    time_ranges={}
    for category in ('mail','notification','conversation'):
        selected=[r for r in rows if r['category']==category]
        dates=[t for r in selected if (t:=timestamp(r.get('timestamp'))) is not None]
        time_ranges[category]=dict(parsed=len(dates),missing_or_unparsed=len(selected)-len(dates),
            earliest=min(dates).isoformat() if dates else None,latest=max(dates).isoformat() if dates else None)
    return dict(pool_rows=len(rows),eligible_pool_rows=sum(r['split']!='quarantine' for r in rows),
        category_totals=dict(collections.Counter(r['category'] for r in rows)),
        quarantine_rows=sum(r['split']=='quarantine' for r in rows),
        group_counts={s:len(v) for s,v in groups.items()},cross_split_group_overlaps=0,
        residual_email_pattern_rows=residual_emails,annotation_counts=counts,
        annotation_decisions_by_split=decision_counts,exact_source_quotes_verified=evidence,
        sft_token_masks_verified=True,target_truncations=0,
        timestamp_ranges_utc=time_ranges,raw_text_in_this_report=False,
        semantic_label_correctness_automatically_proven=False,human_verified_labels=0,
        note='The 48 assistant-reviewed annotations are a seed set. Historical development splits are not a new blind benchmark.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args();result=audit(args.bundle)
    if args.out:dump(args.out,result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
