"""CPU-only tokenizer audit and completion labels for a private reviewed corpus.

Uses the exact local tokenizer.json and chat_template.jinja, without loading
model weights. This does not establish training/runtime support for a model.
"""
import argparse
import collections
import importlib.metadata
import json
from pathlib import Path

from probes.deneb_qlora_dataset import SPLITS, dump, jsonl, load_jsonl, private_destination, sha


def completion_labels(prompt_ids, full_ids, eos_id, max_length):
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('chat/tokenizer boundary mismatch; never guess a loss mask')
    if len(full_ids) <= len(prompt_ids):
        raise ValueError('no supervised completion tokens')
    ids = list(full_ids)
    if ids[-1] != eos_id:
        ids.append(eos_id)
    if len(ids) > max_length:
        return None
    labels = [-100]*len(prompt_ids) + ids[len(prompt_ids):]
    return dict(input_ids=ids, attention_mask=[1]*len(ids), labels=labels)


def distribution(values):
    values = sorted(values)
    return dict(count=len(values), min=values[0] if values else None,
                median=values[len(values)//2] if values else None,
                p95=values[min(len(values)-1,int(len(values)*.95))] if values else None,
                max=values[-1] if values else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft',type=Path,required=True)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-length',type=int,default=4096)
    args=parser.parse_args()
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    tokenizer=Tokenizer.from_file(str(args.metadata/'tokenizer.json'))
    tokenizer.no_truncation();tokenizer.no_padding()
    config=json.loads((args.metadata/'tokenizer_config.json').read_text())
    eos=config['eos_token']
    if isinstance(eos,dict):eos=eos['content']
    eos_id=tokenizer.token_to_id(eos)
    if eos_id is None:raise ValueError('EOS missing in tokenizer')
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,
        extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson']=lambda value,**kw:json.dumps(value,ensure_ascii=kw.get('ensure_ascii',False))
    template=env.from_string((args.metadata/'chat_template.jinja').read_text())
    destination=private_destination(args.out)
    inventory=[];oversized=[];lengths=collections.defaultdict(list);counts={}
    seen={s:set() for s in SPLITS}
    for split in SPLITS:
        out=[]
        for row in load_jsonl(args.sft/(split+'.jsonl')):
            kwargs=dict(tools=[],reasoning_effort='low',clear_thinking=True)
            prompt=template.render(messages=row['prompt'],add_generation_prompt=True,**kwargs)
            full=template.render(messages=row['prompt']+row['completion'],add_generation_prompt=False,**kwargs)
            if not full.startswith(prompt):raise ValueError('rendered chat prefix mismatch')
            prefix=tokenizer.encode(prompt,add_special_tokens=False).ids
            ids=tokenizer.encode(full,add_special_tokens=False).ids
            result=completion_labels(prefix,ids,eos_id,args.max_length)
            if result is None:
                oversized.append(dict(id=row['id'],split=split,tokens=len(ids)+int(ids[-1]!=eos_id)))
                continue
            visible=tokenizer.decode([x for x in result['labels'] if x!=-100],skip_special_tokens=False)
            # GLM's closing </think> is ordinary text in this tokenizer. Keep
            # the template's exact empty-reasoning wrapper and train EOS too.
            expected=full[len(prompt):] + (eos if ids[-1]!=eos_id else '')
            if visible!=expected or row['completion'][0]['content'] not in visible:
                raise ValueError('supervised target roundtrip failed')
            key=sha(json.dumps(prefix))
            if any(key in values for values in seen.values()):raise ValueError('duplicate tokenized input')
            seen[split].add(key)
            out.append(dict(id=row['id'],**result))
            lengths[split].append(len(result['input_ids']))
            inventory.append(dict(id=row['id'],split=split,prompt_tokens=len(prefix),
                supervised_tokens=sum(x!=-100 for x in result['labels']),
                total_tokens=len(result['input_ids'])))
        jsonl(destination/(split+'.jsonl'),out);counts[split]=len(out)
    jsonl(destination/'token-lengths.jsonl',inventory)
    jsonl(destination/'oversized-excluded.jsonl',oversized)
    pool_lengths=collections.defaultdict(list)
    for row in load_jsonl(args.pool/'pool.jsonl'):
        pool_lengths[row['category']].append(len(tokenizer.encode(row['payload'],add_special_tokens=False).ids))
    report=dict(counts=counts,max_length=args.max_length,oversized_excluded=len(oversized),
        sft_sequence_tokens={k:distribution(v) for k,v in lengths.items()},
        pool_payload_tokens={k:dict(distribution(v),over_4096=sum(n>4096 for n in v),
                                   over_8192=sum(n>8192 for n in v)) for k,v in pool_lengths.items()},
        prompt_labels_all_ignored=True,completion_roundtrip_verified=True,eos_token=eos,eos_id=eos_id,
        tokenizer_version=importlib.metadata.version('tokenizers'),jinja_version=importlib.metadata.version('jinja2'),
        metadata_sha256={n:sha((args.metadata/n).read_bytes()) for n in
                         ('tokenizer.json','tokenizer_config.json','chat_template.jinja')},
        sft_manifest_sha256=sha((args.sft/'manifest.json').read_bytes()),
        rendering_options=dict(reasoning_effort='low',clear_thinking=True,tools=[]),
        native_sft_trainer_tested=False,model_weights_loaded=False,
        note='Pre-tokenized files are specific to these tokenizer/template hashes. Retokenize portable SFT for another model.')
    dump(destination/'manifest.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
