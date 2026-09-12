"""Read-only, private calibration snapshots from Deneb client transcripts.

Only main/client UUID conversation files are eligible; cron/system/tool-only
records and thinking blocks are excluded. Sessions never cross dataset splits.
These are reconstructed text conversation windows, not captured provider wire
requests. Raw windows, tokenized captures and provenance must remain private.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re


PATTERNS = [
    (re.compile(r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----', re.S), '[PRIVATE_KEY]'),
    (re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}'), 'Bearer [REDACTED]'),
    (re.compile(r'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})\b'), '[CREDENTIAL]'),
    (re.compile(r'(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\b\s*[=:]\s*)["\']?[^\s,"\'}]{8,}'), r'\1[REDACTED]'),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'(?<!\d)(?:\+82[- .]?)?0?1[016789][- .]?\d{3,4}[- .]?\d{4}(?!\d)'), '[PHONE]'),
]


def sanitize(text):
    replacements = 0
    for pattern, replacement in PATTERNS:
        text, count = pattern.subn(replacement, text)
        replacements += count
    return text, replacements


def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(b['text'] for b in content if isinstance(b,dict)
                         and b.get('type')=='text' and isinstance(b.get('text'),str))
    return ''


def window(history):
    # Keep bounded recent context. Do not include the reply to the current user.
    messages = [dict(role=m['role'],content=m['content'][:3000]) for m in history[-5:]]
    while len(messages)>1 and sum(len(m['content']) for m in messages)>8000:
        messages.pop(0)
    return messages


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--transcripts',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    source=args.transcripts.resolve();destination=args.out.resolve()
    if source==destination.parent or source in destination.parents:
        raise ValueError('output must be outside the production transcript directory')
    if any((p/'.git').exists() for p in destination.parents):
        raise ValueError('private corpus must not be written inside a Git checkout')
    if destination.exists():raise ValueError('use a new immutable output path')
    os.umask(0o077)
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if destination.parent.stat().st_mode & 0o077:
        raise ValueError('private output directory must have mode 0700')
    sessions=[];counters=collections.Counter();source_bytes=0
    eligible=re.compile(r'client:main(?::[0-9a-fA-F-]{36})?\.jsonl')
    for path in sorted(source.glob('client:main*.jsonl')):
        if path.is_symlink() or not eligible.fullmatch(path.name):continue
        size=path.stat().st_size;source_bytes+=size
        if size>64<<20 or source_bytes>128<<20:raise ValueError('source snapshot exceeds bounded inventory')
        with path.open('rb') as f:raw=f.read(size)
        history=[];samples=[]
        for index,line in enumerate(raw.splitlines()):
            try:message=json.loads(line)
            except json.JSONDecodeError:
                if not raw.endswith(b'\n') and index==len(raw.splitlines())-1:
                    counters['incomplete_live_tail']+=1;continue
                raise
            role=message.get('role')
            if role not in ('user','assistant'):continue
            text=text_content(message.get('content')).strip()
            if not text:
                counters['nontext_records_excluded']+=1;continue
            text,count=sanitize(text);counters['redaction_matches']+=count
            history.append(dict(role=role,content=text))
            if role=='user':
                messages=window(history)
                if sum(len(m['content']) for m in messages)<48:continue
                samples.append(dict(messages=messages,line=index,
                                    korean=any('\uac00'<=c<='\ud7a3' for c in text)))
        if samples:
            sessions.append(dict(name=path.name,sha256=hashlib.sha256(raw).hexdigest(),
                                 snapshot_bytes=len(raw),samples=samples))
    if len(sessions)<10:raise ValueError('too few independent conversations')
    # Largest conversational record belongs to training; randomize the other
    # conversations before reserving validation/test sessions. This avoids
    # putting nearly all production history into one evaluation split.
    sessions.sort(key=lambda s:(-len(s['samples']),s['name']))
    largest,others=sessions[0],sessions[1:]
    random.Random(20260912).shuffle(others)
    held=max(3,len(sessions)//5)
    split_sessions={'validation':others[:held],'test':others[held:2*held],
                    'train':[largest]+others[2*held:]}
    caps={'train':160,'validation':40,'test':40}
    rows=[];provenance=[];seen=set()
    for split,items in split_sessions.items():
        # Round-robin sessions, with samples spread across each conversation.
        queues=[]
        for session in items:
            samples=session['samples'];keep=min(len(samples),96 if split=='train' else 20)
            take=sorted({round(i*(len(samples)-1)/max(keep-1,1)) for i in range(keep)})
            queues.append((session,[samples[i] for i in take]))
        count=0;position=0
        while count<caps[split] and any(position<len(q) for _,q in queues):
            for session,queue in queues:
                if position>=len(queue) or count>=caps[split]:continue
                sample=queue[position]
                identity=hashlib.sha256(json.dumps(sample['messages'],sort_keys=True).encode()).hexdigest()
                if identity in seen:continue
                seen.add(identity)
                session_id=hashlib.sha256(session['name'].encode()).hexdigest()[:16]
                row_id=f'deneb-{session_id}-{sample["line"]}'
                rows.append(dict(id=row_id,split=split,messages=sample['messages']))
                provenance.append(dict(id=row_id,session= session['name'],source_sha256=session['sha256'],
                                       snapshot_bytes=session['snapshot_bytes'],korean=sample['korean'],text_sha256=identity))
                count+=1
            position+=1
    counts={s:sum(r['split']==s for r in rows) for s in caps}
    if counts['train']<64 or min(counts['validation'],counts['test'])<16:
        raise ValueError(('insufficient session-disjoint text coverage',counts))
    destination.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    report=dict(scope=__doc__,private=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                split_session_counts={s:len(v) for s,v in split_sessions.items()},prompt_counts=counts,
                source_bytes_read=source_bytes,counters=dict(counters),rows=provenance,
                output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    destination.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(prompt_counts=counts,split_session_counts=report['split_session_counts'],
                          korean_windows=sum(r['korean'] for r in provenance),redaction_matches=counters['redaction_matches'],
                          output_bytes=destination.stat().st_size)))


if __name__=='__main__':
    main()
