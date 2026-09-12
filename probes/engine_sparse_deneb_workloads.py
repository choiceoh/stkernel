"""Private, read-only workload calibration from Deneb mail and event stores.

Mail is reconstructed from archived headers/body. Phone judgments and digest/
briefing inputs use run.start previews (production logs cap these at 4096 bytes).
Neither source is a full provider request. No stored answer is a target label.
All input text, source identities, and provenance must stay outside Git.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re

from probes.engine_sparse_deneb_corpus import sanitize


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    return re.sub(r'\s+', '', str(value)).casefold()


def subject_key(subject):
    return normalized(re.sub(r'^(?:(?:re|fw|fwd|회신|전달)\s*:\s*)+', '', subject, flags=re.I))


class Components:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.root(self.parent[key])
        return self.parent[key]

    def link(self, keys):
        roots = sorted({self.root(k) for k in keys})
        for root in roots[1:]:
            self.parent[root] = roots[0]
        return roots[0]


def mail_keys(row):
    ids = re.findall(r'<([^<>]+)>', ' '.join(str(row.get(k, '')) for k in ('message_id', 'references')))
    if row.get('message_id') and not ids:
        ids = [str(row['message_id']).strip('<> ')]
    keys = ['mail-id:' + x.casefold() for x in ids]
    subject = subject_key(row.get('subject', ''))
    if subject:
        keys.append('mail-subject:' + subject)
    return keys or ['mail-body:' + digest(normalized(row.get('body', '')))]


def phone_payload(message):
    match = re.search(r'^출처:[ \t]*([^\n]+)\n내용:[ \t]*\n(.*?)\n\n위는 사용자 스마트폰', message, re.M | re.S)
    if not match:
        return None
    source, body = match.groups()
    return source.strip(), body.strip()


def read_rows(path, inventory):
    if path.is_symlink():
        raise ValueError('symlink input is not a snapshot')
    size = path.stat().st_size
    inventory['bytes'] += size
    if size > 64 << 20 or inventory['bytes'] > 192 << 20:
        raise ValueError('bounded workload inventory exceeded')
    with path.open('rb') as stream:
        raw = stream.read(size)
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        inventory['records_read'] += 1
        try:
            yield i, json.loads(line.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines)-1 and not raw.endswith(b'\n'):
                inventory['incomplete_tail'] += 1
            else:
                inventory['malformed_records_excluded'] += 1


def split_groups(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    ordered = sorted(groups, key=lambda k: (-len(groups[k]), k))
    if len(ordered) < 10:
        raise ValueError('too few independent workload groups')
    largest, rest = ordered[0], ordered[1:]
    random.Random(20260912).shuffle(rest)
    held = max(3, len(ordered)//5)
    allocation = {'train': [largest]+rest[2*held:], 'validation': rest[:held], 'test': rest[held:2*held]}
    for split, keys in allocation.items():
        for key in keys:
            for row in groups[key]:
                row['split'] = split
    return rows


def spread(rows, count):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    queues = []
    for key in sorted(groups):
        queue = sorted(groups[key], key=lambda r: digest(r['identity']))
        queues.append(queue)
    out = []
    for index in range(max(map(len, queues), default=0)):
        for queue in queues:
            if index < len(queue):
                out.append(queue[index])
                if len(out) == count:
                    return out
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--conversations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    destination, state = args.out.resolve(), args.state.resolve()
    if destination.exists() or state == destination.parent or state in destination.parents:
        raise ValueError('use a new private output outside production state')
    if any((p/'.git').exists() for p in destination.parents):
        raise ValueError('production data must stay outside Git')
    os.umask(0o077)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.parent.stat().st_mode & 0o077:
        raise ValueError('private directory must have mode 0700')
    inventory = collections.Counter()
    candidates = []
    components = Components()
    for path in sorted((state/'mailstore/messages').glob('*.jsonl')):
        for line, row in read_rows(path, inventory):
            body = row.get('body', '').strip()
            if len(body) < 80:
                continue
            keys = mail_keys(row)
            components.link(keys)
            headers = '\n'.join(f'{label}: {row.get(key, "")}' for label, key in
                                [('From','from'), ('To','to'), ('Subject','subject'), ('Date','date')])
            text = '다음 업무 메일의 핵심 사실, 위험, 기한과 필요한 후속 조치를 분석하세요.\n'+headers+'\n\n--- 본문 ---\n'+body
            text, count = sanitize(text)
            inventory['redaction_matches'] += count
            candidates.append(dict(category='mail', text=text, payload=body, keys=keys,
                identity=str(path)+':'+str(line), source_kind='archived mail; reconstructed user input'))
    for row in candidates:
        row['group'] = components.root(row['keys'][0])
    split_groups(candidates)
    phones, background = [], []
    for path in sorted((state/'agent-logs').glob('*.jsonl')):
        if not (path.name.startswith('phone-event:') or path.name.startswith('noti-digest')
                or path.name.startswith('cron:morning-letter')):
            continue
        for line, row in read_rows(path, inventory):
            if row.get('type') != 'run.start':
                continue
            message = row.get('data', {}).get('message', '')
            if not isinstance(message, str) or len(message) < 80:
                continue
            # Ignore likely truncated previews; the log limit is byte based.
            if len(message.encode()) >= 4093:
                inventory['truncated_previews_excluded'] += 1
                continue
            text, count = sanitize(message)
            inventory['redaction_matches'] += count
            item = dict(text=text, identity=str(path)+':'+str(line),
                        source_kind='run.start message preview; no system/history/tools')
            if path.name.startswith('phone-event:'):
                payload = phone_payload(message)
                if payload is None:
                    inventory['unrecognized_phone_preview'] += 1
                    continue
                source, body = payload
                if len(body) < 24:
                    continue
                item.update(category='notification', payload=body, group='phone-source:'+normalized(source))
                phones.append(item)
            else:
                kind = 'digest' if path.name.startswith('noti-digest') else 'briefing'
                item.update(category='background', payload=message, group='background:'+kind, split='train')
                background.append(item)
    split_groups(phones)
    if inventory['malformed_records_excluded'] > max(1, inventory['records_read']//100):
        raise ValueError('more than one percent of source records are malformed')
    # Background batches can reference many mail/phone items. Keep them in train
    # only; exclude evaluation payloads overlapping any selected background text.
    background = spread(background, 8)
    old = json.loads(args.conversations.read_text())
    old_guard = '\n'.join(m['content'] for row in old if row['split'] != 'test' for m in row['messages'])
    guard = normalized(old_guard+'\n'+'\n'.join(r['text'] for r in background))
    candidates += phones
    guard += normalized('\n'.join(r['payload'] for r in candidates if r['split']=='train'))
    unique = set()
    clean = []
    for row in candidates:
        identity = digest(normalized(row['payload']))
        if identity in unique:
            inventory['duplicate_payloads_excluded'] += 1
            continue
        unique.add(identity)
        payload = normalized(row['payload'])
        if row['split'] != 'train' and (payload in guard or any(payload[i:i+128] in guard for i in range(0, len(payload)-127, 32))):
            inventory['cross_source_evaluation_overlap_excluded'] += 1
            continue
        clean.append(row)
    candidates = clean
    chosen = list(background)
    caps = {'train':64, 'validation':16, 'test':16}
    for category in ('mail', 'notification'):
        for split, cap in caps.items():
            pool = [r for r in candidates if r['category']==category and r['split']==split]
            selected = spread(pool, cap)
            if len(selected) != cap:
                raise ValueError(('insufficient workload coverage', category, split, len(selected)))
            chosen.extend(selected)
    rows, provenance = [], []
    for row in chosen:
        key = 'deneb-workload-'+digest(row['identity'])[:20]
        rows.append(dict(id=key, split=row['split'], text=row['text'], category=row['category']))
        provenance.append(dict(id=key, group=digest(row['group']), source=row['identity'],
                               category=row['category'], split=row['split'], source_kind=row['source_kind']))
    for split, cap in [('train',64), ('validation',16), ('test',12)]:
        pool = [r for r in old if r['split']==split]
        # Existing conversation split is retained, including the earlier model's
        # complete training sessions. Selection never promotes an old train row.
        chosen_old = sorted(pool, key=lambda r:digest(r['id']))[:cap]
        for row in chosen_old:
            rows.append(dict(row, category='conversation'))
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    groups = {s:{r['group'] for r in provenance if r['split']==s} for s in caps}
    assert not any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    report = dict(private=True, scope=__doc__, source_sha256=digest(Path(__file__).read_text()),
        counters=dict(inventory), workload_groups_disjoint=True, background_train_only=True,
        counts=dict(collections.Counter(r['category']+'/'+r['split'] for r in rows)), rows=provenance,
        output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    destination.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('counts','counters','workload_groups_disjoint','background_train_only')}))


if __name__ == '__main__':
    main()
