"""Extend the frozen Qwen fit with unused real inputs from its original training sessions.

Read only the original transcript byte prefixes, verified against their saved
digests. Preserve every existing split byte and training prefix. Reconstruct and
sanitize additional windows with the original extractor; never read a held-out
session, repeat a selected prompt, or move an evaluation group into training.
Private input text and row-level provenance stay outside Git and stdout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re

from probes.engine_sparse_deneb_corpus import sanitize, text_content, window
from probes.qwen38_gptq_data import digest, private_directory, snapshot
from probes.qwen38_gptq_feed import load_split


def training_sources(rows, provenance):
    splits = {r["id"]: r["split"] for r in rows}
    sessions, sources = {}, {}
    for row in provenance["rows"]:
        split = splits[row["id"]]
        name = row["session"]
        if not re.fullmatch(r"client:main(?::[0-9a-fA-F-]{36})?\.jsonl", name):
            raise ValueError("unexpected transcript name")
        if sessions.setdefault(name, split) != split:
            raise ValueError("one original session spans splits")
        source = (row["snapshot_bytes"], row["source_sha256"])
        if name in sources and sources[name] != source:
            raise ValueError("one session has conflicting snapshot provenance")
        sources[name] = source
    return {name: sources[name] for name, split in sessions.items() if split == "train"}


def original_windows(path, size, sha):
    if path.is_symlink() or not 0 < size <= 64 << 20:
        raise ValueError("invalid original transcript snapshot")
    with path.open("rb") as stream:
        raw = stream.read(size)
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("original training transcript prefix changed")
    history = []
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                continue
            raise
        role = row.get("role")
        if role not in ("user", "assistant"):
            continue
        value = text_content(row.get("content")).strip()
        if not value:
            continue
        value, _ = sanitize(value)
        history.append(dict(role=role, content=value))
        if role == "user":
            messages = window(history)
            if sum(len(m["content"]) for m in messages) >= 48:
                yield index, messages


def normalized(text):
    return re.sub(r"\s+", "", text).casefold()


def guard_fragments(rows):
    # Guard every possible normalized 128-character fragment of held-out
    # messages; candidate fragments are sampled every 32 chars. This also catches
    # substantial cross-source copies without rejecting generic short templates.
    return {text[i:i+128] for row in rows for m in row["messages"]
            for text in [normalized(m["content"])] for i in range(max(0, len(text)-127))}


def overlaps(messages, guard):
    return any(text[i:i+128] in guard for m in messages for text in [normalized(m["content"])]
               for i in range(0, len(text)-127, 32))


def expand(args):
    manifest = json.loads((args.base / "manifest.json").read_bytes())
    splits = {s: load_split(args.base / (s + ".jsonl"), manifest) for s in ("train", "validation", "test")}
    old, provenance, source_sha = snapshot(args.conversations)
    if source_sha != manifest["source_sha256"]["conversations"]:
        raise ValueError("conversation source differs from the original Qwen fit")
    from engine.profiles.qwen38.boot import chat_renderer, tokenizer
    for filename, field in (("chat_template.jinja", "template_sha256"), ("tokenizer.json", "tokenizer_sha256")):
        if digest((args.ckpt / filename).read_bytes()) != manifest[field]:
            raise ValueError("Qwen tokenizer or template changed")
    render, tok = chat_renderer(args.ckpt), tokenizer(args.ckpt)
    sources = training_sources(old, provenance)
    if sum(size for size, sha in sources.values()) > 128 << 20:
        raise ValueError("bounded snapshot inventory exceeded")
    original_ids = {digest(r["id"]) for r in old}
    all_rows = [row for rows in splits.values() for row in rows]
    messages_seen = {r["messages_sha256"] for r in all_rows}
    tokens_seen = {r["token_sha256"] for r in all_rows}
    train_groups = {r["group"] for r in splits["train"] if r["category"] == "conversation"}
    guard = guard_fragments(splits["validation"] + splits["test"])
    queues, counts = defaultdict(list), Counter()
    for name, (size, sha) in sorted(sources.items()):
        group = digest(["conversation", name])
        if group not in train_groups:
            raise ValueError("expansion source is not an existing training group")
        sid = hashlib.sha256(name.encode()).hexdigest()[:16]
        for line, messages in original_windows(args.transcripts / name, size, sha):
            row_id = digest(f"deneb-{sid}-{line}")
            msha = digest(messages)
            if row_id in original_ids or msha in messages_seen:
                counts["already_selected_or_duplicate"] += 1
                continue
            if overlaps(messages, guard):
                counts["evaluation_text_overlap"] += 1
                continue
            text = render(messages, manifest["chat_template_kwargs"])
            ids = tok.encode(text).ids
            if not 65 <= len(ids) <= manifest["max_prompt_tokens"]:
                counts["length_excluded"] += 1
                continue
            tsha = digest(ids)
            if tsha in tokens_seen:
                counts["duplicate_tokens"] += 1
                continue
            messages_seen.add(msha)
            tokens_seen.add(tsha)
            queues[group].append(dict(id=row_id, split="train", group=group, category="conversation",
                messages=messages, messages_sha256=msha, prompt_tokens=len(ids), token_sha256=tsha,
                rendered_sha256=digest(text.encode()), source_snapshot_sha256=sha, source_line=line))
    rng = random.Random(20260919)
    for group in sorted(queues):
        rng.shuffle(queues[group])
    added, total = [], sum(r["prompt_tokens"] for r in splits["train"])
    if args.target_tokens <= total:
        raise ValueError("expansion target must exceed the original training size")
    for index in range(max(map(len, queues.values()), default=0)):
        for group in sorted(queues):
            if index < len(queues[group]) and total < args.target_tokens:
                row = queues[group][index]
                added.append(row)
                total += row["prompt_tokens"]
        if total >= args.target_tokens:
            break
    if total < args.target_tokens:
        raise ValueError(f"only {total} distinct training tokens available after expansion guards")
    os.umask(0o077)
    out = private_directory(args.out)
    base_train = (args.base / "train.jsonl").read_bytes()
    if not base_train.endswith(b"\n"):
        raise ValueError("original training file lacks its final newline")
    addition = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in added).encode()
    (out / "train.jsonl").write_bytes(base_train + addition)
    for split in ("validation", "test"):
        (out / (split + ".jsonl")).write_bytes((args.base / (split + ".jsonl")).read_bytes())
    report = dict(manifest)
    report.update(version=2, scope=__doc__, script_sha256=digest(Path(__file__).read_bytes()),
        base_manifest_sha256=digest((args.base / "manifest.json").read_bytes()),
        splits_sha256={s: digest((out / (s + ".jsonl")).read_bytes()) for s in splits},
        prompt_tokens=dict(manifest["prompt_tokens"], train=total),
        counts=dict(Counter(r["split"] + "/" + r["category"] for r in all_rows + added)),
        expansion=dict(target_tokens=args.target_tokens, added_prompts=len(added),
            added_tokens=total-manifest["prompt_tokens"]["train"], source_sessions=len(sources),
            added_session_groups=len({r["group"] for r in added}), candidates=sum(map(len, queues.values())),
            counters=dict(counts), original_train_prefix_preserved=True,
            evaluation_bytes_preserved=True, original_snapshot_prefixes_verified=True,
            added_category="conversation", sample_order="original train prefix, then new windows round-robin by session"))
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--conversations", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-tokens", type=int, default=330000)
    expand(ap.parse_args())


if __name__ == "__main__":
    main()
