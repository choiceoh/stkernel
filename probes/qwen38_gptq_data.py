"""Prepare private real-input Qwen GPTQ splits, preserving the source conversation/workload groups.

CPU only. Inputs are the existing sanitized Deneb snapshots and their provenance,
not production state. No source text or generation belongs in Git or stdout.
The snapshots reconstruct user inputs; they are not full captured provider requests
and their old responses are not ground-truth labels for model quality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def snapshot(path):
    path = Path(path)
    raw = path.read_bytes()
    provenance = json.loads(path.with_suffix(".provenance.json").read_bytes())
    if digest(raw) != provenance["output_sha256"]:
        raise ValueError("snapshot digest differs from its provenance")
    return json.loads(raw), provenance, digest(raw)


def messages(row):
    result = row.get("messages") or [{"role": "user", "content": row["text"]}]
    if not result or result[-1]["role"] != "user":
        raise ValueError("a calibration prompt must end with the input user turn")
    for m in result:
        if m.get("role") not in ("user", "assistant", "system") or not isinstance(m.get("content"), str):
            raise ValueError("expected text-only conversation messages")
    return [{"role": m["role"], "content": m["content"]} for m in result]


def grouped_rows(conversations, conversation_provenance, workloads, workload_provenance):
    """Merge the overlapping conversation subset once; never promote a held-out group to train."""
    cp = {row["id"]: row for row in conversation_provenance["rows"]}
    wp = {row["id"]: row for row in workload_provenance["rows"]}
    known, groups, content_splits = {}, {}, {}
    rows = []
    for row in [*conversations, *workloads]:
        key, split = row["id"], row["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("unknown source split")
        value = messages(row)
        identity = digest(value)
        if key in known:
            if known[key] != (split, identity):
                raise ValueError("one source row changed content or split")
            continue
        known[key] = split, identity
        if key in cp:
            group = digest(["conversation", cp[key]["session"]])
            category = "conversation"
        else:
            source = wp[key]
            if source["split"] != split:
                raise ValueError("source split differs from provenance")
            group = digest(["workload", source["group"]])
            category = row["category"]
        if groups.setdefault(group, split) != split:
            raise ValueError("one source group spans calibration and evaluation splits")
        if identity in content_splits:
            if content_splits[identity] != split:
                raise ValueError("identical input spans calibration and evaluation splits")
            continue
        content_splits[identity] = split
        rows.append(dict(id=digest(key), split=split, group=group, category=category,
                         messages=value, messages_sha256=identity))
    return rows


def interleave(rows, seed=20260919):
    """Round-robin categories so a capped collector does not see only the first source."""
    pools = defaultdict(list)
    for row in rows:
        pools[row["category"]].append(row)
    rng = random.Random(seed)
    for key in sorted(pools):
        rng.shuffle(pools[key])
    return [pools[key][i] for i in range(max(map(len, pools.values()), default=0))
            for key in sorted(pools) if i < len(pools[key])]


def private_directory(path):
    path = Path(path).resolve()
    if any((parent / ".git").exists() for parent in [path, *path.parents]):
        raise ValueError("private data must remain outside Git")
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    return path


def prepare(args):
    conversations, cp, csha = snapshot(args.conversations)
    workloads, wp, wsha = snapshot(args.workloads)
    rows = grouped_rows(conversations, cp, workloads, wp)
    from engine.profiles.qwen38.boot import chat_renderer, tokenizer
    render, tok = chat_renderer(args.ckpt), tokenizer(args.ckpt)
    kept, excluded, token_splits = [], Counter(), {}
    for row in rows:
        text = render(row["messages"], {"enable_thinking": True})
        ids = tok.encode(text).ids
        if not 65 <= len(ids) <= args.max_tokens:
            excluded[row["split"] + ("/short" if len(ids) < 65 else "/long")] += 1
            continue
        sha = digest(ids)
        if sha in token_splits:
            if token_splits[sha] != row["split"]:
                raise ValueError("identical Qwen input tokens span splits")
            excluded[row["split"] + "/duplicate_tokens"] += 1
            continue
        token_splits[sha] = row["split"]
        kept.append(dict(row, prompt_tokens=len(ids), token_sha256=sha, rendered_sha256=digest(text.encode())))
    totals = {split: sum(r["prompt_tokens"] for r in kept if r["split"] == split)
              for split in ("train", "validation", "test")}
    if totals["train"] < args.min_train_tokens or min(totals["validation"], totals["test"]) < 4096:
        raise ValueError(f"insufficient distinct real inputs after filtering: {totals}")
    os.umask(0o077)
    out = private_directory(args.out)
    files = {}
    for split in totals:
        selected = interleave([r for r in kept if r["split"] == split])
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
        (out / (split + ".jsonl")).write_text(data)
        files[split] = digest(data.encode())
    report = dict(version=1, scope=__doc__, source_sha256={"conversations": csha, "workloads": wsha},
                  template_sha256=digest((args.ckpt / "chat_template.jinja").read_bytes()),
                  tokenizer_sha256=digest((args.ckpt / "tokenizer.json").read_bytes()),
                  script_sha256=digest(Path(__file__).read_bytes()),
                  chat_template_kwargs={"enable_thinking": True}, max_prompt_tokens=args.max_tokens,
                  counts=dict(Counter(r["split"] + "/" + r["category"] for r in kept)),
                  prompt_tokens=totals, excluded=dict(excluded), splits_sha256=files,
                  source_groups_disjoint=True, exact_messages_disjoint=True, exact_tokens_disjoint=True,
                  generated_labels=False)
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conversations", type=Path, required=True)
    ap.add_argument("--workloads", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--min-train-tokens", type=int, default=131072)
    prepare(ap.parse_args())


if __name__ == "__main__":
    main()
