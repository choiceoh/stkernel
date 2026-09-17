#!/usr/bin/env python3
"""Bisect the incident prompt by LENGTH, on the same content, across the covered boundary.

The incident's controls vary length and content together: a 39-token clean question
reads, a 41-47k context with the incident's system, tools and history does not. So
"the selection" and "the input" are still confounded, and the one shape that tells
them apart is this one -- the SAME ids, cut to different suffix lengths, with the user
question (the last turn) always at the end.

Why the boundary matters: the sparse indexer selects only when a row's visible pools
exceed the top-k budget. With `index_topk = 2048` and `kpool = 4` that is 512 pools =
2048 tokens, so a context at or below it takes the covered path and never selects,
while above it every row runs the selector. If the answers go bad only above the
boundary, the selection path is the cause; if they go bad below it too, the cause is
in the input or in a path every length takes.

    python3 probes/incident_length_bisect.py --record /tmp/<private>.json \
        --door http://127.0.0.1:8001 --out /tmp/bisect

Add `--lengths 1024,1536,2048,3072` to concentrate around the boundary. Private
prompt ids stay outside the repository; only counts, hashes and glyph damage print.
"""
import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.request
import uuid
from pathlib import Path

DEFAULT_LENGTHS = (768, 1536, 2048, 3072, 6144, 12288, 24576, 50005)


def http(door, path, body=None, timeout=900):
    request = urllib.request.Request(door + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def glyph_scan(text):
    """The counters bench/korean-corruption.py gates on, plus the sentence shape."""
    welded = len(re.findall(r"[\uac00-\ud7a3][\u1100-\u11ff\u3130-\u318f]", text))
    hangul = len(re.findall(r"[\uac00-\ud7a3]", text))
    return dict(chars=len(text), hangul=hangul, replacement=text.count("\ufffd"), welded_jamo=welded,
                han=len(re.findall(r"[\u4e00-\u9fff]", text)),
                cyrillic=len(re.findall(r"[\u0400-\u04ff]", text)),
                thai=len(re.findall(r"[\u0e00-\u0e7f]", text)),
                ascii_words=len(re.findall(r"[A-Za-z]{2,}", text)),
                control=sum(1 for c in text if unicodedata.category(c) == "Cc" and c not in "\t\n\r"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True, help="the private engine record with the prompt ids")
    parser.add_argument("--door", default="http://127.0.0.1:8001")
    parser.add_argument("--out", required=True, help="private output directory, outside the repository")
    parser.add_argument("--lengths", default=",".join(str(n) for n in DEFAULT_LENGTHS))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()

    record = json.loads(Path(args.record).read_text())
    prompt = record["tokens"][:record["prompt_len"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for length in (int(x) for x in args.lengths.split(",") if x.strip()):
        if length > len(prompt):
            print(f"{length:>6} tokens: skipped, the prompt is {len(prompt)}")
            continue
        ids = prompt[len(prompt) - length:]                    # the question is the last turn, so it survives every cut
        before = http(args.door, "/")
        if any(before.get("running", [])) or before.get("waiting") or before.get("queued"):
            raise SystemExit("the door is not idle: this bisect needs an idle engine")
        started = time.monotonic()
        result = http(args.door, "/v1/engine/completions",
                      dict(ids=ids, temperature=args.temperature, seed=args.seed, top_p=1, top_k=-1,
                           max_tokens=args.max_tokens, retain=False,
                           cache_salt="incident-bisect-" + uuid.uuid4().hex))
        after = http(args.door, "/")
        text = result.get("text") or ""
        row = dict(length=length, prompt_tokens=result.get("prompt_tokens"),
                   cached_tokens=result.get("cached_tokens"), completion_tokens=result.get("completion_tokens"),
                   served_delta=after["served"] - before["served"], seconds=round(time.monotonic() - started, 1),
                   text_sha256=hashlib.sha256(text.encode()).hexdigest(), glyph=glyph_scan(text))
        rows.append(row)
        (out / f"bisect-{length}.json").write_text(json.dumps(result, ensure_ascii=False))
        print(json.dumps(row, ensure_ascii=False))
    (out / "bisect-summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out}/bisect-summary.json -- read the answers there; a clean answer below 2048 tokens "
          f"and damaged ones above it point at the selection, damage below it does not")


if __name__ == "__main__":
    main()
