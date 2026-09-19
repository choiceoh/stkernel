"""Feed an immutable private GPTQ split to this session's Qwen fleet, then request filing.

This is a collection client, not a throughput or output-quality benchmark.
Use bench/onepass.py separately with collection disarmed for those measurements.
Responses stay private. No commands from corpus text are executed by this client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def request(url, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def verify_owner(path, owner):
    if not owner or not owner.startswith(("session/", "queue/")):
        raise ValueError("collection requires this session's explicit fleet owner")
    if json.loads(Path(path).read_bytes()).get("owner") != owner:
        raise RuntimeError("this session does not hold the fleet; no request sent")


def load_split(path, manifest):
    raw = path.read_bytes()
    split = path.stem
    if hashlib.sha256(raw).hexdigest() != manifest["splits_sha256"][split]:
        raise ValueError("private split changed after preparation")
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError("dataset file contains the wrong split")
    return rows


def collect(args):
    manifest = json.loads(args.dataset.with_name("manifest.json").read_bytes())
    rows = load_split(args.dataset, manifest)
    owner = args.owner or os.environ.get("ST_LEASE_OWNER", "")
    verify_owner(args.lease, owner)
    models = request(args.url, "/v1/models")["data"]
    if args.model not in {m["id"] for m in models}:
        raise RuntimeError("the serving model differs from the prepared Qwen experiment")
    out = args.out.resolve()
    if any((parent / ".git").exists() for parent in [out, *out.parents]):
        raise ValueError("request and response records must remain outside Git")
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    total, cached, start = 0, 0, time.monotonic()
    with (out / "responses.jsonl").open("x") as log:
        for i, row in enumerate(rows):
            verify_owner(args.lease, owner)
            # Corpus rows may share conversation history. A reset makes the
            # Hessian cover every selected row instead of just its uncached tail.
            request(args.url, "/v1/prefix/reset", {})
            body = dict(model=args.model, messages=row["messages"], max_tokens=1, temperature=0.0,
                        seed=20260919, retain=False, stream=False,
                        chat_template_kwargs=manifest["chat_template_kwargs"])
            response = request(args.url, "/v1/chat/completions", body)
            usage = response["usage"]
            actual = int(usage["prompt_tokens"])
            hit = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            record = dict(index=i, id=row["id"], token_sha256=row["token_sha256"],
                          expected_prompt_tokens=row["prompt_tokens"], response=response)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            if actual != row["prompt_tokens"] or hit:
                raise RuntimeError("served token count or prefix reuse differs from the prepared collection")
            total += actual
            cached += hit
            if (i + 1) % 16 == 0 or i + 1 == len(rows):
                print(json.dumps(dict(requests=i + 1, total_requests=len(rows), prompt_tokens=total)), flush=True)
    verify_owner(args.lease, owner)
    filed = request(args.url, "/v1/engine/calibration", {})
    counts = filed.get("rows", {})
    minimum = min(counts.values(), default=0)
    report = dict(split=args.dataset.stem, requests=len(rows), prompt_tokens=total, cached_tokens=cached,
                  min_collected_rows=minimum, rank0_sites=len(counts), save_requested=True,
                  all_rank_file_validation="pending", elapsed_s=time.monotonic() - start,
                  dataset_sha256=manifest["splits_sha256"][args.dataset.stem], owner=owner,
                  scope="collection only; elapsed_s is not inference throughput")
    (out / "filing.json").write_text(json.dumps(filed, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if len(counts) != 193 or minimum < args.min_rows:
        raise RuntimeError("not every target projection reached the declared calibration coverage")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--lease", type=Path, default=Path("/home/choiceoh/glm53-logs/st-fleet.lock"))
    ap.add_argument("--owner", default="")
    ap.add_argument("--min-rows", type=int, default=131072)
    collect(ap.parse_args())


if __name__ == "__main__":
    main()
