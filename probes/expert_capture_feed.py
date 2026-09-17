"""Feed the expert-capture corpus (probes/expert_capture_corpus.py) through a held measurement boot's door, one
conversation at a time, one generated token each: the capture (engine/profiles/glm53/capture.py) records prefill only.

Runs on the controller beside the hold, as a plain HTTP client (stdlib only). It waits for the door and for the
container to name the capture release, sends conversation i with a first line naming it (no two conversations share
a prefix block, so every token is prefilled and captured), and appends one row per conversation to --log: what was
sent, the prompt tokens the door counted, and how long it took. Rows already in --log are skipped, so a restart
resumes. Order is the corpus's: the k-th sequence the capture saw is the k-th row here.

    python3 probes/expert_capture_feed.py --corpus corpus.jsonl --log feed.jsonl --release <sha> [--port 8001]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def call(url, payload=None, timeout=3600.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def served_release(container: str) -> str:
    try:
        out = subprocess.run(["docker", "exec", container, "printenv", "ST_RELEASE"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def wait_for_door(base: str, release: str, container: str, deadline: float) -> None:
    said = None
    while True:
        state = "no answer"
        try:
            call(base + "/v1/models", timeout=10)
            served = served_release(container)
            if release and served[:12] != release[:12]:
                state = f"the door serves release {served or '(unnamed)'}, not {release[:12]}"
            else:
                return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            state = f"no answer ({type(exc).__name__})"
        if state != said:
            print(f"  waiting: {state}", flush=True)
            said = state
        if time.time() > deadline:
            sys.exit(f"ABORT: {state} after the wait")
        time.sleep(15)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--release", required=True, help="the capture commit: the door must serve it")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--container", default="st-glm53")
    ap.add_argument("--wait-minutes", type=float, default=240.0)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many conversations (0: all)")
    ap.add_argument("--cache-salt", default="", help="a prefix-cache salt: a pass of a corpus already fed must prefill again")
    ap.add_argument("--calibration-root", default="",
                    help="the calibration arm: feed every non-held-out conversation, POST the door to file the sums under "
                         "<root>-fit, then the held-out conversations, then file under <root>-heldout")
    ap.add_argument("--filed-glob", default="", help="a local glob of the filed blobs (rank 0's host view); {suffix} is -fit "
                                                      "or -heldout; waited for after each POST")
    ap.add_argument("--filed-count", type=int, default=0, help="how many blobs a filing writes on this rank")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    corpus = [json.loads(line) for line in open(args.corpus)]
    log = Path(args.log)
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok" and "i" in row:
                done.add(row["i"])
    wait_for_door(base, args.release, args.container, time.time() + 60 * args.wait_minutes)
    print(f"  door up at {base}, release {args.release[:12]}; {len(corpus) - len(done)} of {len(corpus)} conversations to send", flush=True)
    sent = tokens = 0
    t_start = time.time()
    if args.calibration_root:
        corpus = [c for c in corpus if c["split"] != "heldout"] + [c for c in corpus if c["split"] == "heldout"]
    fit_filed = not args.calibration_root

    def file_phase(suffix):
        import glob as globmod
        root = args.calibration_root + suffix
        reply = call(base + "/v1/engine/calibration", dict(root=root), timeout=120)
        print(f"  filing {root}: {reply.get('status')}", flush=True)
        if args.filed_glob:
            pattern = args.filed_glob.replace("{suffix}", suffix)
            deadline = time.time() + 1800
            while len(globmod.glob(pattern)) < args.filed_count and time.time() < deadline:
                time.sleep(10)
            print(f"  filed {len(globmod.glob(pattern))} blobs matching {pattern}", flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(dict(event="filed", root=root, t=round(time.time(), 3), status=reply.get("status"))) + "\n")

    for conv in corpus:
        if not fit_filed and conv["split"] == "heldout":
            file_phase("-fit")
            fit_filed = True
        if conv["i"] in done:
            continue
        if args.limit and sent >= args.limit:
            break
        messages = [dict(m) for m in conv["messages"]]
        messages[0]["content"] = f"[calibration document {conv['i']:04d}]\n" + messages[0]["content"]
        payload = dict(model=args.model, messages=messages, max_tokens=1, temperature=0.0, stream=False)
        if args.cache_salt:
            payload["cache_salt"] = args.cache_salt
        row = dict(i=conv["i"], source=conv["source"], split=conv["split"], corpus_tokens=conv["tokens"], t0=round(time.time(), 3))
        for attempt in (1, 2):
            t0 = time.perf_counter()
            try:
                result = call(base + "/v1/chat/completions", payload)
                usage = result.get("usage") or {}
                row.update(status="ok", attempt=attempt, seconds=round(time.perf_counter() - t0, 3),
                           prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                           cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
                break
            except urllib.error.HTTPError as exc:
                row.update(status="http_error", attempt=attempt, code=exc.code, error=exc.read()[:300].decode(errors="replace"))
            except (urllib.error.URLError, OSError, ValueError) as exc:
                row.update(status="error", attempt=attempt, error=f"{type(exc).__name__}: {exc}"[:300])
            if attempt == 1:
                wait_for_door(base, args.release, args.container, time.time() + 600)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        sent += 1
        if row["status"] == "ok":
            tokens += row["prompt_tokens"] or 0
        print(f"  {conv['i']:4d} {conv['source']:9s} {conv['split']:7s} {row['status']} {row.get('prompt_tokens')} tokens "
              f"{row.get('seconds')} s; {tokens} tokens in {time.time() - t_start:.0f} s", flush=True)
    if args.calibration_root:
        if not fit_filed:
            file_phase("-fit")
        file_phase("-heldout")
    print(f"  fed {sent} conversations, {tokens} prompt tokens, {time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
