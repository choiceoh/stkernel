#!/usr/bin/env python3
"""Capture a Korean onepass prefill; stop profiling at its first streamed token.

The canonical onepass still completes and checks the full answer. Controls
and captures use identical request bodies after an explicit prefix-cache reset.
No serving kernels or sampling settings are changed by this diagnostic.
"""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


def first_piece(raw):
    line = raw.decode("utf-8", "replace").strip()
    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
        return False
    try:
        obj = json.loads(line[5:])
    except ValueError:
        return False
    return any(any((c.get("delta") or {}).get(k) for k in
                   ("content", "reasoning_content", "reasoning"))
               for c in obj.get("choices") or [])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo / "bench"))
    import onepass
    from window_metrics import metric_sum, traffic_state

    base = "http://127.0.0.1:" + os.environ.get("GLM53_API_PORT", "18000")
    opener = urllib.request.urlopen
    record = {"name": args.name, "ctx": args.ctx, "capture": args.capture,
              "start_epoch": time.time(), "events": [], "issues": []}

    def save():
        (args.out / (args.name + ".capture.json")).write_text(
            json.dumps(record, indent=2) + "\n")

    def post(path):
        t = time.time()
        with opener(urllib.request.Request(base + path, data=b"", method="POST"),
                    timeout=600) as response:
            status, body = response.status, response.read().decode()
        record["events"].append({"path": path, "start_epoch": t,
                                  "end_epoch": time.time(), "status": status,
                                  "body": body[:200]})
        if status != 200:
            raise RuntimeError(f"{path}: HTTP {status}")
        return body

    def metrics():
        with opener(base + "/metrics", timeout=10) as response:
            raw = response.read().decode()
        return {"traffic": traffic_state(raw),
                "prefix_hits": metric_sum(raw, "vllm:prefix_cache_hits_total"),
                "prefix_queries": metric_sum(raw, "vllm:prefix_cache_queries_total")}

    if metrics()["traffic"]["running"] != 0 or metrics()["traffic"]["waiting"] != 0:
        raise RuntimeError("server must be idle before capture")
    post("/reset_prefix_cache")
    record["before"] = metrics()
    active = False
    stop_future = None
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    class Stream:
        def __init__(self, response):
            self.response = response

        def __enter__(self):
            self.response.__enter__()
            return self

        def __exit__(self, *exc):
            return self.response.__exit__(*exc)

        def __iter__(self):
            nonlocal stop_future
            for raw in self.response:
                if args.capture and stop_future is None and first_piece(raw):
                    record["first_piece_epoch"] = time.time()
                    # Stop is a separate HTTP connection. Do not truncate or
                    # otherwise change the answer consumed by onepass.
                    stop_future = pool.submit(post, "/stop_profile")
                yield raw

    def instrumented_open(request, *a, **kw):
        url = getattr(request, "full_url", request)
        response = opener(request, *a, **kw)
        return Stream(response) if url == base + "/v1/chat/completions" else response

    try:
        if args.capture:
            post("/start_profile")
            active = True
        urllib.request.urlopen = instrumented_open
        sys.argv = ["onepass.py", "--name", args.name, "--ctx", str(args.ctx),
                    "--require-exclusive", "--seed", "7", "--num-spec", "5",
                    "--out", str(args.out / "onepass.jsonl")]
        rc = onepass.main()
        if rc:
            raise RuntimeError(f"onepass exited {rc}")
    finally:
        urllib.request.urlopen = opener
        try:
            if active:
                if stop_future is None:
                    post("/stop_profile")
                else:
                    stop_future.result(timeout=610)
        finally:
            pool.shutdown(wait=True)
            record["end_epoch"] = time.time()
            save()
    record["after"] = metrics()
    before, after = record["before"], record["after"]
    if before["prefix_hits"] is None or after["prefix_hits"] is None:
        record["issues"].append("prefix cache counters unavailable")
    elif after["prefix_hits"] - before["prefix_hits"] != 0:
        record["issues"].append("prefix cache hit during fresh-prefill request")
    rows = [json.loads(line) for line in (args.out / "onepass.jsonl").read_text().splitlines()]
    row = next(r for r in reversed(rows) if r["name"] == args.name)
    if row.get("quality") != {"ok": 3, "total": 3} or row.get("korean", {}).get("dirty") != 0:
        record["issues"].append("quality or Korean gate failed")
    if row.get("evidence_issues") or row.get("traffic", {}).get("issues"):
        record["issues"].append("onepass evidence or traffic gate failed")
    save()
    if record["issues"]:
        raise RuntimeError(str(record["issues"]))


if __name__ == "__main__":
    main()
