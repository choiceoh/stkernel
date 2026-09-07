#!/usr/bin/env python3
"""Run canonical onepass with fresh prefix-cache identity for every request.

Metrics requests are outside ask_stream's measured interval. Model inputs and
sampling stay unchanged; wire and unsalted hashes are saved separately. This
does not boot/deploy anything. Use the owned fleet arm and onepass_memory.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.request
import uuid


class FreshRequests:
    def __init__(self, url, ask, opener, metrics):
        self.url, self.ask, self.opener, self.metrics = url, ask, opener, metrics
        self.records = []
        self.active = None

    def open(self, request, *args, **kwargs):
        if getattr(request, "full_url", request) != self.url:
            return self.opener(request, *args, **kwargs)
        if self.active is None:
            raise RuntimeError("untracked model request")
        body = request.data
        payload = json.loads(body)
        payload["cache_salt"] = self.active["cache_salt"]
        wire = json.dumps(payload).encode()
        self.active.update(unsalted_sha256=hashlib.sha256(body).hexdigest(),
                           wire_sha256=hashlib.sha256(wire).hexdigest())
        salted = urllib.request.Request(self.url, data=wire,
                                        headers=dict(request.header_items()), method=request.get_method())
        return self.opener(salted, *args, **kwargs)

    def call(self, *args, **kwargs):
        before = self.metrics()
        if before["traffic"]["running"] != 0 or before["traffic"]["waiting"] != 0:
            raise RuntimeError("fresh prefill requires idle serving before each request")
        entry = dict(cache_salt="prefill-fresh:" + uuid.uuid4().hex, before=before, issues=[])
        self.records.append(entry)
        self.active = entry
        try:
            result = self.ask(*args, **kwargs)
            entry.update(ttft_s=result[1], prompt_tokens=result[2], completion_tokens=result[3])
        except BaseException as exc:
            entry["error"] = repr(exc)
            raise
        finally:
            self.active = None
        after = self.metrics()
        entry["after"] = after
        hits = [state["prefix_hits"] for state in (before, after)]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in hits):
            entry["issues"].append("prefix cache counters unavailable or invalid")
        elif hits[1] != hits[0]:
            entry["issues"].append("prefix cache hit or counter reset during fresh request")
        if not entry.get("wire_sha256") or entry["prompt_tokens"] <= 0:
            entry["issues"].append("request hash or token evidence missing")
        if entry["issues"]:
            raise RuntimeError(str(entry["issues"]))
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctx", default="2000,32000,128000")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        ap.error("name must contain only letters, digits, underscore or hyphen")
    args.out.mkdir(parents=True, exist_ok=True)
    import onepass
    from window_metrics import metric_sum, traffic_state
    base = os.environ.get("HEAD_URL", "http://127.0.0.1:18000").rstrip("/")
    opener = urllib.request.urlopen
    with opener(base + "/openapi.json", timeout=10) as response:
        api = json.load(response)
    if "cache_salt" not in api["components"]["schemas"]["ChatCompletionRequest"]["properties"]:
        raise RuntimeError("request-level cache isolation is unsupported")

    def metrics():
        with opener(base + "/metrics", timeout=10) as response:
            text = response.read().decode()
        return dict(traffic=traffic_state(text), prefix_hits=metric_sum(text, "vllm:prefix_cache_hits_total"),
                    prefix_queries=metric_sum(text, "vllm:prefix_cache_queries_total"))

    original_ask = onepass.ask_stream
    original_argv = sys.argv
    fresh = FreshRequests(base + "/v1/chat/completions", original_ask, opener, metrics)
    try:
        urllib.request.urlopen = fresh.open
        onepass.ask_stream = fresh.call
        sys.argv = ["onepass.py", "--name", args.name, "--ctx", args.ctx,
                    "--require-exclusive", "--seed", "7", "--num-spec", "5",
                    "--out", str(args.out / "onepass.jsonl")]
        return onepass.main()
    finally:
        onepass.ask_stream = original_ask
        urllib.request.urlopen = opener
        sys.argv = original_argv
        report = dict(schema=1, name=args.name, requests=fresh.records,
                      note="Every TTFT sample is fresh prefill; later samples may have warm compilation.")
        (args.out / (args.name + ".fresh.json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
