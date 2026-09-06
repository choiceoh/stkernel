#!/usr/bin/env python3
"""Prefix-cache hit test for GLM-5.3 + DFlash2 (39차, operator item 1).

For each context size: one COLD request on a Korean document (check-quality's
builder, three planted facts) and then a WARM request that reuses the exact
document prefix with a different question. Reports, per request:
  TTFT, prefix-cache hit/query counters delta, retrieval correctness, and the
  DFlash2 acceptance (raw acc and per-position) over that request's decode --
the last one is the point: a hit that skips the target forward leaves the
drafter's context KV unwritten unless the drafter's own window blocks are
reused (launcher note on vLLM #47926), so acceptance on the WARM request must
stay near the COLD one. A hit counter that rises while acceptance collapses is
worse than no cache.

    python3 probes/apc_hit_test.py [--ctx 32000,100000] [--max-tokens 300]
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "bench")
BASE = os.environ.get("GLM53_BASE", "http://127.0.0.1:8000")


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(BENCH, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    def g(name):
        m = re.search(r"^vllm:%s(\{[^}]*\})?\s+([0-9.e+]+)" % re.escape(name), t, re.M)
        return float(m.group(2)) if m else 0.0
    pos = [float(v) for v in re.findall(r"^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*\}\s+([0-9.e+]+)", t, re.M)]
    return {"hits": g("prefix_cache_hits_total") or g("prefix_cache_hits"),
            "queries": g("prefix_cache_queries_total") or g("prefix_cache_queries"),
            "drafts": g("spec_decode_num_drafts_total"), "draft_tokens": g("spec_decode_num_draft_tokens_total"),
            "accepted": g("spec_decode_num_accepted_tokens_total"), "pos": pos}


def ask(model, content, max_tokens):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": content}],
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; parts = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                obj = json.loads(line[5:].strip())
            except ValueError:
                continue
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(piece)
    return "".join(parts), (ttft if ttft is not None else time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="32000,100000")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    cq = _load("check-quality.py", "apc_quality")
    onepass = _load("onepass.py", "apc_onepass")
    model = cq.MODEL
    facts = cq.FACTS
    instr = onepass.INSTRUCTION
    print(f"{'ctx':>7} {'phase':<5} {'TTFT':>7} {'hit/query':>11} {'retrieval':>9} {'raw acc':>8} {'pos1':>5} {'pos3':>5} {'pos5':>5}  note")
    for ctx in (int(c) for c in args.ctx.split(",")):
        doc = cq.build(ctx, args.seed + ctx)
        for phase, qi in (("cold", 0), ("warm", 1), ("warm2", 2)):
            _, q, _ = facts[qi]
            m0 = metrics()
            text, ttft = ask(model, f"문서:\n{doc}\n\n{instr}{q}", args.max_tokens)
            m1 = metrics()
            low = text.lower()
            good = all(any(alt in low for alt in group) for group in onepass.FACT_EXPECT[qi])
            dd = m1["draft_tokens"] - m0["draft_tokens"]; da = m1["accepted"] - m0["accepted"]; dn = m1["drafts"] - m0["drafts"]
            acc = da / dd if dd else 0.0
            pos = [(b - a) / dn if dn else 0.0 for a, b in zip(m0["pos"], m1["pos"])]
            hq = f"{m1['hits'] - m0['hits']:.0f}/{m1['queries'] - m0['queries']:.0f}"
            note = "" if good else f"MISS {text[:60]!r}"
            print(f"{ctx:>7} {phase:<5} {ttft:>6.2f}s {hq:>11} {'o' if good else 'X':>9} {acc * 100:>7.1f}% "
                  f"{(pos[0] * 100 if pos else 0):>4.0f}% {(pos[2] * 100 if len(pos) > 2 else 0):>4.0f}% {(pos[4] * 100 if len(pos) > 4 else 0):>4.0f}%  {note}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
