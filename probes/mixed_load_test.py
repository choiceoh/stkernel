#!/usr/bin/env python3
"""Mixed-load probe: one decoder's inter-token latency while a long prompt
prefills next to it (39차, operator item 2 -- the decode-first scheduler).

Per context size:
  1. a streaming DECODER request (2K Korean document, long answer) starts;
  2. once it has produced --settle tokens, a PREFILL request (ctx-size Korean
     document, short answer) is fired;
  3. both run to completion.
Reported per context: the decoder's inter-token gaps (median / p95 / max) over
the prefill's lifetime and overall, tokens the decoder produced while the
prefill was in flight, the prefill's TTFT and prompt throughput, and a solo
decoder reference (no prefill) measured first.

    python3 probes/mixed_load_test.py [--ctx 32000,100000] [--decode-tokens 600]
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import threading
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


def stream(model, content, max_tokens, stamps, done):
    """Append an arrival timestamp per content chunk; set done[0] at the end."""
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": content}],
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        stamps.append(time.time())
    finally:
        done[0] = time.time()
        done.append(usage)


def gaps(stamps, lo=None, hi=None):
    pts = [t for t in stamps if (lo is None or t >= lo) and (hi is None or t <= hi)]
    return [b - a for a, b in zip(pts, pts[1:])]


def summarize(g):
    if not g:
        return "   -      -      -  "
    g = sorted(g)
    return f"{statistics.median(g) * 1000:5.0f} {g[int(0.95 * (len(g) - 1))] * 1000:6.0f} {g[-1] * 1000:6.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="32000,100000")
    ap.add_argument("--decode-tokens", type=int, default=600)
    ap.add_argument("--settle", type=int, default=25, help="decoder tokens before the prefill is fired")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    cq = _load("check-quality.py", "mixed_quality")
    model = cq.MODEL
    story = ("문서를 바탕으로, 등장인물과 장소를 살려 한국어로 아주 긴 이야기를 써줘. "
             "최소 스무 문단 이상, 멈추지 말고 계속 이어서 써줘.")
    decoder_doc = cq.build(2000, args.seed)
    decoder_prompt = f"문서:\n{decoder_doc}\n\n{story}"

    # Solo reference.
    stamps, done = [], [None]
    stream(model, decoder_prompt, args.decode_tokens, stamps, done)
    solo = gaps(stamps)
    solo_tokens = (done[1] or {}).get("completion_tokens", 0) if len(done) > 1 else 0
    solo_s = (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    print(f"solo decoder: {len(stamps)} steps, {solo_tokens} tokens ({solo_tokens / solo_s if solo_s else 0:.0f} tok/s), "
          f"ITL med/p95/max ms = {summarize(solo)}", flush=True)
    print(f"{'ctx':>7} {'dec steps/tok':>13} {'ITL(all) med p95 max':>22} {'ITL(overlap) med p95 max':>26} {'steps@ovl':>9} {'dec tok/s@ovl':>13} {'prefill TTFT':>12} {'prompt tok/s':>12}")
    for ctx in (int(c) for c in args.ctx.split(",")):
        doc = cq.build(ctx, args.seed + ctx)
        _, q, _ = cq.FACTS[0]
        prefill_prompt = f"문서:\n{doc}\n\n문서의 내용만 근거로 한 문장으로 답해줘.\n질문: {q}"
        d_stamps, d_done = [], [None]
        t = threading.Thread(target=stream, args=(model, decoder_prompt, args.decode_tokens, d_stamps, d_done), daemon=True)
        t.start()
        while len(d_stamps) < args.settle and d_done[0] is None:
            time.sleep(0.02)
        p_stamps, p_done = [], [None]
        t0 = time.time()
        stream(model, prefill_prompt, 48, p_stamps, p_done)
        ttft = (p_stamps[0] - t0) if p_stamps else float("nan")
        t.join()
        prompt_tokens = (p_done[1] or {}).get("prompt_tokens") if len(p_done) > 1 else None
        if not prompt_tokens:  # usage missing: estimate from the Korean filler ratio
            prompt_tokens = int(len(prefill_prompt) / cq.KO_CHARS_PER_TOKEN)
        rate = (prompt_tokens / ttft) if prompt_tokens and ttft == ttft and ttft > 0 else float("nan")
        overlap = gaps(d_stamps, t0, p_done[0])
        n_overlap = sum(1 for s in d_stamps if t0 <= s <= p_done[0])
        d_tokens = (d_done[1] or {}).get("completion_tokens", 0) if len(d_done) > 1 else 0
        per_step = (d_tokens / len(d_stamps)) if d_stamps else 0.0
        ovl_s = max(p_done[0] - t0, 1e-9)
        ovl_tps = n_overlap * per_step / ovl_s   # tokens/step from the whole answer, steps counted inside the overlap
        print(f"{ctx:>7} {len(d_stamps):>6}/{d_tokens:<6} {summarize(gaps(d_stamps)):>22} {summarize(overlap):>26} {n_overlap:>9} {ovl_tps:>13.1f} {ttft:>11.2f}s {rate:>12.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
