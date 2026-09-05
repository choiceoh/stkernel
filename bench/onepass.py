#!/usr/bin/env python3
"""One pass, every gate: prefill ladder + quality + decode windows +
acceptance + Korean corruption from ONE workload (operator, 35차).

The separate legs each made their own traffic (35 min); here each request
serves two purposes:

  phase A  the quality documents (2K / 32K / 128K, facts planted at 25/50/75%
           depth -- check-quality.py's builder) ARE the prefill ladder: per
           context a max_tokens=1 request twice (cold, warm: prefix caching is
           off on this fleet, so "warm" is the JIT/L2-warm repeat) gives the
           ladder's cold/warm tok/s and TTFT, then the three questions give
           the retrieval score;
  phase B  the Korean prompts (8 x 2 rounds, temperature 0, 400 tokens --
           korean-corruption.py's) ARE the decode stream: the engine's step
           counter is sampled every 2 s while they generate (bracket.py's
           _StepWindows: each full window is one step/s sample, ~70 of them
           instead of 3 x 9), the spec-decode counters before/after give the
           raw acceptance and tokens/step, and the responses are scanned for
           corruption.

Prints the same numbers the legs did (windows median/spread, tok/s, raw acc,
warm throughput, N/9, N/16) and appends one JSON record to
~/glm53-logs/bracket-onepass.jsonl. ~12 min on a healthy boot.

    python3 bench/onepass.py --name PRODV3 [--ctx 2000,32000,128000] [--rounds 2]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics_text(url):
    return urllib.request.urlopen(url, timeout=5).read().decode()


def _gen_tokens(text):
    import re
    m = re.search(r"^vllm:generation_tokens_total\{[^}]*\}\s+([0-9.e+]+)", text, re.M)
    return float(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="onepass")
    ap.add_argument("--ctx", default=os.environ.get("QUALITY_CTX", "2000,32000,128000"))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--num-spec", type=int, default=int(os.environ.get("SPEC_K", "7")))
    ap.add_argument("--out", default=os.path.expanduser("~/glm53-logs/bracket-onepass.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    kq = _load("korean-corruption.py", "onepass_korean")
    cq = _load("check-quality.py", "onepass_quality")
    bd = _load("bench-dec.py", "onepass_bench_dec")
    br = _load("bracket.py", "onepass_bracket")
    rec = {"name": args.name, "t": time.strftime("%F %T"), "git": br._git_sha(),
           "prefill": [], "quality": {}, "decode": {}, "korean": {}}
    t_all = time.time()

    # ---- phase A: prefill ladder on the quality documents + retrieval
    print(f"{'ctx':>7} {'tok':>7} {'cold tok/s':>11} {'warm tok/s':>11} {'cold TTFT':>10} {'warm TTFT':>10}  quality", flush=True)
    ok = total = 0
    for ctx in (int(c) for c in args.ctx.split(",")):
        doc = cq.build(ctx, args.seed + ctx)
        samples = []
        for _ in range(2):
            body = json.dumps({"model": cq.MODEL, "max_tokens": 1, "temperature": 0.0,
                               "messages": [{"role": "user", "content": doc}],
                               "chat_template_kwargs": {"thinking": False}}).encode()
            req = urllib.request.Request(cq.URL, data=body, headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.load(r)
            samples.append((int(out["usage"]["prompt_tokens"]), time.time() - t0))
        tok = samples[0][0]
        cold, warm = samples[0][1], min(s[1] for s in samples[1:])
        hits = []
        for _, q, expect in cq.FACTS:
            ans = cq.ask(doc, q).lower()
            good = all(e in ans for e in expect)
            hits.append("o" if good else "X")
            total += 1
            ok += good
            if not good:
                print(f"    MISS ctx~{ctx // 1000}K q={q!r} -> {ans[:100]!r}", flush=True)
        rec["prefill"].append({"ctx": ctx, "tok": tok, "cold_s": cold, "warm_s": warm,
                               "cold_tok_s": tok / cold, "warm_tok_s": tok / warm})
        print(f"{ctx:>7} {tok:>7} {tok / cold:>11.0f} {tok / warm:>11.0f} {cold:>9.2f}s {warm:>9.2f}s  {' '.join(hits)}", flush=True)
    rows = rec["prefill"]
    if len(rows) >= 2:
        print(f"  warm 처리량: {rows[0]['warm_tok_s']:.0f} -> {rows[-1]['warm_tok_s']:.0f} tok/s "
              f"(over {rows[0]['tok']} -> {rows[-1]['tok']} tokens)")
    rec["quality"] = {"ok": ok, "total": total}
    print(f"=> {ok}/{total} correct", flush=True)

    # ---- phase B: Korean generation as the decode stream
    m0 = bd._parse_spec_metrics(_metrics_text(bd.METRICS))
    g0 = _gen_tokens(_metrics_text(bd.METRICS)) or 0.0
    texts = []
    t0 = time.time()
    with br._StepWindows(bd) as sw:
        for r in range(args.rounds):
            for i, p in enumerate(kq.PROMPTS):
                text, finish = kq.ask(p, args.max_tokens)
                texts.append((r, i, text, finish))
    wall = time.time() - t0
    m1 = bd._parse_spec_metrics(_metrics_text(bd.METRICS))
    g1 = _gen_tokens(_metrics_text(bd.METRICS)) or 0.0
    gen = g1 - g0
    legacy, raw = br._spec_delta(m0, m1)
    rates = sw.rates()
    tok_s = gen / wall if wall > 0 else 0.0
    step_s = br.step_s_of(tok_s, raw, args.num_spec)
    win_med = median(rates) if rates else None
    print(f"decode: {tok_s:.1f} tok/s over {gen:.0f} tokens / {wall:.0f}s, raw acc "
          f"{(raw or 0) * 100:.1f}%, tokens/step {1 + args.num_spec * (raw or 0):.3f}, "
          f"step/s {step_s if step_s is None else round(step_s, 1)}, windows n={len(rates)} "
          f"med {win_med if win_med is None else round(win_med, 1)} "
          f"[{min(rates):.1f}, {max(rates):.1f}]" if rates else "decode: no windows", flush=True)
    rec["decode"] = {"tok_s": tok_s, "gen_tokens": gen, "wall_s": wall, "acc_raw": raw,
                     "acc_legacy": legacy, "step_s": step_s, "windows": rates,
                     "windows_med": win_med, "num_spec": args.num_spec}

    # ---- Korean corruption on the same responses
    dirty = []
    chars = 0
    kinds_tot = {}
    for r, i, text, finish in texts:
        chars += len(text)
        h = kq.scan(text, truncated=(finish == "length"))
        gated = {k: v for k, v in h.items() if k not in kq.INFORMATIONAL}
        kinds = {k: v for k, v in gated.items() if v}
        for k, v in gated.items():
            kinds_tot[k] = kinds_tot.get(k, 0) + v
        if kinds:
            dirty.append((r, i, kinds, text))
    n = len(texts)
    print(f"응답 {n}개 · 문자 {chars:,}")
    print(f"  깨진 응답: {len(dirty)}/{n} ({100 * len(dirty) / max(n, 1):.0f}%)")
    for k in ("replacement", "lone_jamo", "cjk_mixed", "control"):
        v = kinds_tot.get(k, 0)
        print(f"  {k:<14}{v:>4}  {1e6 * v / max(chars, 1):6.1f}/백만자")
    for r, i, kinds, text in dirty:
        ks = " ".join(f"{k}={v}" for k, v in kinds.items())
        for k in kinds:
            pat = {"cjk_mixed": kq.HAN, "lone_jamo": kq.WELDED_JAMO}.get(k)
            m = pat.search(text) if pat is not None else None
            if m:
                a = max(0, m.start() - 40)
                print(f"\n  [round {r} prompt {i}] {ks}\n    …{text[a:m.end() + 40]!r}…")
                break
        else:
            print(f"\n  [round {r} prompt {i}] {ks}")
    rec["korean"] = {"dirty": len(dirty), "n": n, "kinds": kinds_tot,
                     "hits": [(r, i, k) for r, i, k, _ in dirty]}

    print(f"== onepass {args.name}: {time.time() - t_all:.0f}s total", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
