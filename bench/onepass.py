#!/usr/bin/env python3
"""One workload, every gate (operator, 35차): the quality documents are the
prefill ladder AND the questions are answered in Korean, so the same nine
requests give prefill, retrieval quality, Korean corruption and the decode
stream at once.

  for ctx in 2K / 32K / 128K (check-quality.py's documents, facts planted at
  25/50/75% depth), for each of the three facts:
     one streaming request: "answer in Korean, two paragraphs, quote names /
     numbers / dates verbatim", max_tokens 400, temperature 0
       time to the first content chunk  -> prefill tok/s and TTFT (the first
                                           question of a context is the cold
                                           sample, the other two the warm ones;
                                           prefix caching is off on this fleet)
       the answer text                  -> retrieval check (any-of groups with
                                           Korean spellings) + corruption scan
                                           (korean-corruption.py's scanner)
       the engine's step counter,       -> decode windows (2 s, bracket.py's
       sampled throughout                  _StepWindows; windows that span a
                                           prefill read low and are dropped)
       spec-decode counters before/after-> raw acceptance, tokens/step

Nine answers of ~400 tokens are ~45 decode windows; the legacy Korean prompt
set (the campaign's known near-tie sites) can be appended with --korean-extra.
Prints the legs' numbers and appends one JSON record to
~/glm53-logs/bracket-onepass.jsonl. ~5 min on a healthy boot.

    python3 bench/onepass.py --name PRODV3 [--ctx 2000,32000,128000] [--korean-extra]
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# accepted answer forms: every group must match by at least one of its spellings
FACT_EXPECT = [
    [["8127"]],
    [["halvorsen", "할보르센", "할버슨"], ["14 march 1997", "march 14, 1997", "1997년 3월 14일", "1997-03-14", "3월 14일"]],
    [["k-42", "k42"], ["north", "북쪽", "북측", "북벽", "북면"]],
]
INSTRUCTION = ("문서의 내용만 근거로 한국어로 두 문단 정도로 답해줘. 이름·숫자·날짜·장비 번호 같은 고유 표기는 "
               "문서에 적힌 그대로 인용해줘.\n질문: ")


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics_text(url):
    return urllib.request.urlopen(url, timeout=5).read().decode()


def _counter(text, name):
    m = re.search(r"^vllm:%s\{[^}]*\}\s+([0-9.e+]+)" % re.escape(name), text, re.M)
    return float(m.group(1)) if m else 0.0


def ask_stream(url, model, content, max_tokens):
    """(text, ttft_s, prompt_tokens, completion_tokens, finish_reason) of one
    streamed chat completion: ttft = first chunk carrying content."""
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0.0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": content}],
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    parts = []
    usage = {}
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    if ttft is None:
        ttft = time.time() - t0
    return ("".join(parts), ttft, int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0), finish)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="onepass")
    ap.add_argument("--ctx", default=os.environ.get("QUALITY_CTX", "2000,32000,128000"))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--num-spec", type=int, default=int(os.environ.get("SPEC_K", "7")))
    ap.add_argument("--korean-extra", action="store_true",
                    help="also run the legacy 8 Korean prompts x 2 rounds (the known near-tie sites)")
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
    texts = []          # (tag, text, finish) for the corruption scan
    phases = []         # (ctx, t_first_token, t_end): each answer's decode phase
    quality_ok = quality_total = 0
    gen_tokens = 0

    m0 = bd._parse_spec_metrics(_metrics_text(bd.METRICS))
    print(f"{'ctx':>7} {'tok':>7} {'cold tok/s':>11} {'warm tok/s':>11} {'cold TTFT':>10} {'warm TTFT':>10}  quality", flush=True)
    t_dec0 = time.time()
    with br._StepWindows(bd) as sw:
        for ctx in (int(c) for c in args.ctx.split(",")):
            doc = cq.build(ctx, args.seed + ctx)
            ttfts, hits, tok = [], [], 0
            for qi, (_, q, _old) in enumerate(cq.FACTS):
                content = f"문서:\n{doc}\n\n{INSTRUCTION}{q}"
                t_req = time.monotonic()
                text, ttft, ptok, ctok, finish = ask_stream(cq.URL, cq.MODEL, content, args.max_tokens)
                phases.append((ctx, t_req + ttft, time.monotonic()))
                tok = ptok or tok
                gen_tokens += ctok
                ttfts.append(ttft)
                low = text.lower()
                good = all(any(alt in low for alt in group) for group in FACT_EXPECT[qi])
                hits.append("o" if good else "X")
                quality_total += 1
                quality_ok += good
                if not good:
                    print(f"    MISS ctx~{ctx // 1000}K q={q!r} -> {text[:100]!r}", flush=True)
                texts.append((f"ctx{ctx // 1000}K q{qi}", text, finish))
            cold, warm = ttfts[0], min(ttfts[1:]) if len(ttfts) > 1 else ttfts[0]
            rec["prefill"].append({"ctx": ctx, "tok": tok, "cold_s": cold, "warm_s": warm,
                                   "cold_tok_s": tok / cold if cold > 0 else 0.0,
                                   "warm_tok_s": tok / warm if warm > 0 else 0.0})
            print(f"{ctx:>7} {tok:>7} {tok / cold:>11.0f} {tok / warm:>11.0f} {cold:>9.2f}s {warm:>9.2f}s  {' '.join(hits)}", flush=True)
        if args.korean_extra:
            for r in range(2):
                for i, p in enumerate(kq.PROMPTS):
                    t_req = time.monotonic()
                    text, finish = kq.ask(p, args.max_tokens)
                    phases.append((0, t_req + 0.5, time.monotonic()))   # short prompts: ~0 prefill
                    texts.append((f"ko r{r} p{i}", text, finish))
    wall = time.time() - t_dec0
    m1 = bd._parse_spec_metrics(_metrics_text(bd.METRICS))
    rows = rec["prefill"]
    if len(rows) >= 2:
        print(f"  warm 처리량: {rows[0]['warm_tok_s']:.0f} -> {rows[-1]['warm_tok_s']:.0f} tok/s "
              f"(over {rows[0]['tok']} -> {rows[-1]['tok']} tokens)")
    rec["quality"] = {"ok": quality_ok, "total": quality_total}
    print(f"=> {quality_ok}/{quality_total} correct", flush=True)

    # ---- decode: only windows that lie INSIDE an answer's decode phase (after
    # its first token, before its end) count, bucketed by the context length --
    # decode at 128K context is heavier than at 2K, and the legs' number is
    # the short-context one. Acceptance from the counters over the whole run.
    legacy, raw = br._spec_delta(m0, m1)
    k_eff = br.spec_k_eff(m0, m1) or args.num_spec   # 37차: the served k, not the flag
    samp = sw.samples
    by_ctx = {}
    for i in range(len(samp) - 1):
        (ta, sa), (tb, sb) = samp[i], samp[i + 1]
        for ctx, t0p, t1p in phases:
            # 1 s margins: the first window after the first token still holds
            # prefill tail, the last one before the end holds the stream's
            # close (3 windows per answer make one low edge window the median)
            if ta >= t0p + 1.0 and tb <= t1p - 1.0:
                r = (sb - sa) / max(tb - ta, 1e-6)
                if r > 0:
                    by_ctx.setdefault(ctx, []).append(r)
                break
    rates = [r for v in by_ctx.values() for r in v]
    win_med = median(rates) if rates else None
    if rates:
        per = "  ".join(f"{('ko' if c == 0 else str(c // 1000) + 'K')}: n={len(v)} med {median(v):.1f}"
                        for c, v in sorted(by_ctx.items()))
        print(f"decode: windows n={len(rates)} med {win_med:.1f} [{min(rates):.1f}, {max(rates):.1f}] step/s "
              f"(inside the answers only; per context: {per}), raw acc {(raw or 0) * 100:.1f}%, "
              f"tokens/step {1 + k_eff * (raw or 0):.3f} (k={k_eff:.1f}), generated {gen_tokens} tokens over {wall:.0f}s",
              flush=True)
    else:
        print("decode: no windows", flush=True)
    rec["decode"] = {"gen_tokens": gen_tokens, "wall_s": wall, "acc_raw": raw, "acc_legacy": legacy,
                     "tokens_per_step": 1 + k_eff * (raw or 0), "windows": rates,
                     "windows_med": win_med, "windows_by_ctx": {str(k): v for k, v in by_ctx.items()},
                     "num_spec": k_eff}

    # ---- Korean corruption on every answer
    dirty, chars, kinds_tot = [], 0, {}
    for tag, text, finish in texts:
        chars += len(text)
        h = kq.scan(text, truncated=(finish == "length"))
        gated = {k: v for k, v in h.items() if k not in kq.INFORMATIONAL}
        for k, v in gated.items():
            kinds_tot[k] = kinds_tot.get(k, 0) + v
        kinds = {k: v for k, v in gated.items() if v}
        if kinds:
            dirty.append((tag, kinds, text))
    n = len(texts)
    print(f"응답 {n}개 · 문자 {chars:,}")
    print(f"  깨진 응답: {len(dirty)}/{n} ({100 * len(dirty) / max(n, 1):.0f}%)")
    for k in ("replacement", "lone_jamo", "cjk_mixed", "control"):
        v = kinds_tot.get(k, 0)
        print(f"  {k:<14}{v:>4}  {1e6 * v / max(chars, 1):6.1f}/백만자")
    for tag, kinds, text in dirty:
        ks = " ".join(f"{k}={v}" for k, v in kinds.items())
        shown = False
        for k in kinds:
            pat = {"cjk_mixed": kq.HAN, "lone_jamo": kq.WELDED_JAMO}.get(k)
            m = pat.search(text) if pat is not None else None
            if m:
                a = max(0, m.start() - 40)
                print(f"\n  [{tag}] {ks}\n    …{text[a:m.end() + 40]!r}…")
                shown = True
                break
        if not shown:
            print(f"\n  [{tag}] {ks}")
    rec["korean"] = {"dirty": len(dirty), "n": n, "kinds": kinds_tot,
                     "hits": [(tag, k) for tag, k, _ in dirty]}

    print(f"== onepass {args.name}: {time.time() - t_all:.0f}s total", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
