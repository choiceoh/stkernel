#!/usr/bin/env python3
"""THE test (operator, 39차: "원패스 한국어 본문판만 남기고 모든 테스트는 저거
한 개로 통일"). One Korean workload gives every gate at once:

  for ctx in 2K / 32K / 128K: a KOREAN document (check-quality.py: Korean
  Wikipedia paragraphs from bench/ko_filler.txt in seeded order, three facts
  planted in Korean at 25/50/75% depth)
     2K:        three streaming requests, one question each (the first is the
                cold prefill sample, the other two the warm ones; prefix
                caching is off on this fleet)
     32K, 128K: ONE streaming request carrying the three questions (the
                document is prefilled once; a 128K prefill is ~48 s), three
                answers' worth of tokens (max_tokens x 3)
       time to the first content chunk  -> prefill tok/s and TTFT
       the answer text                  -> retrieval (any-of groups with the
                                           Korean spellings) + corruption scan
                                           (korean-corruption.py's scanner)
       the engine's step counter,       -> decode windows (2 s, bracket.py's
       sampled throughout                  _StepWindows; windows that span a
                                           prefill read low and are dropped)
       spec-decode counters before/after-> raw acceptance, tokens/step

There is no other leg: no English document, no separate Korean prompt set,
no decode-only bracket, no C>1 arm. ~2.5 min on a healthy boot. Prints the
numbers and appends one JSON record (harness 39) to
~/glm53-logs/bracket-onepass.jsonl. Records before 2026-09-06 15:30 (English
word-salad documents, separate Korean set) compare only on decode windows,
raw acc and tokens/step.

    python3 bench/onepass.py --name PRODV3 [--ctx 2000,32000,128000]

Optional C=1 follow-up: --fixed-decode-tokens 2048 --fixed-decode-reps 3
--require-exclusive adds fixed-length 2K responses to this same workload.
Their interior windows are the primary decode metric; the normal context
ladder and all quality checks remain. Counter mismatches invalidate the run.
"""
import argparse
import hashlib
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
# 38차: the three questions of a context in ONE request (one prefill instead
# of three): numbered answers of about two paragraphs each
INSTRUCTION_COMBINED = ("문서의 내용만 근거로 아래 질문 세 개에 한국어로 번호를 붙여 각각 두 문단 정도로 답해줘. "
                        "이름·숫자·날짜·장비 번호 같은 고유 표기는 문서에 적힌 그대로 인용해줘.\n질문:\n")


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


def ask_stream(url, model, content, max_tokens, timing=None, min_tokens=0, seed=None):
    """(text, ttft_s, prompt_tokens, completion_tokens, finish_reason) of one
    streamed chat completion: ttft = first chunk carrying content."""
    body = json.dumps({"model": model, "max_tokens": max_tokens, "min_tokens": min_tokens,
                       "seed": seed, "temperature": 0.0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": content}],
                       # 39차: thinking ON, explicitly. The stock template ignored this
                       # kwarg and always reasoned, so every reference (BASE39-*, DEF40, ...)
                       # was measured with reasoning in the stream; the v2 template honours
                       # the kwarg and thinking=false gives answers too short for the 2 s
                       # decode windows (TPL1: no windows). Keep the condition constant.
                       "chat_template_kwargs": {"thinking": True}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    arrivals = []
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
                    arrived = time.monotonic()
                    arrivals.append(arrived)
                    if ttft is None:
                        ttft = arrived - t0
                    parts.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    if ttft is None:
        ttft = time.monotonic() - t0
    if timing is not None:
        elapsed = time.monotonic() - t0
        ctok = int(usage.get("completion_tokens", 0) or 0)
        decode_s = elapsed - ttft
        # Standard request TPOT includes the final stream/usage tail. SSE
        # chunks can contain several speculative tokens; gaps are NOT ITL.
        timing.update(completion_tokens=ctok, prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                      min_tokens=min_tokens, max_tokens=max_tokens, seed=seed,
                      request_sha256=hashlib.sha256(body).hexdigest(),
                      output_sha256=hashlib.sha256("".join(parts).encode()).hexdigest(),
                      ttft_s=ttft, elapsed_s=elapsed,
                      decode_s=decode_s, finish_reason=finish,
                      tpot_ms=1000 * decode_s / (ctok - 1) if ctok > 1 else None,
                      decode_tok_s=(ctok - 1) / decode_s if ctok > 1 and decode_s > 0 else None,
                      chunk_gaps_ms=[1000 * (b - a) for a, b in zip(arrivals, arrivals[1:])])
    return ("".join(parts), ttft, int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0), finish)


def _served_build(repo: str, profile: str = "glm53") -> dict:
    """What the SERVER is running: the deployed overlay stamp and the knobs
    that differ from the profile's defaults.

    NOT this process's environment -- ab-lever.sh boots the server with the
    arm's env and then runs this bench in a plain shell, so os.environ here
    carries none of it. The serving container's own Config.Env is the only
    honest source, and the overlay stamp identifies the BUILD (a bench with
    no deploy reuses the previous build whatever the git sha says).
    An empty `knobs` IS that build's baseline -- bench/baseline.py reads it
    so the next session can skip re-measuring one. Every failure degrades to
    a missing field: a bench must never die over its own label.
    """
    import subprocess
    out = {}
    try:
        stamp = os.environ.get("MK_OVERLAY_STAMP",
                               "/home/choiceoh/glm53-cache/.overlay-sha")
        with open(stamp) as fh:
            out["overlay"] = fh.read().strip()[:12]
    except Exception:
        pass
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True,
                               timeout=10).stdout.split()
        name = next((n for n in names if n.startswith("glm53")), None)
        if not name:
            return out
        boot = subprocess.run(["docker", "inspect", "-f", "{{.Id}}|{{.State.StartedAt}}", name],
                              capture_output=True, text=True, timeout=10)
        if boot.returncode == 0 and "|" in boot.stdout.strip():
            out["boot_id"] = boot.stdout.strip()
        raw = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Env}}", name],
                             capture_output=True, text=True, timeout=10).stdout
        served = dict(e.split("=", 1) for e in json.loads(raw or "[]")
                      if "=" in e and e.startswith("VLLM_"))
        declared = {}
        with open(os.path.join(repo, "profiles", profile + ".env")) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("VLLM_") and "=" in line:
                    k, v = line.split("=", 1)
                    declared[k] = v.strip().strip('"')
                elif line.startswith("SPEC_K="):
                    # The launcher exports this profile setting under a
                    # VLLM alias for the compile-cache key. Matching values
                    # belong to the baseline, not an experimental knob.
                    declared["VLLM_GLM53_SPEC_K"] = line.split("=", 1)[1].strip().strip('"')
        knobs = {k: v for k, v in served.items() if k in declared and v != declared[k]}
        knobs.update({k: v for k, v in served.items()
                      if k.startswith("VLLM_GLM53_") and k not in declared})
        out["knobs"] = dict(sorted(knobs.items()))
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="onepass")
    ap.add_argument("--ctx", default=os.environ.get("QUALITY_CTX", "2000,32000,128000"))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--num-spec", type=int, default=int(os.environ.get("SPEC_K", "7")))
    ap.add_argument("--combine-min-ctx", type=int, default=int(os.environ.get("ONEPASS_COMBINE_MIN_CTX", "32000")),
                    help="contexts at or above this size ask the three questions in ONE request (one prefill "
                         "instead of three; the fleet has no prefix cache). 0 = never combine")
    ap.add_argument("--out", default=os.environ.get("ONEPASS_JSONL",
                                                   os.path.expanduser("~/glm53-logs/bracket-onepass.jsonl")))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fixed-decode-tokens", type=int, default=int(os.environ.get("ONEPASS_FIXED_DECODE_TOKENS", "0")))
    ap.add_argument("--fixed-decode-reps", type=int, default=int(os.environ.get("ONEPASS_FIXED_DECODE_REPS", "3")))
    ap.add_argument("--require-exclusive", action="store_true", default=os.environ.get("ONEPASS_REQUIRE_EXCLUSIVE") == "1")
    args = ap.parse_args()
    if args.fixed_decode_tokens < 0 or args.fixed_decode_reps < 1:
        ap.error("fixed decode needs nonnegative tokens and positive repetitions")

    kq = _load("korean-corruption.py", "onepass_korean")
    cq = _load("check-quality.py", "onepass_quality")
    bd = _load("bench-dec.py", "onepass_bench_dec")
    br = _load("bracket.py", "onepass_bracket")
    rec = {"name": args.name, "t": time.strftime("%F %T"), "git": br._git_sha(),
           "harness": 40, "doc_lang": "ko", "thinking": True, "window_s": 1.0,
           "prefill": [], "quality": {}, "decode": {}, "korean": {}}
    rec["workload"] = {"ctx": [int(c) for c in args.ctx.split(",")], "seed": args.seed,
                       "max_tokens": args.max_tokens, "combine_min_ctx": args.combine_min_ctx,
                       "fixed_decode_tokens": args.fixed_decode_tokens,
                       "fixed_decode_reps": args.fixed_decode_reps if args.fixed_decode_tokens else 0,
                       "require_exclusive": args.require_exclusive}
    if os.environ.get("FLEET_EXPERIMENT_ID"):
        rec["experiment_id"] = os.environ["FLEET_EXPERIMENT_ID"]
        rec["runtime"] = json.loads(os.environ.get("FLEET_CONTEXT", "{}"))
    rec["endpoint"] = {"completion": bd.URL, "metrics": bd.METRICS}
    rec.update(_served_build(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if os.environ.get("FLEET_SESSION"):
        rec["session"] = os.environ["FLEET_SESSION"]          # who held the fleet (fleet.sh run)
    if os.environ.get("MK_COLD_COMPILE") == "1":
        rec["cold_compile"] = True                              # first boot on this build (ab-lever)
    t_all = time.time()
    texts = []          # (tag, text, finish) for the corruption scan
    phases = []         # (ctx, t_first_token, t_end): each answer's decode phase
    fixed_phases = []
    rec["requests"] = []
    quality_ok = quality_total = 0
    gen_tokens = 0

    from window_metrics import traffic_state, exclusive_errors, decode_windows
    metrics_before = _metrics_text(bd.METRICS)
    before_traffic = traffic_state(metrics_before)
    if args.require_exclusive and (before_traffic["running"] != 0 or before_traffic["waiting"] != 0):
        raise RuntimeError("exclusive onepass requires an idle server before sending requests")
    m0 = bd._parse_spec_metrics(metrics_before)
    print(f"{'ctx':>7} {'tok':>7} {'cold tok/s':>11} {'warm tok/s':>11} {'cold TTFT':>10} {'warm TTFT':>10}  quality", flush=True)
    t_dec0 = time.time()
    # 39차: 1 s windows (were 2 s) with 0.5 s margins -- the 2K answers decode
    # for ~3-4 s and produced 0-3 windows per boot; medians of step/s are
    # comparable across window sizes.
    with br._StepWindows(bd, period=1.0) as sw:
        facts = cq.FACTS
        for ctx in (int(c) for c in args.ctx.split(",")):
            doc = cq.build(ctx, args.seed + ctx)
            ttfts, hits, tok = [], [], 0
            combined = bool(args.combine_min_ctx) and ctx >= args.combine_min_ctx
            if combined:
                # 39차: one request carries the three questions, so the document
                # is prefilled ONCE (the 128K document cost 3 x 48 s before);
                # three answers' worth of decode keeps the window count. The
                # cold / warm TTFT pair survives at the contexts below the cut.
                qs = "\n".join(f"{qi + 1}. {q}" for qi, (_, q, _old) in enumerate(facts))
                content = f"문서:\n{doc}\n\n{INSTRUCTION_COMBINED}{qs}"
                t_req = time.monotonic()
                timing = {"ctx": ctx, "question": "all"}
                text, ttft, ptok, ctok, finish = ask_stream(bd.URL, cq.MODEL, content, args.max_tokens * len(facts), timing)
                rec["requests"].append(timing)
                phases.append((ctx, t_req + ttft, time.monotonic()))
                tok = ptok or tok
                gen_tokens += ctok
                ttfts.append(ttft)
                low = text.lower()
                for qi, (_, q, _old) in enumerate(facts):
                    good = all(any(alt in low for alt in group) for group in FACT_EXPECT[qi])
                    hits.append("o" if good else "X")
                    quality_total += 1
                    quality_ok += good
                    if not good:
                        print(f"    MISS ctx~{ctx // 1000}K q={q!r} (combined) -> {text[:100]!r}", flush=True)
                texts.append((f"ctx{ctx // 1000}K q-all", text, finish))
            for qi, (_, q, _old) in enumerate([] if combined else facts):
                content = f"문서:\n{doc}\n\n{INSTRUCTION}{q}"
                t_req = time.monotonic()
                timing = {"ctx": ctx, "question": qi}
                text, ttft, ptok, ctok, finish = ask_stream(bd.URL, cq.MODEL, content, args.max_tokens, timing)
                rec["requests"].append(timing)
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
                                   "warm_tok_s": tok / warm if warm > 0 else 0.0,
                                   "ttft_samples_s": ttfts,
                                   "combined": combined})
            warm_col = f"{tok / warm:>11.0f}" if not combined else f"{'(1 req)':>11}"
            warm_t = f"{warm:>9.2f}s" if not combined else f"{'-':>10}"
            print(f"{ctx:>7} {tok:>7} {tok / cold:>11.0f} {warm_col} {cold:>9.2f}s {warm_t}  {' '.join(hits)}", flush=True)
        if args.fixed_decode_tokens:
            doc = cq.build(2000, args.seed + 2000)
            qs = "\n".join(f"{i + 1}. {q}" for i, (_, q, _) in enumerate(facts))
            content = (f"문서:\n{doc}\n\n{INSTRUCTION_COMBINED}{qs}\n"
                       "세 답의 근거를 먼저 제시한 뒤 문서의 주제, 주요 개념, 사례와 한계를 "
                       "한국어로 길고 상세하게 설명해줘. 내용을 여러 절로 구성해줘.")
            for rep in range(args.fixed_decode_reps):
                timing = {"ctx": 2000, "question": "fixed-all", "rep": rep, "fixed_decode": True}
                t_req = time.monotonic()
                text, ttft, ptok, ctok, finish = ask_stream(
                    bd.URL, cq.MODEL, content, args.fixed_decode_tokens, timing,
                    min_tokens=args.fixed_decode_tokens, seed=args.seed + rep)
                phase = (2000, t_req + ttft, time.monotonic())
                phases.append(phase)
                fixed_phases.append(phase)
                rec["requests"].append(timing)
                gen_tokens += ctok
                for qi in range(len(facts)):
                    good = all(any(alt in text.lower() for alt in group) for group in FACT_EXPECT[qi])
                    quality_total += 1
                    quality_ok += good
                texts.append((f"fixed2K rep{rep}", text, finish))
                print(f"fixed2K rep={rep} tokens={ctok}/{args.fixed_decode_tokens} "
                      f"decode={timing['decode_tok_s']:.2f} tok/s", flush=True)
    wall = time.time() - t_dec0
    metrics_after = _metrics_text(bd.METRICS)
    m1 = bd._parse_spec_metrics(metrics_after)
    traffic_issues = exclusive_errors(before_traffic, traffic_state(metrics_after),
                                      sw.traffic_samples, len(rec["requests"]))
    rec["traffic"] = {"before": before_traffic, "after": traffic_state(metrics_after),
                      "samples": sw.traffic_samples, "issues": traffic_issues}
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
    # 39차: 1 s windows with 0.5 s margins (were 2 s / 1 s) so the 2K answers
    # (~3-4 s of decode) yield windows every boot; medians of step/s are
    # comparable across window sizes.
    by_ctx, fixed_intervals = decode_windows(samp, phases, fixed_phases, margin=0.5)
    rates = [r for v in by_ctx.values() for r in v]
    if args.fixed_decode_tokens:
        rates = [w["steps"] / w["seconds"] for w in fixed_intervals]
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
    if args.fixed_decode_tokens:
        rec["decode"].update(primary="fixed-2K", fixed_intervals=fixed_intervals,
            fixed_pooled_step_s=(sum(w["steps"] for w in fixed_intervals)
                                 / sum(w["seconds"] for w in fixed_intervals)) if fixed_intervals else None)

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
    issues = list(traffic_issues) if args.require_exclusive else []
    if args.fixed_decode_tokens:
        if any(sb < sa for (_, sa), (_, sb) in zip(samp, samp[1:])):
            issues.append("engine step counter reset during workload")
        if any(q["completion_tokens"] != args.fixed_decode_tokens for q in rec["requests"] if q.get("fixed_decode")):
            issues.append("fixed decode token count differs from requested length")
        if len(fixed_intervals) < 20:
            issues.append(f"too few fixed decode windows: {len(fixed_intervals)} < 20")
    if issues:
        rec["evidence_issues"] = issues
        rec["decode"]["raw_windows_med"] = rec["decode"]["windows_med"]
        rec["decode"]["windows_med"] = None
        print("INVALID measurement: " + "; ".join(issues), flush=True)
    # armed != serving: which of the arm's lanes actually ran, from the head log,
    # checked after the traffic above (serving markers appear only then).
    try:
        from proof import check as _proof_check
        _kn = [kk for kk, vv in (rec.get("knobs") or {}).items() if vv not in ("0", "", "off")]
        if _kn:
            rec.update({kk: vv for kk, vv in _proof_check(
                _kn, os.environ.get("MK_HEAD_LOG", "/home/choiceoh/glm53-logs/glm53.log")).items()
                if kk in ("proof", "proof_ok")})
    except Exception:
        pass

    print(f"== onepass {args.name}: {time.time() - t_all:.0f}s total", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 2 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
