#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""base→cand→base 브래킷의 기록과 판정. MEASUREMENTS 원장의 판정 규율을 코드로
옮긴 것: 판정 채널은 C=1 step/s (`step/s = tok/s / (1 + k x raw_acc)`, k =
스페큘레이티브 토큰 수), 유의 판정값은 **base 두 다리의 드리프트** — cand 효과가
같은 설정을 다시 부팅했을 때 벌어지는 차이보다 커야 한다.

재기동은 사람이 한다 (런북 규칙: 자동화는 읽기 전용). 이 도구의 `leg` 은 살아있는
서버에 rep 를 돌려 jsonl 에 기록하고, `judge` 는 기록된 다리들로 판정문을 만든다.
env 스냅샷은 **요청값**이다 — 부팅 로그의 engine-confirmed 값과 대조할 것
(#116: 노브가 안 통한 부팅이 무효 A/B 로 판정된 적이 있다).

    python3 bench/bracket.py leg   --name EXP9 --tag base --reps 6
    python3 bench/bracket.py leg   --name EXP9 --tag cand --reps 6
    python3 bench/bracket.py leg   --name EXP9 --tag base --reps 6
    python3 bench/bracket.py judge --name EXP9

C=2/4 로 `leg` 을 돌면 기록만 하고 판정에서는 빠진다 — 원장: C>1 은 수락률이
요청마다 달라 단일 정규화가 안 먹는다 (CV 12~13.5%).
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
from statistics import median

# 노브 스냅샷 목록 — 이 레포가 지금까지 판정에 쓴 환경 키. 새 노브를 여는 PR 은
# 여기에 한 줄씩 추가한다 (빠뜨리면 judge 가 다리 사이 설정 차이를 못 본다).
ENV_KEYS = (
    [k for k in (
        "ENABLE_EP", "CUSTOM_OPS_AXIS", "BENCH_MODEL", "GRAPH_CAP",
        "VLLM_GLM53_FP8_DENSE", "VLLM_GLM53_FP8_DENSE_BPROJ", "VLLM_DFLASH2_FP8_DENSE",
        "VLLM_GLM53_INDEXER_GATE_SPLITK", "VLLM_GLM53_FUSED_K_GATE",
        "VLLM_GLM53_PREP_FUSED", "VLLM_GLM53_ASYNC_DFLASH",
        "VLLM_GLM53_MHC_SMALLM", "VLLM_DFLASH_PREP_WARMUP",
        "VLLM_GLM53_MK_PDL", "VLLM_GLM53_MK_KSR_OUT",
        "VLLM_GLM53_MK_LOCALQ",
        "VLLM_GLM53_AR_PREFETCH", "VLLM_GLM53_DFLASH_EARLY_FC",
        "VLLM_GLM53_DRAFTER_PREP", "VLLM_GLM53_INDEXER_DECODE_FUSED",
    ) if k in os.environ]
    + sorted(k for k in os.environ
             if k.startswith(("VLLM_GLM53_MK_", "VLLM_GLM53_KPOOL")))
)


def _bench_dec():
    spec = importlib.util.spec_from_file_location(
        "bench_dec", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "bench-dec.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.abspath(__file__))
                              ).stdout.strip() or None
    except Exception:
        return None


def _spec_delta(m0, m1):
    """(legacy acc, exact raw acc) over a metrics counter delta — bench-dec 의
    두 관례를 그대로: legacy 는 원장의 모든 수치가 쓰는 substring 합, raw 는
    정확한 이름 비율."""
    delta = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in m1}
    acc = sum(v for k, v in delta.items() if "accept" in k)
    drafts = sum(v for k, v in delta.items() if "draft" in k and "accept" not in k)
    acc_x = sum(v for k, v in delta.items()
                if k.endswith("num_accepted_tokens_total"))
    draft_x = sum(v for k, v in delta.items()
                  if k.endswith("num_draft_tokens_total"))
    legacy = acc / drafts if drafts > 0 else None
    raw = acc_x / draft_x if draft_x > 0 else None
    return legacy, raw


def step_s_of(tok_s: float, acc_raw, num_spec: int):
    """원장 판정 채널. 토큰/스텝 = 1 + k x raw_acc; 수락률 카운터가 없으면
    None — judge 는 그 다리를 tok/s 로 떨어뜨려 놓고 표시한다."""
    if acc_raw is None:
        return None
    return tok_s / (1.0 + num_spec * acc_raw)


class _StepWindows:
    """Sample the engine's step counter (vllm:iteration_tokens_total_count)
    every `period` s while a rep runs, in a thread. Each full window gives a
    step/s sample of its own, so a 40 s rep yields ~10 of them and three
    reps give the spread six reps used to (32차 item 2: fewer reps, more
    samples). A window that spans an idle gap (between requests) reads low
    and is dropped by the `min_frac` rule: fewer steps than min_frac of the
    rep's median window means the engine was not stepping the whole time."""

    def __init__(self, bd, period: float = 2.0):
        import threading
        self.bd, self.period = bd, period
        self.samples: list[tuple[float, float]] = []   # (t, steps)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _steps(self):
        import re, urllib.request
        try:
            text = urllib.request.urlopen(self.bd.METRICS, timeout=5).read().decode()
        except Exception:
            return None
        m = re.search(r"^vllm:iteration_tokens_total_count\{[^}]*\}\s+([0-9.e+]+)", text, re.M)
        return float(m.group(1)) if m else None

    def _run(self):
        import time
        while not self._stop.is_set():
            st = self._steps()
            if st is not None:
                self.samples.append((time.monotonic(), st))
            self._stop.wait(self.period)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._th.join(timeout=self.period + 5)

    def rates(self, min_frac: float = 0.5) -> list[float]:
        r = [(self.samples[i + 1][1] - self.samples[i][1])
             / max(self.samples[i + 1][0] - self.samples[i][0], 1e-6)
             for i in range(len(self.samples) - 1)]
        r = [x for x in r if x > 0]
        if not r:
            return []
        med = median(r)
        return [x for x in r if x >= min_frac * med]


def cmd_leg(args) -> int:
    bd = _bench_dec()
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "name": args.name, "tag": args.tag, "conc": args.conc,
           "num_spec": args.num_spec, "git": _git_sha(),
           "env": {k: os.environ[k] for k in ENV_KEYS},
           "model": bd.MODEL, "reps": []}
    print(f"[bracket] {args.name} tag={args.tag} C={args.conc} reps={args.reps} "
          f"model={bd.MODEL} git={rec['git']} env={rec['env'] or '{}'}")
    for r in range(args.reps):
        m0 = bd.spec_counters()
        with _StepWindows(bd) as win:
            toks, dt = bd.sweep(args.conc, r + 1 + args.conc * 1000)
        legacy, raw = _spec_delta(m0, bd.spec_counters())
        tok_s = toks / dt
        s = step_s_of(tok_s, raw, args.num_spec)
        rates = win.rates()
        rec["reps"].append({"rep": r + 1, "tok_s": round(tok_s, 2),
                            "acc_legacy": None if legacy is None else round(legacy, 4),
                            "acc_raw": None if raw is None else round(raw, 4),
                            "step_s": None if s is None else round(s, 2),
                            "win_step_s": [round(x, 2) for x in rates]})
        wtxt = ""
        if rates:
            q = sorted(rates)
            wtxt = (f", windows n={len(q)} med {median(q):.1f} "
                    f"[{q[len(q) // 4]:.1f}, {q[(3 * len(q)) // 4]:.1f}]")
        print(f"{args.tag} C={args.conc} rep{r + 1}: {tok_s:.1f} tok/s"
              + (f", raw acc {raw:.1%}" if raw is not None else "")
              + (f", step/s {s:.1f}" if s is not None else "") + wtxt, flush=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[bracket] appended to {args.out}")
    return 0


def judge(records: list, conc_judge: int = 1) -> dict:
    """기록 목록 → 판정 딕셔너리. 순수 함수 — 파일도 네트워크도 없다. 기록 한
    줄은 한 다리(rep 목록)이고, 판정 단위는 rep 다."""
    flat = []
    other = 0
    for r in records:
        for rep in r.get("reps", []):
            if r.get("conc") == conc_judge:
                wins = rep.get("win_step_s") or []
                if wins:
                    # step windows: the engine's own step counter sampled
                    # through the rep -- one sample per window, many per rep
                    for w in wins:
                        flat.append({"tag": r["tag"], "env": r.get("env", {}),
                                     "tok_s": rep["tok_s"], "step_s": w})
                else:
                    flat.append({"tag": r["tag"], "env": r.get("env", {}),
                                 "tok_s": rep["tok_s"],
                                 "step_s": rep.get("step_s")})
            else:
                other += 1
    out: dict = {"ok": True, "conc": conc_judge, "problems": [],
                 "segments": [], "cands": [], "other_conc": other}
    if not flat:
        out["ok"] = False
        out["problems"].append(f"C={conc_judge} 기록이 없다")
        return out

    def chan(p):
        return p["step_s"] if p.get("step_s") is not None else p["tok_s"]

    segments = []
    for p in flat:
        key = (p["tag"], json.dumps(p["env"], sort_keys=True))
        if segments and segments[-1]["key"] == key:
            segments[-1]["reps"].append(chan(p))
        else:
            segments.append({"tag": p["tag"], "key": key, "env": p["env"],
                             "reps": [chan(p)]})
    if len(segments) < 3 or segments[0]["tag"] != "base" \
            or segments[-1]["tag"] != "base":
        out["ok"] = False
        out["problems"].append(
            "브래킷이 아니다: base→cand→base 순의 다리가 필요 (첫/마지막 base)")
        out["segments"] = [{"tag": s["tag"], "n": len(s["reps"])}
                           for s in segments]
        return out
    envs = {json.dumps(s["env"], sort_keys=True) for s in segments}
    if len(envs) != 1:
        out["problems"].append(
            "다리 사이 env 스냅샷이 다르다 — 노브가 한 개만 움직였는지 부팅 로그와 대조")

    for s in segments:
        s["median"] = median(s["reps"])
        out["segments"].append({"tag": s["tag"], "n": len(s["reps"]),
                                "median": round(s["median"], 2)})

    def base_pair():
        return segments[0]["median"], segments[-1]["median"]

    b1, b2 = base_pair()
    mean_b = (b1 + b2) / 2
    drift = abs(b1 - b2) / mean_b
    out["base_drift"] = round(drift, 4)
    cands = [s for s in segments[1:-1] if s["tag"] != "base"]
    if not cands:
        out["problems"].append("cand 다리가 없다 (base 만 있다)")
    for c in cands:
        effect = (c["median"] - mean_b) / mean_b
        if abs(effect) <= drift:
            verdict = ("CV 이하 — 단독 판정 불가 (rep 를 늘리거나 "
                       "비트동일 노브와 스택 부팅)")
        elif effect > 0:
            verdict = "채택 근거 (effect > base 드리프트)"
        else:
            verdict = "기각 근거 (역효과 > base 드리프트)"
        out["cands"].append({"median": round(c["median"], 2),
                             "effect": round(effect, 4),
                             "verdict": verdict})
    if any(c["verdict"].startswith("CV 이하") for c in out["cands"]) \
            and min(len(s["reps"]) for s in segments) < 12:
        out["problems"].append("drift 이하 효과 + rep < 12: rep 를 늘려 drift 를 더 눌러볼 것")
    return out


def cmd_judge(args) -> int:
    records = []
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if args.name in (None, r.get("name")):
                    records.append(r)
    if not records:
        print(f"!! {args.out} 에 기록이 없다")
        return 2
    rep = judge(records, conc_judge=args.conc)
    print(f"# bracket {args.name or ''} — 판정 채널 C={args.conc} "
          f"(step/s = tok/s / (1 + k x raw_acc))")
    for p in rep["problems"]:
        print(f"  !! {p}")
    for s in rep["segments"]:
        print(f"  {s['tag']:<6} n={s['n']:<3} median {s['median']}")
    if rep.get("base_drift") is not None:
        print(f"  base 드리프트 {rep['base_drift']:.2%} (같은 설정 재부팅 차이 — 유의 문턱)")
    for c in rep["cands"]:
        print(f"  cand    n 대비 median {c['median']}  effect {c['effect']:+.2%}  → {c['verdict']}")
    if rep["other_conc"]:
        print(f"  (참고: C≠1 기록 {rep['other_conc']}개는 판정에서 뺐다 — 원장 규칙)")
    return 0 if rep["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("leg", help="살아있는 서버에 rep 를 돌려 기록")
    lg.add_argument("--name", required=True)
    lg.add_argument("--tag", required=True, choices=("base", "cand"))
    lg.add_argument("--reps", type=int, default=3)
    lg.add_argument("--conc", type=int, default=1)
    lg.add_argument("--num-spec", type=int, default=7,
                    help="스페큘레이티브 토큰 수 k (step/s 정규화 계수)")
    lg.add_argument("--out", default="runs/bracket.jsonl")
    lg.set_defaults(fn=cmd_leg)
    jd = sub.add_parser("judge", help="기록된 다리들로 판정")
    jd.add_argument("--name", default=None)
    jd.add_argument("--conc", type=int, default=1)
    jd.add_argument("--out", default="runs/bracket.jsonl")
    jd.set_defaults(fn=cmd_judge)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
