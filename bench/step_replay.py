#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""저장된 증거에서 스텝과 레이턴시를 다시 계산한다 — 4박스가 없어도.

플릿을 못 잡은 날에도 디스크에는 측정이 쌓여 있다: 엔진이 죽으며 남긴 스텝 링
(`steps-*.ring`), onepass 의 jsonl 기록, step_peek 이 저장한 /metrics 스크랩, bracket
다리들. 이 도구는 그것들을 **다시 판다**: 스텝 in-flight 레이턴시 분포, 컨텍스트별
TTFT/처리량, decode 창 step/s, 관측 캐던스 — 그리고 onepass 기록이 두 개 이상이면
**재부팅 없이** 기록 간 델타를 낸다.

    python3 bench/step_replay.py measurements/st_onepass_20260912_0746/result.jsonl
    python3 bench/step_replay.py /tmp/stepsim/steps-sim-*.ring
    python3 bench/step_replay.py peek.jsonl
    python3 bench/step_replay.py a.jsonl b.jsonl            # 두 onepass 기록의 델타

해석 규칙 하나: 링 레코드의 wall 은 launch→readback **in-flight** 시간이다. 동기
스텝에서는 스텝 한 번의 전부이지만, async 스텝은 depth 큐에서 기다린 만큼 길다
(숙주 시뮬레이션에서 device 50 ms/depth 2 가 in-flight med 105 ms 로 나오는 것이
그것이다). 판정 채널은 여전히 플릿 onepass(D17) — 이 도구는 재분석이다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step_peek as peek                          # noqa: E402 -- bench 안에서 flat import (테스트 관례)
from engine.base import scheduler as sched        # noqa: E402
from engine.base.record import HEADER, MAGIC      # noqa: E402
from engine.base.runner import KIND, STEP_RECORD  # noqa: E402

RING_MAGIC = MAGIC


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def ring_stats(path: Path) -> dict:
    """steps-*.ring → 종류별 스텝 in-flight 통계."""
    from engine.base.record import read as read_ring
    import struct
    raw = path.read_bytes()
    magic, _ver, _rb, cap = HEADER.unpack_from(raw, 0)
    if magic != RING_MAGIC:
        raise ValueError(f"{path}: 스텝 링이 아니다")
    (pushed,) = struct.unpack_from("<Q", raw, HEADER.size)
    records = read_ring(path)
    by_kind = {sched.PREFILL: [], sched.DECODE: []}
    tokens = {sched.PREFILL: [], sched.DECODE: []}
    rev = {v: k for k, v in KIND.items()}
    for rec in records:
        _count, wall, kind, _n, ntok, _seq = STEP_RECORD.unpack(rec)
        k = rev.get(kind)
        if k not in by_kind:
            continue
        by_kind[k].append(wall * 1e3)
        tokens[k].append(ntok)
    out = {"path": str(path), "pushed": pushed, "kept": len(records),
           "by_kind": {}}
    for k, walls in by_kind.items():
        if not walls:
            continue
        out["by_kind"][k] = {"steps": len(walls), "med_ms": round(statistics.median(walls), 3),
                             "p95_ms": round(_pct(walls, 0.95), 3), "max_ms": round(max(walls), 3),
                             "tokens_med": int(statistics.median(tokens[k]))}
    return out


def onepass_summary(rec: dict) -> dict:
    """onepass jsonl 한 줄 → 요약. 필드는 bench/onepass.py 의 기록 스키마를 따른다."""
    dec = rec.get("decode") or {}
    out = {"name": rec.get("name"), "ts": rec.get("ts"),
           "git": rec.get("git"), "session": rec.get("session"),
           "engine_shape": rec.get("engine_shape"),
           "prefill": [{"ctx": p["ctx"], "cold_s": p.get("cold_s"), "warm_s": p.get("warm_s"),
                        "cold_tok_s": p.get("cold_tok_s"), "warm_tok_s": p.get("warm_tok_s"),
                        "combined": p.get("combined")}
                       for p in rec.get("prefill", [])],
           "step_s": dec.get("windows_med"),
           "step_s_pooled_fixed": dec.get("fixed_pooled_step_s"),
           "windows": len(dec.get("windows") or []),
           "acc_raw": dec.get("acc_raw"), "tokens_per_step": dec.get("tokens_per_step"),
           "quality": rec.get("quality"),
           "korean_dirty": (rec.get("korean") or {}).get("dirty"),
           "korean_n": (rec.get("korean") or {}).get("n"),
           "issues": rec.get("evidence_issues")}
    # 요청 단위 TPOT/TTFT 분포 — 기록에는 전체 timing dict 이 남는다
    tpot = [t["tpot_ms"] for t in rec.get("requests", []) if t.get("tpot_ms")]
    ttft = [t["ttft_s"] for t in rec.get("requests", []) if t.get("ttft_s") is not None]
    out["tpot_ms_med"] = round(statistics.median(tpot), 2) if tpot else None
    out["ttft_s_med"] = round(statistics.median(ttft), 3) if ttft else None
    out["tpot_p95"] = round(_pct(tpot, 0.95), 2) if tpot else None
    return out


def _pct_change(a, b):
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a


def onepass_delta(base: dict, cand: dict) -> list:
    """두 onepass 요약의 비교 행: 무엇이 얼마나 움직였는지."""
    rows = []
    ch = _pct_change(base.get("step_s_pooled_fixed") or base.get("step_s"),
                     cand.get("step_s_pooled_fixed") or cand.get("step_s"))
    rows.append(("decode step/s", base.get("step_s_pooled_fixed") or base.get("step_s"),
                 cand.get("step_s_pooled_fixed") or cand.get("step_s"), ch))
    bp = {p["ctx"]: p for p in base.get("prefill", [])}
    for p in cand.get("prefill", []):
        b = bp.get(p["ctx"])
        if not b:
            continue
        for field, label in (("warm_s", f"ctx{p['ctx'] // 1000}K warm TTFT"),
                             ("warm_tok_s", f"ctx{p['ctx'] // 1000}K warm tok/s")):
            if b.get(field) and p.get(field):
                rows.append((label, b[field], p[field], _pct_change(b[field], p[field])))
    if base.get("acc_raw") is not None and cand.get("acc_raw") is not None:
        rows.append(("raw 수락률", base["acc_raw"], cand["acc_raw"],
                     cand["acc_raw"] - base["acc_raw"]))
    return rows


def bracket_summary(rec: dict) -> dict:
    """bracket.py 다리 한 줄 → 창 step/s 평탄화 중앙값 (judge 의 판정 단위와 같은 배열)."""
    wins = [w for rep in rec.get("reps", []) for w in (rep.get("win_step_s") or [])]
    toks = [rep.get("tok_s") for rep in rec.get("reps", []) if rep.get("tok_s")]
    return {"name": rec.get("name"), "tag": rec.get("tag"), "conc": rec.get("conc"),
            "git": rec.get("git"),
            "win_step_s_med": round(statistics.median(wins), 2) if wins else None,
            "windows": len(wins),
            "tok_s_med": round(statistics.median(toks), 2) if toks else None}


def peek_summary(samples: dict) -> dict:
    return peek.summarize(samples)


def _load(path: Path):
    """파일 하나 → (종류, 내용). 링은 magic 으로, jsonl 은 줄의 스키마로 판별한다."""
    if path.read_bytes()[:4] == RING_MAGIC:
        return ("ring", ring_stats(path))
    onepass, brackets, pings = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if "series" in rec and "monotonic" in rec:
                pings.append((rec["monotonic"], rec["series"]))
            elif "reps" in rec:
                brackets.append(bracket_summary(rec))
            elif "decode" in rec or "prefill" in rec:
                onepass.append(onepass_summary(rec))
    if onepass:
        return ("onepass", onepass)
    if brackets:
        return ("bracket", brackets)
    if pings:
        return ("peek", peek_summary(pings))
    raise ValueError(f"{path}: 알 수 없는 형식 (ring 도 onepass/bracket/peek jsonl 도 아니다)")


def _fmt_onepass(s: dict) -> str:
    lines = [f"== onepass {s.get('name')} ({s.get('ts')}, git {s.get('git')}, "
             f"session {s.get('session')})"]
    shape = s.get("engine_shape")
    if shape:
        lines.append(f"   engine_shape {json.dumps(shape, ensure_ascii=False)[:120]}")
    if s["prefill"]:
        lines.append(f"   {'ctx':>7} {'cold TTFT':>10} {'warm TTFT':>10} {'cold tok/s':>10} {'warm tok/s':>10}")
        for p in s["prefill"]:
            def f(x, fmt="{:.2f}s"):
                return fmt.format(x) if isinstance(x, (int, float)) else "-"
            lines.append(f"   {p['ctx']:>7} {f(p.get('cold_s')):>10} {f(p.get('warm_s')):>10} "
                         f"{f(p.get('cold_tok_s'), '{:.0f}'):>10} "
                         f"{f(p.get('warm_tok_s'), '{:.0f}'):>10}"
                         + ("  (1요청 결합)" if p.get("combined") else ""))
    step = s.get("step_s_pooled_fixed") or s.get("step_s")
    lines.append(f"   decode step/s {step if step is not None else '-'} (창 {s['windows']}개"
                 + (f", fixed pooled {s['step_s_pooled_fixed']}" if s.get("step_s_pooled_fixed") else "")
                 + ")")
    acc = s.get("acc_raw")
    lines.append("   수락률 " + (f"{acc:.1%}" if acc is not None else "-")
                 + f", tokens/step {s.get('tokens_per_step')}, "
                 f"TTFT med {s.get('ttft_s_med')}s, TPOT med {s.get('tpot_ms_med')} ms "
                 f"p95 {s.get('tpot_p95')}")
    q = s.get("quality")
    if q:
        lines.append(f"   품질 {q.get('ok')}/{q.get('total')}, 깨진 응답 "
                     f"{s.get('korean_dirty')}/{s.get('korean_n')}")
    if s.get("issues"):
        lines.append(f"   !! 증거 이슈: {'; '.join(s['issues'])}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="저장된 증거에서 스텝/레이턴시 재계산")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--base", help="onepass 델타의 기준 이름 (기본: 첫 기록)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    loaded = []
    for p in args.paths:
        try:
            loaded.append(_load(p))
        except (ValueError, OSError) as exc:
            print(f"!! {exc}")
    if not loaded:
        return 2

    results = []
    onepass_recs = []
    for kind, payload in loaded:
        if kind == "ring":
            stats = payload
            results.append(("ring", stats))
            print(f"== {Path(stats['path']).name}: 스텝 {stats['pushed']} 푸시, {stats['kept']} 보존")
            for k, v in stats["by_kind"].items():
                print(f"   {k:<7} n={v['steps']:<5} med {v['med_ms']:>9.3f} ms  "
                      f"p95 {v['p95_ms']:>9.3f}  max {v['max_ms']:>9.3f}  tokens med {v['tokens_med']}")
            print("   (wall 은 launch→readback in-flight — async 스텝은 depth 대기가 포함된다)")
        elif kind == "onepass":
            for s in payload:
                results.append(("onepass", s))
                onepass_recs.append(s)
                print(_fmt_onepass(s))
        elif kind == "bracket":
            for s in payload:
                results.append(("bracket", s))
                print(f"== bracket {s.get('name')} tag={s.get('tag')} C={s.get('conc')} "
                      f"git={s.get('git')}: 창 step/s med {s['win_step_s_med']} (n={s['windows']}), "
                      f"tok/s med {s.get('tok_s_med')}")
        elif kind == "peek":
            results.append(("peek", payload))
            print(f"== peek 관측: {payload['windows']}창, 캐던스 중앙값 "
                  f"{payload['step_s_med']} step/s {payload['step_s_q']}")
            if "pooled" in payload:
                print(f"   전체: {peek._line(payload['pooled'])}")

    if len(onepass_recs) >= 2:
        base = next((s for s in onepass_recs if args.base and s.get("name") == args.base),
                    onepass_recs[0])
        for cand in onepass_recs:
            if cand is base:
                continue
            print(f"-- {cand.get('name')} 대비 {base.get('name')} 의 이동:")
            for label, v0, v1, ch in onepass_delta(base, cand):
                def f(v, pct=False):
                    if v is None:
                        return "-"
                    return f"{v:+.1%}" if pct else (f"{v:.4g}" if isinstance(v, float) else str(v))
                move = f" ({ch:+.1%})" if isinstance(ch, float) else ""
                print(f"   {label:<24} {f(v0)} -> {f(v1)}{move}")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
