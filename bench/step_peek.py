#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""4박스를 잡지 않고, 지금 돌고 있는 부팅의 스텝과 레이턴시를 관측한다.

누군가 플릿을 쥐고 있을 때(서빙 중, 다른 세션의 벤치 중) 남는 측정 경로가 하나 있다:
**아무것도 잡지 않고 /metrics 만 읽는 것.** 요청을 보내지 않으니 독점성도, 큐도,
리스도 건드리지 않는다. 그 대신 이 숫자는 관측이다 — 내 워크로드의 판정이 아니라
지금 부팅의 실제 스텝 캐던스와 요청 레이턴시다(D17: 판정은 플릿 onepass 두 번).

엔진의 /metrics 는 원래 이걸 위해 설계됐다: `vllm:iteration_tokens_total_count` 는
스텝 카운터(bracket._StepWindows 와 같은 계열), `st:step_seconds{kind=...}` 는
"호스트가 본 스텝 종단" 히스토그램, `vllm:time_to_first_token_seconds` 등은 도착부터
잰 요청 레이턴시 히스토그램. 둘 다 **누적**이므로 두 스크랩의 차가 그 창의 분포가
된다 — quantile 은 버킷 경계 안에서 보간한다.

    python3 bench/step_peek.py                              # 헤드 30초 관측
    python3 bench/step_peek.py --seconds 120 --out peek.jsonl   # 저장 → step_replay 로 재분석

구버전(vLLM 포크) 부팅에서도 돈다: st:* 계열이 없으면 스텝 캐던스와 수락률만 나온다.
없는 카운터는 0이 아니다(window_metrics 규칙) — 없으면 칸을 비운다.
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
import urllib.request

DEFAULT_URL = "http://10.10.10.2:8000"          # 플릿 헤드 (bench/fleet.sh HEAD_URL 계열)

# 두 스크랩 사이 증분으로 읽는 계열. 히스토그램은 이름만 적는다(_bucket/_sum/_count).
STEP = "vllm:iteration_tokens_total_count"
COUNTERS = (STEP, "st:steps_prefill_total", "st:steps_decode_total",
            "st:decode_row_steps_total", "st:generation_tokens_committed_total",
            "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
            "vllm:request_success_total",
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_drafts_total")
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
HISTOGRAMS = ("st:step_seconds",                      # kind="prefill"/"decode" 라벨
              "vllm:time_to_first_token_seconds",
              "vllm:inter_token_latency_seconds",
              "vllm:time_per_output_token_seconds",
              "vllm:e2e_request_latency_seconds",
              "vllm:request_queue_time_seconds")


def parse_metrics(text: str) -> dict:
    """Prometheus 텍스트 → {'name{labels}': value}. HELP/TYPE 와 빈 줄은 건너뛴다."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def series(metrics: dict, name: str, **labels):
    """이름(과 필요 라벨의 부분집합)으로 하나 찾기. 없으면 None — 없는 계열은 0이 아니다."""
    want = [f'{k}="{v}"' for k, v in labels.items()]
    for key, value in metrics.items():
        if key == name or (key.startswith(name + "{") and all(w in key for w in want)):
            return value
    return None


def _buckets(metrics: dict, name: str, **labels):
    """(le, 누적 count) 오름차순. +Inf 는 뺀다 — 전체는 _count 계열이 말한다."""
    rows = []
    for key, value in metrics.items():
        if not key.startswith(name + "_bucket{"):
            continue
        fields = dict((k.strip(), v.strip().strip('"'))
                      for part in key[key.index("{") + 1:-1].split(",")
                      for k, _, v in [part.partition("=")])
        if any(fields.get(k) != str(v) for k, v in labels.items()):
            continue
        for part in key[key.index("{") + 1:-1].split(","):
            k, _, v = part.partition("=")
            if k.strip() == "le":
                le = v.strip().strip('"')
                if le != "+Inf":
                    rows.append((float(le), value))
    rows.sort()
    return rows


def hist_delta(a: dict, b: dict, name: str, **labels) -> "dict | None":
    """두 스크랩 사이 그 히스토그램의 분포: count/sum 증분과 보간 quantile."""
    c0, c1 = series(a, name + "_count", **labels), series(b, name + "_count", **labels)
    s0, s1 = series(a, name + "_sum", **labels), series(b, name + "_sum", **labels)
    if None in (c0, c1, s0, s1):
        return None
    count = c1 - c0
    if count <= 0:
        return None
    bounds, cum = [], []
    b0, b1 = _buckets(a, name, **labels), _buckets(b, name, **labels)
    # A changed bucket layout cannot be zipped into a distribution even when
    # it has the same number of buckets. Count/mean remain independently valid.
    if b0 and b1 and [le for le, _ in b0] == [le for le, _ in b1]:
        for (le, v0), (_, v1) in zip(b0, b1):
            bounds.append(le)
            cum.append(max(0.0, v1 - v0))
    out = {"count": int(count), "mean": (s1 - s0) / count}
    for tag, q in (("p50", 0.5), ("p95", 0.95)):
        out[tag] = _quantile(bounds, cum, count, q) if bounds else None
    return out


def _quantile(bounds, cum, total, q):
    """누적 버킷 증분 위의 선형 보간 quantile — 버킷 안을 균등이라고만 가정한다."""
    target = q * total
    prev_bound, prev_cum = 0.0, 0.0
    for bound, c in zip(bounds, cum):
        if c >= target:
            frac = (target - prev_cum) / max(c - prev_cum, 1e-9)
            return prev_bound + frac * (bound - prev_bound)
        prev_bound, prev_cum = bound, c
    return bounds[-1] if bounds else None


def window(a: dict, b: dict, dt: float) -> dict:
    """스크랩 a→b, dt 초짜리 한 창의 요약. 순수 함수 — 네트워크도 시간도 없다."""
    def d(name):
        v0, v1 = series(a, name), series(b, name)
        return None if None in (v0, v1) else v1 - v0

    steps = d(STEP)
    out = {"dt": dt,
           "steps": steps,
           "step_s": steps / dt if steps is not None and dt > 0 else None}
    pre = d("st:steps_prefill_total")
    out["prefill_share"] = pre / steps if None not in (pre, steps) and steps else None
    gen = d("vllm:generation_tokens_total")
    out["gen_tok_s"] = gen / dt if gen is not None and dt > 0 else None
    # ST's legacy vLLM generation counter advances at request completion.
    # This optional counter counts actual IDs as the host observes each step;
    # it is usable while all requests are still live, without an acceptance
    # formula. Neither rate is a measurement of client network delivery time.
    committed = d("st:generation_tokens_committed_total")
    out["committed_tok_s"] = committed / dt if committed is not None and committed >= 0 and dt > 0 else None
    row_steps, decode = d("st:decode_row_steps_total"), d("st:steps_decode_total")
    out["mean_decode_rows"] = row_steps / decode if None not in (row_steps, decode) and row_steps >= 0 and decode > 0 else None
    acc, draft = d("vllm:spec_decode_num_accepted_tokens_total"), d("vllm:spec_decode_num_draft_tokens_total")
    out["acc_raw"] = acc / draft if None not in (acc, draft) and draft else None
    req = d("vllm:request_success_total")
    out["requests"] = int(req) if req is not None else None
    out["step_ms"] = hist_delta(a, b, "st:step_seconds", kind="decode")
    out["ttft"] = hist_delta(a, b, "vllm:time_to_first_token_seconds")
    out["e2e"] = hist_delta(a, b, "vllm:e2e_request_latency_seconds")
    out["gauges"] = {name: series(b, name) for name in GAUGES}
    return out


def fetch(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as r:
        return parse_metrics(r.read().decode())


def track(metrics: dict) -> dict:
    """저장/재분석에 쓸 계열만 남긴다."""
    names = set(COUNTERS) | set(GAUGES)
    keep = {}
    for key, value in metrics.items():
        base = key.split("{", 1)[0]
        if base in names or any(base == h + suffix for h in HISTOGRAMS
                                for suffix in ("_bucket", "_sum", "_count")):
            keep[key] = value
    return keep


def _fmt_ms(h):
    if not h:
        return "-"
    p50 = f"{h['p50'] * 1e3:.1f}" if h.get("p50") is not None else "-"
    p95 = f"{h['p95'] * 1e3:.1f}" if h.get("p95") is not None else "-"
    return f"p50 {p50} p95 {p95} (n={h['count']})"


def _fmt_s(h):
    if not h:
        return "-"
    p50 = f"{h['p50']:.2f}s" if h.get("p50") is not None else "-"
    p95 = f"{h['p95']:.2f}s" if h.get("p95") is not None else "-"
    return f"med {p50} p95 {p95} (n={h['count']})"


def _line(w: dict) -> str:
    def num(x, fmt="{:.1f}"):
        return fmt.format(x) if x is not None else "-"

    g = w["gauges"]
    return (f"steps {num(w['step_s'])}/s"
            + (f" (prefill {w['prefill_share']:.0%})" if w["prefill_share"] is not None else "")
            + f"  committed {num(w.get('committed_tok_s'))} tok/s"
            + f"  gen {num(w['gen_tok_s'])} tok/s"
            + f"  decode rows {num(w.get('mean_decode_rows'))}"
            + (f"  acc {w['acc_raw']:.1%}" if w["acc_raw"] is not None else "")
            + f"  step_ms [{_fmt_ms(w['step_ms'])}]"
            + f"  ttft [{_fmt_s(w['ttft'])}]"
            + f"  run {num(g.get('vllm:num_requests_running'), '{:g}')}"
            f" wait {num(g.get('vllm:num_requests_waiting'), '{:g}')}")


def summarize(samples: "list[tuple[float, dict]]") -> dict:
    """관측 전체의 요약: 창별 캐던스의 중앙값/사분위 + 전체 누적 히스토그램 차."""
    rates = []
    for (ta, sa), (tb, sb) in zip(samples, samples[1:]):
        dt = tb - ta
        if dt <= 0:
            continue
        s0, s1 = series(sa, STEP), series(sb, STEP)
        if None not in (s0, s1) and s1 > s0:
            rates.append((s1 - s0) / dt)
    out = {"windows": len(samples) - 1,
           "step_s_med": round(statistics.median(rates), 2) if rates else None,
           "step_s_q": [round(min(rates), 2), round(max(rates), 2)] if rates else None}
    if len(samples) >= 2:
        total_dt = samples[-1][0] - samples[0][0]
        pooled = window(samples[0][1], samples[-1][1], total_dt)
        out["pooled"] = pooled
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="플릿을 잡지 않고 살아있는 서버 /metrics 관측")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"엔진 베이스 URL (기본 {DEFAULT_URL})")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--out", help="스크랩 샘플을 이 jsonl 에 append (step_replay 가 읽는다)")
    args = ap.parse_args()

    print(f"[step_peek] {args.url} 를 {args.seconds:g}초, {args.period:g}s 주기로 관측한다 — "
          "요청을 보내지 않고, 큐/리스를 잡지 않는다.")
    print("[step_peek] 이 숫자는 지금 부팅의 관측이지 판정이 아니다 (D17: 판정은 플릿 onepass 두 번).")
    samples = []
    fh = None
    if args.out:
        fh = open(args.out, "a", encoding="utf-8")
    try:
        first = fetch(args.url)
    except Exception as exc:                              # noqa: BLE001 -- 사용자에게 그대로 보여준다
        print(f"!! {args.url}/metrics 에 닿지 않는다: {exc}")
        return 2
    t0 = time.monotonic()
    samples.append((t0, first))
    if fh:
        fh.write(json.dumps({"monotonic": t0,
                             "wall": datetime.datetime.now().isoformat(timespec="seconds"),
                             "series": track(first)}) + "\n")
        fh.flush()
    try:
        while True:
            time.sleep(args.period)
            now = time.monotonic()
            if now - t0 >= args.seconds:
                break
            try:
                m = fetch(args.url)
            except Exception as exc:                      # noqa: BLE001 -- 한 창이 날아간 것뿐
                print(f"  (스크랩 실패, 건너뜀: {exc})")
                continue
            w = window(samples[-1][1], m, now - samples[-1][0])
            print(f"t+{now - t0:6.1f}s  {_line(w)}", flush=True)
            samples.append((now, m))
            if fh:
                fh.write(json.dumps({"monotonic": now,
                                     "wall": datetime.datetime.now().isoformat(timespec="seconds"),
                                     "series": track(m)}) + "\n")
                fh.flush()
    except KeyboardInterrupt:
        print("\n(중단 — 관측된 창까지로 요약한다)")
    finally:
        if fh:
            fh.close()
    s = summarize(samples)
    print(f"== {s['windows']}창, 캐던스 중앙값 {s['step_s_med']} step/s {s['step_s_q']}")
    if "pooled" in s:
        print(f"   전체: {_line(s['pooled'])}")
    if args.out:
        print(f"   샘플: {args.out} (step_replay 로 재분석)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
