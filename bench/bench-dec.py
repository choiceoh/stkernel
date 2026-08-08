#!/usr/bin/env python3
"""Decode-only C=1/2/4 bench, built to be low-variance.

The noise source in every earlier decode measurement is the dspark acceptance
rate, which is re-rolled per request (probabilistic draft sampling at temp 0.95).
Averaging over a long generation shrinks that: each sample here spans LONG_OUT
decode steps instead of a couple hundred, so per-sample spread drops enough to
resolve a <10% effect. Prompts are short and unique (no prefix-cache hits, and
prefill is a negligible share of wall time).
"""
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/v1/chat/completions"
METRICS = "http://127.0.0.1:8000/metrics"
MODEL = "deepseek-v4-flash"
LONG_OUT = 768
TOPICS = [
    "조력 발전의 원리", "해수 담수화 공정", "지열 발전의 입지 조건",
    "풍력 터빈의 요잉 제어", "양수 발전의 역할", "송전 손실의 원인",
    "변압기 냉각 방식", "계통 주파수 유지", "수소 저장 방식",
    "태양광 인버터의 MPPT",
]


def one(seed: int) -> int:
    """One long generation; returns completion tokens produced."""
    rng = random.Random(seed)
    topic = rng.choice(TOPICS)
    body = json.dumps({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": f"{topic}에 대해 {LONG_OUT} 토큰 분량으로 자세히 설명해줘. "
                       f"(참조번호 {seed})",
        }],
        "max_tokens": LONG_OUT,
        "temperature": 0.95,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out["usage"]["completion_tokens"]


def sweep(conc: int, seed: int) -> tuple[int, float]:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        toks = list(ex.map(one, [seed * 131 + i for i in range(conc)]))
    return sum(toks), time.time() - t0


def _parse_spec_metrics(text: str) -> dict[str, float]:
    """Sum spec-decode counters from a Prometheus payload (labels collapsed).

    Metric names vary across vLLM forks; match anything containing
    ``spec_decode`` so the acceptance ratio below stays fork-agnostic.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(
            r"([A-Za-z0-9_:]*spec_decode[A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+"
            r"([0-9.eE+-]+)\s*$",
            line,
        )
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def spec_counters() -> dict[str, float]:
    try:
        text = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:
        return {}
    return _parse_spec_metrics(text)


def acceptance_suffix(before: dict[str, float], after: dict[str, float]) -> str:
    """Per-sample dspark acceptance from the /metrics counter delta.

    Acceptance noise is THE variance source in decode rates (probabilistic
    draft sampling at temp 0.95), so printing it beside each sample shows
    whether a rate outlier is acceptance luck or a real effect — and lets
    prefill-batch-size influence on acceptance be tracked (forum #378890).
    """
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    acc = sum(v for k, v in delta.items() if "accept" in k)
    drafts = sum(v for k, v in delta.items() if "draft" in k and "accept" not in k)
    if drafts <= 0:
        return ""
    return f" | acc {acc / drafts:6.1%}"


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    for c in (1, 2, 4):
        rates = []
        for r in range(reps):
            m0 = spec_counters()
            toks, dt = sweep(c, r + 1 + c * 1000)
            m1 = spec_counters()
            rates.append(toks / dt)
            print(f"{tag}C={c} rep{r + 1}: {toks:>5} tok / {dt:>5.1f}s "
                  f"=> {toks / dt:>6.1f} tok/s"
                  f"{acceptance_suffix(m0, m1)}", flush=True)
        rates.sort()
        n = len(rates)
        mean = sum(rates) / n
        med = rates[n // 2] if n % 2 else (rates[n // 2 - 1] + rates[n // 2]) / 2
        print(f"{tag}C={c} SUMMARY mean {mean:.1f} | median {med:.1f} | "
              f"min {rates[0]:.1f} | max {rates[-1]:.1f}", flush=True)
