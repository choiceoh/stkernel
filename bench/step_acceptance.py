#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measured cumulative acceptance at each draft position, from saved /metrics.

    python3 bench/storacle.py acceptance peek.jsonl
    python3 bench/storacle.py acceptance --before before.txt --after after.txt --k 6

Counts follow the engine's accepted-token counter, excluding the bonus token.
Terminal prefixes are censored by its EOS/output-limit convention; these are not
uncensored verifier probabilities.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HIST = "st:spec_accepted_per_step_total"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTED = "vllm:spec_decode_num_draft_tokens_total"
LANE = "st:lane_info"
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*")\s*(,|$)')


def _family(metrics: dict, name: str) -> dict:
    """Canonical label tuples, so label order cannot change counter identity."""
    rows = {}
    for key, value in metrics.items():
        if key != name and not key.startswith(name + "{"):
            continue
        labels, pos = {}, 0
        body = key[len(name)+1:-1] if key != name and key.endswith("}") else ""
        if key != name and not key.endswith("}"):
            raise ValueError(f"malformed {name} labels")
        while pos < len(body):
            m = _LABEL.match(body, pos)
            if not m or m[1] in labels:
                raise ValueError(f"malformed/duplicate {name} labels")
            labels[m[1]] = json.loads(m[2])
            pos = m.end()
        identity = tuple(sorted(labels.items()))
        if identity in rows:
            raise ValueError(f"duplicate {name} series")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError(f"invalid {name} count")
        rows[identity] = int(value)
    return rows


def _groups(metrics: dict) -> dict:
    groups = {}
    for labels, value in _family(metrics, HIST).items():
        fields = dict(labels)
        position = fields.pop("accepted", "")
        if not position.isascii() or not position.isdigit():
            raise ValueError("invalid accepted position")
        group = groups.setdefault(tuple(sorted(fields.items())), {})
        i = int(position)
        if i in group:
            raise ValueError("duplicate accepted position")
        group[i] = value
    return groups


def histogram(a: dict, b: dict, k: int | None = None) -> dict:
    """Validate the complete interval before returning any histogram delta.

    Zero buckets are sparsely emitted by ST. A new bucket within an existing
    group starts at zero; a new group, reset, or changed lane is not an interval.
    Aggregate counters, when available, must agree with every group of bins.
    """
    lanes0, lanes1 = _family(a, LANE), _family(b, LANE)
    if lanes0 != lanes1:
        raise ValueError("lane identity changed between scrapes")
    ks = {dict(labels)["spec_k"] for labels in lanes0 if "spec_k" in dict(labels)}
    if len(ks) > 1:
        raise ValueError("mixed spec_k values")
    if ks:
        recorded_k = int(next(iter(ks)))
        if k is not None and k != recorded_k:
            raise ValueError("--k disagrees with recorded spec_k")
        k = recorded_k
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("spec_k must be positive")
    before, after = _groups(a), _groups(b)
    if not before or not after:
        raise ValueError("acceptance histogram is missing")
    if before.keys() != after.keys():
        raise ValueError("acceptance series groups changed between scrapes")
    totals = []
    for name in (ACCEPTED, DRAFTED):
        left, right = _family(a, name), _family(b, name)
        if left or right:
            if left.keys() != before.keys() or right.keys() != before.keys():
                raise ValueError(f"{name} groups do not match histogram")
            if any(right[g] < left[g] for g in left):
                raise ValueError(f"{name} reset between scrapes")
        totals.append({g: right[g] - left[g] for g in left})
    accepted, drafted = totals
    combined = {}
    for group in before:
        bins = {}
        for i in before[group].keys() | after[group].keys():
            v0, v1 = before[group].get(i, 0), after[group].get(i, 0)
            if v1 < v0:
                raise ValueError("acceptance bucket reset/disappeared between scrapes")
            if k is not None and i > k and (v0 or v1):
                raise ValueError("accepted position exceeds spec_k")
            bins[i] = v1 - v0
            combined[i] = combined.get(i, 0) + bins[i]
        if accepted and sum(i*n for i, n in bins.items()) != accepted[group]:
            raise ValueError("histogram does not cover accepted-token counter delta")
        if drafted and k is not None and sum(bins.values()) * k != drafted[group]:
            raise ValueError("histogram does not cover drafted-token counter delta")
    last = k if k is not None else max(combined)
    return {"k": k, "histogram": [combined.get(i, 0) for i in range(last+1)]}


def histogram_from_samples(samples: list[dict], k: int | None = None) -> dict:
    """Check adjacent intervals too: endpoint subtraction can hide a restart."""
    if len(samples) < 2:
        raise ValueError("at least two scrapes are required")
    for a, b in zip(samples, samples[1:]):
        histogram(a, b, k)
    return histogram(samples[0], samples[-1], k)


def profile(data: dict) -> dict:
    """P(X >= i) and P(X >= i | X >= i-1), with explicit row denominators."""
    k, hist = data["k"], data["histogram"]
    if k is None:
        raise ValueError("spec_k is missing; supply --k for older scrapes")
    if type(k) is not int or k < 1 or not isinstance(hist, list) or len(hist) != k + 1:
        raise ValueError("acceptance histogram must cover positions 0 through K")
    if any(type(count) is not int or count < 0 for count in hist):
        raise ValueError("acceptance histogram needs nonnegative integer counts")
    n = sum(hist)
    if not n:
        raise ValueError("no observed decode rows in this interval")
    accepted = sum(i*count for i, count in enumerate(hist))
    positions, reached = [], n
    for i in range(1, k+1):
        passed = reached - hist[i-1]
        positions.append({"position": i, "accepted_rows": passed, "reached_rows": reached,
                          "cumulative": passed / n,
                          "conditional": passed / reached if reached else None})
        reached = passed
    return {**data, "rows": n, "accepted_tokens": accepted, "raw_acceptance": accepted / (n*k),
            "positions": positions,
            "basis": "engine accepted-token counters; bonus excluded, terminal prefixes censored"}


def load_profile(path: Path, k: int | None = None) -> dict:
    """Consume validated scrape intervals, never a fitted average or summary."""
    samples = []
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("peek records must be objects")
            if 'series' in record:
                if not isinstance(record['series'], dict):
                    raise ValueError("peek series must be a metrics object")
                samples.append(record['series'])
    return profile(histogram_from_samples(samples, k))


def yield_bounds(k: int, *, observed: dict | None = None,
                 reference_k: int = 6, raw_acceptance: float = .45) -> tuple[float, float]:
    """Tokens/row under a transferred prefix distribution and one bonus token.

    For observed prefixes E[min(X,K)] = sum(P(X>=i), i=1..K). Unobserved
    positions have no point estimate: their survival is between 0 and the last
    observed survival. An average alone bounds, but cannot identify, a new K.
    These are scenario bounds, not sampling confidence intervals or GPU proof.
    """
    if type(k) is not int or k < 0:
        raise ValueError('candidate K must be a nonnegative integer')
    if observed is not None:
        data = profile(observed)
        survival = [row['cumulative'] for row in data['positions']]
        known = 1 + math.fsum(survival[:k])
        return known, known + max(0, k - data['k']) * survival[-1]
    if (type(reference_k) is not int or reference_k < 0
            or not math.isfinite(raw_acceptance) or not 0 <= raw_acceptance <= 1):
        raise ValueError('invalid reference K or raw acceptance')
    if reference_k == 0:
        return 1.0, 1.0 + k  # no draft observations, including no tail probability
    accepted = reference_k * raw_acceptance
    if k <= reference_k:
        return 1 + k * raw_acceptance, 1 + min(k, accepted)
    return 1 + accepted, 1 + accepted + (k - reference_k) * raw_acceptance


def economics(data: dict) -> dict:
    """Marginal value of each draft position; no constant/geometric tail fit."""
    data = profile(data)
    rows, before = [], 1.0
    for row in data['positions']:
        marginal = row['cumulative']
        after = before + marginal
        rows.append(dict(k=row['position'], cumulative=marginal, tokens_per_row=after,
                         max_relative_step_increase=marginal / before))
        before = after
    tail = data['positions'][-1]['cumulative']
    return dict(rows=rows,
                next_position=dict(k=data['k'] + 1, cumulative_range=[0.0, tail],
                                   max_relative_step_increase_range=[0.0, tail / before]),
                basis='same prefix distribution and one bonus token per row; terminal censoring retained; not a live K comparison')


def format_economics(data: dict) -> str:
    value = economics(data)
    lines = ['K 증가 손익분기 · 관측된 prefix 분포와 보너스 1토큰을 유지하는 가정',
             '     K    해당 위치 누적    기대 토큰/행    허용 스텝 시간 증가(미만)']
    for row in value['rows']:
        lines.append(f"  {row['k']:>4}    {row['cumulative']:>12.1%}    {row['tokens_per_row']:>12.3f}"
                     f"    {row['max_relative_step_increase']:>12.2%}")
    row = value['next_position']
    lines.append(f"  K={row['k']}: 미관측 · 누적 0–{row['cumulative_range'][1]:.1%},"
                 f" 허용 증가 0–{row['max_relative_step_increase_range'][1]:.2%} 범위만 식별")
    lines.append('  표본 오차의 신뢰구간이 아님; K 변경에 따른 prefix 분포 변화와 실제 스텝 시간은 별도 계측')
    return '\n'.join(lines)


def format_profile(data: dict) -> str:
    lines = [f"수락률 실측 · K={data['k']} · 행-스텝 {data['rows']} · 평균 {data['raw_acceptance']:.1%}",
             "  위치    누적 수락률    직전 위치 통과 후    수락 / 전체 / 직전 통과"]
    for row in data["positions"]:
        conditional = "-" if row["conditional"] is None else f"{row['conditional']:.1%}"
        lines.append(f"  {row['position']:>4}    {row['cumulative']:>9.1%}    {conditional:>16}"
                     f"    {row['accepted_rows']} / {data['rows']} / {row['reached_rows']}")
    lines.append("  엔진 수락 카운터 기준: 보너스 제외, EOS·출력 길이 제한 영향 포함")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scrapes", nargs="?", type=Path, help="step_peek JSONL")
    ap.add_argument("--before", type=Path, help="first Prometheus metrics text")
    ap.add_argument("--after", type=Path, help="last Prometheus metrics text")
    ap.add_argument("--k", type=int, help="draft count if older scrapes lack st:lane_info")
    ap.add_argument("--economics", action="store_true", help="marginal K value and break-even step-time overhead")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if bool(args.scrapes) == bool(args.before or args.after) or bool(args.before) != bool(args.after):
        ap.error("supply a scrape JSONL or both --before and --after")
    try:
        if args.scrapes:
            result = load_profile(args.scrapes, args.k)
        else:
            from step_peek import parse_metrics
            samples = [parse_metrics(p.read_text()) for p in (args.before, args.after)]
            result = profile(histogram_from_samples(samples, args.k))
        if args.economics:
            result['economics'] = economics(result)
    except (OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            print(f"수락률 확인 불가: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False) if args.json else format_profile(result)
          + ('\n' + format_economics(result) if args.economics else ''))
    return 0


if __name__ == "__main__":
    sys.exit(main())
