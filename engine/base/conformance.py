"""Bands and repeat counts, or it is not a gate (base).

D14 accepted that quiet quality loss can pass an end-to-end gate, and named
the two things that make a gate hold anyway:

  a band, not a point   DEF40's tok/step 3.597 was the best of six boots; the
                        "-7%" that followed was spread, not regression, and it
                        was only readable once a band (3.35-3.46) existed.
  a repeat count        the 128K decode anomaly showed in 3 of 10 boots. A
                        gate that passes once has not passed.

A Gate declares both. `judge` takes the samples a campaign produced and says
PASS / FAIL / INSUFFICIENT -- and INSUFFICIENT is what a single boot gets.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class Gate:
    name: str
    low: float                 # band, inclusive
    high: float
    repeats: int               # samples required before a verdict exists
    source: str                # where the band came from (a ledger section)


@dataclass(frozen=True)
class Verdict:
    gate: str
    status: str                # PASS | FAIL | INSUFFICIENT
    samples: int
    center: float | None
    detail: str


def judge(gate: Gate, samples: "list[float]") -> Verdict:
    n = len(samples)
    if n < gate.repeats:
        return Verdict(gate.name, "INSUFFICIENT", n, median(samples) if samples else None,
                       f"{n} sample(s), gate needs {gate.repeats}: no verdict from a single boot")
    center = median(samples)
    inside = sum(gate.low <= s <= gate.high for s in samples)
    if inside == n:
        return Verdict(gate.name, "PASS", n, center, f"all {n} inside [{gate.low}, {gate.high}]")
    return Verdict(gate.name, "FAIL", n, center,
                   f"{n - inside} of {n} outside [{gate.low}, {gate.high}]: "
                   + ", ".join(f"{s:g}" for s in samples if not gate.low <= s <= gate.high))


def _selfcheck() -> None:
    g = Gate("decode tok/step (ko)", 3.35, 3.46, repeats=3, source="ledger 39th: DEF40 band")
    assert judge(g, [3.597]).status == "INSUFFICIENT"           # the DEF40 lesson, exactly
    assert judge(g, [3.40, 3.44, 3.36]).status == "PASS"
    v = judge(g, [3.40, 3.44, 3.20])
    assert v.status == "FAIL" and "3.2" in v.detail
    print("  conformance: single boot is INSUFFICIENT, band + repeats judge OK")


if __name__ == "__main__":
    _selfcheck()
