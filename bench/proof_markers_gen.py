#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Keep bench/proof-markers.tsv complete: find every lane's serving line in the
overlay sources and map it to the knob that arms it.

    python3 bench/proof_markers_gen.py            # report: missing rows, stale rows, proposals
    python3 bench/proof_markers_gen.py --check    # exit 1 when a row is missing or stale
    python3 bench/proof_markers_gen.py --propose  # the missing rows as tsv, ready to paste

Rule (39차 idea 3, operator "3"): a knob whose module prints a serving line
("... serving ...", "[tag] installed", "plan built") MUST have a row, so the
proof (bench/proof.py) follows every new lane without anyone remembering.
A row whose marker is "-" declares "this knob has no serving line of its own"
and silences the proposal for it.

Mapping a line to its knob: the line's tag ([drafter-prep], "smlp lane",
"head lane") is split into tokens; a knob read by the SAME file whose name
contains every token is the owner (DRAFTER_PREP, MK_SMLP2, MK_HEAD_*). A file
that reads exactly one knob maps every serving line to it. Lines that report a
failure or a fallback ("failed", "not mounted", "-> stock", "skipped",
"refused", "NOT selected") are never markers. The marker is the literal prefix
up to the first %-format or {-brace, so it matches the served log as a fixed
string.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TSV = os.path.join(HERE, "proof-markers.tsv")
PROFILE = os.path.join(REPO, "profiles", "glm53.env")

KNOB_RE = re.compile(r"VLLM_GLM53_[A-Z0-9_]+")
LIT_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
SERVING_RE = re.compile(r"(?:\bserving\b|\] installed\b|plan built)", re.I)
NEGATIVE_RE = re.compile(r"(fail|not mounted|-> stock|skipp|refus|NOT selected|NOT SERVING|not serving|declin|disarm|missing|cannot|unavailable|off\b)", re.I)


def profile_knobs() -> set[str]:
    with open(PROFILE, encoding="utf-8") as fh:
        return set(KNOB_RE.findall(fh.read()))


def table(path: str = TSV) -> dict[str, tuple[str, str]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = (p[1], p[2] if len(p) > 2 else p[1])
    return out


def marker_of(lit: str) -> str:
    """The fixed-string prefix of a format literal."""
    lit = lit.replace('\\"', '"')
    cut = len(lit)
    for tok in ("%", "{"):
        i = lit.find(tok)
        if i >= 0:
            cut = min(cut, i)
    return lit[:cut].rstrip(" :,(-")


def tag_tokens(lit: str) -> list[str]:
    """'<name> lane serving' names the lane even under a module tag
    ('[megakernel] head lane serving' is the head lane's line, not the
    megakernel knob's); else the [tag]."""
    m = re.search(r"([a-z0-9]+(?:[ -][a-z0-9]+)?) (?:lane|hook) serving", lit, re.I)
    if m:
        return [t for t in re.split(r"[-_ ]+", m.group(1).upper()) if t]
    m = re.match(r"\s*\[([a-z0-9 _-]+)\]", lit, re.I)
    if m:
        return [t for t in re.split(r"[-_ ]+", m.group(1).upper()) if t]
    return []


def owners(tokens: list[str], knobs: set[str]) -> list[str]:
    """Knobs read by the file whose name carries every tag token."""
    if not tokens:
        return []
    out = []
    for k in knobs:
        name = k[len("VLLM_GLM53_"):]
        parts = set(name.split("_"))
        if all(any(t == p or (len(t) >= 4 and t in p) for p in parts) for t in tokens):
            out.append(k)
    return sorted(out, key=lambda k: (len(k), k))     # the exact tag first (PREP_FUSED before PREP_FUSED_SELFCHECK_EVERY)


def scan(root: str = REPO) -> list[tuple[str, str, str]]:
    """(knob, marker, file) for every serving line the rule can attribute."""
    found = []
    prof = profile_knobs()
    for path in sorted(glob.glob(os.path.join(root, "overlay", "modules", "*", "*.py"))):
        src = open(path, encoding="utf-8").read()
        knobs = set(KNOB_RE.findall(src)) & prof
        if not knobs:
            continue
        for lit in LIT_RE.findall(src):
            if not SERVING_RE.search(lit) or NEGATIVE_RE.search(lit):
                continue
            mk = marker_of(lit)
            if len(mk) < 8:
                continue
            own = owners(tag_tokens(lit), knobs)
            if not own and len(knobs) == 1:
                own = sorted(knobs)
            for k in own:
                found.append((k, mk, os.path.relpath(path, root)))
    return found


def check(root: str = REPO):
    """(missing rows, stale rows, proposals) against the table."""
    tbl = table()
    src_all = ""
    for path in sorted(glob.glob(os.path.join(root, "overlay", "modules", "*", "*.py"))):
        src_all += open(path, encoding="utf-8").read()
    stale = [(k, m, s) for k, (m, s) in tbl.items() if m != "-" and s not in src_all]
    seen, covered = {}, set()
    for k, mk, f in scan(root):
        seen.setdefault(k, (mk, f))
    # a line owned by several knobs (a tag that is a prefix of a sibling's name)
    # is satisfied by the shortest one: PREP_FUSED's row covers the line that
    # PREP_FUSED_SELFCHECK_EVERY also matched
    by_line = {}
    for k, (mk, f) in seen.items():
        by_line.setdefault((mk, f), []).append(k)
    for ks in by_line.values():
        ks.sort(key=lambda k: (len(k), k))
        if ks[0] in tbl:
            covered.update(ks[1:])
    missing = [(k, mk, f) for k, (mk, f) in sorted(seen.items()) if k not in tbl and k not in covered]
    return missing, stale, seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--propose", action="store_true")
    a = ap.parse_args()
    missing, stale, seen = check()
    if a.propose:
        for k, mk, f in missing:
            print(f"{k}\t{mk}\t{mk}")
        return 0
    print(f"serving lines attributed to knobs: {len(seen)}; table rows: {len(table())}")
    for k, mk, f in missing:
        print(f"  MISSING {k}: '{mk}'  ({f})")
    for k, m, s in stale:
        print(f"  STALE   {k}: src literal not in the sources any more: '{s}'")
    if not missing and not stale:
        print("  table complete")
    return 1 if (a.check and (missing or stale)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
