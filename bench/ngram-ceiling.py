#!/usr/bin/env python3
"""Offline ceiling measurement for an ngram/copy (prompt-lookup) draft hybrid.

Hard rule (MEASUREMENTS.md): compute the theoretical upper bound BEFORE any
engine work. This simulates prompt-lookup drafting over real serving-shaped
text and reports how many accepted-tokens/step a dspark+PLD hybrid could add
over dspark alone. Only if the ceiling clears the adoption bar (high single
digits) does an engine implementation (which must fight the FULL-CUDA-graph
draft step) get considered.

Simulation model (documented, deliberately simple):
  * The realized text is treated as THE target sample — valid for this fork
    because the seeded-Gumbel coupling emits argmax(log p + g) regardless of
    the draft, so a deterministic copy proposal is accepted exactly where it
    equals the realized continuation.
  * Engine-style stepping: each step advances 1 + accepted tokens.
  * dspark acceptance is drawn from the measured per-position curve
    (default: ledger 78.5/56.4/39.6/28.6/21.5) with a seeded RNG.
  * hybrid policy: use the copy proposal when the context suffix match is at
    least --threshold tokens, else dspark. An oracle policy (pick whichever
    accepts more, per step) bounds the best possible router.

Inputs: .txt files (first --warmup tokens become context) or .jsonl rows
({"prompt": ..., "completion": ...} — prompt is context, completion is
simulated; or {"text": ...}). Tokenizer: --tokenizer <model dir> (run on a
fleet node / in a throwaway image container); --mode chars is a smoke-test
fallback only — do not quote its numbers.

Example (inside the image, CPU-only, never docker exec into hy4):
  docker run --rm -v /home/choiceoh/models:/home/choiceoh/models \
    -v ~/stkernel/bench:/bench --entrypoint python3 \
    aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6 \
    /bench/ngram-ceiling.py --tokenizer /home/choiceoh/models/DeepSeek-V4-Flash-0731 \
    /home/choiceoh/models/refit-corpus/agent-traffic-*.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys

DEFAULT_CURVE = "78.5,56.4,39.6,28.6,21.5"


def parse_curve(raw: str) -> list[float]:
    """Cumulative per-position acceptance percentages -> fractions."""
    vals = [float(v) / 100.0 for v in raw.split(",") if v.strip()]
    if not vals or any(not 0.0 <= v <= 1.0 for v in vals):
        raise ValueError(f"bad per-pos curve: {raw!r}")
    if any(b > a + 1e-9 for a, b in zip(vals, vals[1:])):
        raise ValueError(f"per-pos curve must be non-increasing: {raw!r}")
    return vals


def accept_len_pmf(curve: list[float]) -> list[float]:
    """P(accepted == m) for m in 0..k from the cumulative curve."""
    padded = [1.0] + list(curve) + [0.0]
    return [padded[m] - padded[m + 1] for m in range(len(curve) + 1)]


def sample_accept_len(pmf: list[float], rng: random.Random) -> int:
    x = rng.random()
    acc = 0.0
    for m, p in enumerate(pmf):
        acc += p
        if x < acc:
            return m
    return len(pmf) - 1


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class SuffixIndex:
    """Incremental last-occurrence index of --nmin-grams over the token list."""

    def __init__(self, tokens: list[int], nmin: int, keep: int = 64):
        self.tokens = tokens
        self.nmin = nmin
        self.keep = keep
        self.table: dict[tuple[int, ...], list[int]] = {}
        self.indexed_to = 0

    def extend_to(self, end: int) -> None:
        """Index gram end-positions in [indexed_to, end)."""
        t = self.tokens
        for pos in range(max(self.indexed_to, self.nmin), end + 1):
            key = tuple(t[pos - self.nmin : pos])
            bucket = self.table.setdefault(key, [])
            bucket.append(pos)
            if len(bucket) > self.keep:
                del bucket[0]
        self.indexed_to = max(self.indexed_to, end + 1)

    def best_match(self, t: int, nmax: int) -> tuple[int, int]:
        """Longest suffix match for position t -> (match_len, source_pos).

        source_pos is the position right AFTER the matched gram (the copy
        source of the next token). Returns (0, -1) when nothing matches.
        """
        toks = self.tokens
        if t < self.nmin:
            return 0, -1
        key = tuple(toks[t - self.nmin : t])
        best_n, best_pos = 0, -1
        for pos in reversed(self.table.get(key, ())):
            if pos == t:
                continue
            n = self.nmin
            while (n < nmax and pos - n - 1 >= 0 and t - n - 1 >= 0
                   and toks[pos - n - 1] == toks[t - n - 1]):
                n += 1
            if n > best_n or (n == best_n and pos > best_pos):
                best_n, best_pos = n, pos
        return best_n, best_pos


def iter_docs(paths: list[str]):
    for path in paths:
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "completion" in row:
                        yield row.get("prompt", ""), row["completion"]
                    elif row.get("text"):
                        yield "", row["text"]
        else:
            with open(path, encoding="utf-8") as handle:
                yield "", handle.read()


def simulate_doc(ctx_tokens, gen_tokens, args, pmf, rng):
    tokens = list(ctx_tokens) + list(gen_tokens)
    start = len(ctx_tokens)
    if start < args.nmin:
        start = max(start, min(args.warmup, len(tokens) - 1))
    if len(tokens) - start < 2:
        return None
    index = SuffixIndex(tokens, args.nmin, keep=args.keep)

    stats = {
        "gen_tokens": len(tokens) - start,
        "steps": {"dspark": 0, "hybrid": 0, "oracle": 0},
        "pld_steps": 0,
        "pld_accepted": 0,
        "match_steps": 0,
    }

    def pld_accept(t: int) -> tuple[int, int]:
        index.extend_to(t)
        n, pos = index.best_match(t, args.nmax)
        if n < args.threshold or pos < 0:
            return -1, n
        # Only already-generated tokens may be proposed: a source too close to
        # t truncates (real PLD cannot copy the future).
        proposal = tokens[pos : min(pos + args.k, t)]
        if not proposal:
            return -1, n
        actual = tokens[t : t + args.k]
        return lcp(proposal, actual), n

    for policy in ("dspark", "hybrid", "oracle"):
        rng_p = random.Random(args.seed)
        index.table.clear()
        index.indexed_to = 0
        t = start
        while t < len(tokens):
            ds = sample_accept_len(pmf, rng_p)
            if policy == "dspark":
                adv = 1 + ds
            else:
                copied, n = pld_accept(t)
                if n >= args.threshold:
                    stats["match_steps"] += policy == "hybrid"
                if copied < 0:
                    adv = 1 + ds
                elif policy == "oracle":
                    adv = 1 + max(copied, ds)
                else:
                    adv = 1 + copied
                    stats["pld_steps"] += 1
                    stats["pld_accepted"] += copied
            stats["steps"][policy] += 1
            t += adv
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer dir")
    p.add_argument("--mode", choices=("tokens", "chars"), default="tokens")
    p.add_argument("--k", type=int, default=5, help="draft length (SPEC_TOKENS)")
    p.add_argument("--nmin", type=int, default=3, help="index gram size")
    p.add_argument("--nmax", type=int, default=16, help="max match extension")
    p.add_argument("--threshold", type=int, default=4,
                   help="min match length to route to the copy proposal")
    p.add_argument("--warmup", type=int, default=64,
                   help="context tokens for raw-text docs")
    p.add_argument("--keep", type=int, default=64,
                   help="occurrences kept per gram key")
    p.add_argument("--per-pos", default=DEFAULT_CURVE,
                   help="dspark cumulative per-position acceptance %% "
                        f"(default: ledger {DEFAULT_CURVE})")
    p.add_argument("--floor", type=float, default=23.8,
                   help="no-spec C=1 tok/s floor for the projection")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", dest="json_out", default=None)
    args = p.parse_args()
    if args.threshold < args.nmin:
        sys.exit(f"--threshold ({args.threshold}) must be >= --nmin ({args.nmin})")

    curve = parse_curve(args.per_pos)
    if len(curve) != args.k:
        sys.exit(f"--per-pos has {len(curve)} positions but --k is {args.k}")
    pmf = accept_len_pmf(curve)

    encode = None
    if args.mode == "tokens":
        if not args.tokenizer:
            sys.exit("--tokenizer is required in tokens mode "
                     "(--mode chars is a smoke test only)")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        encode = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    else:
        print("WARN: chars mode — smoke test only, numbers are not quotable")
        encode = lambda text: [ord(c) for c in text]  # noqa: E731

    rng = random.Random(args.seed)
    totals = {"gen_tokens": 0, "pld_steps": 0, "pld_accepted": 0,
              "match_steps": 0, "docs": 0}
    steps = {"dspark": 0, "hybrid": 0, "oracle": 0}
    for prompt, completion in iter_docs(args.files):
        stats = simulate_doc(encode(prompt), encode(completion), args, pmf, rng)
        if stats is None:
            continue
        totals["docs"] += 1
        totals["gen_tokens"] += stats["gen_tokens"]
        totals["pld_steps"] += stats["pld_steps"]
        totals["pld_accepted"] += stats["pld_accepted"]
        totals["match_steps"] += stats["match_steps"]
        for k in steps:
            steps[k] += stats["steps"][k]
    if totals["docs"] == 0 or min(steps.values()) == 0:
        sys.exit("no simulatable documents (too short?)")

    tps = {k: totals["gen_tokens"] / v for k, v in steps.items()}
    report = {
        "docs": totals["docs"],
        "gen_tokens": totals["gen_tokens"],
        "dspark_curve_tokens_per_step": round(1 + sum(curve), 3),
        "tokens_per_step": {k: round(v, 3) for k, v in tps.items()},
        "projected_c1_tok_s": {
            k: round(args.floor * v, 1) for k, v in tps.items()
        },
        "hybrid_uplift_pct": round(100 * (tps["hybrid"] / tps["dspark"] - 1), 2),
        "oracle_uplift_pct": round(100 * (tps["oracle"] / tps["dspark"] - 1), 2),
        "match_step_share_pct": round(
            100 * totals["match_steps"] / steps["hybrid"], 2
        ),
        "pld_mean_accepted_when_routed": round(
            totals["pld_accepted"] / totals["pld_steps"], 3
        ) if totals["pld_steps"] else None,
        "params": {
            "k": args.k, "nmin": args.nmin, "nmax": args.nmax,
            "threshold": args.threshold, "mode": args.mode,
            "seed": args.seed,
        },
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    verdict = ("CLEARS" if report["hybrid_uplift_pct"] >= 7.0 else "BELOW")
    print(f"\nceiling verdict: hybrid +{report['hybrid_uplift_pct']}% "
          f"({verdict} the high-single-digit adoption bar; oracle bound "
          f"+{report['oracle_uplift_pct']}%)")


if __name__ == "__main__":
    main()
