#!/usr/bin/env python3
"""Per-position CONDITIONAL acceptance, read from the engine's own counters.

Why not tok/s: tok/s mixes acceptance with step cost and swings ~40% at C=1.
Why not the per-request `Draft acceptance rate` line: it is an aggregate.

vLLM exports `spec_decode_num_accepted_tokens_per_pos_total`, which is the
count of drafts whose position i was accepted -- an UNCONDITIONAL (marginal)
profile. Position i is only reachable if i-1 was accepted, so the marginal
decays geometrically even when the per-position conditional rate is flat.
Reading the marginal as if it were conditional makes a healthy drafter look
broken; the conditional rate is pos[i] / pos[i-1].

Deltas around a fixed prompt set, so a warm server can be measured without a
reboot and two boots can be compared on the same workload.
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    # Copy-shaped code: the drafter's best case.
    "Write a Python function that reverses a linked list. Only code.",
    # Structured prose: the middle.
    "Explain in three sentences why B-trees are used for database indexes.",
    # Korean free prose: the drafter's worst case (see the bring-up notes).
    "변압기의 유입자냉식과 유입풍냉식 냉각 방식의 차이를 설명해줘.",
    # Continuation of given text: high redundancy.
    "Repeat the following sentence four times: the quick brown fox jumps.",
]


def _metrics(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
        body = r.read().decode()
    out = {"pos": {}}
    for line in body.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            value = float(value)
        except ValueError:
            continue
        if "num_accepted_tokens_per_pos_total" in name:
            pos = name.split('position="')[1].split('"')[0]
            out["pos"][int(pos)] = value
        elif "num_accepted_tokens_total" in name:
            out["accepted"] = value
        elif "num_drafts_total" in name:
            out["drafts"] = value
        elif "num_draft_tokens_total" in name:
            out["draft_tokens"] = value
    return out


def _generate(base, model, prompt, max_tokens, temperature, seed):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read().decode())
    return body["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--max-tokens", type=int, default=256)
    # The lane benches at 0.95; acceptance is temperature-sensitive, so a
    # comparison is only meaningful between runs at the SAME temperature.
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    before = _metrics(args.base)
    if "drafts" not in before:
        print("no spec_decode counters -- is speculative decoding on?")
        return 1
    for i, prompt in enumerate(PROMPTS):
        text = _generate(
            args.base, args.model, prompt, args.max_tokens,
            args.temperature, args.seed + i,
        )
        print(f"  p{i}: {len(text)} chars")
    after = _metrics(args.base)

    drafts = after["drafts"] - before["drafts"]
    accepted = after["accepted"] - before["accepted"]
    draft_tokens = after["draft_tokens"] - before["draft_tokens"]
    if drafts <= 0:
        print("no drafts in the window -- speculative decoding did not run")
        return 1

    positions = sorted(after["pos"])
    marginal = [after["pos"][p] - before["pos"].get(p, 0.0) for p in positions]

    print(f"\n=== accept profile {args.label} "
          f"(temp {args.temperature}, {int(drafts)} drafts) ===")
    print("  pos  accepted  marginal  conditional")
    previous = drafts
    for pos, count in zip(positions, marginal):
        cond = count / previous if previous else 0.0
        print(f"  {pos:>3}  {int(count):>8}  {count / drafts:>7.1%}  {cond:>10.1%}")
        previous = count
    # +1: the token the target itself verifies and always emits.
    print(f"  accepted/draft {accepted / drafts:.3f}"
          f"  -> tokens/step {1 + accepted / drafts:.3f}")
    print(f"  draft acceptance rate {accepted / draft_tokens:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
