"""What the sampler kernel is worth, and whether it is right, against a serving ST engine (45차 §34).

    python3 probes/st_sampler_serving.py [--url http://10.10.10.2:8000] [--tokens 128]

Everything here runs against ONE boot, because the engine is production now and a window is a
deployment, not a fleet reservation. That rules out before/after, so the comparison this makes is
the one a single boot can carry: **greedy rows against stochastic rows**. A greedy row is picked by
the captured argmax and never enters the sampler kernel; a stochastic row is the kernel's whole
path. The gap between their inter-token latencies is the sampler's share of a decode step, and it
is what the offline number (467 us for 24 rows of 154,880, `probes/engine_sampler_bench.py`) has to
be recognisable in.

Three correctness claims come with it, none of which needs a second engine to judge:

  top_k=1     At any temperature, keeping one token means the argmax. If the threshold search is
              wrong by even one position this diverges from the temperature-0 answer immediately,
              at a real 154,880-token vocabulary on real logits -- which is the part no CPU test
              reaches. This is the decisive one.
  seed        The same seed twice is the same tokens (D12). It is also four ranks agreeing: they
              draw from their own generators and a disagreement would not survive a sentence.
  top_p tight A nucleus of 0.01 keeps one token wherever the model is confident, so it should
              track greedy closely. Reported, not asserted -- the model is entitled to be unsure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPT = "Count from one to twenty in English words, one per line, nothing else."


def ask(url: str, timeout: float, **body):
    body.setdefault("model", "glm-5.3")
    body.setdefault("messages", [{"role": "user", "content": PROMPT}])
    body.setdefault("stream", False)
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    text = out["choices"][0]["message"].get("content") or ""
    used = out.get("usage", {}).get("completion_tokens", 0)
    return text, used, time.monotonic() - start


def histogram(url: str, timeout: float, name: str):
    """(sum, count) of a histogram series from /metrics, or None when it has not been touched."""
    with urllib.request.urlopen(url + "/metrics", timeout=timeout) as r:
        body = r.read().decode()
    total = count = None
    for line in body.splitlines():
        if line.startswith(name + "_sum"):
            total = float(line.rsplit(" ", 1)[1])
        elif line.startswith(name + "_count"):
            count = float(line.rsplit(" ", 1)[1])
    return None if total is None or not count else (total, count)


def gap(before, after):
    """Seconds a token cost over the window between two readings of a histogram."""
    if before is None or after is None or after[1] <= before[1]:
        return None
    return (after[0] - before[0]) / (after[1] - before[1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://10.10.10.2:8000")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3, help="requests per arm; the arms alternate")
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args(argv)
    n, bad = a.tokens, 0

    print("== is it right ==")
    greedy, _, _ = ask(a.url, a.timeout, temperature=0.0, max_tokens=n)
    for t in (0.7, 1.0, 1.3):
        got, _, _ = ask(a.url, a.timeout, temperature=t, top_k=1, max_tokens=n)
        same = got == greedy
        bad += not same
        print(f"  top_k=1 at temperature {t}: {'matches greedy' if same else 'DIVERGES from greedy'}")
        if not same:
            for i, (x, y) in enumerate(zip(greedy, got)):
                if x != y:
                    print(f"    first difference at character {i}: {greedy[i:i+40]!r} vs {got[i:i+40]!r}")
                    break

    one, _, _ = ask(a.url, a.timeout, temperature=1.0, top_p=0.9, seed=1234, max_tokens=n)
    two, _, _ = ask(a.url, a.timeout, temperature=1.0, top_p=0.9, seed=1234, max_tokens=n)
    bad += one != two
    print(f"  seed 1234 twice: {'same tokens' if one == two else 'DIFFERENT tokens (replay is broken)'}")
    other, _, _ = ask(a.url, a.timeout, temperature=1.0, top_p=0.9, seed=99, max_tokens=n)
    print(f"  a different seed: {'different tokens (the draw is live)' if other != one else 'the same -- is anything being drawn?'}")

    tight, _, _ = ask(a.url, a.timeout, temperature=1.0, top_p=0.01, max_tokens=n)
    shared = sum(x == y for x, y in zip(tight, greedy))
    print(f"  top_p=0.01 against greedy: {shared}/{min(len(tight), len(greedy))} characters shared (reported, not judged)")

    print("== what it costs ==")
    print("  (a greedy row never enters the kernel; a stochastic row is all of it)")
    arms = {"greedy": dict(temperature=0.0),
            "temperature only": dict(temperature=0.8),
            "temperature + top_p": dict(temperature=0.8, top_p=0.9),
            "temperature + top_k": dict(temperature=0.8, top_k=50)}
    itl = {}
    for name, body in arms.items():
        before = histogram(a.url, a.timeout, "vllm:inter_token_latency_seconds")
        wall, tokens = 0.0, 0
        for _ in range(a.rounds):
            _, used, seconds = ask(a.url, a.timeout, max_tokens=n, **body)
            wall, tokens = wall + seconds, tokens + used
        after = histogram(a.url, a.timeout, "vllm:inter_token_latency_seconds")
        step = gap(before, after)
        itl[name] = step
        shown = f"{step * 1000:7.2f} ms/step" if step else "   (no step histogram)"
        print(f"  {name:22s} {shown}   wall {wall / max(tokens, 1) * 1000:6.2f} ms/token over {tokens} tokens")
    if itl.get("greedy") and itl.get("temperature + top_p"):
        delta = (itl["temperature + top_p"] - itl["greedy"]) * 1000
        print(f"  the sampler's share of a decode step: {delta:+.2f} ms")
        print("  (45차 §34 measured the kernel at 0.47 ms for 24 rows offline; the sort it replaced was 4.4 ms)")

    print("FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
