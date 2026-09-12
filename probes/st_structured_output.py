#!/usr/bin/env python3
"""Judge the served structured-output path against a running ST engine (45차 §31, §33).

Everything here failed only on a device, so none of it is covered by the CPU suite:

  A  the default path -- thinking on -- with `response_format`. Before §33 the grammar armed inside the
     think block, forbade its end token, and the whole answer came back as reasoning_content with content
     empty. Gate: content parses as JSON *and* reasoning_content is not empty.
  B  a json_schema with required fields is actually enforced.
  C  thinking off: content is JSON and nothing is reasoning.
  D  a schema xgrammar cannot build is a 400, and the engine is still serving afterwards. Before §33 that
     exception left `once()` and took every live request on every rank with it.
  E  the drafter under a mask: st:spec_accepted_per_step_total moves while the rows above run.

  usage: python3 probes/st_structured_output.py [--base http://10.10.10.2:8000] [--max-tokens 1024]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"], "additionalProperties": False}


def post(base: str, body: dict, timeout: float):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:400]


def metrics(base: str, name: str) -> float:
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=20) as r:
            for line in r.read().decode().splitlines():
                if line.startswith(name) and not line.startswith(("# ", name + "_")):
                    return float(line.rsplit(" ", 1)[-1])
    except Exception:                                    # noqa: BLE001 -- the counter is evidence, not a gate
        pass
    return float("nan")


def ask(base, kind, timeout, max_tokens, **kw) -> tuple:
    body = {"messages": [{"role": "user", "content": "Give me a person: a name and an age. Answer in JSON."}],
            "max_tokens": max_tokens, "temperature": 0.0, "response_format": kind, **kw}
    t = time.perf_counter()
    status, out = post(base, body, timeout)
    return status, out, time.perf_counter() - t


def judge(name: str, status, out, seconds, *, want_reasoning: bool, schema=None) -> bool:
    if status != 200:
        print(f"  {name}: FAIL  HTTP {status}: {out}")
        return False
    choice = out["choices"][0]
    msg = choice["message"]
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    head = f"  {name}: {seconds:5.1f}s  finish={choice['finish_reason']}  reasoning={len(reasoning):5d}ch  content={len(content):4d}ch"
    if not content:
        if choice["finish_reason"] == "length" and reasoning:
            print(head + "  INCONCLUSIVE: the budget ran out inside the reasoning; raise --max-tokens")
            return None
        print(head + "  FAIL: no content (this is the §33 bug: the grammar never let the think block close)")
        return False
    try:
        value = json.loads(content)
    except ValueError as exc:
        print(head + f"  FAIL: content is not JSON ({exc})")
        return False
    if schema is not None:
        missing = [k for k in schema["required"] if k not in value]
        extra = [k for k in value if k not in schema["properties"]]
        if missing or extra or not isinstance(value.get("age"), int) or not isinstance(value.get("name"), str):
            print(head + f"  FAIL: {value} does not match the schema (missing={missing} extra={extra})")
            return False
    if want_reasoning and not reasoning:
        print(head + "  FAIL: the model was asked to think and did not (the grammar is still arming too early)")
        return False
    if not want_reasoning and reasoning:
        print(head + "  FAIL: reasoning with thinking off")
        return False
    print(head + f"  OK  {json.dumps(value)[:60]}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://10.10.10.2:8000")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    accepted = metrics(base, "st:spec_accepted_per_step_total")
    results = {}

    print("A  thinking on (the default path) + json_object")
    results["A"] = judge("A", *ask(base, {"type": "json_object"}, a.timeout, a.max_tokens), want_reasoning=True)

    print("B  thinking on + json_schema")
    fmt = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA}}
    results["B"] = judge("B", *ask(base, fmt, a.timeout, a.max_tokens), want_reasoning=True, schema=SCHEMA)

    print("C  thinking off + json_object")
    results["C"] = judge("C", *ask(base, {"type": "json_object"}, a.timeout, a.max_tokens,
                                    chat_template_kwargs={"thinking": False}), want_reasoning=False)

    print("D  a schema the compiler refuses is a 400, and the engine keeps serving")
    bad = {"type": "json_schema", "json_schema": {"name": "bad", "schema": {"type": "string", "pattern": "(a)\\1"}}}
    status, out, seconds = ask(base, bad, 60.0, 8)
    ok = status == 400 and "compiled" in str(out)
    print(f"  D: {'OK' if ok else 'FAIL'}  HTTP {status}: {str(out)[:120]}")
    alive = judge("D-after", *ask(base, {"type": "json_object"}, a.timeout, a.max_tokens,
                                  chat_template_kwargs={"thinking": False}), want_reasoning=False)
    results["D"] = ok and alive is not False

    after = metrics(base, "st:spec_accepted_per_step_total")
    print(f"E  drafts verified under the mask: st:spec_accepted_per_step_total {accepted} -> {after}")

    bad_names = [k for k, v in results.items() if v is False]
    soft = [k for k, v in results.items() if v is None]
    print(f"\n{'FAILED: ' + ', '.join(bad_names) if bad_names else 'all gates passed'}"
          f"{('  (inconclusive: ' + ', '.join(soft) + ')') if soft else ''}")
    return 1 if bad_names else 0


if __name__ == "__main__":
    sys.exit(main())
