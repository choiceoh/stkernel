#!/usr/bin/env python3
"""Opt-in generation check against an explicitly supplied Chat Completions URL.

uv run --with jsonschema python probes/glm53_tool_choice_acceptance.py \
  --url http://HOST:PORT/v1/chat/completions --model glm-5.3-flash
Uses OPENAI_API_KEY if set; prints JSONL verdicts, never executes returned tools.
Run baseline and --required-first on the same image/weights for a quality A/B.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

from jsonschema import validate

LITERAL_TOOL_XML = "<tool_call>record_job</tool_call>"


def read_stream(lines):
    """Assemble tool slots without accepting EOF as a successful SSE finish."""
    slots, content, finish, done = {}, [], None, False
    for line in lines:
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        packet = json.loads(data)
        if "error" in packet:
            raise ValueError("SSE error event")
        for choice in packet.get("choices", []):
            if choice["index"] != 0:
                raise ValueError("Unexpected choice index")
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            content.append(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                slot = slots.setdefault(tc["index"], {"id": "",
                    "function": {"name": "", "arguments": ""}})
                if "type" in tc:
                    if tc["type"] != "function":
                        raise ValueError("Unexpected tool-call type")
                    slot["type"] = tc["type"]
                if tc.get("id"):
                    if slot["id"] and slot["id"] != tc["id"]:
                        raise ValueError("Tool ID changed during streaming")
                    slot["id"] = tc["id"]
                for key in ("name", "arguments"):
                    slot["function"][key] += tc.get("function", {}).get(key) or ""
    if not done or finish is None:
        raise ValueError("Incomplete SSE response")
    return {"content": "".join(content), "tool_calls": [v for _, v in sorted(slots.items())]}, finish


def check_response(message, finish, choice, schema):
    if finish not in ("stop", "tool_calls"):
        raise ValueError(f"Unfinished response: {finish}")
    calls = message.get("tool_calls") or []
    if choice == "none":
        if calls or LITERAL_TOOL_XML not in (message.get("content") or ""):
            raise ValueError("none must preserve the requested XML without tool calls")
        return
    if len(calls) != 1 or not calls[0].get("id"):
        raise ValueError("Expected one complete tool call with an ID")
    if calls[0].get("type") != "function":
        raise ValueError("Expected an explicitly function-typed tool call")
    fn = calls[0]["function"]
    if fn["name"] != "record_job":
        raise ValueError("Unexpected function name")
    validate(json.loads(fn["arguments"]), schema)


def run(args):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
    failures = 0
    for large in (False, True):
        props = {"description": {"type": "string"}, "title": {"type": "string"}}
        if large:
            props.update(location={"type": "object", "properties": {
                "city": {"type": "string"}, "remote": {"type": "boolean"}},
                "required": ["city", "remote"]}, responsibilities={"type": "array",
                "items": {"type": "string"}, "minItems": 10})
        props["tag"] = {"type": "string"}
        schema = {"type": "object", "properties": props, "required": ["tag", "title", *(
            ["location", "responsibilities"] if large else [])]}
        for choice in ("none", "auto", "required", "named"):
            for stream in (False, True):
                for repeat in range(args.repeats):
                    prompt = "Call record_job once for a backend developer in Seoul. Use tag engineering."
                    if large:
                        prompt += " Include a 250-word description and ten detailed responsibilities."
                    if choice == "none":
                        prompt = "Print this XML example literally: " + LITERAL_TOOL_XML
                    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                        "tools": [{"type": "function", "function": {"name": "record_job", "parameters": schema}}],
                        "tool_choice": {"type": "function", "function": {"name": "record_job"}} if choice == "named" else choice,
                        "stream": stream, "temperature": 0, "max_tokens": args.max_tokens,
                        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"}}
                    if args.required_first:
                        body["glm53_required_first"] = True
                    record = {"large": large, "choice": choice, "stream": stream,
                        "repeat": repeat, "required_first": args.required_first, "pass": False}
                    started = time.monotonic()
                    try:
                        req = urllib.request.Request(args.url, data=json.dumps(body).encode(), headers=headers)
                        with urllib.request.urlopen(req, timeout=args.timeout) as response:
                            if stream:
                                message, finish = read_stream(response)
                            else:
                                result = json.load(response)["choices"][0]
                                message, finish = result["message"], result["finish_reason"]
                        record["finish_reason"] = finish
                        check_response(message, finish, choice, schema)
                        record["pass"] = True
                    except Exception as exc:
                        # Avoid dumping credentials, request headers, or model output.
                        record["error"] = type(exc).__name__
                        if isinstance(exc, urllib.error.HTTPError):
                            record["http_status"] = exc.code
                        failures += 1
                    record["seconds"] = round(time.monotonic() - started, 3)
                    print(json.dumps(record), flush=True)
    return bool(failures)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--required-first", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()
    if args.repeats < 1 or args.max_tokens < 1 or args.timeout <= 0:
        ap.error("repeats, max-tokens, and timeout must be positive")
    raise SystemExit(run(args))
