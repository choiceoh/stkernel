#!/usr/bin/env python3
"""Does our V4.1 completion parser agree with DeepSeek's? No GPU, no model.

`dsv41_encoding` was written from the checkpoint's `encoding/README.md` and
corrected once against `encoding/encoding.py`, but it was never RUN against
that file -- every other piece of this bring-up is held to a differential test
and this one was not. It also ships its own fixtures, which is better evidence
than a comparison invented here.

Two things are compared, and only the second is about formatting:

    1. the reference's own fixtures. `encode_messages` over
       encoding/tests/test_input_N.json must reproduce test_output_N.txt.
       That proves the reference runs correctly HERE before anything is
       graded against it.
    2. parsing. For completions built with the reference's own encoder,
       `parse_message_from_completion_text` and our
       `split_thinking` + `parse_tool_calls` must agree on the reasoning /
       content split and on every tool name and argument VALUE.

Container shape is deliberately not compared: the reference returns OpenAI
tool_calls (an id, a "function" wrapper, arguments as a JSON string) because
that is its API's shape, and ours returns names and decoded values because that
is what a vLLM tool parser hands on. Comparing those would be comparing two
correct answers to different questions.

    python3 probes/dsv41_encoding_diff.py --encoding-dir .../encoding
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_encoding"))

import dsv41_encoding as ours  # noqa: E402


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("_dsv41_ref_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_args(value):
    """The reference hands arguments back as a JSON string; ours decodes."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding-dir", required=True,
                    help="the checkpoint's encoding/ directory")
    args = ap.parse_args()
    d = Path(args.encoding_dir)
    ref = load_reference(d / "encoding.py")
    fail = 0

    # -- 1. the reference reproduces its own fixtures here -----------------
    cases = sorted(p.stem.split("_")[-1]
                   for p in (d / "tests").glob("test_input_*.json"))
    for case in cases:
        want = (d / "tests" / f"test_output_{case}.txt").read_text()
        # their harness's own call: load_cases -> encode_case, thinking_mode
        # "chat". Going through encode_messages directly drops the `tools` the
        # fixtures carry.
        payload = ref.load_cases(str(d / "tests" / f"test_input_{case}.json"))[0]
        got, _media = ref.encode_case(payload, thinking_mode="chat")
        ok = got == want
        fail |= not ok
        print(f"  fixture {case}  {'OK' if ok else 'MISMATCH'}"
              f"  ({len(want)} chars)")
        if not ok:
            for i, (a, b) in enumerate(zip(got, want)):
                if a != b:
                    print(f"      first differs at {i}: "
                          f"got {got[i:i+40]!r} want {want[i:i+40]!r}")
                    break

    # -- 2. parsing, on completions the REFERENCE ENCODER produced ---------
    # Hand-writing the markup was how the first version of this probe went
    # wrong: the wire format carries a newline after every tag
    # (`<|DSML| calls>\n<|DSML| invoke ...>\n`) and a hand-written sample
    # without them is rejected outright by the reference -- which means it
    # would have tested our parser against a string the model never emits.
    # Round-tripping through `encode_messages` removes the question.
    def completion_of(messages, mode):
        prompt = ref.encode_messages(messages, thinking_mode=mode,
                                     drop_thinking=False)
        marker = "<\uff5cAssistant\uff5c>"
        body = prompt[prompt.rindex(marker) + len(marker):]
        # the prompt ends with <think> (thinking) or </think> (chat); the
        # model's own output starts after it
        for opener in ("<think>", "</think>"):
            if body.startswith(opener):
                return body[len(opener):]
        return body

    tool = [{"type": "function", "function": {
        "name": "search",
        "arguments": json.dumps({"q": "seoul weather", "limit": 3,
                                 "tags": ["a", "b"]})}}]
    user = {"role": "user", "content": "weather?"}
    trials = [
        ("thinking, plain", "thinking",
         [user, {"role": "assistant", "content": "The answer is 4.",
                 "reasoning_content": "weighing it."}]),
        ("thinking, empty reasoning", "thinking",
         [user, {"role": "assistant", "content": "Fine.",
                 "reasoning_content": ""}]),
        ("chat, plain", "chat",
         [user, {"role": "assistant", "content": "The answer is 4."}]),
        ("thinking, tool call", "thinking",
         [user, {"role": "assistant", "content": "Let me check.",
                 "reasoning_content": "needs a lookup.", "tool_calls": tool}]),
        ("chat, tool call", "chat",
         [user, {"role": "assistant", "content": "Checking.",
                 "tool_calls": tool}]),
        ("thinking, tool call, no prose", "thinking",
         [user, {"role": "assistant", "content": "",
                 "reasoning_content": "needs a lookup.", "tool_calls": tool}]),
    ]
    eos = f"<{ours.BAR}end\u2581of\u2581sentence{ours.BAR}>"
    for label, mode, messages in trials:
        text = completion_of(messages, mode)
        ref_mode = "thinking" if mode == "thinking" else "non_thinking"
        try:
            r = ref.parse_message_from_completion_text(text,
                                                       thinking_mode=ref_mode)
        except Exception as exc:                      # noqa: BLE001
            print(f"  parse {label:28s} reference raised {exc!r}")
            fail = 1
            continue
        thinking = mode == "thinking"
        # ours never sees the EOS: vLLM strips it before the parser runs
        body = text[:-len(eos)] if text.endswith(eos) else text
        reasoning, content = ours.split_thinking(body, thinking=thinking)
        content = ours.strip_tool_calls(content)
        calls = ours.parse_tool_calls(body)

        ok = reasoning == r["reasoning_content"].strip()
        ok &= content == r["content"].strip()
        ref_calls = [(c["function"]["name"],
                      norm_args(c["function"]["arguments"]))
                     for c in r.get("tool_calls", [])]
        our_calls = [(c["name"], c["arguments"]) for c in calls]
        ok &= ref_calls == our_calls
        fail |= not ok
        print(f"  parse {label:28s} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"      reasoning  ref {r['reasoning_content'].strip()!r}")
            print(f"                 our {reasoning!r}")
            print(f"      content    ref {r['content'].strip()!r}")
            print(f"                 our {content!r}")
            print(f"      calls      ref {ref_calls}")
            print(f"                 our {our_calls}")

    print("\n" + ("ENCODING FAIL" if fail else "ENCODING PASS"))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
