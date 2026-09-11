"""GLM's tool-call wire format, as the chat template teaches the model to write it:

    <tool_call>{name}<arg_key>{k1}</arg_key><arg_value>{v1}</arg_value>...</tool_call>

(chat_template_mm_v2.jinja, the "You are provided with function signatures" block; values that are not
strings are written as JSON). The door hands the assistant's content here and gets OpenAI tool calls
back: (name, arguments-as-JSON-text) per call, in order. Text without a complete call parses to None.
"""
from __future__ import annotations

import json
import re

_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _value(raw: str):
    """A JSON value when it is one (numbers, objects, lists, booleans), otherwise the text itself."""
    text = raw.strip()
    if text and text[0] in "{[\"-0123456789tfn":
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def parse_tool_calls(content: str):
    calls = []
    for body in _CALL.findall(content):
        name_end = body.find("<arg_key>")
        name = (body if name_end < 0 else body[:name_end]).strip()
        if not name:
            continue
        args = {k.strip(): _value(v) for k, v in _ARG.findall(body)}
        calls.append((name, json.dumps(args, ensure_ascii=False)))
    return calls or None
