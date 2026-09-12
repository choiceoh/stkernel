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
_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
# the last key whose value has begun and has not ended: everything after it is still arriving
_ARRIVING = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>((?:(?!</arg_value>).)*)$", re.S)
_JSONISH = "{[\"-0123456789tfn"
_END_VALUE = "</arg_value>"


def _without_partial(text: str, tag: str) -> str:
    """`text` without a trailing piece of `tag` that has not finished arriving."""
    for cut in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:cut]):
            return text[:-cut]
    return text


def _value(raw: str):
    """A JSON value when it is one (numbers, objects, lists, booleans), otherwise the text itself."""
    text = raw.strip()
    if text and text[0] in _JSONISH:
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


def partial_tool_calls(content: str):
    """Every call the text has begun, with as much of its arguments as is already certain:
    (name, arguments so far, whether the call closed), in order.

    The door streams `arguments` in fragments, the way OpenAI's tool-call deltas do, so a caller
    does not wait for `</tool_call>` to see the first key -- and on this format the arguments are
    most of the call, so that wait was most of the answer. A fragment need not be JSON on its
    own; the concatenation is. So each call's text here is the finished call's JSON cut off at
    the last thing the model has actually written, and **it only ever grows** -- which is the
    property the door subtracts against, and what the tests pin.

    That is why a value is only streamed when its type is already decided. `_value` reads the
    first character: anything that could still parse as JSON waits for `</arg_value>`, because
    until then its rendering is unknown. A plain string is streamed as it arrives, minus any
    trailing whitespace, because `_value` will strip it and a fragment may not have to be taken
    back.
    """
    calls, at = [], 0
    while True:
        start = content.find(_OPEN, at)
        if start < 0:
            return calls
        body_at = start + len(_OPEN)
        end = content.find(_CLOSE, body_at)
        closed = end >= 0
        body = content[body_at:end] if closed else content[body_at:]
        at = end + len(_CLOSE) if closed else len(content)
        key_at = body.find("<arg_key>")
        if key_at < 0 and not closed:
            return calls                        # the name itself is still arriving
        name = (body if key_at < 0 else body[:key_at]).strip()
        if not name:
            if not closed:
                return calls
            continue
        done = [(k.strip(), _value(v)) for k, v in _ARG.findall(body)]
        text = json.dumps(dict(done), ensure_ascii=False)
        if closed:
            calls.append((name, text, True))
            continue
        text = text[:-1]                        # the object stays open while the call does
        arriving = _ARRIVING.search(body)
        if arriving:
            key = arriving.group(1).strip()
            # `</arg_value` is not value text -- it is a closing tag halfway here, and the
            # negative lookahead above only rejects the whole one. The same holdback the door
            # does for `<tool_call>`, one level down.
            sofar = _without_partial(arriving.group(2), _END_VALUE).strip()
            if sofar and sofar[0] not in _JSONISH:
                text += ((", " if done else "") + json.dumps(key, ensure_ascii=False) + ": "
                         + json.dumps(sofar, ensure_ascii=False)[:-1])
        calls.append((name, text, False))
        return calls
