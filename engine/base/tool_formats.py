"""Tool-call wire formats (base): how a chat template teaches a model to write a call, read back as OpenAI tool calls.

A format is what the door (base/serve.Server) takes for tools: `parse` (text -> [(name, arguments JSON)] or None),
`partial` (text still arriving -> [(name, arguments so far, closed)], only ever growing), `grammar` (tools -> an EBNF
grammar, or None) and `start_token` (the call marker's token id). Every format here opens a call with `<tool_call>` and
closes it with `</tool_call>`: the door splits content from calls on that marker (base/serve._Choice.flush), so a format
with other markers needs the door taught first. Formats are named by their layout; a profile says which one its
template writes, or `detect` reads it off the template.

    ARG_PAIRS      <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
    FUNCTION_XML   <tool_call>\\n<function={name}>\\n<parameter={k}>\\n{v}\\n</parameter>\\n...</function>\\n</tool_call>
                   (a call after the first opens with '\\n')

Both layouts write a string value raw and any other value as JSON. The parsers read a value back as JSON when it is
one (numbers, objects, lists, booleans, null) and as text otherwise; a string that happens to read as JSON comes back
as that JSON, which the written text cannot tell apart either.

A grammar binds the declared names and argument keys in the layout, values free. Lazily (`lazy=True`, the default) it
begins after the marker, for a door that arms it at the marker's token and leaves the prose before a call free; eagerly
it begins with the marker, for `tool_choice` required or a named function, where the answer must be calls. With
`parallel=False` it takes one call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_JSONISH = "{[\"-0123456789tfn"


def _without_partial(text: str, tag: str) -> str:
    """`text` without a trailing piece of `tag` that has not finished arriving."""
    for cut in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:cut]):
            return text[:-cut]
    return text


def tool_call_token(tok, marker: str = _OPEN) -> "int | None":
    """The id of `marker`, where a tool grammar arms -- or None if it is not one token.

    A template writes the marker whole and the vocabularies that serve tools have it whole, which
    is what lets a grammar begin exactly there. A checkpoint that spelled it in pieces gets no tool grammar rather than
    one that arms in the middle of the marker.
    """
    if tok is None:
        return None
    try:
        try:
            out = tok.encode(marker, add_special_tokens=False)
        except TypeError:                                    # a tokenizer without the switch
            out = tok.encode(marker)
        ids = list(getattr(out, "ids", out))                 # the Rust tokenizer's Encoding, or a plain list
        return ids[0] if len(ids) == 1 and tok.decode(ids) == marker else None
    except Exception:                                        # noqa: BLE001 -- no tokenizer, no grammar
        return None


def _tools_named(tools):
    """[(name, sorted argument keys)] for the declared tools, or None when one of them has no name to hold a call to."""
    out = []
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = (fn or {}).get("name")
        if not isinstance(name, str) or not name:
            return None
        out.append((name, sorted(k for k in ((fn.get("parameters") or {}).get("properties") or {}) if isinstance(k, str))))
    return out


# -- ARG_PAIRS: <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call> -----------------------------
_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
# the last key whose value has begun and has not ended: everything after it is still arriving
_ARRIVING = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>((?:(?!</arg_value>).)*)$", re.S)
_END_VALUE = "</arg_value>"


def _value(raw: str):
    """A JSON value when it is one (numbers, objects, lists, booleans), otherwise the text itself."""
    text = raw.strip()
    if text and text[0] in _JSONISH:
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def parse_arg_pairs(content: str):
    calls = []
    for body in _CALL.findall(content):
        name_end = body.find("<arg_key>")
        name = (body if name_end < 0 else body[:name_end]).strip()
        if not name:
            continue
        args = {k.strip(): _value(v) for k, v in _ARG.findall(body)}
        calls.append((name, json.dumps(args, ensure_ascii=False)))
    return calls or None


def partial_arg_pairs(content: str):
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


def grammar_arg_pairs(tools, *, lazy: bool = True, parallel: bool = True) -> "str | None":
    """A grammar for this request's tool calls (45차 §45).

    The template teaches the model a shape, and nothing held it to it: a call could name a tool
    that was never declared, or an argument the tool does not take, and the door would hand the
    caller something it cannot make. llama.cpp binds the declared schema from the `<tool_call>`
    trigger onward (`grammar_lazy`, `grammar_triggers`); every marker here is a single token, so
    that trigger is exactly our `grammar_after`.

    What it binds is the name, the argument keys, and the shape. **Not the values** -- a value is
    written as raw text and may be anything, including `<` and code, so its rule admits every
    character. That ambiguity with the closing tag is the point: inside a value the mask forbids
    nothing, and the model closes when it means to.

    Lazily the grammar begins after the trigger token, so its root is what follows `<tool_call>`. Prose
    after a call is not in it -- and costs nothing, because `_Choice.flush` already drops
    everything from the first `<tool_call>` on. Eagerly (`lazy=False`) the answer itself is the calls:
    blank space, then the marker.
    """
    named = _tools_named(tools)
    if not named:
        return None
    calls, rules = [], []
    for i, (name, keys) in enumerate(named):
        rules.append(f'call{i} ::= {json.dumps(name)} pairs{i} "</tool_call>"')
        rules.append(f'pairs{i} ::= ("<arg_key>" key{i} "</arg_key>" "<arg_value>" value "</arg_value>")*'
                     if keys else f'pairs{i} ::= ""')
        if keys:
            rules.append(f"key{i} ::= " + " | ".join(json.dumps(k) for k in keys))
        calls.append(f"call{i}")
    body = 'call ("<tool_call>" call)*' if parallel else "call"
    root = f"root ::= {body}" if lazy else f'root ::= [ \\n]* "<tool_call>" {body}'
    head = [root, "call ::= " + " | ".join(calls), "value ::= [^\\u0000]*"]
    return "\n".join(head + rules) + "\n"


# -- FUNCTION_XML: <tool_call>\n<function={name}>\n<parameter={k}>\n{v}\n</parameter>\n...</function>\n</tool_call> --------
_FUNCTION = re.compile(r"<function=([^>\n]*)>")
_PARAMETER = re.compile(r"<parameter=([^>\n]*)>(.*?)</parameter>", re.S)
_PARAMETER_ARRIVING = re.compile(r"<parameter=([^>\n]*)>((?:(?!</parameter>).)*)$", re.S)
_END_PARAMETER = "</parameter>"
_END_FUNCTION = "</function>"


def _xml_value(raw: str):
    """A value as the template wrote it: the newline after the opening tag and the one before the closing tag are the
    layout, not the value, and nothing else is taken off -- code keeps its indentation. JSON when it reads as JSON."""
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    text = raw.strip()
    if text and text[0] in _JSONISH:
        try:
            return json.loads(text)
        except ValueError:
            pass
    return raw


def _xml_body(body: str):
    """(name, the text holding its parameters) of one call's body, or None while no whole `<function=...>` is in it."""
    head = _FUNCTION.search(body)
    if head is None:
        return None
    inner = body[head.end():]
    stop = inner.find(_END_FUNCTION)
    return head.group(1).strip(), inner if stop < 0 else inner[:stop]


def parse_function_xml(content: str):
    calls = []
    for body in _CALL.findall(content):
        found = _xml_body(body)
        if found is None or not found[0]:
            continue
        name, inner = found
        args = {k.strip(): _xml_value(v) for k, v in _PARAMETER.findall(inner)}
        calls.append((name, json.dumps(args, ensure_ascii=False)))
    return calls or None


def partial_function_xml(content: str):
    """`partial_arg_pairs`'s contract on this layout: (name, arguments so far, closed) per call begun, the text only ever
    growing. A string value streams once its first non-blank character says it is not JSON; while it arrives, a
    trailing piece of `</parameter>` and one trailing newline are held back, because either may turn out to be the
    closing tag and its layout newline rather than value text."""
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
        found = _xml_body(body)
        if found is None or not found[0]:
            if not closed:
                return calls                    # the name is still arriving
            continue
        name, inner = found
        done = [(k.strip(), _xml_value(v)) for k, v in _PARAMETER.findall(inner)]
        text = json.dumps(dict(done), ensure_ascii=False)
        if closed:
            calls.append((name, text, True))
            continue
        text = text[:-1]                        # the object stays open while the call does
        arriving = _PARAMETER_ARRIVING.search(inner)
        if arriving:
            key = arriving.group(1).strip()
            sofar = _without_partial(arriving.group(2), _END_PARAMETER)
            if sofar.startswith("\n"):
                sofar = sofar[1:]
            if sofar.endswith("\n"):
                sofar = sofar[:-1]
            decided = sofar.strip()
            if decided and decided[0] not in _JSONISH:
                text += ((", " if done else "") + json.dumps(key, ensure_ascii=False) + ": "
                         + json.dumps(sofar, ensure_ascii=False)[:-1])
        calls.append((name, text, False))
        return calls


def grammar_function_xml(tools, *, lazy: bool = True, parallel: bool = True) -> "str | None":
    """`grammar_arg_pairs`'s binding on this layout: the declared names and argument keys in the template's exact
    layout, values free. A call after the first opens with a newline, as the template writes it."""
    named = _tools_named(tools)
    if not named:
        return None
    calls, rules = [], []
    for i, (name, keys) in enumerate(named):
        rules.append(f'call{i} ::= "\\n<function=" {json.dumps(name)} ">\\n" params{i} "</function>\\n</tool_call>"')
        rules.append(f'params{i} ::= ("<parameter=" key{i} ">\\n" value "\\n</parameter>\\n")*'
                     if keys else f'params{i} ::= ""')
        if keys:
            rules.append(f"key{i} ::= " + " | ".join(json.dumps(k) for k in keys))
        calls.append(f"call{i}")
    body = 'call ("\\n<tool_call>" call)*' if parallel else "call"
    root = f"root ::= {body}" if lazy else f'root ::= [ \\n]* "<tool_call>" {body}'
    head = [root, "call ::= " + " | ".join(calls), "value ::= [^\\u0000]*"]
    return "\n".join(head + rules) + "\n"


@dataclass(frozen=True)
class ToolFormat:
    name: str
    parse: Callable
    partial: Callable
    grammar: Callable
    marker: str = _OPEN

    def start_token(self, tok) -> "int | None":
        return tool_call_token(tok, self.marker)


ARG_PAIRS = ToolFormat("arg_pairs", parse_arg_pairs, partial_arg_pairs, grammar_arg_pairs)
FUNCTION_XML = ToolFormat("function_xml", parse_function_xml, partial_function_xml, grammar_function_xml)
FORMATS = (ARG_PAIRS, FUNCTION_XML)

PROBE_TOOL = "st_probe_lookup"
PROBE_ARGUMENTS = {"st_probe_query": "probe text", "st_probe_count": 7}


def detect(render) -> "ToolFormat | None":
    """The format `render` (the door's chat renderer: messages, kwargs) writes a call in, read off the template: a
    conversation whose assistant turn called a declared tool is rendered, and the format whose parser reads that very
    call back -- name and arguments -- is the one. None when the template writes no call either parser reads (or
    refuses the conversation): the door then serves no tools, which it says (/v1/models capabilities)."""
    if render is None:
        return None
    tools = [{"type": "function", "function": {"name": PROBE_TOOL, "description": "probe", "parameters": {
        "type": "object", "properties": {"st_probe_query": {"type": "string"}, "st_probe_count": {"type": "integer"}}}}}]
    messages = [{"role": "user", "content": "probe"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "st_probe_0", "type": "function", "function": {
                    "name": PROBE_TOOL, "arguments": dict(PROBE_ARGUMENTS)}}]},
                {"role": "tool", "tool_call_id": "st_probe_0", "content": "probe result"}]
    try:
        text = render(messages, {"tools": tools})
    except Exception:                                        # noqa: BLE001 -- a template that cannot write a call has no format
        return None
    for fmt in FORMATS:
        for name, arguments in fmt.parse(text) or ():
            if name == PROBE_TOOL and json.loads(arguments) == PROBE_ARGUMENTS:
                return fmt
    return None


__all__ = ["ToolFormat", "ARG_PAIRS", "FUNCTION_XML", "FORMATS", "detect", "tool_call_token",
           "parse_arg_pairs", "partial_arg_pairs", "grammar_arg_pairs",
           "parse_function_xml", "partial_function_xml", "grammar_function_xml"]
