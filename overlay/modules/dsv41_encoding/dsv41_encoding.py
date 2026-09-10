"""DeepSeek-V4.1 prompt format: the three places it is not V4's.

Taken from the checkpoint's own `encoding/README.md`, which states the deltas
explicitly rather than leaving them to be diffed out of a chat template:

  1. DSML tag names carry a LEADING SPACE. `<|DSML| calls>` with
     `<|DSML| invoke>` / `<|DSML| parameter>` inside, where V4 wrote
     `<|DSML|tool_calls>` with no space. A parser built for V4 does not
     half-work on this -- it matches nothing.

     The exact grammar below is transcribed from the checkpoint's own
     `encoding/encoding.py` (`tool_calls_block_name`, `tool_call_tag_name`,
     `tool_parameter_tag_name`, `encode_arguments_to_dsml`) rather than
     inferred from the README's prose, because one detail is invisible from
     the prose and changes results: every parameter carries
     `string="true|false"`, and `false` means the VALUE IS JSON. Reading a
     parameter without honouring that attribute returns the string "42" where
     the tool expects the number 42, silently, for every non-string argument.
  2. Reasoning effort is a NUMBER, 1-100, rendered as
     `Reasoning Effort: {n} (range 1-100, ...)`, replacing V4's natural-language
     descriptions. The string aliases still exist and map onto the scale.
  3. Mid-conversation system messages are legal, via `<|System|>`, and behave
     like a user message for the purpose of appending the generation header.

Only (2) has a default worth arguing about and the card fixes it: "high" (50).

The tokens below are written with ASCII bars in the SOURCE so this file stays
diffable, and are built from the real full-width character at import. The
checkpoint uses U+FF5C FULLWIDTH VERTICAL LINE, not U+007C, and a tokenizer
handed the ASCII form does not fail loudly -- it silently tokenizes the tag as
ordinary text, which is the kind of bug that shows up as a quality regression
three brackets later.
"""

from __future__ import annotations

import json
import re

# U+FF5C, not the ASCII bar.
BAR = "｜"

BOS = f"<{BAR}begin▁of▁sentence{BAR}>"
EOS = f"<{BAR}end▁of▁sentence{BAR}>"
SYSTEM = f"<{BAR}System{BAR}>"
USER = f"<{BAR}User{BAR}>"
ASSISTANT = f"<{BAR}Assistant{BAR}>"

# Change 1: the space that opens each tag name is load-bearing -- and so is a
# newline the README does not mention. The wire format, taken from the golden
# fixture encoding/tests/test_output_1.txt, is
#
#     \n\n<|DSML| calls>\n
#     <|DSML| invoke name="TOOL">\n
#     <|DSML| parameter name="P" string="true|false">VALUE</|DSML| parameter>\n
#     </|DSML| invoke>\n
#     </|DSML| calls>
#
# The reference parser REJECTS the same markup without those newlines
# ("expected '>\\n' but got '>'"), so a hand-written sample is not a smaller
# version of real output -- it is output the model never produces. The regexes
# below tolerate the whitespace rather than requiring it, and
# probes/dsv41_encoding_diff.py tests them against strings the reference's own
# encoder emitted rather than against any written here.
DSML_TOKEN = f"{BAR}DSML{BAR}"
CALLS_BLOCK_NAME = " calls"
INVOKE_TAG_NAME = " invoke"
PARAMETER_TAG_NAME = " parameter"

DSML_CALLS_OPEN = f"<{DSML_TOKEN}{CALLS_BLOCK_NAME}>"
DSML_CALLS_CLOSE = f"</{DSML_TOKEN}{CALLS_BLOCK_NAME}>"

# Change 2.
EFFORT_ALIASES = {"low": 25, "high": 50, "xhigh": 75, "max": 100}
EFFORT_DEFAULT = "high"
EFFORT_TEMPLATE = ("Reasoning Effort: {budget} (range 1-100, the higher the "
                   "value, the more thorough the reasoning)")

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def resolve_effort(effort: "int | str | None") -> int:
    """`"low"`/`"high"`/`"xhigh"`/`"max"` or an int in 1..100."""
    if effort is None:
        effort = EFFORT_DEFAULT
    if isinstance(effort, str):
        key = effort.strip().lower()
        if key not in EFFORT_ALIASES:
            raise ValueError(
                f"unknown reasoning effort {effort!r}; "
                f"use {sorted(EFFORT_ALIASES)} or an int 1-100")
        return EFFORT_ALIASES[key]
    budget = int(effort)
    if not 1 <= budget <= 100:
        raise ValueError(f"reasoning effort {budget} outside 1-100")
    return budget


def render_effort_prefix(effort: "int | str | None", *, thinking: bool,
                         turn_index: int) -> str:
    """The effort line, or "" when it does not belong.

    Two conditions, both from the card: thinking mode only, and the beginning of
    the conversation only (index 0). Rendering it on a later turn is not a
    harmless duplicate -- it puts a system-voice instruction in the middle of a
    dialogue the model was trained to read as settled.
    """
    if not thinking or turn_index != 0:
        return ""
    return EFFORT_TEMPLATE.format(budget=resolve_effort(effort))


def is_generation_anchor(role: str) -> bool:
    """Change 3: a mid-conversation system message anchors generation.

    `system` behaves like `user` here -- after it, the assistant header is
    appended. V4 had no such case, so a port that maps roles one-to-one gets
    this wrong by omission rather than by a wrong branch.
    """
    return role in ("user", "system")


def role_token(role: str, *, turn_index: int) -> str:
    if role == "system":
        # The leading system message is the prompt preamble; a later one is an
        # in-band turn. Both use the same token, which is why turn_index has to
        # travel with the role.
        return SYSTEM
    if role == "user":
        return USER
    if role == "assistant":
        return ASSISTANT
    raise ValueError(f"unknown role {role!r} at turn {turn_index}")


_CALLS_RE = re.compile(
    re.escape(DSML_CALLS_OPEN) + r"(.*?)" + re.escape(DSML_CALLS_CLOSE), re.S)
_INVOKE_RE = re.compile(
    re.escape(f"<{DSML_TOKEN}{INVOKE_TAG_NAME}") + r'\s+name="([^"]*)"\s*>(.*?)'
    + re.escape(f"</{DSML_TOKEN}{INVOKE_TAG_NAME}>"), re.S)
# `string` is optional in the wild but its ABSENCE is not the same as "true":
# the encoder always writes it, so a missing one means malformed output, and
# treating the value as a raw string is the conservative read.
_PARAM_RE = re.compile(
    re.escape(f"<{DSML_TOKEN}{PARAMETER_TAG_NAME}") +
    r'\s+name="([^"]*)"(?:\s+string="(true|false)")?\s*>(.*?)'
    + re.escape(f"</{DSML_TOKEN}{PARAMETER_TAG_NAME}>"), re.S)


def split_thinking(completion: str, *, thinking: bool) -> "tuple[str, str]":
    """(reasoning, content).

    GLM-5.3 taught this the expensive way: a template that ends in an open
    `<think>` while the request said thinking=false leaks the reasoning into
    content, because the server skips the parser for such requests. Here the
    split is unconditional and `thinking=False` simply yields no reasoning, so a
    template change cannot route thoughts into the answer.
    """
    if not thinking:
        return "", completion.replace(_THINK_OPEN, "").strip()
    body = completion
    if body.startswith(_THINK_OPEN):
        body = body[len(_THINK_OPEN):]
    if _THINK_CLOSE not in body:
        # Still inside the reasoning span: nothing is content yet.
        return body.strip(), ""
    reasoning, content = body.split(_THINK_CLOSE, 1)
    return reasoning.strip(), content.strip()


def parse_tool_calls(text: str) -> "list[dict]":
    """Every `<|DSML| invoke>` in the completion, in order, with typed arguments.

    `string="true"` hands the value back verbatim. `string="false"` means the
    encoder wrote JSON, so it is decoded -- and a value that does not parse is
    returned as its raw text rather than raising, because a malformed argument
    is the model's error to report to the caller and not this parser's to
    escalate into a 500.
    """
    calls = []
    for block in _CALLS_RE.findall(text):
        for name, params in _INVOKE_RE.findall(block):
            arguments = {}
            for key, is_string, raw in _PARAM_RE.findall(params):
                if is_string == "false":
                    try:
                        arguments[key] = json.loads(raw.strip())
                        continue
                    except ValueError:
                        pass
                arguments[key] = raw
            calls.append({"name": name, "arguments": arguments})
    return calls


def strip_tool_calls(text: str) -> str:
    """The prose with the call blocks removed."""
    return _CALLS_RE.sub("", text).strip()
