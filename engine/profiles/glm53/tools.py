"""GLM's tool-call wire format, as the chat template teaches the model to write it:

    <tool_call>{name}<arg_key>{k1}</arg_key><arg_value>{v1}</arg_value>...</tool_call>

(chat_template_mm_v2.jinja, the "You are provided with function signatures" block; values that are not
strings are written as JSON). The door hands the assistant's content here and gets OpenAI tool calls
back: (name, arguments-as-JSON-text) per call, in order. Text without a complete call parses to None.

The format lives in engine/base/tool_formats (`GLM`) beside the other templates' formats, so a new model whose
template writes this layout binds it without this profile; these are the names the GLM boot and tests import.
"""
from __future__ import annotations

from engine.base.tool_formats import grammar_glm as tool_grammar
from engine.base.tool_formats import parse_glm as parse_tool_calls
from engine.base.tool_formats import partial_glm as partial_tool_calls
from engine.base.tool_formats import tool_call_token

__all__ = ["parse_tool_calls", "partial_tool_calls", "tool_call_token", "tool_grammar"]
