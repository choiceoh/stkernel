#!/usr/bin/env python3
"""CPU-only replay against the selected image's real parser and tokenizer.

Run inside the serving image, with this repo and the tokenizer mounted read-only:
  python3 probes/glm53_chat_contract.py --tokenizer /models/glm-5.3-flash-nvfp4
No model weights, CUDA context, generation request, or running server is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def run(tokenizer_path: str):
    from transformers import AutoTokenizer
    import vllm
    from vllm.entrypoints.chat_utils import parse_chat_messages
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.glm47_moe import Glm47MoeParser
    from vllm.parser import ParserManager
    from vllm.renderers.hf import resolve_chat_template_content_format
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "glm53_chat", root / "overlay/modules/glm53_runtime/glm53_chat.py")
    chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    # The read-only tokenizer never changes during replay. Materialize its real
    # 154k-entry vocabulary once instead of copying it for every parser instance.
    vocab = tokenizer.get_vocab()
    tokenizer.get_vocab = lambda: vocab
    template = (root / "launchers/chat_template_mm_v2.jinja").read_text()
    dispatched = ParserManager.get_parser(tool_parser_name="glm47",
        reasoning_parser_name="glm45", enable_auto_tools=True)
    for manager, name, getter in (
        (ReasoningParserManager, "glm45", "get_reasoning_parser"),
        (ToolParserManager, "glm47", "get_tool_parser"),
    ):
        cls = getattr(manager, getter)(name)
        assert cls._parser_engine_cls is Glm47MoeParser, (name, cls)

    tools = [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object", "properties": {
            "q": {"type": "string"}, "n": {"type": "integer"},
            "flag": {"type": "boolean"}, "meta": {"type": "object"},
            "items": {"type": "array", "items": {"type": "integer"}},
            "empty": {"type": "null"},
        }}}}, {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}]
    args = {"q": '서울 "</think>" 123', "n": 3, "flag": False,
            "meta": {"s": "한글"}, "items": [1, 2], "empty": None}
    call = "<tool_call>lookup" + "".join(
        "<arg_key>" + key + "</arg_key><arg_value>"
        + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        + "</arg_value>" for key, value in args.items()) + "</tool_call>"
    expected_calls = [("lookup", args), ("ping", {})]
    cases = [
        ("prefilled_think", True, "reason</think>answer", "reason", "answer", []),
        ("explicit_think", True, "<think>reason</think>answer", "reason", "answer", []),
        ("empty_reason", True, "</think>answer", "", "answer", []),
        ("unfinished", True, "unfinished reason", "unfinished reason", "", []),
        ("compat_off", False, "answer", "", "answer", []),
        ("two_tools", True, "reason</think>" + call + "<tool_call>ping</tool_call>", "reason", "", expected_calls),
        ("tool_ends_reasoning", True, "reason" + call + "<tool_call>ping</tool_call>", "reason", "", expected_calls),
        ("off_tools", False, call + "<tool_call>ping</tool_call>", "", "", expected_calls),
    ]
    cases = [(*case, "auto") for case in cases]
    ping = "<tool_call>ping</tool_call>"
    for thinking in (True, False):
        prefix, reason = ("reason</think>", "reason") if thinking else ("", "")
        cases.extend([
            (f"none_literal_{thinking}", thinking, prefix + call, reason, call, [], "none"),
            (f"none_fenced_{thinking}", thinking, prefix + "```xml\n" + ping + "\n```",
             reason, "```xml\n" + ping + "\n```", [], "none"),
            (f"none_partial_{thinking}", thinking, prefix + "example <tool_ca",
             reason, "example <tool_ca", [], "none"),
            (f"surrounding_{thinking}", thinking, prefix + "  앞\n" + ping + "\n  뒤  ",
             reason, "  앞\n\n  뒤  ", [("ping", {})], "auto"),
            (f"between_{thinking}", thinking, prefix + ping + " 사이 " + ping + " 뒤 ",
             reason, " 사이  뒤 ", [("ping", {}), ("ping", {})], "auto"),
            (f"literal_think_{thinking}", thinking, prefix + "Use <think>literal</think> here.",
             reason, "Use <think>literal</think> here.", [], "none"),
            (f"whitespace_{thinking}", thinking, prefix + " \n" + ping + "\n ",
             reason, "", [("ping", {})], "auto"),
        ])
    cases.extend([
        ("required", True, "reason</think>" + call, "reason", "", [("lookup", args)], "required"),
        ("named", True, "reason</think>" + call, "reason", "", [("lookup", args)],
         {"type": "function", "function": {"name": "lookup"}}),
        ("none_implicit_end", True, "reason" + ping, "reason", ping, [], "none"),
        ("none_no_tools", True, "reason</think>" + ping, "reason", ping, [], "none"),
    ])
    replays = 0
    for name, thinking, text, expected_reason, expected_content, expected_tools, choice in cases:
        body = chat.normalize_chat_options({"model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": "test"}], "tools": tools,
            "tool_choice": choice, "chat_template_kwargs": {"thinking": thinking}})
        if name == "none_no_tools":
            body.pop("tools")
        request = ChatCompletionRequest(**body)
        prompt_ids = tokenizer.apply_chat_template(body["messages"], tools=tools,
            chat_template=template, tokenize=True, add_generation_prompt=True,
            **body["chat_template_kwargs"])
        def parser():
            return dispatched(tokenizer, tools=request.tools,
                                 chat_template_kwargs=body["chat_template_kwargs"])
        original_choice = request.tool_choice
        parser().adjust_request(request)
        assert request.tool_choice == original_choice, (name, "changed tool choice")
        assert request.skip_special_tokens is False
        if choice in ("required",) or isinstance(choice, dict):
            assert request.structured_outputs is not None, (name, "missing structural tag")
        expected = (expected_reason, expected_content, expected_tools)
        reasoning, content, calls = parser().parse(text, request, enable_auto_tools=True)
        actual = (reasoning or "", content or "",
                  [(c.name, json.loads(c.arguments)) for c in calls or []])
        assert actual == expected, (name, "non-streaming", actual, expected)
        # Also retain direct-engine coverage: it is a separate upstream entry.
        direct = Glm47MoeParser(tokenizer, tools=request.tools,
                               chat_template_kwargs=body["chat_template_kwargs"])
        r, c, tcs = direct.parse(text, request, enable_auto_tools=True)
        assert (r or "", c or "", [(tc.name, json.loads(tc.arguments)) for tc in tcs or []]) == expected, (name, "direct")
        hidden_request = request.model_copy(update={"include_reasoning": False})
        hidden = parser().parse_delta(text, [], hidden_request,
            prompt_token_ids=prompt_ids, finished=True)
        assert hidden is None or not hidden.reasoning, (name, "reasoning visibility")

        if calls:
            # Exercise the real Chat API request validation and preprocessing:
            # deprecated reasoning_content, JSON-string arguments, null body,
            # and array-format tool results arriving in reverse call order.
            history = body["messages"] + [{"role": "assistant", "content": None,
                "reasoning_content": expected_reason, "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in calls]}] + [
                {"role": "tool", "tool_call_id": c.id,
                 "content": [{"type": "text", "text": "result " + c.id}]} for c in reversed(calls)]
            roundtrip_request = ChatCompletionRequest(model=body["model"], messages=history)
            # Text-only configuration: no media processor or model weights.
            content_format = resolve_chat_template_content_format(template, tools, "auto",
                tokenizer, model_config=None)  # Explicit template requires no model config.
            conversation, mm_data, _ = parse_chat_messages(roundtrip_request.messages,
                SimpleNamespace(multimodal_config=None, enable_prompt_embeds=False),
                content_format=content_format)
            assert mm_data is None
            if expected_reason:
                assert conversation[1]["reasoning_content"] == expected_reason
            rendered = tokenizer.apply_chat_template(conversation, tools=tools,
                chat_template=template, tokenize=False, add_generation_prompt=True)
            assert "None" not in rendered, (name, "null content")
            for c in calls:
                assert rendered.count("result " + c.id) == 1, (name, "lost tool response")

        # Every character cut, one-character chunks, and one combined delta.
        # Text-only splits exercise delimiter buffering without token hints.
        partitions = [[(text, [])], [(char, []) for char in text]]
        partitions += [[(text[:cut], []), (text[cut:], [])] for cut in range(1, len(text))]
        # Every tokenizer boundary whose decoded prefix is complete UTF-8.
        ids = tokenizer.encode(text, add_special_tokens=False)
        for cut in range(1, len(ids)):
            prefix = tokenizer.decode(ids[:cut], skip_special_tokens=False)
            if text.startswith(prefix) and not prefix.endswith("\ufffd"):
                partitions.append([(prefix, ids[:cut]), (text[len(prefix):], ids[cut:])])
        for partition in partitions:
            stream = parser()
            reason_parts, content_parts, slots = [], [], {}
            for index, (delta_text, delta_ids) in enumerate(partition):
                delta = stream.parse_delta(delta_text, delta_ids, request,
                    prompt_token_ids=prompt_ids, finished=index == len(partition) - 1)
                if delta is None:
                    continue
                data = delta.model_dump(exclude_none=True)
                reason_parts.append(data.get("reasoning", data.get("reasoning_content", "")))
                content_parts.append(data.get("content", ""))
                for tc in data.get("tool_calls", []):
                    slot = slots.setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        assert not slot["id"] or slot["id"] == tc["id"], (name, "unstable tool id")
                        slot["id"] = tc["id"]
                    fn = tc.get("function", {})
                    slot["name"] += fn.get("name", "")
                    slot["arguments"] += fn.get("arguments", "")
            actual = ("".join(reason_parts), "".join(content_parts), [
                (slot["name"], json.loads(slot["arguments"])) for _, slot in sorted(slots.items())])
            assert actual == expected, (name, partition, actual, expected)
            assert len({s["id"] for s in slots.values()}) == len(slots), (name, "duplicate tool ids")
            assert all(s["id"] for s in slots.values()), (name, "missing tool ids")
            replays += 1

        print(f"PASS {name}", flush=True)

    sources = [Path(inspect.getfile(Glm47MoeParser)),
               root / "launchers/chat_template_mm_v2.jinja"]
    print(json.dumps({"vllm": vllm.__version__, "tokenizer": tokenizer_path,
        "cases": len(cases), "replays": replays,
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    run(ap.parse_args().tokenizer)
