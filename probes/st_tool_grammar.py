#!/usr/bin/env python3
"""Does the tool-call grammar hold a call to the tools that were declared? (45차 §45)

Runs inside the ST image; no server, no GPU -- the tokenizer and an xgrammar compile. The
grammar is the one the door sends, built by `profiles/glm53/tools.tool_grammar` and armed at
the `<tool_call>` token, so what this walks is what a row would be masked by.

  accept  a Korean value; a value holding `<` and code; a tool that takes nothing; two calls
  refuse  a tool nobody declared; an argument the tool does not take; an argument on a tool
          that takes none

  usage: docker exec -i <container> python3 - < probes/st_tool_grammar.py [--ckpt PATH]
         (ST_REPO points at the engine tree to gate; /repo, the released one, by default)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CKPT = "/home/choiceoh/models/glm53-redhat-nvfp4"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
              "type": "object", "properties": {"city": {}, "days": {}}}}},
         {"type": "function", "function": {"name": "send_mail", "parameters": {
              "type": "object", "properties": {"to": {}, "body": {}}}}},
         {"type": "function", "function": {"name": "ping"}}]

CASES = [
    (True, "a Korean value", 'get_weather<arg_key>city</arg_key><arg_value>서울특별시</arg_value></tool_call>'),
    (True, "a value holding < and code",
     'send_mail<arg_key>body</arg_key><arg_value>if (a<b) { return "<x>"; }</arg_value></tool_call>'),
    (True, "a tool that takes nothing", "ping</tool_call>"),
    (True, "two calls", 'ping</tool_call><tool_call>get_weather<arg_key>days</arg_key>'
                        '<arg_value>3</arg_value></tool_call>'),
    (False, "a tool nobody declared", 'get_stock<arg_key>city</arg_key><arg_value>x</arg_value></tool_call>'),
    (False, "an argument the tool does not take",
     'get_weather<arg_key>zipcode</arg_key><arg_value>x</arg_value></tool_call>'),
    (False, "an argument on a tool that takes none", 'ping<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("ST_CKPT", CKPT))
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import xgrammar as xgr
    from transformers import AutoTokenizer

    sys.path.insert(0, os.environ.get("ST_REPO", "/repo"))
    from engine.profiles.glm53.tools import parse_tool_calls, tool_call_token, tool_grammar

    tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
    vocab = len(tok)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab))
    trigger = tool_call_token(tok)
    print(f"trigger token: {trigger}")
    if trigger is None:
        print("FAIL: `<tool_call>` is not one token here, so no grammar would be armed")
        return 1
    grammar = compiler.compile_grammar(xgr.Grammar.from_ebnf(tool_grammar(TOOLS)))

    def walk(text):
        matcher = xgr.GrammarMatcher(grammar)
        mask = xgr.allocate_token_bitmask(1, vocab)
        for n, tid in enumerate(tok(text, add_special_tokens=False)["input_ids"]):
            matcher.fill_next_token_bitmask(mask)
            if not (mask[0, tid >> 5].item() >> (tid & 31)) & 1:
                return f"refused at {n} ({tok.decode([tid])!r})"
            matcher.accept_token(tid)
        return "accepted"

    failed = []
    width = max(len(label) for _, label, _ in CASES)
    for want, label, text in CASES:
        got = walk(text)
        ok = (got == "accepted") == want
        failed += [] if ok else [label]
        print(f"  {label:<{width}}  want={'accept' if want else 'refuse':6} -> {got:<34} {'ok' if ok else 'MISMATCH'}")
        if want:
            json.loads(parse_tool_calls("<tool_call>" + text)[0][1])   # and the parser reads it back
    print(("FAIL: " + ", ".join(failed)) if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
