#!/usr/bin/env python3
"""The v3 self-distribution conversations as raw text for a /v1/completions prefill (next window): the door's own
rendering (serve.template_messages, qwen38.boot.chat_renderer: the checkpoint's template, tools, thinking) with the
served model's answer closed -- its thinking, text and tool calls, then <|im_end|> -- so the tap holds every position
of the answer and mtp_tune data --answer-after 248045,74455,198 keeps exactly those. A continued final message would
stop at the answer's text and drop a tool call that follows it (438 of Deneb's 986 answers end in one).

    render_all.py CKPT_META OUT.jsonl CONVOS.jsonl        (inside st-engine:qwen38, the tree at /repo)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/repo")
from engine.base.serve import template_messages                     # noqa: E402
from engine.profiles.qwen38.boot import chat_renderer                 # noqa: E402

render = chat_renderer(Path(sys.argv[1]))
n, chars, missing_marker = 0, 0, 0
with open(sys.argv[2], "w") as out:
    for line in open(sys.argv[3]):
        r = json.loads(line)
        options = {"enable_thinking": bool(r.get("thinking"))}
        if r.get("tools"):
            options["tools"] = r["tools"]
        text = render(template_messages(r["messages"]), options, generation_prompt=False, continue_final=False)
        if "<|im_start|>assistant\n" not in text:
            missing_marker += 1
        out.write(json.dumps({"cid": r["cid"], "src": r["src"], "text": text}, ensure_ascii=False) + "\n")
        n += 1
        chars += len(text)
print(json.dumps({"rendered": n, "chars": chars, "missing_assistant_marker": missing_marker}))
