#!/usr/bin/env python3
"""The tool conversations as the door would render them, for a raw /v1/completions prefill (2026-09-19, window 4).

The Qwen door answers any chat request carrying `tools` with 400 -- it arms the tool-call grammar and this boot binds no
grammar compiler -- so the tool conversations are rendered here with the door's own functions (serve.template_messages,
qwen38.boot.chat_renderer: the checkpoint's template, tools, thinking, the last turn continued) and prefilled as text.

    render_tools.py CKPT_META OUT.jsonl CONVOS.jsonl [CONVOS.jsonl ...]     (inside st-engine:qwen38, the tree at /repo)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/repo")
from engine.base.serve import template_messages                     # noqa: E402
from engine.profiles.qwen38.boot import chat_renderer                 # noqa: E402

render = chat_renderer(Path(sys.argv[1]))
n = 0
with open(sys.argv[2], "w") as out:
    for path in sys.argv[3:]:
        for line in open(path):
            r = json.loads(line)
            if not r.get("tools"):
                continue
            text = render(template_messages(r["messages"]), {"enable_thinking": bool(r.get("thinking")), "tools": r["tools"]},
                          generation_prompt=False, continue_final=True)
            out.write(json.dumps({"cid": r["cid"], "src": Path(path).name, "text": text}, ensure_ascii=False) + "\n")
            n += 1
print(json.dumps({"rendered": n}))
