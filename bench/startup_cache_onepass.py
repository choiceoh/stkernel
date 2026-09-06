#!/usr/bin/env python3
"""Run the existing onepass workload while retaining its exact response evidence."""
import hashlib
import json
import os
from pathlib import Path

import onepass


def main():
    path = Path(os.environ["STARTUP_CACHE_RESPONSES"])
    original = onepass.ask_stream

    def recorded(url, model, content, max_tokens):
        result = original(url, model, content, max_tokens)
        text, ttft, prompt_tokens, completion_tokens, finish = result
        record = {"prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
                  "response": text, "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "ttft_s": ttft, "prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens, "finish_reason": finish}
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result

    onepass.ask_stream = recorded
    return onepass.main()


if __name__ == "__main__":
    raise SystemExit(main())
