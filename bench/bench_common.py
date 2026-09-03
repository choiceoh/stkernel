"""Shared harness bits. Today: which model name to send.

Every bench in here hardcoded `deepseek-v4-flash` as the default. Pointed at
the glm53 server that 404s, and a 404 does not read like a failure in these
scripts -- the caller greps for a SUMMARY line, finds none, and the section
reads "measured nothing" rather than "never ran". It silently voided the 9/9
quality gate on this lane for an unknown number of boots.

Ask the server instead. `BENCH_MODEL` still wins when a caller means a
specific name, and the literal default is the last resort.
"""
import json
import os
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"


def resolve_model(default: str = "deepseek-v4-flash",
                  url: str = DEFAULT_URL) -> str:
    """The served name, asked of the server, not assumed."""
    named = os.environ.get("BENCH_MODEL")
    if named:
        return named
    try:
        base = url.split("/v1/", 1)[0]
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode())["data"]
        if len(data) == 1:
            return data[0]["id"]
        for entry in data:
            if entry["id"] == default:
                return default
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return default
