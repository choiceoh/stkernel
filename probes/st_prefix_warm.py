"""Warm the ST engine's prefix cache with known prompts (45차 §23 C): every line of a JSONL file is a request body for
`POST /v1/prefix/warm` -- {"messages": [...]} (through the chat template), {"prompt": "..."} or {"ids": [...]} -- with
`--pin` the boundaries stay out of eviction until `/v1/prefix/unpin`.

    python3 probes/st_prefix_warm.py prompts.jsonl [--url http://10.10.10.2:8000] [--pin]
"""
import argparse
import json
import sys
import urllib.request


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--url", default="http://10.10.10.2:8000")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    total = 0
    for line in open(a.file):
        line = line.strip()
        if not line:
            continue
        body = json.loads(line)
        if a.pin:
            body["pin"] = True
        req = urllib.request.Request(a.url + "/v1/prefix/warm", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            out = json.load(r)
        total += 1
        print(f"  {out['tokens']:>7} tokens, boundaries {out['boundaries']}{' (pinned)' if out.get('pinned') else ''}")
    print(f"  warmed {total} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
