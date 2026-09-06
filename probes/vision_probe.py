#!/usr/bin/env python3
"""Vision smoke probe for GLM-5.3-Flash serving (39차, operator "비전 행도 잡아봐").

Builds a small PNG in pure Python (no PIL): a red square on white with a
blue bar, sends it as a data-URL image with a short question, thinking off,
and reports TTFT / total time / the answer -- or HANG when the request does
not finish inside --timeout seconds (the serving side is then left for the
stall watcher's py-spy snapshot). A second text-only request afterwards
shows whether the engine still serves after the image request.

    python3 probes/vision_probe.py [--timeout 180] [--size 224]
"""
import argparse
import base64
import importlib.util
import json
import os
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "bench")
BASE = os.environ.get("GLM53_BASE", "http://127.0.0.1:8000")


def png(size: int) -> bytes:
    """RGB PNG: white background, red square top-left, blue bar bottom."""
    rows = []
    for y in range(size):
        row = bytearray([0])  # filter type 0
        for x in range(size):
            if y >= size * 3 // 4:
                row += bytes((30, 60, 220))
            elif x < size // 2 and y < size // 2:
                row += bytes((220, 30, 30))
            else:
                row += bytes((255, 255, 255))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def model_name() -> str:
    spec = importlib.util.spec_from_file_location("vp_quality", os.path.join(BENCH, "check-quality.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MODEL


def ask(model: str, content, max_tokens: int, timeout: float):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                       "messages": [{"role": "user", "content": content}],
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                obj = json.loads(line[5:].strip())
            except ValueError:
                continue
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(piece)
            if time.time() - t0 > timeout:
                raise socket.timeout("stream exceeded the budget")
    return "".join(parts), ttft, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()
    model = model_name()
    data_url = "data:image/png;base64," + base64.b64encode(png(args.size)).decode()
    content = [{"type": "image_url", "image_url": {"url": data_url}},
               {"type": "text", "text": "이 이미지에 어떤 색의 도형이 어디에 있는지 한 문장으로 설명해줘."}]
    rc = 0
    for label, payload in (("image", content), ("text-after", "1부터 5까지 세어줘.")):
        try:
            text, ttft, total = ask(model, payload, args.max_tokens, args.timeout)
            print(f"{label:>10}: OK  ttft={ttft if ttft is not None else float('nan'):.2f}s total={total:.1f}s -> {text[:160]!r}", flush=True)
        except (socket.timeout, TimeoutError) as e:
            print(f"{label:>10}: HANG (no completion within {args.timeout:.0f}s: {e})", flush=True)
            rc = 2
        except urllib.error.HTTPError as e:
            print(f"{label:>10}: HTTP {e.code} {e.read()[:300]!r}", flush=True)
            rc = 1
        except Exception as e:  # noqa: BLE001
            print(f"{label:>10}: ERROR {e!r}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
