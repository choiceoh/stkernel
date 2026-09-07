#!/usr/bin/env python3
"""Identical first text/image/video requests before the startup onepass workload."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import vision_probe as vision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    video = (ROOT / "tests/fixtures/startup_red_blue.mp4").read_bytes()
    image = vision.png(224)
    cases = [
        ("text", "1부터 5까지 숫자만 쉼표로 구분해서 적어줘."),
        ("image", [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}},
            {"type": "text", "text": "그림 속 도형 두 개의 색과 위치를 한 문장으로 설명해줘."}]),
        ("video", [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(video).decode()}},
            {"type": "text", "text": "영상 화면의 색이 처음과 끝에 어떻게 바뀌는지 한 문장으로 설명해줘."}]),
    ]
    result = {"video_sha256": hashlib.sha256(video).hexdigest(),
              "image_sha256": hashlib.sha256(image).hexdigest(), "requests": []}
    try:
        for label, content in cases:
            started = time.time()
            text, ttft, total, reasoning = vision.ask(vision.model_name(), content, 120, 180)
            if label == "text":
                ok = all(str(i) in text for i in range(1, 6))
            else:
                ok = any(word in text for word in ("빨", "붉", "적색")) and any(word in text for word in ("파", "푸", "청색"))
            ok = ok and ttft is not None and not reasoning and "\ufffd" not in text
            row = dict(kind=label, started=started, ttft_s=ttft, total_s=total,
                       content=text, reasoning=reasoning, ok=ok)
            result["requests"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            assert ok, f"{label}: first-request output check failed"
        result["ok"] = True
    finally:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
