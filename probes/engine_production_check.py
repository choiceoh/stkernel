"""HTTP cutover acceptance: chat channels, tools, concurrency and chunked prefill.

Only synthetic requests are sent. A Wormhole config may supply its local token;
the token is never included in the report. This probe does not execute tools.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://10.10.10.2:8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--wormhole-config")
    ap.add_argument("--output", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
    if args.wormhole_config:
        token = json.loads(Path(args.wormhole_config).read_text()).get("token")
        if token:
            headers["Authorization"] = "Bearer " + token
    report = dict(base=args.base, model=args.model, started=time.time(), cases=[], passed=False)

    def chat(prompt, *, stream=False, thinking=False, **extra):
        body = dict(model=args.model, messages=[dict(role="user", content=prompt)],
                    max_tokens=128, temperature=0, stream=stream,
                    chat_template_kwargs=dict(thinking=thinking))
        body.update(extra)
        if stream:
            body["stream_options"] = dict(include_usage=True)
        req = urllib.request.Request(args.base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=240) as resp:
            assert resp.status == 200
            if not stream:
                result = json.load(resp)
                message = result["choices"][0]["message"]
                return dict(id=result.get("id"), model=result.get("model"),
                            seconds=round(time.monotonic() - start, 3), content=message.get("content") or "",
                            reasoning=message.get("reasoning_content") or "", tools=message.get("tool_calls", []),
                            usage=result.get("usage"), finish=result["choices"][0].get("finish_reason"))
            content, reasoning, usage, finish, done, first = [], [], {}, None, False, None
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        first = first or time.monotonic() - start
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or "")
                    finish = ch.get("finish_reason") or finish
            assert done and finish, "stream did not finish cleanly"
            return dict(id=chunk.get("id"), model=chunk.get("model"),
                        seconds=round(time.monotonic() - start, 3), ttft=first, content="".join(content),
                        reasoning="".join(reasoning), usage=usage, finish=finish, done=done)

    def record(name, result):
        report["cases"].append(dict(name=name, **result))
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(dict(name=name, **result), ensure_ascii=False), flush=True)

    try:
        r = chat("대한민국의 수도는? 도시 이름만 답하세요.", max_tokens=32)
        record("korean", r)
        assert "서울" in r["content"] and not r["reasoning"]
        r = chat("What is the capital of France? Answer with only the city name.", stream=True, max_tokens=32)
        record("stream-content", r)
        assert "Paris" in r["content"] and not r["reasoning"]
        if not args.quick:
            r = chat("Compute 17 + 25. Explain briefly and give the result.", thinking=True, stream=True, max_tokens=256)
            record("stream-thinking", r)
            assert "42" in r["content"] and r["reasoning"]
            tool = dict(type="function", function=dict(name="get_weather", description="Get the current weather in a city.",
                        parameters=dict(type="object", properties=dict(city=dict(type="string")), required=["city"])))
            r = chat("Use get_weather to check the weather in Seoul. Call the tool before answering.", tools=[tool])
            record("tool-call", r)
            assert r["tools"] and r["tools"][0]["function"]["name"] == "get_weather"
            barrier = threading.Barrier(4)
            prompts = [("Name the capital of Japan, then briefly describe it in two sentences.", "Tokyo"),
                       ("Name the capital of Italy, then briefly describe it in two sentences.", "Rome"),
                       ("Name the capital of Germany, then briefly describe it in two sentences.", "Berlin"),
                       ("Name the capital of Spain, then briefly describe it in two sentences.", "Madrid")]
            def worker(pair):
                barrier.wait()
                return chat(pair[0]), pair[1]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(worker, prompts))
            for index, (r, expected) in enumerate(results):
                record(f"concurrent-{index}", r)
                assert expected in r["content"]
            prompt = "The secret verification number is 7349. Remember it.\n" + (
                "This is unrelated filler about blue skies and green fields.\n" * 850)
            prompt += "\nWhat is the secret verification number at the beginning? Answer only the number."
            r = chat(prompt, max_tokens=32)
            record("chunked-prefill-retrieval", r)
            assert r["usage"]["prompt_tokens"] > 6912 and "7349" in r["content"]
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished"] = time.time()
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
