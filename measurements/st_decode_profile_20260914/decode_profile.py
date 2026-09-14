"""srv2: where a decode step's device time goes on the booted door (:8001).

Long greedy decodes keep the engine in steady bursts; during each, POST /v1/engine/profile profiles the next 32 decode
steps (bursts) on every rank and GET returns rank 0's kernel table. Around each profile window, and around an
unprofiled window of the same length, /metrics gives the host-seen step seconds and the drafted/accepted counters, so
the table can be put per iteration and the profiler's own cost is visible.

    python3 /tmp/decode_profile.py --out ~/expert-capture/decode-profile-<label>.json --label <label>
"""
import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8001"
PROMPT = ("Write a very long, detailed technical essay (at least 5000 words) on the history of numerical linear "
          "algebra, from Gaussian elimination to modern GPU kernels. Use many sections and subsections, explain each "
          "algorithm step by step, and do not stop early.")
LANES = (("one-shot collective", r"oneshot|osar|k_oneshot"),
         ("NCCL collective", r"nccl|all_reduce|allgather|reduce_scatter"),
         ("mHC", r"mhc"),
         ("MLA / DSA", r"mla|sparse|logits|kpool|indexer|topk|deep_gemm|mqa"),
         ("KDA", r"kda|conv1d|fused_recurrent|recurrent|gdn"),
         ("MoE", r"moe|b12x|expert|cute|sf6"),
         ("dense GEMM", r"gemm|cutlass|nvjet|sm90|sm100|sm121|megakernel|mk_"),
         ("memcpy", r"memcpy|memset|dtoh|htod|pinned"),
         ("norm / elementwise", r"norm|elementwise|vectorized|copy|cat|fill|index|gather|scatter|softmax|argmax|reduce"))


def http(path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and path == "/v1/engine/profile" and body is None:
            return None                                  # no profile has run yet on this boot
        raise
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode()


def metrics():
    text = http("/metrics")
    out = {}
    for line in text.splitlines():
        m = re.match(r'^(st:step_seconds_(?:sum|count))\{engine="st",kind="decode"\} (\S+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
        for name in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                     "st:steps_decode_total"):
            if line.startswith(name + "{") or line.startswith(name + " "):
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def delta(a, b):
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b}
    count = d.get("st:step_seconds_count", 0.0)
    drafted = d.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    d["burst_ms"] = 1000 * d.get("st:step_seconds_sum", 0.0) / count if count else None
    d["iterations"] = drafted / 7 if drafted else None
    d["iteration_ms"] = 1000 * d.get("st:step_seconds_sum", 0.0) / d["iterations"] if d["iterations"] else None
    d["acceptance"] = d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) / drafted if drafted else None
    return d


def lane_of(name):
    low = name.lower()
    for lane, pattern in LANES:
        if re.search(pattern, low):
            return lane
    return "other"


def decode(n, stop):
    """n concurrent long greedy requests; each restarts until `stop` is set, so decode never pauses."""
    def one():
        while not stop.is_set():
            body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 6000,
                    "temperature": 0, "stream": False, "chat_template_kwargs": {"thinking": False}}
            try:
                http("/v1/chat/completions", body, timeout=900)
            except Exception as exc:  # noqa: BLE001 -- keep the load up; the table says what ran
                print("request error:", exc, flush=True)
                time.sleep(1)
    threads = [threading.Thread(target=one, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    return threads


def window(profile, seconds_hint):
    before = metrics()
    if profile:
        previous = http("/v1/engine/profile")
        http("/v1/engine/profile", {"steps": 32})
        deadline = time.time() + 120
        table = previous
        while time.time() < deadline:
            time.sleep(2)
            table = http("/v1/engine/profile")
            if isinstance(table, dict) and table.get("kernels") and table != previous:
                break
    else:
        time.sleep(seconds_hint)
        table = None
    after = metrics()
    return table, delta(before, after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    http("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}],
                                  "max_tokens": 16, "chat_template_kwargs": {"thinking": False}}, timeout=300)
    result = {"label": a.label, "runs": []}
    for concurrency, repeats in ((1, 3), (4, 2)):
        stop = threading.Event()
        threads = decode(concurrency, stop)
        time.sleep(15)                                   # past prefill, into steady bursts
        for r in range(repeats):
            table, prof = window(True, 0)
            _, plain = window(False, 8)
            lanes = {}
            for row in (table or {}).get("kernels", []):
                lanes[lane_of(row["kernel"])] = lanes.get(lane_of(row["kernel"]), 0.0) + row["us_per_step"]
            run = {"concurrency": concurrency, "repeat": r, "profiled": prof, "unprofiled": plain,
                   "device_us_per_step": (table or {}).get("device_us_per_step"), "lanes_us_per_step": lanes,
                   "kernels": (table or {}).get("kernels", [])}
            result["runs"].append(run)
            iters = (prof.get("iterations") or 0) / max(1.0, prof.get("st:step_seconds_count") or 1.0)
            print(f"C={concurrency} #{r}: device {run['device_us_per_step']} us/burst, bursts {prof.get('st:step_seconds_count')}, "
                  f"iter/burst {iters:.2f}, burst ms profiled {prof.get('burst_ms')} vs plain {plain.get('burst_ms')}, "
                  f"iteration ms profiled {prof.get('iteration_ms')} vs plain {plain.get('iteration_ms')}, "
                  f"accept {prof.get('acceptance')}", flush=True)
            print("   lanes us/burst: " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(lanes.items(), key=lambda kv: -kv[1])), flush=True)
            with open(a.out, "w") as f:
                json.dump(result, f, indent=1)
        stop.set()
        for t in threads:
            t.join(timeout=900)
    print("PROFILE done", flush=True)


if __name__ == "__main__":
    main()
