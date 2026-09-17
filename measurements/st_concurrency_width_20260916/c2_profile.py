"""Where does C=2 lose? Profile decode steps under one and two concurrent sequences.

`POST /v1/engine/profile` records the kernels of the next few decode steps and does not reserve the
server (no 409, unlike a latency recording), so this runs beside production. Two long generations
make the C=2 width; one makes C=1. The comparison is the per-kernel cost of an 8-row step against a
16-row one -- the ledger's C=2 decomposition (MoE, collective, mHC) named the suspects but the
dense side is measured to win, so this asks the door which kernels actually grew.
"""
import json, sys, threading, time, urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "glm-5.3-flash"
PROMPT = ("Explain, in detail and at length, how a paged KV cache works, why block size matters, "
          "and how speculative decoding interacts with it. Use several paragraphs.")


def post(path, payload, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def generate(tag):
    try:
        post("/v1/chat/completions", dict(model=MODEL, max_tokens=900, temperature=0.7,
                                          messages=[dict(role="user", content=f"[{tag}] " + PROMPT)]))
    except Exception as exc:
        print("  gen %s: %s" % (tag, str(exc)[:90]), flush=True)


def run(width, steps=24):
    threads = [threading.Thread(target=generate, args=(f"c{width}-{i}",), daemon=True) for i in range(width)]
    for t in threads:
        t.start()
    time.sleep(6)                                   # let the step settle at the target width
    post("/v1/engine/profile", dict(steps=steps))
    time.sleep(12)
    prof = get("/v1/engine/profile")
    for t in threads:
        t.join(timeout=180)
    return prof


def main():
    out = {}
    for width in (1, 2):
        print("== C=%d ==" % width, flush=True)
        out[width] = run(width)
        time.sleep(8)
    json.dump(out, open("/tmp/c2profile.json", "w"))
    print("wrote /tmp/c2profile.json", flush=True)
    for width, prof in out.items():
        keys = sorted(prof.keys()) if isinstance(prof, dict) else type(prof).__name__
        print("C=%s keys: %s" % (width, keys), flush=True)


main()
