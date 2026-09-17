"""Pull four ranks' diagnostic decode traces from the live door, then release it no matter what.

A latency recording RESERVES the server: every other request gets 409 until it ends. So the whole
body sits in try/finally, the finally aborts, and the abort is retried -- a recording left active is
a production outage (2026-09-16: a TERM'd probe held one for two minutes).
"""
import base64, gzip, hashlib, json, sys, time, urllib.request

URL = "http://127.0.0.1:8000"
TOKEN = sys.argv[1] if len(sys.argv) > 1 else "skew" + str(int(time.time()))
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/skew-traces"


def post(path, payload, timeout=120, token=None):
    # A recording reserves the door: its own traffic must carry the token, or it is 409 like anyone else.
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-ST-Latency-Token"] = token
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def control(**payload):
    return post("/v1/engine/latency", payload)


def idle():
    with urllib.request.urlopen(URL + "/", timeout=10) as r:
        s = json.loads(r.read())
    return not s["running"] and not s["waiting"] and not s["queued"], s


import os
os.makedirs(OUT, exist_ok=True)

for _ in range(30):                                   # begin refuses unless serving is idle
    ok, status = idle()
    if ok:
        break
    time.sleep(2)
print("idle:", ok, "| steps", status["steps"], "| served", status["served"], flush=True)
if not ok:
    raise SystemExit("door is busy; not taking a recording beside live traffic")

began = False
try:
    r = control(op="begin", token=TOKEN, diagnostic=True, concurrency=1)
    began = True
    print("begin:", [x.get("status") or x.get("error") for x in r["ranks"]], flush=True)

    body = {"model": "glm-5.3-flash", "max_tokens": 64,
            "messages": [{"role": "user", "content": "바다에 대해 네 문장으로 설명해줘."}]}
    t0 = time.time()
    out = post("/v1/chat/completions", body, timeout=300, token=TOKEN)
    print("generated in %.1fs, %d chars" % (time.time() - t0,
          len(out["choices"][0]["message"]["content"])), flush=True)

    report = control(op="end", token=TOKEN, timeout=300)
    began = False
    ranks = report["ranks"]
    print("end: ranks", [x["rank"] for x in ranks],
          "| traces", [len(x.get("traces", [])) for x in ranks], flush=True)

    for rank in ranks:
        for trace in rank.get("traces", []):
            name = "rank%d-%s" % (rank["rank"], trace["file"])
            path = os.path.join(OUT, name)
            offset, digest, buf = 0, hashlib.sha256(), bytearray()
            while offset < trace["bytes"]:
                reply = control(op="artifact", token=TOKEN, rank=rank["rank"],
                                file=trace["file"], offset=offset)
                block = next(b for b in reply["ranks"] if b["rank"] == rank["rank"])
                data = base64.b64decode(block["base64"], validate=True)
                if not data or block["offset"] != offset:
                    raise RuntimeError("bad artifact chunk")
                buf += data
                digest.update(data)
                offset += len(data)
            if digest.hexdigest() != trace["sha256"]:
                raise RuntimeError("artifact checksum mismatch: " + name)
            open(path, "wb").write(bytes(buf))
            print("  %-42s %8d bytes  phase=%s step=%s" %
                  (name, trace["bytes"], trace.get("phase"), trace.get("step")), flush=True)
finally:
    if began:                                          # never leave the door reserved
        for attempt in range(5):
            try:
                control(op="abort", token=TOKEN)
                print("ABORTED the recording (attempt %d)" % (attempt + 1), flush=True)
                break
            except Exception as exc:
                print("abort failed:", exc, flush=True)
                time.sleep(3)
print("door:", urllib.request.urlopen(URL + "/v1/models", timeout=10).status, flush=True)
