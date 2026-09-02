#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run on the head with the engine up: profiles one fresh 32K prefill and runs census.py on it.
# on srv2: profile one fresh 32K prefill, then census it
set -uo pipefail
cd ~/stkernel
before=$(ls ~/vllm-prof/*rank0*.pt.trace.json.gz 2>/dev/null | wc -l)
python3 - <<'PY' &
import json, random, time, urllib.request
WORDS = ["reactor","harbor","lattice","quarry","ember","meridian","syntax","granite","voltage","cirrus","tundra","beacon","ledger","prism","cobalt","willow","cascade","anvil","nocturne","vellum"]
rng = random.Random(4242)
text = " ".join(rng.choice(WORDS) for _ in range(int(32000 / 1.3)))
time.sleep(3)
body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": text + "\n\nSummarize the above in one sentence."}],
                   "max_tokens": 8, "temperature": 0.0, "chat_template_kwargs": {"thinking": False}}).encode()
t0 = time.time()
with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900) as r:
    out = json.load(r)
print(f"prefill request: prompt_tokens={out['usage']['prompt_tokens']} wall={time.time()-t0:.1f}s", flush=True)
PY
req=$!
curl -s -X POST localhost:8000/start_profile >/dev/null; echo "profiler started"
wait $req
sleep 2; curl -s -X POST localhost:8000/stop_profile >/dev/null; echo "profiler stopped"
for i in $(seq 1 30); do now=$(ls ~/vllm-prof/*rank0*.pt.trace.json.gz 2>/dev/null | wc -l); [ "$now" -gt "$before" ] && break; sleep 5; done
f=$(ls -t ~/vllm-prof/*rank0*.pt.trace.json.gz | head -1); echo "trace: $f ($(du -m $f | cut -f1) MB)"
python3 census.py "$f" 2>&1 | tail -60
