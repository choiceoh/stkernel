#!/usr/bin/env bash
set -euo pipefail
cd /home/choiceoh/st-onepass-f4d7-20260912-0746
systemctl --user is-active --quiet st-glm53.service
restore() { systemctl --user start st-glm53.service; }
trap restore EXIT
systemctl --user stop st-glm53.service
python3 - <<'PY'
import json,time,urllib.request
for _ in range(30):
 with urllib.request.urlopen('http://127.0.0.1:8000/',timeout=5) as r:d=json.load(r)
 if not any(d.get(k) for k in ('running','waiting','queued','parking','resuming')): break
 time.sleep(1)
else: raise SystemExit('server not idle; benchmark not started')
print('ST fleet idle; supervisor probes suspended, containers retained',flush=True)
PY
curl -fsS http://127.0.0.1:8000/metrics > metrics-before.txt
set +e
BENCH_MODEL=glm-5.3-flash SPEC_K=5 timeout --signal=TERM --kill-after=15 1200 \
 python3 -u run_onepass_st.py --name ST-PROD-5734-20260912-0746 \
 --ctx 2000,32000,128000 --max-tokens 400 --num-spec 5 --seed 7 \
 --require-exclusive --out /home/choiceoh/st-onepass-f4d7-20260912-0746/result.jsonl
rc=$?
set -e
curl -fsS http://127.0.0.1:8000/metrics > metrics-after.txt || true
printf '%s\n' "$rc" > exit-code.txt
exit "$rc"
