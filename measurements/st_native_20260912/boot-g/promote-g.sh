#!/usr/bin/env bash
# Run on srv2 only after E's exclusive run has restored the old service.
# Preserve the old image, source, tier, env and systemd drop-in for rollback.
set -euo pipefail
QUAL=/home/choiceoh/st-native-f4d7-20260912g
CANDIDATE=/home/choiceoh/st-releases/perf-f4d7-20260912g
BASELINE=/home/choiceoh/st-releases/5734b29fde84
RUN=/home/choiceoh/st-native-promotion-20260912g
ENV=/home/choiceoh/.config/st-glm53.env
DROPIN=/home/choiceoh/.config/systemd/user/st-glm53.service.d/release.conf
IMAGE=st-engine:prod-0abb87ae741e
SHA=0abb87ae741ee2843dc54ef16479c404478f28ff249cee346856eeb1f3b7a331
python3 - "$QUAL" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
assert (p/'exclusive-exit-code.txt').read_text().strip()=='0'
assert (p/'onepass-exit-code.txt').read_text().strip()=='0'
r=json.loads((p/'result.jsonl').read_text().splitlines()[-1])
assert r['quality']=={'ok':9,'total':9},r['quality']
assert r['korean']['dirty']==0 and not r['traffic']['issues']
assert len(json.loads((p/'api-smoke.json').read_text()))==3
assert 'API_SMOKE_PASS 3/3' in (p/'run.log').read_text()
for rank in range(4):
 m=json.loads((p/f'candidate/memory-rank{rank}.json').read_text())
 assert m['ready'] and m['workspace_limit_bytes']==12<<30 and m['os_reserve_bytes']==4<<30
 assert m['phases'][-1]['phase']=='production/ready' and all(x['passed'] for x in m['phases'])
 assert all(x['peak_workspace_bytes']<=12<<30 and x['immediately_free_bytes']>=4<<30 for x in m['phases'])
print('Exclusive serving, API and memory gates passed',flush=True)
PY
mkdir "$RUN"
cd "$RUN"
cp "$ENV" baseline.env
cp "$DROPIN" baseline-release.conf
node_sh() {
  local ip=$1; shift
  if [ "$ip" = 10.10.10.2 ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$ip" "$@"; fi
}
wait_door() {
  local i
  for i in $(seq 1 120); do
    if curl -fsS --max-time 4 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then return 0; fi
    [ "$(docker inspect --format '{{.State.Running}}' st-glm53 2>/dev/null)" = true ] || return 1
    sleep 10
  done
  return 1
}
drain_rule=(-p tcp --dport 8000 '!' -i lo -m conntrack --ctstate NEW -m comment --comment st-native-g-promotion -j REJECT --reject-with tcp-reset)
changed=0
closed=0
adopted=0
finish() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [ "$changed" = 1 ] && [ "$adopted" = 0 ]; then
    bash "$CANDIDATE/launchers/start-st-glm53.sh" stop >> rollback.log 2>&1
    cp baseline.env "$ENV"
    cp baseline-release.conf "$DROPIN"
    systemctl --user daemon-reload
    unset ST_KV_GIB ST_TIER_DIR ST_DUMP_DIR
    set -a; source "$ENV"; set +a
    for ip in 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4; do
      node_sh "$ip" "PYTHONPATH=$CANDIDATE python3 -c 'from engine.base.arena import release_model_cache; print(release_model_cache([\"/home/choiceoh/models\"]))'" >> rollback.log 2>&1
    done
    node_sh 10.10.10.4 "python3 /home/choiceoh/st-perf-f4d7/st-restore-cache-return.py" >> rollback.log 2>&1 &
    bash "$BASELINE/launchers/start-st-glm53.sh" >> rollback.log 2>&1
    wait_door >> rollback.log 2>&1 || rc=1
  fi
  systemctl --user start st-glm53.service
  if [ "$closed" = 1 ]; then sudo -n iptables -D INPUT "${drain_rule[@]}"; fi
  sudo -n systemctl stop st-native-g-promotion-cleanup.timer
  printf '%s\n' "$rc" > exit-code.txt
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
systemctl --user stop st-glm53.service
sudo -n iptables -I INPUT 1 "${drain_rule[@]}"
closed=1
sudo -n systemd-run --quiet --unit=st-native-g-promotion-cleanup --on-active=60m \
  /usr/sbin/iptables -D INPUT "${drain_rule[@]}"
python3 - <<'PY'
import json,time,urllib.request
for _ in range(600):
 with urllib.request.urlopen('http://127.0.0.1:8000/',timeout=5) as response: state=json.load(response)
 if not any(state.get(k) for k in ('running','waiting','queued','parking','resuming')): break
 time.sleep(1)
else: raise SystemExit('production remains active; deployment did not start')
PY
for ip in 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4; do
  node_sh "$ip" "docker tag st-engine:perf-f4d7-g $IMAGE"
done
changed=1
bash "$BASELINE/launchers/start-st-glm53.sh" stop
cat > "$ENV.new" <<EOF
ST_REPO=$CANDIDATE
ST_ENGINE_DIR=$CANDIDATE
ST_IMAGE=$IMAGE
ST_PRODUCTION=1
ST_KV_GIB=16
RANKS_DIR=/home/choiceoh/models/st-glm53-9391-up-gate-full
CKPT=$CANDIDATE/st-glm53-meta
DRAFTER=/home/choiceoh/models/GLM-5.3-Flash-DFlash2
CACHE_DIR=/home/choiceoh/glm53-cache
ST_TIER_DIR=/home/choiceoh/glm53-logs/st-tier-native-0abb87ae741e
ST_DUMP_DIR=/home/choiceoh/glm53-logs/st-dumps-native-0abb87ae741e
PORT=8000
EOF
chmod --reference="$ENV" "$ENV.new"
mv "$ENV.new" "$ENV"
cat > "$DROPIN.new" <<EOF
[Service]
ExecStart=
ExecStart=$CANDIDATE/launchers/st-glm53-supervisor.sh
EOF
mv "$DROPIN.new" "$DROPIN"
systemctl --user daemon-reload
set -a; source "$ENV"; set +a
bash "$CANDIDATE/launchers/start-st-glm53.sh" > launch.log 2>&1
wait_door
r=0
for ip in 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4; do
  node_sh "$ip" 'docker inspect st-glm53' > "container-rank$r.json"
  node_sh "$ip" 'docker exec -w /repo -e PYTHONPATH=/repo st-glm53 python3 -m engine.runtime.verify --gpu' > "manifest-rank$r.json"
  node_sh "$ip" "cat $ST_DUMP_DIR/memory-rank$r.json" > "memory-rank$r.json"
  node_sh "$ip" 'docker logs st-glm53' > "rank$r.log" 2>&1
  r=$((r+1))
done
python3 - "$SHA" "$IMAGE" <<'PY'
import json,sys
from pathlib import Path
for rank in range(4):
 m=json.loads(Path(f'manifest-rank{rank}.json').read_text())
 c=json.loads(Path(f'container-rank{rank}.json').read_text())[0]
 b=json.loads(Path(f'memory-rank{rank}.json').read_text())
 assert m['passed'] and m['engine_source_sha256']==sys.argv[1] and not m['vllm_present']
 assert c['Config']['Image']==sys.argv[2] and c['State']['Running']
 assert not any(e.startswith('STK_') for e in c['Config']['Env'])
 assert b['ready'] and b['phases'][-1]['phase']=='production/ready' and all(p['passed'] for p in b['phases'])
print('All four deployed source, image and memory identities agree',flush=True)
PY
curl -fsS --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"대한민국의 수도 이름만 답하세요."}],"max_tokens":48,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' > chat.json
python3 - <<'PY'
import json
r=json.load(open('chat.json'))
assert '서울' in r['choices'][0]['message']['content'] and r['choices'][0]['finish_reason']=='stop',r
print(r,flush=True)
PY
systemctl --user start st-glm53.service
systemctl --user is-active --quiet st-glm53.service
curl -fsS --max-time 5 http://127.0.0.1:8000/ > status.json
curl -fsS --max-time 5 http://127.0.0.1:8000/metrics > metrics.txt
cp "$ENV" deployed.env
cp "$DROPIN" deployed-release.conf
adopted=1
printf '%s\n' "$SHA" > ADOPTED
echo "ST native release adopted on production port 8000"
