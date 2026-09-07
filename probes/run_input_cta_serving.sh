#!/usr/bin/env bash
# Exact eight-slice CTA probe; same-build controls and supervised production handoff.
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export REPO=$PWD
RESTORE_REPO=${FLEET_PRODUCTION_REPO:-/home/choiceoh/stkernel}
IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
export INPUT_CTA_SERVING_OUT=${INPUT_CTA_SERVING_OUT:-/home/choiceoh/glm53-logs/INPUTCTASERVE20907}
out=$INPUT_CTA_SERVING_OUT
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh LEVER=$REPO/probes/input_cta_lever.sh
session=${FLEET_SESSION:?}
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == boot ]] || exit 2
[[ -z $(git status --porcelain) && -z $(git -C "$RESTORE_REPO" status --porcelain) ]] || exit 2
git fetch origin
git merge-base --is-ancestor origin/main HEAD || { echo 'ABORT before stopping service: candidate needs current main'; exit 2; }
[[ ! -e $out/source.commit ]] || { echo 'ABORT: fresh evidence required'; exit 2; }
mkdir -p "$out/build"
git rev-parse HEAD > "$out/source.commit"
python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_entry.py" idle "$out/before-metrics.txt"

touched=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  docker stop -t 2 "inputcta-$session" >/dev/null 2>&1 || true
  if [[ $touched == 1 && ${FLEET_RESTORE_MANAGED:-0} != 1 ]]; then
    if (
      cd "$RESTORE_REPO"
      git fetch origin || exit 1
      git merge --ff-only origin/main || exit 1
      bash launchers/deploy-overlays.sh glm53 || exit 1
      env -u ONEPASS_JSONL -u ONEPASS_VERDICTS REPO="$RESTORE_REPO" \
        GLM53_API_HOST=0.0.0.0 GLM53_API_PORT=8000 HEAD=10.10.10.2 \
        PREFILL_WARMUP=1 LEGS=none bash bench/ab-lever.sh "${session}RESTORE" ''
    ) > "$out/restore.log" 2>&1; then
      echo restored > "$out/restore.status"
    else
      echo FAILED > "$out/restore.status"; rc=1
    fi
  fi
  echo "$rc" > "$out/runner.exit"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker image inspect "$IMAGE" >/dev/null
if docker inspect glm53 >/dev/null 2>&1; then
  [[ $(docker inspect glm53 --format '{{.Image}}') == "$IMAGE" ]] || exit 2
fi
touched=1
if docker inspect glm53 >/dev/null 2>&1; then docker stop -t 30 glm53 >/dev/null; fi
pids=()
for node in 1 3 4; do
  ssh -o BatchMode=yes "choiceoh@10.10.10.$node" 'if docker inspect glm53-worker >/dev/null 2>&1; then docker stop -t 30 glm53-worker; else docker info >/dev/null; fi' >/dev/null &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
(( available_kib >= 16 * 1024 * 1024 )) || { echo 'ABORT: less than 16 GiB available'; exit 1; }
args=(run --rm --name "inputcta-$session" --gpus device=0 --network=none
      --cpuset-cpus=14-17 --memory=10g --shm-size=1g
      --mount "type=bind,src=$REPO,dst=/repo,readonly"
      --mount "type=bind,src=$out,dst=/evidence"
      --mount "type=bind,src=$out/build,dst=/build"
      --mount 'type=bind,src=/usr/local/cuda/compute-sanitizer,dst=/san,readonly'
      --workdir /repo)
timeout 420 docker "${args[@]}" --entrypoint python3 "$IMAGE" \
  /repo/probes/gemm_input_cta.py --production > "$out/probe.log" 2>&1
for tool in racecheck memcheck; do
  timeout 240 docker "${args[@]}" --entrypoint /san/compute-sanitizer "$IMAGE" \
    --tool "$tool" --target-processes application-only --error-exitcode 77 \
    --kernel-name kns=mk_gemm_input_cta_kernel \
    python3 /repo/probes/gemm_input_cta.py --production --check-only \
    --out "/evidence/$tool.json" > "$out/$tool.log" 2>&1
done

mode=$(python3 - "$out" <<'PYSEL'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);p=json.loads((out/'result.json').read_text())
assert p['status']=='PASS'
assert all(json.loads((out/f'{tool}.json').read_text())['status']=='PASS' for tool in ('racecheck','memcheck'))
t={r['cache']:r for r in p['timings']}
winners=[]
for mode in ('1','2','3'):
    if all(r['reduction_pct'][mode]>=1 and sum(a<b for a,b in zip(r['raw_us'][mode],r['raw_us']['0']))>=29 for r in t.values()):
        winners.append(mode)
assert winners,'no stable winner over enabled input-reuse default; stop before serving'
mode=min(winners,key=lambda m:t['warm']['median_us'][m])
(out/'selection.json').write_text(json.dumps({'mode':int(mode),'eligible_modes':winners,
    'rule':'at least 1 percent lower median and 29/32 faster pairs in both cache regimes; lowest warm median',
    'source_sha256':p['source_sha256']},indent=2)+'\n')
print(mode)
PYSEL
)
bash launchers/deploy-overlays.sh glm53 > "$out/deploy.log" 2>&1
# A peer died during the first baseline's initial MHC call. Exercise that
# initialization on every node in a fresh process before loading the model.
sha=$(sha256sum overlay/modules/glm53_megakernel/glm53_megakernel.cu | cut -d ' ' -f 1)
pids=()
python3 probes/input_cta_node_check.py --session "$session" --source-sha256 "$sha" \
  > "$out/node-check-srv2.log" 2>&1 &
pids+=($!)
for node in 1 3 4; do
  ssh -o BatchMode=yes "choiceoh@10.10.10.$node" python3 - \
    --session "$session" --source-sha256 "$sha" < probes/input_cta_node_check.py \
    > "$out/node-check-srv$node.log" 2>&1 &
  pids+=($!)
done
node_rc=0
for pid in "${pids[@]}"; do wait "$pid" || node_rc=1; done
[[ $node_rc == 0 ]] || { echo 'ABORT: independent node startup check failed'; exit 1; }
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=18000 HEAD=127.0.0.1
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=3 ONEPASS_REQUIRE_EXCLUSIVE=1
export ONEPASS_JSONL=$out/records.raw.jsonl ONEPASS_VERDICTS=$out/verdicts.jsonl
bash bench/chain.sh \
  'ICTAB1=' \
  "ICTAA1=VLLM_GLM53_MK_INPUT_CTA=$mode" \
  "ICTAA2=VLLM_GLM53_MK_INPUT_CTA=$mode" \
  'ICTAB2='
