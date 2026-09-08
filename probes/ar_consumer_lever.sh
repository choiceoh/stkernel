#!/usr/bin/env bash
# Exact same source on both arms; retain normal onepass and SSE quality evidence.
set -euo pipefail
name=${1:?}; knobs=${2:-}; out=${AR_CONSUMER_OUT:?}
mode=0
for pair in $knobs; do
  [[ $pair != VLLM_GLM53_AR_CONSUMER_PDL=* ]] || mode=${pair#*=}
done
[[ $mode == 0 || $mode == 1 ]] || exit 2
expected=$(python3 - "$REPO" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); wanted={'glm53_megakernel.cu','glm53_megakernel.py','dsv4_oneshot_ar.cu','dsv4_oneshot_shim.py'}
files={}
for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
    if not line or line.startswith('#'):continue
    source,target,*_=line.split('\t')
    if source in wanted:files[target]=hashlib.sha256((root/'build/glm53'/source).read_bytes()).hexdigest()
assert len(files)==4,files
print(json.dumps(files))
PY
)
collect() {
  local prefix=$1 status=0
  curl -fsS --max-time 5 http://127.0.0.1:18000/health >/dev/null || return 1
  python3 "$REPO/probes/ar_consumer_runtime_proof.py" "$mode" "$expected" > "$out/$prefix-srv2.json" || status=1
  for node in 1 3 4; do
    # Quote the JSON for the remote shell as one argument.
    printf -v command 'python3 - %q %q' "$mode" "$expected"
    ssh -o BatchMode=yes "choiceoh@10.10.10.$node" "$command" \
      < "$REPO/probes/ar_consumer_runtime_proof.py" > "$out/$prefix-srv$node.json" || status=1
  done
  [[ $(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://10.10.10.2:18000/health || true) == 000 ]] || status=1
  return "$status"
}
rc=0
LEGS=none bash "$REPO/bench/ab-lever.sh" "$name" "$knobs" > "$out/prepare-$name.log" 2>&1 || rc=$?
tail -n 20 "$out/prepare-$name.log"
collect "prepared-$name" || rc=1
if [[ $rc == 0 && ${LEGS:-onepass} != none ]]; then
  export SPEC_K=5 BENCH_MODEL=glm-5.3-flash
  export INPUT_REUSE_CHANNELS_OUT=$out/channels-$name.jsonl MK_COLD_COMPILE=0
  if rg -q 'first boot on build' "$out/prepare-$name.log" 2>/dev/null; then export MK_COLD_COMPILE=1
  elif ! command -v rg >/dev/null && grep -q 'first boot on build' "$out/prepare-$name.log"; then export MK_COLD_COMPILE=1; fi
  # Retain host RAM pressure even if earlyoom terminates the engine and the
  # after-traffic runtime proof is unavailable. This observer is read-only,
  # bounded, and reaped before the next arm; it never changes memory policy.
  timeout 920 vmstat -w -t 2 > "$out/host-memory-$name.log" 2>&1 &
  memory_pid=$!
  timeout 900 python3 "$REPO/probes/input_reuse_channels.py" --name "$name" || rc=$?
  kill "$memory_pid" 2>/dev/null || true
  wait "$memory_pid" 2>/dev/null || true
fi
collect "runtime-$name" || rc=1
cp /home/choiceoh/glm53-logs/glm53.log "$out/boot-$name.log"
exit "$rc"
