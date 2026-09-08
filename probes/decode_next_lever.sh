#!/usr/bin/env bash
# One boot and one unchanged onepass, with four-rank identity and memory proof.
set -euo pipefail
name=${1:?}; knobs=${2:-}; out=${DECODE_NEXT_OUT:?}
trap 'rc=$?; printf "%s\n" "$rc" > "$out/arm-$name.exit"; exit "$rc"' EXIT
candidate='VLLM_GLM53_AR_COMPACT_CTA=1 VLLM_GLM53_AR_PROXY_INLINE=1 VLLM_GLM53_B12X_STATIC_V2=t,r,sf6'
if [[ $knobs == "$candidate" ]]; then mode=candidate
elif [[ -z $knobs ]]; then mode=baseline
else echo 'exact all-three candidate or profile baseline required'; exit 2; fi
# Keep chain's baseline argument empty while making the launched baseline
# independent of inherited target overrides. Candidate argv overrides these.
export VLLM_GLM53_AR_COMPACT_CTA=0 VLLM_GLM53_AR_PROXY_INLINE=0 VLLM_GLM53_B12X_STATIC_V2=t,r
export VLLM_GLM53_AR_CONSUMER_PDL=1 VLLM_GLM53_MK_PDL=1
expected=$(python3 - "$REPO" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
    if not line or line.startswith('#'):continue
    source,target,*_=line.split('\t')
    assert target not in files, target
    files[target]=hashlib.sha256((root/'build/glm53'/source).read_bytes()).hexdigest()
assert len(files)>50, 'complete overlay manifest required'
print(json.dumps(files,sort_keys=True))
PY
)
printf '%s\n' "$expected" > "$out/expected-$name.json"
collect() {
  local prefix=$1 status=0 command node
  curl -fsS --max-time 5 http://127.0.0.1:18000/health >/dev/null || return 1
  python3 "$REPO/probes/decode_next_runtime_proof.py" "$mode" "$expected" > "$out/$prefix-srv2.json" || status=1
  for node in 1 3 4; do
    printf -v command 'python3 - %q %q' "$mode" "$expected"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@10.10.10.$node" "$command" \
      < "$REPO/probes/decode_next_runtime_proof.py" > "$out/$prefix-srv$node.json" || status=1
  done
  [[ $(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://10.10.10.2:18000/health || true) == 000 ]] || status=1
  return "$status"
}
rc=0
LEGS=none bash "$REPO/bench/ab-lever.sh" "$name" "$knobs" > "$out/prepare-$name.log" 2>&1 || rc=$?
tail -n 20 "$out/prepare-$name.log"
collect "prepared-$name" || rc=1
if [[ $rc == 0 ]]; then
  export SPEC_K=5 BENCH_MODEL=glm-5.3-flash
  export INPUT_REUSE_CHANNELS_OUT=$out/channels-$name.jsonl MK_COLD_COMPILE=0
  if python3 - "$out/prepare-$name.log" <<'PY'
from pathlib import Path
import sys
sys.exit(0 if 'first boot on build' in Path(sys.argv[1]).read_text() else 1)
PY
  then export MK_COLD_COMPILE=1; fi
  timeout 900 python3 "$REPO/bench/onepass_memory.py" --report "$out/memory-$name.jsonl" \
    -- python3 "$REPO/probes/input_reuse_channels.py" --name "$name" \
    > "$out/onepass-$name.log" 2>&1 || rc=$?
  tail -n 40 "$out/onepass-$name.log"
fi
collect "runtime-$name" || rc=1
cp /home/choiceoh/glm53-logs/glm53.log "$out/boot-$name-srv2.log" || rc=1
for node in 1 3 4; do
  scp -q -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@10.10.10.$node:glm53-logs/glm53.log" \
    "$out/boot-$name-srv$node.log" || rc=1
done
exit "$rc"
