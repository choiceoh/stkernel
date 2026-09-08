#!/usr/bin/env bash
# Compatibility wrapper: one canonical onepass, then passive all-rank proof.
set -euo pipefail
name=${1:?}; knobs=${2:-}; out=${AR_CONSUMER_OUT:?}
# Every comparison arm names its mode so a profile promotion cannot silently
# change an arm while retaining its old runtime-proof expectation.
mode=
for pair in $knobs; do
  [[ $pair != VLLM_GLM53_AR_CONSUMER_PDL=* ]] || mode=${pair#*=}
done
if [[ -z $mode ]]; then
  mode=$(sed -nE 's/^VLLM_GLM53_AR_CONSUMER_PDL=([01])$/\1/p' "$REPO/profiles/glm53.env" | tail -1)
fi
[[ $mode == 0 || $mode == 1 ]] || { echo 'explicit VLLM_GLM53_AR_CONSUMER_PDL=0 or 1 required'; exit 2; }
expected=$(python3 - "$REPO" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); wanted={'glm53_megakernel.cu','glm53_megakernel.py','dsv4_oneshot_ar.cu','dsv4_oneshot_transport.h','dsv4_oneshot_shim.py'}
files={}
for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
    if not line or line.startswith('#'):continue
    source,target,*_=line.split('\t')
    if source in wanted:files[target]=hashlib.sha256((root/'build/glm53'/source).read_bytes()).hexdigest()
assert len(files)==5,files
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
# Source hashes and logs add no model requests. All GPU quality and timing
# evidence comes from the canonical arm; there is no boot-only staging pass.
rc=0
bash "$REPO/bench/ab-lever.sh" "$name" "$knobs" > "$out/onepass-$name.log" 2>&1 || rc=$?
tail -n 20 "$out/onepass-$name.log"
if [[ $rc == 0 ]]; then
  collect "runtime-$name" || rc=1
fi
cp "${LOGD:-/home/choiceoh/glm53-logs}/glm53.log" "$out/boot-$name.log" || rc=1
exit "$rc"
