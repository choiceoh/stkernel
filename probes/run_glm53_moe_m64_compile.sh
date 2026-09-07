#!/usr/bin/env bash
# CPU only: no NVIDIA runtime, no network, no GPU devices, no serving changes.
set -euo pipefail
int8=0
if [[ $# == 1 && $1 == --int8 ]]; then int8=1
elif [[ $# != 0 ]]; then echo 'only --int8 is accepted' >&2; exit 2; fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
revision=$(git -C "$REPO" rev-parse HEAD)
[[ -z $(git -C "$REPO" status --porcelain) ]] || { echo 'clean source required' >&2; exit 3; }
echo "M64 CPU compile source=$revision image=$IMAGE"
args=(docker run --rm --runtime runc --network none --memory 4g --cpus 2
  -v "$REPO:/repo:ro" --entrypoint python3)
while IFS=$'\t' read -r source target _; do
  [[ -z $source || $source == \#* ]] && continue
  args+=(-v "$REPO/build/glm53/$source:$target:ro")
done < "$REPO/build/glm53/manifest.tsv"
args+=("$IMAGE" /repo/probes/glm53_moe_m64_compile.py)
timeout --signal=TERM --kill-after=10s 120s "${args[@]}"
args[${#args[@]}-1]=/repo/probes/glm53_sanitizer_order_canary.py
timeout --signal=TERM --kill-after=10s 120s "${args[@]}" --compile-only
if [[ $int8 == 1 ]]; then
  args[${#args[@]}-1]=/repo/probes/glm53_prefill_int8_check.py
  args+=(--compile-only)
  timeout --signal=TERM --kill-after=10s 120s "${args[@]}"
fi
