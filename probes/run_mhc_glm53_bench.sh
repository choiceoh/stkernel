#!/usr/bin/env bash
# Run the GLM53 MHC probe against the exact composed overlay sources.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-glm53:v13-b12x}
BUILD="$REPO/build/glm53"
MANIFEST="$BUILD/manifest.tsv"

bash "$REPO/launchers/compose-overlays.sh" glm53 >&2

mounts=()
for source in tilelang.py tilelang_kernels.py; do
  target=$(awk -F '\t' -v source="$source" '$1 == source {print $2}' "$MANIFEST")
  [ -n "$target" ] \
    || { echo "ABORT: $source is missing from $MANIFEST" >&2; exit 1; }
  [ "${target#*$'\n'}" = "$target" ] \
    || { echo "ABORT: duplicate $source rows in $MANIFEST" >&2; exit 1; }
  [ -f "$BUILD/$source" ] \
    || { echo "ABORT: composed source missing: $BUILD/$source" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$BUILD/$source,dst=$target,readonly")
done

run_probe() {  # run_probe <extra docker args...> -- <probe args...>
  local docker_args=()
  while [ "$1" != "--" ]; do docker_args+=("$1"); shift; done
  shift
  docker run --rm --gpus all \
    --mount "type=bind,src=$REPO,dst=/repo,readonly" \
    "${mounts[@]}" \
    -e STKERNEL_MHC_OVERLAY_BUILD=/repo/build/glm53 \
    "${docker_args[@]}" \
    "$IMAGE" python3 /repo/probes/mhc_glm53_bench.py "$@"
}

if [ "${1:-}" = "--passes" ]; then
  # VLLM_GLM53_MHC_PASSES freezes into the kernels at import, so each combo
  # is its own container. The stock combo runs FIRST and saves the pair's
  # outputs as the cross-combo numerics reference; every other combo loads
  # it and must hold rel_err <= 1e-4. ONEPASS is on in all four so each row
  # times both paths under that combo's pass set.
  refdir=$(mktemp -d /tmp/mhc-passes-ref.XXXXXX)
  trap 'rm -rf "$refdir"' EXIT
  for combo in "" tma ws "tma,ws"; do
    echo "=== pass combo: ${combo:-stock(disabled)} ===" >&2
    # A combo that fails to compile or numerically diverges IS the verdict
    # for that combo (e.g. TMA lowering broken on sm_121) -- record it and
    # keep sweeping. Only a failed STOCK combo aborts: there is nothing to
    # compare against without its reference.
    if [ -z "$combo" ]; then
      run_probe -e VLLM_GLM53_MHC_PASSES=none \
        -e VLLM_GLM53_MHC_ONEPASS=1 \
        -v "$refdir:/refshare" \
        -- --passes --ref-save /refshare/stock_ref.pt
    else
      if ! run_probe -e VLLM_GLM53_MHC_PASSES="$combo" \
        -e VLLM_GLM53_MHC_ONEPASS=1 \
        -v "$refdir:/refshare" \
        -- --passes --ref-load /refshare/stock_ref.pt; then
        echo "VERDICT: combo '$combo' FAILED (compile/divergence) -- axis closed for this combo" >&2
      fi
    fi
  done
  rm -rf "$refdir"
  exit 0
fi

exec docker run --rm --gpus all \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${mounts[@]}" \
  -e STKERNEL_MHC_OVERLAY_BUILD=/repo/build/glm53 \
  "$IMAGE" python3 /repo/probes/mhc_glm53_bench.py "$@"
