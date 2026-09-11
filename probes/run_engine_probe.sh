#!/usr/bin/env bash
# Run an ST probe in a fresh container with no overlay mounts.
# Build the runtime first: bash engine/runtime/build.sh
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
probe=${1:?usage: run_engine_probe.sh probes/engine_kernel_check.py [args...]}
shift
[ -f "$repo/$probe" ] || { echo "missing probe: $repo/$probe" >&2; exit 1; }
image=${ST_IMAGE:-st-engine:glm53}
cache=${ST_CACHE:-$HOME/.cache/st}
mkdir -p "$cache"
mounts=(--mount "type=bind,src=$repo,dst=/repo,readonly"
        --mount "type=bind,src=$cache,dst=/cache")
models=${MODELS:-/home/choiceoh/models}
if [ -d "$models" ]; then
  mounts+=(--mount "type=bind,src=$models,dst=$models,readonly")
fi
envs=(-e PYTHONPATH=/repo -e "MAX_JOBS=${MAX_JOBS:-2}")
for name in $(compgen -v ST_); do envs+=(-e "$name=${!name}"); done
gpu=(--gpus all)
if [ "${ST_PROBE_NO_GPU:-0}" = 1 ]; then
  gpu=()
  envs+=(-e CUTE_DSL_ARCH=sm_121a)
fi
exec docker run --rm "${gpu[@]}" "${mounts[@]}" "${envs[@]}" \
  --entrypoint python3 "$image" -u "/repo/$probe" "$@"
