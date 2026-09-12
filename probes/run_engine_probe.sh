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
# ST_* (cache paths, probe switches) and STK_* (the profile's declared D11 knobs) reach the container
for name in $(compgen -v ST_ || true) $(compgen -v STK_ || true); do envs+=(-e "$name=${!name}"); done
gpu=(--gpus all)
if [ "${ST_PROBE_NO_GPU:-0}" = 1 ]; then
  gpu=()
  envs+=(-e CUTE_DSL_ARCH=sm_121a)
fi
# A probe takes the same GPUs as a boot, so it takes the same lease -- the last way in
# was a bare `docker run` that no launcher and no queue could see. Short, named (so the
# container is the lease's evidence), and released however the probe ends.
LEASE_PATH=${ST_LEASE_PATH:-$HOME/st-fleet.lock}
LEASE_OWNER=${ST_LEASE_OWNER:-probe/$(whoami)@$(hostname -s)/$$}
NAME=st-probe-$$
if [ "${ST_PROBE_NO_LEASE:-0}" != 1 ]; then
  python3 "$repo/engine/base/fleet_lease.py" acquire --owner "$LEASE_OWNER" --path "$LEASE_PATH" \
    --container "$NAME" --est-minutes "${ST_PROBE_MINUTES:-20}" --note "$probe" >/dev/null \
    || { echo "ABORT: $(python3 "$repo/engine/base/fleet_lease.py" read --path "$LEASE_PATH")" >&2; exit 1; }
  trap 'python3 "$repo/engine/base/fleet_lease.py" release --owner "$LEASE_OWNER" --path "$LEASE_PATH" >/dev/null 2>&1 || true' EXIT INT TERM
fi
docker run --rm --name "$NAME" "${gpu[@]}" "${mounts[@]}" "${envs[@]}" \
  --entrypoint python3 "$image" -u "/repo/$probe" "$@"
