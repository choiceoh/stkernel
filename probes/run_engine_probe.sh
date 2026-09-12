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
# A probe takes the same GPUs as a boot, so it takes the same lease -- a bare `docker run`
# was the last way in that no launcher and no queue could see. The lease is the head
# node's one file (launchers/lib/fleet-lease.sh), not this node's: a lease written here
# would be one nobody else can read. Named container, heartbeat while it runs, released
# however the probe ends.
LEASE_OWNER=${ST_LEASE_OWNER:-probe/$(whoami)@$(hostname -s)/$$}
NAME=st-probe-$$
# A probe that asks for no GPU (ST_PROBE_NO_GPU=1: import and source checks) reserves
# nothing -- taking four Sparks for a CPU check is the opposite of smooth.
if [ "${ST_PROBE_NO_LEASE:-0}" != 1 ] && [ "${ST_PROBE_NO_GPU:-0}" != 1 ]; then
  # Set, then source: an assignment prefix on `.` is temporary in bash, so the helper's
  # own defaults would be discarded the moment the builtin returns.
  FLEET_REPO=$repo
  . "$repo/launchers/lib/fleet-lease.sh"
  # Wait, do not abort: a probe the queue just sent in should not lose its turn because
  # the previous holder is still tearing down (2026-09-12: three reservations died in two
  # seconds each on a lock that was removed moments later).
  until fleet_lease acquire --owner "$LEASE_OWNER" --container "$NAME" \
          --est-minutes "${ST_PROBE_MINUTES:-20}" --note "$probe" >/dev/null 2>&1; do
    [ "$(date +%s)" -lt "$(( ${LEASE_DEADLINE:=$(( $(date +%s) + 60 * ${ST_PROBE_WAIT_MINUTES:-10} ))} ))" ] \
      || { echo "ABORT: the fleet stayed held for ${ST_PROBE_WAIT_MINUTES:-10} min: $(fleet_lease read 2>/dev/null || echo unreachable)" >&2; exit 1; }
    echo "  waiting for the fleet: $(fleet_lease read 2>/dev/null || echo unreachable)" >&2
    sleep "${ST_PROBE_POLL_S:-10}"
  done
  BEAT=$(fleet_lease_beat "$LEASE_OWNER")
  # If this shell is killed outright (SIGKILL) the trap cannot run and the lease leaks.
  # It is not lost: with its evidence on another node it goes stale after the grace and
  # the next acquirer reclaims it, and `read` reports it as free (stale: ...) meanwhile.
  trap 'kill $BEAT 2>/dev/null; fleet_lease release --owner "$LEASE_OWNER" >/dev/null 2>&1 || true' EXIT INT TERM
fi
docker run --rm --name "$NAME" "${gpu[@]}" "${mounts[@]}" "${envs[@]}" \
  --entrypoint python3 "$image" -u "/repo/$probe" "$@"
