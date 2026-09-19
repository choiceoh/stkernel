#!/usr/bin/env bash
# Is the single-GPU lane's box actually fit to run a check? Run this FROM a controller.
#
# The lane fails in layers -- ssh, then docker, then the NVIDIA runtime, then the image,
# then what is inside it, then the queue's own admission -- and a failure in one reads like
# a failure in another. `fleet.sh run` says "ABORT: no ST image" whether the image is
# missing, the daemon is down, or the box never answered. This walks the chain in order and
# names the first link that does not hold, so the answer is the layer and not the symptom.
#
#   bash tools/ost-97x-selftest.sh                 # the check lane's host, from FLEET_CHECK_GPU_HOST
#   bash tools/ost-97x-selftest.sh ost-97x         # or an ssh alias, as ~/.ssh/config spells it
#   ST_IMAGE=... bash tools/ost-97x-selftest.sh    # a tag other than the box's default
#
# Read-only: it starts one throwaway container and writes nothing on the box.
set -uo pipefail

host=${1:-${FLEET_CHECK_GPU_HOST:-ost-97x}}
image=${ST_IMAGE:-st-engine:glm53-sm120-x86}
budget=${ST_PROBE_GIB:-4}
repo=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)
failed=0

# The queue logs what it runs, and an escape sequence in a log is noise nobody asked for:
# colour only when a terminal is actually watching.
if [ -t 1 ]; then GREEN=$'\033[32m'; RED=$'\033[31m'; OFF=$'\033[0m'; else GREEN=''; RED=''; OFF=''; fi
ok()   { printf '  %sPASS%s  %-26s %s\n' "$GREEN" "$OFF" "$1" "${2-}"; }
bad()  { printf '  %sFAIL%s  %-26s %s\n' "$RED" "$OFF" "$1" "${2-}"; failed=$((failed + 1)); }
skip() { printf '  ----  %-26s %s\n' "$1" "${2-}"; }

echo "ost-97x self-test: $host (image ${image}, budget ${budget} GiB)"

# 1. the door. Everything below is meaningless if this does not hold, so stop here.
if answer=$("${SSH[@]}" "$host" 'echo "$(hostname) $(uname -m) $(nproc)cpu"' 2>&1); then
  ok ssh "$answer"
else
  bad ssh "$(printf '%s' "$answer" | tail -1)"
  echo "the box did not answer: nothing below can be judged. $failed failed."
  exit 1
fi
case $answer in
  *x86_64*) ok arch 'x86_64 -- an ARM64 image will not run here' ;;
  *) bad arch "expected x86_64, got: $answer" ;;
esac

# 2. the card, as the driver sees it. Capability decides which lanes could ever run.
if gpu=$("${SSH[@]}" "$host" 'nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv,noheader' 2>&1); then
  ok gpu "$gpu"
  case $gpu in
    *' 12.0'*) skip 'gpu arch' 'sm_120: engine/kernels/b12x only; native lanes are compute_121a' ;;
    *' 12.1'*) ok 'gpu arch' 'sm_121a -- the fleet lanes own this' ;;
  esac
else
  bad gpu "$(printf '%s' "$gpu" | tail -1)"
fi

# 3. docker, as the lane's own evidence query runs it: not root, not sudo. A box whose
#    `docker ps` needs a password answers the queue with rc != 0 and reads as "no room".
if "${SSH[@]}" "$host" 'docker ps --filter name=st-probe- --format "{{.Names}}"' >/dev/null 2>&1; then
  ok docker 'the lane query runs unprivileged'
else
  bad docker 'docker ps failed for this user -- the queue will read the box as unreachable'
fi

# 4. the NVIDIA runtime, proven by the driver's own tools appearing inside a stock image.
if seen=$("${SSH[@]}" "$host" "docker run --rm --gpus all ubuntu:24.04 nvidia-smi --query-gpu=name --format=csv,noheader" 2>&1); then
  ok 'nvidia runtime' "a container sees: $seen"
else
  bad 'nvidia runtime' "$(printf '%s' "$seen" | tail -1)"
fi

# 5-6. the image, and whether what the checks import is actually importable in it. A build
#      can install cleanly and still be unusable: the first image built here installed all
#      42 locked wheels and could not import flashinfer (no tvm_ffi).
if "${SSH[@]}" "$host" "docker image inspect $image" >/dev/null 2>&1; then
  ok image "$image"
  probe='
import sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "cap", ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else "-")
import flashinfer
from flashinfer.utils import supported_compute_capability
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
print("flashinfer", flashinfer.__version__, "| bf16 matmul on device ok")
'
  if out=$("${SSH[@]}" "$host" "docker run --rm --gpus all $image -c '$probe'" 2>&1); then
    ok 'image imports' "$(printf '%s' "$out" | head -1)"
    ok 'image on device' "$(printf '%s' "$out" | tail -1)"
  else
    bad 'image imports' "$(printf '%s' "$out" | tail -1)"
  fi
else
  bad image "$image is not on $host -- build it with engine/runtime/build-x86_64.sh"
  skip 'image imports' 'no image to judge'
fi

# 7. the queue's own admission, by the module the queue uses. Anything else is a guess at it.
if [ -f "$repo/bench/fleet_single.py" ]; then
  if why=$(python3 "$repo/bench/fleet_single.py" evidence --host "$host" --gib "$budget" 2>&1); then
    ok admission "room for a ${budget} GiB check"
  else
    bad admission "${why:-refused}"
  fi
else
  skip admission 'run this from a checkout to judge it'
fi

# 8. what the runner needs on the far side before it can stage anything.
if "${SSH[@]}" "$host" 'command -v rsync >/dev/null && mkdir -p .cache/st && test -w .cache/st' 2>/dev/null; then
  ok staging 'rsync present, ~/.cache/st writable'
else
  bad staging 'the probe runner cannot stage a tree here'
fi

echo
if [ "$failed" = 0 ]; then
  echo "fit to run a check: bench/fleet.sh run --gpu --check <session> -- <one-GPU ST check> (FLEET_CHECK_GPU_HOST=$host)"
else
  echo "$failed check(s) failed -- fix the FIRST one; the rest usually follow from it."
fi
exit $((failed > 0))
