#!/usr/bin/env bash
# Build the x86_64 sm_120 CHECK image. Explicit network fetch, then an offline, GPU-free build.
#
# This is the x86_64 counterpart of build-seed.sh, and it is deliberately NOT build.sh's
# input: there is no seed to pin, because the ARM64 seed's parent (glm53:v13-b12x-it) has no
# x86_64 twin -- see make_x86_64_lock.py and bench/OST_97X_LANE.md. The output tag is
# therefore its own, never st-engine:glm53, and `verify.py`'s manifest does not apply to it.
#
#   bash engine/runtime/build-x86_64.sh
#   ST_X86_ARTIFACTS   reusable download/build context (default ~/.cache/st/cuda132-x86_64)
#   ST_X86_IMAGE       output tag (default st-engine:glm53-sm120-x86)
set -euo pipefail
runtime=$(cd "$(dirname "$0")" && pwd)
artifacts=${ST_X86_ARTIFACTS:-$HOME/.cache/st/cuda132-x86_64}
image=${ST_X86_IMAGE:-st-engine:glm53-sm120-x86}

case "$(uname -m)" in
  x86_64) ;;
  *) echo "build-x86_64.sh builds an x86_64 image and needs an x86_64 docker: this is $(uname -m)" >&2; exit 2;;
esac

python3 "$runtime/fetch_x86_64.py" "$artifacts"
cp "$runtime/install_x86_64.py" "$runtime/cuda132.x86_64.lock.json" "$artifacts/"
exec docker build --network none -f "$runtime/Dockerfile.x86_64" -t "$image" "$artifacts"
