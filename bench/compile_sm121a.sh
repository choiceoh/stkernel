#!/usr/bin/env bash
# Compile ST's b12x CuTe kernels for sm_121a on a box that is NOT a Spark, with no device.
#
# sm_121a here is a compile target for ptxas, not this box's card. Nothing this runs opens a
# CUDA context (`m64_compile`-style harnesses assert it), so the verdict it can give is
# "this lowers, and these are its registers/shared/stack" -- never a number. A number from
# another card is that card's (bench/OST_97X_LANE.md).
#
#   bash bench/compile_sm121a.sh <script.py> [args...]
#   ST_X86_IMAGE   the x86_64 check image (default st-engine:glm53-sm120-x86)
#   ST_VENDORED    the Sparks' flashinfer, unpacked (default ~/st-x86-flashinfer/vendored)
#   ST_EVIDENCE    bound at /evidence (default ~/st-compile-evidence)
#
# Why ST_VENDORED exists. The x86_64 image resolves flashinfer from pip (0.6.18.post1); the
# Sparks run the vendored 0.6.18.dev20260819. Only the dev build carries
# `Sm120B12xBlockScaledDenseGemmKernel._collapse_to_vmk`, and
# engine/kernels/b12x/_moe_dynamic/generic.py borrows that staticmethod AT IMPORT TIME -- so
# on the stock image the entire b12x path dies with `AttributeError` before any kernel is
# named. The package is pure Python (0 .so), so the Sparks' copy carries. Produce it once:
#
#   ssh <spark> 'docker run --rm --entrypoint bash <st image> -c "cd /usr/local/lib/python3.12/dist-packages \
#     && tar czf - flashinfer flashinfer_cubin flashinfer_*.dist-info"' > flashinfer-dev.tgz
#   mkdir -p ~/st-x86-flashinfer/vendored && tar xzf flashinfer-dev.tgz -C ~/st-x86-flashinfer/vendored
#
# The image installs into site-packages, which SHADOWS dist-packages: mounting over the
# latter changes nothing and the import silently keeps resolving to post1.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
image=${ST_X86_IMAGE:-st-engine:glm53-sm120-x86}
vendored=${ST_VENDORED:-$HOME/st-x86-flashinfer/vendored}
evidence=${ST_EVIDENCE:-$HOME/st-compile-evidence}
site=/usr/local/lib/python3.12/site-packages

[ $# -ge 1 ] || { echo "usage: bash bench/compile_sm121a.sh <script.py> [args...]" >&2; exit 2; }
[ -d "$vendored/flashinfer" ] || {
  echo "no vendored flashinfer at $vendored -- see the header of $0 for the one-time export" >&2
  exit 2; }
mkdir -p "$evidence"

mounts=()
for d in "$vendored"/*; do
  mounts+=(-v "$d:$site/$(basename "$d"):ro")
done

exec docker run --rm \
  --network=none \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e CUTE_DSL_ARCH=sm_121a \
  -e PYTHONPATH=/repo \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$repo":/repo:ro \
  -v "$evidence":/evidence \
  "${mounts[@]}" \
  -w /repo \
  --entrypoint python3 \
  "$image" "$@"
