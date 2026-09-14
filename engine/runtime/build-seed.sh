#!/usr/bin/env bash
# Explicit network preparation followed by an offline, GPU-free image build.
set -euo pipefail
runtime=$(cd "$(dirname "$0")" && pwd)
artifacts=${ST_RUNTIME_ARTIFACTS:-$HOME/.cache/st/cuda132-seed}
expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bootstrap_seed_image_id"])' "$runtime/cuda132.lock.json")
seed=${ST_BOOTSTRAP_IMAGE:-glm53:v13-b12x-it}
actual=$(docker image inspect "$seed" --format '{{.Id}}')
if [ "$actual" != "$expected" ]; then
  echo "ST bootstrap image mismatch: $actual (expected $expected)" >&2
  exit 1
fi
mkdir -p "$artifacts/wheels"
python3 "$runtime/fetch_cuda132.py" "$artifacts/wheels"
cp "$runtime/"{cuda132.lock.json,install_cuda132.py,fetch_cuda132.py,promote_deep_gemm.py} "$artifacts/"
pinned="st-engine-bootstrap:${expected#sha256:}"
docker tag "$expected" "$pinned"
exec docker build --network none --build-arg "ST_BOOTSTRAP_IMAGE=$pinned" \
  -f "$runtime/Dockerfile.seed" -t "${ST_CUDA132_SEED_IMAGE:-st-engine:cuda13.2.1-runtime}" "$artifacts"
