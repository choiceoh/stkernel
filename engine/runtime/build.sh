#!/usr/bin/env bash
# Pin the local fleet image ID before using its tag as a Dockerfile base.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
seed=${ST_SEED_IMAGE:-glm53:v13-b12x-it}
expected=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
actual=$(docker image inspect "$seed" --format '{{.Id}}')
if [ "$actual" != "$expected" ]; then
  echo "ST seed image mismatch: $actual (expected $expected)" >&2
  exit 1
fi
# A separate local tag keeps the build bound to the checked image ID.
pinned="st-engine-seed:${expected#sha256:}"
docker tag "$expected" "$pinned"
exec docker build --network none --build-arg "ST_SEED_IMAGE=$pinned" \
  -f "$repo/engine/runtime/Dockerfile" -t "${ST_IMAGE:-st-engine:glm53}" "$repo"
