#!/usr/bin/env bash
# Pin the local fleet image ID before using its tag as a Dockerfile base.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
seed=${ST_SEED_IMAGE:-st-engine:cuda13.2.1-runtime}
expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed_image_id"])' "$repo/engine/runtime/dependencies.json")
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
