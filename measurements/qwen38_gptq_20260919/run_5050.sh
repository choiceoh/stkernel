#!/usr/bin/env bash
# Own only the RTX 5050 lane; no fleet lease or serving changes occur here.
set -euo pipefail
root=${1:?task root required}
sha=${2:?frozen source SHA required}
engine=${3:?frozen engine tree required}
image=sha256:9f496f0dabe3a7b495d9b97181913cc20be1e4b3d3fcf2694407e34f24b3981b
mkdir -p "$HOME/.cache/st" "$root/out/torch-kernels"
exec 9>"$HOME/.cache/st/gpu-probe.lock"
echo 'Waiting for the RTX 5050 lane'
flock -w 43200 9 || { echo '5050 lane wait expired'; exit 1; }
if [[ -n $(docker ps --filter name=st-probe- --format '{{.Names}}') ]]; then
  echo 'another ST GPU probe is running without the lane lock'; exit 1
fi
python3 - <<'PY'
from pathlib import Path
mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
assert mem['MemAvailable'] >= 8 * 1024 * 1024, 'need 4 GiB budget and 4 GiB host floor'
PY
nvidia-smi --query-gpu=name,uuid,driver_version,memory.free --format=csv > "$root/out/gpu-before.csv"
name=st-probe-qwen38-gptq-330k-4436
trap 'docker stop -t 10 "$name" >/dev/null 2>&1 || true' EXIT
for rank in 0 1 2 3; do
  timeout --signal=TERM 14400 docker run --rm --name "$name" --gpus all \
    --network none --cpus 2 --memory 4g --pids-limit 256 --user "$(id -u):$(id -g)" \
    -e PYTHONPATH=/work -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 \
    -e ST_QWEN_5050_LOCK=/lane.lock -e PYTORCH_KERNEL_CACHE_PATH=/out/torch-kernels \
    -v "$HOME/.cache/st/gpu-probe.lock:/lane.lock:ro" \
    -v "$root/code:/work:ro" -v "$root/inputs:/inputs:ro" -v "$root/out:/out" \
    -w /work --entrypoint python3 "$image" -u -m probes.qwen38_gptq_offline \
    --inputs "/inputs/rank$rank" --out "/out/rank$rank" --source-sha "$sha" --engine-tree "$engine" \
    > "$root/out/rank$rank.log" 2>&1
done
nvidia-smi --query-gpu=name,uuid,driver_version,memory.free --format=csv > "$root/out/gpu-after.csv"
echo 'All four 330K pack/error jobs finished on RTX 5050; serving validation is separate'
