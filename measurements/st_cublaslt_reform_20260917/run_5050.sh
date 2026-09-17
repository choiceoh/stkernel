#!/usr/bin/env bash
# Owned direct RTX 5050 comparison; no fleet queue or service control.
set -euo pipefail
root=/home/choiceoh/st-cublaslt-compare-0917
image=sha256:9f496f0dabe3a7b495d9b97181913cc20be1e4b3d3fcf2694407e34f24b3981b
exec 9>"$HOME/.cache/st/gpu-probe.lock"
flock -n 9 || exit 1
[[ -z $(docker ps --filter name=st-probe- --format '{{.Names}}') ]] || exit 1
name=st-probe-cublas-split-final-0917
trap 'docker stop -t 3 "$name" >/dev/null 2>&1 || true' EXIT
run() {
  timeout 900 docker run --rm --name "$name" --gpus all --network none --cpus 2 --memory 4g \
    -e PYTHONPATH=/work:/work/DeepGEMM/build/lib.linux-x86_64-cpython-312 \
    -e ST_NATIVE_BUILD_ROOT=/out/native -e TRITON_CACHE_DIR=/out/triton -e DG_JIT_CACHE_DIR=/out/deep_gemm \
    -e MAX_JOBS=2 -e OMP_NUM_THREADS=1 -e ST_TEST_CUBLASLT_GPU=1 \
    -v "$root:/work:ro" -v "$root/out:/out" -w /work --entrypoint python3 "$image" -u "$@"
}
nvidia-smi --query-gpu=name,uuid,driver_version,utilization.gpu,memory.free --format=csv >"$root/out/split-gpu-before.csv"
run -m unittest tests.test_engine_cublaslt.NativePreparationTests -v >"$root/out/split-native-tests.log" 2>&1
run probes/engine_cublaslt_check.py --gpu --timing-target sm120-probe \
  --packed-weight /work/packs/e29b30a8de75e0e3e6fc6895abd857dac5a74910db589ab28f5306e98fc50009.pt \
  --shape 8x4096x20480 --shape 16x4096x20480 --output /out/split-fc.json >"$root/out/split-fc.log" 2>&1
run probes/engine_cublaslt_check.py --gpu --timing-target sm120-probe \
  --packed-weight /work/packs/7e3c089b56122e41d05b7fcfc7163749d4f5a6eaa8f4186e2199fc71762d394e.pt \
  --shape 7x38784x4096 --shape 8x38784x4096 --shape 14x38784x4096 --shape 16x38784x4096 \
  --output /out/split-head.json >"$root/out/split-head.log" 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,utilization.gpu,memory.free --format=csv >"$root/out/split-gpu-after.csv"
