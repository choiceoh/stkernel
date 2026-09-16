#!/usr/bin/env bash
set -euo pipefail
root=/home/choiceoh/st-cublaslt-compare-0917
image=sha256:9f496f0dabe3a7b495d9b97181913cc20be1e4b3d3fcf2694407e34f24b3981b
exec 9>"$HOME/.cache/st/gpu-probe.lock"
flock -n 9 || exit 1
[[ -z $(docker ps --filter name=st-probe- --format '{{.Names}}') ]] || exit 1
name=st-probe-cublas-layout-final-0917
trap 'docker stop -t 3 "$name" >/dev/null 2>&1 || true' EXIT
run() {
 timeout 900 docker run --rm --name "$name" --gpus all --network none --cpus 2 --memory 4g \
  -e PYTHONPATH=/work:/work/DeepGEMM/build/lib.linux-x86_64-cpython-312 \
  -e ST_NATIVE_BUILD_ROOT=/out/native -e TRITON_CACHE_DIR=/out/triton -e DG_JIT_CACHE_DIR=/out/deep_gemm \
  -e ST_TEST_CUBLASLT_GPU=1 -e MAX_JOBS=2 -e OMP_NUM_THREADS=1 \
  -v "$root:/work:ro" -v "$root/out:/out" -w /work --entrypoint python3 "$image" -u "$@"
}
nvidia-smi --query-gpu=name,uuid,driver_version,utilization.gpu,memory.free --format=csv >"$root/out/layout-final-before.csv"
run -m unittest tests.test_engine_cublaslt.NativePreparationTests tests.test_engine_cublaslt_serving -v >"$root/out/layout-final-native.log" 2>&1
for kind in fc head; do
 if [[ $kind == fc ]]; then pack=e29b30a8de75e0e3e6fc6895abd857dac5a74910db589ab28f5306e98fc50009
 else pack=7e3c089b56122e41d05b7fcfc7163749d4f5a6eaa8f4186e2199fc71762d394e; fi
 run probes/engine_cublaslt_reform_check.py --gpu --timing-target sm120-probe --kind "$kind" \
  --packed-weight "/work/packs/$pack.pt" --baseline-reader /work/baseline_reader.py \
  --output "/out/layout-final-$kind.json" >"$root/out/layout-final-$kind.log" 2>&1
done
nvidia-smi --query-gpu=name,uuid,driver_version,utilization.gpu,memory.free --format=csv >"$root/out/layout-final-after.csv"
