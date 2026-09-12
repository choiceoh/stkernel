# The pinned fleet seed supplies the tested CUDA 13/PyTorch/CuTe ABI.
# Override only with a seed whose dependency versions have been validated.
ARG ST_SEED_IMAGE=glm53:v13-b12x-it
FROM ${ST_SEED_IMAGE}
COPY engine/runtime/promote_deep_gemm.py /tmp/st-promote-deep-gemm.py
RUN python3 /tmp/st-promote-deep-gemm.py \
    && python3 -m pip uninstall -y vllm \
    && python3 -c "from pathlib import Path; import shutil, sysconfig; shutil.rmtree(Path(sysconfig.get_path('purelib')) / 'vllm', ignore_errors=True)" \
    && python3 -c "import importlib.util; assert importlib.util.find_spec('vllm') is None; import deep_gemm; assert callable(deep_gemm.fp8_fp4_mqa_logits); assert callable(deep_gemm.tf32_hc_prenorm_gemm)" \
    && rm /tmp/st-promote-deep-gemm.py
WORKDIR /opt/st
COPY engine /opt/st/engine
RUN python3 -m engine.runtime.verify > /opt/st/runtime-manifest.json
ENV PYTHONPATH=/opt/st \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRITON_CACHE_DIR=/cache/triton \
    TILELANG_CACHE_DIR=/cache/tilelang \
    DG_JIT_CACHE_DIR=/cache/deep_gemm \
    FLASHINFER_WORKSPACE_BASE=/cache \
    ST_MLA_BUILD_ROOT=/cache/mla \
    ST_DENSE_BUILD_ROOT=/cache/st-dense \
    ST_ONESHOT_BUILD_ROOT=/cache/st-oneshot \
    MAX_JOBS=2
ENTRYPOINT ["python3", "-m", "engine.profiles.glm53.boot"]
CMD ["--help"]
