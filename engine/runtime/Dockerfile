# The independently built, immutable CUDA 13.2.1 runtime seed owns all libraries.
ARG ST_SEED_IMAGE=st-engine:cuda13.2.1-runtime
FROM ${ST_SEED_IMAGE}
WORKDIR /opt/st
COPY engine /opt/st/engine
RUN CUDA_VISIBLE_DEVICES= python3 -m engine.runtime.verify > /opt/st/runtime-manifest.json
ENV PYTHONPATH=/opt/st \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_CACHE_PATH=/cache/cu132/driver \
    TRITON_CACHE_DIR=/cache/cu132/triton \
    TILELANG_CACHE_DIR=/cache/cu132/tilelang \
    DG_JIT_CACHE_DIR=/cache/cu132/deep_gemm \
    FLASHINFER_WORKSPACE_BASE=/cache/cu132 \
    ST_MLA_BUILD_ROOT=/cache/cu132/mla \
    ST_DENSE_BUILD_ROOT=/cache/cu132/st-dense \
    ST_ONESHOT_BUILD_ROOT=/cache/cu132/st-oneshot \
    MAX_JOBS=2
ENTRYPOINT ["python3", "-m", "engine.profiles.glm53.boot"]
CMD ["--help"]
