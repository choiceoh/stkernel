# Qwen3.8-Flash-Next on the FlashInfer this fleet proves b12x with.
#
# The entire b12x MoE campaign in this repo -- shared workspace, EP lanes,
# static v4/v5, tile-major weights, SF6 packers -- overrides moe_dispatch.py at
# preimage f6923850..., which is FlashInfer 0.6.18.dev20260819's file. The
# stock Qwen image ships 0.6.17 (3,047 lines, 0a8f4397...), where none of it
# applies. Qwen is the first model since GLM-5.3 whose NVFP4 group_size 16 can
# use those lanes at all, so the campaign follows the model onto one base.
#
# Both images carry torch 2.13.0+cu130, so the ABI matches.
#
# Grafting subdirectories was tried first and does not close:
#   - blackwell_sm12x alone -> ImportError, quantize_block_mxfp4 moved
#   - plus both cute_dsl trees -> vLLM's NvFp4 oracle refuses the backend
#     ("kernel does not support current device"): a 0.6.17 install carrying
#     0.6.18 subpackages is a chimera its own support probe can see
#   - package + dist-info only -> flashinfer-cubin 0.6.17 against flashinfer
#     0.6.18, which FlashInfer itself refuses
# The cubins are version-keyed precompiled kernels, so FLASHINFER_DISABLE_
# VERSION_CHECK=1 would mix them rather than fix them. All three move together.
FROM glm53:v13-b12x-it AS fi
FROM vllm/vllm-openai:qwen38-flash-next
ARG D=/usr/local/lib/python3.12/dist-packages
RUN rm -rf $D/flashinfer $D/flashinfer_cubin $D/flashinfer_jit_cache \
           $D/flashinfer_python-*.dist-info $D/flashinfer_cubin-*.dist-info \
           $D/flashinfer_jit_cache-*.dist-info
COPY --from=fi $D/flashinfer $D/flashinfer
COPY --from=fi $D/flashinfer_cubin $D/flashinfer_cubin
COPY --from=fi $D/flashinfer_python-0.6.18.dev20260819.dist-info $D/flashinfer_python-0.6.18.dev20260819.dist-info
COPY --from=fi $D/flashinfer_cubin-0.6.18.dev20260819.dist-info $D/flashinfer_cubin-0.6.18.dev20260819.dist-info
RUN python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"
