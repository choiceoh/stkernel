# glm53_dflash_warmup

Boots-time fix: the DFlash input-prep triton kernel
(`_prepare_dflash_inputs_kernel`) JIT-specializes on its BLOCK_SIZE bucket
— `min(256, next_pow2(max_query_len + num_query_per_req))` — and the boot
log showed it **compiling during inference**
(`jit_monitor: Triton kernel JIT compilation during inference ... This
causes a latency spike`), i.e. the first real request of each new
query-length bucket paid a multi-second compile.

The image's spec-decode warmup (`spec_decode_rejection_warmup.py`) covered
only the rejection kernels. This takeover extends it: after the rejection
warmup, the production `prepare_dflash_inputs` wrapper is invoked once per
BLOCK_SIZE bucket (16/32/64/128/256 — query lengths `B - num_query_per_req`,
one dummy request with a 1-token context so every kernel load stays in
bounds, scalars at production values because triton specializes ints on
`==1` and divisibility).

Behavior beyond the warmup window is unchanged (the kernel and its launch
path are untouched; warmup adds ~5 compiles ≈ seconds at boot). Fail-open:
any warmup exception logs and the rejection warmup still runs.
`VLLM_DFLASH_PREP_WARMUP=0` disarms.

Preimage: `cd3bce82…` (glm53:v13-b12x).
