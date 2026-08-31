# glm53_mhc_tilelang

Takeover of the image's `vllm/model_executor/kernels/mhc/tilelang.py`
(GLM-5.3 MHC TileLang dispatcher) to make the small-M decode launch heuristic
env-tunable. STEP_KERNEL_MAP #108 §4: `mhc_fused` + `mhc_pre_big_fuse_with_norm`
= 185 kernels/step (9.8%) — more than our own AllReduce — and both run from
this dispatcher on every one of the 45 layers.

The base heuristic is the one the upstream author left as
`TODO(gnovack): investigate autotuning`:

```python
tile_n = 2 if num_tokens < 8 else 3
n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
```

C=1 decode rides the `num_tokens < 8` arm (tile_n=2, n_splits=8). The dsv4
lane swept the identical TODO heuristic and adopted `(6, 4)` at M<8 for
+16.3% kernel-time at M=6 (dsv4_mhc_tilelang R1, bit-exact residual) — GLM's
shapes (hc_mult=4 → n_out=24, hidden=4096) are in the same family, so the
same sweep applies here.

## Knob

`VLLM_GLM53_MHC_SMALLM="tile_n,n_splits"` — e.g. `6,4`. Read once at import
(capture-safe frozen constant); the per-call validator re-checks the kernel's
shape contracts and falls back to stock on ANY doubt:

- `tile_n` must divide `n_out = hc_mult*(hc_mult+2)` (= 24 here)
- `n_splits` ∈ {1,2,4,8} (the dispatcher's own assert)
- `hidden_size % n_splits == 0` and `(hidden_size//n_splits) % 256 == 0`
  (default n_thr=256; a non-exact h-loop silently drops elements)

Unset (the profile default) = byte-identical stock behavior. The value gets
set only after `probes/mhc_glm53_bench.py` finds a winner on real shapes and
a bracket boot confirms it (quality 9/9 + Korean 0/16 + C=1 step/s, the
standard gates).

## One-pass decode kernel — `VLLM_GLM53_MHC_ONEPASS` (default off)

`tilelang_kernels.py` is also taken over now (preimage `03aeb3f7…`): the stock
kernels are untouched, and one new kernel is appended — `mhc_onepass_tilelang`,
the small-M pair (`mhc_fused` FMA → `big_fuse_with_norm` mixes/sinkhorn/norm)
folded into ONE launch. Grid = one CTA per token; the single tile spans all
`n_out=24` and `split_k=1`, so the `gemm_out_mul/sqrsum` intermediates and
their global roundtrip disappear — −45 launches/step across the 45 layers
(the largest single slice of the #108 §4 axis).

The math is a line-by-line transcription of the two stock kernels;
`tests/test_mhc_onepass_math` proves formula equivalence bitwise in pure
python, and `probes/mhc_glm53_bench.py --onepass` is the GPU validation
harness (rel ≤ 1e-4 vs the stock pair, plus timing). The gate stays CLOSED —
no boot serves through this kernel — until that probe runs clean in an
engine-down window and a bracket adopts it.

## Not in this takeover

- The stock `mhc_pre_big_fuse*` / `mhc_fused` / `mhc_post` kernels are
  byte-identical to the image; only the appended onepass kernel is new.
- The tf32 prenorm `compute_num_split` path (prefill, ≤16-token decode never
  reaches it): dsv4's P2b already rejected prenorm n_splits sweeps on this
  hardware family; not re-opened.

## Recovering the preimage

```bash
# on srv4 -- never docker-exec CUDA into the serving container; create+cp runs nothing
ssh srv4 'docker rm -f tmp-src 2>/dev/null; \
  docker create --name tmp-src glm53:v13-b12x true >/dev/null; \
  docker cp tmp-src:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py /tmp/glm53_tilelang.py; \
  sha256sum /tmp/glm53_tilelang.py; docker rm tmp-src >/dev/null'
# the hash must equal the manifest's third column
scp srv4:/tmp/glm53_tilelang.py .
```
