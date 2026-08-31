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

## Not in this takeover

- The `mhc_pre_big_fuse*` kernel internals live in `tilelang_kernels.py`
  (stock, not taken over) — only launch configs are touched here.
- The tf32 prenorm `compute_num_split` path (prefill, ≤16-token decode never
  reaches it): dsv4's P2b already rejected prenorm n_splits sweeps on this
  hardware family; not re-opened.
- Fusing `mhc_fused` + `big_fuse_with_norm` into one kernel (−89/step ≈ 1.5%
  ceiling) means editing TileLang kernel source — a different, larger job,
  only worth visiting if the launch-config sweep comes back flat.

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
