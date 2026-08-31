# glm53_sm121_mla_prefill

Opt-in dense-MHA prefill for the fresh short/full-attention region of
GLM-5.3-Flash sparse MLA on DGX Spark. The image currently selects
`FLASHINFER_MLA_SPARSE_SM90` on SM121 and logs that no MLA prefill backend
accepts GLM's `(qk_nope, qk_rope, v) = (256, 0, 256)` dimensions. Every
prefill consequently takes the top-k MQA path.

This module extends `FlashAttnPrefillBackend.supports_mla_dimensions` and adds
an exact-arm request guard in `prepare_metadata`.

When the gate admits the layer, vLLM's existing `SparseMLACommonImpl` routes
prefill with `prefill_max_seq_len <= index_topk` through its FlashAttention 2
`forward_mha`; GLM's
`index_topk=2048` means that region is semantically full attention. Longer
prefills, cached-prefix/multi-turn prefills, and all decode tokens stay on the
existing top-k MQA path.

## Gate and rollback

The arm is enabled only by the exact value
`VLLM_GLM53_SM121_MLA_PREFILL=1`. It additionally requires every observed
production contract to match:

- compute capability 12.1 (GB10 / SM121),
- model type `glm5_next_text`, 64 total heads, KV LoRA rank 512,
- `(qk_nope, qk_rope, v) = (256, 0, 256)`, `index_topk=2048`,
  `index_kpool=4`,
- standard `fp8_e4m3` HND KV cache, and
- outer attention backend `FLASHINFER_MLA_SPARSE_SM90`, with the selected
  dense-prefill kernel version resolving to FlashAttention 2.

Unset, `0`, malformed values, a future image selecting the SM120 backend, a
packed `fp8_ds_mla` cache, or any shape drift all fail closed to the image's
current behavior. Rollback is therefore the env value alone; the profile keeps
it at `0` until a bracket run adopts it.

Expected enabled boot fingerprint:

```text
Using FLASHINFER_MLA_SPARSE_SM90 attention backend ...
Using FLASH_ATTN MLA prefill backend.
```

The existing `No MLA prefill backend supports this model` warning is the
disabled/fail-closed fingerprint.

## Deliberate exclusions and validation

This does **not** enable the SM100-only FA4 custom-mask path and does not select or
modify `FLASHINFER_MLA_SPARSE_SM120`. Upstream marks that implementation as
having no dense-MHA prefill path; reported GB10 and packed-cache failures make
flipping its class flag unsafe. The current HND `fp8_e4m3` path is important:
the generic context gather can dequantize it, unlike the packed
`fp8_ds_mla` regression reported in vLLM #48611.

The GLM opt-in also overrides `prepare_metadata` for this exact instance only.
If `chunked_context` exists, it clears `use_dense_mha` before the indexer or
attention forward consumes it, routing the whole batch to MQA. This excludes
`run_prefill_context_chunk(causal=False)`, which can hit the SM121 FA2
device-side assertion reported in vLLM #50707. Stock FlashAttention dimensions
and fresh GLM prefills are unchanged by that request-level guard.

There is no safe Python exception fallback after a CUDA kernel launch: an
asynchronous illegal launch can poison the context before an exception is
observed. Admission is therefore pre-launch and fail-closed, not a `try/except`
around FlashAttention. Before adoption, run an engine-down bracket with:

1. a 1-2048 token **fresh** prefill sweep and a >2048 MQA control,
2. prefix-cache/multi-turn and mixed fresh+cached batches proving MQA remains
   and the non-causal FA2 context call is never reached,
3. output comparison against the `=0` arm, including retrieval and Korean
   corruption gates, and
4. the 75K prompt benchmark; expected end-to-end gain is only 0-5% because
   attention is about 11% of prefill time.

Base contract: `glm53:v13-b12x` and official vLLM `main` at
`dafbef15a1c879c64ebb99427917e4ca8d5bca1e` contain the same 497-line
`flash_attn.py`, SHA-256
`6eb45bb113d29a1a2728703dcc171bb76929973c00801057481b31dff886b3f1`.
A read-only probe against the live srv4 image (no attention-kernel launch)
confirmed compute capability `(12, 1)`, backend availability `True`, and
`get_flash_attn_version(head_size=256, head_size_v=256) == 2`. This proves the
selection prerequisites only; enabled-path correctness and performance still
require the engine-down bracket above.
