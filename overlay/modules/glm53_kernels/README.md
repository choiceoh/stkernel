# glm53_kernels

GLM-5.3 층 커널 — kpool 희소 인덱서(tail-select 융합), tail 슬롯, KDA 프리필(regime 버킷 + #368 direct-out), MHC TileLang(프리필 big_fuse 오버라이드 + MK 훅).

**34차 §8 (2026-09-06, 운영자 "전부 지워")**: radix top-k 확장(`glm53_kpool_topk.cu/.py`, `KPOOL_FUSED_TOPK`), SM121 MLA 프리필(`flash_attn.py`, `SM121_MLA_PREFILL`), MHC 디코드 쪽 오버라이드(`MHC_SMALLM`·`MHC_ONEPASS`·`mhc_onepass_tilelang`)를 삭제했다 — opt-in 이었고 판정 기록이 없거나(top-k·SM121) mk_mhc 가 대체했다(ONEPASS/SMALLM). KDA 프리필 버킷(`kda.py`·`chunk_delta_h.py`, `KDA_PREFILL_REGIME`)도 후보였으나 같은 날 #368 이 그 파일에 direct-out 프리필 경로를 얹어 **유지**했다. 아래 해당 절은 기록이다.

2026-09-05 (34차, 운영자 "디폴트화된 모듈들을 4~5개씩 하나로 묶어라") 에 아래 모듈들을 이 디렉터리 하나로 합쳤다. **매니페스트 행·베이스 계약·소스 파일·노브·기본값은 그대로**이고 디렉터리와 `manifest.tsv`·`requires`·README 만 합쳐졌다(합성 결과 `build/glm53/` 의 파일은 바이트 동일(메가커널 .cu 주석의 경로 한 줄 제외), 행 순서만 바뀜). 옛 이름은 원장·런북·커밋에 그대로 남아 있고, 아래 절이 옛 모듈 하나씩이다.

| 옛 모듈 | 파일 | 무엇 |
|---|---|---|
| `glm53_kpool_tail_select` | `sparse_attn_indexer_kpool.py` | kpool 인덱서 op 접수 + tail-select 융합 (`INDEXER_DECODE_FUSED`, `KPOOL_UPDATE_DIRECT_POS`); radix top-k 확장은 34차 §8 일몰 |
| `glm53_tail_slot_persistent` | `glm53_kpool_indexer.py` | kpool tail 슬롯 고정 버퍼 (이 이미지에서는 잠들어 있음) |
| `glm53_kda_prefill_regime` | `chunk_delta_h.py`, `kda.py` | KDA 프리필 autotune 버킷 (`KDA_PREFILL_REGIME`, 기본 off) + #368 direct-out 출력 (`KDA_PREFILL_DIRECT_OUT`, 기본 off) |
| `glm53_mhc_tilelang` | `tilelang.py`, `tilelang_kernels.py` | MHC TileLang 접수 (프리필 big_fuse `MHC_BIGFUSE`·패스 `MHC_PASSES` 오버라이드; MK-MHC 훅) |

---

## glm53_kpool_tail_select (was `overlay/modules/glm53_kernels/`)

## glm53_kpool_tail_select

GLM-5.3's indexer pools keys four at a time (`index_kpool = 4`) and selects
pools, not tokens, so the pool-level top-k decides what the model may look at.

### Skip dead short-prefill scoring under dense MLA

The stock sparse indexer has a request-level bypass for fresh short prefills
that the selected MLA backend will execute as dense MHA. The kpool fork
predates that bypass, so even when `glm53_sm121_mla_prefill` admitted dense MHA
it still filled and causally masked a `[tokens, 2048]` top-k buffer in every
sparse layer. Dense MLA never reads that buffer.

The kpool path now derives the sibling MLA metadata key from the exact
`*.indexer.k_cache -> *.attn` GLM module layout. After it writes the pooled
index-K cache and the persistent tail cache, it returns before the unused
top-k buffer work when the MLA metadata says `use_dense_mha=True`. Mixed
prefill/decode batches, CUDA graph capture, cached-context or long MQA
prefills, and any module-name drift retain the old path. This makes the change
inert unless the separate SM121 dense-prefill arm is admitted.

The remaining changes to the kpool selection path are below.

### Keep prefill pool windows as views

Every prefill cache write compresses four consecutive raw K and gate rows.
The old caller built an `[tokens, 4]` index grid and advanced-indexed both
inputs, materializing two overlapping `[tokens, 4, 128]` tensors before the
compression kernel. The kernel ABI already accepts independent row and
pool-slot strides, but its public wrapper immediately called `contiguous()`
and discarded that capability.

The kpool prefill caller now forms `[window, 4, 128]` with
`unfold(...).movedim(...)`, which is a zero-copy view with strides
`[128, 128, 1]`, and invokes the pinned stock Triton kernel with those strides.
Destination slots are shifted to each window's completion token, preserving
the previous mask and cache layout. This removes the index grid and both K/gate
gathers for every prefill size; the last head dimension remains contiguous as
required by the kernel.

The dense cache-only model path now emits K and gate as two views of one fused
projection, so their row strides differ from their logical width. Pool
compression already carries explicit row strides; the persistent-tail seeder
now does the same instead of assuming `row_stride == head_dim`. This keeps the
fused projection zero-copy through both cache writers.

### Reuse the dense-prefill write plan across sparse layers

All sparse layers in one dense prefill receive the same immutable pool-level
`slot_mapping`. Previously each layer formed the same completion slice and
launched the same `loc >= 0` mask construction before compressing its own
K/gate values. The first layer now stores that destination/mask pair in
the current `ForwardContext`; the remaining layers reuse it by exact tensor
pointer, shape, stride, dtype, device, and pool-width key.

This cache exists only after the shared dense-MLA no-consumer predicate admits
an eager request. Capture, full CUDA graphs, mixed batches, MQA prefill, or a
different metadata allocation never reuse it, and the context releases all
entries when the forward ends. Index-K and persistent-tail writes remain in
their original order.

### Make short MQA prefill a one-pass index write

When a fresh context is within `index_topk`, the MQA fallback attends to every
causal token and does not run sparse logits. The old fast path nevertheless
performed three full-buffer passes: initialize `[tokens, 2176]` to `-1`,
broadcast an arange over it, then build/apply a boolean causal mask. It also
requested the sparse logits gather workspace before iterating an intentionally
empty chunk list.

A 2-D Triton kernel now writes `column` or `-1` directly from each row's token
position. Pure short-prefill batches skip the overwritten sentinel fill;
mixed batches retain it for their decode/padded rows. The shared logits
workspace is requested only in the non-short branch that actually consumes
it. This path remains the fallback when the separate dense-MLA arm is off;
when dense MLA is admitted, the earlier no-consumer return skips the top-k
buffer completely.

### Honor `index_kpool_always_select_tail`

The model config sets this True. Nothing reads it — across the whole install the
name appears in the config dataclass, in the assignment that stores it, and in
one docstring on `append_tail_to_topk` claiming it "keeps the (incomplete)
trailing pool so the most recent tokens are always attended to".

That is what the code does, and the incomplete trailing pool is
`[pool_len*4, seq_len)`, which is **empty whenever `seq_len % 4 == 0`**. At those
steps the four most recent tokens are visible only if their just-completed pool
outranks the entire history, and the top-k ranks by relevance, not recency. The
declared guarantee has never held.

The fix raises that pool's logit before the top-k rather than editing its output:
neither `top_k_per_row_decode` nor `persistent_topk` documents the order it
writes indices in, so evicting "the weakest" from the result would be a guess,
while biasing the input lets each kernel drop its own weakest. Written with
gather/scatter rather than `nonzero`, so it adds no host sync and keeps static
shapes under CUDA graph capture. Applied only on the token-granular path — the
`positions is None` fallback reads `decode_metadata.seq_lens`, which is already
pool-granular, and dividing it again would aim the bias at the wrong pool.

**Inert below the top-k budget.** With `index_topk = 2048` and `kpool = 4`, every
pool is selected until ~2048 tokens, so short contexts are unaffected. This is a
long-context correctness fix and does **not** address the Korean-token corruption
tracked in MEASUREMENTS.md, which reproduces at ~1000 tokens.

Not yet exercised by a boot: static checks only (AST, scope of the hoisted
`dec_seq`, no undefined names).

### Route small-SM parts away from `persistent_topk`

Carried forward from the bring-up patch. `persistent_topk` oversubscribes past
~24K context on a 48-SM part and its FilteredTopK fallback wants 128 KB of shared
memory against GB10's 99 KB. The stock guard excludes capability family 120,
which already covers GB10; keying on SM count instead states the actual
constraint and keeps larger family-120 parts on the fast kernel.

### Reuse inactive top-k storage for packed pool ids

The model owns one persistent `int32` top-k buffer shared by all 11 sparse-MLA
layers. With `index_topk=2048` and `index_kpool=4`, each layer previously made
a fresh packed `[rows, 512]` `int32` buffer and then widened it to `int64`
before expansion. The image's CUDA top-k kernels fully write all 512 outputs,
and the fused expand kernel accepts any integer input dtype, so neither the
fresh `-1` fill nor the widening is needed on the GLM CUDA path.

A column slice of the output buffer is deliberately not used: its row stride
is 2176 while the CUDA top-k ABI assumes packed stride 512. Instead, the code
uses a packed view over persistent storage after the active output rows. A
capacity/contiguity/dtype guard falls back to the original temporary buffer.
Because scratch is wholly outside the active prefix, padded decode rows cannot
overwrite prefill results in mixed batches, and the rounded output columns keep
their required `-1` mask.

Base contract from `glm53:v13-b12x`.

### Fused decode tail-select (34차, `VLLM_GLM53_INDEXER_DECODE_FUSED`)

The decode branch of `sparse_attn_indexer_kpool` ran, per full-attention
layer, ten aten launches between the paged-MQA logits and the top-k:
`_decode_topk_seq_lens` (int32 cast + add) and `_force_tail_pool_into_logits`
(int64 cast, floor-div, sub, clamp, gather, full_like, compare, where +
scatter), then one more copy of the expanded indices into the persistent
top-k buffer -- 11 x 11 layers = ~120 graph nodes and ~0.25 ms of a decode
step in the 09-05 production trace (profiler-inflated).

With the knob exactly `1` and a uniform spec-verify layout
(`not requires_padding`), `_glm53_indexer_tail_select_kernel` does the
seq_len and the tail bias in one launch (the same arithmetic, the same
`finfo.max`), and `expand_pools_and_append_tail_into` runs the image's expand
kernel straight into the buffer. Padded batches, prefill, and any layout the
direct write refuses keep the stock chain. Integer index math only:
`probes/indexer_decode_fused_check.py` (via `run_indexer_fused_check.sh`, a
fresh container) is the bit-exact gate and prints the graph-replay time of
both chains. The op logs `[indexer-fused] tail-select fused ... [capture]`
once, the proof that the captured decode graph carries the fused launch.

Measured (srv4, idle fleet, 2026-09-05): 40/40 random and edge-case trials
bit-exact; CUDA-graph replay per layer, 1,250 pools: stock chain 24.7 us ->
fused 5.2 us at 8 rows (C=1), 24.6 -> 6.2 us at 32 rows -- about -0.21 ms per
decode step over the 11 full-attention layers, un-profiled.

---

## glm53_tail_slot_persistent (was `overlay/modules/glm53_kernels/`)

## glm53_tail_slot_persistent

The kpool tail slot mapping is rebuilt every step and read from inside a
captured cudagraph. It was returned as a fresh `clone()`, so the graph recorded
the address of whichever copy existed at capture and every replay since has read
that one.

### Why the original was written that way

`compute_kpool_tail_slot_mapping` says so:

> Pure torch (no Triton, no device sync): the indexer op consumes the tail slot
> mapping **on its eager break**, so the returned tensor need not be the
> persistent `BlockTables` buffer.

The eager break does not happen here. `eager_break_during_capture` returns the
function unchanged when `VLLM_USE_BREAKABLE_CUDAGRAPH` is off — it is, `=0` —
and even when on it steps aside under `CUDAGraphMode.FULL`:

```python
if not is_breakable_cudagraph_enabled():
    return fn
...
if mode == CUDAGraphMode.FULL:
    return fn(*args, **kwargs)
```

`FULL_DECODE_ONLY` is what this model gets, because the sparse indexer backend
reports `AttentionCGSupport.UNIFORM_BATCH` and vLLM downgrades to it. So the
indexer is captured, and `KpoolTailMetadataBuilder` keeps allocating tensors
nothing reads.

### Why this is the shape of the Korean corruption

The tail holds the most recent tokens — the ones not yet folded into a pool. A
frozen slot mapping sends their writes to capture-time slots, so a decode step
can miss the byte it just emitted:

| observation | this mechanism |
|---|---|
| decode corrupts, prefill never does | prefill runs eager; only decode replays a graph |
| exactly one single-byte glue token lost | the tail is precisely the not-yet-pooled recent tokens |
| rare, ~1 per 1000 tokens | the tail matters only when `seq_len % kpool != 0`, and stale slots sometimes coincide |
| non-deterministic at temperature 0 | depends on which descriptor replays and what capture saw |
| worse with speculation | eight tokens per step instead of one, so eight times the tail traffic |
| the replay address assertion never fired | it inspects positional args; attention metadata arrives via the forward context |
| unmoved by `MOE_CUTOVER`, fp8 KV, MoE backend | different subsystem entirely |

### The fix

Write through a caller-owned buffer the builder allocates once, so the address
the graph baked in stays valid while the contents move. This is what the
decorator asks of anything it wraps:

> **In-place output buffer required.** Decorated ops must write into a
> caller-provided output tensor; a fresh tensor returned by `fn` would change
> address each replay and break downstream graph segments.

Allocated lazily on first build, when dtype and length are known — converting a
pre-made buffer with `.to()` would hand back a new tensor and reintroduce the
bug. It grows if a later step needs more room; that reallocation moves the
address, so it must happen before capture, which it does since capture runs the
largest descriptors first.

`out=None` keeps the old clone behaviour for any caller outside a graph.

Base contract from `glm53:v13-b12x`. **Not yet confirmed by measurement** — the
mechanism explains every observation, but the corruption rate after this change
has not been measured.

---

## glm53_sm121_mla_prefill (was `overlay/modules/glm53_kernels/`; 34차 §8 일몰 — 기록)

## glm53_sm121_mla_prefill

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

The `glm53_kpool_tail_select` module also recognizes this request-level
decision after completing the index-K/tail-cache writes and skips the unused
`[tokens, 2048]` top-k buffer fill/mask. Both modules are required for that
scoring bypass; with this dense-prefill arm disabled, the kpool change is
inert.

### Gate and rollback

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

### Deliberate exclusions and validation

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

---

## glm53_kda_prefill_regime (was `overlay/modules/glm53_kernels/`)

### Pure-prefill direct output (2026-09-06, default off)

`VLLM_GLM53_KDA_PREFILL_DIRECT_OUT=1` lets the existing final KDA output
kernel write into the layer's `core_attn_out` prefix. Previously it wrote to
the contiguous V scratch and the model copied the result into that buffer.
The arithmetic, final recurrent state, state scatter, and gated RMSNorm are
unchanged. The model admits BF16 pure-prefill batches only; mixed/spec/plain
decode keep their current routes. The destination must have matching shape,
dtype and CUDA device, be contiguous, and share no storage with any input.
Invalid explicit `out` fails before launching, since a silent fallback would
cause the caller to skip a necessary copy.

For TP4's `[1,8192,16,128]` BF16 output this removes a 32 MiB copy per KDA
layer, or 1.0625 GiB of copied payload across 34 layers per rank and full
8K chunk. This is source-derived traffic accounting, **not a measured speedup**.
The input V contiguous copy is still required by the preceding kernels.

Validation in an idle fleet window (fresh containers, no serving workers or
other GPU tenants; the runner does not acquire a fleet reservation):

```bash
bash probes/run_mk_probe.sh probes/kda_prefill_direct_out.py --order long-first \
  | tee /tmp/kda-direct-long-first.log
bash probes/run_mk_probe.sh probes/kda_prefill_direct_out.py --order short-first \
  | tee /tmp/kda-direct-short-first.log
```

Leave `PROBE_CACHE` unset so each run starts with fresh caches. The probe
checks bitwise output/final-state equality, input integrity, padded output
sentinels and poisoned-output graph replay. It includes chunk boundaries,
uneven packed sequences, contiguous V and both QKV split layouts. It records
source hashes, library versions, first prewarm shape and autotuner selections.
ABBA eager timing excludes input reset; graph timing includes reset only when
stock actually overwrites V (including singleton strided fixtures).

Before promotion, additionally compare 2K/32K/128K onepass prefill and the
existing quality/Korean/acceptance controls at explicit knob values 0 and 1.
Arm on the launcher as a caller variable, not through `EXTRA_ENV`:
`VLLM_GLM53_KDA_PREFILL_DIRECT_OUT=1 bash launchers/start-glm53-nvfp4-tp4.sh`.
Rollback is the same caller variable set to 0. The import-time arming message
alone does not prove this lane served: confirm a pure-prefill trace loses the
KDA output-merge copies. GPU and service measurements remain pending.

## glm53_kda_prefill_regime

Default-off, two-bucket Triton autotune-cache split for the generic
flash-linear-attention KDA chunk path that GLM-5.3 actually imports.

The live `glm53:v13-b12x` model layer imports
`chunk_kda_with_fused_gate` from
`vllm.third_party.flash_linear_attention.ops.kda`; it does **not** call the
Kimi-K3 vendored KDA fork. This module therefore replaces the generic FLA
`kda.py` plus its external `chunk_delta_h.py` dependency only.

### Contract and bucket

The arm requires the exact string `VLLM_GLM53_KDA_PREFILL_REGIME=1` and all of
the following live contracts:

- compute capability 12.1, model type `glm5_next_text`, TP=4,
  `max_num_batched_tokens=8192`;
- packed/varlen chunk KDA with B=1, H=16, K=V=128;
- bf16 q/k/v/raw-g, the production fp32 sigmoid-beta and fp32 initial state;
- bounded gate (`safe_gate=True`, `lower_bound=-5.0`) and
  `output_final_state=True`.

Any mismatch returns regime `0`, the unsplit stock/short config domain. On an
admitted call, the model/config/device portion of the contract is latched once while
vLLM constructs the GLM KDA layer (the only interval in which vLLM exposes its
current config). Forward dispatch reads that latch and computes the bucket once
from tensor and `cu_seqlens` shapes:

```text
regime = int(total_T >= 1024 * (cu_seqlens.numel() - 1))
```

Thus packed average sequence length below 1024 stays in bucket 0 and long
prefill uses bucket 1. No `cu_seqlens` value is read and raw T is never an
autotune key, so each existing shape key has at most two config-cache entries.
The 1024 boundary is an unmeasured hypothesis; the profile remains off until
an engine-down bracket shows a repeatable prefill win.

Arm a bracket from the head checkout with the profile knob as a caller env:

```bash
VLLM_GLM53_KDA_PREFILL_REGIME=1 \
  launchers/start-glm53-nvfp4-tp4.sh
```

Do not pass this profile-owned key through `EXTRA_ENV`: the launcher's later
profile env block would otherwise emit the default `0` after it. A direct
caller value is preserved across profile loading and reaches every rank as
`1`.

### Core-six ownership

One call-level scalar is threaded through exactly the production chunk
Autotuners:

1. KKT inter-subchunk,
2. KKT intra-subchunk,
3. W/U recompute,
4. gated-delta state propagation (`chunk_delta_h.py`),
5. GLA output, and
6. bounded gate + chunk cumsum.

`AUTOTUNE_REGIME` is a regular runtime scalar, appears in each autotune key,
and is explicitly listed in `do_not_specialize`. It therefore splits config
selection without creating separate kernel-code specializations. Low-level
wrappers default to `0`; only `chunk_kda_with_fused_gate_fwd` derives the
request bucket, once, and forwards the identical value to all six.

The decode path uses `fused_recurrent_kda`, not these chunk Autotuners. This
module intentionally leaves fused recurrent, l2norm, solve-tril and the
standalone non-cumulative gate kernel unchanged.

### Validation and rollback

Regime-selection rollback is the env value alone; malformed values, `0`, and
unset all retain regime 0 and skip even the model-init capability/config
lookup. The mounted source still has the added unused scalar ABI and therefore
is not byte-for-byte stock; full rollback removes this module from the profile
and recomposes the overlay. Before adoption, run `probes/kda_prefill_bench.py`
in a fresh GPU container. The first admitted long request creates a second
autotune entry for each core kernel (including the 24-config KKT-inter and
36-config GLA-output sweeps), so prewarm both short and long shapes before
timing. Use matched fresh-cache arms in both `short -> long` and
`long -> short` order; do not let the first 75K user request absorb the tune
sweep. Then bracket a 75K prompt and shorter/multi-sequence controls with
retrieval, Korean-corruption and output/state comparisons.

The standalone probe imports the base image's production entry and directly
sweeps its config inventory. It does not construct the GLM layer or prove that
this overlay's init latch and 0/1 dispatch arm in service; that proof belongs to
the full-engine bracket above.

Live preimages from srv4 `glm53-worker`:

- `kda.py`: `ac2260c84a36936ad7d56ef63dbceb4618b2c499d7637e08b407f0cd706f9d02`
- `chunk_delta_h.py`: `1b3ad391f939d9443c6b7adb19e57fe381bd5dccea064e8417a4f85b0e713b26`

No GPU kernel was launched while preparing this overlay.

---

## glm53_mhc_tilelang (was `overlay/modules/glm53_kernels/`)

## glm53_mhc_tilelang

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

### Knob

`VLLM_GLM53_MHC_SMALLM="tile_n,n_splits"` — e.g. `6,4`. Read once at import
(capture-safe frozen constant); the per-call validator re-checks the kernel's
shape contracts and falls back to stock on ANY doubt:

- `tile_n` must divide `n_out = hc_mult*(hc_mult+2)` (= 24 here)
- `n_splits` ∈ {1,2,4,8} (the dispatcher's own assert)
- `hidden_size % n_splits == 0` and `(hidden_size//n_splits) % 256 == 0`
  (default n_thr=256; a non-exact h-loop silently drops elements)

Unset (the profile default) = byte-identical stock behavior. The value gets
set only after `probes/run_mhc_glm53_bench.sh` finds a winner on real shapes
and a bracket boot confirms it (quality 9/9 + Korean 0/16 + C=1 step/s, the
standard gates). The wrapper mounts the composed sources into a fresh image;
running the Python file directly is deliberately rejected.

### Prefill big_fuse retune — `VLLM_GLM53_MHC_BIGFUSE` (default off)

dsv4 precedents ported here (`dsv4_mhc_tilelang` R2+R3):

- **R3** — big_fuse `h_blk=4096` (a single pipelined block instead of the
  stock gcd(1024, hidden)) won +5.6% at M=4096, while M≤64 was stock-optimal.
  Both stock big_fuse kernels take an optional `h_blk` (default = the stock
  gcd); the dispatcher passes the value only when `num_tokens > 64`.
- **R2** — `mhc_post` prefill `(n_thr 512, h_blk 4096)` beat its stock
  `(128, 1024)` by +3.6% at M=4096. The second env field feeds it.

Format: `VLLM_GLM53_MHC_BIGFUSE="h_blk[,post_thr]"` — e.g. `"4096"` or
`"4096,512"`. Decode is never touched (the M>64 gate is structural).
`probes/run_mhc_glm53_bench.sh --prefill` sweeps the big_fuse side on real
shapes. Adopt via bracket + the standard gates; numerics class is bf16
reduce-order (R3 measured layer_input rel 1.2e-3).

### One-pass decode kernel — `VLLM_GLM53_MHC_ONEPASS` (34차 §8 일몰 — 기록; mk_mhc 가 쌍을 대체)

`tilelang_kernels.py` is also taken over now (preimage `03aeb3f7…`): the stock
kernels are untouched, and one new kernel is appended — `mhc_onepass_tilelang`,
the small-M pair (`mhc_fused` FMA → `big_fuse_with_norm` mixes/sinkhorn/norm)
folded into ONE launch. Grid = one CTA per token; the single tile spans all
`n_out=24` and `split_k=1`, so the `gemm_out_mul/sqrsum` intermediates and
their global roundtrip disappear — −45 launches/step across the 45 layers
(the largest single slice of the #108 §4 axis).

The math is a line-by-line transcription of the two stock kernels;
`tests/test_mhc_onepass_math` proves formula equivalence bitwise in pure
python, and `probes/run_mhc_glm53_bench.sh --onepass` is the GPU validation
harness (rel ≤ 1e-4 vs the stock pair, plus timing). It proves both imported
source hashes and that ONEPASS was frozen on before the eager MHC package
import. The gate stays CLOSED — no boot serves through this kernel — until
that probe runs clean in an engine-down window and a bracket adopts it.

#### Small-M split ownership

The optional ONEPASS return must not own the fallback for the dispatcher.
When ONEPASS is off, `num_tokens <= 16` keeps the small-M kernel's supported
`n_splits` value (4 or 8); only the non-small path may call DeepGEMM's generic
`compute_num_split`. On GB10 with GLM's `hc=4, hidden=4096`, that generic
planner returns 48, which is outside this dispatcher's supported split set and
leaves zero complete 256-thread hidden iterations. `tests/test_logic.py` locks
the control-flow ownership and this concrete SM121a shape.

### Not in this takeover

- The stock `mhc_pre_big_fuse*` / `mhc_fused` / `mhc_post` kernels are
  byte-identical to the image; only the appended onepass kernel is new.
- The tf32 prenorm `compute_num_split` path (prefill, ≤16-token decode never
  reaches it): dsv4's P2b already rejected prenorm n_splits sweeps on this
  hardware family; not re-opened.

### Recovering the preimage

```bash
## on srv4 -- never docker-exec CUDA into the serving container; create+cp runs nothing
ssh srv4 'docker rm -f tmp-src 2>/dev/null; \
  docker create --name tmp-src glm53:v13-b12x true >/dev/null; \
  docker cp tmp-src:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py /tmp/glm53_tilelang.py; \
  sha256sum /tmp/glm53_tilelang.py; docker rm tmp-src >/dev/null'
## the hash must equal the manifest's third column
scp srv4:/tmp/glm53_tilelang.py .
```

### TileLang pass-config re-enable — `VLLM_GLM53_MHC_PASSES` (default off, 2026-09-04)

The image's `tilelang_kernels.py` disables BOTH `TL_DISABLE_TMA_LOWER` and
`TL_DISABLE_WARP_SPECIALIZED` for every mhc kernel with no recorded reason,
while GB10 does have TMA (the deep_gemm sm120 impls use CUtensorMap loads).
`VLLM_GLM53_MHC_PASSES` is the offline A/B for that choice:

- `"tma"` / `"ws"` / `"tma,ws"` flip the matching `TL_DISABLE_*` to False for
  EVERY kernel in the module (stock pair + onepass) at import; `"none"` is the
  explicit stock combo. Unset or unparseable = byte-identical stock dict.
- The value compiles in at import (like the sibling knobs); an armed boot logs
  `[deneb] VLLM_GLM53_MHC_PASSES=... TL_DISABLE_...=...` for engine-confirmed
  verification.
- `probes/run_mhc_glm53_bench.sh --passes` loops the four combos, each in its
  own container (pass configs freeze at import), with the stock combo saving
  the numerics reference and the others gated at rel ≤ 1e-4 against it.
- Contrast PDL: `ENABLE_PDL` is dead here on purpose — the image seals PDL on
  SM12x ("unvalidated, races on KDA state kernels", `platforms/cuda.py`), so
  that axis is closed, not merely untried.
