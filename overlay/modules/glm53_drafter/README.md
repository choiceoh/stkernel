# glm53_drafter

GLM-5.3 DFlash 드래프터 — fp8 헤드·로더·워밍업·early-fc·준비 캐시, fp8 lm_head.

2026-09-05 (34차, 운영자 "디폴트화된 모듈들을 4~5개씩 하나로 묶어라") 에 아래 모듈들을 이 디렉터리 하나로 합쳤다. **매니페스트 행·베이스 계약·소스 파일·노브·기본값은 그대로**이고 디렉터리와 `manifest.tsv`·`requires`·README 만 합쳐졌다(합성 결과 `build/glm53/` 의 파일은 바이트 동일(메가커널 .cu 주석의 경로 한 줄 제외), 행 순서만 바뀜). 옛 이름은 원장·런북·커밋에 그대로 남아 있고, 아래 절이 옛 모듈 하나씩이다.

| 옛 모듈 | 파일 | 무엇 |
|---|---|---|
| `glm53_dflash2_fp8_head` | `qwen3_dflash2.py` | DFlash2 드래프터 모델 접수 (fp8 헤드, early-fc 소비자) |
| `glm53_dflash_loader_fp8` | `dflash_utils.py` | 드래프터 로더 fp8 (`DFLASH2_FP8_DENSE` 패스 호출) |
| `glm53_dflash_warmup` | `spec_decode_rejection_warmup.py` | DFlash 입력 준비 커널 부팅 워밍업 (`DFLASH_PREP_WARMUP`) |
| `glm53_dflash_early_fc` | `glm53_dflash_early_fc.py` | 드래프터 fc 를 타깃 헤드·샘플러 아래로 (`DFLASH_EARLY_FC`, EXP-15) |
| `glm53_drafter_prep` | `glm53_drafter_prep.py` | FULL 재생 드래프터 스텝의 메타데이터 빌드 생략 (`DRAFTER_PREP`, EXP-24) |
| `fp8_lm_head` | `fp8_lm_head.py` | W8A16 fp8 lm_head (타깃·드래프트 헤드) |

---

## glm53_dflash2_fp8_head (was `overlay/modules/glm53_drafter/`)

## glm53_dflash2_fp8_head

Builds DFlash2's candidate logits through `Fp8HeadLogitsProcessor`, so the
single head GEMM can use the fp8 copy. Everything else in
`get_top_k_tokens` is vLLM's.

The candidate selector also keeps its final rank-256 bilinear edge scores in
FP32. Its codebooks remain BF16, but casting the gathered rows before the
modulation and reduction prevents distinct path scores from being rounded to
the same BF16 value before `_selector_walk` ranks them. The score tensor is
only `B x 7 x 16 x 16` for this profile, and the walk already consumes FP32.

Base contract from `glm53:v13-b12x`.

---

## glm53_dflash_loader_fp8 (was `overlay/modules/glm53_drafter/`)

## glm53_dflash_loader_fp8

Calls `build_fp8_lm_head` on the freshly loaded drafter. dflash2 has no
loader of its own, so this is dflash's.

Base contract from `glm53:v13-b12x`.

---

## glm53_dflash_warmup (was `overlay/modules/glm53_drafter/`)

## glm53_dflash_warmup

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

---

## glm53_dflash_early_fc (was `overlay/modules/glm53_drafter/`)

## glm53_dflash_early_fc

The DFlash drafter's `fc` (aux hidden states [tokens, 5 x 4096] -> [tokens,
4096]; 168 MB bf16, 301 us on the MK W4 lane) needs only the target's aux
hidden states, which exist when the target forward returns -- but the stock
speculator computes it inside `propose()`, after the head GEMM, the logits
AllGather and the rejection sampler. That window is not DRAM-bound, so the fc
streams its weights there for free (ceiling ~0.3 ms/step, MEASUREMENTS 27차
census: fc is the only tail GEMM whose inputs are ready that early).

- producer: a wrapper on `GPUModelRunner.execute_model` (installed from the
  GLM model module import, like `glm53_prep_fused`); after the forward it
  cats the aux states and runs the drafter's `fc` on a side stream into a
  persistent buffer, recording an event
- consumer: `DFlash2Qwen3ForCausalLM.combine_hidden_states` (the drafter
  overlay) takes the pending buffer for this step and token count after
  waiting on the event, else runs the stock computation

The consumer runs before `precompute_and_store_context_kv` and the drafter
graph, so the fc's MK launch never overlaps another megakernel launch (the
lane's ticket barrier is one-launch-at-a-time). Numerics identical: same
kernel, same inputs. Knob `VLLM_GLM53_DFLASH_EARLY_FC=1`, default 0; any
producer failure disables it for the boot and logs. Speed only: bracket on
C=1 step/s stacked on the EXP-10 arm (the fc must be on the lane to matter).

---

## glm53_drafter_prep (was `overlay/modules/glm53_drafter/`)

## glm53_drafter_prep

The DFlash drafter's per-step host build of its attention metadata, skipped
when the drafter step is a FULL cudagraph replay. Knob
`VLLM_GLM53_DRAFTER_PREP` = `0` (default, stock) | `time` | `shadow` | `1`.

### What the trace showed (34차, 2026-09-05, production boot, rank 0 and rank 3)

Between the last `reshape_and_cache` of `precompute_and_store_context_kv`
and the first kernel of the drafter graph the GPU idles **420 us on rank 0
and 640 us on rank 3** every decode step (profiler-inflated; the un-profiled
number is what the `time` mode logs). `tools/trace_step_nodes.py --gap`
lists what the host does in that window:

| rel us (rank 0) | event | what |
|---|---|---|
| 52 492 | `Memcpy DtoH (Device -> Pageable)` 3.6 us | `seq_lens.cpu()` |
| 52 502 | `cudaStreamSynchronize` | the same `.cpu()` |
| 52 505 -> 52 793 | (nothing on the GPU) | ~290 us of Python |
| 52 793 | `cudaGraphLaunch` 192 us | the drafter graph |
| 52 923 | `k_oneshot` **245 us** | rank 0 waiting for rank 3's longer gap |

The Python is `DFlashSpeculator.propose` -> `_build_draft_attn_metadata` ->
`build_attn_metadata` -> the FlashInfer builder. The drafter's block
attention is non-causal, so the builder's `all_uses_trtllm` is False and it
reads `common_attn_metadata.seq_lens_cpu` -- a property that copies the GPU
`seq_lens` to pageable host memory (the DtoH + sync above) -- and then
builds metadata objects. `propose` then takes the FULL branch:

```python
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    self.query_cudagraph_manager.run_fullgraph(batch_desc)   # replays; never reads draft_attn_metadata
else:
    self._generate_draft(num_reqs, num_tokens_padded, draft_attn_metadata, ...)
```

The replayed graph reads the GPU buffers `_prepare_dflash_inputs_kernel`
wrote (input ids, positions, seq_lens, query_start_loc) and the runner's
block-table / slot-mapping gathers; the metadata dict is consumed only by
the eager branch. So on the FULL branch the build is pure host cost, and
the sync in front of it turns the Python into GPU idle.

### What this module does

`install_glm53_drafter_prep()` (called from the GLM model wiring, like
`glm53_prep_fused` and `glm53_dflash_early_fc`) wraps
`DFlashSpeculator._build_draft_attn_metadata` once, after verifying the
sha256 of the three image files whose internals it relies on (the DFlash
speculator, its base, the cudagraph manager -- drift means DISARM, stock).

The wrapper decides FULL-ness the way `propose` does -- it asks the
drafter's own cudagraph manager to dispatch `(num_reqs, num_reqs x
num_query_per_req)` and requires a FULL descriptor of exactly the padded
token count it was handed -- and then, by mode:

| mode | build | serves | logs |
|---|---|---|---|
| `time` | stock, every step | stock | `[drafter-prep] stock build host us: n= median= p90= max=` every 256 calls |
| `shadow` | stock, every step | stock | the dict cached for this shape is compared field by field: GPU tensors by identity/shape/dtype/stride, host values by equality with the per-step scalars (`max_seq_len`, `seq_lens_cpu_upper_bound`, ...) counted apart; `[drafter-prep] shadow: full= drift= ...` every 256 |
| `1` | once per shape `(num_reqs, num_reqs_padded, num_tokens_padded, step, causal)` | the cached dict on FULL replays | `[drafter-prep] serving: FULL replay -> draft metadata cached for key=...` once, `tally: served= stock=` every 1024 |

Anything the FULL test rejects -- eager batches, a `query_start_loc_np`
override, DP > 1, profile runs (the manager answers NONE before capture)
-- runs the stock build. Any exception inside the wrapper disables it for
the boot, loudly, and returns the stock build.

Nothing numeric changes: the same captured graph replays the same buffers.
The step gets shorter only by the host time removed, so the judgement is
the C=1 step/s bracket with prefill alongside (RUNBOOK EXP-24); `time`
mode on a production boot gives the honest ceiling first.

### Verification ladder

1. `tests/test_logic.py` -- contracts (knob, preimages, wiring, the FULL
   test and the compare on fakes).
2. A `time` boot: the median build time is the ceiling.
3. A `shadow` boot: `drift=0` over a bench run (per-step scalars may
   differ; anything else is a bug).
4. `1` as one arm of the fusion bundle (EXP-24).

---

## fp8_lm_head (was `overlay/modules/glm53_drafter/`)

## fp8_lm_head

Block-quantized fp8 (W8A16) vocabulary head, for either end of speculative
decoding. The deepgemm variant adopted on this fleet -- not the rowwise
`_scaled_mm` one that was measured and rejected.

`Fp8HeadLogitsProcessor` takes the env name that arms it, because the risk
differs: a bad draft head costs acceptance only, while the target's logits
decide the sampled token and the accept/reject.

- `VLLM_SPEC_FP8_LM_HEAD` -- draft head
- `VLLM_TARGET_LM_HEAD_FP8` -- target head (needs a divergence gate)

New file, so portable across images. DSV4 still carries its own copy inside
`dspark_drafter/dspark_v2.py`; the block here was taken from it verbatim and
verified byte-identical.
