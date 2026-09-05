# glm53_runtime

GLM-5.3 런타임 — 디코드 준비 통합(prep-fused), 샘플러 가드, 부팅 스탬프, 개발 랩, one-shot AR 배선.

2026-09-05 (34차, 운영자 "디폴트화된 모듈들을 4~5개씩 하나로 묶어라") 에 아래 모듈들을 이 디렉터리 하나로 합쳤다. **매니페스트 행·베이스 계약·소스 파일·노브·기본값은 그대로**이고 디렉터리와 `manifest.tsv`·`requires`·README 만 합쳐졌다(합성 결과 `build/glm53/` 의 파일은 바이트 동일(메가커널 .cu 주석의 경로 한 줄 제외), 행 순서만 바뀜). 옛 이름은 원장·런북·커밋에 그대로 남아 있고, 아래 절이 옛 모듈 하나씩이다.

| 옛 모듈 | 파일 | 무엇 |
|---|---|---|
| `glm53_prep_fused` | `glm53_prep_fused.py` | 디코드 입력 준비 통합 (`PREP_FUSED`, EXP-7, 기본 on) |
| `glm53_v2_sampler_guards` | `sampler.py` | V2 샘플러 가드 |
| `glm53_boot_stamps` | `deneb_boot_stamps.py`, `zz_deneb_boot_stamps.pth` | 부팅 단계 타이밍 스탬프 |
| `glm53_dev_lab` | `glm53_dev_lab.py`, `glm53_lab_middleware.py` | 개발 랩 `/glm53/lab` (`DEV_LAB`, 개발 부팅 전용) |
| `glm53_oneshot_wiring` | `cuda_communicator.py` | one-shot AR 의 glm53 이미지 배선 (`cuda_communicator.py`) |

---

## glm53_prep_fused (was `overlay/modules/glm53_runtime/`)

## glm53_prep_fused

The decode step's host-side input preparation, replaced by one Triton launch.

### What the trace showed (2026-09-01, rank 3, 229 steps)

The V2 runner prepares each step between the drafter graph and the target
graph: `prepare_inputs`, `prepare_attn`, and `model_state.prepare_attn` over
the seven KV-cache groups (MLA+indexer, kpool tail, four mamba groups, the
drafter). Per step that is ~1,000 aten calls, ~45 memcpys and ~100 kernel
launches of 1-3 us each, and with DFlash the scheduler is synchronous, so the
GPU idles through all of it.

| profiled step (72.1 ms) | idle |
|---|---|
| eager prep between the two graphs (6.6-12.3 ms) | ~5.7 ms |
| `cudaGraphLaunch` of the 1,640-node target graph | 1.43 ms |
| DtoH wait before the drafter graph, step turnaround | ~1.4 ms |
| total | 8.9 ms (12%) |

nvidia-smi without the profiler puts decode at 93% busy, i.e. 4-5 ms of a
66 ms step. The prep region is the largest part of it. Ceiling by the ledger's
rule: the region itself, 4-6% of the step; nothing in it reads bytes.

### What this module does

For the steady-state decode step -- every request in spec-verify with the
full draft, FULL cudagraph dispatched, no request padding -- the patched
`prepare_inputs` issues:

1. one pinned H2D copy of `idx_mapping`,
2. one Triton launch (`_glm53_prep_fused_kernel`, grid = requests + 1 + groups)
   that writes every persistent buffer the captured graph reads:
   input ids (last sampled + drafts), positions, seq_lens, query_start_loc,
   expanded idx mapping, block-table gathers and slot mappings for all groups,
   the four GDN builders' FULL-graph buffers, the sparse-MLA req ids, and the
   indexer's expanded block table / compressed context lengths / translated
   table / compressed slot mapping, plus every tail the stock path re-pads,
3. the deep_gemm `get_paged_mqa_logits_metadata` call and its copy.

`prepare_attn` then returns the same views the stock path returns without
launching, and `model_state.prepare_attn` serves the attention-metadata dict
from a per-shape cache (it is only handed to the speculator, and the plan
refuses any speculator but DFlash, which ignores it; under FULL replay the
graph reads buffers, not Python objects). Anything outside that shape --
prefill, partial drafts, padded graphs, adaptive verification, DCP/PCP/PP/LoRA
-- takes the stock path untouched.

Live handles. The kernel's inputs that the runner re-points every step --
`num_blocks.gpu` and `prefill_len.gpu` are UvaBackedTensors whose `.gpu`
rotates through a round-robin pool on every `apply_staged_writes` /
admission, and the block-table pointer tensors are re-made on a KV-cache
wake-up -- are read off the owning objects at every launch, exactly as the
stock kernels do. The idx_mapping staging goes through the image's own
`UvaBufferPool` (sized to the in-flight batch count), so an async-scheduling
boot cannot overwrite a buffer the previous step's copy still reads.

The drafter's KV-cache group carries a target-side FlashInfer builder that
nothing reads (the drafter builds its own metadata from its own builders);
the plan identifies that group by its layer names and only gathers its block
table and slot mapping.

### Contract: bit-exact with the stock buffers

`VLLM_GLM53_PREP_FUSED=shadow` runs the fused path first, then the WHOLE
stock chain (prepare_inputs, the gather/slot kernels, the seven builders)
over the same buffers, diffs every buffer above plus the InputBatch fields
and the kpool tail builder's dormancy, and -- when clean -- lets the fused
batch drive the rest of the step, so the armed control flow (view-based
InputBatch through sampler, rejection sampler and drafter, the metadata
cache) is exercised under shadow too. On drift the stock batch is used.
`[prep-fused] shadow: ... drift=0` over a full bench boot is the gate before
`=1`. Armed mode repeats the same verification every
`VLLM_GLM53_PREP_FUSED_SELFCHECK_EVERY` fused steps (default 64, 0 disables)
and DISARMs for the rest of the boot on the first drift.

Offline first (a fresh container of the PROFILE image, never the serving
one; the wrapper verifies every mounted row's base preimage against the image
the way the launcher does, so srv4's `glm53:sm121-fi618` -- five mounted files
differ -- is refused; use srv2 with `glm53:v13-b12x`):

```bash
bash probes/run_prep_fused_check.sh --trials 60
```

runs the stock building blocks (input_batch kernels, BlockTables gather /
slot kernels, the GDN copies, the indexer's uniform-decode kernel and
compressed slot mapping) and the fused kernel on randomized batches at the
production geometry and requires every tensor bit-exact.

The probe drives the request state through the image's own UvaBackedTensor
and BlockTables objects and rotates their `.gpu` handles between the two
arms, so a plan that snapshots a handle instead of reading it live fails the
diff (the first version of this module did exactly that).

2026-09-02, srv4, `glm53:sm121-fi618` (a hybrid-base run the wrapper now
refuses; superseded by the srv2 run on the profile image recorded in
MEASUREMENTS.md): 60/60 randomized batches (C=1..4), 46 tensors bit-exact.
Per C=1 step the stock building blocks take 2,556 us of wall time (~30
kernels + ~8 memcpys, host-bound); the fused path 201 us (3 kernels + 1
memcpy). Those are HOST issue times: a CUDA-graph replay and CUPTI put the
fused kernel at ~23 us of GPU time and the whole fused launch at ~27-40 us
(the "88 us / 64 us GPU" of the first ledger entry were the Triton launcher
and the deep_gemm wrapper on the Grace CPU). The stock column excludes the
runner's ~1,000 aten calls around those blocks, so serving saves more than
the difference; the ceiling is the trace's prep region. Serving numbers:
none yet (EXP-7).

### Guards

The installer pins the sha256 of every runner and builder file whose control
flow it bypasses (as shipped in `glm53:v13-b12x`; `mla/indexer.py` is the
`glm53_tail_slot_persistent` copy). Any drift logs `preimage drift -> DISARM`
and leaves the runner stock. The plan is built on the first eligible decode
step from the live runner and refuses (stock for the boot, logged) on any
geometry it was not read against: mamba cache mode other than `none`, an
indexer builder off the flattened uniform-decode path, unknown builders, a
draft width that is not `decode_query_len - 1`.

### What was noticed but not changed

The runner's `MambaHybridModelState.prepare_attn` never passes `positions`
to the builders, so `KpoolTailMetadataBuilder`'s circular tail mapping (and
`glm53_tail_slot_persistent`'s persistent buffer for it) is dormant: the tail
group serves the generic per-group slot mapping, which its own docstring
says collapses concurrent requests onto tail block 0 for `pos >= kpool`.
This module reproduces the generic mapping bit for bit and GUARDS the
assumption: the tail builder's circular buffer must stay unallocated (it is
allocated only when positions arrive), checked at plan build and on every
verification pass; if it ever appears the boot DISARMs. Enabling the
circular mapping is a numerics change for C >= 2 and belongs to its own
bracket.

### Arming

```
VLLM_GLM53_PREP_FUSED=0        # default: inert; any other unknown value also DISARMs, loudly
VLLM_GLM53_PREP_FUSED=shadow   # bench boot: fused + stock chain, diff logged, fused batch served when clean
VLLM_GLM53_PREP_FUSED=1        # armed after a clean shadow + EXP-7 bracket
VLLM_GLM53_PREP_FUSED_SHADOW_EVERY=16      # shadow: verify every Nth fused step (default 1)
VLLM_GLM53_PREP_FUSED_SELFCHECK_EVERY=64   # armed: verify every Nth fused step, DISARM on drift (0 = off)

VLLM_GLM53_PREP_FUSED=shadow bash launchers/start-glm53-nvfp4-tp4.sh   # caller env: the key is profile-declared, EXTRA_ENV is refused
```

Rollback is the env line. Base contract from `glm53:v13-b12x`.

---

## glm53_v2_sampler_guards (was `overlay/modules/glm53_runtime/`)

## glm53_v2_sampler_guards

Keeps V2 sampler-side request guards on the logits-processing path.

`Sampler.apply_sampling_params()` returns the input logits unchanged when
`_requires_logits_processing()` says that the active requests need no
processing. The predicate covered logit bias, penalties, bad words,
temperature, min-p, top-k, and top-p, but omitted `ThinkingBudgetState`.
Consequently, a request whose only active processor was a thinking-token
budget skipped `ThinkingBudgetState.apply()` entirely and did not force the
configured reasoning end marker when its budget was reached.

The guard now treats any active per-request thinking budget as requiring logits
processing. The `enabled` check is required because `ThinkingBudgetState`
intentionally does not allocate `use_thinking_budget` when the model has no
complete reasoning-token configuration.

Base contract from `glm53:v13-b12x`. This is a whole-file replacement for the
V2 model runner sampler, not the older `vllm/v1/sample/rejection_sampler.py`
path, so the manifest pin must be refreshed on an image bump.

---

## glm53_boot_stamps (was `overlay/modules/glm53_runtime/`)

## glm53_boot_stamps

Boot phase timing for the serving engine. Additive: two new files, no image
file replaced.

### Why

The 2026-09-02 boot reported one number for the whole middle of the boot --
`init engine (profile, create kv cache, warmup model) took 192.52 s` -- and
the only way to split it was to read the timestamps of whatever else
happened to log inside the window:

| what | measured |
|---|---|
| osar build + connect | 60 s |
| TileLang MHC pair (first) | 5 s |
| megakernel compile + arm | 43 s |
| **no marker at all** | **42 s** |
| torch.compile (2 ranges) | 10 s |
| cudagraph memory profiling | 12 s |
| **no marker at all** | **15 s** |
| cudagraph capture | 2 s |
| **no marker at all (tail)** | **6 s** |

The three compile items are cached since 2026-09-03 (MEASUREMENTS 13차,
14차), so the window should now be ~90 s, of which **63 s has no marker in
it**. That is the largest unattributed boot cost this repo has, and the
switch-fan investigation is the standing reminder of what guessing at an
unmeasured number costs.

### What it does

A `.pth` file imports `deneb_boot_stamps` at interpreter start -- the image
already ships `/usr/lib/python3.12/sitecustomize.py`, which must not be
shadowed, so a `.pth` is the additive way in. An audit hook on `import`
installs timing wrappers as soon as the vLLM worker modules land in
`sys.modules`, around:

- `Worker.load_model`
- `Worker.determine_available_memory` and `GPUModelRunner.profile_run`
- `Worker.initialize_from_config` (KV cache allocation)
- `Worker.compile_or_warm_up_model` and `GPUModelRunner.capture_model`

Each wrapper calls the original, returns its value, and logs
`[boot-stamp] <phase> took X.Xs (at Y.Ys since interpreter start)` to
stderr, which the supervisor already captures into `glm53.log`.

It measures and nothing else. Any failure to install leaves the originals in
place and logs one line. `DENEB_BOOT_STAMPS=0` disables it.

### Reading the output

The absolute offsets are what close the gaps: two consecutive stamps whose
`at` values differ by more than their own durations mean the time went
somewhere neither of them covers, and that is the next thing to name.

---

## glm53_dev_lab (was `overlay/modules/glm53_runtime/`)

## glm53_dev_lab — 부팅 없는 커널 반복 루프 (32차 item 5)

`VLLM_GLM53_DEV_LAB=1` 로 부팅한 **개발용** 플릿에서:

```
curl -s -X POST http://10.10.10.2:8000/glm53/lab -H 'content-type: application/json' -d '{"op":"info"}'
curl ... -d '{"op":"replay","args":{"n":50}}'      # 서빙 디코드 그래프 50회 재생, 랭크별 us/step
curl ... -d '{"op":"reload","args":{"src":"/overlays/glm53/glm53_megakernel.cu"}}'   # 새 .cu 로 확장 재빌드 + 셀프테스트
curl ... -d '{"op":"recapture"}'                    # 그래프 재캡처(새 커널이 박힘)
```

루프: `.cu` 수정 → 4 노드의 오버레이 경로로 scp → reload → recapture → replay. 25 분 브래킷 대신 1~2 분.
`replay` 는 마지막 배치의 KV/상태 슬롯을 덮어쓴다 — 트래픽 없는 개발 부팅에서만. 프로덕션 기본값 0.

---

## glm53_oneshot_wiring (was `overlay/modules/glm53_runtime/`)

## glm53_oneshot_wiring

Hooks `CUDACommunicator.all_reduce` so small decode tensors try the
one-shot path first. Whole-file replacement, so the contract is
image-specific: `glm53:v13-b12x`.

The kernel and shim are in `tp_oneshot_ar`.
