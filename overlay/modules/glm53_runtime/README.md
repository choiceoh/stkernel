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
| `glm53_prefill_sp` (신규, 368차) | `glm53_prefill_collectives.py`, `parallel_state.py` | 순수 프리필 TP4 시퀀스 병렬화 (`PREFILL_SP` 기본 1, `PREFILL_SP_FP8` 기본 3; 4096토큰 미만 BF16) |
| 채팅 옵션 계약 | `glm53_chat.py` | Chat Completions 옵션 검증과 정규화 |
| GLM 본문 보존 | `glm47_moe.py` | none의 리터럴 XML, 도구 호출 사이·뒤의 본문, 인자의 think 태그 보존 |

## Chat Completions 옵션과 템플릿

`start-glm53-nvfp4-tp4.sh`는 `ChatContractMiddleware`를 등록하고, 기존
`glm45` reasoning / `glm47` tool 파서를 사용한다. HTTP 응답과 SSE 청크,
`finish_reason`은 수정하지 않는다. 추론 중 length 종료를 답변으로 변환하지 않는다.

- `chat_template_kwargs.thinking`과 `enable_thinking`은 boolean만 받으며,
  둘을 함께 지정하면 값이 같아야 한다. 명시적인 `false`는 기존 클라이언트의
  `<think></think>` 호환 경로를 유지한다. 생략한 값은 서버 기본값에 맡긴다.
- `reasoning_effort`는 `low`, `high`, `max`를 지원한다. 최상위 필드와
  `chat_template_kwargs` 값이 충돌하거나 지원하지 않는 값을 지정하면 400이다.
  `null`/생략은 기본값이며, 공식 템플릿의 기본 effort는 `max`다.
- `clear_thinking`은 기존 기본값 false를 유지한다. 일반 채팅 클라이언트는
  true를 명시할 수 있다. 이전 사용자 턴의 reasoning만 지우며 현재 도구 호출
  라운드의 reasoning은 유지한다.
- 이력의 reasoning은 `reasoning_content` 필드로 전달한다. 일반 본문의
  `<think>`/`</think>`는 재해석하지 않는다. 과거의 태그 포함 이력을 복원할 때만
  `legacy_reasoning_content=true`를 지정한다. 이때도 본문 맨 앞의 완성된 블록
  하나만 분리하며 뒤의 본문과 태그는 보존한다. 명시적 reasoning 필드가 우선이다.
- `content=null`은 빈 본문으로 렌더링한다. 도구 인자의 JSON 문자열→mapping
  변환은 vLLM의 Chat API 전처리 책임이다.
- `glm53_required_first=true`를 요청 최상위에 명시하면, 렌더링 전에 도구
  스키마의 필수 필드를 `required` 순서로 앞에 배치한다. 기본 false인 실험
  옵션이며 OpenAI SDK에서는 `extra_body={"glm53_required_first": True}`로
  전달한다. 중첩 스키마와 `$defs`도 처리하되 default/enum 값과 입력 객체는
  변경하지 않는다. `tool_choice` 의미와 JSON Schema 검증 규칙은 그대로다.
  [vLLM 초안 #55558](https://github.com/vllm-project/vllm/pull/55558)의 후보이며
  우리 NVFP4에서 필수 인자 누락률 A/B를 확인하기 전에는 기본값으로 올리지 않는다.

이 검증은 `/v1/chat/completions`에 적용된다. Responses API의 별도 입력 규약을
추가로 구현한 것은 아니다. Hugging Face 직접 호출에는 Jinja 검증이 적용된다.

런처는 저장소의 `launchers/chat_template_mm_v2.jinja`를 기준으로
`chat_template.<SHA256>.jinja`를 모든 모델 디렉터리에 원자적으로 배치한다.
기존 이름의 파일은 덮어쓰지 않는다. 전송된 파일과 실제 컨테이너 마운트의
해시 검증이 끝나야 시작을 진행한다. 다른 템플릿을 명시하면 저장소 파일을
우선하고, 없을 때 모델 폴더에서 읽는다. 런처 로그에 이름·해시·파서가 남는다.

검증:

```bash
uv run --with jinja2 python tests/test_glm53_chat.py
bash launchers/check-glm53-chat.sh /home/choiceoh/models/glm53-redhat-nvfp4 glm53:v13-b12x-it
```

두 번째 검사는 이미지의 원본 파서 SHA를 확인하고 후보 파서를 읽기 전용으로
마운트한 CPU 컨테이너에서 실행한다. 실제 `ParserManager`의 glm45/glm47 조합과
직접 엔진의 추론 시작·종료, 미완료 추론, off, none/auto/required/named, 복수
도구 호출, 인자 타입, 본문과 청크 경계, 도구 결과 재입력을 검사한다.
재입력은 실제 `auto` content-format 판별 경로를 사용한다. 프롬프트·응답의
영상/이미지 처리 자체나 강제 생성의 수렴성을 CPU 재생으로 증명하지는 않는다.
`deploy-overlays.sh glm53`도 파일 배포 전에 이 검사를 실행한다.
모델을 재시작하거나 생성 품질을 측정하는 검사는 별도로 수행해야 한다.

파서 교체는 GLM 모듈에 한정한다. `tool_choice=none`에서는 도구 문자열을
소비한 뒤 호출만 숨기는 대신 인식을 건너뛴다. 실제 호출 사이·뒤의 텍스트는
동일 청크에서 반환하며 들여쓰기를 보존한다. 최초 reasoning 블록 이후의
`<think>`/`</think>`는 리터럴로 취급하므로 API reasoning 전처리가 도구 인자를
손상하지 않는다. HTTP/SSE 종료 사유와 미완료 호출의 인자를 복구하거나
`required`/named 요청을 auto로 바꾸는 동작은 추가하지 않는다.

공식 템플릿 기준은 `tests/fixtures/glm53_official_chat_template.json`의 HF
리비전·SHA로 고정한다. 비교 검사는 low/high/max, clear_thinking, 도구 결과,
이미지·영상·오디오 자리표시자를 포함한다. off/legacy 이력/공백 보존처럼 의도한
차이는 별도 회귀 검사로 유지한다. [공식 템플릿 수정 이력](https://huggingface.co/zai-org/GLM-5.3-Flash/discussions/18).

강제 도구 생성의 수렴성은 [vLLM #55541](https://github.com/vllm-project/vllm/issues/55541)
보고와 별도로 실제 모델 검증이 필요하다. 아래 opt-in 프로브는 지정한 엔드포인트에
16개 요청(작은/큰 중첩 스키마 × none/auto/required/named × stream)을 보내며,
도구를 실행하지 않고 JSON Schema·필수 인자·함수명·종료 사유·SSE 완료를 판정한다.
`length`, 끊긴 SSE, 불완전 JSON은 성공으로 세지 않는다. 기본 파서 릴리스 게이트는
이 네트워크 검사를 호출하지 않는다. GPU 캠페인 예약/동일 이미지 조건을 충족한
엔드포인트에서 실행하고 `--required-first` 유무를 비교한다. `OPENAI_API_KEY`는
환경 변수로만 읽으며 결과 JSONL에 응답 본문이나 인증 정보를 기록하지 않는다.

```bash
uv run --with jsonschema python probes/glm53_tool_choice_acceptance.py \
  --url http://HOST:PORT/v1/chat/completions --model glm-5.3-flash
```

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

---

## glm53_prefill_sp

Pure-prefill TP4 sequence parallelism (`VLLM_GLM53_PREFILL_SP=1`, default
gated FP8 transport `VLLM_GLM53_PREFILL_SP_FP8=3`). MHC/residual
work is row-sharded across ranks; full-token attention and MoE see an
all-gathered view; the terminal TP all-reduce is deferred and replaced by a
reduce-scatter (`partial_tp_output` / `maybe_partial_all_reduce` hooks).
Uneven token counts (T=8185 etc.) pad and trim exactly. Guards fail closed
on metadata, topology, dtype, layout, graph capture, and reduction-contract
drift — any miss keeps the untouched stock path; exactly one deferred
terminal reduction per scope is enforced at scope exit. SP became the
profile default after the 39차 +12.6%/+13.3% 32K/128K BF16 prefill result.

FP8 v1 agrees on common block scales before native FP8 SUM;
v2 sends rank-local values and scales in two all-to-alls and sums decoded
terms in FP32. v3 packs values and scales by destination directly into one
all-to-all, preserving v2's numerical recipe. Its packet stride is rounded
to 128 bytes for both collectives; the pack kernel initializes the alignment
gap. BF16 already has one native reduce-scatter. Ungated v3's matched
serving comparison gained 1.7%/3.4% at 32K/128K but lost 9.2% at warm 2K.
Every FP8 mode requires its own serving-quality gate.

v3 now selects BF16 before codec/scratch allocation when the actual scheduled
chunk has fewer than `VLLM_GLM53_PREFILL_SP_FP8_MIN_TOKENS=4096` real global
rows. This is an initial boundary, not a measured optimum. All model gathers
(including auxiliary/final output) supply the real count, so TP padding does
not promote a short chunk to FP8. Standalone gathers without `num_tokens`
use their represented padded row count. The gate has no device-data read or
extra collective; every rank uses the same profile and chunk shape. Set the
minimum to `0` for the ungated v3 control. v1/v2 remain unchanged. The
operator promoted gated v3 to the profile default on 2026-09-07 (PR #425).
Set `VLLM_GLM53_PREFILL_SP_FP8=0` to use BF16 at every length. Long-context
uplift and the optimal threshold remain unresolved; see
`docs/GLM53_PREFILL_40.md` for current serving evidence.

`probes/glm53_prefill_collectives_check.py` (WORLD_SIZE=4 torchrun) covers
all four transports. `probes/run_glm53_prefill_transport_check.sh` checks
v3 packets and sums against v2 on one GPU with a simulated exchange;
`--compile-only` compiles the new kernels on the CPU for SM121. Neither
the compile nor the simulated exchange proves real NCCL speed or quality.

Three independent follow-up controls are available for matched experiments:

* `VLLM_GLM53_PREFILL_SP_FUSE_MHC=1` keeps the v3 receive packet until MHC
  post, where one kernel performs source-ordered FP32 decode/sum, the same
  intermediate BF16 rounding, and the residual mix. The next MHC pre uses
  its existing implementation. Packet and output storage belong to each
  invocation, including repeated auxiliary and next-layer consumers.
* `VLLM_GLM53_PREFILL_SP_FP8_AG_MIN_TOKENS` and
  `VLLM_GLM53_PREFILL_SP_FP8_RS_MIN_TOKENS` independently select the gather
  and reduction crossover. Both default to `-1`, inheriting the shared
  4096 threshold; `0` means ungated. They do not affect v1/v2. Use identical
  settings on all ranks; these are real global chunk rows, not request size.
* `VLLM_GLM53_PREFILL_SP_DIRECT_NCCL=1` exchanges the same v3 byte packets
  with grouped native PyNCCL sends/receives on the current CUDA stream.
  Peer order, including self, remains 0..3. Validation precedes the group;
  a communication error propagates without attempting another collective.

Fusion and direct exchange default to `0` until numerical and serving
evidence supports promotion. Their improvements must not be added to the
earlier percentages or treated as reaching the original 40% target.

Validation in a current fleet turn (same checkout path on all four nodes):

```bash
# CPU compilation is GPU-free; run it through fleet.sh run --cpu.
bash probes/run_glm53_prefill_transport_check.sh --compile-only
# GPU probes require fleet.sh run --gpu --probe.
bash probes/run_glm53_prefill_transport_check.sh --skip-scale --fuse-mhc
bash probes/run_glm53_prefill_tp4_check.sh fp8-v3 \
  --rows 128 129 2128 4095 4096 6143 6144 6912 8185 \
  --ag-min-tokens 6144 --rs-min-tokens 4096 --direct-nccl --fuse-mhc --timing
```

The deliberately different probe thresholds exercise both dispatch choices;
they are not tuned production values. The single-GPU fusion probe compares
bit patterns against the mounted production TileLang MHC post and post/pre
continuation, and checks repeated consumers and nondefault streams. TP4
adds actual exchange plus MHC equality, threshold-edge padding cases, and
mirrored BF16/v2/v3/direct timings using the slowest rank in each sample.
Neither gate substitutes for a clean, matched serving throughput/quality pair.
