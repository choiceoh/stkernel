# Profiles

A profile names the modules a model loads. Nothing else decides what gets
mounted.

```
MODULES="tp_oneshot_ar spec_fp8_head mla_indexer dsv4_model ..."
```

`launchers/compose-overlays.sh <profile>` renders `build/<profile>/`: the flat
directory and single `manifest.tsv` that the deployer, the launcher preflight
and the 4-node SHA-256 verification already expect. Splitting the repo into
modules did not change anything the fleet sees.

A module may declare what it cannot run without in a `requires` file, and a
profile that omits a requirement aborts the compose rather than failing later
as an ImportError inside a rank.

Two modules may not claim the same source filename, and two rows may not bind
the same container path -- either aborts the compose. That is the check that
keeps a module honest: if a second model needs a different version of a module's
file, the module was never model-agnostic and has to be split, not overridden.

| profile | model | modules | state |
|---|---|---|---|
| `dsv4` | DeepSeek-V4-Flash-0731 | 18 | production |
| `glm53` | GLM-5.3-Flash NVFP4 | 25 | kernel campaign -- boots daily; the megakernel set is its default (ledger 28차 §8) |
| `qwen38` | Qwen3.8-Flash-Next NVFP4 | 1 | bring-up (never composed or deployed) |

`glm53` carries its own modules and can load none of `dsv4`'s: its image
installs to dist-packages rather than the venv site-packages, and one of its
modules targets flashinfer rather than vllm at all. That is what `TARGET_PREFIX`
is for -- the allowlist for container paths describes an image, so it belongs to
the profile. A module binding outside it aborts the compose.

The other direction has one exception, and it is the rule working rather than
bending: `glm53_megakernel` binds two NEW files under a RELATIVE
`vllm/model_executor/layers/`, so it lands wherever a profile's prefix points
and has no preimage to drift. `dsv4` mounts it (2026-09-03, every knob 0). The
model-bound halves that used to sit in the same module -- GLM's kda.py hook --
moved to `glm53_mk_kda_wiring` to make that true, which is what "a module that
is not model-agnostic has to be split, not overridden" looks like when it is
actually applied. The name still says `glm53`; renaming it touches the ledger,
so it waits for a measured win on the second model.

`qwen38` stays at one module: it ran on stock image code, and its b12x path is
closed rather than pending (MEASUREMENTS.md).

A profile also carries the serving knobs that are the model's rather than the
fleet's -- backend, speculative depth, draft placement -- and, where a bring-up
is blocked, says so and names the one flip that would isolate the cause.

## 프로필별 구성

| | `dsv4` | `glm53` | `qwen38` |
|---|---|---|---|
| 상태 | production | 커널 캠페인 대상 · 매일 부팅 | bring-up |
| 이미지 | `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6` | `glm53:v13-b12x`(서빙은 `-it` 태그, 4노드 ID 일치 요구) | 미고정 |
| 패키지 루트 | `site-packages` | `dist-packages` | 기본값 |
| 모듈 수 | 18 | **25** | 1 |
| 오버레이 파일 | 23 | **36** | 2 |
| 기본 노브 | 노브 전부 off 가 기준선 | **메가커널 세트**(`MEGAKERNEL`·`MK_MHC`·`MK_GEMM`·`MK_MLA`=1, `MK_KDA`=0) + 드래프터 W4 (28차 §8) + `MK_PDL`(27차 프로브, PR #290 — 종단 수치는 아직 없다) | — |

`glm53` 의 기본값이 곧 브래킷된 cand 구성이다 — 그래서 A/B 의 base 팔은
`VLLM_GLM53_MEGAKERNEL=0` 을 **명시**한다(`launchers/ab-glm53.sh`). 명시하지 않으면
base 가 조용히 메가커널 세트가 된다. 채택 게이트는 부팅마다 품질 9/9 · 한국어 0/16 ·
pos-1 수용률 ±2 pct 이고, 그리디 텍스트 diff 는 부팅 간 재현되지 않으므로 판정에 쓰지
않는다(28차 §8).

`glm53_drop_audit`와 `glm53_sparse_q`는 V1 Model Runner 파일만 교체한다.
`glm53:v13-b12x`는 V2 Model Runner를 사용하므로 이 둘과
`VLLM_SPEC_GATHER_Q`는 프로필에서 제외한다. 설정이 켜졌지만 실제
DFlash2 경로에는 전혀 적용되지 않는 상태를 정상 구성으로 취급하지 않는다.
대신 `glm53_v2_sampler_guards`가 thinking budget만 활성인 요청도 logits
처리 경로를 반드시 통과시킨다.

### 모듈 × 프로필

범위가 모델을 넘는 것을 위에, 한 모델에 묶인 것을 아래에 둔다. **이식 가능**은 신규 파일이면서 타깃이 상대경로인 것 — 이미지가 달라도 그대로 실린다. 나머지는 파일 전체 교체라 계약이 이미지에 묶인다.

| 모듈 | 범위 | 파일 | 이식 | dsv4 | glm53 | qwen38 |
|---|---|---:|:---:|:---:|:---:|:---:|
| `moe_gate_sm121` | GB10의 모든 MoE | 1 | ✓ | ● | ● | · |
| `tp_oneshot_ar` | 어느 모델이든 | 2 | ✓ | ● | ● | ● |
| `glm53_megakernel` | sm_121a 디코드 커널 코어 (opt-in; dsv4 는 MK_SEG_MHC 만 해당) | 2 | ✓ | ○ | ○ | · |
| `mla_indexer` | DeepSeek-MLA | 1 | — | ● | · | · |
| `mla_sparse_swa` | DeepSeek-MLA | 1 | — | ● | · | · |
| `spec_fp8_head` | 드래프터 일반 — **기각** | 1 | ✓ | ○ | · | · |
| | | | | | | |
| `glm53_model` | **묶음(34차)**: 모델·어텐션·KDA·MLA 파일 접수 + 밀집 GEMM fp8/W4 패스 + KDA 원패스 (옛 `glm53_model_wiring`·`glm53_indexer_gate_splitk`·`glm53_mk_kda_wiring`·`glm53_mk_mla_wiring`·`glm53_kda_onepass`·`glm53_fp8_dense`) + MTP 블록 FP8 MoE 백엔드 오버레이(37차 `mtp.py`) | 10 | 일부 | · | ● | · |
| `glm53_kernels` | **묶음(34차)**: kpool 인덱서 op·top-k 커널·tail-select 융합, tail 슬롯, SM121 MLA 프리필, KDA 프리필 버킷, MHC TileLang (옛 `glm53_kpool_tail_select`·`glm53_tail_slot_persistent`·`glm53_sm121_mla_prefill`·`glm53_kda_prefill_regime`·`glm53_mhc_tilelang`) | 9 | 일부 | · | ● | · |
| `glm53_drafter` | **묶음(34차)**: DFlash2 드래프터 접수, fp8 로더, 워밍업, early-fc, 준비 캐시, fp8 lm_head (옛 `glm53_dflash2_fp8_head`·`glm53_dflash_loader_fp8`·`glm53_dflash_warmup`·`glm53_dflash_early_fc`·`glm53_drafter_prep`·`fp8_lm_head`) | 6 | 일부 | · | ● | · |
| `glm53_moe` | **묶음(34차)**: b12x 공유 워크스페이스·EP 마이크로커널 레인·직접 출력 (옛 `b12x_shared_workspace`·`b12x_zero_weight_micro`·`glm53_b12x_out`) + 정적(디코드) MoE 커널 v2/v3/v4/v5(35·38·39차, `moe_static_kernel_v2.py`·`moe_static_kernel_v3.py`·`moe_static_kernel_v4.py`·`moe_static_kernel_v5.py`(타일 우선 가중치 `t`), 프로필 기본값 `u` = v4) | 8 | — | · | ● | · |
| `glm53_runtime` | **묶음(34차)**: prep-fused, 드래프터 학습 덤프(37차, `VLLM_GLM53_DRAFT_DUMP` 없으면 비활성), 샘플러 가드, 부팅 스탬프, 개발 랩, one-shot AR 배선 (옛 `glm53_prep_fused`·`glm53_v2_sampler_guards`·`glm53_boot_stamps`·`glm53_dev_lab`·`glm53_oneshot_wiring`) | 8 | 일부 | · | ● | · |
| `deepseek_reasoning` | 모델 전용 | 1 | — | ● | · | · |
| `deepseek_tool_parser` | 모델 전용 | 1 | — | ● | · | · |
| `dspark_drafter` | 모델 전용 | 3 | — | ● | · | · |
| `dsv4_attention` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_eager_scratch` | 모델 전용(신규 파일이라 계약은 이식 가능) | 1 | ✓ | ● | · | · |
| `dsv4_flashinfer_sparse` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_mhc_tilelang` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_model` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_oneshot_wiring` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_ops_cache_utils` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_ops_fused_indexer_q` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_tokenizer` | 모델 전용 | 2 | — | ● | · | · |

매니페스트의 모든 행이 `absent`(=대체할 베이스가 없는 신규 파일)인 모듈은 이제 **다섯 개**다 — `tp_oneshot_ar`, `moe_gate_sm121`, `spec_fp8_head`, `dsv4_eager_scratch`, `glm53_megakernel`. 34차(2026-09-05)에 glm53 전용 모듈 25개를 다섯 묶음(`glm53_model`·`glm53_kernels`·`glm53_drafter`·`glm53_moe`·`glm53_runtime`)으로 접으면서 이식 가능한 행(옛 `fp8_lm_head`·`glm53_fp8_dense`·`glm53_prep_fused`·`glm53_dflash_early_fc`·`glm53_boot_stamps` 등)은 묶음 안에서 이미지 계약 행과 섞였다 — 행 단위 계약은 그대로다(표의 "이식" 열 `일부`). 그래서 이미지가 달라도 계약이 성립한다 — 단 **형식이 이식 가능하다는 것과 내용이 모델 무관이라는 것은 다른 명제다**: `glm53_fp8_dense` 는 GLM 의 선형 이름 패턴에, `glm53_prep_fused` 는 러너의 준비 체인에 묶여 있다. 표의 "이식" 열은 앞의 뜻(계약 형식)이고, "범위" 열이 뒤의 뜻이다. 나머지가 한 이미지에 묶이는 이유는 기능이 특수해서가 아니라 오버레이가 **파일 전체 교체**이기 때문이고, 그래서 `*_wiring`·`glm53_*` 계열이 짝으로 존재한다: 이식 가능한 알맹이와 이미지별 배선.

`spec_fp8_head`는 ○로 표시했다: dsv4에 마운트돼 있지만 `VLLM_DSPARK_FP8_DRAFT_HEAD=0`으로 꺼져 있다. rowwise `_scaled_mm` 판본이고 실측에서 60.6 vs 61.7·수용률 무이동으로 기각됐다(MEASUREMENTS.md:419). 채택된 쪽은 `spec_fp8_lm_head`(deepgemm)이며 dsv4는 아직 `dspark_drafter` 안의 사본을 쓴다.

`glm53_sm121_mla_prefill`의 ○도 마운트되지만 기본은 꺼진다는 뜻이다.
`VLLM_GLM53_SM121_MLA_PREFILL=1`과 SM121·GLM 형상·SM90/HND
`fp8_e4m3` 계약이 모두 맞을 때만 2048 토큰 이하 dense-MHA prefill을
열고, 나머지는 기존 top-k MQA를 유지한다.

`glm53_kda_prefill_regime`도 마운트만 되고 기본은 꺼져 있다. exact
`VLLM_GLM53_KDA_PREFILL_REGIME=1`과 현재 GLM/TP4/SM121/bf16 chunk 계약이
모두 맞을 때만 packed 평균 1024 토큰 이상을 별도 Triton autotune cache로
분리한다. raw T는 key가 아니며 short/long 두 bucket만 허용한다. 기본 0은
레짐 분리를 끄지만 오버레이된 Triton ABI가 달라지므로 최초 stock bucket의
재컴파일·autotune까지 없애는 byte-identical rollback은 아니다.

`qwen38`은 이미지를 고정하지 않았다. 그 브링업은 스톡 이미지에서 돌았고 b12x 경로는 열린 문제가 아니라 닫힌 것이라(MEASUREMENTS.md), 프로필은 기록으로만 있다 — 실제로 합성해 배포한 적은 없다.
