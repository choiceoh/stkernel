# Profiles

> 살아 있는 참조 — **프로필이 무슨 모듈을 싣는지. 프로필이 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

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
| `qwen38` | Qwen3.8-Flash-Next NVFP4 | 6 | TEP=4 (TP=4 + EP) bring-up |

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

GLM53's C=1 M6/N6416/K4096 projection defaults to
`VLLM_GLM53_MK_INPUT_CTA=4` (operator promotion, 2026-09-08), which builds on
the value `2` route described here. That route retains the
eight original K slices and sums their partials within one CTA. The independent
startup gate falls back to the existing input-reuse kernel; setting the knob
to `0` selects that previous route explicitly. Repeated kernel measurements
show 24.5% lower warm latency and 7.9% lower read-evicted latency. That
2026-09-08 promotion ran no serving bracket of its own, so its serving
step/output and quality acceptance were left unmeasured after baseline boot
failures; the separate value `4` bracket described below is not evidence for
it. [Measurements and failure receipts](../measurements/glm53_input_cta_20260907/README.md).

The value `4` retains the CTA2 N6416 kernel and adds a three-slice
CTA for foreground M6/N4096 or N6144/K4096. A failed three-slice startup
check falls back to separately validated CTA2. Other shapes, background
work, low-rank correction, and non-three-slice overrides retain their
existing route. The profile default is now `4` (operator promotion,
2026-09-08); set the knob to `2` to roll back to the previous default.
Its N6144 warm kernel latency is 32.544 -> 26.208 us (-19.5%) with -5.7%
read-evicted, and N4096 is -7.9% read-evicted with no warm gain. Racecheck
reported zero hazards and memcheck zero errors, and all four nodes proved
actual CTA4 serving capture. Serving throughput is **not** established: the
same-build CTA2/CTA4/CTA2 bracket had pooled step/s 21.752/21.962/21.892, so
the candidate's +0.627% lies inside the baselines' own 0.641% spread.
[Kernel, sanitizer and serving evidence](../measurements/glm53_input_cta_next_20260908/README.md).

GLM53 selects EP4 MoE with `ENABLE_EP=1`, `VLLM_GLM53_EP_TILED=1` and
`VLLM_GLM53_TP_SF6_Q0=0`, retaining K6 and `VLLM_GLM53_PREP_FUSED=1`.
The canonical pooled decode result was 72.62743 tok/s against the operator's
absolute 67 target; TP measured 80.49311 tok/s. Quality and required execution
proof passed. This is not a relative non-regression or full-warm prefill claim.
[Acceptance scope and explicit TP rollback](../docs/GLM53_EP_TILED.md).

GLM53 MoE defaults to `VLLM_GLM53_B12X_STATIC_V2=t,r,sf6` (operator adoption,
2026-09-09). SF6 losslessly packs scales for direct decode and prefill reads,
then releases eligible raw scale Parameters and their aliases before profiling.
Observed scale storage is 4.42969 GiB raw -> 3.35687 GiB packed per rank;
this 1.07281 GiB difference is tensor accounting, not a matched memory benchmark.
Restart with `t,r` to retain original scales. The SF6 comparison has a different
MHC activation outcome and is not a matched speedup verdict.
[Adoption and retained results](../measurements/glm53_sf6_default_adoption_20260909/README.md).

The underlying `t,r` geometry was promoted on 2026-09-08 in PR #461.
For M<=8 this combines M16 padding, FC1 N128/K256 and
FC2 N256; larger M retains the existing `t` geometry and weight storage.
The corrected bundle passed 13 GPU shapes and 130 numerical/graph comparisons.
The same-build C=1 A-B measured pooled step/s 21.727898 -> 22.076208 (+1.603%)
and output tok/s 70.612170 -> 72.787236 (+3.080%), with quality 18/18 and
Korean corruption 0/8 in both arms. Window median rose 0.116%; one boot per
arm does not independently establish repeatability. Set the knob to `t` to
restore the previous geometry.
[Full evidence](../measurements/glm53_decode_reform_20260908/README.md).

## 프로필별 구성

| | `dsv4` | `glm53` | `qwen38` |
|---|---|---|---|
| 상태 | production | 커널 캠페인 대상 · 매일 부팅 | bring-up |
| 이미지 | `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6` | `glm53:v13-b12x`(서빙은 `-it` 태그, 4노드 ID 일치 요구) | 미고정 |
| 패키지 루트 | `site-packages` | `dist-packages` | 기본값 |
| 모듈 수 | 18 | **9**(34차 묶음 8 + 39차 프리픽스 캐시; 접기 전 25) | 1 |
| 오버레이 파일 | 23 | **54** | 2 |
| 기본 노브 | 노브 전부 off 가 기준선 | **메가커널 세트**(`MEGAKERNEL`·`MK_MHC`·`MK_GEMM`·`MK_MLA`=1, `MK_KDA`=0) + 드래프터 W4 (28차 §8) + `MK_PDL`(27차 프로브, PR #290 — 종단 수치는 아직 없다) | — |

`glm53` 의 기본값이 곧 브래킷된 cand 구성이다 — 그래서 A/B 의 base 팔은
`VLLM_GLM53_MEGAKERNEL=0` 을 **명시**한다(`launchers/ab-glm53.sh`). 명시하지 않으면
base 가 조용히 메가커널 세트가 된다. 채택 게이트는 부팅마다 품질 9/9 · 한국어 0/16 ·
pos-1 수용률 ±2 pct 이고, 그리디 텍스트 diff 는 부팅 간 재현되지 않으므로 판정에 쓰지
않는다(28차 §8).

`glm53_drop_audit`와 `glm53_sparse_q`(V1 Model Runner 파일만 교체하던 고아 모듈)는
34차 §8 에서 삭제했다. `glm53:v13-b12x`는 V2 Model Runner를 사용하므로
`VLLM_SPEC_GATHER_Q`는 프로필에서 제외한다. 설정이 켜졌지만 실제
DFlash2 경로에는 전혀 적용되지 않는 상태를 정상 구성으로 취급하지 않는다.
대신 `glm53_v2_sampler_guards`가 thinking budget만 활성인 요청도 logits
처리 경로를 반드시 통과시킨다.

### 모듈 × 프로필

범위가 모델을 넘는 것을 위에, 한 모델에 묶인 것을 아래에 둔다. **이식 가능**은 신규 파일이면서 타깃이 상대경로인 것 — 이미지가 달라도 그대로 실린다. 나머지는 파일 전체 교체라 계약이 이미지에 묶인다.

| 모듈 | 범위 | 파일 | 이식 | dsv4 | dsv41 | glm53 | qwen38 |
|---|---|---:|:---:|:---:|:---:|:---:|:---:|
| `moe_gate_sm121` | GB10의 모든 MoE | 1 | ✓ | ● | ● | ● | · |
| `tp_oneshot_ar` | 어느 모델이든 | 3 | ✓ | ● | ● | ● | ● |
| `qwen38_moe` | Qwen3.8-Flash-Next 전용 (공유 전문가를 라우팅 grouped GEMM 의 11번째 슬롯으로 융합; 랭크 로컬 센티넬 −2, all-to-all 비참여) | 1 | ✓ | · | · | · | ● |
| `qwen38_ple` | Qwen3.8-Flash-Next 전용 (51 GiB PLE n-gram 표: 호스트 RAM 오프로드 + NVFP4 체크포인트용 온디바이스 FP8 임베딩; TP>1 필수) | 2 | — | · | · | · | ● |
| `qwen38_qsa` | Qwen3.8-Flash-Next 전용 (GB10 48 SM 용 QSA split-K 상한; 상위는 GB300 튜닝) | 1 | — | · | · | · | ● |
| `qwen38_spec` | Qwen3.8-Flash-Next 전용 (MTP 스펙: n-gram 순서 수정, 적응 K) | 2 | — | · | · | · | ● |
| `qwen38_b12x` | Qwen3.8-Flash-Next 전용 (b12x 워크스페이스 용량 바운드 체크 래퍼; IMA 를 숫자 적힌 부등식으로) | 2 | ✓ | · | · | · | ● |

| `sched_decode_first` | 어느 모델이든 (AsyncScheduler 서브클래스; 모델·커널·형상 임포트 0) | 1 | ✓ | · | ● | ● | · |
| `boot_stamps` | 어느 모델이든 (부팅 단계 계측) | 2 | ✓ | · | ● | ● | · |
| `dsv41_vllm` | V4.1 을 vLLM 아키텍처로 등록 (transformers config 타입 + ModelRegistry + DSV4 모델 파생 + bf16 그룹 o-projection + E8M0 스케일 fp32 변환 + mHC 지연 짝짓기; 전부 신규 파일이라 이미지 불필요) | 8 | ✓ | · | ● | · | · |
| `dsv41_model` | DeepSeek-V4.1 전용 (CED 층 계획·후보 인덱서·packed FP4 캐시·전체 텐서 형상·KV 컴프레서·rope·가중치 라우팅·sparse_attn 인덱스 계약·프리샤드 랭크 파일 적재·슬라이딩 윈도 KV 링; 명시적 reference opt-in) | 14 | ✓ | · | ● | · | · |
| `dsv41_engram` | DeepSeek-V4.1 전용 (SSD 룩업표: 설정·I/O·해시·게이트) | 4 | ✓ | · | ● | · | · |
| `dsv41_encoding` | DeepSeek-V4.1 전용 (V4.1 프롬프트 형식 파싱) | 1 | ✓ | · | ● | · | · |
| `glm53_megakernel` | sm_121a 디코드 커널 코어 (opt-in; dsv4 는 MK_SEG_MHC 만 해당) | 2 | ✓ | ○ | · | ○ | · |
| `mla_indexer` | DeepSeek-MLA | 1 | — | ● | · | · | · |
| `mla_sparse_swa` | DeepSeek-MLA (V4.1 의 compress_ratio 2 도 레이어 타입을 얻는다) | 1 | — | ● | ● | · | · |
| `spec_fp8_head` | 드래프터 일반 — **기각** | 1 | ✓ | ○ | · | · | · |
| | | | | | | |
| `glm53_model` | **묶음(34차)**: 모델·어텐션·KDA·MLA 파일 접수 + 밀집 GEMM fp8/W4 패스 + KDA 원패스 + 순수 프리필 SP/NVFP4 후보 + 영상 자리표시 수정(39차) + FP8·랭크별 부팅 캐시 | 14 | 일부 | · | · | ● | · |
| `glm53_kernels` | **묶음(34차)**: kpool 인덱서 op·tail-select 융합, tail 슬롯, MHC TileLang 프리필 big_fuse 오버라이드 + MK 훅 (옛 `glm53_kpool_tail_select`·`glm53_tail_slot_persistent`·`glm53_mhc_tilelang`; 34차 §8 일몰: radix top-k 확장, SM121 MLA 프리필, MHC SMALLM/ONEPASS; KDA 프리필 버킷(`kda.py`·`chunk_delta_h.py`)은 #368 이 direct-out 을 얹어 유지) | 6 | 일부 | · | · | ● | · |
| `glm53_drafter` | **묶음(34차)**: DFlash2 드래프터 접수, fp8 로더, 워밍업, early-fc, 준비 캐시, fp8 lm_head (옛 `glm53_dflash2_fp8_head`·`glm53_dflash_loader_fp8`·`glm53_dflash_warmup`·`glm53_dflash_early_fc`·`glm53_drafter_prep`·`fp8_lm_head`) | 6 | 일부 | · | · | ● | · |
| `glm53_moe` | **묶음(34차)**: b12x 공유 워크스페이스·EP 마이크로커널 레인·직접 출력 (옛 `b12x_shared_workspace`·`b12x_zero_weight_micro`·`glm53_b12x_out`) + 정적(디코드) MoE 커널 v4(35·38차, `moe_static_kernel_v4.py` + 공유 헬퍼 `moe_static_common.py`, 프로필 기본값 `u`; v2/v3 은 34차 §8 일몰) + 순수 프리필 dynamic 재사용 후보(#368) + v5 `moe_static_kernel_v5.py`(타일 우선 가중치, 셀 `t`, 39차; `z`·`h` 는 39차 §3g/§3h 일몰) + 그 배치를 읽는 gated 프리필 커널 서브클래스 `moe_dynamic_gated_tiled.py` + NVFP4 블록 스케일 6-bit 패커 `moe_sf_pack.py`(39차 §4c) + E72 전체 토큰 프리필 후보 `moe_dynamic_ep_local.py` 및 단일 실행 remap `glm53_ep_route_remap.py` + SF6 직접 읽기 프리필 `moe_dynamic_gated_sf6.py` 및 `moe_reform_sf_pack.py` + EP 시작 수치 검증 `glm53_ep_local_selftest.py` + TP Q0 프리필 `moe_dynamic_gated_sf6_q0.py` 및 실제 가중치 시작 검증 `glm53_tp_sf6_q0_selftest.py` + EP tile-major 공통 가중치·static 디코드 및 시작 검증 | 21 | — | · | · | ● | · |
| `glm53_runtime` | **묶음(34차)**: prep-fused, 드래프터 학습 덤프, 샘플러 가드, 개발 랩, one-shot AR 배선 및 순수 프리필 collectives, 채팅 옵션 검증 및 GLM 본문 보존, KV 블록 zeroing 커널의 블록 인덱스 경계 가드(40차). 41차에 부팅 스탬프는 `boot_stamps` 로, 디코드 우선 스케줄러는 `sched_decode_first` 로 **내용 그대로** 빠져나갔다(DeepSeek-V4.1 이 필요로 하고 둘 다 GLM 의 것이 아니다; 합성된 build/glm53 71개 파일은 바이트 불변) | 14 | 일부 | · | · | ● | · |
| `glm53_prefix_cache` | 하이브리드 KV 프리픽스 캐시 조정자 수정(39차; 스톡은 이 레이아웃에서 히트 0). 기본값 `PREFIX_CACHE=1`(같은 접두사 재질문 92~99.6% 재사용, warm 수용률 = cold) | 1 | — | · | · | ● | · |
| `deepseek_reasoning` | 모델 전용 | 1 | — | ● | · | · | · |
| `deepseek_tool_parser` | 모델 전용 | 1 | — | ● | · | · | · |
| `dspark_drafter` | 모델 전용 | 3 | — | ● | · | · | · |
| `dsv4_attention` | 모델 전용 | 1 | — | ● | · | · | · |
| `dsv4_eager_scratch` | 모델 전용(신규 파일이라 계약은 이식 가능) | 1 | ✓ | ● | · | · | · |
| `dsv4_flashinfer_sparse` | 모델 전용 | 1 | — | ● | · | · | · |
| `dsv4_mhc_tilelang` | 모델 전용 | 1 | — | ● | · | · | · |
| `dsv4_model` | 모델 전용 | 1 | — | ● | · | · | · |
| `dsv4_oneshot_wiring` | 모델 전용 | 1 | — | ● | · | · | · |
| `dsv4_ops_cache_utils` | 모델 전용 | 1 | — | ● | · | · | · |
| `dsv4_ops_fused_indexer_q` | 모델 전용 | 1 | — | ● | · | · | · |
| `dsv4_tokenizer` | 모델 전용 | 2 | — | ● | · | · | · |

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

`VLLM_GLM53_KDA_PREFILL_DIRECT_OUT=1`은 별도의 기본-off 실험이다.
순수 프리필에서 KDA 마지막 출력 커널이 층 버퍼에 직접 써서 병합 복사를
제거한다. 혼합·디코드 경로는 그대로이며 GPU 수치·재생·속도 검증 전에는
승격하지 않는다. [검증 명령과 계약](../overlay/modules/glm53_kernels/README.md#pure-prefill-direct-output-2026-09-06-default-off).

`qwen38`은 이미지를 고정하지 않았다. 그 브링업은 스톡 이미지에서 돌았고 b12x 경로는 열린 문제가 아니라 닫힌 것이라(MEASUREMENTS.md), 프로필은 기록으로만 있다 — 실제로 합성해 배포한 적은 없다.
