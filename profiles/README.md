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
| `dsv4` | DeepSeek-V4-Flash-0731 | 16 | production |
| `glm53` | GLM-5.3-Flash NVFP4 | 2 | bring-up, blocked |
| `qwen38` | Qwen3.8-Flash-Next NVFP4 | 1 | bring-up |

`glm53` carries two modules of its own and can load none of `dsv4`'s: its image
installs to dist-packages rather than the venv site-packages, and one of its
modules targets flashinfer rather than vllm at all. That is what `TARGET_PREFIX`
is for -- the allowlist for container paths describes an image, so it belongs to
the profile. A module binding outside it aborts the compose.

`qwen38` stays at one module: it ran on stock image code, and its b12x path is
closed rather than pending (MEASUREMENTS.md).

A profile also carries the serving knobs that are the model's rather than the
fleet's -- backend, speculative depth, draft placement -- and, where a bring-up
is blocked, says so and names the one flip that would isolate the cause.

## 프로필별 구성

| | `dsv4` | `glm53` | `qwen38` |
|---|---|---|---|
| 상태 | production | bring-up, Korean-token corruption open | bring-up |
| 이미지 | `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6` | `glm53:v13-b12x` | 미고정 |
| 패키지 루트 | `site-packages` | `dist-packages` | 기본값 |
| 모듈 수 | 17 | 9 | 1 |
| 오버레이 파일 | 21 | 11 | 2 |

### 모듈 × 프로필

범위가 모델을 넘는 것을 위에, 한 모델에 묶인 것을 아래에 둔다. **이식 가능**은 신규 파일이면서 타깃이 상대경로인 것 — 이미지가 달라도 그대로 실린다. 나머지는 파일 전체 교체라 계약이 이미지에 묶인다.

| 모듈 | 범위 | 파일 | 이식 | dsv4 | glm53 | qwen38 |
|---|---|---:|:---:|:---:|:---:|:---:|
| `moe_gate_sm121` | GB10의 모든 MoE | 1 | ✓ | ● | ● | · |
| `spec_fp8_lm_head` | 드래프터 일반 | 1 | ✓ | · | ● | · |
| `tp_oneshot_ar` | 어느 모델이든 | 2 | ✓ | ● | ● | ● |
| `b12x_swiglu_clamp` | flashinfer 0.6.18 b12x | 2 | — | · | ● | · |
| `flashinfer_b12x_collapse` | flashinfer 0.6.18 b12x | 1 | — | · | ● | · |
| `mla_indexer` | DeepSeek-MLA | 1 | — | ● | · | · |
| `mla_sparse_swa` | DeepSeek-MLA | 1 | — | ● | · | · |
| `spec_fp8_head` | 드래프터 일반 — **기각** | 1 | — | ○ | · | · |
| | | | | | | |
| `deepseek_reasoning` | 모델 전용 | 1 | — | ● | · | · |
| `deepseek_tool_parser` | 모델 전용 | 1 | — | ● | · | · |
| `dspark_drafter` | 모델 전용 | 3 | — | ● | · | · |
| `dsv4_attention` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_eager_scratch` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_flashinfer_sparse` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_mhc_tilelang` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_model` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_oneshot_wiring` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_ops_cache_utils` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_ops_fused_indexer_q` | 모델 전용 | 1 | — | ● | · | · |
| `dsv4_tokenizer` | 모델 전용 | 2 | — | ● | · | · |
| `glm53_dflash2_fp8_head` | 모델 전용 | 1 | — | · | ● | · |
| `glm53_dflash_loader_fp8` | 모델 전용 | 1 | — | · | ● | · |
| `glm53_model_gate` | 모델 전용 | 1 | — | · | ● | · |
| `glm53_oneshot_wiring` | 모델 전용 | 1 | — | · | ● | · |

이식 가능한 모듈은 셋뿐이다 — `tp_oneshot_ar`, `moe_gate_sm121`, `spec_fp8_lm_head`. 셋 다 새로 만드는 파일이라 대체할 베이스가 없고, 그래서 이미지가 달라도 계약이 성립한다. 나머지가 한 이미지에 묶이는 이유는 기능이 특수해서가 아니라 오버레이가 **파일 전체 교체**이기 때문이고, 그래서 `*_wiring`·`glm53_*` 계열이 짝으로 존재한다: 이식 가능한 알맹이와 이미지별 배선.

`spec_fp8_head`는 ○로 표시했다: dsv4에 마운트돼 있지만 `VLLM_DSPARK_FP8_DRAFT_HEAD=0`으로 꺼져 있다. rowwise `_scaled_mm` 판본이고 실측에서 60.6 vs 61.7·수용률 무이동으로 기각됐다(MEASUREMENTS.md:419). 채택된 쪽은 `spec_fp8_lm_head`(deepgemm)이며 dsv4는 아직 `dspark_drafter` 안의 사본을 쓴다.

`qwen38`은 이미지를 고정하지 않았다. 그 브링업은 스톡 이미지에서 돌았고 b12x 경로는 열린 문제가 아니라 닫힌 것이라(MEASUREMENTS.md), 프로필은 기록으로만 있다 — 실제로 합성해 배포한 적은 없다.

