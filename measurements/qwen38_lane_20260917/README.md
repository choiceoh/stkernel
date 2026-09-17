# Qwen3.8 단일 GPU 레인 첫 실행 — 레인 판정, dense 전환, MoE 셀 (2026-09-17)

> 그대로 두는 기록 — 이 날 srv4 단일 GPU 레인에서 돈 Qwen3.8 이식 캠페인(`engine/QWEN38_CARRY.md`) 티켓 네 개의 결과와 그 원시 로그다.
> 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

체크포인트 없이 `probes/qwen38_config.json`(srv2 체크포인트 config 와 바이트 동일)에서 읽은 형상으로, 합성 가중치와 입력을 썼다.
속도 주장은 없다. dense·MoE 의 시간은 이 실행에서 한 칸도 재지 못했다(아래 "멈춘 곳").

| 티켓 | 커밋 | 프로브 | 결과 | 로그 |
|---|---|---|---|---|
| `qwen38-cells-0917` | 7cd39bb9 | `--lanes qwen38_cells`(C1) | qualify 통과, GPU 케이스 7건 중 3건 오류(테스트 픽스처) | [cells-7cd39bb9.log](cells-7cd39bb9.log) |
| `qwen38-cells2-0917` | 8c25c625 | 같은 프로브, M1·S1·K1 GPU 케이스 추가 | qualify 통과, 19건 중 3건 오류(같은 픽스처) | [cells2-8c25c625.log](cells2-8c25c625.log) |
| `qwen38-dense-0917` | 79ff030d | `--lanes qwen38_dense`(C2) | 두 형상 게이트 통과 뒤, 비서빙 FP8 팔 실패로 멈춤 | [dense-79ff030d.log](dense-79ff030d.log) |
| `qwen38-moe-0917` | 4148ea03 | `--lanes qwen38_moe`(C4) | 디코드 네 크기 통과 뒤, 프리필 반복 불일치로 멈춤 | [moe-4148ea03.log](moe-4148ea03.log) |

장치: NVIDIA GB10, torch 2.13.0+cu132, CUDA 13.2, 이미지 `st-engine:glm53`. `qwen38-kda-0917`(C3)은 이 기록 시점에 대기 중이다.

## 1. Qwen3.8 자체 레인 qualify (C6) — 통과

`engine/profiles/qwen38/lanes.qualify` 가 서빙 레인을 `engine/modules` 오라클에 대조했다. 두 cells 실행의 값이 같다.
대역은 max 5e-2 / rms 2e-2 다. 값은 (최대 오차/최대 크기, 오차 rms/rms) 이다.

| 레인 | max | rms |
|---|---|---|
| 게이트 잔차 enter / inject / leave_norm / leave / close | 0.0063 / 0.0050 / 0.0050 / 0.0050 / 0.0056 | 0.0012 / 0.0020 / 0.0025 / 0.0025 / 0.0027 |
| GDN decay / beta / raw beta / 출력 norm | 1.9e-7 / 0 / 0 / 7.0e-5 | 1.2e-7 / 0 / 0 / 2.4e-6 |
| QSA norm+partial rope 6×256 / 4×128 | 0.0062 / 0.0069 | 0.0015 / 0.0023 |

이 행이 C6 의 GPU 판정이고, 수치를 바꾸는 뒤 작업의 기준점이다.

## 2. GPU 케이스 (C1) — 픽스처 오류 3건 말고 모두 통과

cells2 에서 통과한 것:
- KDA decay 진입점(KDA 게이트와 같음, rows 접기)
- MHCV41 이음매, MLA 글루(`selftest mla rel=3.07e-03`), 패딩 dense
- M1 `gated_sum` 5건(GPU 반올림이 torch 와 같음 포함)
- S1 `swiglu(pad_to)` 4건(BF16 이 torch 쌍과 바이트 동일)
- K1 GDN 링 게이트 3건(BF16, 4/12×128 에서 gates→decay 진입점과 바이트 동일)

오류 3건은 `KdaDecayKernelTests.test_the_ring_decay_entry_is_the_recurrence_and_the_functional_lane` 의 GPU 전용 7 토큰 케이스다.
테스트가 링을 6칸으로 만들어, 런처가 `1 <= T <= R` 위반으로 거부했다(커널 결함 아님). 이 기록의 PR 에서 링을 8칸으로 바꿨다.

## 3. dense W4A8/FP8 전환 (C2) — 멈춤, 발견 하나

- **통과:** `gdn.in_proj` 4120×2560 과 `gdn.out_proj`+`attn.o_proj` 2560×1536 은 1–4096 행 모든 게이트를 통과했다.
  - W4A8 오차 0.085–0.087, FP8 0.037–0.039
  - 디스패치 바이트 동일, fp32 쌍둥이와 ≤2e-5
  - 캡처 재생 바이트 동일
- **발견:** shared expert `moe.sh_gate_up` **320×2560** 의 **FP8 팔이 4행에서 상대 오차 0.220**(대역 0.05)이었다. 1·2 행은 0.040·0.037 이다.
  - 그 행 수를 서빙한 W4A8 은 0.086 으로 정상이었다.
  - FP8 레인은 320 행을 384 로 채워 DeepGEMM `fp8_gemm_nt` 를 부르므로, (N=384, K=2560, M=4) 의 네이티브 GEMM 이 의심된다.
  - 서빙 전환(>32행 FP8)이 이 형상에서 맞는지는 48행 이상을 재야 안다.
- **멈춘 곳:** 프로브가 서빙하지 않는 팔의 실패에도 멈췄다. 이 기록의 PR 에서 그런 팔은 `broken_arms` 로 기록하고 그 칸만 빼고 계속하게 했다.

## 4. MoE EP 셀 (C4) — 멈춤, 발견 하나

- **디코드(캡처, micro M64 MAC 48, sentinel 128 스킵):** 2·4·6·8 토큰 모두 통과했다.
  - 오라클 0.53–0.64%(게이트 2%)
  - 반복과 재생 바이트 동일
  - 가중치 0·전부 다른 랭크 경로에서 정확히 0
- **프리필 128 토큰(로컬 pair 316, 정적 커널 [128,128] MAC 48):** 오라클 0.43% 는 통과했지만, **같은 입력 두 번의 eager 호출이 한 원소에서 0.0039**
  (최대 크기 0.45 의 0.87%, BF16 값 순서로 29,428 칸) 갈라졌다.
  - FP32 합산 순서로는 이만큼 벌어질 수 없다.
  - 정적 커널 호출 자체인지 `index_add_` 합산인지는 이 실행이 가르지 못했다.
- **멈춘 곳:** 프로브가 이 반복 불일치에서 멈춰 프리필 스윕과 micro 변형은 돌지 않았다. 이 기록의 PR 에서 불일치를 호출·합산으로 진단해 기록하고 계속하게 했다.

## 다음

- cells(링 8칸), dense(`broken_arms`), MoE(반복 진단) 재실행. 결과는 새 기록으로 남긴다.
- FP8 320×2560 소행 오차: 행 수·N 스윕으로 경계를 찾고 DeepGEMM 호출을 확인한다.
- 정적 MoE 커널 반복 불일치: 진단이 호출을 가리키면 커널, 합산을 가리키면 레인을 고친다. 그 전에는 C4 셀을 admit 하지 않는다.
