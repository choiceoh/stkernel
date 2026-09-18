# Qwen3.8-Flash-Next 첫 서빙 — 플릿 부팅 세 번과 C4 프로브 (2026-09-18)

> 그날의 조사 — 운영자 지시 "올려봐" → "두 문제 해결해" → "폴백경로 말고 자체 경로 활용하는 방향으로" → "부팅도 한번 해보지그래".
> 레이아웃 v3(PR #1179 머지 6ad27304)의 산출물 `~/models/st-qwen38-tep4` 를 `launchers/start-st-qwen38.sh` 로 네 Spark 에 올린 기록.
> 코드는 PR #1180.

## 창 (프로덕션 양보)

`fleet_lease.py yield --kind session` 으로 프로덕션에 청하면 quiet gate 가 몇 초 만에 넘겨준다(감독기는 "fleet taken (session)" 로
대기). 이미지 `st-engine:qwen38` 은 창 밖에서 네 노드에 미리 빌드(캐시 레이어, 초 단위). 끝은 `... stop` 뒤 `fleet_lease.py release`.

| 창 | 시각 | 부팅 | 결과 |
|---|---|---|---|
| 1 | 16:14~16:47 | one-shot(기본) 107 s ready, NCCL 재부팅 40 s | 토큰 0. 즉시 프리필의 정적 MoE 커널이 (행, 전문가) 조합마다 JIT(랭크 0 이 12 분에 75 개, 약 4 s/개); one-shot 은 피어가 컴파일하는 동안 `STALL wait(peer-flags)` 를 찍었다 |
| 2 | 17:01~17:10 | C4 프로브(srv4 단일 GPU, checks-only) 뒤 one-shot 기본으로 53.5 s ready | **토큰 나옴**, STALL 0. 아래 |

## 고친 것 (자체 경로)

- **프리필**: `select_sm120_moe_backend` 에 규칙 하나 — expert-local 즉시 프리필(한 행에 한 라우트, 모든 전문가가 로컬, 디코드
  스텝(8 행)보다 많은 행)은 b12x 의 **dynamic 프리필 커널**로. 아티팩트 이름이 `dynamic_e128_k2560_n640_t1_<key>` 로 행 수와
  무관하다(정적 커널은 `static_m{행}_…_r{용량}` 으로 행마다 하나). 이번 부팅 랭크 0 의 CuTe-DSL 컴파일: **2 개**(첫 창 75 개).
- **one-shot**: 고칠 것이 없었다. 창 2 에서 기본(one-shot RDMA, inline flags)으로 STALL 0 — 창 1 의 스톨은 피어들이 커널을
  컴파일하느라 늦어 생긴 증상이었다(NCCL 로도 같은 스텝이 같은 시간 걸렸다). `--no-oneshot`(`ST_ONESHOT=0`) 은 옵션으로 남긴다.

## C4 프로브 (srv4, `ST_PROBE_CHECKS_ONLY=1`, `moe-checks-srv4.jsonl`)

합성 전문가 128개(프리샤드와 같은 NVFP4 인코딩), 서빙 라우터의 라우트. 오라클 = `engine/modules/moe.expert_gemm` 의 데이터플로
(reciprocal quantiser), 문턱 2%.

| 검사 | 로컬 pairs | 커널 | tile | 오라클 상대오차 | 반복 |
|---|---|---|---|---|---|
| decode 8/6/4/2 토큰 | 14/17/8/4 | micro(캡처) | M64 | 0.0053 / 0.0063 / 0.0056 / 0.0064 | 안정(0 ulp), 재생 바이트 동일 |
| prefill 16 토큰 | 43 | dynamic | 16 | 0.0080 | 불안정: ±1 bf16 ulp 급(max_abs 2^-7), 상대 0.011 |
| prefill 64 | 156 | dynamic | 16 | 0.0069 | 불안정 0.0034 |
| prefill 128 | 339 | dynamic | 16 | 0.0089 | 불안정 0.0045 |
| prefill 1024 | 2,563 | dynamic | 32 | 0.0054 | 불안정 0.0054 |
| prefill 4096 | 10,239 | dynamic | 64 | 0.0074 | 불안정 0.0074 |

전부 오라클 안. 프리필의 반복 불안정은 dynamic 커널의 BF16 원자 scatter 순서(GLM 프리필도 같은 커널)로, 프로브가 "기록하는 발견,
멈춤 아님" 으로 두는 것. 정적 프리필의 반복 불일치(첫 C4 실행)와 달리 오라클은 매번 든다.

## 플릿 (창 2, C=1, greedy, thinking off, 도어 = srv2:8000)

| 요청 | 프롬프트 | 생성 | 벽시계 | 비고 |
|---|---|---|---|---|
| 17×23 | 32 | 26 | 16.61 s | 첫 요청: dynamic 커널 컴파일 포함. 답 "391 …" 정확 |
| 하늘이 파란 이유 | 31 | 149 | 6.33 s | 23.5 tok/s incl. prefill, 문장 정상 |
| 한국의 수도 | 35 | 23 | 0.85 s | 27.1 tok/s, "서울특별시 … 약 940만 명" — 한글 정상 |
| 트랜스포머 설명 350 단어 | 52 | 450 | 11.43 s | **39.4 tok/s incl. prefill** |
| 1,761 토큰 한글 요약(콜드) | 1,761 | 39 | 34.38 s | 새 pair 대역의 dynamic 아티팩트·QSA 프리필 커널 JIT |
| 같은 길이 다른 프롬프트(웜) | 1,767 | 39 | 1.33 s | 프리필 약 0.5 s ≈ 3.5K tok/s |
| 1,286 토큰 영어(콜드) | 1,286 | 41 | 4.47 s | 또 다른 대역의 아티팩트 |

카운터: decode 스텝 394 회 13.90 s → **35.3 ms/스텝**; 생성 687 토큰 → **1.74 토큰/스텝**(MTP K=1 수용 82/114 = 72%) →
디코드만 약 **49 tok/s**. 프리필 스텝 5 회 52.9 s 는 콜드 JIT 이 대부분(웜 1,767 토큰 ≈ 0.5 s).

## 안 잰 것

C=2/4, T=1 수용률, 긴 컨텍스트(≥8K), one-shot 지연 실측(`ONESHOT_MEASURED_HIDDEN` 기록은 별도 프로브), 프리필의 SSD PLE 모음
비용 분리, 정확도 벤치. 콜드 JIT 은 `/cache` 에 남아 다음 부팅부터 사라진다(P3 의 768 토큰 블록당 forward 는 그대로).
