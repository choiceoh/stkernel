# CPU 후보 선별 / 오라클 한계 — 2026-09-13

**목표 미달. 새 커널의 tok/s는 아직 계산할 근거가 없다. GPU 큐 제출 0회.**

목표는 GLM-5.3-Flash, 4×GB10, C=1, 접두사 재사용 0, 프로파일러 OFF에서 실제 입력 토큰 / 클라이언트 TTFT:
2K 3,300 tok/s, 128K 4,000 tok/s. CPU 컴파일, 오라클 시나리오, 실제 성능을 구분한다.

## 이번에 준비한 후보

- `engine/kernels/b12x/moe_prefill_q0_batch8.py`: 짧은 프리필의 Q0 입력 준비를 4행에서 8행으로 늘린다.
  288개 전문가의 누적 위치 계산은 lane 0 직렬 루프에서 9개 워프의 병렬 prefix로,
  토큰별 top8 예약은 lane 0 루프에서 8개 lane의 동시 예약으로 바꿨다. 중복 전문가와 0·음수 가중치도 보존한다.
  입력이 기존 sC뿐 아니라 sA도 사용하므로, sA에 있던 라우팅 메타데이터를 sB로 옮겼다.
  실제 컴파일 레이아웃에서 65,536 B 입력 영역과 3,456 B 메타데이터 영역의 비중첩을 확인했다.
  `_prefill_q0_batch8=True` 명시 호출 전용이며, M128·65~8,192행의 eager 경로만 허용한다.
  packed SF6·raw scale·N128 조합을 각각 준비했고 기존 FP32 scatter와 BF16 반올림은 유지했다.
- `engine/kernels/prefill_collectives/sum_pack.py`: routed/shared MoE의 BF16 합산을 FP8 통신 패킹에 합친다.
  FP8 양자화 전에 기존 합산과 같은 BF16 반올림을 수행하고, 패딩은 통신 패킷에서만 만든다.
  `PrefillCollectives(comm, fuse_sum=True)`로 선택하며 일반 실행과 layer-major 실행에 모두 연결했다.
  FP8_MIN_ROWS 미만은 기존 합산·통신을 사용한다. 32,256×4,096 BF16 기준 MoE 층당 rank마다
  252 MiB 중간 텐서 한 개와 그 텐서의 쓰기·재읽기 504 MiB를 제거한다. 지속 상주 메모리나 통신량 감소는 아니다.
- `engine/kernels/prefill_mhc.py`: 잔차 post와 다음 prenorm을 합친다. 기존 BF16 반올림을 유지하고,
  이미 손실 없는 것으로 확인한 BF16 계수 팩을 재사용한다. FP32 누산 순서는 달라 GPU 수치·품질 검증이 필요하다.
  `MHC(weights, prefill=True)`로만 선택한다. 기본값은 OFF다.
- `engine/kernels/b12x/moe_dynamic_prefill_n128_tiled.py`: 기존 tiled 가중치에 N128 FC1의 A/SFA 재사용을 연결한다.
  `_prefill_scale_expansion=True, _prefill_n128=True` 명시 호출 전용이다. FC2의 네 조각을 레지스터에 모두
  유지하는 방식은 자원 사용이 불리해 기존 shared-memory 재읽기를 유지한다. 디코드 자동 선택에는 들어가지 않는다.
- `bench/step_kernels.py`: 사전 계수가 없는 프로파일의 fallback 회귀를 고쳤다. 총시간을 실제 전체 토큰 수와
  청크 개수의 두 비용으로 적합한다. 청크 용량을 전체 토큰으로 쓰거나 두 단위를 더하지 않는다.
  전체 토큰이 없거나 두 비용을 분리할 수 없으면 추정값을 만들지 않는다.
  기존 저장 계수와 앞선 L7 오라클 계산은 이 fallback을 사용하지 않아 이전 보고 수치에 영향이 없다.

## 실제 CPU 결과

동일 이미지 `sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`.
CPU 2개, 메모리 4 GiB, runc, 네트워크 없음, NVIDIA 장치 없음. CUDA context 생성 없음.

| 후보 | 첫 컴파일 | 수정 / 최종 선택 | 판정 |
|---|---:|---:|---|
| mHC BM32/BK128 → BM16/BK64 | REG 255 / STACK 1,136 B | REG 214 / STACK 0 B | r3에서 splits 1·2·4 확인 |
| N128 FC2 네 조각 유지 → 재읽기 | STACK 912 B | STACK 528 B | r2 유지, 속도 이득 미검증 |
| N128 FC1 unroll 4 → 1 | STACK 528 B | STACK 784 B | r3 기각, r2 소스로 원복 |
| Q0 batch8 packed / raw | 공유 메모리 주소의 IR 지역성 오류 | REG 168 / STACK 112 B | 정적 레이아웃 오프셋으로 수정; 두 변형 컴파일 통과 |
| Q0 batch8 + N128 | — | REG 168 / STACK 528 B | 조합 컴파일 통과 |
| BF16 합산·FP8 패킹 융합 | — | REG 40 / STACK 0 B | 컴파일 통과, GPU 수치·속도 미검증 |

`cpu-r1`, `cpu-r2`, `cpu-r3`에는 실제 컴파일 로그·자원 보고·소스 해시를 남겼다.
STACK은 컴파일러 자원 지표이며 실행 시간이나 실제 spill 트래픽 측정값이 아니다.
mHC 최종 커널은 r3, N128 최종 커널은 r2 결과에 대응한다.
각 실행의 CPU 검증은 30개 중 23 통과·7 생략. 수정한 오라클 검증은 55개 통과.
최종 CPU 검증은 BF16 반올림, packed 좌표, 꼬리 행, split 3의 비정렬 H 범위와 667행을 포함한다.
CPU shim의 FMA·dot은 GPU 연산 순서/수치 오차를 증명하지 않는다.

추가 Q0 후보는 `q0-batch8-r1`의 실패와 `q0-batch8-r2`의 수정 결과를 함께 보존한다.
최신 `sum-pack-r2`는 소스 `eb58d244`에서 **49개 검증 중 42 통과·7 생략**, 9개 변형 컴파일 통과
(36.38초, CUDA 미초기화)다. 각 커널의 소스 해시는 `result.json`, 실제 선택한 검증은 `test-scope.json`에 있다.
실제 Q0 본문의 prefix·예약·입력 staging을 CPU에서 실행했고,
합산·패킹 본문은 독립적인 BF16 합산 후 패킹 결과와 바이트 단위로 비교했다.
TP4 CPU 모델에서는 새 합산 전달을 켜기 전후의 hidden·KDA 상태·KV 페이지가 일반/층별 실행 모두 정확히 일치했다.
이는 같은 실행 배치 안에서 새 합산 전달의 동등성을 확인한 것이며 모든 SP 상태 검증의 통과는 아니다.

### 별도로 남은 기존 검증 실패

`tests.test_engine_token_shards.TokenShardTests.test_model_prefill_and_next_decode_keep_hidden_aux_kda_and_kv`
의 130·131행이 unsharded와 SP 비교에서 실패했다. 새 합산 융합 이전 소스 `8fcd20a1`을
**동일 CPU 이미지에서 이 테스트만 실행**해도 두 경우 모두 같은 1,657개 원소 불일치와
최대 절대 오차 0.58984375가 재현됐다. 원인은 아직 확정하지 않았다.
`sum-pack-r1/cpu-tests.log`와 `sum-pack-before-tests.log`에 두 결과를 남겼다.
기준 엔진 부팅이나 실제 GLM 가중치·GPU 실행은 아니며, 이 실패를 완화하거나 성공으로 집계하지 않았다.

`run_cpu.py`의 기본 전체 검증에는 해당 테스트를 그대로 유지했다. 최신 결과는 `--tests`로 새 후보와
영향 범위를 명시한 집중 검증이며 전체 상태 검증 통과나 GPU 품질 통과를 뜻하지 않는다.

## 오라클에서 말할 수 있는 범위

`retained-L7.json`은 이전 실측을 보존한 입력이고, `retained-oracle-scenarios.json`은 이전 스케줄링 분석 결과다.
그 파일의 `candidate_commit`은 이전 수리판이며 이번 새 커널의 성능 기록이 아니다.

| 입력 | L7 실측 | 청크 분할만 수정한 기존 시나리오 | 목표 |
|---|---:|---:|---:|
| 2K, 2,672 토큰 | 2,413.7 tok/s | 2,413.7 tok/s | 3,300 |
| 2K, 2,632 토큰 | 2,378.5 tok/s | 2,378.5 tok/s | 3,300 |
| 2K, 2,618 토큰 | 1,740.7 tok/s | 2,313.7 tok/s | 3,300 |
| 128K, 129,775 토큰 | 3,305.7 tok/s | 긴 입력에는 짧은 회귀를 적용하지 않음 | 4,000 |

이미 한 청크인 2K는 TTFT를 약 297~309 ms, 128K는 약 6,814 ms 더 줄여야 한다.
N128의 요청 TMA 바이트 23.4% 감소는 커널 시간 23.4% 감소가 아니다. 캐시·동기화·레지스터·연산 비중을
재지 않았기 때문이다. mHC의 읽기/런치 감소 역시 새 비용 계수 없이 tok/s로 환산하지 않는다.
따라서 이번 후보의 `candidate_tok_s`는 미측정이며 **오라클 목표 달성으로 승인하지 않았다**.

실측 없는 임의 계수로 목표에 맞추면 도구가 목표를 맞춘 것이지 엔진이 개선된 것이 아니다.
현재의 오라클은 스케줄링 비교와 필요한 개선 폭 계산에 유용하고, CPU 컴파일은 나쁜 커널 후보를 거른다.
새 커널의 실행 시간/수치/품질은 이후 승인된 GPU 검증이 채워야 한다. 기준 엔진 재실행이나 큐 재등록은 하지 않았다.
