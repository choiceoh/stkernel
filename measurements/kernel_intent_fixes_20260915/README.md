# 커널 계약 수정과 FP32 라우터 검토 — 2026-09-15

## 결과

검증 난수 경계, greedy 동점, 비연속 텐서 주소 계산을 수정했다. FP32 라우터 가중치를 실제 읽는 계산도 비교한 뒤, **기존 native BF16 입력 / FP32 출력 계산을 유지하고 사용하지 않는 FP32 복사본을 제거**했다.

브랜치: `codex/kernel-intent-fixes`. 수정 기준 커밋: `24240e36944f869c3d6f96a0c6863817d4a4589c`.
이전 문제 재현 결과는 `../kernel_intent_audit_20260915/`에 그대로 보관했다.

## 구현 변경

1. **Block verification:** CUDA, scalar host, batch Torch 참조의 승인 비교를 모두 `<`로 맞췄다. `[0,1)` 난수에서 확률 0을 승인하지 않는다. Dense test oracle도 같은 경계 계약을 따른다.
2. **Greedy:** signed zero와 NaN을 vocabulary argmax와 동일하게 정규화한다. BF16/FP16/FP32에서 mixed-temperature 배치와 전용 argmax의 토큰이 일치한다.
3. **Stride:** sampler의 temperature/top-k/top-p/uniform 벡터, drafter의 slot/context 벡터는 자신의 stride를 사용한다. Verifier는 target/draft/IDs/probabilities/uniform/residual 각각의 stride를 받는다. 후보 IDs의 stride를 확률에 재사용하지 않는다.
4. **지원하지 않는 배열 형태:** RMSNorm와 residual RMSNorm, SwiGLU의 CUDA wrapper는 비연속 열/weight를 조용히 읽지 않고 거부한다. Sampler의 distribution 출력도 shape와 연속 열 조건을 확인한다. 단일 norm+RoPE는 기존 입력/position 정리 방식처럼 weight view도 연속 복사하여 계산한다. 이미 연속인 weight는 추가 복사하지 않는다.
5. **라우터:** `prepare_routers()`는 BF16 바인딩을 검증하고 준비한 레이어 집합을 기록한다. 복사본 기반 준비 상태를 제거했고, arena 예약 및 budget의 FP32 router 항목도 제거했다. 기존 `router_resident_bytes` 계측은 0을 기록한다. Native boot의 실제 실행 검사는 준비한 레이어 집합과 실행한 레이어 집합을 계속 대조한다.

새 회귀 테스트는 `tests/test_engine_kernel_contracts.py`에 있다. 값을 바꾼 CUDA Graph 재생, padding만 바꾼 경우, 확률 0/1 및 정확한 승인 경계, 서로 다른 텐서 stride, 지원하지 않는 출력 형태를 포함한다. 라우터 준비의 추가 GPU 할당량이 0이고 기존 native expert IDs/weights와 비트 단위로 같은지도 검사한다.

라우터 준비 상태를 사용하는 기존 테스트 fixture를 갱신했다. 검증 중 발견한 기존 MLA test double의 누락된 `branch` 인자도 실제 함수의 현재 시그니처에 맞췄다.

## FP32 라우터를 실제로 읽었을 때

### 방법

- RTX 5050 / SM120, PyTorch 2.13.0+cu130. 실제 모델 가중치가 아닌 합성 BF16 `[288,4096]` router와 BF16 hidden을 사용했다.
- 행 수 `1,7,8,28,64,2304,9216`, 총 11,628행을 비교했다.
- Native: 작은 행 수는 기존 `glm_pointwise.router_logits`, 9,216행은 기존 `prefill_router.router_logits`를 사용했다. 둘 다 그대로인 구현이다.
- FP32 IEEE: 준비한 FP32 weight를 읽어 `x.float() @ resident.T`를 계산했다. TF32는 비활성화했고 activation 변환 비용을 포함했다.
- FP32 cached-x: activation 변환을 제외하여 projection 자체도 확인했다.
- FP32 TF32: 같은 FP32 resident를 읽되 TensorFloat-32 연산을 허용하는 별도 arm이다. 이것은 엄격한 IEEE FP32 연산과 구분한다.
- FP64 참조는 각 크기의 앞 `min(rows,64)`행에 대해 계산했다. Expert 선택 비교는 전체 행에 대해 수행했다.
- CUDA event 구간에서 여러 호출을 실행하고, arm 순서를 뒤집은 3회 측정의 중앙값을 사용했다. 작은 행 수는 호스트 launch 지연의 영향을 받으므로 미세한 순위 차이를 확정하지 않는다. GPU를 독점하지 않은 로컬 측정이다.
- 프로세스의 TF32 설정은 종료 시 복원했다. 모델/서버를 실행하거나 전역 서비스를 변경하지 않았다.

### 주요 측정

| 행 수 | 기존 native | FP32 resident + activation 변환 | 소요 시간 비율 |
| --- | ---: | ---: | ---: |
| 28 | 17.06 µs | 40.96 µs | 2.40배 |
| 2,304 | 210.98 µs | 1,101.37 µs | 5.22배 |
| 9,216 | 852.12 µs | 4,204.40 µs | 4.93배 |

9,216행의 앞 64행에서 FP64 참조 대비 출력 상대 L2 오차:

| 계산 | 상대 L2 오차 |
| --- | ---: |
| 기존 native BF16 입력 / FP32 누산·출력 | 약 `4.40e-6` |
| FP32 resident, IEEE FP32 | 약 `6.09e-7` |
| FP32 resident, TF32 허용 | 약 `7.31e-6` |

IEEE FP32 계산은 이 입력에서 누산 결과의 오차를 약 7.2배 줄였다. 다만 **선택한 top-8 expert 집합은 전체 11,628행에서 차이가 없었다.** 9,216행 중 한 행은 선택 집합 안의 정렬 순서가 달랐다. 정확한 모든 값과 arm별 비교는 `router_precision.json`에 있다.

### 결정과 범위

BF16 checkpoint 값을 FP32로 넓힌 복사본 자체에는 추가 가중치 정보가 없다. 전체 FP32 GEMM의 계산 순서는 수치 오차를 줄일 수 있지만, 이번 합성 비교에서는 expert 집합 개선을 보이지 않았고 큰 prefill의 projection 시간이 약 5배였다. 이에 따라 기존 native projection을 유지하고 불필요한 복사본을 없앴다.

기존 stock/reference 실행의 `x.float() @ gate.float().T` 계산은 유지한다. FP32를 native 기본값으로 도입하려면 실제 GLM weight와 activation, GB10/SM121, TP4의 quality/latency 비교가 필요하다. 로컬 합성 수치 오차를 언어 모델 정확도 향상률이나 전체 토큰 속도 변화로 해석하지 않는다.

제거한 예약량은 GLM-5.3의 MoE 42개 × expert 288 × hidden 4096 × 4 bytes = **rank당 189MiB, TP4 합계 756MiB**다. 모델 구조와 제거한 할당 수식으로 계산한 값이다. 합성 한 레이어에서는 수정 후 준비 단계의 실제 추가 CUDA 할당량 0을 회귀 검사했다.

## 검증 기록

- `regression_tests.log`: 최초 8개 새 회귀 테스트 모두 통과.
- `router_regression.log`: 추가 GPU 할당 0 및 native 라우팅 결과 유지 테스트 통과.
- `rotary_weight_recheck.log`: 추가 RoPE weight view / 값 변경 graph 재생 테스트 통과. 새 회귀 사례는 총 10개다.
- `targeted_tests.log`: 주변 테스트 157개를 실행했다. 최초 결과는 141 통과, 14 skip, 1 수치 불일치, 1 test-double 오류였다.
- `mla_fixture_recheck.log`: test double의 `branch` 인자를 고친 뒤 해당 사례 통과. 이를 반영한 주변 테스트 결과는 142 통과, 14 skip, 아래 기존 수치 불일치 1개다.
- `cpu_engine_check.log`: 저장소 CI와 같은 `tools/check.py --list`에서 **221파일 / 2,000테스트, 실패 0, 실행 불가 0, skip 391**. 이 실행의 새 회귀 모듈 집계는 9개이며, 이후 추가한 GPU 전용 RoPE weight 사례는 별도 로그에서 통과했다.
- `onepass_checks.log`, `oracle_checks.log`: CI의 onepass recording **122개**, CPU oracle **77개** 모두 통과.
- 최종 집계와 소스/산출물 체크섬은 `manifest.json`에 기록한다.

### 남은 검증 한계: 기존 RTX 5050 라우터 수치 테스트

`tests.test_engine_decode_seven.RouterTensorCoreTests`는 2,304행에서 기존 BF16 matmul과 FP32 matmul의 원소별 오차 허용치 `rtol=3e-5, atol=5e-6`를 비교한다. 이 환경에서는 663,552원소 중 1,105개(약 0.2%)가 허용치를 넘고, 보고된 해당 원소의 절대 차이는 `1.694262e-5`다.

라우터 projection 두 파일 및 해당 테스트가 수정 기준 HEAD와 바이트 단위로 동일함을 `router_baseline_sources.json`에 기록했다. 수정하지 않은 이 구현/테스트만 새 프로세스로 독립 실행해 같은 불일치를 재확인했다(`router_baseline_recheck.log`). 허용치를 느슨하게 바꾸지 않았다. 이 결과 때문에 GB10의 native 라우터 정밀도 적합성을 이번 로컬 실행으로 보증하지 않는다.

## 실행 방법

저장소 루트에서 CUDA PyTorch/Triton Python을 사용한다. 이번 실행기는 `/tmp/st-draft-topk-venv/bin/python`이었다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m unittest tests.test_engine_kernel_contracts -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python measurements/kernel_intent_fixes_20260915/router_precision.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python tools/check.py --list
```

성능 스크립트는 `router_precision.json`을 덮어쓴다. 이전 조사 폴더의 `reproduce.py`는 수정 전 오류를 assert하는 역사적 재현이므로 수정 후 성공 판정에는 새 회귀 테스트를 사용한다.
