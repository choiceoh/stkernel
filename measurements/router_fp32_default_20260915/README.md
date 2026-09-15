# GLM 라우터 기본값을 IEEE FP32로 변경 — 2026-09-15

## 결과

운영자 요청에 따라 MoE 라우터 projection을 **FP32 입력·가중치·누산·출력**으로 바꿨다.
Decode, 짧은 prefill, 긴 prefill, TP4 sender 및 packet 소비 경로에 적용했다.
별도 BF16 성능 옵션은 두지 않았다. 이전 `kernel_intent_fixes_20260915`의 BF16 유지 결정은
그 시점의 기록이며, 현재 구현은 이 문서가 설명하는 FP32 경로다.

구현 기준 커밋: `1d51304cb6f4ae76a4a87c9557345a447882a05f`.
브랜치: `codex/kernel-intent-fixes`.

## 연산과 실제 가중치 읽기

- `net.prepare_routers(arena)`가 체크포인트의 BF16 gate를 한 번 FP32로 복사한다.
  Native `route()`와 `_sender_routes()`는 `_router_weights[L]`의 FP32 저장소를 직접 읽는다.
  BF16 checkpoint 바인딩을 0으로 바꿔도 준비된 FP32 라우팅 결과가 유지되는 테스트로 실제 소비를 확인했다.
- `router_fp32.cpp`에서 활성값을 FP32로 승격하고 `at::mm`를 호출한다. 가중치가 BF16이면 거부한다.
  `at::NoTF32Guard` 및 CUDA autocast 제외 guard는 해당 스레드의 호출 범위에만 적용한다.
  다른 연산의 전역 TF32 설정과 autocast 상태는 그대로다.
- 긴 prefill의 BF16 Triton GEMM을 제거했다. Sender도 같은 FP32 projection을 호출한다.
  Packet 경로는 FP8 × scale을 기존처럼 BF16으로 반올림한 뒤 FP32 입력으로 쓴다.
  따라서 transport 값을 임의로 바꾸지 않고 라우터 계산의 정밀도를 높인다.
- Sigmoid, correction bias, expert top-k, 정규화된 routing weight는 FP32를 유지한다.
- 부팅의 native 실행 증거는 `router_fp32`로 기록하며 모든 준비된 layer의 실제 실행을 요구한다.
  `router-fp32` C++ 확장도 첫 collective 전에 빌드해 rank별 JIT 대기가 통신 안으로 들어가지 않게 한다.

BF16 체크포인트 값을 FP32로 승격해도 저장 전에 사라진 가중치 정보는 복원되지 않는다.
이번 변경은 실제 행렬곱과 누산을 IEEE FP32로 수행하게 한다.

## 메모리

- Resident gate: 42 MoE layers × 288 experts × 4096 hidden × 4 bytes = **189 MiB/rank**.
- TP4는 라우터를 복제하므로 네 rank의 합계는 **756 MiB**다.
- 추가 상주 공간과 carve 정렬 공간을 arena 예약에 포함했다. Budget 표 및
  `router_resident_bytes` 계측이 실제 예약을 반영한다.
- 활성값 승격은 기존 workspace ceiling 안에서 처리한다. 32,768행 전체 prefill에서는
  최대 512 MiB, 8,192행 TP4 sender에서는 최대 128 MiB의 FP32 입력이다.
  Packet workspace 설명에도 sender FP32 임시 공간을 명시했다.

## 검증

로컬 환경: RTX 5050 / SM120, PyTorch 2.13.0+cu130, Python 3.12.
실제 GLM 체크포인트 없이 합성 BF16 gate/activation과 CPU model fixture를 사용했다.

- `gpu_tests.log`: **21개 통과**. Decode 1/7/8/16/28행과 prefill 2,304/9,216행의
  FP32 로짓, expert IDs, weights가 IEEE FP32 참조와 정확히 일치한다. 입력을 바꾼 CUDA Graph도 일치한다.
- `high`/`medium` matmul 설정과 BF16 autocast 하에서도 FP32 low bits를 가진 입력·가중치의
  결과가 IEEE 참조와 정확히 일치한다. 호출 후 외부 precision/autocast 설정도 유지된다.
- 8,193~32,768행 packet 복원 결과와 top-k/weights가 일반 gather 후 FP32 projection과
  비트 단위로 일치한다. 8,193/9,216/32,768행에서는 sender shard와 full projection도 일치한다.
- `targeted_cpu_tests.log`: **48개 실행, 47개 통과, CUDA 전용 1개 skip**.
  모든 행 수의 FP32 resident 선택, 메모리 예산, native 부팅 증거, packet forward, native 선행 빌드를 포함한다.
- `merge_tests.log`: 최신 main 병합 후 **73개 실행, 72개 통과, CUDA 전용 1개 skip**.
  새 decode-topk와 router-fp32를 모두 선행 빌드하도록 목록 충돌을 해결했다.
- 전체 CPU 회귀 검사는 실행 중이다. 최종 결과와 소스 해시는 `manifest.json` 및 각 로그에 기록한다.

GPU 테스트 중 allocator가 20 MiB 요청의 OOM을 한 번 기록하고 캐시 회수 후 재시도해 통과했다.
공유 로컬 GPU의 allocator 기록이며, GB10의 전체 모델 메모리 여유를 증명하는 결과는 아니다.

**GB10 TP4 전체 모델의 실행시간, DFlash 수용률, 생성 결과 및 최종 workspace peak는 이번에 측정하지 않았다.**
앞선 대화의 전체 시간 약 3%는 추정치이며 이번 변경의 실측 성능으로 인용하지 않는다.
앞서 실패했던 BF16 projection 대 FP32 참조 비교는 허용치를 완화하지 않고 FP32 경로에서
`rtol=0, atol=0`으로 통과했다.

## 재현

저장소 루트에서 CUDA PyTorch/Triton 환경을 사용한다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest tests.test_engine_decode_seven.RouterFp32Tests tests.test_engine_decode_residency.RouterResidencyTests tests.test_engine_kernel_contracts tests.test_engine_prefill_fp8_consumer -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/check.py --list
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_onepass*.py'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_step_tools tests.test_oracle_accuracy
```

이전 BF16 성능 스크립트는 당시 커밋의 API와 연산을 측정한 역사적 재현이다.
현재 FP32 기본값의 검증에는 위 명령과 이 폴더의 결과를 사용한다.
