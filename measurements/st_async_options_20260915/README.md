# 요청 옵션을 사용하는 비동기 디코드

## 적용 범위

기준 체크아웃은 `3acae017`, 작업 브랜치는 `codex/async-request-options`다.
`seed`, presence/frequency/repetition penalty, `logit_bias`, `logprobs`, 미충족 `min_tokens` 때문에
비동기 체인을 비우던 조건을 제거하고 각 옵션을 장치 상태와 연결했다. 별도 활성화 플래그는 없다.

| 요청 | 변경 후 경로 |
| --- | --- |
| 요청별 seed | 기존 요청별 난수 키와 일치하는 device nonce를 사용. 행 위치·다른 요청의 난수 소비와 독립 |
| 페널티·logit bias | 블록 전체의 FP32 로짓을 한 Triton launch로 변환. speculative prefix와 실제 출력 이력을 구분 |
| min_tokens | 생성 수 + 블록 내 위치를 기준으로 EOS/stop IDs 차단. 호스트가 최소 길이 충족을 확인하면 기본 샘플러로 복귀 |
| logprobs | 선택 토큰·상위 후보만 연속 핀 버퍼로 복사하고 기존 이벤트 뒤에서 공개 |
| 문법·reasoning 경계 | 호스트 경계 유지. 로짓 변환과 logprobs 처리는 블록 단위로 병합 |
| 기본 greedy·중립 옵션 | 기존 캡처 샘플러와 4회 버스트 유지. 옵션 요청이 떠난 뒤 일반 행도 이 경로로 복귀 |

페널티·bias·logprobs·미충족 minimum은 **깊이 2의 일반 비동기 체인**을 사용한다.
문법 matcher가 커밋 결과를 알아야 다음 마스크를 만들 수 있으므로 문법 요청은 아직 완전 비동기가 아니다.
문법의 speculative walk, 거절된 draft의 rollback, committed token의 advance는 기존 규칙을 따른다.
reasoning 종료 토큰 강제와 min_tokens가 충돌하면 기존처럼 min_tokens를 우선한다.

이력의 seen은 프롬프트와 출력을 포함하고 frequency/presence count는 출력만 센다.
GPU commit 후 클리핑된 토큰만 count에 추가하므로, 호스트가 아직 읽지 않은 두 번째 스텝도 정확한 이력을 사용한다.
새 행의 최초 이력은 CPU에서 한 번 구성해 핀 메모리로 비동기 전송하고, 행 합류·재정렬은 장치의 진행 상태를 보존한다.
새 턴은 옵션 정책과 stop-ID 캐시를 무효화한다.

## 검증

- [CPU 회귀 로그](cpu-tests.log): 253개 실행, **226개 통과 / 27개 건너뜀**. CUDA 전용 항목 등을 CPU 실행에서 제외했다.
- [CUDA 및 실제 문법 로그](cuda-tests.log): **30개 통과**.
- seed=0, 음수/64비트 초과 seed, 혼합 행·재정렬, 기존 VERIFY/FRESH 난수 및 기각 샘플링과의 일치.
- 거절 draft·중복 토큰·EOS·생성 한도·유령 스텝·미수거 두 스텝·행 합류/이탈·새 턴 캐시 무효화.
- BF16/FP16/FP32 입력, 어휘 조각의 global ID, 비연속 입력/출력과 주변 메모리 보존.
- CUDA Graph replay에서 변경된 생성 수·draft·이력을 실제로 다시 읽는지 확인.
- xgrammar의 실제 JSON matcher와 CUDA mask로 `{}` 및 EOS 출력, rollback과 commit, logprobs 확인.
- TP1과 어휘 분할 형태의 부팅 준비 검사. 새 FP32 샘플러 변형과 강제 토큰 커널도 준비한다.
- `git diff --check` 및 변경 Python 파일의 컴파일 검사 통과.

환경: RTX 5050 8 GB, PyTorch 2.13.0+cu132, Triton 3.7.1, CUDA 13.2,
xgrammar 0.2.6, transformers 5.17.0, tokenizers 0.23.2.
테스트 환경은 별도 `/tmp/st-async-options-venv`에 구성했다.
모델 가중치를 적재하지 않았으며 GPU 메모리 할당 한도는 전체의 12%로 제한했다.
로그의 synthetic tokenizer RAW 경고와 의도적으로 발생시킨 release/disk 오류 메시지는 테스트 실패가 아니다.

## 처리 단계 측정

[원시 수치와 소스 해시](component-rtx5050.json), [재실행 프로브](../../probes/engine_sampling_options.py).

어휘 154,880, K=7, BF16 입력, 요청 1/2개, 2K 프롬프트 이력을 사용했다.
기존 `process_logits`의 위치별 실행과 새 블록 변환을 비교했다.
각 arm을 준비한 뒤 A/B/B/A 순서로 4회, 회당 10번 실행했다.
logprobs는 상위 후보 5개를 요청하고 같은 GPU에서 기존 함수와 일괄 함수의 점수·순서 일치도 검사했다.
아래 시간은 **호스트 호출부터 CUDA 완료까지의 반복당 wall time 중앙값**이며 JSON에는 stream 시간도 있다.

| 처리 단계 | 요청 1개: 기존 → 변경 | 요청 2개: 기존 → 변경 |
| --- | ---: | ---: |
| 옵션 로짓 변환 | 4.189 → **0.048 ms** | 6.276 → **0.078 ms** |
| logprobs 수집 | 6.528 → **0.704 ms** | 9.162 → **0.513 ms** |

변환 로짓의 최대 절대 오차는 **2.384185791015625e-7**이고 이 프로브의 모든 greedy 선택이 일치했다.
부동소수점 연산 병합을 끄고 기존 변환 순서를 유지했다.
다른 요청의 seeded 출력 전체까지 모든 환경에서 비트 동일함을 보장하는 결과는 아니다.

공유 데스크톱 GPU에서 측정했으므로 시스템 부하에 따른 시간 변동이 있다.
이 수치는 로짓 변환과 logprobs 수집의 **구성 요소 측정**이며, 모델 forward·TP4 통신·요청 전체 지연 시간을 포함하지 않는다.
엔진 전체 tok/s 개선율이나 구조화 출력의 최종 응답 속도로 해석하지 않는다.

## 플릿 상태와 남은 검증

[플릿 조회 기록](fleet-status.log)에서 다른 실험의 boot 점유, 읽을 수 없는 lease,
`fleet_single.py` 등 보조 모듈 누락 및 6개 대기 작업을 확인했다.
이번 작업은 점유자를 변경하거나 플릿을 재기동하지 않았다.
**TP4 모델 부팅·옵션 요청 혼합 실행·전체 onepass 성능/품질 검증은 미실행**이다.
코드의 기본 적용과 엔진 전체 성능의 최종 채택 여부는 구분한다.

다음 플릿 검증은 기본 채팅과 seed/penalties/min_tokens/logprobs/JSON 요청을 C=1·2에서 섞어
샘플링 옵션 준수·토큰 순서·수용률·출력 품질을 먼저 확인한 뒤, 동일 조건의 전체 onepass와 비교해야 한다.
`engine/CHARTER.md` D17에 따른 성능 최종 채택은 그 결과가 필요하다.

## 재실행

저장소 루트에서 위 버전의 Python 환경으로 실행한다.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python3 -m unittest \
  tests.test_engine_device_options tests.test_engine_pipeline tests.test_engine_draws \
  tests.test_engine_burst_decode tests.test_engine_sampling_options tests.test_engine_grammar \
  tests.test_engine_sampling tests.test_engine_release tests.test_engine_replay_metadata \
  tests.test_engine_runtime_memory tests.test_engine_draft_tuning_integration \
  tests.test_engine_charter tests.test_engine_kernel_common -q

OMP_NUM_THREADS=2 python3 -m unittest \
  tests.test_engine_device_options.CudaPolicyTests \
  tests.test_engine_device_options.CudaAsyncOptionTests \
  tests.test_engine_draws.TensorAgreementTests tests.test_engine_grammar.GrammarTests -q

OMP_NUM_THREADS=2 python3 probes/engine_sampling_options.py --vocab 154880 --spec 7 --iterations 10
```
