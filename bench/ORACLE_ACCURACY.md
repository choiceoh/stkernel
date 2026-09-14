# ST 오라클 정확도와 검증 기록

2026-09-14: 요청 형상·종료 토큰·계수 선택·비교 통계를 수정했다.
이번 변경은 **계산의 오류를 고치는 것**과 **별도 기록의 예측 오차를 드러내는 것**이다.
모든 새 빌드의 tok/s나 TTFT가 일정 오차 안에 들어온다는 판정은 아니다.

## 바뀐 계산

| 조건 | 이전 | 수정 |
|---|---|---|
| K=0 / 수용률=0 | K=5 / 45%로 대체 | 관측값 0 유지 |
| 1토큰 응답 | 프리필 뒤 불필요한 디코드 | 프리필에서 완료 |
| 생성 한도 직전 speculative 묶음 | 초과 토큰까지 집계 | 남은 토큰만 커밋 |
| 비동기에서 이미 종료된 행 | 수용량 분모에 포함 | 실제 활성 행-스텝을 별도 집계 |
| 디코드 phase step/s | 항상 C1·ctx=0의 역수 | 실행한 폭·문맥의 모형 시간 합으로 계산 |
| 클라이언트 채널 보정 | 서버의 문맥 계단이 덮어씀 | 클라이언트의 문맥 계단 사용 |
| C4에서 얻은 기본 스텝 시간 | C1 기본값처럼 사용 | `decode_reference_width=4`로 기준 폭 보존 |
| C4 요청 | 순차 실행 또는 별도 배열 무시 | 동시 파동 전체가 끝난 뒤 다음 파동 입장 |
| 프롬프트 길이 | 컨텍스트 요약의 마지막 길이 | 요청의 실제 길이 우선, 대체 출처 기록 |
| warm TTFT 비교 | 시뮬 쪽은 cold까지 포함 | 양쪽에서 동일한 warm 요청 집합 비교 |
| C4 전체 출력률 | 비동기 drain까지의 벽 시간 | 첫 도착부터 마지막 응답 완료까지 |
| CPU 첫 요청 | 메모리 계측이 torch를 지연 import | 시뮬레이터 인스턴스의 메모리 계측 끔 |

`Recorder`의 서빙 기본값은 메모리 계측 활성 상태다. CPU 오라클이 사용하는
인스턴스에만 `memory_sampling=False`를 준다. 전역 상태나 GPU 컨텍스트를 바꾸지 않는다.

## 보정 재현과 별도 검증

기존 명령은 비용을 같은 기록에서 폴딩한다. 결과에는 `reconstruction`이 붙고,
보정 입력·그 파생값·재구성 값을 분리한다. cold−warm으로 보정한 cold TTFT나
수용률에서 만든 tokens/step을 독립적인 예측 성공으로 세지 않는다.

```sh
python3 bench/storacle.py sim --against recorded.jsonl --json --validation-output reconstruction.json
```

별도 기록으로 검증하려면 비용을 먼저 고정한다. 검증 기록의 요청 길이와
생성 길이는 작업 조건으로 사용하고, 비용·K·수용률은 보정 자료에서 가져온다.
이는 답변의 생성 길이나 품질을 예측하는 기능이 아니다.

```sh
python3 bench/storacle.py sim --fit-from calibration.jsonl --against validation.jsonl \
  --json --validation-output holdout.json
```

`--fit-from`은 파일의 마지막 C1 기록을 선택한다. 같은 관측 자료의 이름만
바꾼 복사본도 검증 자료로 거부한다. 엔진 소스·이미지·노브가 다르면
`cross_runtime_transfer`로 표시하고 바뀐 식별자를 저장한다. 다른 빌드로 계수를
옮긴 결과를 같은 실행 조건에서 확인한 정확도와 혼동하지 않는다.
`--compose` 또는 `--cost-json`도 `--against`에 적용할 수 있으며 재보정하지 않는다.

JSON에는 보정/검증 자료 해시, 실행 식별자, 프롬프트 길이의 출처, 파동 ID,
요청별 도착·완료 시각, 커밋 토큰, 모형 시간 합, 모든 비교 행과 제외 이유를 남긴다.
기존 `confidence`는 결측 계수에 따른 휴리스틱이며 **실측 예측 정확도는 아니다**.

## C4와 부족한 자료

최근 원패스의 `requests`/`c4`와 과거 `c1_requests`/`prefill_c1`/`decode_c1` 형식을
읽는다. C4 파동은 네 요청이 함께 입장한다. C1 카운터를 C4 측정값으로 복사하지
않으며, 폭 계수가 없으면 C4 비교를 제외하고 이유를 기록한다.

구성 모형은 `--compose`로 지정한다. 실측한 추가 행 비용이 있다면
`--decode-ms-per-row`로 명시한다. C4 시간으로 보정한 경우 그 값은 C4에 고정되고,
C1 시간으로 보정한 경우에는 C1에 고정된다. 음수 스텝 시간을 만드는 조합은 거부한다.
`--fold-width C1.jsonl C4.jsonl`은 같은 소스/이미지/노브·K·프롬프트 형상·측정 채널과
완결된 파동을 요구한다. JSONL의 최신 기록을 선택한다.

실패·오염으로 무효 표시된 자료, 잘못된 rate, 생성 토큰 수가 빠진 요청을
임의의 기본값으로 검증하지 않는다. 비교 가능한 결과가 하나도 없으면 종료 코드 2다.
원패스의 실제 청크 계약, prefix-cache 상태, 준비 여부와 맞는 비용 자료를 사용해야 한다.

## 검증 범위

기존 오라클 58개와 새 정확도 계약 19개를 CPU에서 검증한다. 전용 CI 단계에도
두 모듈을 연결했다. 실제 GPU 성능·수용률·답변 품질은 이 검사 범위 밖이다.

```sh
python3 -m unittest tests.test_step_tools tests.test_oracle_accuracy
python3 measurements/st_oracle_accuracy_20260914/reproduce.py
```

[수정 전후의 정확한 반례와 별도 기록 결과](../measurements/st_oracle_accuracy_20260914/README.md).
