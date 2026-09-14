# 오라클 정확도 수정의 CPU 증거

기준: `2ac7de6f`의 `bench/step_sim.py`. GPU·큐·부팅·배포 없이 수행했다.
`reproduce.py`는 기준 Git 소스와 현재 구현을 읽어 양쪽의 지연과 장치 메모리
계측을 끈다. 따라서 아래는 **정확한 계산 반례**이며 마이크로벤치 속도가 아니다.

| 반례 | 이전 | 수정 | 독립적으로 계산한 정답 |
|---|---:|---:|---:|
| 기록된 K=0 | 5 | 0 | 0 |
| 기록된 수용률=0 | 45% | 0% | 0% |
| 클라이언트 문맥 계수 | 1 / 2 ms | 10 / 20 ms | 10 / 20 ms |
| 1토큰 응답의 디코드 횟수 | 1 | 0 | 0 |
| 응답 길이 2·10의 디코드 토큰 합 | 24 | 10 | `(2−1)+(10−1)=10` |
| 혼합 폭 phase step/s | 500 | 89.66 | `스텝 수 / Σ(폭별 시간) = 89.655…` |

[regressions.json](regressions.json)에 소스 해시와 모든 값을 보관했다.
과거 `c4-20260912T232937.json`에서 기존 경로는 요청을 찾지 못했다. 새 경로는
C4 파동 3개·총 12요청을 읽으며 각 요청의 실제 2,465/2,493/2,498 토큰 길이를 유지한다.
그 기록 전체의 생성·레이턴시를 이번에 실측하거나 전부 시뮬레이션한 것은 아니다.

## 별도 기록을 넣었을 때의 한계도 기록

`onepass-a`로 비용을 고정하고 `onepass-h`의 2K 요청 세 개를 입력한 결과가
[holdout_report.json](holdout_report.json)이다. [holdout_2k.json](holdout_2k.json)은
원본 H 기록에서 `ctx=2000`인 requests/prefill만 고른 자료이며 다른 측정 필드는
원본 그대로다. 원본 위치와 선택 규칙도 파일에 들어 있다.

이 두 기록은 **엔진 소스·이미지·노브가 다르다**. 새 보고서는 이를
`cross_runtime_transfer`로 분리한다. 비교 행 7개의 절대 상대오차 중앙값은 약
175%, 최대 약 642%였다. 특히 보정 자료의 수용률과 준비 비용이 검증 빌드에
맞지 않는다. 다른 빌드의 계수를 그대로 옮겨도 정확하다는 근거로 사용할 수 없다.
이번 변경으로 실제 서빙 예측 오차가 일정 비율 이하가 됐다고 주장하지 않는다.

```sh
python3 measurements/st_oracle_accuracy_20260914/reproduce.py
python3 bench/storacle.py sim \
  --fit-from measurements/st_native_20260912/onepass-a/result.jsonl \
  --against measurements/st_oracle_accuracy_20260914/holdout_2k.json \
  --no-calib --json --validation-output build/oracle-holdout.json
```

정확한 산술·형상 회귀는 `tests/test_oracle_accuracy.py` 19개, 기존 도구 검사는
`tests/test_step_tools.py` 58개다. 로컬 macOS에서 함께 시도한 Linux loader 검사는
`os.O_DIRECT` 부재로 8개가 실행 오류였으며, 해당 경로는 Linux 엔진 CI로 확인한다.
