# ST 최적화의 모델 의존도 — 분류 기록 (2026-09-17)

운영자 질문은 이것이었다. "st엔진과 st커널 들의 최적화 중에 모델관련 없이 사용가능한게 몇%나 되지". 이 기록은 실측이
아니라 머지된 PR 의 분류다. 해석과 결론은 [`engine/MODEL_DEPENDENCE_20260917.md`](../../engine/MODEL_DEPENDENCE_20260917.md)
에 있고, 여기에는 재료와 재현 절차를 둔다.

## 무엇을 어떻게 분류했나

- **대상.** `3398a72b`(#1070)까지 main 의 first-parent 커밋 중 2026-09-10 이후 `engine/` 을 건드린 398 건이다.
  squash PR 255 건, merge PR 141 건, 직접 커밋 2 건이고 날짜는 09-11 ~ 09-17 이다. 그 뒤의 #1072 는 `tests/` 만 바꿨다.
- **읽은 것.** `collect.py` 가 건마다 커밋 메시지와 diffstat 을 뽑는다. merge PR 은 PR 제목, 브랜치 커밋 제목,
  `^1..` diffstat 을 쓴다.
- **분류.** 에이전트 8 개가 50 건씩 나눠 같은 규칙 [`rubric.md`](rubric.md) 로 판정했다. 메시지만으로 모호한 건은
  `git show` 나 기록 README 를 더 읽었다. 건마다 적은 칸은 다음과 같다.
  - `opt`: 최적화인가.
  - `status`: default / optin / rejected.
  - `scope`: 모델 의존도. 값은 아래 표.
  - `mixed`: 섞인 다른 scope.
  - `feature`: 무엇을 최적화했나.
  - `hw`: GB10/SM121/TP4 RoCE 전용인가.
  - `measured`: 근거 종류. e2e / component / none.
  - `why`: 판정 이유 한 줄.
- **scope 다섯.**

  | scope | 뜻 |
  |---|---|
  | U | 모델의 구조나 형상에 기대지 않는다 |
  | S | 모든 모델에 있는 연산이지만 타일·행 수·컴파일 인스턴스를 GLM-5.3 셀에서 정했다 |
  | F | 그 특징이 있는 모델만 쓴다. MoE, MLA/DSA, 인덱서, KDA, mHC, 드래프터 |
  | G | GLM-5.3 체크포인트에 묶였다 |
  | O | 다른 모델 전용이다 |

- **모델별 이식.** 사람 판단이 아니다. `summarize.py` 가 F·S 건의 feature 를 그 건이 최적화한 레인에 잇고,
  `engine/kernels/cells.admission()` 을 그 모델의 형상에 돌려 분류한다. 형상은 `tests/test_engine_kernel_shape.py` 의
  Qwen3.8·DeepSeek-V4.1 픽스처이고, GLM-5.3 은 `kernel_shape.MEASURED` 다.

  | 레인 판정 | 분류 |
  |---|---|
  | admitted | 그대로 |
  | unmeasured, 같은 커널 | 재측정 |
  | glue | 어댑터 경유 |
  | refused, 다른 커널, 없는 레인 | 안 닿음 |

## 파일

| 파일 | 내용 |
|---|---|
| `rubric.md` | 에이전트에게 준 판정 규칙 전문 |
| `collect.py` | 항목 목록 `items.json` 과 판독용 `batch_<n>.txt` 를 git 에서 다시 만든다 |
| `classification.jsonl` | 398 행. 항목 정보(hash·date·kind·pr·subject)와 판정 칸 |
| `summarize.py` | 비중·흔들기·특징표·하드웨어·근거·모델별 이식·U 의 코드 위치·커널 코드량 → `summary.txt`, `--table` → `table.md` |
| `summary.txt` | 위 출력. 문서의 수치는 모두 여기서 왔다 |
| `table.md` | 최적화 212 건의 PR별 표. scope 별, 기각 포함 |

## 재현

```
python3 measurements/st_model_dependence_20260917/collect.py /tmp/st-model-dependence   # 항목·판독 파일 (git 만)
python3 measurements/st_model_dependence_20260917/summarize.py > measurements/st_model_dependence_20260917/summary.txt
python3 measurements/st_model_dependence_20260917/summarize.py --table
```

- `collect.py` 의 `items.json` 과 `batch_3.txt` 는 분류에 쓴 파일과 바이트 단위로 같다.
- `summarize.py` 의 모델별 이식 절은 `engine.kernels.cells` 와 테스트 픽스처를 임포트한다. torch 없이 돈다.
- 분류 자체는 에이전트 판독이라 다시 돌리면 경계 건이 달라질 수 있다. 흔들기 범위는 `summary.txt` 의 scope shares 절에 있다.

## 알려진 한계

- **상태 칸.** 일부 배치만 현재 main 의 기본값과 대조했다(#812·#813·#817·#857·#862·#863·#865 등). 나머지는
  머지 시점 메시지를 따랐다. rejected 8 건을 빼는 것 말고는 비중에 쓰지 않는다.
- **정확도 PR.** 규칙이 "저비트 레인을 가능케 하는 GPTQ 팩" 을 최적화로 쳤다. 그래서 정확도가 목적인
  GPTQ·스무딩 4 건(#650·#659·#661·#669)이 S 에 들어갔다. 흔들기 행이 이를 뺀다.
- **경계 U 7 건.** 이득이 한 계열의 커널에만 떨어진다(#981·#983·#580·#540·#590·#549·#553). 흔들기 행이 이를 F 로 옮긴다.
- **이식 PR.** vLLM 시절 오버레이에서 이식된 커널 최적화(메가커널·b12x 정적 커널·SF6·Q0 등)는 이식 PR
  (#541·#591 등) 한 건으로만 들어왔다. 개별로 세지 않았다.
