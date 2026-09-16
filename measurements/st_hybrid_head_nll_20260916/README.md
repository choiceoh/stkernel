# GPTQ 하이브리드의 프리필 head NLL — PR #911 이 "Not measured" 로 남긴 비교 (2026-09-16)

원장 항목: [MEASUREMENTS.md](../../MEASUREMENTS.md) 의 `2026-09-16 — GPTQ 전문가 하이브리드는 프로덕션보다 나쁘다`.
**부팅 없음** — 2026-09-14 에 이미 찍힌 세 팔의 `head.jsonl` 을 짝지어 다시 읽었을 뿐이다.

PR #911 은 하이브리드 레이아웃(ModelOpt 라우팅 전문가 + GPTQ 코드 + BF16 dense MLP + unit 활성 스케일)을
채택하면서 마지막 줄에 이렇게 남겼다:

> **Not measured:** head NLL of the hybrid against Red Hat at the same commit. The operator adopted the layout
> without that comparison.

그 비교가 이 디렉터리다. 도구는 srv2 의 `~/expert-capture/head_compare.py`(저장소 밖) — 같은 코퍼스를 먹인 두
부팅의 `head.jsonl` 을 **위치 단위로 짝지어**(두 부팅이 같은 타깃 토큰으로 채점한 위치만 센다) 평균 NLL 차이와
문서 단위 부트스트랩 95% 신뢰구간을 낸다.

## 팔

| 팔 | 랭크 | 디렉터리 |
|---|---|---|
| RH (프로덕션) | `st-glm53-9391-up-gate-full` | `eval-A` (커밋 `e032a600`) |
| N (NVIDIA ModelOpt) | `st-glm53-nvidia-tp4-9391` | `eval-N` |
| HY (하이브리드) | `st-glm53-hybrid-gptq-v1` | `eval-HY` |

`eval-RH` 라는 빈 디렉터리가 9월 14일 01:04 에 만들어져 있고, `rh-main.log` 의 Red Hat 팔은 onepass 만 돌렸다 —
그래서 PR 이 "Not measured" 라고 쓴 것이다. **`eval-A` 가 곧 프로덕션 랭크의 팔**이라는 것은 아래 재현이 말한다.

## 결과 (fit 스플릿, 108 문서 / 27,622 위치)

| 비교 | ΔNLL | 95% CI | 해석 |
|---|---|---|---|
| N − RH | **+0.03444** | [+0.01427, +0.05485] | PR #911 의 "+0.034 nats worse than production" **정확히 재현** |
| **HY − RH** | **+0.05008** | **[+0.02920, +0.07174]** | **하이브리드가 프로덕션보다 나쁘다 — 0 을 포함하지 않는다** |
| HY − N | +0.01564 | [−0.00459, +0.03707] | BF16 dense 복원이 N 의 손해를 되찾지 못했다 |

held-out 스플릿(25·22 문서)은 부호가 같지만 CI 가 0 을 걸친다 — 문서 수가 적어 검정력이 없다.

읽기:

1. **하이브리드는 자기가 고치려던 것보다 나쁘다.** N 의 +0.034 를 BF16 dense MLP 로 되돌리려 만든 레이아웃인데
   결과는 +0.050 이다. HY−N 이 +0.016(CI 가 0 을 걸침)이라 "되찾은 게 없다" 가 맞는 읽기다.
2. **전문가 가중치 충실도가 2배 좋아졌는데도 그렇다.** PR #911 의 held-out 전문가 출력오차는
   RH 21.9% → ModelOpt 14.6% → **GPTQ from BF16 10.9%** 였다. 전문가 오차 절반이 예측을 더 나쁘게 만들었다.
   **오프라인 가중치 오차에서 서빙 이득으로 가는 추론은 여기서 정면으로 깨진다.**
3. **차이는 몸통에 있지 head 양자화에 있지 않다.** 각 부팅의 자기 hidden state 에 BF16 head 를 씌운 열이
   서빙 열과 소수점 다섯 자리까지 같다(+0.05011 vs +0.05008).

## 안 한 것 · 함정

- **커밋이 다르다.** `eval-A` 는 `e032a600`, HY 는 `st-glm53-hybrid.env` 의 릴리스다. PR 이 말한
  "at the same commit" 통제는 **여전히 없다** — 커밋 효과를 배제하지 못한다. 이 숫자는 "같은 코퍼스·같은 도구로
  짝지은 두 부팅의 차이"이지 랭크만 바꾼 차이가 아니다.
- **하이브리드는 변경 셋을 묶고 있다**(GPTQ 전문가 + BF16 dense MLP + unit 활성 스케일). 어느 것이 nats 를
  쓰는지는 이 비교가 못 가른다.
- 도구가 문서 번호가 어긋난다고 경고한다(133 vs 174 vs 186). 짝짓기는 **같은 타깃 토큰**을 요구해 막혀 있고,
  fit 스플릿은 세 비교 모두 108 문서 / 27,622 위치로 같은 자리에 떨어진다.
- **서빙 판정이 아니다.** tok/s·수용률·품질은 PR #911 에 있다(수용률·tokens per step 동일, 속도차는 부팅
  CV 1.7% 안, reasoning 6/9 → 5/9).

## 재현

srv2 에서, 부팅 없이:

```bash
cd ~/expert-capture
python3 head_compare.py --a eval-A --b eval-HY --feed feed-eval-HY.jsonl --names RH,HY
python3 head_compare.py --a eval-A --b eval-N  --feed feed-eval-N.jsonl  --names RH,N
```

원시 출력은 [head_compare.log](head_compare.log).
