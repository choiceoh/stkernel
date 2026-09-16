# 실제 캡처의 W4 → FP8/BF16 단일 연산 비교

**이번 16사례에서는 교체할 후보가 없다. W4를 유지한다.** 5개 block의 dense 연산 30개를 각각 FP8 RTN, BF16으로 하나씩 교체했다. 60개 후보 모두 사례별 연속 수용 길이가 baseline과 정확히 같았다. 개선과 악화가 평균에서 상쇄된 결과도 아니다. 모든 후보의 지연·활성 가중치 바이트 점추정치는 증가했다.

| 단일 연산 교체 | 후보 | 평균 연속 수용 길이 | 사례별 개선 / 악화 | 추가 proposal 시간 | 추가 활성 가중치 / rank |
|---|---:|---:|---:|---:|---:|
| W4 baseline | — | 2.875 | — | — | — |
| FP8 RTN | 30 | 모두 2.875 | 0 / 0 (480회 비교) | +3.6~95.1 µs | +1.74~10.48 MiB |
| BF16 | 30 | 모두 2.875 | 0 / 0 (480회 비교) | +23.0~165.8 µs | +5.73~34.48 MiB |

480회 비교는 **서로 독립인 480표본이 아니다.** 같은 16사례를 30개 후보에 반복 사용했다. 이번 표본에 채택 근거가 없다는 결론이며 다른 요청·상태에서의 개선 가능성을 배제하지 않는다.

**2.875는 0.001토큰 정밀도로 효과가 없음을 측정한 수치가 아니다.** 각 사례의 연속 수용 길이는 0~7의 정수이고 총 46토큰 / 16사례 = 2.875다. 이 표본에서 평균의 최소 비영 변화는 1/16 = **0.0625토큰**이다. 다만 반올림에 가려진 것이 아니라 아래와 같이 사례별 정수도 모두 같았다. 현재 증거는 “효과 없음”의 확정이 아니라 “이 작은 표본에서 개선을 관측하지 못함”이다.

## 수용 이득과 비용

baseline 연속 수용 길이는 `[0, 2, 7, 7, 1, 7, 0, 3, 0, 2, 7, 7, 0, 1, 0, 2]`다. K=7을 모두 수용한 5사례는 개선 여지가 없고 나머지 11사례에서도 이득이 없었다. 모든 FP8/BF16 후보가 같은 배열을 유지했다.

- FP8은 baseline 대비 총 232개 draft 위치를 바꿨다. 230개는 첫 기각 뒤였고, 첫 기각 위치의 2개 변화도 여전히 정답과 불일치했다.
- BF16은 231개 위치를 바꿨다. 230개는 첫 기각 뒤였고, 첫 기각 위치의 1개 변화도 정답과 불일치했다.
- 이미 수용된 prefix는 한 번도 훼손하지 않았지만 더 길게 수용하게 만들지도 못했다. **토큰 변화량이나 국소 양자화 오차 감소를 수용 이득으로 대체할 수 없다.**

60개 후보 모두 적어도 한 사례의 실제 draft ID를 변경했다. 전혀 적용되지 않은 교체를 비교한 결과는 아니다. 첫 기각 위치가 바뀐 3건은 모두 `case-00006`의 첫 토큰으로, baseline `46645` → candidate `69782`였으나 정답은 `10759`였다. 따라서 수용 길이는 0으로 남는다.

사용자의 동일 수치 지적 후 실제 HTTP 응답에서 정답을 다시 잘라 누적 일치 곱의 합으로 독립 재계산했고, 960개 paired prefix가 모두 같은 것을 확인했다. 분석 대조군으로 한 사례의 첫 오답만 정답으로 바꾸자 평균 +0.0625를 검출했다. 이는 분석 계산의 대조군이며 추가 GPU precision 실험은 아니다. 표본은 요청별 첫 eligible step과 16 step 뒤에 치우쳐 있고, 5개의 0수용 사례는 모두 첫 캡처이며 정답 시작 3토큰도 같다. 다양한 기각 상태를 충분히 대표하는 표본으로 해석하지 않는다.

아래 범위는 같은 연산 종류의 layer index 0~4에 대한 관측값이다. 추가 시간은 각 사례에서 TP4 최장 rank 시간의 중앙값을 구한 뒤 candidate−baseline 차이를 16사례 평균한 값이다.

| 연산 | FP8 추가 시간 (µs) | BF16 추가 시간 (µs) | FP8 추가 MiB/rank | BF16 추가 MiB/rank | 수용 길이 변화 |
|---|---:|---:|---:|---:|---:|
| attention_conv projection | 3.62~17.98 | 27.01~31.21 | 1.75 | 5.75 | 모두 0 |
| MLP down projection | 47.73~52.37 | 81.87~85.78 | 5.24 | 17.23 | 모두 0 |
| MLP gate/up | 91.68~95.11 | 158.80~165.80 | 10.48 | 34.48 | 모두 0 |
| mlp_conv projection | 14.77~17.51 | 29.37~33.41 | 1.75 | 5.75 | 모두 0 |
| attention output projection | 19.91~25.08 | 23.03~29.64 | 1.74 | 5.73 | 모두 0 |
| attention QKV | 16.73~19.69 | 36.91~39.74 | 2.62 | 8.62 | 모두 0 |

이전 국소 오차 비교에서 가장 눈에 띄었던 **`layers.2.mlp.down_proj.weight`도 수용 이득은 0**이다. FP8은 +49.04 µs·+5.24 MiB/rank, BF16은 +81.87 µs·+17.23 MiB/rank였다. FP8 잔여 오차 때문에 이 reader의 효과를 놓쳤다는 가설도 이번 BF16 단일 교체에서는 뒷받침되지 않았다.

raw JSON의 `cost_screen`은 교체 후보끼리만 비교한다. 그 안의 `non_dominated`는 W4보다 유리하다는 뜻이 아니다. **W4를 포함하면 이번 점추정치에서 모든 후보는 수용 이득 없이 추가 지연·메모리가 든다.** 작은 시간 차이의 통계적 유의성은 판정하지 않았고 서빙 채택 후보도 지정하지 않았다. BF16은 `torch.nn.functional.linear`를 사용하는 진단 경로이며 최적화된 BF16 서빙 커널의 비용을 뜻하지 않는다.

## 측정 범위와 검증

- [실제 캡처](../live/README.md)의 C=1 greedy 요청 8개, 사례 16개를 사용했다. 짧은 문맥 8사례, 32K 4사례, 128K 4사례다. 각 사례에는 완전한 7토큰 greedy 정답이 있다.
- TP4에서 각 rank의 해당 reader 하나만 교체하고 나머지 block·head·selector를 다시 실행했다. baseline draft의 정확한 재현, 반복 실행 시 출력 안정성, context ring 불변 검사를 모두 통과했다.
- B/A/A/B를 사례마다 10회 수행해 각 arm의 timing 20개를 남겼다. CUDA graph 준비·컴파일은 측정 구간 밖이다. 원본 timing 배열을 보존했고, 중앙값·평균 차이·prefix 지표를 별도 CPU 스크립트로 다시 계산했다.
- baseline proposal 시간은 FP8 비교에서 2,604.7~2,645.0 µs, BF16 비교에서 2,611.4~2,647.9 µs였다. 이 범위 역시 reader별 16사례 평균이다.
- 시간은 **embedding이 미리 계산된 block/head/selector graph**에 한정한다. embedding, target verification, observe, 전체 decode step은 포함하지 않는다. **실제 tok/s와 완결 출력 품질은 이번에 측정하지 않았다.**
- 가중치 비용은 해당 reader의 활성 payload와 scale 차이다. replay는 양쪽 arm을 함께 보유하므로 실제 peak resident memory 실측이 아니다.
- 고정된 context ring, 단일 reader 교체 결과다. 여러 reader의 동시 변경, 후보를 적용한 장기 generation, FC/context projection과 selector 자체의 정밀도 변경은 비교하지 않았다.

이 결과로 단일 dense reader의 정밀도 승격을 추진하지 않는다. 추가 조사를 한다면 최초 기각이 발생하는 상태를 더 확보하고 FC/context 또는 복수 연산의 영향을 별도로 검증해야 한다. 현재 자료만으로 그 원인이나 효과를 확정하지 않는다.

## 실행 신원과 재현

두 비교 모두 **서버 엔진을 재부팅하지 않았다.** fleet에 등록한 standalone TP4 프로세스로 캡처된 drafter와 head만 로드했고, 완료 후 컨테이너를 정리하고 lease를 반납했다.

- controller/probe commit: `8f0a8dff23d8df7d0ac68912d57a2262037d7079`
- 캡처 및 replay engine commit: `4e6698a1a47ffe447c4293d3cfc3122065bc1c69`
- FP8 fleet session: `draftcompare2-0916`; BF16: `draftcompare-bf16-0916`
- image tag: `st-engine:bracket-4e6698a1a47f`; rank별 실제 image digest는 두 run에서 같으며 runtime JSON에 보존했다.
- OneShot: rails=2, inline_flags=true, consumer_max_elements=65536; PyTorch `2.13.0+cu132`.

결과: [FP8 raw](fp8.json), [BF16 raw](bf16.json), [사례별 재계산과 SHA-256](summary.json). 실행 신원: [FP8](fp8-runtime.json), [BF16](bf16-runtime.json). 완료·반납: [FP8](fp8-fleet.log), [BF16](bf16-fleet.log). 각 rank의 원본 로그도 같은 디렉터리에 보존했다.

저장된 결과 재검증(CPU만 사용):

```sh
python3 measurements/st_draft_sensitivity_20260916/audit_comparison.py
```

GPU 재실행은 srv2의 깨끗한 checkout에서 **새 session/output 이름**으로 fleet에 등록한다. 아래는 실행 형태이며 기존 결과를 덮어쓰는 명령이 아니다.

```sh
ST_IMAGE=st-engine:bracket-4e6698a1a47f bash bench/fleet.sh run --gpu --detach NEW_SESSION 12 \
  "TP4 fixed-state precision sensitivity" -- \
  python3 /ABSOLUTE/CHECKOUT/bench/draft_replay.py \
  --capture /home/choiceoh/expert-capture/draft-sensitivity-0916-7e62/captured \
  --checkpoint /home/choiceoh/models/GLM-5.3-Flash-DFlash2/model.safetensors \
  --output /NEW/OUTPUT.json --reader all --precision fp8-rtn --rounds 10 \
  --engine-revision 4e6698a1a47ffe447c4293d3cfc3122065bc1c69
```

`--precision bf16`으로 BF16 비교를 실행할 수 있다. admission/controller가 최신 main을 요구해도 `--engine-revision`이 캡처 시 engine source를 별도로 고정한다. capture source·checkpoint·prepared-state 해시가 다르면 probe가 결과 생성을 거부한다.
