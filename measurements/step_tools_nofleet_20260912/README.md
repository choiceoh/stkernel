# 노플릿 스텝 측정 도구 — 이 맥에서 잰 숙주 비용 (2026-09-12)

4박스를 잡지 못한 상태에서 스텝/레이턴시를 잰다는 질문에 답하는 도구 셋과, 그 도구로
**이 노트북(ost, macOS 26.6 arm64, CPU — GB10 이 아니다)** 에서 직접 잰 숫자.
원장 규칙 6 을 이 항목에도 적용한다: 아래 숫자는 **형상이 다른 상한/시뮬레이션이지
엔진 판정이 아니다**(D17 유지 — 판정은 플릿 onepass 두 판).

## 도구 (bench/)

| 파일 | 잰 것 | 함정 |
|---|---|---|
| `step_sim.py` | 진짜 `engine/base/runner.Runner`·스케줄러·`Ring` 을 장치 없는 모델로 돌려 스텝당 숙주 ms(종류별 med/p95). `--device-ms` 는 장치를 직렬 단일 서버 스레드로 모델링해 `decode_async` 겹침을 실제로 만든다 | 커널 시간이 없다 — 서빙 속도와 무관 |
| `step_peek.py` | 살아있는 부팅의 `/metrics` 관측(요청 전송 없음, 큐/리스 미접촉): 창별 step/s·수용률·`st:step_seconds`/TTFT 히스토그램 차 | 남의 부팅을 보고 있다 — 독점성·판정 없음 |
| `step_replay.py` | 저장 증거 재계산: `steps-*.ring`, onepass jsonl, bracket 다리, peek 샘플; onepass 기록 둘 이상이면 재부팅 없이 델타 | 재분석이지 재측정이 아니다 |

## 이 맥에서 잰 숫자

`sim_default.json` — 캘리브레이션(장치 0, C=1): decode med **0.039 ms/스텝**,
프롬프트 2K/32K/128K 워크로드(생성 256)에서 decode med 0.039~0.044 ms, prefill chunk
med 0.042~0.05 ms, 루프 캐던스 16,732 step/s. 실측 세계(~20 step/s, 스텝 50 ms)에서
**숙주는 병목이 아니다**를 코드 수준으로 확인.

`sim_meta.json` — `StepMeta.build` med **6.3 µs/스텝**(128K 컨텍스트, draft 3).
커널 호출자의 flat array 비용, 루프와 별도 측정.

`sim_device50.json` — 장치 50 ms 주입(직렬 서버 + depth 2 겹침): 측정 캐던스
**18.76 decode step/s**(이론 상한 20), 숙주 여유 **99.9%**. ring 의 decode in-flight
med **105.2 ms ≈ depth(2) × 50 ms** — "async 스텝의 ring wall 은 launch→readback
in-flight 이라 depth 대기만큼 길다"는 해석 규칙의 실증.

`replay_st_onepass_20260912.txt` — 실제 플릿 증거
(`../st_onepass_20260912_0746/result.jsonl`)의 재계산: decode 창 84개 med
**10.96 step/s**, 수용률 46.2%, tokens/step 3.309 — 원본 증거 README 와 동일한
숫자를 노트북에서 재생성.

`steps-sim-*.ring` — 시뮬레이션이 남긴 표준 스텝 링(`record.DeathDump` 형식,
`replay_ring_d50.txt` 가 다시 판 것).

## 검증

- `python3 -m unittest discover -s tests -p 'test_step_tools.py'` — 13 케이스 통과
  (GPU·네트워크 없음; peek 파싱·창 계산·히스토그램 보간은 순수 함수로 검증).
- `tools/check.py --pattern 'test_step_*'` — 1 ok.
- peek 의 네트워크 루프는 로컬 가짜 `/metrics` 서버로 검증(스크랩당 20스텝 증가 →
  39.7 step/s 관측). 이 날 노트북에서 플릿 헤드(10.10.10.2)는 도달 불가 — 실물 관측값은
  아직 없다.
