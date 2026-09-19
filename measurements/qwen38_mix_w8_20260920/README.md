# Qwen3.8 디코드 믹서 W8A16 — 단일 GPU 증거 (2026-09-20)

상태: **기본 끔, PR 미개설, TP4 품질·실제 tok/s 판정 전**. 목표는 출력 품질을 유지하면서 C=1도 개선하고 C=2~4 전체 처리량을 높이는 것이다. C=1 하락 허용치는 최대 5%다.

기존 `--hc-fp8`(H6)은 BF16 입력도 양자화하며 발사가 늘어 플릿에서 느려졌다. 이 후보는 BF16 입력과 중간 반올림을 유지하고 가중치만 block-128 FP8로 저장한다. 다운+게이트, 업+평균의 두 발사를 유지한다. FP32 부분 내적 뒤에 블록 스케일을 적용한다. 원소마다 가중치 스케일을 푸는 첫 시도는 느려 폐기했다.

## 실제 가중치의 믹서 하나

- srv4 NVIDIA GB10, `st-engine:glm53`, PyTorch 2.13.0+cu132 / CUDA 13.2. 단일 GPU 레인, 플릿 통신 없음.
- 체크포인트 rank 3, 앞 8층의 두 사이트와 closing 1개, 총 17개. BF16 가중치 224,133,120 B, 패딩 포함 FP8+scale 133,726,080 B.
- 17개 사이트를 순회해 L2보다 큰 가중치 묶음을 읽는다. 무작위 BF16 입력 둘, 10개 순서 교대 cold 표본. 판별 수치는 실제 문장 생성의 품질 수치가 아니다.
- 재현: 각 트리에서 `ST_PROBE_GIB=8 REPO=$PWD bash bench/fleet.sh run --gpu --detach SESSION 10 NOTE -- bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_mix_w8 --output /cache/SESSION.json`.

| 행 | 기존 BF16 µs | W8A16 µs | 커널 속도 비 | FP8 recipe 최대 오차 | 원래 BF16 대비 최대 오차 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64.012 | 41.894 | 1.528 | 0.003165 | 0.044304 |
| 4 | 62.647 | 40.800 | 1.535 | 0.003145 | 0.051798 |
| 8 | 65.327 | 42.120 | 1.551 | 0.002841 | 0.098090 |
| 16 | 95.077 | 71.769 | 1.325 | 0.002941 | 0.055990 |

FP8 recipe를 BF16로 풀어 계산한 참조와는 2^-7 이내지만, 원래 모델과의 오차는 따로 존재한다. 출력 동등성·품질 통과로 해석하지 않는다. CUDA graph replay와 split scratch/arrival 재사용을 확인했다. 16행 시간 표본은 다른 폭보다 흔들림이 컸다. 전체 표본은 JSON에 있다.

| 세션 / ticket | 소스 | 결과·원시 |
|---|---|---|
| qwen-mix-w8-0920a / 17898318583987442 | f3a40c84 | 원소별 스케일 prototype: C1 66.39→148.53µs, 폐기. `probe-a.log/json` |
| qwen-mix-w8-0920b / 17898319924002194 | a7451bae | 위 표. `probe-b.log/json` |

## 실제 네 층 + MTP 그래프, A B B A

`qwen-mix-step-0920d`, ticket `17898326254083948`, source `f8f7cf1c`, srv3 GB10, image `st-engine:glm53`. rank 2의 실제 가중치, 타깃 층 4/5/6/7과 MTP, `OneRankComm`으로 **TP 통신을 대체**했다. C=1..4, 6블록, K=3. 6GiB 프로세스 상한. peak 4.55GiB→4.64GiB. 실패/누락 그래프 없음. 원시 `step-d.log/json`.

재현: `ST_PROBE_GIB=6 REPO=$PWD bash bench/fleet.sh run --gpu --detach SESSION 12 NOTE -- bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_step_mix_w8 --output /cache/SESSION.json`.

| C | 그래프 | 기존 두 판 µs | 후보 두 판 µs | 평균 시간 변화 |
|---:|---|---|---|---:|
| 1 | target | 2543.8, 2554.4 | 2397.9, 2422.5 | −5.45% |
| 1 | draft | 4334.7, 4414.0 | 4160.6, 4274.5 | −3.58% |
| 2 | target | 2677.1, 2696.9 | 2520.9, 2549.5 | −5.65% |
| 2 | draft | 4507.7, 4517.7 | 4260.9, 4377.4 | −4.29% |
| 3 | target | 4042.5, 4161.0 | 3834.0, 3821.1 | −6.68% |
| 3 | draft | 5072.4, 4916.1 | 4676.1, 4814.1 | −4.99% |
| 4 | target | 4227.4, 4246.9 | 4073.3, 4063.3 | −3.98% |
| 4 | draft | 5098.9, 5144.2 | 4714.4, 4871.3 | −6.42% |

호출 수는 각 폭·그래프에서 동일하다. 서빙 옆에서 잰 축소 그래프이므로 전체 48층의 실제 출력 tok/s를 뜻하지 않는다. 두 판의 순서를 바꾸었지만 모든 부하·열 변동을 배제한 것은 아니다. `served_loop`의 토큰은 불완전 모델이 고른 것이므로 출력 품질이나 수용률 증거로 쓰지 않는다.

앞선 `qwen-mix-step-0920c`는 메모리 대기 뒤 준비 스냅샷 불일치로 멈췄다. 지원되는 새 제출로 d를 만들었고, declared/process 한도를 모두 6GiB로 맞췄다. 다른 임대는 해제하지 않았다.

## 서빙 연결

`--hc-w8a16` / `ST_HC_W8A16=1`, 기본 꺼짐. 1..16행의 타깃·MTP 믹서만 사용한다. 더 큰 프리필은 기존 BF16 경로다. 부팅 때 실제 사이트의 recipe 수치 검사를 하고, `hc_fp8`과 동시 선택하면 실패한다. 주소 기반 prefetch도 사용 가중치의 바이트 범위를 따르게 했다. calibration identity를 구분해 기존 모델 분포의 Hessian을 조용히 재사용하지 않는다.

후보는 원본 BF16 가중치도 유지해 전체 모델의 추가 메모리가 필요하다. full-model peak, 출력 hash·검색 품질·한국어 오염·수용률·tokens/step, C1 실 tok/s, C2~4 aggregate tok/s는 아직 판정하지 않았다. TP4 비교 계획은 [별도 기록](../qwen38_decode_ab_20260920/README.md)에 있다.
