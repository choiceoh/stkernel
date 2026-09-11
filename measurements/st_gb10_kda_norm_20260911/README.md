# GB10 KDA 출력 RMS norm + sigmoid gate 통합

KDA 출력의 FP32 변환·제곱·평균·rsqrt·weight 곱·sigmoid gate·BF16 저장을 하나의 Triton
커널로 합쳤다. 비교 기준은 앞선 상태 전치 제거 PR #561을 포함한 **`b0e50398`**이다.
실제 L0 KDA 블록의 1토큰/6토큰 지연이 각각 **4.57% / 4.35% 추가 감소**했고,
검사한 출력·상태 바이트는 기준과 동일했다.

## 변경과 GB10 실행 구성

`net._kda`의 기존 torch 표현식을 `engine.modules.linear_attention.kda_output_norm`에
독립 참조로 보존했다. `Lanes.kda_output_norm`이 참조와 실제 커널을 바인딩하며,
서빙은 `engine/kernels/kda/output.py`를 호출한다. 참조 호출로의 자동 대체는 없다.

GB10의 128차원 head 하나를 **1 warp / 1 CTA**가 처리한다. 각 thread의 레지스터에서
FP32 제곱합과 gate 계산을 끝내고, 마지막에 BF16으로 한 번 저장한다. 측정한 최종 커널은
**31 registers/thread, shared memory 0, spill 0**이다. small decode의 launch 비용을 줄이고,
큰 prefill에서는 UMA에 여러 FP32 중간 tensor를 쓰고 읽는 비용을 없앤다.

4~32행을 묶는 타일도 비교했다. 일부 중간 형상에서는 빨랐지만 reduction 순서가 바뀌어
BF16 결과가 달라졌다. 최종 경로는 1행·1 warp로 선택했다. 근사 sigmoid 대신 libdevice exp와
명시적 **`div.rn.f32`**를 사용한다. 이 runtime의 `libdevice.div_rn`은 `.ftz`로 컴파일되어
gate=-88 부근의 FP32 subnormal을 0으로 만들었다. non-FTZ 나눗셈은 BF16에 표현 가능한
작은 출력까지 보존했다. FP fusion도 꺼서 기존 연산 사이의 FP32 반올림을 유지했다.

초기 sweep 원본은 `initial-tuning.json`과 `initial-tuning-large.json`이다. 당시의 precise
후보는 libdevice의 FTZ division을 사용했으므로 최종 커널의 수치 증거로 쓰지 않는다.
큰 형상은 초기 768 MiB allocator 한도에 도달해 별도로 1.5 GiB 한도에서 측정했다.
최종 커널의 수치·시간·레지스터·프로파일러 결과는 `perf.json`과 회귀 검사에 기록했다.

## 실제 L0 KDA 블록

실제 rank 0의 L0 KDA 가중치 **70,492,480 bytes**와 seeded synthetic BF16 activations를
사용했다. 비교 기준 `net.py`를 git에서 추출해 별도 모듈에 로드하고, 기존 `_kda` 메서드와
현재 메서드를 같은 가중치에서 비교했다. projection·gate projection·conv·recurrent KDA·
출력 norm·출력 projection·상태 링 읽기/쓰기를 모두 포함한다.

| 실행 | 기존 | 변경 | 지연 감소 |
| --- | ---: | ---: | ---: |
| 1토큰 decode | 474.848 µs | 453.152 µs | **4.57%** |
| 6토큰 verify | 462.464 µs | 442.368 µs | **4.35%** |

같은 srv4 GB10에서 CUDA graph를 AB/BA 순서로 11회 번갈아 측정한 중앙값이다.
각 샘플 전에 동일 그래프를 한 번 replay했다. 측정 context는 4096, physical slot은 1이다.
TP4의 rank 0 계산을 분리했으며 collective는 항등 처리했다. mHC·MLP·나머지 layer는
포함하지 않는다. 단일 실행의 결과이며 시간대·기기를 달리한 성능 분포는 아니다.

prefill 64→verify 6→4개 draft 거절 후 재검증→decode를 검사했다. 그래프에서는
T=1/6, context=0/1/7/8/4095/4096, physical slot=1/2를 바꿨다. **16조건 모두 이전/현재
출력과 전체 상태 링 바이트가 같았고, 현재 eager와 graph replay의 바이트도 같았다.**
원본 시간, 가중치별 해시, frozen net/config 해시는 `real.json`에 있다.

## 출력 정규화만 분리한 측정

아래는 실제 public 함수의 출력 할당까지 포함한다. `warm`은 네 번 호출을 묶은 그래프의
호출당 시간, `evicted`는 64 MiB를 먼저 써서 L2를 밀어낸 후 한 번 호출하는 그래프 시간이다.
eviction 자체는 측정 구간 밖이다. 각 조건에서 AB/BA 11회 중앙값을 기록했다.

| 토큰 / 128차원 행 | warm 기존 → 변경 | evicted 기존 → 변경 |
| --- | ---: | ---: |
| 1 / 16 | 21.904 → 2.256 µs | 26.624 → 4.128 µs |
| 6 / 96 | 21.888 → 2.248 µs | 26.912 → 5.792 µs |
| 24 / 384 | 22.984 → 2.400 µs | 30.400 → 7.648 µs |
| 64 / 1,024 | 27.592 → 3.016 µs | 48.800 → 11.264 µs |
| 256 / 4,096 | 62.896 → 7.160 µs | 105.312 → 28.832 µs |
| 1,024 / 16,384 | 383.160 → 24.008 µs | 407.232 → 86.496 µs |
| 4,096 / 65,536 | 2,412.744 → 214.472 µs | 2,462.528 → 247.456 µs |
| 6,912 / 110,592 | 4,060.384 → 351.416 µs | 4,105.920 → 383.712 µs |

이 구간만의 6토큰 warm 지연 감소는 89.73%다. 위 실제 블록의 4.35%와 범위가 다르다.
8조건 모두 기존 함수와 BF16 output bits가 같았으며, 타이밍 후 graph 출력도 대조했다.

## 메모리와 호출 수

모든 시간 측정이 끝난 뒤 profiler를 별도로 실행했다. `perf.json`에 CUDA 이벤트와
aten 연산별 횟수, allocator high-water delta를 기록했다.

| 항목 | 기존 | 변경 |
| --- | ---: | ---: |
| 정규화 호출의 CUDA 커널 수 | 12 | 1 |
| 6토큰 호출의 추가 할당 최고 사용량 | 196,608 B (192 KiB) | 24,576 B (24 KiB) |
| 6,912토큰 호출의 추가 할당 최고 사용량 | 226,492,416 B (216 MiB) | 29,360,128 B (28 MiB) |

추가 할당 peak는 출력 tensor를 포함한다. 큰 출력 tensor의 논리적 크기는 27 MiB이고,
고정 runtime allocator의 요청 반올림을 포함한 측정값은 28 MiB다. 최대 prefill 조건의
peak 감소는 **188 MiB**다. 이 수치를 모델 전체의 상주 메모리 감소로 해석하면 안 된다.

## 수치와 회귀 검사

- GLM D=128: 빈 입력부터 110,592행까지 10가지 크기와 5가지 크기 배율
  (1e-20/1e-3/1/1e3/1e12)의 **50조건에서 BF16 비트 일치**.
- 65,536개 BF16 bit pattern 중 비유한값을 0으로 치환한 입력에 10가지 포화 gate 적용,
  그리고 0/-0/±1 입력에 해당 bit pattern을 gate로 적용해 BF16 비트 일치.
- D=1/8/17/33/64/127/128/129/511/512, FP32 weight, epsilon=1e-6/1e-4 비교.
  D=128에는 비트 일치를 요구하고, 다른 차원은 상대 최대 오차 0.008 미만을 요구한다.
- 그래프에서 입력·gate·weight를 바꾼 replay, 입력 불변성, 독립 CUDA stream 검사 통과.
- 잘못된 dtype·stride·shape·장치·epsilon은 launch 전에 거절한다.

bit exact라는 판정은 위 GLM 형상과 검사한 입력·고정 runtime 범위의 결과다.
모든 가능한 FP32 weight·epsilon·GPU·컴파일러에서의 수학적 보장은 아니다.

최신 main **`5c6c649f` (PR #562/#563)** 을 통합한 소스에서 전체 회귀 검사
**261개가 통과했으며 skip은 없었다** (`gpu-tests.log`). norm 검사만의 결과는
`norm-tests.log`, 최종 프로브의 정상 종료 기록은 `execution.json`에 있다.
통합 전후 norm·참조·net·lanes 파일은 측정 당시와 같았고, 기준으로 추출한 `net.py`도
최신 main의 해당 파일과 같았다. `GraphCaches`의 KDA 읽기/쓰기도 통합 과정에서 바뀌지 않았다.
전체 엔진 해시를 실제 suite에 mount된 소스와 대조했다.

## 재현과 출처

runtime은 `st-engine:9391`, Torch **2.13.0+cu130**, CUDA **13.0**, NVIDIA GB10이다.
이미지 ID는 `runtime-image.txt`, 최종 커널·독립 참조·net·lanes 해시는 `perf.json`에 있다.
전체 엔진은 측정 당시 `measured-engine-code-sha256.json`, 최종 통합 검사는
`engine-code-sha256.json`에 기록했다. 가중치/config/baseline net 해시는 `real.json`에 있다.

```bash
git show b0e50398:engine/profiles/glm53/net.py > /work/baseline-net.py
python3 probes/engine_kda_norm_tune.py --output /work/tune.json
python3 probes/engine_kda_norm_perf.py --output /work/perf.json
python3 probes/engine_kda_state_real.py --baseline-net /work/baseline-net.py \
  --checkpoint /meta --rank-file /ranks/rank0of4.safetensors --output /work/real.json
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
```

source를 `/work`에 mount하고 `-w /work -e PYTHONPATH=/work`를 사용한다. 이미지의 기본
entrypoint를 덮어써야 한다. 가중치와 checkpoint는 read-only로 mount한다. 전체 suite의
체크포인트 기반 검사는 `/home/choiceoh/models/glm53-redhat-nvfp4`에서 실제 파일을 읽는다.
config만 있는 fixture로는 loader 검사를 실행할 수 없다.

task 컨테이너는 5~6 GiB 메모리·2 CPU로 제한했다. 성능 프로브의 allocator는 1.5 GiB,
실가중치 프로브는 1 GiB, 전체 suite는 2 GiB로 제한했다. 기존 서비스를 재시작하거나
교체하지 않았다. 전체 모델 품질·DFlash acceptance·NCCL을 포함한 TP4 tokens/s·ITL은
이 실험에서 측정하지 않았다.
