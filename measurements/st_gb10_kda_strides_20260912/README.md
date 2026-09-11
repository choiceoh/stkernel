# GB10 recurrent KDA 입력 stride 직접 읽기

**Q/K/V·beta의 연속 배열 복사 4개를 없애 6토큰 recurrent 호출의 CUDA 커널 수를 5 → 1로 줄였다.**
최종 warm 측정에서 이 구간 지연은 **19.35% 감소**했다. 기준 driver와 recurrent kernel은
conv 최적화 PR #566까지 포함한 `e9702f7b`에서 추출했다. 최종 코드에는 main `f5de3168`의
MoE graph workspace 수명 수정도 통합했다.

## 변경과 적용 범위

실제 GLM TP4 conv 출력 `[T,6144]`를 Q/K/V `[1,T,16,128]`로 나누면 token stride는 6144이고,
각 tensor를 따로 연속화한 stride 2048과 다르다. Beta도 `[T,6416]` projection에서 16채널을
잘라낸 view다. 기존 `fused_recurrent_kda`는 이 네 view를 `.contiguous()`로 복사했다.

`state_layout="kv"`에서 원래 token/head/channel stride를 constexpr로 전달하고, recurrent
커널이 입력 주소를 직접 계산한다. 연산 순서·Q/K 정규화·gate·beta·FP32 상태 계산·매 토큰
snapshot 저장은 유지한다. 이미 연속인 입력은 기존 dense 주소 계산을 사용하므로 일반적인
1토큰 호출에는 stride 전용 특수화를 만들지 않는다. 기본 legacy `vk`의 복사·주소 경로도 유지한다.

출력은 항상 연속 배열이다. 특히 전치된 V 입력에 `empty_like`를 그대로 적용해 잘못된 stride의
출력을 만드는 일이 없도록 명시했다. 입력 shape/dtype/device를 검증하고, 호출자 제공 출력이
입력과 같은 저장소를 쓰면 launch 전에 거절한다. 입력과 initial state를 수정하지 않는다.

GB10 실행 구성은 기존과 같은 **BK=128, BV=16, 1 warp/CTA, 128 CTA**다.
T=1/6의 컴파일 결과는 모두 **154 registers/thread, shared memory 512 bytes/CTA, spill 0**으로
기존과 같다. 추가 메모리 이동과 launch를 제거하는 변경이며 reduction tile이나 계산 정밀도를 바꾸지 않는다.

## 실행 순서를 균형 있게 맞춘 측정

각 조건에서 **12회**, AB와 BA를 각각 6회 실행했다. 인접한 AB/BA 두 round를 하나의 cycle로
묶어 각 경로의 평균 시간을 구한 뒤, 6개 cycle의 중앙값과 쌍별 차이를 기록했다.
표의 대표 시간은 cycle 평균의 중앙값, 감소 시간/비율은 cycle별 차이/비율의 중앙값이다.
독립 중앙값의 단순 비율과 작은 차이가 날 수 있다.

초기 11회 측정은 AB/BA 횟수가 6:5로 달랐고, 블록 측정에서 먼저 실행한 쪽에 큰 비용이 붙는
패턴이 나타났다. 따라서 해당 수치를 최종 개선율로 사용하지 않았다. `exploratory/`에 원본을
남겼다. 공유 GPU의 부하 변화도 있었으며 telemetry만으로 부하 원인을 분리하지는 못했다.
최종 수치 역시 같은 호스트에서 얻은 한 번의 균형 측정이며, 독점 GPU나 여러 기기의 통계는 아니다.

### Recurrent 호출 전체: 입력 복사와 snapshot 포함

`warm`은 같은 그래프를 먼저 한 번 replay한 뒤 8회 호출을 담은 그래프의 호출당 시간이다.
`evicted`는 측정 전에 64 MiB를 써서 L2를 밀어낸 뒤 한 번 호출한 그래프 시간이다.
출력/상태 할당과 입력 복사를 포함하는 public 함수를 capture하지만 CPU allocator 시간은 replay에 포함되지 않는다.
아래는 initial state가 있는 경우이며, 없는 경우까지 포함한 16조건의 원본은 `perf.json`에 있다.

| 토큰 | warm 기존 → 변경 | paired 감소 | evicted 기존 → 변경 |
| --- | ---: | ---: | ---: |
| 1 | 4.855 → 4.852 µs | -0.06% | 26.136 → 28.080 µs |
| 2 | 15.642 → 7.247 µs | 53.68% | 46.384 → 34.504 µs |
| 3 | 18.151 → 9.429 µs | 48.04% | 53.384 → 41.840 µs |
| 4 | 25.797 → 12.402 µs | 51.78% | 61.704 → 49.440 µs |
| 5 | 32.966 → 27.229 µs | 17.25% | 67.584 → 57.328 µs |
| 6 | 40.650 → 32.777 µs | 19.35% | 73.760 → 62.656 µs |
| 7 | 51.528 → 43.058 µs | 16.39% | 83.720 → 72.064 µs |
| 12 | 82.225 → 71.627 µs | 12.86% | 112.216 → 99.288 µs |

1토큰에는 기존에도 입력 복사가 없으므로 구조적인 개선을 기대하지 않는다.
7/12토큰은 보수적인 BV=8 경로의 호환성도 확인하기 위한 저수준 호출이다.

Profiler는 모든 타이밍 뒤에 별도로 실행했다. 초기 상태가 있는 6토큰 호출은 CUDA 커널
**5 → 1**, 추가 할당 peak는 **6,390,272 → 6,316,032 bytes**, 즉 **72.5 KiB 감소**다.
Q/K/V 72 KiB와 beta 192 bytes의 임시 저장소를 제거하며, allocator의 512-byte 반올림을 포함한
peak가 72.5 KiB다. 논리적인 복사 read+write는 호출당 147,840 bytes 줄어든다.
출력과 6 MiB의 FP32 token별 상태 snapshot은 계속 생성한다. 모델 전체 상주 메모리 절감량은 아니다.

### 실제 L0 KDA 블록

실제 rank 0 L0 KDA 가중치 **70,492,480 bytes**와 seeded synthetic activations를 사용했다.
현재 net/lane/cache에서 frozen recurrent callable만 교체한다. projection·conv·recurrent·
출력 norm·출력 projection·상태 링 읽기/쓰기를 포함하고 collective는 항등 처리한다.
각 timed sample은 graph를 8번 replay한 호출당 시간이며, context=4096, physical slot=1이다.

| 토큰 | 기존 | 변경 | paired 감소 | paired 감소율 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 740.883 µs | 740.799 µs | 0.789 µs | 0.10% |
| 6 | 733.637 µs | 723.136 µs | 14.160 µs | 1.92% |

이 블록 수치는 공유 GPU의 부하 변동이 남아 있는 **참고 관측값**이다. 6토큰 cycle의 차이는
0.932~198.758 µs로 넓었고, 중앙 차이는 14.160 µs였다. 확정적인 서빙 개선율로 취급하지 않는다.
위 recurrent 구간의 감소율과 범위가 다르며 전체 모델 tokens/s의 보장으로 해석하지 않는다. `real.json`에 각 샘플, AB/BA cycle 평균과 차이, 가중치/config/baseline 해시가 있다.

## 수치와 통합 검사

전체 엔진 검사 **276개 통과, skip 없음** (`gpu-tests.log`). 최종 소스에서 확인한 항목:

- 실제 conv와 beta stride, T=1~7/12, 입력 크기 배율 0.001/1/10, initial state 유무의
  48조건에서 연속화한 입력과 출력·모든 FP32 snapshot 바이트 일치.
- FP16/BF16/FP32, token/head/channel stride, broadcast view, grouped heads와 vector beta 검사.
- 변경된 입력과 initial state의 CUDA graph replay, accepted prefix를 통한 rollback 검사.
- 별도 출력 버퍼, 입력 불변성, 잘못된 shape/dtype/device와 입력/출력 저장소 중복 거절.
- 공유 kernel의 legacy scalar-decay 경로도 독립 Torch recurrence와 비교.

실제 가중치 검사에서는 prefill 64→verify 6→draft 4개 거절 후 재검증→decode,
T=1/6 및 context=0/1/7/8/4095/4096, physical slot=1/2의 **16조건**을 확인했다.
기존/현재 출력과 전체 상태 링, 현재 eager/graph 결과가 부호 있는 0까지 바이트 단위로 일치했다.
바이트 일치는 검사한 입력·형상과 고정 runtime에서의 결과이며 다른 GPU/컴파일러에 대한 보장은 아니다.

main의 `graph_resources` 추가를 통합한 첫 suite에서 기존 executor 검사 2개 subtest가 실패했다.
이 함수는 kernel이 아니라 capture 이후 수집하는 host metadata이므로, direct resource 조회와
executor에 묶인 kernel 호출을 각각 검사하도록 수정했다. 운영 kernel 동작을 바꾸지 않았다.
`integration-initial.log`에 최초 결과를, `gpu-tests.log`에 수정 후 전체 결과를 남겼다.

## 재현과 출처

```bash
mkdir -p /work/baseline
git show e9702f7b:engine/kernels/kda/kda.py > /work/baseline/kda.py
git show e9702f7b:engine/kernels/kda/fused_recurrent.py > /work/baseline/fused_recurrent.py
python3 probes/engine_kda_strides_perf.py --baseline-dir /work/baseline --output /work/perf.json
python3 probes/engine_kda_state_real.py --baseline-strides /work/baseline \
  --checkpoint /meta --rank-file /ranks/rank0of4.safetensors --timing-replays 8 --output /work/real.json
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
```

srv4 GB10, `st-engine:9391`, Torch 2.13.0+cu130, CUDA 13.0을 사용했다. source를 `/work`에
mount하고 `-w /work -e PYTHONPATH=/work`를 지정하며 이미지 entrypoint를 덮어쓴다.
`/meta`는 실제 checkpoint config, `/ranks`는 실제 preshard 파일이다. 전체 suite에는
실제 checkpoint를 `/home/choiceoh/models/glm53-redhat-nvfp4`에 read-only로 mount했다.
컨테이너는 5~6 GiB / 2 CPU, allocator는 성능 1.5 GiB·실가중치 1 GiB·suite 2 GiB로 제한했다.

원본/현재 kernel 해시는 `engine/kernels/SOURCES.json`, 전체 137개 엔진 소스는
`engine-code-sha256.json`, runtime 및 종료 정보는 `runtime-image.txt`·`execution.json`,
출처는 `provenance.json`에 있다. 실제 suite/probe에 mount된 엔진 소스를 로컬과 대조했다.
PTX 진단 해시는 `.file`/`.loc`/주석행을 제거한 텍스트의 값으로, 명령어 동일성 판정으로 사용하지 않는다.

기존 서비스를 재시작하거나 교체하지 않았다. 전체 모델 품질, DFlash acceptance,
NCCL을 포함한 TP4 tokens/s·ITL은 이번 실험에서 측정하지 않았다.
