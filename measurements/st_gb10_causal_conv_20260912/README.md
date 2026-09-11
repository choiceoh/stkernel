# GB10 단일 시퀀스 causal conv

**KDA의 conv 배치 메타데이터·임시 상태 테이블을 제거해 CUDA 커널 9개를 1개로 합쳤다.**
기준은 정규화 epsilon 수정 PR #565까지 포함한 `b7a6e44c`다. 실제 rank 0 L0 KDA 블록의
두 차례 측정에서 1토큰 지연은 **1.11~2.50%**, 6토큰은 **1.99~3.21%** 감소했다.
최종 검증에서는 기존/현재 출력과 전체 상태 링, 현재 eager/graph 출력이 바이트 단위로 일치했다.

## 변경과 하드웨어 구성

`engine/profiles/glm53/lanes.py`가 `engine/kernels/causal_conv_single.py`를 직접 호출한다.
기존 배치 어댑터는 두 행의 상태 테이블을 0으로 채우고, 이전 상태를 복사하고, 시퀀스 길이·
배치 ID·청크 오프셋·캐시 인덱스·초기 상태 여부 tensor를 만든 뒤 범용 커널을 호출했다.
단일 시퀀스에서 이미 알려진 이 정보는 이제 launch 인자와 CTA 좌표로 처리한다.

CTA 하나는 **128채널 × 최대 8토큰, 4 warps**를 담당한다. GLM TP4의 6,144채널에서
1토큰도 48 CTA를 발행할 수 있다. 기존 256채널 타일은 24 CTA였다.
[조사한 GB10의 48 SM](../sm121a_architecture_20260911/README.md)에 충분한 작업을 주면서,
weight 4개와 rolling history를 레지스터에서 재사용한다. 마지막 time tile만 별도 최종 상태를 쓴다.
실측 컴파일 결과는 T=1/6/6,912에서 **23/24/30 registers/thread, shared memory 0, spill 0**이다.

입력은 실제 projection `[T,6416]`에서 잘라낸 `[T,6144]` view를 그대로 읽는다.
contiguous 복사나 전역 FP32 중간 결과가 없다. 기존 multiply/add 순서, SiLU 식,
initial state를 입력 dtype으로 반올림하는 동작을 유지한다. 입력·weight·이전 상태는 수정하지 않는다.
출력 `[T,C]`와 최종 상태 `[C,K-1]`는 독립 저장소다. width 2/3/4와 FP16/BF16/FP32를 지원한다.

동작 참조인 `engine/modules/causal_conv.py`와 기존 범용 `engine/kernels/causal_conv.py`는 보존했다.
새 파일은 기존 conv forward recurrence의 특수화이며, 라이선스 표기와 원본/현재 해시는
`engine/kernels/SOURCES.json`에 기록했다. 환경 변수나 실행 중 autotuning은 추가하지 않았다.

## 실제 L0 KDA 블록

실제 L0 KDA 가중치 **70,492,480 bytes**, rank 0 / TP4 형상, seeded synthetic activations를 쓴다.
두 경로는 같은 현재 `_kda`, 가중치, 다른 lane을 사용한다. 기준 `lanes.py`를 git에서 추출해
conv callable만 교체하므로 비교군에 새 conv가 섞이지 않는다. projection·conv·KDA recurrence·
출력 norm·출력 projection·상태 링 읽기/쓰기를 포함한다.

| 측정 | 토큰 | 기존 | 변경 | 지연 감소 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 451.424 µs | 440.128 µs | 2.50% |
| 1 | 6 | 445.120 µs | 430.848 µs | 3.21% |
| 2 | 1 | 459.776 µs | 454.656 µs | 1.11% |
| 2 | 6 | 440.640 µs | 431.872 µs | 1.99% |

각 측정은 CUDA graph AB/BA 11회 중앙값이며, 각 샘플 전에 같은 그래프를 한 번 replay했다.
context=4096, physical slot=1이다. 두 측정 사이 블록/커널 소스는 같고, 최종 검증은 output의
부호 있는 0까지 판정하도록 byte view를 사용한다. 1차 원본도 `tuning/real-initial.json`에 남겼다.
변동을 숨기지 않기 위해 한 수치 대신 두 결과를 함께 제시한다. 다른 기기나 시간대에 대한 분포는 아니다.

최종 `real.json`의 16조건 모두 기존/현재 출력·전체 상태 링 바이트가 동일했다.
prefill 64→verify 6→draft 4개 거절 후 재검증→decode를 확인했다. 그래프에서는 T=1/6,
context=0/1/7/8/4095/4096, physical slot=1/2를 바꿨고 현재 eager와 graph의 바이트도 대조했다.
실제 가중치별 바이트 수·SHA256, config·기준 lane 해시는 `real.json`에 있다.

## conv 호출만 분리한 측정

할당을 포함하는 public 함수 호출을 capture했다. `warm`은 16회 호출 그래프의 호출당 시간,
`evicted`는 측정 전에 64 MiB를 써서 L2를 밀어낸 뒤 한 번 호출한 그래프 시간이다.
CUDA graph의 할당은 capture 때 이루어지므로 표의 replay 시간에는 CPU allocator 시간이 들어가지 않는다.
각 조건에서 AB/BA 11회 중앙값을 기록했다. 아래는 초기 상태가 있는 경우다.
초기 상태가 없는 경우까지 포함한 16조건의 모든 원본 샘플은 `perf.json`에 있다.

| 토큰 | warm 기존 → 변경 | evicted 기존 → 변경 |
| --- | ---: | ---: |
| 1 | 11.470 → 1.672 µs | 18.080 → 6.144 µs |
| 6 | 12.284 → 3.180 µs | 20.128 → 10.080 µs |
| 24 | 14.064 → 3.950 µs | 22.688 → 14.336 µs |
| 64 | 15.218 → 4.850 µs | 32.448 → 24.256 µs |
| 256 | 22.894 → 12.176 µs | 64.384 → 64.160 µs |
| 1,024 | 131.544 → 119.346 µs | 172.768 → 157.344 µs |
| 4,096 | 534.766 → 495.224 µs | 567.296 → 525.984 µs |
| 6,912 | 899.800 → 817.784 µs | 931.936 → 861.888 µs |

1토큰/6토큰 warm conv 구간은 **85.42% / 74.11%** 감소했다. 이 비율은 위 블록 전체와 범위가 다르다.
256토큰 evicted 조건처럼 개선이 미미한 경우도 있다. 모든 조건에서 출력·최종 상태 바이트를
capture 전과 replay 후에 확인했다.

시간 측정이 끝난 뒤 profiler와 CUDA allocator high-water delta를 별도로 측정했다.
초기 상태가 있는 conv의 CUDA 커널 수는 **9 → 1**이다. 6토큰 호출의 추가 할당 peak는
**150,016 → 110,592 bytes (146.5 → 108 KiB)**로, **38.5 KiB** 줄었다.
출력·최종 상태 저장소를 포함한 호출당 수치이며 모델 전체 상주 메모리 감소량은 아니다.

## 검증과 재현

전체 `test_engine_*.py` **268개 통과, skip 없음** (`gpu-tests.log`). 신규 conv 검사에서는:

- 실제 projection stride, T=1/2/3/6/7/8/9/24/64/256/1024/4096/6912, 초기 상태 유무를 대조했다.
- width 2/3/4, 채널 tail, strided 입력·weight·state, FP16/BF16/FP32와 FP32 상태 반올림을 확인했다.
- 65,536개 BF16 bit pattern 중 비유한값을 0으로 치환한 입력, 4개 weight scale,
  0/-0 및 ±20/±88/±100 입력을 기존 GPU 커널과 바이트 단위로 비교했다.
- 여러 길이로 나눈 chunk carry, 독립 Torch convolution, 입력 불변성을 검사했다.
- 변경된 입력·weight·state의 graph replay, 독립 stream, 빈 입력과 잘못된 인자 검사를 통과했다.

바이트 동일 판정은 검사한 형상·입력과 고정 runtime의 결과다. 다른 컴파일러/GPU에 대한 보장은 아니다.
초기 단독 검사 로그는 `conv-initial-tests.log`; 최종 suite는 BF16 pattern 검사를 포함한다.

`tuning/`은 8개 rolling 구성과 8개 token-parallel 구성의 탐색 소스·원본, 최초 public A/B를 보존한다.
탐색 타이밍은 후보별 순차 측정이고 최종 AB/BA 성능 근거로 사용하지 않는다. 최종 구성은 단순한
rolling recurrence를 유지하며, 실제 서빙 파일에는 선택하지 않은 병렬 커널을 넣지 않았다.

```bash
git show b7a6e44c:engine/profiles/glm53/lanes.py > /work/baseline-lanes.py
python3 probes/engine_causal_conv_perf.py --baseline-lanes /work/baseline-lanes.py --output /work/perf.json
python3 probes/engine_kda_state_real.py --baseline-lanes /work/baseline-lanes.py \
  --checkpoint /meta --rank-file /ranks/rank0of4.safetensors --output /work/real.json
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
```

srv4의 `st-engine:9391`, Torch 2.13.0+cu130, CUDA 13.0, NVIDIA GB10에서 실행했다.
이미지 기본 entrypoint를 덮어쓰고 source를 `/work`, 실제 rank file을 `/ranks`에 mount한다.
`-w /work -e PYTHONPATH=/work`를 지정한다. `/meta`는 실제 checkpoint의 config다.
전체 suite의 checkpoint 검사는 실제 파일을 `/home/choiceoh/models/glm53-redhat-nvfp4`에서 읽는다.

task 컨테이너는 5~6 GiB / 2 CPU, allocator는 성능 1.5 GiB·실가중치 1 GiB·전체 suite 2 GiB로 제한했다.
이미지 ID는 `runtime-image.txt`, 종료·자원 제한은 `execution.json`, 출처는 `provenance.json`이다.
`engine-code-sha256.json`의 137개 엔진 소스를 실제 mount된 파일과 대조했다.
기존 서비스의 재시작·교체는 수행하지 않았다.

TP4 rank 0의 collective를 항등 처리한 블록 검사다. 전체 모델 품질, DFlash acceptance,
NCCL을 포함한 TP4 tokens/s·ITL은 측정하지 않았다.
