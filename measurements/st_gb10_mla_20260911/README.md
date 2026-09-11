# GB10 하드웨어 특성을 적용한 ST MLA 최적화

## 결과와 범위

2026-09-11, srv4의 GB10에서 기존 ST MLA와 변경된 커널을 같은 입력과 split 계획으로
비교했다. 32~64행에서 split 2·3을 cluster로 묶고 DSMEM에서 부분값을 합치는 경로가
유리했다. 이 조건을 기본 디스패치에 반영하고, 나머지는 기존 실행 경로를 유지했다.
MLA의 softmax 최대값에는 정수 warp reduction을 적용했다.

이 결과는 MLA 커널과 엔진의 실제 Python/CUDA 드라이버에 대한 것이다.
TP=4 전체 모델의 tokens/s, ITL, TTFT 또는 NVFP4 MoE의 속도 향상률을 뜻하지 않는다.
기존 서빙을 계속 실행한 상태에서 별도 메모리 제한 컨테이너로 측정했고 GPU 클록을
고정하지 않았다. 전체 모델에서의 효과는 MLA의 실행 비중과 실제 T/W 분포에 달려 있다.

## 하드웨어 근거와 구현

[앞선 아키텍처 조사](../sm121a_architecture_20260911/README.md)에서 GB10의 cluster와
DSMEM이 실제로 동작하고, SM당 shared memory는 100 KiB임을 확인했다.
이를 바탕으로 다음 두 곳을 바꿨다.

1. **DSMEM split 병합.** 하나의 query 행에 해당하는 split을 같은 cluster에 배치한다.
   QK·online softmax·PV 계산을 끝내고 async copy와 block 내 접근을 기다린 뒤,
   기존 shared memory를 FP32 부분값 저장에 재사용한다. cluster barrier 후 다른 block의
   shared memory를 읽고 기존 split 순서대로 합산한다. 마지막 cluster barrier는 모든
   peer 읽기가 끝나기 전에 shared memory가 해제되는 것을 막는다.
2. **warp max reduction.** float의 비트를 정수 정렬 키로 변환한 뒤 `__reduce_max_sync`를
   사용한다. 기존 5단계 shuffle/fmax를 대체한다. 무한대와 signed zero를 처리하며,
   NaN은 `fmaxf`처럼 숫자보다 우선하지 않는다. warp 합계의 연산 순서는 바꾸지 않았다.

NVFP4의 warp MMA 경로는 기존대로다. 지원되지 않는 WGMMA·tcgen05/TMEM을 도입하지
않았다. FP8 변환도 기존 구현을 유지했다. 이 변경은 quantization, attention 슬롯 선택,
split 수, FP32 합산 순서 또는 BF16 출력 형식을 바꾸지 않는다.

컴파일된 MLA의 block당 shared memory는 전후 모두 **46,976 B**다. 일반 경로의 register는
thread당 113개, cluster variant는 64개다. 일반 경로의 resident grid는 96 blocks였다.
32~64행 중 split 2·3인 경우만 선택하며, W는 1~2176으로 제한한다. 4~8-block cluster는
직접 비교했으나 대체로 느리거나 이득이 없어 선택하지 않았다. 지원 여부만으로 경로를
활성화하지 않고, 커널의 cluster 수용량과 측정한 형상 조건을 함께 적용한다.

예를 들어 T=48, split=2는 호출당 FP32 partial 3 MiB의 전역 쓰기와 읽기, 합계 약 6 MiB를
DSMEM 접근으로 바꾼다. 이 경로는 전역 partial·ticket counter를 요구하지 않으며, 출력
버퍼를 제공하면 추가 할당 없이 실행한다. 다른 형상이 쓰는 엔진 workspace는 여전히
존재하므로 엔진 전체의 예약 메모리가 그만큼 줄었다는 뜻은 아니다.

## 시간 비교

각 variant를 단일 커널 CUDA Graph로 캡처해 CUDA event로 측정했다. variant의 실행
순서를 번갈아 가며 7회 측정한 중앙값이다. `warm`은 직전 같은 graph 재생 후이고,
`evicted`는 GB10의 24 MiB L2보다 큰 64 MiB 버퍼를 갱신한 후다. eviction 작업 자체는
측정 구간에 포함하지 않았다. cache는 64 MiB FP8, 슬롯은 무작위이며 유효 길이는 W다.
아래 시간의 단위는 µs이고, 감소율은 `(기존 - 변경) / 기존`이다.

| T | W | split | 기존 warm | 변경 warm | 감소 | 기존 evicted | 변경 evicted | 감소 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2048 | 3 | 188.5 | 157.5 | 16.5% | 239.3 | 218.9 | 8.5% |
| 36 | 2048 | 3 | 274.0 | 243.5 | 11.1% | 324.3 | 304.8 | 6.0% |
| 48 | 512 | 2 | 96.1 | 65.4 | 31.9% | 140.5 | 107.3 | 23.6% |
| 48 | 2048 | 2 | 270.1 | 216.9 | 19.7% | 311.0 | 280.2 | 9.9% |
| 64 | 2048 | 3 | 365.6 | 306.7 | 16.1% | 477.7* | 366.5* | 23.3%* |

W=2048의 선택 형상들에서 warm은 약 10~20%, evicted는 변동이 큰 T=64를 제외하면
약 6~12% 감소했다. *T=64/W=2048 evicted는 기준 실행이 409~631 µs로 흔들렸다.
이 조건의 23.3%는 관측 중앙값일 뿐 대표 개선율로 사용하지 않는다.
작은 W에서는 전역 barrier와 부분값 왕복 제거의 비중이 더 커졌다.
warp reduction만 쓰는 나머지 형상은 대체로 차이가 작으므로 별도의 큰 향상률을 주장하지
않는다. 전체 33개 형상, 선택하지 않은 큰 cluster, 모든 개별 측정값은
[kernel-results.json](kernel-results.json)에 있다.

## 검증

- 원본과 변경된 translation unit에서 MLA 본문을 그대로 추출해 각각 `sm_121a`로 빌드했다.
  33개 T/W 형상마다 full·ragged·empty·duplicate 슬롯 입력을 비교했고, BF16 저장 비트를
  정수 view로 비교했다. 모든 비교가 일치했다. FP32 독립 참조는 각 형상의 첫·마지막 유효
  행을 표본 검사했으며 상대 오차 2% 이하였다.
- 각 형상에서 입력과 유효 길이를 바꾸면서 CUDA Graph를 3회 재생해 저장 비트 일치를
  확인했다. cluster의 전역 partial·counter 포인터는 null로 전달했다.
- FP8 byte pair 65,536개를 contiguous·strided 변환으로 비교해 비트 일치를 확인했다.
  warp max는 무작위 FP32 비트와 NaN·무한대·signed zero를 포함한 4,096개 warp를 비교했다.
  warp max 검사는 숫자와 NaN mask의 동등성을 검사하며 NaN payload 보존을 요구하지 않는다.
- 실제 엔진의 전체 CUDA JIT, pybind, `maybe_arm()` 부팅 판정을 통과했다.
  실제 `mla_decode`로 14개 형상의 eager와 70회 graph 재생을 비교했다. 비어 있는 행의
  슬롯을 -1로 채운 경우와 중복 슬롯을 포함하며 모두 BF16 저장 비트가 일치했다.
- 실제 드라이버의 cluster 경로에서 legacy workspace를 제거하고 기존 출력 버퍼를
  전달했을 때 추가 CUDA 할당 없이 동작했다. 두 CUDA stream에서 각각 8회 연속 실행한
  결과도 일치했다. 이는 cluster 경로의 검사이며 legacy 경로의 동시 실행 보장은 아니다.
- GPU 없는 개발 환경의 engine unit suite는 172개 중 111개 통과, Torch가 필요한 61개
  건너뜀이다. 별도 dispatch 검사는 형상 경계, cluster 수용량, 비활성화 설정,
  workspace 미사용과 기존 경로 선택을 확인한다.
- 실행 이미지의 Torch가 설치된 환경에서도 dispatch·커널 패키지 검사 11개를 모두 통과했다.

Compute Sanitizer는 검사 이미지에 설치되어 있지 않아 memcheck/racecheck는 실행하지
못했다. 수치 비교·graph 재생·두 stream 검사를 통해 관찰한 동작을 기록한 것이며,
모든 입력에 대한 race 부재 또는 전체 모델 품질 검증을 주장하지 않는다.

## 재현과 출처

- 기준 Git commit: `879b8942ce09990efbd49e59bc779d573509d98c` (조사문서 PR #551 병합).
- 기준 CUDA SHA256: `61f76b048b1991bfa449c5ef86056f414b7a750840e87383d6367e1ae2ae70a1`.
- 변경 CUDA SHA256: `52de53ada86de8c4f6534a34cf2917c49c01d40b37849f0918a1dad7d6da82b7`.
- GPU: NVIDIA GB10, CC 12.1, 48 SM. Driver 580.159.03, nvcc 13.0.88.
- 실행 이미지: `st-engine:9391`, ID
  `sha256:fa1183e3dd660f09f42ee72918948d53466a96789babc58d6a4bd3a21147a9a7`.
- 컨테이너 RAM 제한: 커널 비교 4 GiB, 전체 JIT 검사 6 GiB, CPU 2개, Torch CUDA allocator
  512 MiB. 빌드는 `MAX_JOBS=1`. 사용자 서빙 컨테이너는 변경하지 않았다.
- [baseline-compile.log](baseline-compile.log), [candidate-compile.log](candidate-compile.log),
  [integration.log](integration.log), [CPU 검사](cpu-tests.log),
  [파일 해시와 환경](provenance.json).

저장소 루트에서 기준 파일을 추출한 후, CUDA 13과 PyTorch가 있는 GB10 환경에서 실행한다.
`--output`에 생성되는 CUDA C ABI wrapper는 빌드 시간 단축용이다. 전체 production 빌드는
두 번째 probe가 검증한다. 이 단일 커널 probe에서 PDL launch attribute는 꺼져 있다.

```bash
git show 879b8942ce09990efbd49e59bc779d573509d98c:engine/kernels/mla/glm53_megakernel.cu > /tmp/st-mla-baseline.cu
PYTHONPATH=. python3 probes/engine_mla_hardware_check.py \
  --baseline /tmp/st-mla-baseline.cu \
  --candidate engine/kernels/mla/glm53_megakernel.cu \
  --output /tmp/st-mla-hardware-results
MAX_JOBS=1 PYTHONPATH=. python3 probes/engine_mla_integration_check.py
python3 -m unittest discover -s tests -p 'test_engine_*.py'
```

일반 엔진 부팅에서는 cluster가 기본으로 활성화된다. 비교할 때는 시작 전에
`ST_GLM53_MK_MLA_CLUSTER=0`을 설정하면 기존 split reduction을 선택한다.
