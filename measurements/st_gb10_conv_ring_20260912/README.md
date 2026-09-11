# GB10 짧은 causal conv와 상태 링 갱신 결합

**KDA decode/verify의 conv 이력 읽기, convolution, 원본 입력 링 저장을 한 커널로 합쳤다.**
초기/최종 이력 버퍼를 없애 호출당 추가 할당 **72 KiB**를 줄이고, CUDA 커널 수를 **3 → 1**로
줄였다. GB10에서 해당 구간의 warm 지연은 1토큰 **4.4535 → 1.9015 µs**, 6토큰
**7.2730 → 3.7040 µs**였고, AB/BA paired 감소율은 각각 **57.11%, 48.82%**였다.

기준 main은 KDA recurrent ring PR #571을 포함한 **`6b36873f`**, 구현은 **`c54d341f`**다.
이 보고서의 큰 감소율은 conv와 그 상태 이동 구간에 한정된다. 실제 L0 블록 전체는 1토큰
약 1.95% 감소를 관찰했으나 변동이 컸고, 6토큰은 약 0.08%로 개선이 명확하지 않았다.
4기기 collective를 포함한 serving throughput이나 ITL 개선율로 사용하지 않는다.

## 구현

GLM TP4의 conv 입력은 projection `[T,6416]`에서 잘라낸 `[T,6144]` view다.
기존 경로는 `_read_conv`로 `[6144,3]` BF16 이력을 모으고, `causal_conv1d_single`이
출력과 새로운 `[6144,3]` 이력을 만들고, `_write_conv`가 원본 입력을 별도의 링에 저장했다.
모델은 conv가 반환한 새 이력을 사용하지 않았다.

새 `causal_conv1d_ring`은 `causal_conv_single.py::_single_conv`의 산술을 재사용한다.
초기 이력은 `[slots,C,R]` 링에서 직접 읽고, 각 토큰의 원본 입력을 `(context+token)%R`에
기록한다. 별도 초기 이력 및 최종 이력 tensor를 생성하지 않고 dense convolution 출력만 반환한다.
곱셈/덧셈 순서, history의 입력 dtype, `acc / (1 + exp(-acc))` SiLU 식은 유지했다.

런처는 한 time tile만 허용하여 **1 ≤ T ≤ min(8,R)**로 제한한다. CTA마다 128채널의 전체
초기 이력과 현재 토큰을 소유하므로 링이 돌아가도 다른 CTA가 해당 채널의 이력을 덮지 않는다.
모든 초기 이력은 첫 ring write 전에 읽는다. 8토큰보다 큰 입력은 이 런처에서 거절한다.
기존 functional conv는 여러 time tile과 긴 prefill을 계속 처리한다.

GLM net은 recurrent ring lane과 conv ring lane이 활성화된 짧은 단계에 연결한다.
현재 GLM은 **T≤6, conv width K=4, conv ring R=8**이다. R=8은 draft rollback에 필요한
원본 입력 이력을 기존과 동일하게 보존한다. `GraphCaches.kda_rings`는 두 링과 device slot의
view만 반환한다. Context/slot은 eager에서는 Python 정수, graph에서는 CUDA singleton이며
replay마다 바뀌어도 host read가 없다.

`reference_for=("conv_prefill",)`은 conv ring lane을 비활성화하고 선언한 reference conv를
사용한다. Reference table 및 기존 functional API의 입력 불변성과 반환 이력 계약은 유지한다.
기존 conv adapter 비교 probe도 optional ring lane을 끄도록 하여 비교 대상이 우회되지 않게 했다.

입력 shape/dtype/device, 링 stride, host index 범위, 입력·weight·device index와 링 주소 범위의
겹침을 검사한다. Device slot/context 값의 범위와 서로 다른 호출의 슬롯 소유권은 기존 호출자
계약이다. 이를 확인하려고 GPU 값을 host로 복사하지 않는다. 링 bounding range 사이의 padding도
보수적으로 겹침 검사에 포함한다.

## GB10 실행 구성

기존과 같은 **48 CTA, 128 channels/CTA, 4 warps/CTA**, 최대 8토큰 time tile이다.
각 CTA가 맡은 채널을 계속 처리하여 이미 읽은 history를 재사용하고 별도 gather/write launch를 없앤다.

| 토큰 | register/thread 기존 → 변경 | shared memory | spill |
| --- | ---: | ---: | ---: |
| 1 | 23 → 25 | 0 | 0 |
| 6 | 24 → 28 | 0 | 0 |

상태는 계속 BF16으로 보존하고 convolution 산술은 기존 커널과 동일하다. 새로운 tensor-core
명령을 사용하는 변경이 아니라 짧은 decode 단계의 kernel launch와 임시 메모리 이동을 줄인다.

## 구간 성능

`engine_conv_ring_perf.py`는 `6b36873f`에서 추출한 frozen conv와 기존
`conv_history`/`write_conv` 조합을 새 ring 커널과 비교한다. 실제 projection stride와 padded
slot stride를 사용한다. 12 round에서 AB/BA를 각각 6회 실행했다. 인접 AB/BA의 경로별 평균을
하나의 cycle로 묶고 cycle 중앙값과 paired 차이/비율 중앙값을 계산한다. 독립 중앙값을 단순히
빼거나 나눈 값과 paired 수치는 조금 다를 수 있다.

warm은 16회 호출을 담은 graph를 한 번 replay한 후 호출당 시간을 측정한다. evicted는
64 MiB buffer를 써서 cache를 밀어낸 후 1회 호출 graph를 측정한다. Python 검사 및 allocator의
host 시간은 graph replay에 포함되지 않는다. 타이밍이 끝난 후 별도로 profiler를 실행한다.

아래는 context=4096이다. Context=0/1 및 T=1/2/6/8의 12조건 원본은 `perf.json`에 있다.
8토큰은 저수준 API 경계 검사이며 GLM의 현재 6토큰 verify 경로와 구분한다.

| 토큰 | warm 기존 → 변경 | paired 감소 | evicted 기존 → 변경 | paired 감소 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.4535 → 1.9015 µs | 57.11% | 12.024 → 7.216 µs | 39.58% |
| 2 | 4.9765 → 2.2825 µs | 54.13% | 12.016 → 7.936 µs | 34.62% |
| 6 | 7.2730 → 3.7040 µs | 48.82% | 16.016 → 11.936 µs | 25.42% |
| 8 | 8.6805 → 4.7210 µs | 45.65% | 18.176 → 12.160 µs | 33.13% |

| 토큰 | 기존 추가 할당 peak | 변경 | 감소 | CUDA 커널 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 86,016 B (84 KiB) | 12,288 B (12 KiB) | 72 KiB | 3 → 1 |
| 6 | 147,456 B (144 KiB) | 73,728 B (72 KiB) | 72 KiB | 3 → 1 |

제거한 것은 36 KiB짜리 이력 버퍼 두 개다. 출력과 영구 ring 자체는 유지하므로 모델 전체
상주 메모리 절감량으로 확대 해석하지 않는다. old/new는 별도 링을 사용하며, correctness
검사 전 동일한 원본을 복원한다. 1회 및 16회 연속 state-mutating graph의 출력과 전체 storage도
바이트 단위로 비교했다.

## 실제 L0 KDA 블록

실제 rank 0 L0 KDA 가중치 **70,492,480 bytes**, seeded synthetic activations,
identity all-reduce를 사용했다. Frozen `net.py::_kda`와 현재 메서드를 같은 lane/cache에
연결한다. Projection, conv, recurrent, output norm, output projection과 상태 접근 전체를 포함한다.
12 round AB/BA, 샘플마다 8 replay의 호출당 시간, context=4096, physical slot=1이다.

| 토큰 | 기존 cycle 중앙값 | 변경 cycle 중앙값 | paired 감소 시간 | paired 감소율 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 453.517 µs | 435.978 µs | 8.678 µs | 1.95% |
| 6 | 432.075 µs | 431.679 µs | 0.328 µs | 0.08% |

공유 GPU에서 측정했으며 1토큰의 cycle 차이는 **-15.930~49.916 µs**, 6토큰은
**-2.172~3.438 µs**였다. 1토큰 개선율은 변동을 포함한 한 번의 참고 관찰이고,
6토큰 전체 블록의 개선은 확인하지 못했다. 구간 microbenchmark에서 줄어든 시간이
다른 연산과 cache를 공유하는 블록 전체에 그대로 더해진다고 가정하지 않는다.
기존 서비스는 실행한 상태에서 측정했다.

## 검증과 재현

- GB10 전체 `test_engine_*.py`: **290 tests, 94.329 s, 실패 0 / skip 0**.
- 신규 conv ring tests 7개: T1~8, context0/1/2/3/7/8/32768, slot0/1/2,
  padded storage, input/weight stride, C67 channel tail, K2/3/4, BF16/FP16/FP32,
  독립 Torch convolution, NaN이 남은 링의 zero context, BF16 finite bit patterns 및
  saturated/signed-zero 입력, graph 슬롯 변경·accepted prefix 재시도, 독립 CUDA stream,
  잘못된 shape/범위/입력 겹침 거절, reference binding.
- 실제 L0 **22조건**의 기존/변경 출력 및 전체 cache arena가 bytes exact.
  Prefill64, verify6, reject4 후 retry6, decode1, T1/6 graph context0/1/7/8/4095/4096,
  physical slot1/2와 두 시퀀스 배치를 포함한다. Current eager/graph도 bytes exact.
- Frozen conv 대비 구간 12조건과 각 조건의 1회/16회 graph도 bytes exact.
- 전체 engine 소스 **139파일**의 로컬/원격 SHA256 일치 확인.
- `compileall`, `git diff --check` 통과. Shared conv의 provenance hash 갱신.

srv4 NVIDIA GB10, driver `580.159.03`, compute capability `12.1`,
PyTorch `2.13.0+cu130`, CUDA runtime `13.0`. 기존 image `st-engine:9391` ID:
`sha256:fa1183e3dd660f09f42ee72918948d53466a96789babc58d6a4bd3a21147a9a7`.

작업 디렉터리는 `/tmp/st-gb10-conv-ring-9391`, shared Triton cache는
`/tmp/st-gb10-kda-9391/cache/triton`이다. 별도 task container에 CPU 2개, host memory
5 GiB(전체 suite 6 GiB), timeout 600초를 적용했다. CUDA allocator cap은 perf 1.5 GiB,
real 1 GiB, 전체 suite 2 GiB다. Baseline 두 파일은 `6b36873f`에서 추출했다.

동일 image에 작업 디렉터리를 `/work`로 마운트하고 `PYTHONPATH=/work`로 실행한다.

```bash
python3 probes/engine_conv_ring_perf.py --baseline-conv /work/baseline-conv.py --output /work/perf.json
python3 probes/engine_kda_state_real.py --baseline-net /work/baseline-net.py \
  --checkpoint /meta --rank-file /ranks/rank0of4.safetensors \
  --timing-replays 8 --output /work/real.json
python3 /work/st-kda-gpu-suite-9391.py
```

`/meta`는 원본 GLM config fixture, `/ranks`는
`/home/choiceoh/models/st-glm53-9391-up-gate-full`의 read-only mount다. 실제 probe는 필요한
L0 KDA key 범위만 읽고 hash했다. 전체 suite에는 원본
`/home/choiceoh/models/glm53-redhat-nvfp4`를 같은 경로에 read-only mount했다.
Suite runner는 2 GiB CUDA allocator cap을 설정한 뒤 `unittest.defaultTestLoader.discover('tests',
pattern='test_engine_*.py')`를 실행하고 실패하면 nonzero로 종료한다.

실행 인자·종료 상태·자원 제한은 `containers.txt`, image/device는 `runtime.txt`,
source/probe/test hash는 `provenance.json` 및 `engine-code-sha256.json`, 원시 샘플과 출력은
`perf.json`, `real.json` 및 각 `.log` 파일에 보존했다.
