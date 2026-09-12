# GB10 KDA 상태 링 직접 읽기·쓰기

**recurrent KDA가 초기 FP32 상태를 링에서 직접 읽고, 매 토큰의 상태를 같은 링에 바로 저장한다.**
기존의 초기 상태 gather와 출력 snapshot 복사를 제거했다. GB10에서 상태 처리 구간의 CUDA
커널은 **3 → 1**, 6토큰 추가 할당 peak는 **7,364,608 → 24,576 bytes**로 줄었다.
6토큰 warm 지연은 **70.304 → 15.816 µs**, AB/BA 쌍별 감소율은 **77.54%**였다.

기준은 입력 stride 최적화 PR #569까지 병합한 main **`9e5b082e`**다. 구현은 **`99ad1337`**.
실제 L0 가중치 블록은 두 번의 측정에서 6토큰 약 **5.85~6.07% 감소**를 관찰했다.
전체 블록의 1토큰 개선은 확인하지 못했다. 모든 수치는 srv4 한 기기의 격리된 TP4 rank 0
블록 또는 상태 구간 측정이며, 4기기 collective를 포함한 serving throughput/ITL 결과는 아니다.

## 구현과 하드웨어 근거

TP4의 KDA 상태 하나는 `[16,128,128]` FP32, **1 MiB**다. 현재 verify 폭과 recurrent ring
폭은 모두 6이다. 기존 graph 경로는 다음 메모리 이동을 수행했다.

1. `_read_rec`: 링의 이전 상태를 1 MiB 임시 버퍼로 복사.
2. recurrent: 초기 버퍼를 읽고 토큰별 상태를 `[T,16,128,128]` 임시 버퍼에 저장.
3. `_write_ring`: 임시 snapshot을 읽어 `(context+i)%6` 링 셀에 저장.

새 `engine/kernels/kda/ring.py::recurrent_kda_ring`은 기존 recurrent 계산을 재사용하면서
초기/최종 상태 주소만 링으로 연결한다. 출력은 dense tensor 하나이고 링 변경은 명시적인
별도 lane 계약이다. 기존 functional recurrent API는 초기 상태를 수정하지 않고 snapshot을 반환한다.
`reference_for=("kda_recurrent",)`와 reference table은 functional 경로를 계속 사용한다.
큰 prefill의 chunk 경로는 그대로다.

초기 상태가 있는 6토큰의 논리적 state load/store는 **21 → 7 MiB**, 즉 **14 MiB 감소**한다.
1토큰은 **6 → 2 MiB**다. 이는 코드에서 계산한 global-memory 요청량이며 DRAM counter로
측정한 양은 아니다. L2 hit 여부에 따라 실제 LPDDR 트래픽은 다르다. 중복 global-memory
접근을 줄이는 방향은 [NVIDIA Blackwell tuning guide의 기본 권고](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#cuda-best-practices)와도 일치한다.

GB10에서 기존과 같은 **BK128, BV16, 1 warp/CTA, 128 CTA**로 실행한다.
컴파일 결과는 register/thread **154 → 167**, shared memory **512 bytes/CTA**, spill **0**이다.
새 tensor-core 명령을 사용하는 변경은 아니며, 기존 FP32 recurrence의 연산 순서와 reduction
형태를 유지하면서 상태 이동과 launch 비용을 줄인다.

## 상태 소유권과 graph

- eager는 한 슬롯의 링 view와 Python context를, graph는 전체 arena의 링 view와 CUDA
  slot/context singleton을 전달한다. graph replay에서 값이 바뀌어도 host read가 없다.
- 각 CTA는 모든 링 행에서 서로 겹치지 않는 `[head,K,V]` 조각을 소유한다. `T == ring width`
  때문에 마지막 출력이 초기 상태 행을 덮어써도 해당 조각은 같은 CTA가 먼저 register에 읽었다.
  다른 CTA의 초기 상태를 덮어쓰는 교차 의존성이 없다.
- context 0은 초기 load를 mask하여 NaN이 남은 링에서도 0 상태로 시작한다. 이전 구현과
  새로운 eager/graph의 출력 및 전체 상태 arena가 바이트 단위로 일치했다.
- 모든 토큰의 FP32 snapshot을 보존하므로 accepted prefix의 위치로 돌아가서 계속 실행할 수 있다.
  초기 상태 행의 수명도 기존 ring-write 계약과 같다.
- 다른 슬롯과 arena padding은 수정하지 않는다. 서로 다른 CUDA stream의 호출은 서로 다른
  슬롯을 소유해야 한다. device slot/context의 값 범위는 기존 scheduler/cache 계약이 보장한다.
- wrapper는 shape/dtype/stride, host index 범위, 입력과 링 주소 범위 겹침을 검사한다.
  device 값은 host로 읽어 검사하지 않는다. 슬롯 사이 padding을 읽는 입력도 보수적으로 거절한다.

## 성능 측정

각 조건은 12 round, AB/BA 각각 6회다. 인접 AB/BA 두 round에서 각 경로의 평균을 구하고,
6개 cycle의 중앙값 및 쌍별 차이/감소율 중앙값을 보고한다. 두 시간 중앙값을 단순히 빼거나
나눈 값과 paired 수치가 조금 다를 수 있다. 모든 원시 sample은 JSON에 포함했다.

### Recurrent와 상태 이동 전체

`probes/engine_kda_ring_perf.py`는 frozen `kda.py`/`fused_recurrent.py`와 기존 `_read_rec` 및
`write_ring`의 조합을 새 ring lane과 비교한다. 실제 conv split 및 projection beta stride를 사용한다.
conv 자체와 conv history는 이 구간에 포함하지 않는다. old/new가 별도 링을 사용하며,
correctness 검사 전에는 동일한 원본 링을 복원한다. 8회 연속 상태 갱신 graph도 바이트 일치를 확인한다.

warm은 8회 호출 graph를 먼저 replay한 뒤 호출당 시간을 측정한다. evicted는 측정 전에
64 MiB 버퍼를 써서 cache를 밀어낸 뒤 1회 호출 graph를 측정한다. Python 검사/allocator의
host 시간은 graph replay에 포함되지 않는다. Profiler는 모든 타이밍이 끝난 뒤 별도로 실행했다.
아래는 context=4096이며 context=0 결과도 `perf.json`에 있다.

| 토큰 | warm 기존 → 변경 | paired 감소 | evicted 기존 → 변경 | paired 감소 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11.735 → 5.098 µs | 56.48% | 43.160 → 28.672 µs | 32.89% |
| 2 | 16.364 → 7.391 µs | 54.92% | 55.424 → 36.208 µs | 34.18% |
| 6 | 70.304 → 15.816 µs | 77.54% | 104.008 → 63.120 µs | 39.80% |

| 토큰 | 기존 추가 할당 peak | 변경 | 감소 | CUDA 커널 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2,101,248 B | 4,096 B | 2 MiB | 3 → 1 |
| 6 | 7,364,608 B | 24,576 B | 7 MiB | 3 → 1 |

출력 tensor는 계속 할당하고, 링 자체도 계속 유지한다. 이 표는 호출의 임시 저장소 절감이며
모델 전체 상주 메모리나 전체 graph pool의 절감량으로 확대 해석하면 안 된다.

### 실제 L0 KDA 블록

rank 0 L0의 실제 가중치 **70,492,480 bytes**, seeded synthetic activation, identity all-reduce를
사용했다. frozen main `net.py::_kda`와 변경된 메서드를 같은 current lane/cache에 연결하여
projection, conv, recurrent, ring 접근, output norm, output projection 전체를 비교했다.
샘플마다 graph 8회 replay. timed context=4096, physical slot=1.

| 실행 | 토큰 | 기존 cycle 중앙값 | 변경 cycle 중앙값 | paired 감소 시간 | paired 감소율 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 최초 | 1 | 727.438 µs | 729.222 µs | -2.804 µs | -0.39% |
| 최초 | 6 | 749.098 µs | 706.617 µs | 43.896 µs | 5.85% |
| 2시퀀스 검사 확장 후 | 1 | 717.397 µs | 724.805 µs | -1.426 µs | -0.20% |
| 2시퀀스 검사 확장 후 | 6 | 717.924 µs | 675.193 µs | 43.665 µs | 6.07% |

첫 실행은 `real-first.json`, 두 번째는 `real.json`에 보존했다. 두 번째 실행은 정확성 검사를
2시퀀스로 확장한 probe를 실행한 것이며 해당 검사는 타이밍 뒤에 수행했다.
공유 GPU에서 큰 지연 변동이 있었다. 예를 들어 두 번째 6토큰 cycle별 차이는
**-98.172~52.164 µs**였다. 원인별 부하를 격리하지 않았으므로 블록 수치는 참고 관찰로 취급한다.
6토큰 두 실행의 중앙 개선 방향은 같고, 1토큰 전체 블록의 개선은 주장하지 않는다.

## 검증과 재현

- GB10에서 `test_engine_*.py` **283 tests, 115.730 s, 실패 0 / skip 0**.
- 신규 ring tests 7개: T1~6의 모든 snapshot, 초기 행을 덮는 ring wrap, slot 0/1/2,
  padded arena, zero-context NaN masking, graph slot/context 변경 및 accepted-prefix 재시도,
  grouped heads 및 K33/V17 tail, BF16/FP16/FP32, 독립 Torch recurrence, 입력 겹침 거절,
  두 CUDA stream의 서로 다른 슬롯, reference bisect binding.
- 실제 L0 **22조건**에서 이전/현재 출력 및 전체 cache arena bytes exact. prefill64, verify6,
  reject4 후 retry6, decode1, T1/6 graph context0/1/7/8/4095/4096 및 physical slot1/2,
  두 시퀀스의 서로 다른 context/slot 배치를 포함한다. current graph와 eager도 bytes exact.
- 기준 frozen recurrent와 직접 ring 구간 비교 **6조건**, 1회 및 8회 연속 graph도 bytes exact.
- 전체 engine 소스 **138파일**의 로컬/원격 SHA256 일치를 확인했다.
- `compileall` 및 `git diff --check` 통과. 포팅한 shared kernel의 `SOURCES.json` 해시를 갱신했다.

첫 targeted 실행에서 grouped-head Torch reference 입력을 그대로 전달하여 shape 오류가 났다.
reference는 동수 head를 요구하므로 Q/K와 gate parameter를 value-head 수에 맞게 반복하여
독립 oracle 비교를 수정했다. 커널 수치 불일치는 없었으며 수정 후 targeted 및 전체 suite가 통과했다.
처음 오류와 성공 로그 모두 `targeted-tests.log`에 보존했다.

srv4 `NVIDIA GB10`, driver `580.159.03`, PyTorch `2.13.0+cu130`, Triton `3.7.1`, CUDA runtime `13.0`.
기존 `st-engine:9391` image ID는
`sha256:fa1183e3dd660f09f42ee72918948d53466a96789babc58d6a4bd3a21147a9a7`다.
별도 task container를 사용했고 실행은 CPU 2개, host memory 5 GiB(전체 suite 6 GiB), timeout 600s로
제한했다. CUDA allocator cap은 perf 1.5 GiB, 실제 가중치 1 GiB, 전체 suite 2 GiB다.
기존 서비스는 계속 실행한 상태에서 측정했다.

작업 디렉터리는 `/tmp/st-gb10-ring-9391`, 재사용 Triton cache는
`/tmp/st-gb10-kda-9391/cache/triton`이다. baseline 세 파일은 `9e5b082e`에서 추출했다.
아래 probe는 `/work`에 repo와 baseline을 마운트한 동일 image에서 실행한다.

```bash
python3 probes/engine_kda_ring_perf.py --baseline-dir /work/baseline --output /work/perf.json
python3 probes/engine_kda_state_real.py --baseline-net /work/baseline-net.py \
  --checkpoint /meta --rank-file /ranks/rank0of4.safetensors \
  --timing-replays 8 --output /work/real.json
python3 /work/st-kda-gpu-suite-9391.py
```

`/meta`는 원본 GLM config fixture, `/ranks`는
`/home/choiceoh/models/st-glm53-9391-up-gate-full`의 read-only mount다. 전체 suite에는 원본 checkpoint
`/home/choiceoh/models/glm53-redhat-nvfp4`를 같은 경로에 read-only mount했다.
실제 rank 파일 전체를 로드하지 않고 필요한 L0 KDA key 범위만 읽고 hash했다.
실행 인자·종료 상태·자원 제한은 `containers.txt`, source/probe/test hash는 `provenance.json` 및
`engine-code-sha256.json`, 원시 출력은 각 `.log`/`.json`에 있다.
