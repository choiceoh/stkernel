# GB10: KDA recurrent 상태 전치 제거

DGX Spark / GB10에서 엔진의 `[H,K,V]` 상태를 KDA recurrent 커널이 직접 읽고 쓴다.
기준은 이 작업의 실제 분기점인 **`e80882d8` (PR #558)** 이다. vLLM 분리 이후의 ST 경로와
비교했으며, 기준 커널과 드라이버를 해당 커밋에서 별도로 추출해 같은 프로세스에 로드했다.

## 변경과 하드웨어 근거

GB10은 CPU/GPU가 UMA와 LPDDR5x 대역폭을 공유한다. GLM TP4의 rank별 KDA 상태는
16 heads × 128 × 128 × FP32 = **1 MiB**다. 기존 adapter는 입력 상태를 `[K,V]→[V,K]`로
복사하고, 계산한 모든 토큰의 상태를 `[V,K]→[K,V]`로 다시 복사했다.

커널에 명시적 `STATE_KV` 주소 계산을 추가했다. 실제 recurrence는 기존 `[BV,BK]` 타일에서
계산하고, 읽기·쓰기 주소만 canonical `[K,V]`를 가리킨다. 1~6토큰·16 heads·128×128에는
실측한 BV16·1 warp를 사용한다. 128 CTAs이며 튜닝 결과 154 registers/thread,
512 bytes shared/CTA다. 더 큰 형상에는 BV8·1 warp를 유지한다. 기존 VK 경로는 BV8이다.

초기 상태를 덮어쓰지 않고 **모든 토큰의 FP32 상태**를 반환하므로 거절된 draft의 이전
위치로 이어지는 계약을 유지한다. `kv` API는 dense 단일 sequence, 별도 출력 상태만 허용한다.
비정방형 상태도 검사하며, 이 과정에서 발견한 padded K의 무마스크 gate 읽기를 수정했다.

context 0에서는 영 상태 버퍼를 유지한다. 이를 생략한 첫 구현은 Triton이 합산 순서를 바꿔
6토큰의 일반 실행과 그래프 실행 결과가 달라졌다. 두 실행이 같은 초기 상태 specialization을
사용하도록 수정한 후 실제 가중치 검사에서 출력과 전체 상태 링의 바이트가 일치했다.

## 실제 L0 KDA 블록

실제 rank 0의 L0 KDA 가중치 **70,492,480 bytes**를 읽었다. 입력은 seeded synthetic BF16
activation이다. `Glm53Net._kda`를 그대로 호출하므로 projection, gate projection, causal conv,
recurrent KDA, 출력 norm/gate/projection과 그래프의 상태 링 읽기·쓰기가 포함된다.
통신은 단일 rank를 분리해 항등 처리했다. mHC·MLP·나머지 layer는 이 시간에 포함되지 않는다.

| 형상 | 기존 블록 | 변경 블록 | 지연 감소 |
| --- | ---: | ---: | ---: |
| 1토큰 decode | 566.688 µs | 500.704 µs | **11.64%** |
| 6토큰 verify | 531.136 µs | 441.024 µs | **16.97%** |

동일 srv4, 동일 이미지에서 CUDA graph를 AB/BA 순서로 11회 번갈아 측정한 중앙값이다.
각 샘플 직전에 같은 그래프를 한 번 replay했다. 측정 context는 4096, physical slot은 1이다.
서빙 서비스가 함께 실행 중인 환경의 단일 실행 결과이며, 여러 기기·날짜에 걸친 통계는 아니다.
원시 샘플과 가중치별 해시는 `real.json`에 있다.

## recurrent adapter만 분리한 시간

아래는 dense q/k/v 입력에서 기존/변경 adapter의 출력과 모든 상태 할당까지 포함한다.
`warm`은 8회 호출을 묶은 그래프의 호출당 시간, `evicted`는 64 MiB를 먼저 써서 L2를
밀어낸 뒤 1회 호출 그래프를 잰 시간이다. cache eviction 쓰기는 측정 구간 밖이다.
실제 projection의 strided q/k/v를 contiguous로 만드는 비용은 양쪽에 동일하며 이 표에는 없다.

| 토큰 | 초기 상태 | warm 기존 → 변경 | evicted 기존 → 변경 |
| --- | --- | ---: | ---: |
| 1 | 없음 / context 0 | 21.296 → 7.660 µs | 42.720 → 28.480 µs |
| 1 | 있음 | 31.748 → 5.096 µs | 51.936 → 26.272 µs |
| 6 | 없음 / context 0 | 96.100 → 35.256 µs | 127.648 → 64.448 µs |
| 6 | 있음 | 105.068 → 33.240 µs | 137.312 → 63.136 µs |

T=1~6,12와 초기 상태 유무의 14조건, 각각 11회 AB/BA 측정이다. 전체 원본은 `perf.json`이다.
`tune.json`은 타일을 고르는 초기 실험으로, 출력 버퍼를 재사용하는 낮은 수준 커널 호출이다.
위 adapter 측정과 버퍼 생명주기·working set이 달라 절대 시간을 직접 비교하면 안 된다.

## 복사·메모리와 수치 검증

6토큰·초기 상태 있음·dense 입력을 별도로 프로파일했다. 측정 중에는 프로파일러를 켜지 않았다.

| 항목 | 기존 | 변경 |
| --- | ---: | ---: |
| 상태 전치 복사 CUDA 커널 | 2 | 0 |
| recurrent CUDA 커널 | 1 | 1 |
| 호출의 추가 allocator 최고 사용량 | 13,656,064 B | 6,316,032 B |

차이는 **7 MiB**다. 입력 1 MiB와 결과 6 MiB의 임시 전치 사본을 없앴다.
이로부터 계산되는 불필요한 읽기+쓰기 감소는 **14 MiB/호출/rank/KDA layer**이며,
이는 논리적 tensor traffic 계산이다. DRAM 하드웨어 카운터로 측정한 traffic은 아니다.
세부 CUDA 이벤트와 aten 연산은 `legacy-profile.json`에 있다.

주소 배치 변경은 Triton의 reduction 배치도 바꾸므로 기존 경로와 bit exact라고 주장하지 않는다.
14조건에서 변경 경로와 기존 경로의 `max(abs(delta))/max(abs(reference))`는
BF16 출력 최대 **2.79018e-4**, FP32 모든 상태 최대 **1.41258e-7**이었다.
독립 PyTorch recurrence 대비는 출력 **3.32447e-4**, 모든 상태 **2.36166e-7**이었다.
이는 전체 tensor 최대값으로 정규화한 오차이며, 작은 개별 원소의 상대 오차 상한이 아니다.

실제 L0 블록의 16조건에서는 기존 대비 출력 최대 **2.53808e-3 (0.254%)**,
상태 링 최대 **6.05794e-8**이었다. output norm과 projection을 지난 출력 수치다.
prefill 64→verify 6→4개 draft 거절 후 재검증→decode를 검사했고,
그래프는 T=1/6, context=0/1/7/8/4095/4096, slot=1/2를 바꿔 검사했다.
각 조건에서 **변경 경로의 일반 실행과 그래프 실행은 출력·전체 링 바이트가 동일**했다.

회귀 테스트는 다음 계약을 검사한다.

- 독립 recurrence에 대해 매 토큰 상태·출력·입력 불변성: 3가지 K/V 형상,
  T=1/2/6/7/12/64, 초기 상태 유무의 36조건.
- 그래프 입력/초기 상태 변경, 허용한 draft prefix의 상태에서 다음 replay를 시작하기.
- 32×6=192토큰 연속 실행, gate lower bound -5와 -0.01에서의 상태 누적.
- context 0의 eager와 그래프 영 상태 입력 간 바이트 일치.
- 잘못된 배치·dtype·stride·in-place/varlen/인덱스 테이블 조합의 명시적 거절.
- 기본 VK API는 원래 드라이버·커널과 9조건에서 출력·상태 비트가 동일함을 별도 검사.

최신 main **`33f26afb` (PR #559/#560)** 을 통합한 소스에서 전체 GPU 회귀 검사
**239개가 통과했으며 skip은 없었다** (`gpu-tests.log`). 각 프로브의 정상 종료 기록은
`execution.json`에 있다. 통합 과정에서 KDA 커널·드라이버·레인 소스는 측정 당시와 바이트가
같음을 확인했다. 이 파일들의 SHA256은 `perf.json`에 별도로 기록되어 있다.

## 재현

테스트한 runtime은 `st-engine:9391`, image id는 `runtime-image.txt`에 있다.
GPU는 GB10, Torch 2.13.0+cu130 / CUDA 13이다.
baseline/source 해시는 `perf.json`, 측정 당시 엔진 전체는 `measured-engine-code-sha256.json`,
main 통합 후 전체 suite를 실행한 엔진은 `engine-code-sha256.json`이다.
통합한 엔진 전체 해시를 실제 mount된 소스와 대조했다.
가중치와 checkpoint config는 `real.json`에 기록했다.

```bash
mkdir -p /work/baseline
git show e80882d8:engine/kernels/kda/kda.py > /work/baseline/kda.py
git show e80882d8:engine/kernels/kda/fused_recurrent.py > /work/baseline/fused_recurrent.py
python3 probes/engine_kda_state_tune.py --baseline /work/baseline/fused_recurrent.py --output /work/tune.json
python3 probes/engine_kda_state_perf.py --baseline-dir /work/baseline --output /work/perf.json
python3 probes/engine_kda_state_real.py --baseline-dir /work/baseline \
  --checkpoint /meta --rank-file /ranks/rank0of4.safetensors --output /work/real.json
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
```

source를 `/work`에 mount하고 `-w /work -e PYTHONPATH=/work`를 설정한다.
이미지 entrypoint를 명시적으로 덮어써야 한다. 예: `--entrypoint timeout ... 900 python3 ...`.
GPU 프로브는 allocator를 768 MiB/1 GiB로 제한했고 전체 suite는 2 GiB로 제한했다.
컨테이너는 5~6 GiB 메모리·2 CPU로 제한했다. 전체 suite의 checkpoint 기반 검사는
실제 checkpoint를 `/home/choiceoh/models/glm53-redhat-nvfp4`에 read-only mount해야 한다.
config만 제공하면 loader의 실제 tensor 비교 검사가 실패한다.

전체 모델 생성 품질·DFlash acceptance·4노드 NCCL 포함 성능·tokens/s·서비스 ITL은
측정하지 않았다. 기존 서비스를 재시작하거나 배포하지 않았다.
