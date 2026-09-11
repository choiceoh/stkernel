# GB10 인덱서 query 양자화 launch 최적화

## 적용한 변경

GB10의 48 SM과 64K registers/SM에 맞춰 `fwht128_quant_fp8`의 launch geometry를
선택한다. 기존 Hadamard-128 커널, BF16 반올림, FP32 power-of-two scale, FP8 저장의
산술 코드는 AST로 동일함을 검사했다. 새 수치 근사나 양자화 형식은 도입하지 않았다.

| Query 행 수 R | 기존 행/block · warp/block | 적용 행/block · warp/block |
| ---: | ---: | ---: |
| 1~1,024 | 32 · 2 | 1 · 1 |
| 1,025~65,536 | 32 · 2 | 8 · 1 |
| 65,537 이상 | 32 · 2 | 32 · 2 |

빈 입력은 이전과 같이 커널을 실행하지 않는다. 입력·출력·할당 계약은 같다.
GLM의 인덱서 head는 32개이고 TP rank에 복제되므로, decode 1/6/24토큰은
각각 R=32/192/768이다. 예를 들어 6토큰에서 grid는 6 blocks에서 192 blocks로 늘어
더 많은 SM이 참여할 수 있다. 메모리나 레지스터를 많이 쓰는 block 하나로 행을 묶는
방식의 작은 입력 비용을 줄인다.

R이 tile에 정렬된 측정 사례에서 Triton이 보고한 register/thread는 기존 125개에서
작은 입력 20개, 중간 입력 66개로 줄었다. Shared memory/block은 4,096 B에서
각각 512 B, 2,048 B로 줄었다. 마스킹이 필요한 경계 형상은 register 수가 다를 수 있다.
실제 metadata는 [results.json](results.json)에 형상별로 기록했다.

별도 XOR butterfly 커널도 비교했지만 기존 커널의 tile 조정만으로 충분한 개선을 얻었다.
런타임에는 새 butterfly 구현을 추가하지 않았다. 15개 행 수와 11개 설정의 탐색 결과는
[tuning.json](tuning.json), 재현 probe는 `probes/engine_indexer_quant_tune.py`다.
탐색의 `batch_after_eviction`은 32회 실행 묶음 전에 한 번 eviction한 평균이며,
커널마다 cache를 비운 결과가 아니다. 최종 probe는 이 둘을 분리한다.

## 커널 측정

동일 srv4 GB10, 동일 `st-engine:9391` 이미지에서 기준/변경을 번갈아 실행한 9회 중앙값이다.
warm은 같은 커널 32회가 들어 있는 CUDA Graph의 실행 시간을 32로 나눈 값이다.
evicted는 64 MiB 버퍼를 갱신한 뒤 **커널 한 번**을 실행한 CUDA Graph의 시간이다.
eviction, 출력 할당, Python launch와 검사 시간은 kernel event 구간 밖에 있다.
행 수별 입력·출력은 기준/변경에 동일하게 제공한다. 단위는 µs다.

| 토큰 수 | R | 기존 warm | 변경 warm | 감소 | 기존 evicted | 변경 evicted | 감소 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 5.367 | 1.656 | 69.1% | 8.192 | 5.728 | 30.1% |
| 6 | 192 | 5.432 | 1.720 | 68.3% | 8.544 | 5.760 | 32.6% |
| 24 | 768 | 5.496 | 2.180 | 60.3% | 10.240 | 6.528 | 36.2% |
| 256 | 8192 | 12.676 | 7.815 | 38.3% | 27.648 | 20.480 | 25.9% |
| 512 | 16384 | 20.537 | 13.113 | 36.1% | 37.888 | 32.768 | 13.5% |

R=65,537 이상은 코드 경로가 동일한 대조 조건이다. 여기서 관찰되는 시간 차이를
최적화 효과로 해석하지 않는다. 기존 서비스는 계속 실행되었고 GPU 클록·호스트 부하는
고정하지 않았다. 특히 L2보다 큰 입력과 아주 짧은 단일 커널 시간에는 측정 변동이 있다.
원시 9개 sample을 모두 보존했다.

## 실제 인덱서와 정확성

실제 L3의 7개 인덱서 tensor, 총 15,207,424 B를 읽고 synthetic activation으로
`Glm53Net._indexer`를 실행했다. 기준 lane table은 query quantizer만 Git 기준 구현으로
교체한다. 나머지 scoring, pooling, 슬롯 선택 및 cache 동작은 두 경로가 같다.
weight 파일은 읽기 전용 입력이며 이 저장소에 포함하지 않는다.

- context 63/256/2,048에서 prefill·6토큰 verify·2토큰 수락 후 rollback, 총 9개 조건의
  선택 슬롯·유효 개수·pooled KV/scale·tail-ring 바이트가 일치했다.
- 공개 query quantizer의 25개 경계 형상 × 5개 입력 크기(1e-20~1e12)가 기준과
  FP8 저장 byte 및 FP32 scale 저장 bit까지 일치했다. 입력을 변경하지 않는다.
- zero·signed zero·constant·alternating·128개 impulse pattern과 6개 지수 크기를
  독립 Torch 참조와 비교했다. BF16 비트 패턴을 열거한 추가 입력은 비유한 값과
  절대값 1e30 이상의 값을 0으로 치환해 Hadamard 중 overflow를 피했다.
- 7개 cutoff 전후 형상에서 query를 바꾸는 graph replay 28회를 통과했다.
  두 독립 CUDA stream, 입력 형상·dtype·contiguity 검사도 통과했다.
- 최종 성능 probe의 15개 형상은 별도로 graph replay 60회와 bit 일치를 확인했다.
  실제 served lane table을 통한 LocalTP 논리 rank 4개의 호출도 일치했다.

실제 인덱서의 CUDA Graph 측정은 다음과 같다. 두 형상 각각 9개 교대 라운드이며,
graph 출력과 cache를 매 라운드 비교했다. 단위는 µs다.

| 전체 L3 인덱서 | 기존 graph | 변경 graph | 감소 |
| --- | ---: | ---: | ---: |
| decode, 6토큰 | 231.072 | 226.976 | 1.8% |
| prefill, 256토큰 | 336.192 | 326.656 | 2.8% |

eager wall 시간은 회차마다 개선 방향이 바뀌었다. 첫 회차는 decode 2,023→2,114 µs,
prefill 2,218→2,253 µs였고, graph를 추가한 회차는 각각 2,742→2,736 µs,
2,880→2,874 µs였다. CPU 제출·호스트 부하의 변동이 크므로 eager 전체 호출의 개선율은
확정하지 않는다. 첫 회차도 [repeat-before-graph.json](repeat-before-graph.json)에 보존했다.
R>=32,768의 warm kernel 시간 역시 반복 간 차이가 커 대표 표에서 제외했다.

GPU engine unit suite는 **193개 통과, 건너뜀 0개**다. Torch가 없는 개발 환경은
**128개 통과, 65개 건너뜀**이다.

이 결과는 인덱서 구성요소와 실제 가중치를 사용하는 인덱서의 검사다. 45-layer 전체 모델의
tokens/s·ITL·TTFT, TP4 네트워크 성능 또는 DFlash 수락률 개선을 입증하지 않는다.
일반 실행의 wall 측정은 CPU 제출·동기화·스케줄링을 포함한다. Graph와 kernel 시간으로
대체하거나 서로 더하지 않는다. 계산 결과의 bit 일치가 모든 모델 품질 검사를 대신하지는 않는다.

## 실행 환경과 재현

- 기준 commit: `355c8600` (PR #554 MLA 병합 후).
- srv4 NVIDIA GB10, CC 12.1, 48 SM; driver 580.159.03, CUDA 13.0, PyTorch 2.13.0+cu130.
- 이미지 ID: `sha256:fa1183e3dd660f09f42ee72918948d53466a96789babc58d6a4bd3a21147a9a7`.
- 최종 probe: 컨테이너 RAM 5 GiB, CPU 2개, OMP threads 2, CUDA allocator 1.5 GiB 제한.
  단위 검사 컨테이너는 6 GiB. 기존 서비스를 재시작하지 않았고 task 컨테이너는 검사 후 제거했다.
- [provenance.json](provenance.json): 기준/변경 source, probe, 검사, config 및 weight SHA256.
- [gpu-tests.log](gpu-tests.log), [local-tests.log](local-tests.log), [probe.log](probe.log).

CUDA와 Triton이 있는 GB10 환경에서 저장소 루트를 작업 디렉터리로 사용한다.
전체 suite와 큰 입력 검사는 충분한 RAM과 CUDA allocator 여유가 필요하다.

```bash
git show 355c8600:engine/kernels/kpool.py > /tmp/st-baseline-kpool.py
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_engine_indexer_quant.py' -v
PYTHONPATH=. python3 probes/engine_indexer_quant_check.py \
  --baseline /tmp/st-baseline-kpool.py --output /tmp/st-quant-results.json \
  --checkpoint /path/to/config-directory --rank-file /path/to/indexer-only.safetensors
```

실제 L3 weight 검사 없이 커널만 재현하려면 `--checkpoint`와 `--rank-file`을 함께 생략한다.
전체 엔진 단위 검사는 `python3 -m unittest discover -s tests -p 'test_engine_*.py' -v`로 실행한다.
