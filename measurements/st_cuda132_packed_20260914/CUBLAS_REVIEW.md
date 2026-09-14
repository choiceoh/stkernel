# cuBLASLt를 다듬어 ST GEMM을 이길 수 있는가

**가능성은 있다. 큰 FP8 프리필, 이미 FP8인 drafter FC와 vocabulary head가 우선 대상이다.**
최근 크게 줄인 C1 W4A8 커널을 통째로 대체하는 것은 우선순위가 낮다.
이 검토는 `de8bfff6`의 실행 경로와 CUDA 13.2.1 CPU 이미지의 소스를 읽고,
스케일 표현·저장량을 계산한 결과다. cuBLAS 실행 시간이나 디코드 개선율은 측정하지 않았다.
운영자의 큐 금지 지시에 따라 GPU·플릿·원패스를 실행하지 않았다.

## 실제 비교 대상

| 대상 | 현재 코드와 크기 | 튜닝 가능성 | 주의할 비용 |
|---|---|---|---|
| 큰 FP8 프리필 선형층 | `FP8Linear.project_quantized` → `deep_gemm.fp8_gemm_nt`; M=1024/8192/32256 등 | 가장 유망. 큰 M,N에서 타일·파이프라인 선택의 차이를 활용 | 스케일 배치, 출력·후처리, 긴 컨텍스트의 메모리 여유 |
| 디코드 drafter FC | 기본 `SERVING_POLICY.fc_precision='fp8'`; N=4096, K=4096×target layer 수(5개일 때 20480) | 별도 유망 후보. 긴 K에 맞는 타일과 지원되는 Split-K를 탐색 | 작은 M의 낭비, reduction 추가 발사와 쓰기 비용 |
| 디코드 vocabulary head | 항상 FP8; TP4 N=38720, 패딩 N=38784, K=4096 | W4 변환 부담 없이 비교 가능. 실제 target/draft 호출 크기별로 판단 | 큰 가중치의 읽기량, 작은 M에서의 대역폭·패딩 |
| C1 W4A8 dense | 예: M=8,N=6144,K=4096; 정렬된 K 부분곱과 기존 반올림을 유지하는 전용 커널 | 상대적으로 낮음. unpack 제거가 읽기량 증가를 상쇄해야 함 | cuBLAS FP8로 풀면 weight+scale payload 약 1.83배 |
| 융합 shared MLP·routed MoE | gate/up, 제한된 SwiGLU, BF16 반올림, 다음 FP8 입력 또는 route scatter까지 결합 | 단독 GEMM 수치만으로 대체하기 어려움 | 추가 중간 텐서·발사, 연산 순서 및 route 소유권 |

실행 근거는 [DenseLinear/FP8Linear](../../engine/kernels/dense/__init__.py),
[drafter 기본 정책](../../engine/profiles/glm53/draft_policy.py),
[drafter 구성](../../engine/profiles/glm53/drafter.py),
[target head](../../engine/profiles/glm53/net.py)다.
M=1/4/7/8/28/32는 작은 행 수의 검토 셀이다. 최종 목록은 실제 캡처되는 target/draft 형상에서 뽑아야 한다.

기존 C1 gate/up의 **76.73→30.27µs(-60.5%)**는 [#946의 실제 가중치 RTN W4, warm captured B/A/A/B](../st_forward_register_20260914/README.md)다.
CUDA 13.0에서의 구성요소 기록이며 현재 CUDA 13.2의 cuBLAS 비교 기준값이 아니다.
같은 기록의 evicted 결과에는 변동과 회귀가 있어 30.27µs를 모든 캐시 상태의 하한으로 취급하지 않는다.

NVIDIA는 CUDA 13.2에서 DGX Spark의 큰 M,N MXFP8/NVFP4 일부 형상을 개선했다고 명시한다.
이것은 ST 대비 개선율이 아니다. 새 MXFP8 grouped GEMM의 지원 기종도 GB10과 구분해야 한다.
[CUDA 13.2 release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)

검사한 이미지의 DeepGEMM SM120 구현도 이미 block-scaled FP8 MMA, persistent loop,
스케일 레지스터 재사용, Split-K 구현을 갖고 있다. [코드 지문](codegen.json)의 해당 헤더·행을 기록했다.
따라서 새 하드웨어 명령의 사용 자체가 cuBLAS만의 이점은 아니다. 실제 선택된 커널·타일의 효율을 비교해야 한다.

## 형식 변환을 별도 커널로 만들 필요는 없다

ST는 activation의 128개 묶음과 weight의 128×128 블록에 FP32로 저장한 2의 거듭제곱 스케일을 쓴다.
[양자화](../../engine/kernels/dense/fp8.py)와 [FP8 packing](../../engine/kernels/dense/packing.py)의 실제 계약이다.
GB10에서 쓸 cuBLAS MXFP8은 32개 단위 UE8M0 스케일이다. cuBLAS의 기존 128개 단위 FP32 scaling 모드는
CC9.0 전용이므로 그 포인터를 그대로 넘길 수 없다.
[cuBLAS scaling/layout 및 FP8 계약](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html#narrow-precision-data-types-usage)

우리 입력에 맞춰 다음처럼 구성할 수 있다.

1. 기존 FP8 값 바이트를 그대로 사용한다. 한 128개 묶음의 스케일을 네 32개 묶음에 동일하게 반복한다.
   activation 양자화 커널이 UE8M0 네 바이트를 한 워드로 바로 쓰면 별도의 스케일 변환 발사를 없앨 수 있다.
2. weight의 스케일만 준비 단계에서 MX 배치로 펼친다. 이미 FP8인 weight는 다시 복사하거나 양자화하지 않는다.
   기존 pack storage와 공유할 소유권·수명, 추가 상주 바이트는 arena 준비 단계에서 반영한다.
3. 입력과 출력의 물리 전치를 피한다. cuBLAS column-major TN에서 `Yᵀ = W × Xᵀ`로 기술하면
   기존 row-major W/X와 BF16 출력 버퍼를 그대로 해석할 수 있다. A/B 스케일 포인터와 bias 축도 이에 맞춘다.
4. [packet consumer](../../engine/kernels/prefill_collectives/consumer.py)의 BF16 반올림을 레지스터에서 유지하고
   최종 스케일 저장 주소만 바꾼다. 큰 BF16 중간 텐서를 다시 만들지 않는다.

MX 스케일은 128행×4열 타일 안에서 배치된다. G=K/128일 때 row r, 기존 scale group g의 네 바이트 시작은
`(r//128*G+g)*512 + (r%32)*16 + ((r%128)//32)*4`다.
[cublas_review.py](cublas_review.py)는 지수 255개, 257행의 패딩·주소·scale-value 대응 13,824바이트를 검사했다.
[CPU 결과](cublas-feasibility.json)는 PASS다. 이 증거는 동일한 dequantized 입력값을 표현할 수 있다는 뜻이며,
cuBLAS가 선택할 MMA 누적 순서·최종 BF16 출력·수용률까지 같다는 뜻은 아니다.

| N=6144,K=4096의 저장량 | 기존 | MXFP8 | 차이 |
|---|---:|---:|---:|
| FP8 weight+scale | 24.0059 MiB | 24.75 MiB | +762 KiB, 약 3.10% |
| W4 weight+scale+row scale와 비교 | 13.5234 MiB | 24.75 MiB | 약 1.83배 |
| activation scale, M=8192 | 1 MiB | 1 MiB | 같은 payload 바이트 수 |
| activation scale, M=8 | 1 KiB | 16 KiB | +15 KiB의 scale 행 패딩 |

이 표는 저장량이지 실제 DRAM 전송량이나 소요 시간이 아니다. activation의 swizzled store가 덜 합쳐지면
바이트 수가 같아도 느려질 수 있다. 작은 M에서 scale 패딩과 GEMM 자체의 형상 지원을 따로 확인해야 한다.
weight 스케일만 늘릴 때 head는 약 4.70 MiB, 5-layer FC는 약 2.48 MiB가 추가된다.
DeepGEMM 내부의 임시 scale 변환·버퍼 재사용 여부는 외부 Python 계약만으로 확정하지 않았다.

## cuBLAS를 다듬을 구체적인 순서

공개 cuBLASLt API의 지원 capability를 조회하고, 그 범위 안에서 알고리즘·tile·stages·Split-K/reduction·swizzle을 고른다.
지원하지 않는 조합은 `cublasLtMatmulAlgoCheck`에서 제외한다. 이 API와 workspace 제한은
[cuBLASLt tuning API](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html#cublasltmatmulalgoconfigsetattribute)에 정의돼 있다.

- 큰 프리필과 작은 FC/head를 다른 shape plan으로 준비한다. 첫 heuristic 한 개에 고정하지 않고,
  예를 들어 32~64개 후보에서 중복을 제거해 지원되는 후보만 남긴다.
- workspace 0/8/32/64 MiB를 검토하되 모델 전체의 메모리 여유를 제한으로 둔다.
  handle·descriptor·workspace는 캡처 전에 만들고, 겹치지 않는 호출은 stream 소유 버퍼를 재사용한다.
- plan key는 M/N/K, stride, type, scaling mode, GPU, cuBLAS/컴파일러 버전을 포함한다.
  실행 중 검색·할당·전치 없이 지정된 출력에 바로 쓴다.
- FP8 바이트와 스케일, FP32 accumulation/BF16 출력 계약을 유지한다. FAST_ACCUM이나 A4 재양자화로
  정밀도를 낮춰 얻은 결과를 동일 계산의 개선으로 분류하지 않는다.
- host alpha=1, beta=0부터 시작한다. 현재 ST의 PDL-off 정책을 유지하며, shared-weight strided batched 호출을
  성급하게 추가하지 않는다. CUDA 13.2 release notes의 sm120/121 broadcast-input 접근 문제가 관련되기 때문이다.
- FC/head의 작은 M에서는 패딩에 따른 실제 연산량과 reduction 비용까지 계산한다.
  head는 FP8 가중치 읽기만으로 한계에 가까울 수 있어, FC와 같은 개선 폭을 가정하지 않는다.

정상적으로 지원되는 형상에서만 아래 부등식을 확인하면 된다.

`T(cuBLAS tuned GEMM) + Δ(입력 준비) + Δ(후처리·출력) < T(현재 ST GEMM)`

Δ는 현재 경로 대비 추가/절감 비용이다. 양쪽이 원래 하는 양자화 비용을 cuBLAS 쪽에만 다시 더하면 안 된다.
동일 CUDA 13.2 이미지·pack·입력·출력 목적지·graph·캐시 상태의 B/A/A/B로 먼저 구성요소를 비교하고,
그다음 전체 step/s와 수용률에서 영향이 남는지 봐야 한다. GPU 실행은 이번 검토에 포함하지 않았다.

**구현 판단:** 독립적인 cuBLAS 호출을 넓게 추가하는 것보다, FP8Linear의 양자화/스케일 생산과
shape plan을 함께 설계하는 것이 타당하다. 큰 프리필 → FP8 FC → FP8 head 순서로 우위를 확인할 가치가 있다.
현재 근거로 W4A8·융합 MoE까지 일괄 교체하거나 24 step/s 달성을 예측할 수는 없다.
