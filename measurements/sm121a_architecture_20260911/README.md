# DGX Spark / GB10 / SM121a architecture investigation

2026-09-11. NVIDIA 공식 문서, srv4의 CUDA driver 속성, CUDA 13.0.88 PTX
컴파일, 작은 CUDA 실행 프로브를 대조했다. 성능 벤치마크나 전체 모델 검증은 아니다.

**GB10은 warp 단위 NVFP4 MMA와 TMA, thread-block cluster, DSMEM을 지원한다.
WGMMA 및 `tcgen05`/TMEM은 지원하지 않는다. SM당 shared memory는 100 KiB다.**
기존 megakernel 헤더의 `NO ... clusters / DSMEM` 및 `128 KB smem/SM` 설명은
실제 장치와 다르다. 이 조사에서는 실행 경로를 변경하지 않았다.

## 장치와 실행 환경

실측: srv4, NVIDIA GB10, compute capability 12.1, aarch64,
driver 580.159.03, CUDA compiler 13.0.88. 이미지 digest 및 소스 hash는
[provenance.json](provenance.json)에 기록했다.

| 항목 | 결과 | 근거 |
|---|---|---|
| GPU | GB10 Blackwell, 48 SM | driver 조회 |
| CUDA cores | 6,144 | NVIDIA 제품 문서 |
| CPU | Arm 20 cores: Cortex-X925 10 + Cortex-A725 10 | NVIDIA 제품 문서 |
| 메모리 | 128 GB LPDDR5x, 256-bit, 공칭 273 GB/s | NVIDIA 제품 문서 |
| CPU–GPU | NVLink-C2C coherent memory model | NVIDIA 발표 |
| L2 | 25,165,824 bytes = 24 MiB | driver 조회 |
| SM shared memory | 102,400 bytes = 100 KiB | driver 조회 |
| block shared memory 기본 / opt-in 최대 | 48 KiB / 99 KiB | driver 조회 |
| block당 driver shared memory 예약 | 1 KiB | driver 조회 |
| SM당 registers | 65,536 × 32-bit = 256 KiB | driver 조회 |
| warp 크기 | 32 threads | driver 조회 |
| SM당 상주 한도 | 1,536 threads = 48 warps, 24 blocks | driver 조회 |
| block당 threads 한도 | 1,024 | driver 조회 |
| cluster / cooperative launch / tensor-map access | 모두 1 | driver 조회 |
| integrated / managed / concurrent managed / pageable memory | 모두 1 | driver 조회 |
| host native atomics | 1 | driver 조회 |
| CUDA GPU Direct RDMA attribute | 0 | driver 조회; NIC RoCE 지원 여부와 다른 속성 |
| copy engine | CUDA `ASYNC_ENGINE_COUNT=1`; 제품 표는 2 | 아래 문서 차이 참조 |

공칭 1 PFLOP는 **sparsity를 적용한 FP4 AI 성능**이다. 일반 dense 모델의
지속 성능이나 모든 연산의 성능을 뜻하지 않는다. Tensor Core가 FP4를 지원한다는
사실도 FP4 일반 산술 명령 전체를 지원한다는 뜻이 아니다.
[NVIDIA hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html),
[NVLink-C2C 발표](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers).

128 KiB는 SM의 **L1/texture/shared 통합 용량**이며, 전부 shared memory로
쓸 수 없다. 48 KiB 초과 shared memory에는 dynamic allocation과 opt-in이 필요하다.
Tensor Core 입력은 TF32/BF16/FP16/FP8/FP6/FP4/INT8이 문서화되어 있다.
FP64 scalar 연산과 FP64 Tensor Core 지원은 구별해야 한다. 12.x에는 후자가 없다.
[CUDA compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

## 컴파일 타깃

장치가 보고하는 capability는 **12.1**이다. `sm_121a`의 `a`는 별도 제품이나
실리콘 stepping을 뜻하지 않고 architecture-specific feature set을 선택한다.

| 타깃 | 의미 / 이번 NVFP4 MMA 컴파일 |
|---|---|
| `sm_121` | baseline; block-scaled NVFP4 MMA 거부 |
| `sm_121a` | GB10의 architecture-specific 명령; 통과 |
| `sm_120f` | 12.0·12.1에 공통인 family feature set; 통과 |
| `sm_121f` | 현재 12.1 family subset; 통과 |
| `sm_120a` | SM12.0 전용; 문법 비교용 컴파일만 수행 |
| `sm_100a` | B200 계열 전용; GB10용 바이너리가 아님 |

`sm_120a` 바이너리가 GB10에서 호환된다는 결론을 내리지 않는다. 공식 문서상
`a` 타깃은 정확히 해당 compute capability용이고 `120f`는 12.0·12.1을 포함한다.
현재 ST의 `-gencode arch=compute_121a,code=sm_121a` 선택은 적절하다.
[CUDA target compatibility](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#feature-set-compiler-targets),
[CUTLASS target architecture](https://docs.nvidia.com/cutlass/latest/overview.html#target-architecture).

## 지원 명령어: 범용 명령과 아키텍처 전용 명령을 구분

아래는 모델 엔진과 관련된 대표 PTX 명령이다. 모든 opcode·타입·modifier의
조합을 망라한 목록은 아니다. **컴파일 통과는 수치 정확성·성능 검증이 아니다.**

| 범주 | 대표 명령 | 이번 증거 |
|---|---|---|
| BF16 / TF32 / FP8 Tensor Core | `mma.sync.aligned.m16n8k*` | SM121a 컴파일 통과 |
| FP6 Tensor Core | `mma.sync...kind::f8f6f4...e3m2.e2m3` | 컴파일 통과 |
| NVFP4 Tensor Core | `mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3` | 컴파일 통과 |
| MXFP4 Tensor Core | `mma.sync...kind::mxf4.block_scale...ue8m0` | 컴파일 통과 |
| global → shared 비동기 복사 | `cp.async`, `commit_group`, `wait_group` | 컴파일 통과 |
| TMA load / store | `cp.async.bulk.tensor.2d` 양방향 | 컴파일 통과; tensor-map attribute=1 |
| 비동기 barrier / proxy 순서 | `mbarrier.init`, `arrive.expect_tx`, `fence.proxy.async.shared::cta` | 컴파일 통과 |
| matrix shared load/store | `ldmatrix` b16·b8, `stmatrix` b8 | 컴파일 통과 |
| FP4 pack / unpack | `cvt.rn.satfinite.e2m1x2.f32`, `cvt.rn.f16x2.e2m1x2` | 컴파일 통과 |
| PDL | `griddepcontrol.launch_dependents`, `.wait` | 컴파일 통과 |
| cluster barrier / DSMEM | `barrier.cluster`, `mapa.shared::cluster`, `ld.shared::cluster` | 컴파일 및 서로 다른 SM 사이 실행 통과 |
| Cluster Launch Control | `clusterlaunchcontrol.try_cancel.async` | 컴파일 통과; 실제 work-stealing 실행 미검증 |
| BF16 pair atomic add | `atom.global.add.noftz.bf16x2` | 컴파일 통과 |
| warp 정수 reduction | `redux.sync.max.s32` | 컴파일 통과 |
| Hopper warpgroup MMA | `wgmma.mma_async`, commit/wait | SM121a 거부; 동일 문법 SM90a 통과 |
| Tensor Memory 할당 | `tcgen05.alloc` | SM121a 거부; 동일 문법 SM100a 통과 |
| FP4 stochastic rounding | `cvt.rs.satfinite.e2m1x4.f32` | SM121a 거부; 동일 문법 SM100a 통과 |

TMA의 특정 multicast/gather/scatter modifier나 최신 PTX 추가 명령까지
일괄 지원으로 해석하면 안 된다. 예를 들어 TMA 사용 가능과 `tcgen05` 사용 가능은
독립적이다. 정확한 변형은 [PTX ISA의 Target ISA Notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)를
기준으로 검사해야 한다. 이번 실험은 현재 설치된 CUDA 13.0의 PTX 9.0을 사용했다.
웹의 최신 PTX 9.4 문서에 나온 명령을 모두 사용할 수 있다는 의미가 아니다.

PDL은 의존 커널을 일찍 시작할 수 있게 하는 실행 기능이며, 앞 커널의 결과가 필요한
구간에서는 wait로 순서를 보장해야 한다. 지원 자체가 실제 overlap을 보장하지 않는다.
[CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## NVFP4 경로의 정확한 의미

GB10의 CuTe `MmaMXF4NVF4Op`는 **32-thread warp**, instruction tile
**M16 × N8 × K64**, A/B **E2M1**, **16개 값당 UE4M3 scale**, **FP32 accumulation**이다.
CTA tile과 instruction tile은 다르다. 더 큰 CTA는 여러 warp/instruction을 조합한다.
[CUTLASS warp MMA API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op).

NVFP4 표현은 보통 E4M3 block scale 위에 tensor-level FP32 scale을 둔다.
MXFP4의 32-element/E8M0 scaling과 구별해야 한다. 데이터 bit width만 같다고
scale 레이아웃이나 수치 특성이 같지 않다.
[NVIDIA NVFP4 설명](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).

현재 ST의 `engine/kernels/b12x/moe_static_kernel.py`는 이 warp MMA를 선택하며
TMA pipeline도 이미 사용한다. `cluster_shape_mnk=(1,1,1)`은 이 구현의 선택이다.
하드웨어가 다중 CTA cluster를 지원하지 않아서 생긴 제약으로 해석하면 틀린다.

MLA megakernel에는 별도로 W4 저장값을 FP8로 확장해 `mma.sync ... e4m3`로 계산하는
W4A8 경로가 있다. **4-bit weight 저장과 native NVFP4 MMA 사용은 다른 선택**이다.
모델의 모든 레이어가 native FP4 연산으로 실행된다고 판단해서는 안 된다.

## 클러스터 / DSMEM 실행 확인

[cluster_probe.cu](cluster_probe.cu)는 CTA마다 dynamic shared memory 64 KiB를
예약해 GB10의 같은 SM에 두 CTA가 함께 들어갈 수 없게 했다. 각 CTA는 shared
정수를 쓰고 cluster barrier 후 다음 CTA의 shared 정수를 읽는다. `%smid`도 수집했다.
최대 8 CTA × 32 threads, 출력 64 bytes를 사용했다.

| cluster 크기 | 이웃 shared read | 서로 다른 SM 확인 | 관측한 SM ID |
|---|---|---|---|
| 1 | PASS | PASS | 0 |
| 2 | PASS | PASS | 0, 1 |
| 4 | PASS | PASS | 0, 1, 8, 9 |
| 8 | PASS | PASS | 0, 1, 8, 9, 16, 17, 24, 25 |

`cudaOccupancyMaxPotentialClusterSize` 결과는 8이었다. 이 값과 SM ID는 해당
장치·커널·설정의 결과다. 모든 커널의 최적 cluster 크기가 8이라는 뜻은 아니다.
64 KiB/CTA 설정에서 8-CTA cluster의 active cluster 수는 4였다. cluster packing이
점유율에 영향을 주므로 `48 SM / cluster size`만으로 계산하면 안 된다.

DSMEM은 **동일 GPU 내부 cluster 범위**다. 네 Spark 사이의 TP collective를
이 barrier 또는 shared load로 대체할 수 없다.

## TP=4와 메모리 설계에 주는 의미

각 Spark의 128 GB는 CPU·GPU가 공유하는 로컬 물리 메모리다. 네 노드를 묶으면
분산 용량이 늘지만 단일 주소 공간의 512 GB GPU가 되는 것은 아니다.

Spark의 ConnectX-7은 포트당 최대 200 Gb/s Ethernet이고, SoC와 NIC 사이에는
**독립 PCIe Gen5 x4 링크 2개**가 있다. 물리 포트 하나가 두 PCIe 경로와 두 HCA로
보이므로 포트 개수와 HCA 개수를 혼동하면 안 된다. NIC를 사용한 RoCE 통신은
노드 내 NVLink-C2C와 다른 경로다.
[NVIDIA ConnectX-7 topology](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).

이번 driver 조회의 `GPU_DIRECT_RDMA_SUPPORTED=0`만으로 RoCE 미지원이나
NCCL의 특정 복사 횟수를 단정하지 않는다. UMA에서 어떤 메모리를 등록하고 어떤
transport를 실제 사용하는지는 NCCL 로그와 측정으로 확인해야 한다.

아래는 구조에서 도출한 ST 최적화 판단이며, 성능 향상을 실측한 결과가 아니다.

1. **decode에서는 데이터 이동량과 launch/collective 횟수 감소를 우선한다.**
   작은 M에서는 높은 FP4 peak를 채우기 어렵다. 압축 weight 스트리밍, 상태의
   필요한 위치만 읽고 쓰기, fusion, CUDA Graphs가 적절한 방향이다.
2. **NVFP4는 scale 배치까지 함께 최적화한다.** A/B와 scale의 load 정렬,
   shared bank conflict, register pressure가 warp MMA 이용률에 영향을 준다.
3. **occupancy는 100 KiB SMEM·64K registers·48 warps를 기준으로 계산한다.**
   48 SM은 동시 CTA의 절대 상한이 아니다. software grid barrier는 해당 커널의
   전체 grid가 동시에 상주할 수 있는지 별도로 보장해야 한다.
4. **cluster/DSMEM 및 CLC는 새 후보가 된다.** GPU 내부 split-K reduction이나
   타일 공유·작업 분배에 검토할 수 있다. 추가 barrier와 cluster packing 손해를
   함께 측정해야 한다. 기존 TMA 경로에 단순히 cluster 크기만 늘려 적용하면 안 된다.
5. **TP=4는 실제 네트워크 경로를 기준으로 최적화한다.** layer collective의
   latency, tensor 크기, HCA 사용을 확인한다. B200 NVLink/NVSwitch 전제를 옮겨서는 안 된다.
6. **UMA 예산은 시스템 전체를 고려한다.** weight·KV·workspace·GPU context와
   CPU runtime·OS가 같은 DRAM 용량을 사용한다. 별도 GPU VRAM 풀처럼 계산하면 안 된다.

## 문서 차이와 코드 주석 정정 대상

- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)의
  12.0 항목에는 128 KB shared memory와 32 blocks/SM이 적혀 있다. 이 값을
  GB10에 그대로 적용하면 실제 100 KiB / 24 blocks와 어긋난다. 현재 compute
  capability 표와 장치 속성은 100 KiB / 24 blocks로 일치한다.
- 제품 표의 copy engine 2와 CUDA의 async-engine attribute 1은 서로 다르다.
  이번 조사에서는 둘의 차이를 설명할 내부 구현 자료를 확보하지 않았다.
  TMA engine 수나 copy concurrency를 이 숫자에서 추정하지 않는다.
- `engine/kernels/mla/glm53_megakernel.cu:27`의 헤더는 cluster/DSMEM 미지원,
  SMEM 128 KB, fixed 48-block grid, no PDL이라고 설명한다. 본문은 이미
  102,400-byte SMEM과 occupancy 계산을 사용하며 `griddepcontrol`도 포함한다.
  따라서 헤더에는 하드웨어 오류와 오래된 구현 설명이 함께 남아 있다.

## 재현과 증거

2026-09-12 후속 [NVFP4 instruction audit](../st_gb10_nvfp4_instructions_20260912/README.md)은
`nvdisasm` 13.0.85로 SASS까지 확인했다. 현 도구체인의 GB10에서는 3입력
`max.abs.f32`가 `FMNMX` 두 개로, `mul.rn.f32x2`가 scalar `FMUL` 두 개로
내려간다. 지원되는 PTX의 입력 폭이 실제 명령 한 개의 처리량을 뜻하지 않는다.
현재 ST MoE의 native NVFP4 MMA는 `OMMA`로 확인했다. 후속 변경은 amax를
독립 연산이 있는 트리로 바꾸며, 기계어 개수 감소나 FP4 peak 달성을 주장하지 않는다.

- [srv4-device.json](srv4-device.json): context·allocation·kernel 없이 driver 조회.
- [srv4-ptx.json](srv4-ptx.json): 25종 probe, 타깃 조합 47건.
  기대한 성공 43건 / 거부 4건. 거부 3종은 다른 지원 아키텍처에서 문법 통과를 확인했고,
  나머지 1건은 baseline `sm_121`에서의 NVFP4 거부다.
- [srv4-cluster.log](srv4-cluster.log): 네 cluster 크기의 실행·결과·SM ID 확인.
- `nvdisasm`이 이미지에 없어 SASS disassembly는 수행하지 않았다.
  PTX 통과를 SASS별 throughput 측정으로 표현하지 않는다.
- 프로브는 종료 시 임시 디렉터리를 지우는 disposable container에서 수행했다.
  기존 서비스를 재시작하지 않았다. 전체 모델이나 성능 부하를 실행하지 않았다.

동일 CUDA image 안에서, 이 폴더를 작업 디렉터리로 두고 재현한다.

```bash
python3 device_attributes.py
python3 ptx_compile_probe.py
nvcc -std=c++17 -arch=sm_121a cluster_probe.cu -o /tmp/st-sm121-cluster-probe
timeout 25 /tmp/st-sm121-cluster-probe
```

처음 두 Python script에는 `cuda-python` bindings와 CUDA compiler가 각각 필요하다.
컴파일 전용 script는 GPU 접근 없이 실행 가능하다. cluster 실행에는 GB10 GPU가 필요하다.
