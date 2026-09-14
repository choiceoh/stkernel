# ST CUDA 13.2 전체 런타임 이전 — 2026-09-14

ST의 기본 이미지, Torch CUDA ABI, 컴파일러, 런타임 라이브러리, JIT 선택과 캐시를
CUDA **13.2.1**로 전환했다. 기존 호스트 드라이버/툴킷과 실행 중인 서비스는 변경하지 않았다.
운영자의 “큐 태우지마”에 따라 큐 등록·GPU 부팅·onepass는 하지 않았다.

## 고정한 스택

| 구성 | 적용 버전 |
|---|---|
| CUDA toolkit | 13.2.1 |
| NVCC / PTXAS / NVRTC / NVVM / nvJitLink | 13.2.78 |
| cudart | 13.2.75, 실제 로드 버전 13020 |
| PyTorch / torchvision / torchcodec | 2.13.0+cu132 / 0.28.0+cu132 / 0.16.0+cu132 |
| CUDA Python / bindings | 13.2.0 |
| cuBLAS | 13.4.0.1, CUDA 13.2.1 구성품 버전 |
| Triton / CuTe DSL / TileLang | 3.7.1 / 4.6.2 / 0.1.12 |
| nvdisasm | 13.3.73 — CuTe가 요구하는 진단 도구 |

Torch의 source git은 이전 cu130과 같은 `cf30153c4c131c8164ee7798e5022d810682e2cb`이다.
CuTe/FlashInfer와 DeepGEMM의 확장 및 JIT 파일 879개를 보존했다. nvdisasm은 컴파일·실행 도구가 아니다.
34개 wheel의 URL·SHA256·버전은 `engine/runtime/cuda132.lock.json`에 고정했다.

- 원래 bootstrap: `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`
- 네 노드 공통 runtime seed: `sha256:a9b53fd066bb4fa0c4d12982f7c5dcdd5a2591900a0088c3ffeb41ba0868425c`
- 최종 엔진 소스: `95472dc4ba420ff43114794ba5542ac8abf6f8afda24460d47b1c849d53129fe`
- 소스 기준: `22be9f92`, main `b8bef130` 포함. 개별 engine image ID는 [fleet-images.json](fleet-images.json)에 있다.

네 노드에서 독립 빌드했으므로 engine image ID는 다르다. seed ID와 전체 engine source SHA256은 같고,
각 노드에서 runtime verifier를 실행한 뒤 `st-engine:glm53` 기본 태그를 전환했다. 서비스는 재시작하지 않았다.

## 실물 검증

- [runtime.json](runtime.json): 실제 로드된 cudart/NVRTC/라이브러리 경로, 모든 고정 패키지,
  NVCC/PTXAS, Triton의 일반·Blackwell assembler, TileLang/DeepGEMM CUDA_HOME, DeepGEMM 및 엔진 해시 통과.
- [native-extensions.json](native-extensions.json): 실제 production builder 일곱 종류
  (MLA, dense, oneshot, mapped_staging, decode_queue, bounded_graph, prefill_topk)의 compile + dlopen 통과.
  Dense의 최종 하드웨어 조회만 선언된 GB10 값으로 대체했다. 이 결과는 GPU 자격 검사가 아니다.
- [sf6-native.json](sf6-native.json): SF6 전체 CuTe 커널 일곱 변형 통과.
  [sf6-operands.json](sf6-operands.json): helper 4종과 CPU operand 24,576개 통과.
- [triton-kda.json](triton-kda.json): KDA 55개 specialization 컴파일 통과.
  [tilelang-mhc.json](tilelang-mhc.json): 실제 TileLang MHC lowering과 NVCC cubin 생성 통과.
- [cpu-tests.txt](cpu-tests.txt): 최신 main을 포함한 Linux 검사 **203 files / 1,872 tests,
  203 ok / 0 failed / 0 cannot run / 314 skipped**. GPU가 필요한 검사는 건너뛰었다.
- [MLA 정적 명령 비교](mla-instructions.json): 같은 13.2.78로 기준·후보를 컴파일했다.
  자세한 SF6/MLA 변화와 계측용 변형의 레지스터 증가도 [이식 기록](../st_upstream_microopts_20260914/README.md)에 남겼다.

CPU 컨테이너에는 `--runtime=runc`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`를 사용했다.
테스트/컴파일에 2 CPU, 메모리 4~5 GiB 한도를 두었다. GPU를 점유 중인 재양자화 세션과 임대는 건드리지 않았다.
처음 소스 복사에서 빠졌던 bench/측정 자료, PID 1의 자식 회수 조건을 보완한 뒤 전체 CPU 검사를 통과했다.

## 발견한 제약

`pip check`는 여전히 두 가지를 보고한다([원문](pip-check.txt)). FlashInfer 개발 빌드의 메타데이터가
CuTe 4.7.0을 요구하지만 실제 패치와 ST 커널은 4.6.2를 사용한다. NVIDIA 공식 ARM64 cuSPARSELt wheel은
내부 WHEEL에 `manylinux2014_sbsa`를 적어 pip가 인식하지 못한다. 설치된 `.so`는 ELF e_machine=183(AArch64)다.
실패를 숨기기 위해 외부 패키지 메타데이터를 바꾸지 않았다.

#947의 **실험용 트리 양자화 경로**는 기존 통과 기록이 Torch 2.14 CPU / Triton 3.8이었다.
현재 고정한 Triton 3.7.1에서는 NVFP4 3종이 MLIR에서 실패하고 W4A8 4종은 요구한 FP8 MMA 검사를 통과하지 못한다.
트리 KDA/BF16 10종은 컴파일된다. [전체 검사](tree-compile.json)에서 실패를 그대로 보존했다.
이전 cu130 이미지에서도 17종의 판정과 7종의 실패 이유가 모두 같았다
([전체 대조](tree-compile-cu130.json), [NVFP4 오류 원문 요약](tree-existing-failure.json)).
Triton 3.7.1의 [scaled-MMA 변환](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)은
SM120에 한정되어 있다. 이 실험 경로를 현재 ST 런타임에서 검증된 경로로 계산하지 않는다.
기본 ST CuTe NVFP4 및 CUDA W4A8 커널은 위의 별도 컴파일 증거를 따른다.

현재 R580 계열 호스트 드라이버에서 새 커널의 GPU 실행은 측정하지 않았다.
CUDA 13.x의 [minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)는
native cubin을 대상으로 하며 최신 PTX JIT에는 해당 PTX를 이해하는 드라이버가 필요하다.
GPU 수치·그래프 재생·통신·step/s·tok/s·수용률의 신규 결과는 **없다**.

## 재현

`engine/runtime/build-seed.sh`로 wheel 검증/오프라인 seed 빌드 후 같은 ID를 배포하고,
각 노드에서 `engine/runtime/build.sh`와 `engine.runtime.verify`를 실행한다.
새롭게 빌드한 seed는 manifest의 승인된 ID와 대조해야 한다. 임의 태그를 허용하지 않는다.
원시 검증 자료는 srv2 `/tmp/st-microopts-0914.G0PTLU/out`에 보존했다.

네이티브 캐시 key는 NVCC/PTXAS 경로와 실제 버전을 포함하고, CuTe key는 CUDA lock을 포함한다.
launcher/image 기본 캐시는 `/cache/cu132/`로 격리했다. 이전 SDK 경로나 assembler를 주입하면 부팅 전에 거부한다.
