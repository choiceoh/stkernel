# one-shot 연산별 헤더와 최초 빌드 — 2026-09-13

PR #784 이후의 `torch/types.h`도 전체 `ATen/ATen.h`, 연산 목록, 라이브러리 등록 헤더를
읽는다. one-shot에서 이 묶음을 제거하고 Tensor 본체, 기존 Tensor factory와 pybind
헤더를 직접 읽는다. `AT_PER_OPERATOR_HEADERS`로 factory가 전체 `ATen/Functions.h`
대신 필요한 연산 헤더를 사용하게 한다.

`torch::empty_like`는 그대로 호출한다. `torch::Tensor`·dtype 별칭은 그 정의인
`at::Tensor`, `at::kBFloat16`, `at::kLong`으로 직접 표기한다. Tensor 생성·autograd
처리와 통신·연산 구현을 바꾸지 않는다.

## 현재 main과 같은 실행에서 비교

baseline은 PR #784가 반영된 main `543f9f8841b10b9b06fcf37981c062e24cafad79`의
one-shot 소스와 바이트 단위로 일치한다. 이전 측정값과의 간접 비교가 아니다.

| 쌍 | 실행 순서 | 현재 main | 연산별 헤더 |
|---|---|---:|---:|
| 1 | main → 후보 | 32.302 s | 20.767 s |
| 2 | 후보 → main | 26.397 s | 16.960 s |
| 3 | main → 후보 | 27.370 s | 18.119 s |
| 중앙값 | 각 3회 | **27.370 s** | **18.119 s** |

최초 빌드·링크·로드 중앙값이 **33.8% 감소**했고 세 쌍 모두 빨랐다. 각 표본은 새
Python 프로세스와 빈 빌드 캐시를 사용하며, 6회 모두 NVCC를 정확히 1회 실행했다.
Torch import 이후 소스 준비부터 확장 로드까지를 측정한다. 디스크 페이지 캐시는
비우지 않으며 공유 호스트의 벽시계 시간이다.

| 보조 측정 중앙값 | 현재 main | 연산별 헤더 |
|---|---:|---:|
| 컴파일러 자식 프로세스 CPU 시간(user + system) | 27.349 s | 18.111 s |
| 컴파일러 자식 프로세스 최대 RSS | 1,792,564 KiB | 1,437,488 KiB |

CPU 시간도 33.8% 줄었다. 자식 프로세스 최대 RSS는 약 **1.71 → 1.37 GiB, 19.8% 감소**다.
이는 `getrusage(RUSAGE_CHILDREN)`의 Linux 값으로, 컨테이너 전체의 합산 메모리는 아니다.

## 코드·바인딩 검증

6회 모두 실제 `build()`로 컴파일·로드했다. 공개 API 11개가 같고 BF16 합산과 int64
MAX의 실제 Tensor 타입 변환 및 CUDA 입력 거절 검사를 통과했다. GPU나 transport는
초기화하지 않았고, Torch CUDA 초기화 상태는 실행 전후 false였다.

추출한 GPU cubin은 기존 검사와 동일하게 `.strtab`의 NVCC 내부 파일 식별자 29곳만
정규화한 후 **전체 바이트가 일치**했다. 3개 GPU 코드 섹션·상수·실행 메타데이터·
심벌 오프셋은 마스킹하지 않는다. 모든 실행의 정규화 SHA256은
`7617b744bc100df726341c6e9b7cbade36d0e7289ae69f6a4b8382c3a9a92f19`다.
정규화 규칙과 변경 감지 검사는 [이전 측정](../st_native_compile_20260913/README.md)에 있다.

## 재현과 증거

- 소스 커밋: `b6b1cd800759b1ff2211efbde9401638e9139dad`.
  probe가 main과 연산별 헤더 변형을 만들며, 후보 해시는 최종 CUDA 파일과 일치한다.
- srv2, aarch64, Torch `2.13.0+cu130`, CUDA `13.0`, `sm_121a`, `-O2`, `MAX_JOBS=2`.
- 이미지 `st-engine:main-ff728f43`, ID
  `sha256:a13be2698a579983329b82a1fc2a98d6462346a323981ba17e97f27c3b7fbd7b`.
- Docker CPU 2개·메모리 4 GiB 제한, GPU와 네트워크 연결 없음.
- [result.json](result.json)에 소스·후보·cubin 해시, CPU/벽시계 시간, RSS,
  컴파일 횟수와 바인딩 검사가 있다. `0-lean.log`부터 `2-operators.log`까지 원본 로그 6개를 보관한다.
- 원격 원본: `/home/choiceoh/.cache/st-native-cache-evidence/oneshot-operators-0913-v1`.

```sh
docker run --rm --network none --memory 4g --cpus 2 \
  -v "$PWD:/repo:ro" -v "$ST_EVIDENCE:/evidence" -w /repo \
  -e PYTHONPATH=/repo -e CUTE_DSL_ARCH=sm_121a \
  --entrypoint python3 st-engine:main-ff728f43 \
  probes/engine_native_compile_check.py --comparison lean-operators \
  --output /evidence/operators-new --repeats 3
```

기존 출력은 덮어쓰지 않는다. 기존 `--comparison full-lean`도 새 소스에서 원래의 두
변형을 복원한다. 이전 날짜의 시간과 합쳐 누적 개선율을 계산하지 않는다.
이 수치는 one-shot 네이티브 빌드·로드에 한정된다. 전체 모델 부팅·TTFT·tok/s와
GPU 통신·수치·그래프 실행은 측정하지 않았다. 새 헤더는 새 캐시 키로 한 번 빌드된다.

## 회귀 검사

최종 소스 `f36b07d449a8d38b70f3508a7c317684369700c6`에서 같은 ST 이미지의 CPU 검사
31개가 통과했다([로그](cpu-tests.log)). macOS에서는 헤더 변형 복원·cubin 비교·캐시·
one-shot 정수 계약 검사 13개가 통과했다.

```sh
python3 -m unittest tests.test_engine_native_compile tests.test_engine_native_cache \
  tests.test_engine_oneshot_integer tests.test_engine_mla_hardware \
  tests.test_engine_kernels tests.test_engine_drafter_storage -q
```

프로브의 기본 `full-lean` 비교가 새 소스에서도 원래의 두 소스를 복원하고,
`lean-operators`가 Tensor factory 호출을 유지하는지도 검사한다.
