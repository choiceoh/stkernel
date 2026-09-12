# one-shot 최초 NVCC 컴파일 — 2026-09-13

one-shot은 Tensor API와 pybind 바인딩을 사용한다. `torch/extension.h`가 읽는
전체 C++ 프런트엔드(`torch/all.h`)와 추가 Python 프런트엔드 대신 `torch/types.h`와
`torch/csrc/utils/pybind.h`를 직접 포함한다. 연산·통신 코드와 빌드 옵션은 그대로다.
내용이 같은 체크아웃의 재컴파일을 없앤 [#781](https://github.com/choiceoh/stkernel/pull/781)에
이어, 처음 컴파일하는 비용을 줄이는 변경이다.

## 비교 방법

같은 ST 이미지에서 A/B, B/A, A/B의 순서로 세 쌍을 실행한다. 매 표본은 별도의 새
빌드 디렉터리와 Python 프로세스를 사용한다. 양쪽 모두 실제 one-shot `build()`를
호출하며 소스 차이는 위 include 블록 하나다. 모든 표본에서 NVCC가 1회 실행되어야 한다.
시간은 Torch import 이후 소스 준비부터 컴파일·링크·로드 완료까지의 벽시계 시간이다.
디스크 페이지 캐시를 비우지는 않는다.

| 쌍 | 실행 순서 | 기존 헤더 | 축소 헤더 | 감소 |
|---|---|---:|---:|---:|
| 1 | 기존 → 축소 | 50.538 s | 32.268 s | 36.2% |
| 2 | 축소 → 기존 | 25.045 s | 22.901 s | 8.6% |
| 3 | 기존 → 축소 | 34.281 s | 28.313 s | 17.4% |
| 중앙값 | 각 3회 | **34.281 s** | **28.313 s** | **17.4%** |

3쌍 모두 축소 헤더가 빨랐다. 공유 호스트의 벽시계 변동이 있으므로 이 환경의 표본으로
한정한다. [result.json](result.json)과 `0-full.log`부터 `2-lean.log`까지 여섯 로그가
원본이며, 6회 모두 새 캐시에서 NVCC 1회, 공개 API 11개와 두 입력 검증을 통과했다.

## 생성 코드와 바인딩 검사

CUDA 함수를 실행하지 않고 `cuobjdump --extract-elf all`로 실제 `.so`의 cubin을 추출한다.
NVCC가 경로별로 만드는 내부 심벌의 파일 식별자 때문에 원본 cubin 해시는 달라진다.
`.strtab` 안의 `_INTERNAL_<8hex>_18_dsv4_oneshot_ar_cu_`와
`_GLOBAL__N__<8hex>_18_dsv4_oneshot_ar_cu_` 두 패턴의 8자리만 0으로 바꾼 뒤
**전체 cubin 바이트**가 일치해야 통과한다. 명령어·상수·심벌 오프셋·실행 메타데이터·
다른 이름은 마스킹하지 않는다. 포맷이 달라지면 비교를 중단한다.

추출 cubin은 3개 `.text.*` 섹션을 갖는다. 위 파일 식별자 29곳을 정규화한 SHA256은
`7617b744bc100df726341c6e9b7cbade36d0e7289ae69f6a4b8382c3a9a92f19`다.
정규화 검사는 명령어·상수·실행 메타데이터·공개 심벌을 바꾸면 해시가 달라지는지도
독립 CPU 테스트로 확인한다.

모듈의 공개 함수 11개가 일치해야 하고, BF16 합산과 int64 MAX 함수에 CPU Tensor를
넘기면 원래의 CUDA 입력 검증에서 거절되어야 한다. Tensor 타입 변환까지는 실제
바인딩을 통과한다. transport 초기화·CUDA stream 조회·GPU 커널 실행은 하지 않는다.
각 프로세스의 Torch CUDA 초기화 상태는 검사 전후 false여야 한다.

## 재현

- 소스: `20624d82c91c5fb7933f4b5361aaf1d230f65263`. probe가 두 include 변형을 만든다.
- srv2, aarch64, Torch `2.13.0+cu130`, CUDA `13.0`, `sm_121a`, `-O2`, `MAX_JOBS=2`.
- 이미지 `st-engine:main-ff728f43`, ID
  `sha256:a13be2698a579983329b82a1fc2a98d6462346a323981ba17e97f27c3b7fbd7b`.
- Docker CPU 2개·메모리 4 GiB 제한. GPU와 네트워크를 연결하지 않았다.
- 전용 원본 증거 디렉터리: `/home/choiceoh/.cache/st-native-cache-evidence/oneshot-headers-0913-v3`.

```sh
docker run --rm --network none --memory 4g --cpus 2 \
  -v "$PWD:/repo:ro" -v "$ST_EVIDENCE:/evidence" -w /repo \
  -e PYTHONPATH=/repo -e CUTE_DSL_ARCH=sm_121a \
  --entrypoint python3 st-engine:main-ff728f43 \
  probes/engine_native_compile_check.py --output /evidence/headers-new --repeats 3
```

기존 출력 디렉터리는 덮어쓰지 않는다. 각 실행의 JSON은 원본 cubin 해시, 정규화 해시,
공개 API·입력 검증, 빌드 횟수·시간과 소스 SHA256을 기록한다. 프로브는 현재 소스가
전체 헤더 또는 축소 헤더 중 어느 쪽이어도 같은 두 변형을 만들 수 있다.

v1은 이미지에 `nvdisasm`이 없어 어셈블리 덤프에서 중단했다. v2는 두 표본의 원본
cubin 해시 차이에서 중단했고, 조사 결과 46개 섹션 중 내부 파일 식별자가 있는
`.strtab`만 달랐다. 위 두 패턴만 정규화하도록 보완한 뒤 새 캐시로 v3을 실행했다.
불완전한 두 시도의 시간은 최종 비교에 합치지 않는다.

이 측정은 네이티브 확장 컴파일·로드에 한정된다. 전체 모델의 콜드 부팅·TTFT·tok/s,
GPU 통신·수치·그래프 실행을 판정하지 않는다. 헤더 블록의 소스 바이트가 달라지므로
새 버전을 처음 사용할 때는 새 키로 컴파일한다.
