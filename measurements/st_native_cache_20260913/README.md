# ST 네이티브 빌드 캐시 — 2026-09-13

내용이 같은 소스를 touch하거나 다른 체크아웃으로 옮길 때의 불필요한 NVCC 재컴파일을
없앤다. 기존 dense·MLA·one-shot도 내용으로 빌드 디렉터리를 골랐지만, Ninja에는 원본
체크아웃의 절대 경로를 넘겼다. 그래서 경로나 수정 시각만 달라져도 같은 디렉터리에서
다시 컴파일했다. 이제 내용별 디렉터리의 `src/`에 소스와 로컬 헤더를 한 번 기록한다.

실제 코드·헤더·옵션·Torch/CUDA 버전 변경은 별도 빌드를 만들고, Torch/Ninja의
빌드·의존성 검사·잠금·로드를 그대로 사용한다. 바이너리를 직접 import하는 우회는 없다.
파일 준비에도 프로세스 잠금을 사용해 같은 입력을 동시에 기록하면서 수정 시각이
바뀌는 것을 막는다. 키와 저장 파일은 동일한 한 번의 읽기에서 만든다.

## 작은 CUDA 확장으로 재컴파일 조건 확인

동일 이미지에서 각 칸을 새 Python 프로세스로 실행했다. 각 행은 순차 단일 표본이며
행마다 A/B 순서를 바꿨다. 시간은 Torch import 이후 소스 준비부터 확장 로드 완료까지다.
컴파일 수는 `.ninja_log`의 CUDA object 빌드 기록 증가분이다.

| 조건 | 기존 로드 시간 | 변경 후 로드 시간 | 기존 → 변경 후 NVCC 횟수 |
|---|---:|---:|---:|
| 소스 touch | 7.343 s | 5.930 ms | 1 → 0 |
| 같은 내용의 다른 체크아웃 | 5.765 s | 7.259 ms | 1 → 0 |
| 헤더 touch | 4.666 s | 6.736 ms | 1 → 0 |
| 변경 없는 재실행 | 6.760 ms | 6.498 ms | 0 → 0 |
| 헤더 내용 변경 | 7.135 s | 8.497 s | 1 → 1 |
| 소스 내용 변경 | 7.332 s | 7.571 s | 1 → 1 |
| 컴파일 옵션 변경 | 4.668 s | 7.178 s | 1 → 1 |

최초 빌드도 양쪽 모두 1회 실행했다. 16개 실행 모두 host 함수 반환값이 예상한
41 → 43 → 45 → 48과 일치했다. 재사용해야 할 조건과 무효화해야 할 조건을 함께
확인한 결과다. 원본 결과와 컴파일 출력은 [mini/result.json](mini/result.json), `mini/*.log`에 있다.

## 실제 one-shot 모듈

`--fixture oneshot`은 ST의 실제 `.cu`·헤더·`build()`를 사용한다. baseline은 변경 전
빌드 함수를 같은 입력·옵션·`-libverbs` 링크로 재현한다. candidate의 loader 래퍼는
빌드 디렉터리와 NVCC 횟수를 기록하고 로그만 켠다. 통신 초기화나 CUDA 함수 실행은 없다.

| 조건 | 기존 로드 시간 | 변경 후 로드 시간 | 기존 → 변경 후 NVCC 횟수 |
|---|---:|---:|---:|
| 최초 빌드 | 33.490 s | 40.964 s | 1 → 1 |
| 변경 없는 재실행 | 23.939 ms | 18.443 ms | 0 → 0 |
| 같은 내용의 다른 체크아웃 | 42.438 s | 17.774 ms | 1 → 0 |

6개 실행 모두 공개 함수 11개가 일치했다. 재배치 후 candidate 로그에는
`ninja: no work to do.`가 있고 baseline에는 NVCC와 링크 명령이 있다. 원본은
[oneshot/result.json](oneshot/result.json)과 `oneshot/*.log`에 있다.
콜드 빌드 시간은 위 단일 표본에서 candidate가 더 길었으며, 콜드 컴파일 개선을
주장하지 않는다. 이 변경이 확인한 효과는 재배치·수정 시각 변경에 따른 재컴파일 제거다.

## 재현과 범위

- 호스트: srv2, aarch64. 이미지: `st-engine:main-ff728f43`.
- 이미지 ID: `sha256:a13be2698a579983329b82a1fc2a98d6462346a323981ba17e97f27c3b7fbd7b`.
- Torch `2.13.0+cu130`, CUDA `13.0`, `sm_121a`, `-O2`, `MAX_JOBS=2`.
- Docker는 `--network none --memory 4g --cpus 2`, GPU 연결 없이 실행했다.
- mini 증거의 소스 커밋: `c92e357b555c71dd26b80fa249c68904f235cc1c`.
  그 이후 probe에 실제 one-shot 검사를 추가하고 helper의 docstring 들여쓰기를 정리했다.
- one-shot 증거의 소스 커밋: `099bff50c76544bd88b5df1fb084e95a3f5d0745`.
- JSON의 `source_sha256`이 각 실행의 정확한 소스 바이트를 기록한다.
- 각 프로세스의 Torch CUDA 초기화 상태는 실행 전후 모두 false다.

리포지터리 체크아웃과 빈 증거 디렉터리를 각각 `/repo:ro`, `/evidence`로 연결해 실행한다.
출력 디렉터리가 이미 있으면 probe가 거절하므로 새 이름을 사용한다.

```sh
docker run --rm --network none --memory 4g --cpus 2 \
  -v "$PWD:/repo:ro" -v "$ST_EVIDENCE:/evidence" -w /repo \
  -e PYTHONPATH=/repo -e CUTE_DSL_ARCH=sm_121a \
  --entrypoint python3 st-engine:main-ff728f43 \
  probes/engine_native_cache_check.py --fixture mini --output /evidence/mini-new
# 같은 명령에서 --fixture oneshot --output /evidence/oneshot-new
```

이 증거는 네이티브 확장의 컴파일·로드 비용에 한정된다. 최초 콜드 컴파일 자체의
단축, 전체 모델 부팅 시간, GPU 수치·통신·그래프 실행, tok/s·TTFT는 판정하지 않는다.
코드 생성 옵션과 CUDA 소스 바이트는 변경하지 않았다. 캐시 경로와 키가 바뀌므로
기존 캐시에서 이 방식으로 전환하는 첫 실행에는 한 번의 새 빌드가 필요하다.

런타임 Dockerfile에도 dense와 one-shot의 `/cache` 경로를 추가했다. 프로덕션
`start-st-glm53.sh`는 이미 이 경로를 지정하고 있었고, 이번 추가는 이미지 기본값으로
실행하는 컨테이너·프로브에도 같은 영속 캐시를 적용한다. `/cache` 볼륨 연결은 필요하다.

## 회귀 검사

최신 main 병합 후 `82a2312e`에서 같은 ST 이미지로 다음 CPU 검사 21개가 모두 통과했다.
새 캐시 검사 8개에는 프로세스 동시 접근, 불완전 입력 복구, 소스 스냅샷 일관성,
실제 세 빌더가 전달하는 입력과 링크 옵션, 무효화 조건이 포함된다.

```sh
python3 -m unittest tests.test_engine_native_cache tests.test_engine_mla_hardware \
  tests.test_engine_kernels -v
```

macOS에서는 전체 묶음 중 기존 Linux `O_DIRECT` 검사와 Torch 임포트 검사를 실행할 수
없어 Linux 이미지에서 위 묶음을 검증했다. 두 probe의 모든 소스 해시는 기록된 커밋과,
22개 실행의 JSON 행은 저장된 컴파일 로그와 대조했다. one-shot 증거의 소스는 PR #781의
최종 브랜치에서도 같은 바이트다.
