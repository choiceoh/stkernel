# 네이티브 확장을 첫 집합통신 전에 모든 랭크에서 한꺼번에 빌드한다 (2026-09-15)

## 무엇인가 — main 의 첫 콜드 부팅이 컴파일을 기다리다 죽었다

main `3acae017` 을 프로덕션에 처음 올린 부팅(2026-09-15 03:25 UTC)은 문을 열기 전에 네 랭크가 모두 죽었다.

- **캐시가 비어 있었다.** 이 배포는 네이티브 확장 네 개를 `/cache/cu132/st-native` 에서 새로 빌드했다(#984 가 경로를 옮겼다).
- **빌드는 쓰는 자리에서 일어났다.**
  - 각 확장은 처음 쓰이는 단계에서 컴파일됐다.
    - prefill top-k: 32K 프리필 패스 안
    - bounded graph: 타깃 캡처 안
    - mapped staging 과 decode queue: 버스트 파이프라인 안
  - 랭크는 저마다의 속도로 컴파일한다. 먼저 끝난 랭크는 집합통신 안에서 느린 랭크를 기다린다.
- **노드별 빌드 시각.** 각 노드 `/cache/cu132/st-native/<이름>/<키>/` 에서 `.sources.lock` 생성 시각과 `.so` 기록 시각을 읽었다([native-times-srv*.txt](native-times-srv2.txt)). 시각은 UTC 이고, 칸은 `끝 시각 (걸린 초)` 이다.

| 확장 | srv2 (r0) | srv1 (r1) | srv3 (r2) | srv4 (r3) | 끝 시각 차 |
|---|---|---|---|---|---:|
| prefill-topk (시작 03:26:35.0) | 03:27:15.1 (40.1) | 03:27:32.7 (57.7) | 03:27:21.3 (46.3) | 03:27:34.4 (59.4) | 19.3 s |
| bounded-graph (시작 03:28:44.6) | 03:29:22.4 (37.7) | 03:29:23.0 (38.3) | 03:29:31.3 (46.6) | 03:29:13.7 (29.0) | 17.6 s |
| mapped-staging (시작 03:29:58.9) | 03:30:48.4 (49.5) | 03:30:25.6 (26.7) | 03:30:42.6 (43.7) | 03:30:48.4 (49.5) | 22.8 s |
| decode-queue (mapped-staging 직후) | 03:31:35.4 (47.0) | **03:31:01.5** (35.9) | 03:31:28.4 (45.8) | 03:31:28.5 (40.1) | **33.9 s** |
| 네 빌드 합 | 173.9 s | 158.6 s | 182.4 s | 178.0 s | |

- **로그(`docker logs -t`, [cold-boot-rank-excerpts.txt](cold-boot-rank-excerpts.txt)).**
  - bounded graph 가 가장 먼저 끝난 rank 3 은 03:29:16.9 부터 `[oneshot] STALL phase=wait(peer-flags)` 를 찍었다.
  - decode queue 를 가장 먼저 끝낸 rank 1 은 03:31:04.7 부터 한 one-shot 합(seq 2796)에서 세 피어를 기다렸다. 03:31:32.8 에 `Triton Error [CUDA]: unspecified launch failure` 로 죽었다. 약 31 s 를 기다린 뒤였다.
  - 나머지 세 랭크는 03:31:38–39 에 rank 1 을 기다리다 `WC error 12 on rail N; proxy exiting`(RoCE 재전송 소진)을 냈고, 03:32:06 에 같은 오류로 죽었다.
  - 같은 커밋의 다음 부팅(03:38, 캐시가 채워진 뒤)은 문을 열었다.

## 바꾼 것

- **`engine/profiles/glm53/natives.py`**
  - fleet 부팅이 싣는 네이티브 확장 일곱 개(dense, mla, prefill-topk, mapped-staging, bounded-graph, decode-queue, one-shot)의 빌드 진입점을 나열한다.
  - `NativeBuilds` 는 확장마다 스레드 하나를 두고 한꺼번에 빌드한다. `wait()` 는 확장별 초를 돌려주고, 실패한 확장이 있으면 이름과 오류를 모두 담아 올린다.
  - 모듈 import 는 호출 스레드에서 먼저 한다. 서로를 import 하는 패키지를 여러 스레드가 동시에 처음 import 하면 import 시스템이 교착할 수 있다.
  - one-shot 은 서빙 설정의 rails/inline 값으로 빌드한다. 나머지는 모듈의 기존 빌드 함수 그대로다.
- **`boot.fleet`**
  - `declared()` 와 커널 형상 바인딩 직후, `Comm.init()` 전에 빌드를 시작한다.
  - `Comm.init()` 뒤 새 단계 `native builds` 에서 기다린다. 이어 `comm.wait_prepared("native-builds")`(준비용 Gloo 그룹, 1800 s)에서 랭크가 만난 뒤에야 one-shot 전송의 첫 합으로 간다.
  - 한 랭크의 빌드가 실패하면 `failed: rank N: ...` 단계로 랑데부에 가서 피어들을 즉시 멈춘다. 1800 s 를 기다리게 두지 않는다.
  - 확장별 초는 그 단계의 게이지(`native_<이름>_s`)와 랭크당 한 줄(`rankN: native builds in X s (...)`)로 남는다.
- **`engine/kernels/dense`.** `extension()` 을 `build()`(컴파일·로드, 장치 접근 없음)와 `extension()`(장치 확인)으로 나눴다. 빌드 키·플래그·경로는 그대로다.
- **바뀌지 않는 것.**
  - 커널 소스, 빌드 키, 캐시 경로, 수치, 노브는 그대로다.
  - 각 레인은 첫 사용 때 장치 확인·자기 검사를 여전히 한다: `dense.extension` 의 probe, `mla.maybe_arm` 의 selftest, one-shot 의 연결·자기 검사.
  - `--local`/`--test` 부팅은 바꾸지 않았다.

## CPU 측정

### 키가 이미 있을 때 (보통의 재부팅)

- **방법.** srv4 의 `/cache/cu132/{st-dense,mla,st-oneshot,st-native}` 를 스크래치로 복사했다. 이를 프로덕션 이미지(`st-engine:glm53`, `sha256:848e493f…`) 안의 같은 경로 `/cache/cu132` 에 붙이고, CUDA 를 숨긴 채 돌렸다.
- **해당성.** main `3acae017` 과 이 브랜치 사이에 커널 소스 변경이 없으므로 키가 같다. 원시 출력은 [native_builds_kept.log](native_builds_kept.log) 에 있다.

| 순서 | 방식 | 벽시계 | 확장별 |
|---:|---|---:|---|
| 1 | 하나씩(첫 사용 순서) | 0.273 s | 0.029–0.051 s |
| 2 | `NativeBuilds` | 0.091 s | 0.040–0.089 s |
| 3 | `NativeBuilds` | 0.096 s | 0.044–0.094 s |
| 4 | 하나씩 | 0.278 s | 0.029–0.051 s |

- 일곱 개 모두 컴파일 없이 로드됐다.
- 오늘은 이 로드들이 여러 단계에 흩어져 하나씩 일어난다. 앞당겨 한꺼번에 해도 재부팅은 느려지지 않는다.
- 모듈 import 1.0–1.7 s 는 부팅이 lanes 단계에서 어차피 치르는 비용이다.

### 키가 없을 때 (새 커널 소스·툴킷의 첫 부팅)

- **방법.** 빈 빌드 루트(스크래치)에서 CUDA 를 숨긴 채 이 브랜치의 빌드 진입점을 그대로 돌렸다. 이미지는 `st-engine:glm53`, 컨테이너 메모리 상한은 16 GiB 다.
  - 순서: 한꺼번에(새 루트) → 같은 루트로 다시(키 있음) → 하나씩(새 루트, 첫 사용 순서).
  - 원시 출력은 [native_builds_cold.log](native_builds_cold.log) 에 있다.

| 확장 | 하나씩 (s) | 한꺼번에 (s) |
|---|---:|---:|
| one-shot | 25.98 | 32.72 |
| dense | 77.48 | 65.71 |
| prefill-topk | 36.99 | 50.68 |
| mla | 69.57 | 48.34 |
| bounded-graph | 47.93 | 33.44 |
| mapped-staging | 58.82 | 39.66 |
| decode-queue | 56.11 | 50.45 |
| **벽시계** | **372.88** | **65.71** |
| cgroup 메모리 최고 | 3.37 GiB | 15.57 GiB |
| 가장 큰 자식 프로세스 RSS | 2.98 GiB | 2.98 GiB |

- 한꺼번에 빌드하면 가장 긴 dense(65.7 s) 하나로 끝난다. 하나씩은 372.9 s 다.
- **배경 부하가 달랐다.**
  - 한꺼번에 샘플(04:24 UTC)은 플릿이 재시작하던 중이라 MemAvailable 이 100 GiB 였다.
  - 하나씩 샘플(04:25–04:31)은 같은 노드에서 큐 부팅이 올라와 서빙하던 중이었다.
  - 그래서 확장별 초는 두 표본 사이에서 서로 비교하지 않는다.
- **메모리.** cgroup 최고치 15.57 GiB 는 16 GiB 상한 아래에서 잰 값이고 페이지 캐시를 포함한다. 부팅에서 이 빌드는 arena admission(60 GiB 할당) 한참 전에 끝난다(랑데부가 one-shot 준비 앞이다). 그 시점 노드마다 90 GiB 이상이 비어 있다(03:25 부팅의 `box:` 줄: 90–107 GiB free).
- **부팅에 옮기면(추정).**
  - 03:25 콜드 부팅은 네 확장을 노드마다 158.6–182.4 s 동안 하나씩 빌드했다.
  - 한꺼번에라면 가장 긴 하나(이 측정에서 50 s 안팎)로 줄고, 그 시간은 부팅 첫머리 comm 초기화와 겹친다.
  - 새 dense `kernels.cu`(대부분의 성능 PR)만 바뀐 부팅은 dense 한 번(65–77 s)이 그대로 남는다. 하지만 그 시간이 랭크들의 집합통신 안이 아니라 랑데부 앞에 온다.

## CPU 테스트

```bash
python3 -m unittest tests.test_engine_glm53_natives tests.test_engine_native_cache tests.test_engine_bootpaths \
  tests.test_engine_kernels tests.test_engine_kernel_common tests.test_engine_draft_acceptance \
  tests.test_engine_dense_store_digest tests.test_engine_dense
```

- **결과.** 같은 이미지(`st-engine:bracket-9c45086a0622`), CUDA 숨김, main `937666be` 위로 리베이스한 트리에서 66 테스트 OK(7 스킵, CUDA 전용). 로그는 [cpu-tests.log](cpu-tests.log) 에 있다.
- **새 `tests/test_engine_glm53_natives.py`**
  - `engine/kernels` 아래 `cpp_extension.load` 로 빌드하는 모든 모듈이 목록에 있는지, 그 진입점 함수가 있는지 확인한다. 새 확장이 목록 없이 들어오면 실패한다.
  - dense `build()` 에 장치 확인이 없고, `extension()` 이 `build()` 뒤에 확인하는지 본다.
  - 빌드 넷이 겹쳐 도는지(0.4 s × 4 가 1.2 s 안), 순서대로 보고하는지, 호출 스레드가 아닌 스레드에서 도는지 본다.
  - 한 빌드가 실패해도 나머지가 끝나고, 실패가 이름과 함께 올라오는지 본다.
  - 모듈 import 가 호출 스레드에서 일어나는지, one-shot 이 서빙 rails/inline 으로 빌드되는지 본다.
- **`tests/test_engine_bootpaths.py`.** fleet 모드에서 `native-builds` 랑데부가 one-shot 준비보다 먼저 오는지, local 모드는 목록을 부르지 않는지 본다.
- **`tests/test_engine_native_cache.py`.** dense 빌더 이름을 `build` 로 바꿨다.

## 재지 않은 것

- **GPU 부팅.** 다음 부팅에서 볼 것:
  - 각 랭크 로그의 `rankN: native builds in X s (...)` 줄과 `boot rendezvous native-builds` 줄
  - rank 0 표의 `native builds` 행
  - 키가 있는 재부팅이면 확장별 수십 ms 여야 한다.
- **콜드 부팅.** 새 커널 소스가 들어간 첫 부팅에서 빌드가 랑데부 앞에서 끝나는지, 그 뒤 단계에 `STALL` 이 줄었는지 볼 것.
- **남은 같은 종류의 위험.** 이 PR 이 다루지 않는다.
  - CuTe-DSL MoE 커널과 Triton 커널은 여전히 첫 사용 때 랭크마다 컴파일한다(한 번에 3–9 s).
  - 03:51 의 같은 커밋 부팅은 문을 연 뒤 요청마다 `static2_m{22,51,45,48,39,41,42,69,49}` 를 3–6 s 씩 컴파일했다. 그동안 rank 1·2 가 `STALL` 을 찍었다.
