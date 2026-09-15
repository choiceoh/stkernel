# 문 열기 전 워밍업 — 프리필 길이 여섯 개를 뺀다 (2026-09-15)

## 무엇인가

부팅 원장의 `production/ready` 행은 `warmup shapes` + `qualify vision` + `qualify grammar` + 마지막 투표다. main `3acae017` 프로덕션 두 번째 부팅에서 19.2 s 였다.

- **어떻게 이루어져 있나.** `warmup shapes` 는 두 가지를 돈다([boot-lines.txt](boot-lines.txt)).
  - 프리필 여섯 번: 64·256·1,024·2,048·2,304·4,096 토큰.
  - 디코드 폭 1·2 의 동기 한 번과 비동기 한 번.

| 부팅(rank 0) | `warmup shapes` | 프리필 여섯 번 합 | 디코드 네 번 합 |
|---|---:|---:|---:|
| 프로덕션 03:38 UTC | 5.249 s | 4.656 s | 0.590 s |
| 큐 hold 03:51 UTC | 5.148 s | 4.558 s | 0.587 s |

- **프리필 여섯 번이 새로 도는 레인은 없다.** 서빙 레인이 갈리는 문턱은 모두 앞의 두 워밍업이 이미 지난다.
  - 앞의 두 워밍업: 메모리 워밍업(128·1,024·32,256 토큰, #987 재사용 부팅도 앞의 셋은 돈다)과 커널 워밍업(1·8·64·512·4,095 토큰).
  - 32행 이하 dense W4 / 그 위 FP8: `kernels/dense.DenseLinear.__call__`
  - 64행 mHC: `net._hc_post_pre`
  - 128 토큰 토큰 샤딩: `net.forward`, `N >= 128`
  - 640 routed pair 정적 컴팩트 전환: `b12x/moe_dispatch`
  - 8,192행 초과 프리필 라우터: `net.route`
- **디스크에 쓴 JIT 산출물도 없다.** 커널 워밍업이 끝난 뒤부터 문이 열릴 때까지 노드의 `/cache` 전체에 새로 쓰인 파일은 없었다([cache_writes.sh](cache_writes.sh)).
  - 이 구간은 타깃·드래프터 캡처, `warmup shapes`, 비전, 문법을 포함한다.
  - srv2(rank 0)와 srv4(rank 3)에서 두 부팅 모두 0 개였다([cache-writes-srv2.txt](cache-writes-srv2.txt), [cache-writes-srv4.txt](cache-writes-srv4.txt)).
  - 즉 이 두 부팅에서 여섯 번의 프리필은 이미 컴파일된 커널을 다시 돈 것이다.

## 바꾼 것

- `boot.fleet` 이 `engine.warmup_shapes(lengths=())` 를 부른다. 디코드 폭은 그대로 돈다.
  - 디코드 경로의 호스트 쪽(고정 id 스테이징, 버스트 큐, readback)은 문 앞에서 여기서만 처음 돈다.
- `warmup_shapes` 함수, 테스트, 다른 부팅 방식은 바꾸지 않았다. 테스트는 이미 `lengths=()` 로 부른다.
- 새 노브는 없다. 서빙 수치는 바뀌지 않는다: 워밍업은 판정하지 않고, 요청 행을 비워 돌려준다.

## 검증

- **CPU.** 같은 이미지(`st-engine:bracket-9c45086a0622`), CUDA 숨김. 로그는 [cpu-tests.log](cpu-tests.log) 에 있다.
  - `tests.test_engine_warmup_admission`, `tests.test_engine_warmup_draws`, `tests.test_engine_bootpaths`.
  - 앞의 둘이 `lengths=()` 경로를 이미 돈다.
- **줄어들 값(추정).** 부팅마다 `production/ready` 에서 약 4.6 s(위 두 부팅의 프리필 합). 추정이지 측정이 아니다.

## 재지 않은 것

- **GPU 부팅.** 다음 부팅의 rank 0 `warmup:` 줄에 `prefill/…` 가 없어야 한다. 표의 `warmup shapes` 는 0.6 s 안팎이어야 한다.
- **첫 요청 TTFT.** 캐시가 빈 노드에서 첫 요청이 정확히 256·2,048·2,304·4,096 토큰이면, 길이를 constexpr 로 받는 Triton 커널(conv 의 `T`)이 그 요청 안에서 컴파일한다.
  - 이것은 여섯 개가 아닌 모든 길이가 이미 치르던 비용이다.
  - 한 번 컴파일하면 노드 디스크에 남는다.
