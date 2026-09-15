# 네이티브 확장 넷이 부팅마다 다시 컴파일되고 있었다 (2026-09-15)

## 무엇인가

ST 엔진의 torch 확장은 일곱 개다. 그중 셋(`mla`, `dense`, `oneshot`)은 `/cache` 아래(`ST_*_BUILD_ROOT`)에 빌드한다.
나머지 넷 — `bounded_graph`, `decode_queue`, `mapped_staging`, `prefill_topk` — 은 `Path.home()/.cache/st/<이름>` 에 빌드했다.
컨테이너 안의 `$HOME` 은 `/root` 이고, 여기에는 마운트가 없다. 이미지 `st-engine:bracket-9c45086a0622` 에서 `HOME=/root` 이고 `/root/.cache/st` 가 없음을 확인했다.
런처는 매번 `docker rm -f` 로 멈춘다. 그래서 **네 빌드는 부팅마다 사라지고, 네 랭크가 부팅마다 넷을 다시 컴파일했다.**

## 1. 실제 부팅에서 — tempab5-0915 hold, srv4 = rank 3

- 증거 하나: 살아 있는 컨테이너 안 빌드 디렉터리의 시각(`tempab5-rank3-root-cache-st.txt`).
- 증거 둘: 같은 부팅의 rank-0 원장(`tempab5-ledger-utc.tsv`). UTC = 02:25:27.9 + `at_seconds` 로 맞췄다.
- `/root/.cache/st` 자체가 02:27:31 에 만들어졌다. 이 부팅이 처음 만든 디렉터리다.

| 모듈 | 부르는 곳 | 디렉터리 생성 → `.so` (UTC) | 초 | 그 초가 들어간 원장 행 |
|---|---|---|---:|---|
| mapped-staging | `NvmeTier(mapped_staging=True)`, boot `runner` 단계 (티어를 끄면 `BurstDecode` 의 `SharedDecodeQueue`) | 02:27:31.29 → 02:28:17.35 | 46.06 | `prefill/128/0/before` +46.2 s |
| prefill-topk | `net.py` `native_select`, 풀이 512 개를 넘는 첫 프리필(32,256 토큰, 문맥 0) | 02:28:43.22 → 02:29:28.74 | 45.52 | `prefill/32256/0/prepared` +65.5 s |
| bounded-graph | `Glm53DecodeGraphs.__init__` (`decode_iterations` 4) | 02:30:40.93 → 02:31:37.05 | 56.12 | `target/(2, 8, 1048576)` +68.4 s |
| decode-queue | `BurstDecode.__init__` → `SharedDecodeQueue` | 02:32:04.42 → 02:32:51.80 | 47.38 | `bounded-decode/(2, 8, 1048576)/4` +47.8 s |
| **합** | | | **195.07** | 473.3 s 부팅 중 |

rank 3 로그에서 설명되지 않던 두 공백이 여기서 풀린다(`tempab5-rank3-log-excerpt.txt`).
- `boot rendezvous weights-loaded`(02:27:31) → 첫 프리필 집합통신(02:28:17): mapped-staging 빌드다.
- 32K 패스 안의 `routed_rows=8192`(02:28:41) → `[megakernel] selftest mla ... -> ARM`(02:29:29): prefill-topk 빌드다.

같은 커밋의 두 부팅(expert-c1c2-0914-main963, c2opt-profile-main963)이 이 네 행에서 같은 초를 낸 것도 같은 이유다.
TP 랭크는 다음 집합통신에서 가장 느린 랭크를 기다린다. 그래서 부팅은 랭크별 최댓값을 낸다(rank 0 의 `runner` 는 25.6 s, rank 3 은 46 s).

## 2. CPU 에서 — 같은 이미지, 두 컨테이너가 `/cache` 하나를 나눠 쓴다

```bash
flock -w 7200 /tmp/c2opt-heavy.lock bash measurements/st_native_build_root_20260915/native_twice.sh
```

- 이미지: `st-engine:bracket-9c45086a0622` (`sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5`).
- 환경: `CUDA_VISIBLE_DEVICES=`, `ST_NATIVE_BUILD_ROOT=/cache/cu132/st-native`.
- 소스는 이 브랜치(`ostcode/native-build-root`)다. GPU 는 건드리지 않는다(빌드와 `.so` 로드뿐이다).
- 원시 출력: `native_twice.log`.

| 모듈 | 첫 컨테이너(차가움) s | 둘째 컨테이너 s | 둘째의 `.cuda.o` mtime |
|---|---:|---:|---|
| mapped-staging | 56.93 | 0.10 | 첫 컨테이너와 같다 |
| prefill-topk | 51.17 | 0.03 | 같다 |
| bounded-graph | 50.13 | 0.03 | 같다 |
| decode-queue | 58.87 | 0.04 | 같다 |
| **합** | **217.10** | **0.20** | |

네 키(`57cf2633…`, `6a747841…`, `28681e63…`, `fe97e13a…`)는 살아 있던 부팅 컨테이너의 키와 같다.
즉 소스·플래그·torch·툴킷이 같은 다음 부팅은 이 빌드를 그대로 읽는다.

## 3. 바꾼 것

- `engine/kernels/common/native_cache.build_root(name)` 을 더했다. 값은 `$ST_NATIVE_BUILD_ROOT/<name>` 이고, 변수가 없으면 예전 `$HOME/.cache/st/<name>` 이다. 네 모듈이 이것을 쓴다.
- `engine/runtime/Dockerfile` ENV 와 `launchers/start-st-glm53.sh` 의 `-e` 에 `ST_NATIVE_BUILD_ROOT=/cache/cu132/st-native` 를 넣었다.
- `probes/run_engine_probe.sh` 도 기본으로 넘긴다. 핀된 옛 이미지(ENV 없음)로 도는 단일 GPU 레인도 `/cache` 에 남긴다.
- `bench/st_bracket.sh` 는 릴리스의 `launchers/start-st-glm53.sh` 로 띄우므로 따로 바꿀 것이 없다.
- `/cache` 는 노드마다 따로다(`/home/choiceoh/glm53-cache`). srv2 와 srv4 의 `cu132` 디렉터리는 inode 도 mtime 도 다르다.
- 한 노드 안의 동시 빌드는 torch 의 빌드 디렉터리 락과 `prepare_sources` 의 flock 이 가른다.

## 4. 재지 않은 것

- **GPU 부팅.** 다음 부팅의 원장에서 읽을 것.
  - 배포 뒤 노드마다 첫 부팅은 여전히 한 번 컴파일한다(가장 느린 랭크 기준 약 200 s).
  - 그다음 부팅부터 줄 행은 이렇다.
    - `prefill/128/0/before`(티어 켬): 약 −45 s
    - `prefill/32256/0/prepared`: 약 −45 s
    - `target/(2, 8, 1048576)`: 약 −56 s
    - `bounded-decode/(2, 8, 1048576)/4`: 약 −47 s
- 네 소스(`loop.cu`, `queue.cu`, `buffer.cu`, `prefill_topk.cu`)나 torch·툴킷이 바뀌면, 그 모듈만 새 키로 한 번 다시 빌드한다(의도된 무효화).
