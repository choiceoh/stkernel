# 프리필 메모리 게이트 재사용 — 같은 빌드·설정·노드의 재부팅은 먼 끝 패스를 기록으로 대신한다 (2026-09-15)

운영자 지시(2026-09-15): "두개 다 최대한 줄이고 컴파일 캐시 지우지 말고". 재사용 설계는 "이거 좋네" 로 승인됐다.

## 1. 무엇인가

- **게이트가 하는 일.** 문을 열기 전에 `adapter._warmup_prefill_memory` 가 가장 큰 프리필 청크(32,256 토큰)를 서빙 문맥의 양 끝(위치 0, 1,016,320)에서 돌린다(#549).
- **판정.** 행마다 모든 랭크가 할당자 상한, OS 예약, 상자의 kill 선에 대해 투표한다.
- **먼 끝 패스의 비용.** 가장 비싸고, 피크도 이 패스가 낸다.

| 부팅(rank 0 원장) | 빌드 | 먼 끝 패스 s | 위치 0 피크 워크스페이스 B | 먼 끝 피크 워크스페이스 B |
|---|---|---:|---:|---:|
| expert-c1c2-0914-main963 | 9c45086a | 29.87 | 7,059,170,264 | 10,624,328,664 |
| c2opt-profile-main963 | 9c45086a | 30.22 | 7,059,170,264 | 10,624,328,664 |
| tempab2-0915 | 4c128a79 | 30.57 | 7,059,170,264 | 10,624,328,664 |
| tempab5-0915 | 7166a270 | 29.54 | 7,059,170,264 | 10,624,328,664 |
| selfeat-0915 | 745a8dd3 | 29.52 | 7,080,141,784 | 10,414,613,464 |

- **프로덕션 부팅.** main `3acae017` 이 12:38 KST 에 떴다(티어 끔, #981·#984 캐시 적중, 부팅 168.9 s). 게이트는 84.8 s 로 가장 큰 부팅 항목이 됐고, 그중 먼 끝 패스가 **47.40 s** 다.
  - 128: 15.41 s, 1024: 2.07 s, 32K@0: 19.49 s, 32K@1M: 47.40 s. 원장은 `srv2:~/glm53-logs/st-dumps/memory-rank0.json` 이고, 사본은 `memory-rank0-production-3acae017.json` 이다.
  - 먼 끝 패스가 브래킷 다섯 번(29.5–30.6 s)보다 17.9 s 길다. 원인은 못 가렸고, 아래 셋이 **아님**만 확인했다.
    - 컴파일: 네 노드 모두 그 창(12:39:41–12:40:30 KST)에 `glm53-cache` 쓰기가 0 개이고, 로그에도 줄이 없다.
    - 회수: 행의 `reclaimed_bytes` 가 0 이다.
    - 단일 GPU 레인 경합: 12:37:47 release 와 12:40:59 GO 사이에 티켓이 없었다.
  - 비티어 스냅샷(4.25 GiB)은 스냅샷 슬롯 수만 48 → 96 으로 늘린다. 이 패스가 쓰는 마크는 `range(768, 32256, 768)` 의 41개로 같다. 캡처 모양당 초는 두 부팅이 같다(0.31–0.35 s).
- **결정성.** 네 부팅, 세 빌드에서 두 피크가 **바이트까지 같다.**
  - 같은 빌드·설정·가중치·노드의 부팅은 먼 끝 패스에 같은 바이트를 쓴다.
  - 그러면 재부팅마다 30 초를 다시 낼 필요가 없다.

## 2. 바꾼 것

- `engine/base/prefill_record.py`(base, 모델 이름 없음): 키 구성, 기록 읽기·쓰기, 판정.
- `engine/base/runtime_memory.py`: `agree()`(한 번의 집합통신 투표)와 `reused()`(투영 행)를 더했다. `measured()` 에는 `reused_phases` 가 붙는다.
- `engine/profiles/glm53/adapter.py`
  - 위치 0 패스 뒤, 먼 끝 패스 앞에서 투표한다.
  - 모든 랭크가 재사용하면 먼 끝 패스 대신 이어 쓰기(continuation) 패스를 돈다. `PREFILL_CONTINUATION_TOKENS=1024`, 문맥 32,256.
  - 먼 끝을 돌았으면 기록에 알린다.
- `engine/profiles/glm53/boot.py`
  - `fleet()` 이 캡처 전에 모든 랭크에 기록을 묶는다.
  - 기록은 `production/ready` 가 통과한 뒤에만 쓴다.
  - `prefill_record_components()` 와 `--full-memory-gate` 를 더했다.
- `launchers/start-st-glm53.sh`: `ST_FULL_MEMORY_GATE=1` 이면 `--full-memory-gate` 를 넘긴다.

## 3. 안전 요구 — 무엇이 어떻게 지키나

1. **첫 부팅은 전체 게이트.**
   - 기록은 먼 끝 패스를 돌고, 원장의 모든 행이 통과한 부팅만 쓴다(`PrefillRecord.write`). 실패 행, 재사용 행, 먼 끝 행 부재는 거절한다.
   - 쓰기는 원자적이다(같은 디렉터리 임시 파일 + fsync + `os.replace`).
   - 위치는 노드별 `/cache/st-gate/`(= 노드의 `~/glm53-cache/st-gate`)다.
   - 파일은 **랭크·키마다 하나**다. 프로덕션과 그 사이 티켓 부팅(다른 트리·설정)이 서로의 기록을 덮지 않게 하려는 것이다.
   - 최근 사용 순으로 32개를 남긴다. 재사용하면 mtime 이 갱신된다.
2. **키**(`prefill_record_components`). 아래 중 하나라도 다르면 전체 게이트다.
   - 엔진 트리 전체: 경로와 내용의 sha256, 바이트코드 제외.
   - 런타임: `engine.runtime.verify.PACKAGES` 버전(torch, triton, flashinfer-python, tilelang, nvidia-cutlass-dsl …), `torch.version.cuda`·git, 이미지 매니페스트 `/opt/st/runtime-manifest.json`.
   - 가중치: 랭크, 드래프터, 비전 safetensors 의 크기, mtime, 헤더 해시.
   - 체크포인트 메타: 내용 해시. 런처가 런치마다 `cp` 로 새로 복사하므로 mtime 은 넣지 않는다.
   - 선언 설정: `cfg.values` 전부(port 제외), `STK_*`·`PYTORCH_*` 환경, rank/world, `kv_gib`, 티어, 아레나, 워크스페이스, OS 예약, 호스트 예산 바이트, `max_context`, 프리필 청크, `max_seqs`, 블록, 스냅샷, 디코드 폭(K+1), 레인 이름, `lane_info`(부팅마다 재는 `oneshot_latency_us` 제외), 비전·문법 유무.
   - 노드: hostname, GPU UUID·이름·메모리, `/proc/driver/nvidia/version` 해시.
   - 기록이 없으면 이유가 "가장 최근 기록과 어느 성분이 다른지"를 점 표기로 말한다.
3. **오늘의 상자가 판정한다.**
   - 위치 0 패스는 그대로 돌고, 보통의 행 투표를 거친다.
   - 그 피크는 기록의 위치 0 피크 + 64 MiB 안이어야 한다. 기록이 **이 부팅**을 설명해야 하기 때문이다.
   - 먼 끝 피크를 이렇게 투영한다: `max(기록 먼 끝, 오늘 위치 0 + (기록 먼 끝 − 기록 위치 0))`.
   - 투영값이 이 부팅의 할당자 상한 안이어야 한다.
   - 위치 0 행이 **방금 읽은** immediately-free/available 에서 피크까지 더 필요한 바이트를 빼고, 그 값이 OS 예약 이상, SIGTERM 선 이상이어야 한다. 전체 게이트의 실패선인 SIGKILL 보다 엄격하다.
   - 하나라도 모자라면 먼 끝 패스를 돈다.
   - main963 원장으로 계산하면 투영 available 은 25.19 GiB 다. 전체 게이트가 해제 뒤에 읽은 28.96 GiB 보다 보수적이다.
   - 입장(`prepare_allocation`)은 여전히 선언 상한(12 GiB) + OS 예약으로 오늘의 상자를 잰다.
4. **모든 랭크 합의.**
   - `RuntimeMemory.agree`: 값 하나의 `all_reduce_max` 다. 재사용이면 0, 패스가 필요하면 2.
   - 한 랭크라도 2 면 모든 랭크가 같이 먼 끝 패스를 돈다.
   - 1 은 동료의 실패 투표와 짝지어진 것이므로 모든 랭크가 `MemoryError` 로 멈춘다.
   - 기록이 없거나 키 계산이 실패한 랭크도 같은 자리에서 투표한다(`PrefillRecord.build` 는 예외를 던지지 않는다).
5. **기능 검사.**
   - 재사용 부팅은 먼 끝 대신 1,024 토큰을 문맥 32,256 에서 돈다. finite 검사, 자기 원장 행과 투표를 그대로 거친다.
   - 이것은 부팅에서 유일한 **비영 문맥 프리필**이다. 재귀·conv 상태 읽기, top-k 보다 많은 풀에서의 선택, 커버되지 않은 sparse MLA 경로를 문 앞에서 한 번 통과시킨다.
   - **재사용 부팅이 그날 더는 증명하지 않는 것**: 먼 끝(문맥 1,016,320)에서 가장 큰 청크의 실행과 finite 출력. 그 피크 바이트는 같은 키의 기록과 오늘 위치 0 피크의 일치로 대신한다.
6. **컴파일 빚.**
   - tempab5 부팅의 먼 끝 패스 창(11:29:47–11:30:17 KST)에 srv2·srv4 의 `glm53-cache/cu132`(Triton·CuTe·dense·mla·oneshot)에 **쓰인 파일은 0 개**다.
   - rank 3 로그에도 그 창에 CuTe 컴파일 줄이 없다. 먼 끝 패스는 새로 컴파일한 것이 없고, 위치 0 패스와 이전 부팅이 컴파일한 것을 불렀을 뿐이다.
   - 재사용은 같은 노드에서 같은 빌드가 전체 게이트를 통과한 뒤에만 일어난다. 그래서 먼 끝 패스만 쓰던 특수화는 디스크 캐시에서 **읽기만** 한다: Triton `_single_conv`(T=32,256, 상태 있음), 먼 문맥 인덱서 로짓 모양. native 확장(#984)과 CuTe(#981)도 캐시에 있다.
   - Triton 오토튠 키(`chunk_delta_h` `H,K,V,BT,AUTOTUNE_REGIME,HAS_MARKS`, `solve_tril`, `cumsum`, `l2norm`, `kda.py` 의 키들)에는 초기 상태 유무도 문맥도 없다. 먼 끝 패스의 키는 같은 프로세스의 위치 0 패스가 이미 튠한 키와 같다. 재벤치마크 빚은 없다.
   - 남는 것은 첫 먼 문맥 긴 요청이 내는 캐시 로드와 할당자 성장이다(약 3.3 GiB 페이지 매핑, 26 GiB/s 기준 약 0.13 s). 둘 다 **미측정 추정**이다.
7. **위치 0 대 1M.** 위치 0 은 **떨어뜨리지 않았다.**
   - `_single_conv` 는 `T` 와 `HAS_STATE` 가 constexpr 다(`engine/kernels/causal_conv_single.py:16, :103`). 문맥 0 은 이력 없이(`net.py:564`, `:560`) 부른다.
   - 그래서 (T=32,256, 상태 없음) 특수화는 위치 0 패스만 부른다. 128·1024·4095·`warmup_shapes` 는 다른 T 다.
   - 위치 0 을 빼면 새 프롬프트 ≥32K 의 첫 요청이 그 로드를 낸다. 그 파일을 바꾼 빌드의 첫 부팅이라면 컴파일을 낸다.
   - 나머지 문맥 0 전용 경로는 다른 곳에서 덮인다.
     - 커버된 프리픽스(`_mla_prefix` `net.py:914`, `covered_pool_ids`)는 128·1024(전부 커버)와 4095 커널 웜업(프리픽스 2,051 + 선택)이 부른다.
     - MLA prefill32 은 문맥 0 에서도 1M 에서도 같은 커널이다. 로그 `T=30205 W=2051` 의 W 는 선택 폭 2,048+3 이고, 커버 여부가 아니다.
     - 스냅샷 마크는 길이만 따른다.
   - 대신 위치 0 은 재사용 부팅이 기록을 보증하는 패스가 됐다. 전체 게이트의 모양 집합은 바뀌지 않는다.
8. **노브 아님, 강제 방법.**
   - STK_ 노브는 없다.
   - 전체 게이트를 강제하는 법은 셋이다: 노드의 `~/glm53-cache/st-gate` 삭제, `boot.py --full-memory-gate`, 런처 `ST_FULL_MEMORY_GATE=1`.
   - 강제된 전체 게이트가 통과하면 그 키의 기록을 다시 쓴다.

## 4. 절감 추정 — 같은 빌드 재부팅 한 번

- 먼 끝 패스: 프로덕션 main `3acae017` 에서 **47.40 s**, 브래킷에서 29.52–30.57 s(표).
- 대신 도는 이어 쓰기: 약 2 s(추정). 원장의 `prefill/1024/0` 이 프로덕션 2.07 s, 브래킷 1.54–1.66 s 이고, 문맥 32,256 의 선택이 조금 더한다.
- 키 계산: 약 0.3 s(추정). 엔진 트리 6.4 MiB, 헤더 셋, tokenizer.json 20 MiB.
- **합: 프로덕션 재부팅 한 번에 약 −45 s(게이트 84.8 → 약 40 s), 브래킷 모양으로는 약 −28 s. 둘 다 추정이고, GPU 부팅은 재지 않았다.**
- 집합통신 사이의 랭크별 일은 키 계산(약 0.3 s, 캡처 앞)과 판정(호스트 산술)뿐이다. 콜드 부팅을 죽인 컴파일 길이의 지연(RDMA WC 12)과는 자릿수가 다르다.
- 해당하는 경우: 티켓 뒤 프로덕션 복원, 슈퍼바이저 재시작.
- 해당하지 않는 경우: 새 커밋의 첫 부팅(엔진 트리가 다르다), 설정 변경, 강제.

## 5. 검증

CPU, 이미지 `st-engine:bracket-9c45086a0622`, `CUDA_VISIBLE_DEVICES=`:

```bash
flock -w 7200 /tmp/c2opt-heavy.lock docker run --rm -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/repo -v "$PWD":/repo:ro -w /repo \
  --entrypoint python3 st-engine:bracket-9c45086a0622 -m unittest tests.test_engine_prefill_record \
  tests.test_engine_prefill_outputs tests.test_engine_runtime_memory tests.test_engine_bootpaths tests.test_engine_budget \
  tests.test_engine_charter tests.test_engine_kernels tests.test_engine_fleet_ops tests.test_engine_knobs tests.test_engine_native_execution
```

결과: 147 테스트 OK(12 스킵).

바꾼 파일을 이름으로 부르는 테스트 모듈 전부(64개)도 같은 이미지에서 돌렸다. 브랜치는 934, origin/main `3acae017` 내보내기는 910 이다.
- 오류·실패 목록이 **같다**: `test_fleet_st_bracket`·`test_fleet_approval` 26 오류, `test_engine_host_reclaim` 브로커 2 실패.
- 원인은 컨테이너 환경이다. `/repo` 에 git 저장소가 없고, 브로커 프로세스 수명을 본다.

`tests/test_engine_prefill_record.py` 는 24 테스트이고, 다룬 것은 이렇다.
- **키 구성.** 성분 23곳 각각이 키를 바꾸고, 이유가 그 점 표기 이름을 말한다. 트리 해시는 바이트코드를 무시한다. 가중치는 헤더·크기·mtime 이고, 메타는 내용이며 런치마다의 `cp` 를 견딘다.
- **불일치 → 전체 게이트.** 기록 없음, 키 다름, 스키마, 깨진 JSON, 불완전 행, 강제, 키 계산 실패.
- **랭크 불일치.** 동료 하나가 2 면 모든 랭크가 먼 끝 패스를 돈다. 기록 없는 랭크도 투표한다. 동료 실패(1)는 멈춘다.
- **전체 게이트를 통과한 뒤에만 쓴다.** 먼 끝 미실행, 실패 행, 재사용 행, 행 부재, 키 없음은 거절한다. 원자적 쓰기에서 중단돼도 앞 기록은 온전하다.
- **바닥 재확인.** 위치 0 피크가 기록 +64 MiB 를 넘으면, OS 예약 아래이면, SIGTERM 선 아래이면, 상한이 줄었으면 전체 게이트다. 경계값은 1 바이트 단위로 본다.
- **프로덕션과 티켓 기록의 공존**, 최근 사용 순 정리.
- **어댑터(CPU 가짜 캐시).** 어떤 패스가 도는지, 투표 수, 투영 행, 이어 쓰기 행 이름, 해제.
- **런처·부트 배선.** 캡처 전 바인딩, `production/ready` 뒤 쓰기, `ST_FULL_MEMORY_GATE`, `--full-memory-gate`.

## 6. 재지 않은 것

- **GPU 부팅.** 다음 부팅들의 로그에서 읽을 것.
  - 첫 부팅(노드마다 기록 없음): `memory gate: rank N no record on this node` → `ran the full gate` → `kept its full gate at /cache/st-gate/prefill-rankN-<key>.json`.
  - 같은 빌드의 둘째 부팅: `memory gate: rank N the record matches` → `reused ... ran prefill/1024/32256 in its place`. 원장에 `prefill/32256/1016320/reused` 행이 있고, `spend` 의 `prefill` 이 약 28 s 줄어야 한다.
  - 둘째 부팅이 `runs prefill/32256/1016320 -- …` 를 찍으면 그 이유가 곧 진단이다. 예: 부팅마다 바뀌는 성분이 키에 남아 있는 경우.
- 이어 쓰기 패스의 실제 초와 키 계산 시간.
- 먼 문맥 긴 요청의 첫 호출 로드 비용.
