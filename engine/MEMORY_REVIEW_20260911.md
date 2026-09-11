# ST 엔진 메모리 검토 (2026-09-11, main d44e3825 기준)

코드(`engine/base`, `engine/profiles/glm53`, `engine/kernels` 워크스페이스), 헌장 D1·D16, 원장 45차,
`measurements/st_engine_*` 를 읽고, 프로필 코드로 아레나 구성을 다시 계산한 결과다. GPU 실측은 없다:
검토 시점에 srv2·srv4 모두 프로덕션 vLLM(glm53, 74,437 MiB)이 떠 있어 ST 부팅·할당 시험이 불가했다.
아래 수치 중 "계산" 은 `facts/specs/caches/drafter` 로 계산한 것, "원장" 은 MEASUREMENTS.md 의 실측이다.

## 1. 판정

**메모리는 부족하지 않다. 미할당·미측정이다.** 랭크당 아레나 55.41 GiB 는 119.7~121.6 GiB 상자의 절반이고,
KV 예산 8.73 GiB 는 vLLM 40차 부팅의 잔여를 그대로 옮긴 값이다(`boot.py: KV_GIB`). 4 동시 × 128K 의 KV 는
2.93 GiB 라 지금 예산에서도 3 배 남는다. 전체 45층 부팅이 막힌 것은 양이 아니라 **할당의 모양**이다:
55.4 GiB 를 `cudaMalloc` 한 번으로 달라고 했고(원장 §completion: srv2, 09-11 10:42 UTC, 로드 전 `torch.empty`
에서 CUDA OOM), 같은 상자에서 vLLM 은 `expandable_segments:True`(20 MiB 물리 조각) 로 63 GiB 를 매번 받아 왔다.

## 2. 지금 그림 (랭크 하나)

| 항목 | GiB | 출처 |
|---|---:|---|
| 상자 MemTotal | 121.6 (srv4) / 119.7 (srv2) | `/proc/meminfo` 09-11 |
| earlyoom 바닥(5%) | 6.0 | 원장 D1 |
| **아레나** | **55.41** | 계산 (`boot.build`, KV 8.73, max_seqs 4) |
| ├ 가중치 | 44.50 | 계산 = 랭크 파일 |
| │ ├ 전문가 packed (w13/w2) | 35.44 | NVFP4 nibble |
| │ ├ 전문가 스케일 (sf, e4m3/16) | 4.43 | 형식 고유 (packed 의 12.5%) |
| │ ├ KDA / MLA / dense·shared / embed·head | 2.23 / 0.73 / 0.70 / 0.59 | bf16 |
| │ └ 인덱서 / mHC / 라우터 / 노름 | 0.16 / 0.13 / 0.09 / 0.00 | 복제 텐서 |
| ├ 드래프터 (DFlash2, 랭크마다 통째) | 2.18 | 계산 |
| ├ 페이지드 KV | 7.52 | 580 블록 × 13.28 MiB |
| └ 상태 슬롯 | 1.21 | 5 슬롯 × 247.2 MiB (null 슬롯 포함) |
| 아레나 밖: CUDA ctx + NCCL 16ch | 5.54 | 원장 40차 (vLLM) — ST 재측정 필요 |
| 아레나 밖: 프리필 활성화 @6,912 | ≤ 3.6 (+§3.5 의 인덱서 과도) | 원장 0.52 GiB/1K (vLLM) — ST 재측정 필요 |
| 아레나 밖: 디코드 그래프 40 + 샘플링 80 + 드래프터 7 | 미측정 | `capture decode` 페이즈의 dev 델타가 답 |
| 아레나 밖: 커널 워크스페이스 (MLA 6 MB, b12x 캐시, DeepGEMM) | 미측정 | 부팅 표 |
| 아레나 밖: NVMe 스테이징 | 0.13 | 64 MiB pinned + 64 MiB device |

블록 = 2304 토큰 × 11 DSA 층 = 13,922,304 B (5.90 KiB/토큰, 정렬 낭비 0.79%). 슬롯 247.2 MiB 의 구성:
recurrent 링 204.0 (K+1=6 상태 × 34 층) + 드래프터 링 40.0 + conv 링 3.2 + 꼬리 0.04.

## 3. 발견 (우선순위)

### 3.1 부팅 실패 = 한 번의 55.4 GiB `cudaMalloc` (가설, 근거 넷)

- 실패 지점은 `Arena.__init__` 의 `torch.empty(55.4 GiB)` 이고 로드 전이다. 그 뒤 codex 가 넣은 관문
  `prepare_allocation`(랭크·드래프터 파일 `fadvise DONTNEED` + `MemFree ∧ device-free ≥ 아레나 + 16 GiB` + TP 투표)
  은 단위 테스트 둘만 있고 플릿 부팅으로 검증되지 않았다.
- 같은 상자의 프로덕션 런처(`start-glm53-nvfp4-tp4.sh:434`)는 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  로 돈다. **ST 런처·이미지에는 그 설정이 없다.** 즉 vLLM 은 20 MiB 물리 조각 수천 개, ST 는 55 GiB 한 덩어리.
- 실패 직전 srv2 의 페이지 캐시는 fan-out 으로 막 쓴 랭크 파일 44.5 GiB 로 차 있었을 가능성이 높다
  (`full-fanout.log`: 노드당 47.8 GB, 340~378 MB/s). UMA 드라이버가 한 덩어리 요청에 캐시를 회수해 주지
  않으면 `MemAvailable` 이 99 GiB 여도 실패한다 — codex 의 진단과 같다.
- 실패한 프로브 컨테이너는 `--memory 80g`(`full_fleet_probe.py:26`) 였다. 프로덕션은 `--memory 112g`,
  ST 런처는 제한 없음. cgroup 이 드라이버 할당을 어떻게 세는지는 미확인이라 변수에서 빼야 한다.

**대책**: (a) ST 이미지 ENV 와 `start-st-glm53.sh` 에 `expandable_segments:True` — 아레나는 그대로 텐서 하나
(D16 의 "선언된 하나" 유지), 드라이버는 vLLM 과 같은 조각을 본다; (b) 관문은 유지; (c) 프로브에서
`--memory 80g` 제거; (d) 빈 노드에서 5 분 실험으로 확정: 파일 60 GB 읽어 캐시를 채운 뒤 `torch.empty(55.4 GiB)` 를
평범한 할당자 / expandable / 익명 메모리 터치 펌프 뒤, 세 조건에서 MemFree·MemAvailable·성패를 기록.
게이트: 45층 4노드 부팅이 관문·아레나·로드·캡처를 지나고 `Recorder` 표에 페이즈별 dev 델타가 찍힌다.

### 3.2 D16 이 절반만 있다: 파킹이 블록만 내리고 슬롯·행은 쥔다

- `TieredKV.park` 는 `pool.release` 만 한다. `Runner.park` 는 `slot_of`/`SlotPool` 을 건드리지 않는다.
  파킹된 대화도 247 MiB 슬롯과 요청 행을 계속 소유한다.
- `Server._free_rows = min(max_seqs, max_running, slots) = 4`. 다섯째 새 요청은 `_evict_idle` 로 가장 오래된
  유휴 대화를 **NVMe 파일까지 지운다**. 즉 보존 대화 ≤ 4 이고, 디스크 용량("srv4 787 GB ≈ 6천만 토큰")은
  아무 역할이 없다.
- 원장 §9 의 "다중 영역 티어" 는 현재 `kv_tier.py` 에 없다(단일 storage). 슬롯 바이트를 같은 파일의 둘째
  세그먼트로 내리고, 파킹 뒤 `slots.give` + 행 반납, 대화 id → 파일만 남기면 된다. 드래프터 링은 슬롯 안이라
  같이 내려간다.

### 3.3 D10 위반: 파킹·복귀가 스텝 스레드에서 동기로 돈다

- `Server.once()` 는 끝난 행을 그 자리에서 `runner.park` → `tier.demote`(창마다 `stream.synchronize()` + `pwritev`)
  한다. `_admit` 은 `runner.resume` 을 동기로 부른다. `run_async`/Future 는 서브 루프가 쓰지 않는다.
- 128K 대화 = 57 블록 × 13.28 MiB = 757 MiB; 원장 §9 실측 demote 1.05 GiB/s, promote 4.0 GiB/s → 파킹 **~0.7 s**,
  복귀 ~0.2 s 동안 돌던 디코더 전부가 선다. "무방해 1.001" 은 `run_async` 프로브의 수치이지 서브 루프의 수치가 아니다.
- 대책: 스텝이 돌아온 뒤 `run_async` 로 파킹(완료까지 블록 예약 유지, `done()` 뒤 release), 복귀는 "resuming" 상태로
  큐에 두고 `done()` 뒤 입장. 게이트: `probes/kv_tier_interference.py` 를 서브 루프 경로로, 128K 파킹·복귀 중 p50 비 ≤ 1.01.

### 3.4 예산이 vLLM 표다: `profiles/glm53/budget.py` 가 없다

- qwen38·dsv41 프로필에는 D1 의 줄(출처 붙은 예산)이 있고 glm53 에는 없다. `KV_GIB=8.73` 은 "40차 부팅의 잔여"
  (vLLM 의 로드 스크래치 8.77·fp8 fold·프로파일 런 9.17 을 뺀 값)다. ST 는 그 셋 중 아무것도 내지 않는다.
- 줄을 세우면: 121.6 − OS 12.2(2×5%) − 바닥 5.54(원장, 재측정) − 가중치 44.50 − 드래프터 2.18 − 활성화(측정)
  − 그래프·워크스페이스(측정) − 스테이징 0.13 ≈ **50 GiB 대의 KV+슬롯**, 미측정 두 줄이 4 GiB 안이면. 지금 8.73.
- 다만 4 동시에서 KV 는 묶이지 않는다(2.93 GiB). 남는 예산의 용도는 동시성·컨텍스트·보존 대화 수 중 **운영자
  결정**이고, 예산 파일은 그 결정을 수치로 만든다. 동시성 1 증가의 값: 슬롯 +247 MiB, 토큰당 5,995 B, 대상 그래프 +10.

### 3.5 프리필 인덱서 선택의 과도 활성화: 2 × T × ctx 바이트

- `net._indexer`: `logits [T, ctx/4] f32` + `topk_positions` 의 `masked_fill` 사본. T=6,912 청크에서 128K 컨텍스트
  **1.69 GiB**, 256K **3.4 GiB** 가 DSA 층마다 잠깐 선다(캐싱 할당자, 아레나 밖). 이것이 ST 활성화 피크의 지배항이다.
- 대책: 질의 행 ≤ 1,024 로 나눠 선택(256K 에서 0.25 GiB) + 제자리 마스킹. 게이트: `check.py` chunked/verify/rollback 이
  같은 top-k(정확히 동일한 슬롯 집합).

### 3.6 디코드 그래프가 스텝마다 슬롯 전체를 복사한다 (대역폭 = 지연)

- `GraphCaches.gather/commit` 이 그래프 안에서 n 슬롯의 conv·rec·tail 전부를 `index_select`/`index_copy_` 한다:
  시퀀스당 207 MiB × (읽기+쓰기) × (gather+commit) = 828 MiB → n=4 에서 **3.3 GiB/스텝 ≈ 12 ms** (273 GB/s).
- 커널은 이미 인덱스를 받는다: `fused_recurrent_kda(ssm_state_indices=…)`(IS_CONTINUOUS_BATCHING), `causal_conv1d_fn(cache_indices=…)`.
  슬롯 id 를 커널에 넘기면 복사가 사라진다(vLLM 방식). 게이트: 기존 17 케이스 그래프==eager 바이트 동일.
- **main #549(codex, 같은 날)가 이미 이렇게 바꿨다**: 타깃 그래프가 물리 슬롯 id 를 Triton 상태 커널에 넘기고 상태 링 전체의 gather/commit 을 하지 않는다
  (`measurements/st_engine_four_optimizations_20260911`). 이 항목은 병합으로 닫힌다.
- 캡처 사다리는 4096→1,336,320 의 10 단 × 4 = 40 개 전체 모델 그래프. 서빙 최대 컨텍스트로 사다리를 자르면 그래프 수와
  인덱서 gather 상한이 같이 준다. 그래프 인스턴스 메모리는 `capture decode` 페이즈 델타로 먼저 읽는다.

### 3.7 조건부·작은 것

- **슬롯 압축(조건부)**: rec 링 204 MiB 는 K+1 상태를 시퀀스마다 쥐는 설계. K+1 상태를 스텝 공용 스크래치에 쓰고 수락된
  하나만 슬롯에 남기면 247 → ~80 MiB. 동시성을 올리거나(32 슬롯: 7.97 → 2.6 GiB) 3.2 의 파킹 바이트를 줄일 때만 값이 있다.
- **null 슬롯** 247 MiB: 커널이 0 을 건너뛰므로 저장소는 없어도 된다(id 만 예약). −247 MiB.
- **드래프터 복제** 2.18 GiB × 4: 코드북 151 MiB(2 × 154,880 × 256 bf16) 는 vocab-parallel, 5층은 TP 로 −1.6 GiB/랭크 가능.
  vLLM 도 DRAFT_TP=1 이라 동률; 보류.
- 정렬 낭비 0.79%, 로더 호스트 버퍼 2 × 1 GiB(일시), `--shm-size 32g` 는 `--ipc host` 아래서 무시(무해), 스테이징 128 MiB 고정.

## 4. 제안 순서 (이 브랜치에서)

1. **3.1** 할당 모양: ENV + 프로브 `--memory` 제거 + 5 분 실험 → 45층 부팅 관문 통과. 다른 모든 것의 전제.
2. **3.4** 부팅 표(`Recorder`)의 페이즈 델타로 `profiles/glm53/budget.py` 를 세우고 KV·max_seqs 를 그 표에서 선언.
3. **3.3 + 3.2** 티어를 스텝 밖으로, 슬롯까지 파킹, 행 반납 → D10·D16 성립. 게이트는 간섭 프로브 + 바이트 동일 복귀.
4. **3.5** 인덱서 선택 청킹, **3.6** 슬롯 인덱스 전달 + 사다리 상한.
5. **3.7** 은 동시성 결정 뒤.

확인 못 한 것: 실패 당시 srv2 커널 로그(`journalctl -k` 비어 있음), cgroup 이 GB10 드라이버 할당을 세는지, 한 덩어리 vs
조각 할당의 실제 성패(실험 3.1(d)가 답).

## 5. 조치 (2026-09-11, 같은 브랜치) — 3.1 과 3.2 를 코드로

**3.1 할당 모양**
- `base/arena.Arena`: 할당 전에 `expandable_segments:True` 를 켠다(`torch._C._accelerator_setAllocatorSettings`, 구 API 폴백).
  한 텐서, 20 MiB 물리 조각 — vLLM 이 이 상자에서 63 GiB 를 받는 방식. `arena.expandable` 과 `table()` 에 표시. 이 상자에서
  할당 뒤에 켜도 새 세그먼트에 적용됨을 확인(1 GiB 세그먼트 `is_expandable: True`).
- `boot.py` 는 `import torch` 전에 `PYTORCH_CUDA_ALLOC_CONF` 를 setdefault, ST 이미지 ENV 와 `start-st-glm53.sh` 에도 같은 값.
- `prepare_allocation`: 파일 캐시 버리기 뒤에도 `MemFree` 가 아레나 + 16 GiB 에 못 미치면, `MemAvailable` 이 부족분을 허락할 때만
  익명 페이지를 `MAP_POPULATE` 로 잠깐 잡았다 놓아(`touch_pages`) 캐시를 회수하고 다시 잰다. 펌프 뒤에도 모자라면 닫힌 실패.
  부팅 표에 `boot_reclaimed_GiB`·`arena_expandable`.
- `probes/engine_alloc_shape_check.py`: 빈 노드에서 캐시를 채운 뒤 plain / expandable / reclaim 세 조건을 각각 자식 프로세스로
  시험해 JSON 한 줄씩 남긴다. **플릿 창에서 돌릴 것** — 검토 시점엔 srv2·srv4 모두 프로덕션 vLLM 이 GPU 를 쥐고 있어 미실행.

**3.2 D16 완성**
- `NvmeTier.demote/promote`: 파일 = [블록들][슬롯 바이트, 섹터 패딩], 옆에 JSON 기록; manifest 에 `extra`·`record`; 용량은
  파일시스템 여유(예비 1 GiB) 또는 선언 `capacity_bytes`, 넘치면 쓰기 전 `TierFull`; `keys()`·`oldest()`·`record()`.
- `TieredKV.park(row, key, extra, record)` / `resume(row, key, extra)`: 대화 키와 행을 분리.
- `Runner.park`: `Model.park` 기록 + 슬롯 바이트를 내리고 **행·슬롯 반납**; `Runner.resume(row, key)`: 빈 행·빈 슬롯으로 복귀,
  `Model.resume` 은 슬롯을 지우지 않음. 실패 시 각각 상주 유지 / 받은 것 반납 + 디스크 사본 보존.
- `Server`: 끝난 턴은 대화 id 로 파킹되어 행이 곧바로 빈다; 이어가기는 파킹 기록의 문맥으로 예산을 확인하고 빈 행으로 복원;
  `TierFull` 이면 가장 오래된 파킹 대화부터 잊고, 잊을 것이 없으면 보존하지 않는다; 요청 번호는 티어의 최대 대화 id 위에서
  시작하고 엔진이 죽어도 파킹 대화는 남는다(재부팅 뒤 같은 id 로 이어가기 가능).
- `Glm53Engine.park/resume/state_bytes`, `Glm53Caches.slot_bytes`.
- 검증: `tests/test_engine_{tier,serve,arena_admission}.py` 갱신·추가(키 분리, 슬롯 바이트 이동, 티어 초과 정책, 재시작 뒤 이어가기,
  회수 펌프 경계); GPU 에서 `kv_tier`·`tiered_kv` 자가검증과 `probes/engine_cuda_io_check.py`(슬롯·기록 왕복, TierFull) 통과.
- **남은 것**: 3.3(파킹·복귀를 스텝 밖으로)은 그대로다 — 이 변경으로 파킹 바이트에 슬롯 247 MiB 가 더해졌으니 더 급해졌다.
- **사고 (21:24:55 KST)**: 위 검증 중 `boot.py --local --park --layers 0-1`(호스트, 4 랭크 스레드)을 ST 이미지 안 유닛 스위트 컨테이너와
  **동시에** srv4 에서 돌렸고, 프로덕션 vLLM(glm53-worker)이 GPU 를 쥔 상태에서 MemAvailable 이 4.93% 로 떨어져 earlyoom 이
  프로덕션 워커(rank 3)와 스모크를 죽였다. srv2 의 `fleet-idle-recovery.service` 가 스택을 자동 재기동(그 절차는 `drop_caches` 를
  sudo 로 한다 — 3.1 의 페이지 캐시 문제를 플릿 복구 경로는 이미 알고 있었다). 두 번째 스모크(`--serve --park`, 12 스텝, HTTP 3 건,
  파킹된 대화 0 의 둘째 턴 4 토큰, 네 랭크 락스텝 PASS)는 워커가 죽은 뒤의 빈 상자에서 돈 결과라 **프로덕션을 세운 대가로 얻은
  수치**다. 첫 스모크(`--park` 왕복 == 한 번에 돌린 꼬리)는 죽어서 미완. 재현 규칙은 메모리 `feedback-no-gpu-smoke-beside-production`.

## 6. 남은 순서의 2·3·4 (2026-09-11 밤, 같은 브랜치)

- **3.4 예산**: `profiles/glm53/budget.py` — 줄마다 출처(OS 예비 declared, 바닥 ledger, 가중치·드래프터·슬롯 read, 작업공간 상한 12 GiB
  declared = `runtime_memory` 가 강제하는 값, 부팅 원장을 주면 실측 피크를 증거에 적음). 121.6 GiB 상자: 남는 자리 **43.9 GiB**, 선언 KV 8.73 은
  그중 7.5 → **36.4 GiB 미할당**. 전체 모델 부팅이 rank 0 에서 표를 찍고 `budget_unassigned_GiB` 를 게이지로 남긴다. KV·max_seqs 의 새 값은
  운영자 결정(동시성/컨텍스트/보존 대화) — 표가 그 결정의 근거.
- **3.3 D10**: 파킹·복원이 `Runner.park_begin/park_finish`, `resume_begin/resume_finish` 두 반쪽이 되어 티어 스레드에서 돌고, `Server._settle` 이
  매 스텝 `transfer_done` 만 묻는다. 완료·성패를 `all_reduce` 로 네 랭크가 합의한 뒤 적용(락스텝 유지): 한 랭크라도 실패 → 그 대화는 모든 랭크에서
  버림(복원 실패는 503, 엔진은 살아 있음), 모두 TierFull → 가장 오래된 파킹 대화를 잊고 재시도. 정지 때는 진행 중 전송을 기다려 마무리.
  게이트(플릿 창): `probes/kv_tier_interference.py` 를 서브 루프 경로로 — 128K 파킹·복원 중 디코더 p50 비 ≤ 1.01.
- **3.5 인덱서 과도**: `net._select_pools` 가 질의 1,024 행씩 선택(정확히 같은 top-k; `topk_positions(inplace=True)` 로 마스크 사본 제거).
  6,912 청크 × 128K: 층당 1.69 GiB → 0.13 GiB. **캡처 사다리**는 `facts.max_position`(1,048,576) 에서 끝나고 문은 그 너머 문맥을 400 으로 거부.
- 검증: CPU 스위트 178 통과(새 테스트: 예산 2, 선택 청킹 2, 티어 반쪽 1, 서브 비동기 4 + 4랭크 락스텝 1).
