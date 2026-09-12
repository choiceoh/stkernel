# 메모리: vLLM 과 ST — 부팅과 서빙 (2026-09-12, srv4, GPU 없음)

같은 상자 이미지의 vLLM(`v1/worker/gpu_worker.py`, `config/cache.py`, `v1/core/sched/scheduler.py`)을 읽고,
`MEMORY_REVIEW_20260911`·`OOM_STUDY_20260911`·`base/{arena,cache_spec,runtime_memory}` 와 줄줄이 맞춰 봤다.
**결론: 부팅은 vLLM 이 재고 우리는 선언한다. 서빙은 둘 다 눈이 없는데, 눈이 없어서 죽는 쪽은 우리다.**

## 1. 부팅 — vLLM 은 재고, 우리는 선언한다

vLLM 의 KV 예산은 **측정에서 나온다**(`determine_available_memory`):

```
available_kv = total × gpu_memory_utilization
             − non_kv_cache_memory        (가중치 + 과도 피크 + torch 밖)
             − cudagraph_memory_estimate
```

- 과도 피크는 **`max_num_batched_tokens` 로 더미 포워드 한 번**(`profile_run`)을 돌려 잰다.
- 그래프 메모리는 **`profile_cudagraph_memory()`** 로 따로 추정해 뺀다(`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`).
- 재는 게 싫으면 `kv_cache_memory_bytes` 로 숫자를 박고 프로파일을 건너뛴다 — 다만 로그가 "이건 util 을 안 지킨다" 고 경고한다.
- 할당 모양은 `expandable_segments:True`(20 MiB 조각).

우리는 `KV_GIB = 8.73` 을 **선언한다**. 그 값의 출처는 40차 vLLM 부팅의 잔여다(`MEMORY_REVIEW` 3.4). `budget.py` 가 줄을 세웠지만
**그 줄의 두 항 — 프리필 활성화 피크와 그래프·워크스페이스 — 은 아직 미측정**이고, 그게 검토가 남긴 숙제다.

| | vLLM | ST |
|---|---|---|
| KV 예산 | 더미 포워드로 **측정** | **선언**(vLLM 잔여에서 물려받음) |
| 그래프 메모리 | 추정해 차감 | 미측정(`RuntimeMemory.checkpoint(phase)`·`weigh()` 가 페이즈별 reserved 델타를 남기는 기구는 있다) |
| 수동 고정 | `kv_cache_memory_bytes` | 선언값 자체가 고정이다 |
| 할당 모양 | expandable 20 MiB 조각 | **한 할당 + 범프**(D1) |
| 가중치 로드 호스트 비용 | 텐서별 safetensors 가 기본 | 합친 O_DIRECT + 핀 버퍼 둘 = **1.125 GiB**(45차 §49) |

**우리가 나은 곳**: 로드 경로(합치기·O_DIRECT·뷰), 그리고 아레나가 하나라 **부팅 뒤 발자국이 상수**다.
**vLLM 이 나은 곳**: 예산이 **그 상자에서 잰 값**이라 기계가 바뀌어도 따라간다. 우리 8.73 은 다른 엔진의 부팅에서 온 상수다.

## 2. 서빙 — 둘 다 눈이 없다. 그런데 죽는 쪽은 우리다

- **블록이 모자랄 때**: vLLM 은 **선점**한다(재계산, 또는 `--swap-space` 기본 4 GiB 의 CPU 로 스왑). 우리는 **애초에 과다 입장을 안 한다** — 입장 때 지평선 전체를 예약하고 선점이 없다(D3). 이건 의도된 차이고 vLLM 의 기본값(`full_sequence_must_fit=True`)도 같은 선택이다.
- **런타임에 메모리를 돌려주기**: vLLM 은 `sleep(level)`/`wake_up` 으로 `CuMemAllocator` 를 통해 **가중치를 OS 로 반납**할 수 있다. 우리는 **구조적으로 불가능**하다 — 아레나는 한 할당 + 범프이고 NVRM 페이지는 핀이다(`OOM_STUDY` §2). 우리에게 그런 요구가 없으므로 손해는 아니지만, "런타임 축소" 라는 선택지는 존재하지 않는다.
- **대화를 상자 밖으로**: vLLM 의 스왑은 **선점된 블록만**, CPU 4 GiB. 우리 NVMe 티어(D16)는 **파킹된 대화 전체**를 LRU 로 들고 턴을 넘겨 산다. **우리가 더 많다.**
- **디바이스 메모리를 스텝마다 보는 눈**: **vLLM 에 없다.** `vllm:gpu_cache_usage_perc` 는 **블록 점유율**이지 바이트가 아니고, 메트릭 로거에 `mem_get_info`·`memory_reserved` 가 없다. 우리도 `/metrics` 에 없었다.

그런데 우리 `OOM_STUDY` 의 결론이 정확히 그 자리다:
- 이 기계의 캐싱 할당자는 churn 에서 해제 블록을 재사용하지 않고 새 페이지를 매핑한다 → **봐야 하는 건 `memory_reserved`**(예약−할당은 캐시가 아니라 잃은 메모리).
- **26 → 5 GiB 낙하가 4 초** → 감시 주기는 스크레이프가 아니라 스텝이어야 한다.
- earlyoom 은 **절대 6 GiB**(SIGTERM) / 4.5 GiB(SIGKILL)에서 **엔진을 1순위로** 쏜다.

그리고 오늘 13:47, 그 일이 실제로 났다(45차 §48): `mem avail 4,538 / 124,546 MiB (3.64%)`, 엔진에 SIGTERM 셋.
**엔진이 내보내던 어떤 숫자도 그게 오는 걸 보여줄 수 없었다.**

### 그래서 한 것 (45차 §50)

`/metrics` 가 상자의 메모리를 말한다 — vLLM 도 SGLang 도 안 내는 것들이다:

| 계열 | 왜 |
|---|---|
| `st:host_memory_available_bytes` | **earlyoom 이 실제로 보고 결정하는 그 숫자**(MemFree 아님) |
| `st:device_memory_reserved_bytes` | 이 기계에서 봐야 하는 값(할당이 아니라 예약) |
| `st:device_memory_reserved_peak_bytes` | 스크레이프 사이의 고수위. **4초 절벽을 15초 스크레이프가 못 보는 문제의 답** — torch 가 공짜로 들고 있어 스텝당 비용 0 |
| `st:device_memory_allocated_bytes` | 예약과의 간극이 곧 잃은 메모리 |
| `st:device_memory_free_bytes` / `_total_bytes` | 드라이버가 말하는 값 |

`/metrics` 는 없는 숫자에 대해 **줄을 빼지 스크레이프를 깨뜨리지 않는다**(CUDA 없는 상자, /proc 이 답 안 할 때).

## 3. 남은 숙제 (전부 장비가 필요하다)

1. **프리필 활성화 피크**와 **그래프·워크스페이스** 실측 → `budget.py` 의 두 빈 줄을 채운다. vLLM 은 이걸 `profile_run` 한 번으로 얻는다. 우리도 부팅에 같은 걸 넣을 수 있다(`RuntimeMemory.checkpoint` 가 이미 페이즈를 남긴다).
2. 그 값이 나오면 **KV 8.73 GiB 를 다시 정한다** — 검토의 계산으로는 50 GiB 대가 비어 있다.
3. 널 슬롯 247.2 MiB(§49)의 관문.
