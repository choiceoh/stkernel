# LMDeploy / TurboMind 에서 배워올 것 — 조사 (2026-09-12)

`InternLM/lmdeploy` `309d2b5`(2026-09-11, 하루 전)를 읽고 ST 와 대조했다. 읽은 곳:
`src/turbomind/engine/{README.md,scheduler.cc,engine.cc}`, `src/turbomind/{models,kernels}/`,
`docs/en/inference/turbomind.md`, `docs/en/advance/pytorch_new_model.md`.

고른 이유 둘: 지금까지 본 넷(vLLM·SGLang·TRT-LLM 파이토치 백엔드·Mooncake)은 **전부 파이썬 호스트**라
헌장 D7 이 고른 갈림길을 아무도 답하지 않았고, TurboMind 는 **C++ 엔진**이다. 그리고 뜻밖에 —
**선형 어텐션 상태를 접두사 캐시에 넣는 문제를 우리 말고 푼 유일한 곳**이었다.

---

## 1. 가져올 것 — 전부 "선형 어텐션 + 접두사 캐시" 한 자리에서 나온다

ST 의 접두사 캐시는 청크 경계에만 산다. KDA 의 상태 스냅샷이 거기에만 있기 때문이고, vLLM 대조는 그걸
*"모델이 강제하는 것이지 빈 곳이 아니다"* 라고 닫았다. **그 벽에 부딪힌 엔진이 하나 더 있고, 정책이 우리보다 많다.**

### 1.1 캐시는 도는 요청을 미루게 하지 못한다 — 입장이 두 층이다

TurboMind 의 입장은 **필수 층**과 **선택 층**으로 갈린다(`contracts.scheduler-admission`):

- **필수**: 그 요청이 포워드를 돌기 위해 반드시 있어야 하는 것 — 접두사 블록 + **frontier 하나**.
  못 맞추면 요청을 미루고 그 패스를 멈춘다.
- **선택**: 체크포인트 발행, fork 채우기. **모든 필수 포워드가 자리를 잡은 뒤에만** 돌고,
  **비활성 슬롯만** 회수하며, **안 맞으면 그냥 버린다** — *"never evicts active state and never defers a forward"*.

즉 **캐시를 남기는 일이 도는 일을 절대 밀어내지 못한다.** ST 의 D10 이 디스크에 대해 말하는 것을
(*"디코더는 디스크를 안 기다린다"*) 메모리에 대해 말한 것이다. ST 에서 이에 해당하는 자원은
**스냅샷 슬롯**이다 — `restore_begin` 은 `MemoryError("no snapshot is free")` 로 실패할 수 있고,
경계가 스냅샷을 쥔다. 우리는 `spill_low_water` 로 **미리** 뱉어 그 상황을 피하는데, 그건 예방이지 계약이 아니다.
*"안 맞으면 캐시 쪽이 진다"* 를 규칙으로 적으면 예방이 실패해도 도는 요청은 안전하다.

### 1.2 발행된 체크포인트는 보호 집합에서 뺀다

`invariants.protection-set`:

> Published block checkpoints are resume-time optimizations, not run-time state, and are deliberately
> excluded so they stay evictable: a high-priority sequence runs whenever memory fits its prefix blocks
> **+ one frontier** and may **reclaim its own prior checkpoints**.

요청이 자기가 남긴 체크포인트 때문에 못 도는 일이 없다. ST 는 §32 에서 등급 있는 자유 목록을 넣어
블록 쪽은 이 성질을 얻었지만, **스냅샷에는 같은 규칙이 없다.**

### 1.3 체크포인트 간격이 정책이고, 스케줄러가 포워드를 거기서 자른다

`cache_checkpoint_interval`(`CacheRegistry::checkpoint_min_interval`)이 **재귀 체크포인트 간격**이고,
스케줄러가 그걸 위해 **포워드의 끝을 자른다**:

> when a prompt-region forward would run past the checkpoint-due position
> `last_ckpt_pos + checkpoint_min_interval`, its end is truncated to the last block boundary in the
> admitted range ... so the full-block checkpoint can be taken there, with the remainder running in
> the next pass

그리고 중복 체크포인트는 **지우지 않고 evict-first 로 강등**한다(timestamp 0).

**ST 의 간격은 정책이 아니라 부수효과다** — 프리필 청크 크기(6,912)가 그대로 스냅샷 간격이다.
40차가 청크 6,912 의 출처를 추적하는 데 홀드 여럿을 쓴 바로 그 숫자이고(D2 의 근거), 그게 지금
**스냅샷을 얼마나 자주 남길지까지 정하고 있다.** 둘은 다른 질문이다.

---

## 2. 이미 맞는 것 (확인함)

| 항목 | TurboMind | ST |
|---|---|---|
| **persistent batch** | N 개의 미리 잡은 배치 슬롯, 요청이 빈 슬롯에 들어오고 끝나면 반납. 배치가 자동으로 늘고 준다 | 같다. `max_seqs` 슬롯 + 상태 슬롯 id + 배치 크기별 캡처 그래프 |
| KV 를 슬롯 풀로 | 매니저가 전부 할당하고 LRU 로 회수 — *"cache of KV caches"* | 같다(아레나 + `BlockPool`) |
| 발행 결정이 랭크마다 같아야 | *"a pure function of cross-rank-identical sequence attributes ... so it is consistent across ranks"* | 같은 이유로 우리 해시는 토큰 id 의 함수다 — 메시지 없이 합의한다 |

---

## 3. 의도적으로 다른 것 — D7, 이번엔 숫자로

| | |
|---|---:|
| TurboMind **엔진 전체**(요청 큐·게이트웨이·스케줄러·실행기·블록) | **5,548 줄 C++** (스케줄러만 1,585) |
| 같은 레포가 **나란히 유지하는 파이썬 엔진**(`lmdeploy/pytorch`) | **115,650 줄** |
| ST 의 대응물(`runner.py` + `scheduler.py` + `kv.py`) | ~1,300 줄 파이썬 |

**C++ 엔진을 가진 프로젝트가 파이썬 엔진을 20배 크기로 함께 유지한다.** 그 이유를 자기 문서가 적어 뒀다 —
`lmdeploy.pytorch` 는 *"designed to simplify the support for new models and the development of prototypes"*.
C++ 엔진의 값은 **모델을 붙이는 속도**이고, TurboMind 는 그래서 FasterTransformer 에서 물려받은 기능들을
*"dropped ... because of the difference in objectives"* 하고 좁게 간다.

ST 의 D7 은 이미 그 값을 쟀다: `idle`(런치 글루 + 호스트 스톨) = **15.9 ± 0.5 ms**, 프리필 스텝 고정비 214.7 ms 중이고,
디코드는 CUDA 그래프 재생이라 파이썬이 핫패스 밖이다 — **호스트 언어로 건드릴 수 있는 최대치가 스텝의 3% 미만**.
LMDeploy 는 그 3% 를 위해 두 엔진을 유지한다. **우리 결론이 바뀌지 않는다는 외부 증거**이고, I2(스케줄러·KV·스텝
루프를 평평한 배열로)가 왜 값싼 보험인지도 같이 보여 준다 — 들어낼 자리가 5.5k 줄이면 옮기는 것이 불가능하지 않다.

**축출된 대화를 어떻게 두느냐도 다르다.** TurboMind 는 희생된 시퀀스를 *"most compact form, i.e. token IDs"* 로
강등하고 다음에 부르면 **다시 계산한다** — *"infinite device memory"* 가 그 값이다. ST 는 NVMe 에 실물을 둔다.
우리에겐 그 선택을 정당화하는 숫자가 있다(D16: 128K 복귀 **44 s → 0.3 s**). 재계산은 44 초짜리 답이다.

---

## 4. 안 가져올 것

**INT8 KV 캐시** — 그쪽의 배치 확대 수단인데, 우리 KV 는 이미 NVFP4 세계이고 배치는 4 다.

**`cache_generation` 의 세 모드** — 생성된 블록을 캐시에 넣을지의 노브. 우리는 대화를 통째로 파킹하므로
같은 목적을 다른 곳에서 이미 이룬다.

**FasterTransformer 유래의 모델 계층** — LLaMa 계열에 특화된 가중치 레이아웃(원본 LLaMa 기준, transpose 하나 차이)은
우리 사전샤딩과 무관하다.

---

## 5. 그래서 다음에 잴 것

1.1~1.3 은 셋 다 **스냅샷 슬롯**이라는 한 자원을 향한다. ST 에는 그 자원에 대한 계약이 없다 —
`spill_low_water` 라는 예방과 `MemoryError` 라는 결과가 있을 뿐이다. 재야 할 것: **스냅샷이 모자라서
복원이 실패하거나 경계가 안 남는 일이 실제로 얼마나 일어나는가.** 지금은 세는 카운터도 없다.
