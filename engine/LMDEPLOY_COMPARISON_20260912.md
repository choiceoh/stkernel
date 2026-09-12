# LMDeploy / TurboMind 에서 배워올 것 — 조사 (2026-09-12)

`InternLM/lmdeploy` `309d2b5`(2026-09-11, 하루 전)를 읽고 ST 와 대조했다. 읽은 곳:
`src/turbomind/engine/{README.md,scheduler.cc,engine.cc}`, `src/turbomind/{models,kernels}/`,
`docs/en/inference/turbomind.md`, `docs/en/advance/pytorch_new_model.md`.

고른 이유 둘: 지금까지 본 넷(vLLM·SGLang·TRT-LLM 파이토치 백엔드·Mooncake)은 **전부 파이썬 호스트**라
헌장 D7 이 고른 갈림길을 아무도 답하지 않았고, TurboMind 는 **C++ 엔진**이다. 그리고 뜻밖에 —
**선형 어텐션 상태를 접두사 캐시에 넣는 문제를 우리 말고 푼 유일한 곳**이었다.

**결론부터**: 그쪽 재귀 체크포인트 계약은 세 조각인데 **ST 가 이미 둘을 갖고 있다**(코드에서 확인).
남은 하나가 진짜 빈 곳이고, 우리 숫자를 넣어 보면 그냥 빈 곳이 아니라 **이미 일어나고 있는 일**이다.

---

## 1. 가져올 것 — 체크포인트 간격이 정책이어야 한다

`cache_checkpoint_interval`(`CacheRegistry::checkpoint_min_interval`)이 재귀 체크포인트의 **최소 간격**이고,
스케줄러가 그걸 위해 **포워드의 끝을 자른다**:

> when a prompt-region forward would run past the checkpoint-due position
> `last_ckpt_pos + checkpoint_min_interval`, its end is truncated to the last block boundary in the
> admitted range ... so the full-block checkpoint can be taken there

그리고 간격 안에 든 중복 체크포인트는 지우지 않고 **evict-first 로 강등**한다(timestamp 0).

**ST 에는 간격이라는 개념이 없다.** `runner._marks()` 는 프리필 스텝 안의 **모든** 블록 경계를 훑고,
경계마다 `take_snapshot()` 을 한다 — `boot.py:320` 의 주석 그대로 *"boundaries = every 768 block"*.
간격은 정책이 아니라 **블록 크기의 부수효과**다.

숫자를 넣으면 이건 이론이 아니다:

| | |
|---|---:|
| 스냅샷 슬롯 (`boot.PREFIX_SNAPSHOTS`) | **96** (랭크당 ~45 MiB) |
| 블록 경계 간격 (`facts.BLOCK`) | **768** |
| 128K 프롬프트 하나의 경계 수 | **170** |

**프롬프트 하나가 슬롯 전체보다 1.8배 많은 체크포인트를 요구한다.** `take_snapshot()` 은 빈 게 없으면
`_victim()` 을 골라 `_fade()` 하므로 — **긴 프롬프트가 프리필 도중 자기 앞부분 경계를 자기 뒷부분으로 밀어낸다.**
페이드된 경계는 티어에 있으면 블록을 지키지만, 계산해서 만든 체크포인트 74개분이 그 패스에서 버려진다.

간격을 2블록(1,536)으로만 두면 85개로 **들어맞는다**. 지금 이 선택은 아무도 한 적이 없다.

**`_checkpoint()` 도 같은 자리**: 프리필 스텝 끝의 경계를 무조건 찍는다. 간격 정책이 생기면 둘 다 그걸 본다.

**세는 것이 없다**(grep 으로 확인). `_marks` 의 `break`, `_checkpoint` 의 `return`, `_fade` — 스냅샷이
모자라서 경계를 못 남긴 일은 **카운터도 게이지도 없다**. 가져올 것의 절반은 간격이고 절반은 이 계량기다.

---

## 2. 이미 맞는 것 (코드에서 확인함)

**TurboMind 의 2단 입장을 ST 는 세 자리에서 이미 지키고 있다.** 그쪽 문장:

> the optional tier ... **never evicts active state and never defers a forward**

| 자리 | ST 의 코드 | 하는 일 |
|---|---|---|
| 프리필 중 경계 | `runner._marks()` | 슬롯 없으면 **`break`** — 나머지 경계는 그냥 안 남긴다 |
| 스텝 끝 경계 | `runner._checkpoint()` | 슬롯 없으면 **`return`** (*"this one goes uncached"*) |
| 티어에서 복원 | `serve.py:1553` | `MemoryError` 를 잡아 **프리필로 되돌린다** (*"no snapshot / no blocks / no tier: prefill it instead"*) |

앞의 둘이 그쪽 **선택 층**(캐시가 도는 일을 절대 못 미룬다), 셋째가 **필수 층**(못 맞추면 이 요청만 미룬다)이다.
**D10 의 "디코더는 디스크를 안 기다린다" 가 메모리 쪽에서도 이미 성립한다.**

**발행된 체크포인트가 보호 집합에 없는 것**도 이미 맞는다. 그쪽:
*"Published block checkpoints are ... deliberately excluded so they stay evictable"*.
ST 의 `take_snapshot()` 이 바로 그것 — 빈 슬롯이 없으면 살아 있는 경계에서 하나를 **뺏는다**(`_victim` → `_fade`).
PR #609 의 등급 있는 자유 목록이 블록 쪽에서 같은 성질을 준다.

그 밖에:

| 항목 | TurboMind | ST |
|---|---|---|
| **persistent batch** | N 개의 미리 잡은 슬롯, 요청이 빈 슬롯에 들어오고 끝나면 반납 | 같다. `max_seqs` 슬롯 + 상태 슬롯 id + 배치 크기별 캡처 그래프 |
| KV 를 슬롯 풀로 | 매니저가 전부 할당하고 LRU 로 회수 — *"cache of KV caches"* | 같다(아레나 + `BlockPool`) |
| 발행 결정이 랭크마다 같아야 | *"a pure function of cross-rank-identical sequence attributes"* | 같은 이유로 우리 해시는 토큰 id 의 함수다 — 메시지 없이 합의한다 |

---

## 3. 의도적으로 다른 것 — D7, 이번엔 숫자로

| | |
|---|---:|
| TurboMind **엔진 전체**(요청 큐·게이트웨이·스케줄러·실행기·블록) | **5,548 줄 C++** (스케줄러만 1,585) |
| 같은 레포가 **나란히 유지하는 파이썬 엔진**(`lmdeploy/pytorch`) | **115,650 줄** |
| ST 의 대응물(`runner.py`+`scheduler.py`+`kv.py`) | **1,445 줄** 파이썬 |

**C++ 엔진을 가진 프로젝트가 파이썬 엔진을 20배 크기로 함께 유지한다.** 이유를 자기 문서가 적어 뒀다 —
`lmdeploy.pytorch` 는 *"designed to simplify the support for new models and the development of prototypes"*.
C++ 엔진의 값은 **모델을 붙이는 속도**이고, TurboMind 는 그래서 FasterTransformer 에서 물려받은 기능들을
*"dropped ... because of the difference in objectives"* 하고 좁게 간다.

ST 의 D7 은 이미 그 값을 쟀다: `idle`(런치 글루 + 호스트 스톨) = **15.9 ± 0.5 ms**, 프리필 스텝 고정비 214.7 ms 중이고,
디코드는 CUDA 그래프 재생이라 파이썬이 핫패스 밖이다 — **호스트 언어로 건드릴 수 있는 최대치가 스텝의 3% 미만**.
LMDeploy 는 그 3% 를 위해 두 엔진을 유지한다. **우리 결론이 바뀌지 않는다는 외부 증거**이고, I2(평평한 배열)가
왜 값싼 보험인지도 같이 보여 준다 — 들어낼 자리가 5.5k 줄이면 옮기는 것이 불가능하지 않다.

**축출된 대화를 어떻게 두느냐도 다르다.** TurboMind 는 희생된 시퀀스를 *"most compact form, i.e. token IDs"* 로
강등하고 다음에 부르면 **다시 계산한다** — *"infinite device memory"* 가 그 값이다. ST 는 NVMe 에 실물을 둔다.
우리에겐 그 선택을 정당화하는 숫자가 있다(D16: 128K 복귀 **44 s → 0.3 s**). 재계산은 44 초짜리 답이다.

---

## 4. 안 가져올 것

**INT8 KV 캐시** — 그쪽의 배치 확대 수단인데, 우리 KV 는 이미 NVFP4 세계이고 배치는 4 다.

**`cache_prompt`/`cache_generation` 의 모드 노브** — 무엇을 캐시에 넣을지를 요청이 고르게 하는 것.
우리는 대화를 통째로 파킹하므로 같은 목적을 다른 곳에서 이미 이룬다.

**FasterTransformer 유래의 모델 계층** — LLaMa 계열에 특화된 가중치 레이아웃은 우리 사전샤딩과 무관하다.

---

## 5. 그래서 다음에 잴 것

§1 은 **재 보기 전에는 고치지 말 것**이다. 필요한 계량기 둘, 둘 다 지금 없다:

1. `_marks`/`_checkpoint` 가 슬롯이 없어 경계를 포기한 횟수 (`st:prefix_snapshot_denied_total`).
2. `_fade` 가 **살아 있는 프리필의 자기 경계**를 밀어낸 횟수 — §1 의 170 대 96 이 실제로 일어나는지.

둘을 켜고 프로덕션 형상에서 하루 보면 `checkpoint_min_interval` 의 기본값이 숫자로 나온다.
지금 고르면 그건 추측이다.
