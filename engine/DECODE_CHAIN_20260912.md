# 디코드를 GPU 위에서 안 끊기게 — 구조 진단 (2026-09-12)

지시 넷(블록 단위 융합 · all_reduce 를 PDL 로 겹치기 · 하나로 이어진 decode chain · pending 비움 줄이기)을
`origin/main` 에 대조했다. 결과: **넷째는 오늘 다른 세션이 했고, 셋째는 절반 있고, 첫째·둘째는 창이 필요하다.**
그리고 **아무도 "왜 체인을 벗어났는지" 를 안 세고 있었다.**

---

## 0. 넷째(pending 비움)는 PR #671 이 했다 — 남은 건 "왜"

`pipeline.ready_for(seqs, slots)` 가 들어오면서 **독립적인 새 행은 살아 있는 readback 을 비우지 않고 합류**한다.
계량기 둘도 같이 왔다: `st:async_decode_steps_total`(앞서 돈 스텝), `st:sync_drain_steps_total`(비운 횟수).

**남은 빈 곳은 이유다.** "1만 번 비웠다" 는 정보가 아니다. 그중 9,800 이 `logprobs` 인지 `rows_churned` 인지에
따라 **정반대의 일**을 해야 한다 — 전자는 스케줄링 문제고, 체인 안쪽을 아무리 융합해도 안 줄어든다.

이 PR 이 더하는 것: **`st:decode_chain_exits_total{reason}`**
— `logprobs` · `grammar` · `seed` · `min_tokens` · `logit_bias` · `min_p` · 세 penalty · `rows_churned` · `no_pipeline`.

두 가지를 일부러 따로 센다:
- **`min_tokens` 는 그 행이 충분히 뽑으면 스스로 풀린다.** 나머지는 요청이 끝날 때까지 안 풀린다. 합치면 못 읽는다.
- **배치는 다 같이 앞서 가거나 아무도 못 간다**(`async_ready` 가 `all(...)`). `max_seqs=4` 에서
  **logprobs 하나가 나머지 셋을 끌고 나간다.** 그래서 이유를 그 행 하나의 일이 아니라 **스텝 전체의 일**로 센다.

---

## 1. 셋째(decode chain)는 절반 있다 — 없는 건 **가운데**

`profiles/glm53/pipeline.py` 가 이미 그 체인이다:

> target graph → sampler → commit → masked observe → proposal → 다음 스텝의 ids

**결과만** 이벤트 뒤로 호스트에 건너간다. 동기 경로(`adapter.decode`)와 대조하면 값이 분명하다 — 동기 경로는
행마다 `propose(...).tolist()`, 끝에 `sampled.tolist()`. 네 행이면 **스텝당 5번의 디바이스→호스트 왕복**이고
매번 스트림이 마른다. 비동기 경로엔 없다.

**그런데 체인 가운데가 캡처 밖이다.** `launch` 한 스텝(확률 경로 = 프로덕션 기본):

| | 상태 |
|---|---|
| `decode_graphs.run_device` | **캡처된 그래프** |
| `comm.all_gather(local)` | eager (집합 통신) |
| `distribution_batch` → `sampler_rows` | eager |
| `note_ceilings` | eager |
| `block_verify_batch` | eager, 커널 여럿 |
| 커밋 + 행 전진 | **런치 하나** — `kernels/decode_commit.advance` (CUDA일 때. CPU 경로만 `commit_batch` + 원소별 여섯) |
| `caches.stage_boundaries` | eager |
| `positions = ctx_before.view + arange` | eager |
| `observe_rows` | **캡처된 그래프** |
| `propose_rows` | **캡처된 그래프** |

**커밋 자리는 오늘 이미 접혔다** — `engine/kernels/decode_commit.py` 가 커밋과 여섯 개의 원소별 갱신을
한 런치로 합쳤고 `pipeline.py:272` 가 CUDA 경로에서 그걸 쓴다. 남은 eager 는 위 표의 나머지,
특히 **`all_gather` → `distribution_batch` → `note_ceilings` → `block_verify_batch`** 네 덩어리다.
지시의 *"각각 별도 그래프로 이어 붙이는 수준에서 더 나아가"* 가 가리키는 자리가 그것이다.

**왜 나머지가 캡처 밖인지는 코드에 안 적혀 있다.** 후보 둘, **둘 다 가설**:
`block_verify_batch` 가 호스트 `torch.Generator` 를 쓴다(TRT-LLM 대조 §3: 그쪽은 캡처 아래에선
seed/offset **디바이스 텐서**가 필수라고 적었다), 그리고 `all_gather` 가 집합 통신이다.
**창에서 확인할 것이지 지금 단정할 것이 아니다.**

---

## 2. 레이어 루프 — 첫째·둘째가 사는 곳

`net.py` 의 한 레이어:

```
_hc_post_pre(L, ..., "attn")     # mHC
_dsa/_kda(L, ...)                # 끝이 all_reduce(o_proj)
_hc_post_pre(L, ..., "ffn")      # mHC  <- all_reduce 직후의 x 를 바로 먹는다
_moe/_dense(L, ...)              # 끝이 all_reduce(down)
```

**레이어마다 리덕션이 둘이고, 둘 다 곧바로 mHC 입력이 된다.** 지시 둘째가 가리키는 자리가 정확히 여기다.
PDL 배관은 이미 있다 — `kernels/{deep_gemm,causal_conv,dense/mhc,mhc/tilelang_kernels}.py` 가 쓴다.
**리덕션과 mHC 사이에는 없다.**

지시 첫째(블록 단위 융합)도 같은 줄이다: 네 호출이 각각 `x`·`res`·`post`·`comb` 를 BF16 로 내놓고 다시 읽는다.
`res` 는 `[N, hc, hidden]` 이라 가장 크다.

---

## 3. 순서와 관문

1. **이유 계량기**(이 PR). 2~4 의 값이 전부 체류율에 곱해지고, 체류를 막는 게 무엇인지는 이유가 있어야 안다.
2. **체인 가운데를 캡처**(셋째의 남은 절반): `all_gather` · `distribution_batch` · `note_ceilings` ·
   `block_verify_batch`. 커널을 새로 안 써도 되고, 위 가설 둘만 확인하면 된다.
3. **리덕션 ↔ mHC PDL**(둘째). 배관이 있으니 자리만 잇는다. 레이어마다 두 번.
4. **블록 단위 융합**(첫째). 가장 크고 가장 늦다 — 2~3 뒤라야 무엇이 남는지 보인다.

**2~4 는 GPU 창이 필요하고, 프로덕션 GLM-5.3 이 곧 이 엔진이라 창은 다운타임이다.**
창에서 잴 것: (a) `st:async_decode_steps_total / st:steps_decode_total` 체류율, (b) `exits` 의 1위 이유,
(c) 확률 경로 한 스텝의 커널 런치 수, (d) 리덕션 뒤 mHC 가 실제로 기다리는 시간.

**병렬 세션 주의**: 이 영역은 오늘 붐빈다 — #671(파이프라인 처리량), `decode_commit.py`, `test_engine_decode_fusion`,
`test_engine_oneshot_integer` 가 전부 오늘 들어왔다. 2~5 를 잡기 전에 `git log origin/main` 을 다시 볼 것.
