# TensorRT-LLM 에서 배워올 것 — 조사 (2026-09-12)

NVIDIA/TensorRT-LLM `a76e0b2`(2026-09-11, 하루 전)의 파이썬 백엔드(`tensorrt_llm/_torch`)를 sparse clone 으로
읽고 ST 와 대조했다. TRT engine 빌드 경로가 아니라 **PyTorch 백엔드**가 우리와 비교 가능한 쪽이다.

**결론부터: 가장 큰 것은 "가져올 것" 이 아니라 "이미 상자 안에 있었다" 이고, 그게 오늘 내가 쓴 커널을 무효로 만든다.**

---

## 1. 가져올 것

### 1.1 샘플러 커널은 쓸 필요가 없었다 — FlashInfer 가 이미 이미지 안에 있다 (가장 크다)

TRT-LLM 의 샘플러는 백엔드 셋으로 갈린다(`_torch/pyexecutor/sampler/ops/{vanilla,flashinfer,triton}.py`).
**빠른 경로는 자기 커널이 아니라 `flashinfer.sampling.*` 을 부르는 것**이다. 그리고 그 flashinfer 는
**우리 ST 이미지에 이미 들어 있다**(`0.6.18.dev20260819` — b12x 가 쓰고 있어서 들어온 것):

| 우리가 손으로 쓴 것 | 이미 있던 것 |
|---|---|
| `kernels/sampler.py` 의 분포 모드 | `flashinfer.sampling.top_p_renorm_probs` / `top_k_renorm_probs` |
| `kernels/sampler.py` 의 id 모드 | `top_k_top_p_sampling_from_probs` / `..._from_logits` |
| `base/sampler.block_verify_batch` | `chain_speculative_sampling` |
| `process_logits` 의 마스킹 | `top_k_mask_logits` |

**실측**(ST 이미지 안, 24행 × 154,880, top_p 0.9, 같은 실행):

| | 우리 | FlashInfer |
|---|---:|---:|
| 분포(스펙 경로가 쓰는 것) | **726.1 µs** | **532.4 µs** |
| id 만 | **733.1 µs** | **417.4 µs** |

그리고 **정확성은 무승부다** — 정렬 참조(`_rows_by_sorting`)와 대조하면 둘 다 **최악 행에서 0 토큰 차이**,
최대 확률차는 우리 3.73e-09, flashinfer 2.98e-08. 우리가 하루를 들여 얻은 "정렬 없이 정확히" 를
그쪽은 이미 갖고 있고 1.4~1.8× 빠르다.

**채택을 막는 것이 있는지 확인했다 — 없다**:
- CUDA 그래프 캡처: `top_p_renorm_probs`, torch 생성기 샘플링, seed/offset 샘플링 **셋 다 캡처됨**(실측).
- 행마다 다른 top_p: 텐서를 받는다(실측, `Union[torch.Tensor, float]`).
- 라이선스·의존성: 새 의존성이 아니다. 이미 이미지에 있고 b12x 가 임포트한다.

**남는 차이는 둘뿐이고 둘 다 작다**: (a) flashinfer 는 `probs` 를 받으므로 softmax 를 먼저 해야 한다
(우리는 그 패스를 융합했다 — 실측 125.7 µs, 그래도 합이 더 빠르다), (b) greedy 행과 `decodable` 컷은
따로 다뤄야 한다(ST 는 이미 캡처된 argmax 로 greedy 를 따로 다룬다).

**그래서 제안**: `kernels/sampler.py` 를 flashinfer 호출로 갈아타고, 우리 커널은 **참조 옆에 남기거나 지운다.**
`base/sampler.rows()` 가 이미 "CUDA 면 커널, 아니면 정렬 참조" 로 갈리는 모양이라 자리는 그대로다.
**단, 드롭인은 아니다** — 동점 정책과 `valid` 컷을 우리 스윕(1,800 케이스)으로 다시 통과시켜야 한다.

### 1.2 `SpeculationGate` — 수용률이 떨어지면 스펙을 끈다 (82줄)

`_torch/speculative/speculation_gate.py`: 최근 N 회의 **진짜 수용률** 롤링 평균이 임계 아래로 내려가면
**스펙 디코딩을 영구 비활성**한다. O(1) 갱신, deque 하나.

ST 에는 이 문이 없다. 그런데 **입력은 이미 있다** — `adapter.accepted_per_step` 과
`st:spec_accepted_per_step_total`. 드래프터가 헛도는 워크로드(코드·반복 없는 긴 생성)에서 스펙은
순수 오버헤드이고, 지금 우리는 그걸 끌 방법이 없다.

### 1.3 백엔드를 고르는 자리를 둔다

`ops/{vanilla,flashinfer,triton}.py` + `sampler_strategy.py` 의 전략 층. **vanilla 는 그들도 정렬 + multinomial** 이다
— 즉 "정렬 없는 top-k/top-p" 는 TRT-LLM 이 쓴 게 아니라 flashinfer 를 부르는 것이다. 우리 `rows()` 의
`is_cuda` 분기가 같은 자리이고, 거기에 셋째를 넣으면 된다.

---

## 2. 이미 맞는 것 (확인함)

| 항목 | TRT-LLM | ST |
|---|---|---|
| **드래프트 길이** | `suggest_spec_config`: `max_draft_len = 5 if max_batch_size <= 4 else 3` | **`max_seqs=4`, `spec_k=6`.** ~~NVIDIA 휴리스틱이 외부 확인~~ — **철회(원장 45차 §65)**. 그건 일반 기본값이고 우리 수용률은 일반이 아니다. 현재 ST 프로파일은 k=6으로 고정한다. **k 가 천장이다** |
| 결정성이 계약 | 모든 flashinfer 호출에 `deterministic=True`. 주석이 이유를 적어 뒀다 — *"the default collect pass races"* | 우리는 네 랭크가 같은 토큰을 골라야 해서 결정성이 더 강한 계약이다. **flashinfer 로 갈아탈 때 이 플래그는 필수다** |
| 수용 통계 | 위치별 + confidence 보정 히스토그램(`accept_stats.py`) | `st:spec_accepted_per_step_total` 이 수용 **개수의 분포**라 위치별 생존 곡선을 포함한다(vLLM 대조 §1-b 정정) |
| 페널티·min_p·token ban 이 별도 모듈 | `penalties.py`(1,068줄), `token_ban.py`(708줄) | `process_logits` 하나. 규모가 다를 뿐 순서는 같다 |
| DFlash 드래프터 | `speculative/dflash.py`(1,338줄) **1차 지원** | 우리 드래프터가 DFlash2 다. **NVIDIA 가 같은 것을 1차로 지원한다는 사실 자체가 그 선택의 외부 확인**이다 |

---

## 3. 의도적으로 다른 것

**캡처 아래의 랜덤성.** 그들은 `seed`/`offset` **디바이스 텐서**를 "CUDA 그래프 아래에서는 필수" 로 적고
호스트 `torch.Generator` 는 eager 전용이라고 쓴다. 우리는 호스트 생성기로 행마다 uniform 하나를 뽑아
텐서로 넘기고, **그게 캡처된다**(실측). 우리 쪽이 요청별 시드(`gens[seq]`)를 다루기 쉽다 — 행마다 다른
생성기에서 뽑아도 커널은 숫자만 본다.

**엔진 빌드가 없다.** TRT engine AOT 컴파일은 D2(모양 계약)와 정신은 같지만 우리 규모에는 과하다.
우리의 등가물은 CUDA 그래프 사다리다.

**beam search 를 안 서빙한다.** 그쪽 `beam_search.py` 1,609줄 + `sampler.py` 2,894줄. 우리 문은 빔을 안 받는다.

---

## 4. 안 가져올 것

**`dflash.py` 를 이식하지 않는다** — 우리 드래프터는 이미 돌고 그쪽은 `MambaHybridCacheManager`·
`trtllm-gen` 배관에 묶여 있다. **읽을 것**이지 가져올 것이 아니다.

**`auto_heuristic.py` 의 ngram 폴백** — `spec_mode == AUTO` 일 때 n-gram 으로 떨어지는 경로. 우리는
드래프터가 하나뿐이고 D3 가 폴백을 금지한다.

**`penalties.py` 의 규모** — 1,068줄이 배치 전체를 다루는 구조인데, `max_seqs=4` 에서 우리 증분 History
(vLLM 대조 §1.1, 456 ms → 0.130 ms)가 이미 그 문제를 다르게 풀었다.

---

## 5. 이 조사가 스스로에 대해 말하는 것

오늘 샘플러 커널을 쓰기 전에 **이 조사를 먼저 했어야 했다.** 상자 안에 이미 있는 것을 확인하는 데
든 시간은 `docker run ... python3 -c "import flashinfer.sampling"` 한 줄이었다. 커널은 정확하고
(정렬 참조와 0 토큰 차이) 배운 것도 있지만(비트 패턴 임계값 탐색, 그리고 그 과정에서 잡은 실제 버그 둘),
**1.4~1.8× 느린 것을 하루 들여 다시 만든 것**이 정직한 요약이다.

규칙으로 올릴 것: **커널을 쓰기 전에 이미지 안의 패키지를 먼저 grep 한다.** `engine/kernels/SOURCES.json` 이
무엇을 들여왔는지는 적지만, **무엇이 이미 있는지**는 아무 데도 안 적혀 있다.
