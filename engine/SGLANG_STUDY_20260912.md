# SGLang 전면 대조 — 남은 영역 (2026-09-12)

`sgl-project/sglang` **`e91c948`** 체크아웃(`python/sglang/srt`, 1,852 파일 · 791k 줄)을 ST 가 실제로 내린 결정에
닿는 영역만 읽었다. vLLM 대조(`VLLM_COMPARISON_20260912.md`)와 같은 방식이다.

**이미 본 둘은 여기 없다**: 접두사 캐시 블록 수명은 `SGLANG_PREFIX_20260912.md`(원장 §32 보충), 구조화 출력은
`SGLANG_COMPARISON_20260912.md`(원장 §33). **소스 주의**: 구조화 출력 쪽은 PyPI 휠 `0.5.19` 를 읽었는데,
`mamba_radix_cache.py`·`evict_policy.py`·`adaptive_spec_params.py` 는 **그 휠에 없다**. 다음에 대조할 때는 체크아웃으로.

**결론부터: 가져올 것 넷. 셋은 스케줄링, 하나는 계측이고, 전부 우리가 "아직 안 재 봤다"고 적어 둔 자리다.**

## 1. 가져올 것

### 1.1 대기열 **안에서** 접두사 중복을 거른다 (`schedule_policy._compute_prefix_matches`)

SGLang 은 대기 큐 전용 **임시 라딕스 트리**를 매 스케줄마다 새로 지어, 큐 안의 요청끼리 접두사를 맞춰 본다.
캐시 적중이 짧은(`≤ IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`) 요청 여럿이 **서로** 같은 접두사를 공유하면
**하나만 보내고 나머지는 그 라운드에서 강등**한다(`temporary_deprioritized`). 이유를 주석이 적어 뒀다: 오래 도는
엔진에서 `"the"` 같은 짧은 접두사는 흔해서, 임계를 0 으로 두면 아무거나 다 걸린다.

우리 `shared_ahead`(45차 §23 B)는 **이미 도는 프리필**하고만 비교한다. 대기열에 같은 접두사 요청이 셋 있으면,
하나가 *돌기 시작한 뒤에야* 나머지가 양보한다 — **같은 스텝에 둘이 입장하면 둘 다 프리필한다.** `MAX_SEQS=4` 라
드문 일이 아니고, 128K 프롬프트 둘이면 그 한 번이 비싸다. 우리는 이미 요청마다 체인을 들고 다니므로(문에서 한 번
해시, 원장 §23) 큐 안의 비교는 **해시 비교뿐**이다 — 라딕스 트리를 지을 필요도 없다.

### 1.2 스타베이션을 벽시계가 아니라 **처리한 토큰**으로 센다 (`_sort_by_hrrn`)

```
ratio = 1 + wait_sec / est_prefill_time = 1 + (processed_tokens - arrival_processed_tokens) / uncached
```

스케줄러가 프리필 토큰 누적 카운터를 들고, 요청이 큐에 들어올 때 그 값을 찍어 둔다. 나이가 **엔진이 실제로 한 일**로
매겨지므로 부하와 무관하다.

우리 D10 의 밸브는 `max_wait_s = 20.0` **벽시계** 하나다. GPU 가 느려지거나 긴 프롬프트가 줄줄이 들어오면
**대기열 전체가 동시에 만료된다** — 밸브가 열리는 게 아니라 터진다. 토큰 기반이면 그런 동시 만료가 없다.
D10 을 바꾸자는 게 아니라, 밸브의 **눈금**을 바꾸자는 것이다.

### 1.3 드래프트 길이를 관측으로 정한다 (`adaptive_spec_params.AdaptiveStepSlot`)

```
target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)     # 관측된 수락 길이보다 한 스텝 더 탐침
```

배치 크기별 슬롯으로 나누고(`_route(batch_size)`), CUDA 그래프가 캡처된 BS 로 패딩해서 라우팅하며, 전환은
**그래프가 존재하는 step 값으로만** 한다(`cuda_graph_bs_for_step`). 상태 교체는 원자적이다.

우리 k 는 **고정 5** 이고, vLLM 대조 §1.3 이 *"우리 `max_seqs` 는 4 라 위험이 작지만, 재 본 적이 없다"* 라고 적어
뒀다. 필요한 관측은 **이미 나가고 있다** — `st:spec_accepted_per_step_total` 이 수락 개수의 전체 분포다(§1-b 정정).
EMA 는 공짜다. **진짜 비용은 그래프 캡처**다: 부팅 87 초의 75% 가 캡처인데(`BOOT_TIME_STUDY_20260911.md`),
k 후보를 둘로 늘리면 그만큼 늘어난다. 그래서 이건 "k 를 몇 개 캡처할 값어치가 있나" 라는 **측정 먼저**의 문제다.

### 1.4 그래프가 먹은 메모리를 **단계별로** 센다 (`model_executor/graph_memory_usage.py`)

```python
GRAPH_MEMORY_USAGE_KEYS = ("prefill", "decode", "target_verify",
                           "draft_prefill", "draft_decode", "draft_extend")
```

캡처가 쓴 메모리를 여섯 단계로 나눠 합산하고 스키마를 고정해 둔다. 우리는 `rec.phase("capture decode")` 로 **시간은**
재지만 **메모리는 나눠 재지 않는다** — 부팅 87 초의 75% 가 캡처인데, 그 캡처들이 각각 얼마를 가져갔는지는 모른다.
`GB10 OOM 물리`(earlyoom 절대 6 GiB, *예약 − 할당 = 잃은 메모리*)를 생각하면 이건 실제로 빈 곳이다. 그리고 1.3 의
"k 후보를 늘릴 값어치가 있나" 는 **시간과 메모리 둘 다**를 알아야 답할 수 있다.

## 2. 이미 맞는 것 (확인함)

| 항목 | SGLang | ST |
|---|---|---|
| 호스트가 디바이스보다 앞서 감 | `FutureMap` — 아직 안 나온 토큰을 자리표로 두고 다음 스텝을 짓는다(`overlap_utils`) | 같다(비동기 디코드, `depth`/`async_steps`, PR #585) |
| 청크 프리필 | `chunked_prefill_size` | `chunk_for(...)`, D9 의 정렬 규칙까지 |
| 페널티 범위 | repetition = 프롬프트∪출력, presence/frequency = 출력만 (`penaltylib/`) | 같다 |
| 접두사 캐시 · 구조화 출력 | — | 별도 문서 둘 |

## 3. 의도적으로 다른 것

**선점(retract)과 `new_token_ratio`.** SGLang 은 입장할 때 요청이 **앞으로 쓸 토큰을 비율로 추정**하고, 메모리가
모자라면 디코딩 중인 요청을 뒤에서부터 물린다(`retract_decode`). 선점이 나면 비율을 올려 보수적으로 굴다가
`decay_step()` 으로 서서히 되돌린다. 우리는 **입장 때 지평선 전체를 예약하고 선점이 없다** — D3 의 직접 귀결이고,
`kv.py` 가 그 이유를 적어 뒀다. `MAX_SEQS=4` · KV 24 GiB · 128K 컨텍스트면 추정의 여지가 애초에 작다.
(추정을 쓰면 `max_new=4096` 을 걸고 50 토큰만 쓰는 요청의 낭비를 줄일 수 있다 — 우리 상자에서는 그 낭비보다
선점 기계의 값이 더 크다는 판단이다.)

**혼합 청크**(`enable_mixed_chunk`: 한 포워드에 프리필 청크와 디코드를 섞음) — D9 가 균질 스텝을 요구하고,
39차에서 **순차가 인터리빙보다 빨랐다**는 실측이 근거다. vLLM 때와 같은 결론.

**대기열 정책 일곱**(LPM · DFS_WEIGHT · HRRN · FCFS · LOF · RANDOM · ROUTING_KEY). 우리는 FCFS + 양보한 것
되돌리기(`_reorder_waiting`)뿐이고, 노브로 일곱을 여는 것은 D11 위반이다. 1.2 는 **정책 교체가 아니라 밸브 눈금**이라
다른 얘기다.

## 4. 안 가져올 것

- **PD 분리**(`disaggregation/`, 29.7k 줄) — 프리필과 디코드를 다른 노드에 두는 구조. 우리는 한 상자 네 랭크다(D1).
- **빔 서치**(`beam_search/`, `beam_retraction_order`) — 서빙 표면에 없다.
- **LoRA · elastic EP · weight_sync · checkpoint_engine** — D5 의 범위 밖.
- **어텐션 백엔드 선택**(`layers/attention/`, 72k 줄, 백엔드 수십) — 우리는 D8/D11 로 **박아** 쓴다. 고르는 기계가
  없는 것이 결정이지 빈 곳이 아니다.

## 5. 찾으러 갔다가 **없다고 확인한 것** — 캡처를 싸게 하는 기계

부팅의 75% 가 캡처이므로 `model_executor/` 에 레버가 있기를 기대하고 봤다. **없다.**
`get_batch_sizes_to_capture` 는 설정으로 받은 BS 목록을 거르기만 한다 — 어텐션 TP 정렬로 한 번
(`bs * alignment_width % mul_base == 0`), `max_running_requests` 로 한 번, 정렬·중복 제거. 캡처를 지연시키거나
병렬로 하거나 건너뛰는 장치는 없고, 순서도 그냥 오름차순이다.

분리 하나는 있다: **`capture_bs` 와 `compile_bs` 가 다르다** — torch.compile 은 `torch_compile_max_bs` 이하만 하고
CUDA 그래프는 전체를 캡처한다. 우리는 torch.compile 경로가 없으므로(D8: CuTe-DSL · Triton · CUDA) 해당 없다.

즉 **캡처 비용은 저쪽도 그냥 낸다.** 우리 87 초의 75% 를 줄이는 답은 여기 없고, 우리 쪽 `BOOT_TIME_STUDY` 의
레버 목록이 여전히 그 자리의 전부다.
