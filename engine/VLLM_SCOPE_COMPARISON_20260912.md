# 구현 범위 안에서의 vLLM 대조 (2026-09-12)

운영자 "구현 범위 내에서 vllm과 st엔진 비교". 같은 상자의 프로덕션 vLLM
(`glm53:v13-b12x-it`, `0.1.dev20051+g487ecf187`, GLM5Next 지원이 들어간 포크)의 **소스를 꺼내**
ST 코드와 파일 단위로 맞대 봤다. `VLLM_COMPARISON_20260912.md` 가 *"가져올 것"* 이라면 이 문서는
*"지금 있는 것끼리"* 다 — **ST 가 구현한 자리만** 세고, 우리가 아예 안 만든 것은 "뒤처짐" 이 아니라
**범위 밖**으로 따로 적는다.

규모 (파이썬 줄 수, 주석 포함):

| | ST | vLLM (v1 해당 영역) |
|---|---:|---:|
| 엔진 전체 | 161 파일 **80,671** | (전체는 비교 대상 아님) |
| 문(OpenAI 표면) | 2,842 (`base/serve.py`) | 13,930 (`entrypoints/openai`) |
| 스케줄러 + 스텝 루프 | 1,003 | 3,925 (`v1/core/sched`) |
| KV 블록 + 접두사 | 929 | 11,907 (`v1/core`) |
| KV 오프로드 | 714 | 11,459 + 2,008 (`v1/kv_offload`, `simple_kv_offload`) |
| 샘플러 | 583 | 4,620 (`v1/sample`) |
| 구조화 출력 | 280 | 2,507 (`v1/structured_output`) |
| 스펙 디코딩 | 626 | 6,062 (`v1/spec_decode`) |
| 그래프 캡처 | 823 | (`v1/cudagraph_dispatcher` + 컴파일 스택) |
| 커널 | 60,024 (`engine/kernels`) | 47,951 (`v1/attention`) + 백엔드 |

**줄 수는 기능이 아니다.** vLLM 은 모델 수백 개 × 하드웨어 여러 세대 × 백엔드 여러 개를 덮고,
ST 는 **GB10 한 상자 · TP=4 · GLM-5.3 · NVFP4** 하나를 덮는다(D1·CHARTER). 아래는 그 하나 안에서의 비교다.

---

## 요약

| 영역 | 판정 |
|---|---|
| 디토크나이저 | **ST 앞섬** — 같은 러스트 스트림 + 복구 둘에서 우리가 토큰을 안 잃는다 |
| 페널티 | **ST 앞섬** — 증분 카운트(128K 스텝 456 ms → 0.130 ms). vLLM 은 자기 코드를 "quite inefficient" 라 적어 둠 |
| 구조화 출력(적용 경로) | **ST 앞섬** — 스텝 하나의 비트마스크, 죽은 위치는 게더조차 안 함 |
| 블록 수명 | **동등** — vLLM 의 규칙(해시 단 free 리스트·등급·꼬리부터)을 45차 §32 에서 그대로 |
| 접두사 캐시(하이브리드) | **동등, 설계가 다름** — vLLM `mamba_cache_mode="align"`, ST 는 청크 경계 스냅샷 96개 |
| 스케줄링 | **의도적으로 다름** — 선점 없음·혼합 스텝 없음(D3·D9), vLLM 기본값도 같은 쪽 |
| 메모리 회계 | **ST 앞섬(부팅 뒤 상수), vLLM 앞섬(측정)** — §50·§51 에서 측정 쪽을 메움 |
| 서빙 중 메모리 관측 | **ST 앞섬** — `memory_reserved` 고수위까지, vLLM 은 블록 점유율만 |
| 종료 시 반납 | **ST 앞섬** — 아레나·티어·보유자 전부 + `allocated_after` 증명 (§51·§52) |
| NVMe 대화 티어 | **ST 앞섬(대화 단위·부팅 넘어 생존)**, **vLLM 앞섬(생태계)** |
| 관측 지표 | **ST 앞섬(개수·상자 메모리)** — 69 대 42 |
| 죽음 기록 | **ST 만 있음** — `record.py` 스텝 링 + DeathDump |
| 부팅 | **ST 앞섬(로더·아레나)**, **vLLM 앞섬(예열 일반성)** |
| 샘플링 파라미터 | **vLLM 앞섬** — `min_p`·`bad_words`·`allowed_token_ids`·`prompt_logprobs` 등이 우리에겐 없음 |
| 스펙 디코딩 | **vLLM 앞섬** — 제안자 여러 종 + 동적 k. 우리는 DFlash2 고정 k |
| 구조화 출력(백엔드) | **vLLM 앞섬** — 백엔드 4종, 우리는 xgrammar 하나 |
| 문의 표면 | **vLLM 앞섬** — 23개 v1 경로 대 우리 7개 |

---

## 1. 문 (OpenAI 표면)

**ST 가 여는 것** (`base/serve.py`, 라우트 표 `serve.py:2812`): `/v1/chat/completions`, `/v1/completions`,
`/v1/models`, `/v1/engine/completions`, `/v1/engine/calibration`, `/v1/prefix/warm`, `/v1/prefix/unpin`,
`/tokenize`, `/detokenize`, `/metrics`, `/health` — **11개**.

**vLLM 이 여는 것**: 위에 더해 `/v1/embeddings`, `/v1/rerank`(+`/v2/`), `/v1/score`, `/v1/classify`,
`/v1/responses`(+cancel), `/v1/messages`(Anthropic 호환), `/v1/audio/transcriptions`·`translations`,
`/v1/chat/completions/render`·`derender`·`batch`, `/v1/load_lora_adapter`, 그리고 운영 경로
(`/sleep`·`/wake_up`·`/pause`·`/resume`·`/start_profile`·`/reset_prefix_cache`·`/update_weights`…).

**판정: vLLM 앞섬.** 다만 대부분은 **범위 밖**이다 — 임베딩·리랭크·오디오·LoRA·Responses API 는
ST 가 하겠다고 한 적이 없다. 범위 안에서 진짜 차이는 둘:
- **`/reset_prefix_cache` 류 운영 훅이 없다.** `/v1/prefix/unpin` 은 **한 프롬프트의 핀을 푸는 것**이지
  캐시를 비우는 것이 아니다. 통째로 버리는 문이 없다.
- **`/sleep`·`/wake_up` 이 없다.** 구조적으로 못 한다(한 아레나 + 범프). 대신 **종료 반납**을 45차 §51·§52 로 만들었다.

생성 파라미터는 겹치는 쪽이 더 많다. ST 가 받는 것: `n`, `best_of`, `echo`, `stop`, `stop_token_ids`,
`min_tokens`, `logit_bias`, `logprobs`/`top_logprobs`, `seed`, `temperature`, `top_p`, `top_k`,
`presence/frequency/repetition_penalty`, `response_format`, `tools`/`tool_choice`, `stream_options`,
`suffix`, `cache_salt`/`prompt_cache_key`, `reasoning_effort`/`reasoning_budget`, `chat_template_kwargs`,
`continue_final_message`. **없는 것**: `min_p`, `bad_words`, `allowed_token_ids`, `prompt_logprobs`,
`ignore_eos`, `include_stop_str_in_output`, `spaces_between_special_tokens`.

## 2. 스케줄러와 스텝 루프

**의도적으로 다르다.** vLLM 은 블록이 모자라면 러닝 요청을 **선점**하고 대기열 앞에 되돌린다;
한 포워드에 **디코드와 프리필 청크를 섞는다**. ST 는 둘 다 안 한다 — 과다 입장을 안 하고(D3, `kv.py:18`),
균질 스텝을 요구한다(D9). 근거는 취향이 아니라 실측이다: 39차에서 **순차가 인터리빙보다 빨랐다**.
그리고 vLLM 의 기본값 `full_sequence_must_fit=True` 가 우리와 같은 선택이다.

**동등한 것**(대조 확인): 청크 프리필과 그 사이 디코드, 적어도 한 토큰은 계산, 회수 가능 블록을 여유로 계산,
스타베이션 밸브.

## 3. KV 블록과 접두사 캐시

**블록 수명은 동등하다.** vLLM 의 규칙 — 해제된 블록을 해시를 단 채 free 리스트에 남기고 O(1) 로 다시 집기,
해시 없는 블록부터 쓰기, 한 요청의 블록은 꼬리부터 풀기 — 를 45차 §32 에서 그대로 가져왔다. ST 쪽에는 등급이
하나 더 있다(익명 → **바랜** → 캐시됨): 스냅샷이라는 둘째 자원이 있기 때문이다.

**하이브리드(선형 어텐션) 접두사 캐시는 설계가 다르다.** 이 vLLM 포크는 GLM5Next 를 알고 있다 —
`KpoolTailSpec`, `mamba_cache_mode`, *"mamba state is only checkpointed at block boundaries in align mode"*
(`v1/core/kv_cache_utils.py:645-680`). 즉 **블록 경계마다 상태를 체크포인트**하고, align 이 아니면
부분 히트를 포기한다. ST 는 **프리필 청크 경계**(블록의 배수)에서만 체크포인트하고, 스냅샷을 희소 자원으로
명시해 96개를 예산에 세운다(`base/prefix.py`, `boot.PREFIX_SNAPSHOTS`).

- vLLM 쪽 장점: 입자가 더 곱다. 저쪽은 **블록**(GLM5Next 기본 1,152 토큰) 단위로 공유하고,
  우리는 **프리필 청크**(6,912 토큰; 블록 자체는 768) 단위로만 공유한다 → 6,912 토큰을 못 넘는 공유 접두사는
  우리가 통째로 놓친다.
- ST 쪽 장점: 스냅샷이 **예산의 한 줄**이고, 바랜 경계가 티어에서 **스냅샷만** 읽어 되살아난다(1 GB → 77 MiB).

**판정: 동등, 설계가 다름.** 어느 쪽이 이 워크로드에서 나은지는 아직 안 쟀다.

## 4. KV 오프로드

**ST**: NVMe 티어 하나(`kv_tier.py` 487L + `tiered_kv.py` 227L). 대화 단위, O_DIRECT, **부팅을 넘어 생존**(D16),
그리고 접두사 경계용 둘째 티어. 45차 §53 에서 상한을 선언했다(64 GiB 대화 + 16 GiB 경계).

**vLLM**: CPU 티어링(`v1/kv_offload`, 11,459L)에 더해 커넥터 생태계 — LMCache, Mooncake, NIXL, hf3fs,
flexkv, multi-connector, P/D 분리.

**판정: 반반.** 대화를 **디스크에 턴 너머로 남기는 것**은 우리가 더 하고(vLLM 의 `--swap-space` 는 선점된
블록만, 기본 4 GiB CPU), **분산·다중 백엔드**는 vLLM 이 압도한다. 우리는 4노드 한 상자라 커넥터가 필요 없다.

## 5. 샘플러

**동등**(대조 확인): 처리 순서(마스크 → 바이어스 → 페널티 → 온도 → top-k/p), 페널티 범위,
드래프트 위치별 페널티, 잔여 분포와 수락 판정.

**ST 앞섬**: 페널티가 **증분**이다. 행마다 `seen`(bool [V])·`counts`(f32 [V])를 들고 토큰 하나에 스캐터 하나.
128K 프롬프트 디코드 한 스텝 **456 ms → 0.130 ms**. vLLM 은 같은 자리에 *"currently quite inefficient
and will be reworked anyhow"* 라고 적어 뒀다(`v1/sample/ops/penalties.py:27-28`).

**vLLM 앞섬**: 파라미터 폭(`min_p`, `bad_words`, `allowed_token_ids`, `prompt_logprobs`).
`min_p` 는 이미지 안 `flashinfer.sampling.min_p_sampling_from_probs` 로 바로 닿을 수 있다 — 미구현.

## 6. 스펙 디코딩

**vLLM 앞섬.** 제안자가 여럿이다: `dflash.py`(우리와 같은 계열), `eagle.py`, `medusa.py`,
`ngram_proposer.py`(+GPU 판), `suffix_decoding.py`, `draft_model.py`, `step3p5.py`, `gemma4.py`,
그리고 **배치 크기 → k 의 동적 표**(`v1/spec_decode/dynamic/`).

**ST**: DFlash2 하나, **고정 k=5**. `max_seqs=4` 라 "배치가 커지면 드래프팅이 순수 오버헤드" 위험은 작지만
**재 본 적이 없다.** 다만 계측은 우리가 더 낸다 — `st:spec_accepted_per_step_total` 은 수락 개수의
**전체 분포**로, vLLM 의 위치별 생존 카운트보다 정보가 많다.

## 7. 구조화 출력

**적용 경로는 ST 앞섬**(45차 §31): 스텝 하나의 비트마스크, 전용 커널, 걸음을 포워드 앞으로, 죽은 드래프트
위치는 **게더조차 안 함**. 한 스텝(4행 × 6위치, 어휘 154,880): 걸음 0.266 → **0.055 ms**, 전송 4 → **1**,
텐서 연산 44 → **4**.

**백엔드는 vLLM 앞섬**: xgrammar / guidance / outlines / lm-format-enforcer 넷. ST 는 xgrammar 하나이고
`json_object`·`json_schema`·`ebnf` 세 형태 + 도구 문법을 낸다. (이미지에 `outlines` 는 **없다**.)

## 8. 메모리

전문은 `MEMORY_VS_VLLM_20260912.md`(§50)와 45차 §51·§52·§53. 요지만:

- **부팅**: vLLM 은 `profile_run` 한 번으로 KV 예산을 **측정**한다. ST 는 **선언**한다(D1) — 그리고 §51 에서
  프리필 활성화 피크와 그래프 워크스페이스를 **원장에서 되읽어** 그 두 줄을 실측으로 바꿨다(활성화 5.72 GiB;
  빌려 쓰던 vLLM 기울기는 39% 낮았다).
- **부팅 뒤 발자국**: ST 는 아레나 하나 + 범프라 **상수**다. vLLM 은 캐싱 할당자가 계속 움직인다.
- **서빙 중 관측**: vLLM 의 `gpu_cache_usage_perc` 는 **블록 점유율이지 바이트가 아니다.** ST 는
  `st:device_memory_reserved_bytes`/`_peak_bytes`/`_allocated_bytes` 와 **`st:host_memory_available_bytes`**
  (earlyoom 이 결정에 쓰는 바로 그 줄)를 낸다. **ST 앞섬.**
- **런타임 반납**: vLLM 은 `sleep`/`wake_up` 으로 가중치를 OS 에 돌려줄 수 있다. ST 는 못 한다.
  대신 **서빙 종료 시 반납**을 만들었다(§51·§52) — 아레나 스토리지 해제, 티어 스테이징, 아레나 밖 보유자 열넷,
  그리고 `allocated_after` 로 **증명**. vLLM 에는 그런 종료 증명이 없다.

## 9. 관측과 사후 분석

**지표 개수**: ST **69** 대 vLLM `v1/metrics` **42**. 이름은 겹치는 것을 `vllm:` 접두사 그대로 쓴다.
ST 에만 있는 계열: 상자 메모리 6종, 블록 등급별 점유(`st:kv_blocks_{anonymous,faded,cached,…}`),
접두사 캐시 14종, 티어 6종, 스펙 디코딩 분포, 디토크나이저 복구, 핸드오버.

**ST 에만 있는 것 하나 더**: `base/record.py` 의 **스텝 링**(마지막 N 스텝을 항상 들고 있다가 프로세스가
죽을 때만 기록)과 `DeathDump`. vLLM 에는 대응물이 없다 — 이 상자에서 earlyoom 이 엔진을 1순위로 쏘기 때문에
만든 것이다(`OOM_STUDY`).

## 10. 커널과 실행

**범위가 다르다.** vLLM `v1/attention` 87 파일 47,951줄은 모델·하드웨어 세대·백엔드를 덮는다.
ST `engine/kernels` 60,024줄은 **GB10(SM121) 한 세대에서 GLM-5.3 하나**를 덮되, 그 안에서는 우리가 더 깊다 —
b12x 정적 MoE(스톡 대비 +9.7~11.6%, 38차), KDA 융합, 희소 MLA, NVFP4 W4A4, 준비 커널 통합, 메가커널 실험.
프로덕션 vLLM 이 이 상자에서 쓰는 b12x 레인도 같은 이름이다(`flashinfer.fused_moe.B12x*`).

**판정: 비교 불가.** 일반성은 vLLM, 이 형상에서의 깊이는 ST.

---

## 범위 밖 (우리가 아예 안 만든 것 — "뒤처짐" 이 아니다)

임베딩·리랭크·스코어·분류, 오디오(transcription/translation), Responses API, Anthropic `/v1/messages`,
LoRA, 프롬프트 어댑터, 데이터 병렬과 P/D 분리, 파이프라인 병렬, 인코더-디코더, 풀링 모델,
멀티 커넥터 KV 전송(LMCache/Mooncake/NIXL/hf3fs), `sleep`/`wake_up`, 우선순위 스케줄링,
엘라스틱 EP 스케일링, 배치 API(`run_batch`), CPU/TPU/ROCm 백엔드.

## 이 범위 안에서 진짜 뒤처진 것 (할 일)

1. **`min_p`** — 이미지 안 flashinfer 로 바로 닿는다. 미구현.
2. **`bad_words` / `allowed_token_ids` / `prompt_logprobs`** — 문의 표면 빈칸 셋.
3. **동적 드래프트 길이** — k 가 고정 5. 위치별 수락 곡선은 이미 내고 있으니 **재기만 하면 된다**.
4. **접두사 캐시 입자** — 우리 공유 단위는 청크 6,912 토큰, vLLM 은 블록 1,152. **6,912 토큰 미만의 공유
   접두사는 우리가 전부 놓친다** — 대화형 트래픽에서 이게 제일 클 수 있고, 아직 안 쟀다.
5. **운영 훅** — 접두사 캐시를 통째로 버리는 문이 없다(`/v1/prefix/unpin` 은 한 프롬프트의 핀만 푼다).

*(지표 개수는 ST `base/serve.py` 의 노출 이름, vLLM `v1/metrics` 의 노출 이름을 각각 센 값이다.)*
