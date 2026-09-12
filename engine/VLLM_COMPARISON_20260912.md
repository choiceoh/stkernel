# vLLM 에서 항목별로 배워올 것 — 조사 (2026-09-12)

같은 상자 이미지의 vLLM(`glm53:v13-b12x-it`, `0.1.dev20051`)을 영역별로 읽고 ST 코드와 대조했다.
**결론부터: 대부분 이미 맞거나 의도적으로 다르다. 실제로 가져올 것은 넷이고, 그중 하나가 크다.**

부팅·캡처는 `BOOT_TIME_STUDY_20260911.md` §5-l/§5-m 에서, 문·템플릿은 PR #588·#592 에서 이미 끝냈다. 여기는 나머지다.

## 1. 가져올 것

### 1.1 페널티 재계산 — 128K 프롬프트에서 스텝당 456 ms (가장 크다)

`repetition_penalty` 가 켜지면 `adapter._row_logits` 가 **행마다, 드래프트 위치마다** 이걸 다시 한다:

```
prompt = self.tokens[seq][: self.prompt_len[seq]]      # 127,800 개짜리 리스트 사본
generated = self.tokens[seq][self.prompt_len[seq]:] + list(drafts_before)
seen = sorted(set(prompt) | set(generated))            # 파이썬 집합을 매번 새로
```

호스트에서 실측(디코드 한 스텝 = 4 행 × 6 드래프트 위치):

| 프롬프트 | 행·위치 하나 | 한 스텝 |
|---:|---:|---:|
| 1,000 | 0.06 ms | 1.4 ms |
| 8,000 | 1.08 ms | 26 ms |
| 32,000 | 2.62 ms | 63 ms |
| 128,000 | 20.4 ms | **456 ms** |

**로짓 하나 건드리기 전에 드는 시간이다.** 긴 문맥에서 ITL 전체를 삼킨다.
vLLM 도 여기서는 좋지 않다 — 자기 코드가 *"currently quite inefficient and will be reworked anyhow"* 라고 적어 뒀다
(`v1/sample/ops/penalties.py:27-28`). 다만 vLLM 은 (a) 출력 리스트를 **복사하지 않고 살아 있는 참조**로 들고
(`logits_processor/interface.py:45-48`), (b) 배치 전체를 **한 번의 scatter-add** 로 센다.
우리가 할 일은 **행마다 누적 카운트를 증분 유지**하는 것이다 — 토큰 하나가 늘 때 하나만 더하면 된다. vLLM 보다 나아지는 자리다.

### 1.2 계측의 빈칸 넷

- **드래프트 위치별 수락 생존 곡선.** `vllm:spec_decode_num_accepted_tokens_per_pos`(`v1/spec_decode/metrics.py:247-264`),
  분모는 `num_drafts` 다. **k 를 고르는 근거가 이 곡선**인데 우리에겐 총계만 있다.
- **큐 시간과 추론 시간의 분리.** vLLM 은 `request_queue_time` / `request_inference_time` / `prefill_time` / `decode_time`
  을 **같은 버킷**으로 내보내 대시보드에서 바로 뺄 수 있게 한다(`v1/metrics/loggers.py:889-960`).
- **`finished_reason` 라벨.** 우리 `vllm:request_success_total` 은 라벨이 없어 길이 초과와 정지 토큰 종료가 한 숫자다.
- **토큰 수 버킷을 `max_model_len` 에서 생성**(`build_1_2_5_buckets`, `loggers.py:1302-1308`). 4K 와 1M 배포에 같은 고정 버킷은 쓸모가 없다.

그리고 **스펙 디코딩의 함정 하나**: 우리 `self.itl` 은 스텝 시간을 새 토큰 수로 나눠 관측하므로 **TPOT** 이고,
이름도 `vllm:time_per_output_token_seconds` 로 맞다. 없는 것은 **스텝당 간격**(`vllm:inter_token_latency_seconds`)이다.
둘의 차이가 정확히 평균 수락 길이이고, 하나만 내보내면 *"스펙 디코딩이 스텝을 느리게 했는지, 토큰을 싸게 했는지"* 를 구분할 수 없다.

### 1.3 드래프트 길이가 고정 k=5 다

vLLM 은 배치 크기 → k 의 **개루프 표**(`v1/spec_decode/dynamic/utils.py:77-148`)를 둔다. 이유는 품질이 아니라 용량이다:
배치가 커지면 타깃이 이미 연산 한계라 드래프팅이 순수 오버헤드가 되고, **기준선보다 느려진다.**
우리 `max_seqs` 는 4 라 위험이 작지만, **재 본 적이 없다.** 1.2 의 생존 곡선이 있어야 답할 수 있다.

### 1.4 접두사 캐시의 테넌트 분리 (잠재)

vLLM 은 `cache_salt` 를 블록 해시의 `extra_keys` 에 넣되 **첫 블록에만** 넣는다(`kv_cache_utils.py:560-562`),
그리고 루트를 `os.urandom` 으로 무작위화한다(`:88-116`). 우리는 미디어 다이제스트는 섞지만 테넌트 솔트가 없다.
**루트 무작위화는 우리가 못 한다** — 랭크마다 같은 해시가 나와야 한다(그게 메시지 없이 합의하는 방법이다). 솔트는 가능하다.

## 1-b. 그래서 한 것 (2026-09-12)

**페널티는 이제 증분이다.** `base/sampler.History` 가 행마다 `seen`(프롬프트∪출력, bool [V])과
`counts`(출력만, f32 [V])를 들고, 프롬프트는 **행이 시작할 때 한 번** 걷고 이후 토큰은 스캐터 하나다.
이 스텝의 드래프트는 히스토리에 쓰지 않고 **보정으로** 얹는다(≤5 개라 다시 짓는 것보다 싸다).
다시 짓는 것은 셋 중 하나일 때뿐이다 — 새 행, 프롬프트 길이가 움직였을 때(이어진 턴), 리스트가 줄었을 때(되돌린 드래프트).

| | 전 | 후 |
|---|---:|---:|
| 128K 프롬프트, 디코드 한 스텝 | **456 ms** | **0.130 ms** |
| 행이 시작할 때 한 번 | — | 40 ms (4 행) |

**계측 넷을 더한다**: `vllm:inter_token_latency_seconds`(스텝당 간격 — 기존 TPOT 과 짝),
`vllm:request_queue_time_seconds`, `vllm:request_inference_time_seconds`(같은 버킷이라 빼면 된다),
`vllm:request_success_by_reason_total{finished_reason=...}`.

**정정**: 위에서 "드래프트 위치별 수락 곡선이 없다"고 적었는데 **틀렸다**. `st:spec_accepted_per_step_total` 이
이미 나가고 있고, 그건 수락 개수의 **전체 분포**라 vLLM 의 생존 카운트보다 많은 정보다(꼬리를 더하면 생존 곡선이 된다).

### 1.5 스트리밍 디토크나이저 — 러스트 디코드 스트림과 복구 둘 (조사 뒤 추가)

vLLM 의 `v1/engine/detokenizer.py` 는 `tokenizers` 의 **러스트 `DecodeStream`** 을 쓰고, 그 위에 **프로덕션에서 터져서 달린 복구 둘**이 있다
(`_protected_step`): 토큰 id 가 아닌 id(`OverflowError`/`TypeError`, vllm#21951)와 `Invalid prefix encountered`(vllm#17448).
우리는 같은 알고리즘을 파이썬으로 들고 있었고 **복구는 없었다** — `Tokenizer.decode` 의 `OverflowError` 는 HTTP 스레드에서 SSE 를 중간에 끊는다.

**한 것은 45차 §28** 에 있다. 요지: 러스트 스트림(스텝 배치 하나로), 복구 둘(B 는 미정산 id 로 새 스트림을 프리필해 **잃는 토큰이 없다** — vLLM 은 하나 잃는다),
그리고 vLLM 에 없는 정지 가드. 다만 **문의 시간은 디코드에 있지 않았다**: 러스트 스트림만으로 −8%, 나머지는 `partial_suffix` 와 답 전체를 매 스텝 훑던 정지 문자열 스캔이었다.
스텝당 6.22 → 3.97 µs, 그리고 16K 답에서 21.51 → 4.29 µs (답 길이에 대해 평평해졌다).

## 2. 이미 맞는 것 (확인함)

| 항목 | vLLM | ST |
|---|---|---|
| 샘플러 순서 | 마스크 → 바이어스 → 페널티 → 온도 → top-k/p | 같다(`sampler.process_logits` → `distribution`). **페널티가 온도 앞**, **top-p 가 온도 뒤** — 둘 다 맞다 |
| 페널티 범위 | repetition = 프롬프트∪출력, presence/frequency = 출력만 | 같다 |
| 드래프트 위치별 페널티 | 위치 i 는 `출력 + 드래프트[:i]` | 같다(`_row_logits(..., drafts_before)`) |
| 문법 롤백 | 마스크 만들며 전진한 만큼 되감고, **수락된 토큰으로만** 진짜 전진 | 같다(`grammar.Matcher.masks` 가 `walked` 만큼 `rollback`, `advance` 는 따로). `max_rollback_tokens` 도 넘긴다 |
| 미검증 토큰 캐싱 금지 | `num_tokens_to_cache` 를 `request.num_tokens` 로 상한 | 구조적으로 불가 — 우리 캐시 항목은 **프리필 청크 경계**에만 생긴다 |
| 수락 판정의 0 확률 | fp64 uniform(fp32 는 정확히 0.0 이 나와 p=0 을 수락) | fp32 지만 **엄격한 `<`** 라 `0 < 0` 이 거짓이다. 안전 |
| 잔여 분포 | `max(p−q,0)`, 정규화 생략하고 Gumbel | 같다(`speculative_pick`) |
| 선형 어텐션 드래프트 상태 | 수락 수가 정해질 때까지 들고 있다가 이동 | 링을 `spec_k+1` 로 잡고 위치로 되감는다 |
| 회수 가능 블록을 여유로 계산 | `_get_num_evictable_blocks` | `BlockPool.available()` = free + `reclaimable()` |
| 전체가 안 들어가면 안 받는다 | `full_sequence_must_fit`, **기본 True** | 우리 정책 그대로(입장 때 지평선 전체 예약, 선점 없음) |
| 청크 프리필 + 그 사이 디코드 | 예산 둘, 러닝 루프 먼저 | `decode_due`, `chunk_for(...)`, 스타베이션 밸브 |
| 적어도 한 토큰은 계산 | 캐시 히트를 `num_tokens-1` 로 상한 | `lookup` 이 **경계를 프롬프트 안쪽으로 엄격히** 제한 |

## 3. 의도적으로 다른 것

**선점이 없다.** vLLM 은 블록이 모자라면 러닝 요청을 선점하고(가장 최근 것부터), 블록을 모두 반납시키고,
대기열 앞에 되돌린다 — 그리고 라이브락을 막으려고 *선점이 일어난 스텝에는 대기 요청을 하나도 받지 않는다*
(`scheduler.py:703`). 우리는 아예 과다 입장을 하지 않는다. `kv.py:18` 이 그 이유를 적어 뒀다:
*"그 계약이고, 그걸 축출 뒤에 숨기면 버그를 숨기는 것"*. vLLM 의 기본값(`full_sequence_must_fit=True`)이 같은 선택이다.

**혼합 스텝이 없다.** vLLM 은 한 포워드에 디코드와 프리필 청크를 섞는다. 우리는 D9 로 균질 스텝을 요구하고,
39차에서 **순차가 인터리빙보다 빨랐다**는 실측이 근거다.

**블록이 아니라 청크가 재사용 단위다.** vLLM 은 완성된 블록마다 해시를 남기고, 해제된 블록도 **해시를 단 채 free 큐에**
머물러 나중에 다시 집힌다. 우리는 항목이 블록을 핀하고 풀이 필요할 때 LRU 로 회수한다 — 효과는 같지만
**경계가 청크뿐이다.** 선형 어텐션의 상태 스냅샷이 청크 경계에만 존재하기 때문이고, 모델이 강제하는 것이지 빈 곳이 아니다.

## 4. 안 가져올 것

**캐스케이드 어텐션용 공통 접두사 블록 수**(`get_num_common_prefix_blocks`) — 우리 어텐션은 희소 MLA + KDA 라 해당 없음.
**DP 프리필 균형**(`prefill_schedule_interval`) — 데이터 병렬을 안 쓴다.
**우선순위 스케줄링** — vLLM 의 PRIORITY 모드에는 **에이징이 없어** 낮은 우선순위가 영구히 굶는다. 가져오면 그 구멍까지 온다.
