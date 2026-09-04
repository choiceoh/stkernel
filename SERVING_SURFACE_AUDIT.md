# dsv4 서빙 표면 감사

Date: 2026-08-11
Scope: `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6`
(image ID `sha256:b763d81b57f7611378a514fa0faf859c3b0d0ec1010f8c5115bea11a60d49ec3`,
라이브 핑거프린트 `vllm-0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626-tp4`)
의 서빙 레이어 전체 — 챗 템플릿/인코딩, 토크나이저 래퍼, 툴 파서, 리즈닝 체인,
DelegatingParser, 프로토콜 직렬화, tool_choice/response_format, OpenAI·Anthropic 엔드포인트.
트리거: 외부 2x-Spark 레포(tonyd2wild/DeepSeek-v4-Flash-...) PR#17 도입 검토.

방법: ①이미지 내 2단계 하네스(스톡 기준선 런 → 오버레이 마운트 런, srv1
`/var/tmp/dsv4audit/`) ②hy4 실서버 라이브 SSE 프로브 ③모델 동봉 공식 레퍼런스
(`<MODEL>/encoding/encoding_dsv4.py` + 골드 테스트)와 바이트 대조.

## 판정 요약

| # | 표면 | 판정 | 조치 |
|---|---|---|---|
| 1 | 외부 PR#17 (tool_calls:[] 스트리밍 픽스) | **도입 불필요** | 우리 이미지는 프로토콜 레이어에서 이미 상위 수정 |
| 2 | 인코딩/래퍼 — 공식 레퍼런스 drift 5건 + 와이어 위생 2건 | **수정·머지됨** | PR #13 (오버레이 4파일+manifest+런처) — ~~배포 대기~~ → **배포됨**(아래 상태 주) |
| 3 | 리즈닝 체인 (R1/base/DelegatingParser) | 건강 | 잔여 저영향 2건 보류 (§3) |
| 4 | tool_choice named/required | **미강제 (라이브 실증)** | 개선 대기 — 옵션 3안 (§4) |
| 5 | response_format (json_object/json_schema/structural_tag) | 문법 배선 정상 | 프롬프트 힌트 훅 미사용은 업스트림 관례 — 보류 |
| 6 | Anthropic `/v1/messages` | **라이브·건강** | 3종 프로브 통과 (§5) |
| 7 | n>1 멀티초이스 · content-parts 배열 | 안전 | choice별 파서 인스턴스 분리 확인 |
| 8 | srv1 `vllm-tp2` 컨테이너 | 서빙 표면 아님 | `ray start --head` — spark-arena 별개 인프라, 종결 |

## 1. 외부 PR#17 — 도입 불필요 판정

그쪽 버그: 스톡 파서가 콘텐츠 델타에 `tool_calls=[]`를 명시 전달 →
`exclude_unset` 직렬화에서 `"tool_calls": []` 방출 → JS에서 truthy →
에이전트 클라이언트가 답변 전체를 드랍. 그쪽 이미지(`/opt/env`, 307줄 파서,
업스트림 #42879 이전 계보)에서는 실재하며 레포 오너가 라이브 재현.

우리 이미지 판정 근거:

- `DeltaMessage`(entrypoints/openai/engine/protocol.py:350)에 빈 `tool_calls`를
  직렬화 시 제거하는 커스텀 `@model_serializer` 실존 — 모든 툴 파서에 일괄
  적용되는 상위 수정.
- 라이브 SSE 실증: tools 실린 요청의 콘텐츠 델타에 `tool_calls` 키 자체가 없음.
- 우리 스톡 파서는 567줄 신세대(`_process_streaming_buffer` 증분 인자 스트리밍,
  `recovers_tool_calls_in_reasoning=True`, `structural_tag_model`) — PR 파일(321줄,
  버퍼-일괄 방출)을 이식하면 증분 스트리밍·missing-`</think>` 복구를 잃는 퇴행.

## 2. 인코딩/래퍼 drift — 수정·머지 (PR #13)

공식 레퍼런스와 전체 diff 대조로 확인, 오버레이 4파일(스톡 대비 최소 diff)로 수복:

| 결함 | 스톡 동작 | 수복 |
|---|---|---|
| reasoning_effort 3단계 붕괴 | "high" 무동작, 공식 max 텍스트 소실, "low" 거부(assert) | 공식 low/high/max 원문 복원 + 래퍼 매핑 정합 |
| tools 주입 위치 | 빈 system 캐리어 삽입 → `BOS+"## Tools"+{클라 system}` | 선두 system 메시지에 부착 → 공식 골드 순서 `BOS+{system}+"## Tools"` |
| arguments JSON 실패 | 예외 → 요청 실패 (`""` 재전송 포함) | 공식 fallback(`{"arguments": raw}`) 복원 + non-dict 가드 |
| reasoning_content 키 | 무시 (자사 `reasoning`만) | 입력 시 양 키 수용 |
| effort=none 정합 | 인코더 chat ↔ 파서 R1 불일치 → 응답 전체가 reasoning 필드로 | 파서도 chat(Identity) 선택 |
| 스트리밍 null 잡음 | continuation 청크마다 `"id"/"type"/"name"/"content": null` | 미설정 필드 미전달 |
| 런처 사문 kwargs | template-kwargs temperature/top_p — 래퍼가 안 읽음 | 제거; EFFORT 노브 신설(기본 빈 값=종전 바이트 동일) |

검증: 하네스 패치판 런에서 effort 4단계 × encode, 멀티턴 tools 라운드트립(양 키),
실토크나이저 end-to-end 렌더 전부 공식 레퍼런스와 **바이트 동일**; 기본 트래픽
7픽스처(plain chat/think, 비툴 멀티턴, effort 미지정, 무system tools 등)는 스톡과
**바이트 동일** 어서션; 스트리밍 시뮬(인자 재조립·continuation 위생·EOS finalize)
green. 배포 후 A/B: MEASUREMENTS.md 신규 A/B 큐 3(EFFORT)·4(tools 위치 품질 확인).

> **상태 주 (2026-09-05)**: 이 문서는 2026-08-11 시점의 감사다. 그 뒤 확인된 것 하나 —
> 2번의 인코딩/래퍼 오버레이는 `dsv4_tokenizer` 모듈로 프로필에 실려 있고 4노드에
> **배포돼 있다**(srv2 `~/hybrid-stack/overlay-b12x/` 에 `tok_deepseek_v4.py` ·
> `tok_deepseek_v4_encoding.py` 존재). 나머지 열린 항목(4번 tool_choice 등)의 상태는
> 재확인하지 않았다 — 판정을 갱신하려면 이 문서의 하네스를 다시 돌려야 하고, 그
> 결과는 원장에 들어간다.

## 3. 리즈닝 체인 — 건강, 잔여 저영향 2건 보류

정상 확인: V4의 "`<think>` 안에서 생성 시작"은 R1 서브클래스 재분류 +
DelegatingParser의 prompt-state 초기화(`is_reasoning_end_for_prompt`) 이중 방어;
마커 감지는 토큰 ID 기반이라 분할 문제 구조적 부재; in-reasoning 툴콜 복구는
balanced-block 요구 + partial-tag hold-back으로 견고.

보류 잔여 (상한 ~0 판단):

1. `count_reasoning_tokens`가 V4에서 항상 0 (depth 카운터가 start 토큰 전제) —
   소비처는 Responses API usage 한 곳(responses/serving.py:892). Chat Completions 무영향.
2. R1 split delta의 `content=None` 명시 → `"content": null` 잡음 1청크 (falsy, 무해).
   ※ 1·2 모두 R1 파서 67줄 파일 하나의 오버레이로 동시 해결 가능 — Responses API
   사용 시점에 착수.

## 4. tool_choice named/required — 미강제 (라이브 실증)

- named + 무관 프롬프트("2+2는? 툴 쓰지 마") → tool_calls 0, 산문 "2+2 equals 4",
  finish_reason "length". OpenAI 계약(named=무조건 해당 함수 호출) 위반.
- required → 런마다 모델 자유선택(한 런 자발 호출, 한 런 산문) — 강제 아님, temp 0.8 복불복.
- 근본 원인: named/required → 문법(structured outputs) 변환 경로가 포크에 부재.
  protocol.py는 형태 검증만(750–835), serving.py는 파서 결과 포장만(876–899),
  xgrammar 내장 `deepseek_v4` structural tag는 response_format 경로에만 연결.
- 부수 결함: serving.py:941 — required인데 미호출이면 빈 tool_calls로
  finish_reason "tool_calls" 오표시 가능.
- 영향: auto만 쓰는 트래픽은 무영향. forced-function 패턴(LangChain/OpenAI SDK
  구조화 추출 헬퍼) 클라이언트는 조용히 깨짐. Anthropic 경로에도 상속 추정.
- 수정 옵션: ①named/required→structural tag 문법 배선 (정공, protocol/serving
  대형 파일 오버레이 부담) ②기보유 툴 파서 오버레이의 adjust_request에서 강제
  지시 프롬프트 주입 (경량, 비보장) ③미지원 400 명시. **API 로그로 named/required
  사용 여부 확인 후 결정.**

## 5. Anthropic `/v1/messages` — 라이브·건강

entrypoints/anthropic/serving.py(1040줄)는 OpenAI chat 경로 위 번역 레이어 —
§2 오버레이 수정이 그대로 적용됨. 라이브 3종:

- 비스트리밍: `thinking` 블록(합성 signature) + `tool_use` 블록(`{"city":"Seoul"}`)
  + `stop_reason:"tool_use"` — 정확한 Anthropic 형태.
- 스트리밍: `message_start` → 블록별 `content_block_start/delta/stop` ×3
  (thinking/text/tool_use) → `message_delta` → `message_stop` — 정석 시퀀스.
- tool_result 라운드트립: 번역 → OpenAI tool 메시지 → 인코딩 `<tool_result>` 완주,
  결과 반영 응답 + `stop_reason:"end_turn"`.

`ANTHROPIC_BASE_URL`로 Claude 계열 도구를 hy4에 직결하는 구성 실사용 가능.

## 6. 종결 항목

- n>1: serving.py:446–452가 choice별 파서 인스턴스 생성(`parsers[i]`) — 상태 공유 버그 부재.
- content-parts 배열 user 메시지: 파트 병합 정상 (라이브).
- `vllm-tp2`(srv1): `ray start --head --port=6379` 실행 컨테이너 — spark-arena
  (eugr-nightly) ray 클러스터 헤드. vLLM 서빙/파서 표면 아님.

## 검증 인프라 (재사용)

- srv1 `/var/tmp/dsv4audit/harness.py` + `stock.clean.json` 기준선: 스톡 런(마운트
  없음) → 패치 런(오버레이 4파일 실경로 `:ro` 마운트 + `/model` + 기준선) 순서.
  공식 바이트 대조·기본 트래픽 무변경·스트리밍 위생·파서 선택을 어서션.
- 배포 후 라이브 재검증 3종: ①tools 실린 콘텐츠 델타에 `tool_calls`/null 키 부재
  ②툴콜 continuation 청크에 `id/type/name` 키 부재 ③`reasoning_effort:"none"`
  요청의 content 정상 방출.

## 잔여 큐

1. **PR #13 오버레이 배포** — srv2에서 `launchers/deploy-overlays.sh` + 재기동 →
   위 라이브 재검증 3종 → MEASUREMENTS A/B 큐 3·4.
2. tool_choice 강제 — API 로그 확인 후 §4 옵션 선택.
3. (조건부) R1 파서 1파일 오버레이 — Responses API 채택 시.
