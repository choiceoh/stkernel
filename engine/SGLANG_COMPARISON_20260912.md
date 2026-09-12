# SGLang 에서 구조화 출력을 배워올 것 — 조사 (2026-09-12)

`sglang 0.5.19` 의 `srt/constrained/` 전부(2,242 줄)와 스펙 디코딩 쪽 문법 경로(`speculative/spec_utils.py`)를
읽고 ST 와 대조했다. 소스는 PyPI 휠을 스크래치에 풀어 읽었다(GPU·부팅 없음).

**결론부터: 가져올 것은 셋이고 전부 버그다.** 성능 얘기는 vLLM 쪽(§29→§31)에서 이미 끝났고, SGLang 이 더
가진 것은 *동작*이었다 — 사고와 문법의 순서, 컴파일 실패의 주인, 컴파일 시간의 자리. 그리고 SGLang 의 간판
기능이던 **점프 포워드는 0.5.19 의 스케줄러가 더 이상 부르지 않는다**(`srt/constrained/` 밖 호출자 0).

## 1. 가져온 것 — 셋 다 고쳤다 (원장 45차 §32, PR 는 아래)

### 1.1 문법이 모델의 사고 안에서 시작하면 안 된다 — **실물 버그였다**

GLM-5.3 의 기본 경로는 사고를 켠다. 템플릿을 실제로 렌더해 보면:

| chat_template_kwargs | 프롬프트 끝 | 문이 보는 것 |
|---|---|---|
| `{}` (기본) | `<|assistant|>` `<think>` | `reasoning=True` |
| `{"thinking": true}` | `<think>` | `reasoning=True` |
| `{"reasoning_effort": "low"}` | `<think>` | `reasoning=True` |
| `{"thinking": false}` | `<think>` `</think>` | `reasoning=False` |

그리고 `json_object` 문법에서 **첫 생성 토큰으로 허용되는 것은 23 개**(`{`, `[`, `{"`, `[]` …)뿐이고
`</think>`(154842)는 그 안에 없다. 셋을 합치면:

> 기본 설정 + `response_format` = 모델이 사고 블록 **안에서** JSON 을 쓰고, 블록을 닫을 방법이 없고,
> 문의 `split()` 이 `reasoning_end` 를 못 찾아 **답 전체가 `reasoning_content` 로 가고 `content` 는 빈다.**

45차 §22 가 고친 것과 **같은 종류의 버그**가 문법 경로로 다시 들어와 있었다. SGLang 은 `ReasonerGrammarObject`
(사고 중에는 `fill_vocab_mask` 가 아무것도 안 쓰고, 종료 토큰 **열**을 매칭하며, 롤백까지 추적)로, vLLM 은
`structured_output_request.reasoning_ended` 로 막는다.

**우리 것**: 문이 `options["grammar_after"] = reasoning_end` 를 넣는다(그 자리에서 `reasoning` 을 이미 계산한다).
`base/grammar.Matcher` 는 그 토큰이 **커밋될 때까지 잠들어 있다** — 마스크를 안 쓰고, 매처를 안 움직이고,
커널도 안 부른다. 드래프트가 그 토큰을 건너는 스텝 하나만 특별하다: 건너기 전 위치들은 `-1`(전부 허용)로 열어
두고 뒤쪽만 문법이 가린다(커널이 행 슬라이스를 통째로 적용하므로). 드래프트가 깨운 것은 **롤백된다** —
깨우는 것은 커밋된 토큰뿐이고, 그건 `advance` 의 일이다. 우리 종료 토큰은 하나라 SGLang 의 열 매칭은 불필요.

### 1.2 컴파일 실패가 엔진을 죽인다 — **원격 DoS 였다**

`compile_json_schema` 는 스키마가 나쁘면 C++ 에서 던진다. 실측(xgrammar 0.2.6, 어휘 1,024):

| 스키마 | 결과 |
|---|---|
| `pattern: "(a)\1"` (역참조) | `RuntimeError: Regex parsing error ... Backreference` |
| `$ref: "#/definitions/nope"` | `RuntimeError: Cannot find field definitions` |
| `pattern: "(?=abc)x"` (전방탐색) | 경고만, 무시하고 컴파일 |
| `pattern` 에 NUL | `RuntimeError`(0.2.6 은 고쳐졌다; 옛 버전은 **세그폴트** — SGLang 이 `_grammar_key_contains_nul` 로 막는 이유) |

우리 경로에서 그 예외는 `engine.add()` → `_admit()` → `once()` 를 나가고, `once()` 의 `except BaseException` 은
**살아 있는 모든 요청을 중단시키고 다시 던진다**. 즉 잘 만들어진 HTTP 요청 하나로 **네 랭크가 같이 죽는다.**
SGLang 은 `InvalidGrammarObject` 로, vLLM 은 요청 실패로 가둔다.

**우리 것**: `Grammars._build` 가 `(RuntimeError, TypeError, ValueError, UnicodeError)` 를 `ValueError` 로 번역하고,
문이 `prepare_options` 로 **입장 전에** 물어 400 으로 답한다. D3 는 엔진의 실패에 대한 규칙이고, 남의 스키마는
엔진의 실패가 아니다.

### 1.3 컴파일이 디코드 루프 위에 있다 — 283 ms 실측

| 스키마 | 컴파일 |
|---|---:|
| `json_object` 내장 | 4.3 ms |
| 40 필드 × 20 enum | 15.0 ms |
| 40 단계 중첩 배열 | 7.0 ms |
| 400 갈래 정규식 `{1,20}` | **282.8 ms** |

`_bind_options` 는 스케줄러 루프에서 돈다 — 그 시간 동안 **돌던 디코더가 전부 선다**(D10). SGLang 은
`ThreadPoolExecutor` + `Future` + `grammar_queue` 로 루프 밖에 두고, 준비 여부를 **랭크끼리 교집합**으로
합의하고(`all_gather_object`), 타임아웃까지 둔다.

**우리 것**: 컴파일은 스레드로 보내고(`Grammars.compile` 이 핸들을 돌려준다), **첫 마스크에서** 기다린다.
합의는 필요 없다 — 문이 결정하고 스케줄은 어차피 방송되며, 다른 랭크는 같은 지점에서 막힐 뿐이다. 사고하는
행이면 그 지점은 **사고가 끝난 뒤**라, 긴 컴파일이 사고 밑으로 통째로 숨는다.

### 1.4 (덤) 엔진의 종료 토큰이 문법의 stop 토큰이다

SGLang 은 `TokenizerInfo.from_huggingface(..., stop_token_ids=model_eos_token_ids)` 로 **모델의 EOS 를 권위**로
준다. 우리는 안 줬다. GLM-5.3 에서는 우연히 안전했다(실측): 토크나이저의 eos `<|endoftext|>`(154820)가
generation_config 의 셋 `[154820, 154827, 154829]` 안에 있다. 겹치지 않는 모델이라면 **문법이 완성된 뒤
엔진이 멈추지 않는 토큰만 허용**하게 되고, 행은 상한까지 같은 토큰을 뱉는다. 이제 엔진의 eos 를 넘긴다.

## 2. 이미 맞는 것 (확인함)

| 항목 | SGLang | ST |
|---|---|---|
| 배치 하나짜리 비트마스크 | `allocate_vocab_mask(batch)` + `fill_vocab_mask(idx)` | 같다(§31, `Grammars.prepare`) |
| 핀 메모리 + non_blocking | 핀해서 H2D 가 진짜 비동기가 되게(주석까지 같은 말) | 같다 + 스테이징 재사용을 이벤트로 가드 |
| 적용 커널 | xgrammar v0.1.17 트리톤 커널을 **그대로 베껴** 벤더링 | 같은 커널을 xgrammar 에서 부른다 |
| 호스트 일을 포워드 밑으로 | *"Call it after the target verify launch — every step here is host work, so it all overlaps that forward"* | 같다(§31: 포워드 **앞**에서 채우고 큐에 넣는다) |
| 드래프트 걸음 + 롤백 | 트리 DFS, 부모 마스크로 수락 판정, 끝나면 롤백 | 체인 워크, `accept_token` 반환으로 판정, `walked` 만큼 롤백 |
| 캐시 단위 | 컴파일된 문법을 키로 캐시, 행마다 매처 `copy()` | 같다(`_cache[key]`, 행마다 `Matcher`) |
| 종료된 문법 | stop 토큰만 허용 | 같다 |

## 3. 의도적으로 다른 것

- **랜딩 버퍼**: SGLang 은 스텝마다 `vocab_mask.to(device)` 로 새 디바이스 텐서를 만든다. 우리는 하나를 들고
  재사용하고, 스테이징 덮어쓰기를 CUDA 이벤트로 막는다.
- **행 재배치**: 그들은 배치 한 번에 커널을 걸어야 해서 재정렬이 필요하다. 우리는 행마다 자기 슬라이스를 준다.
- **죽은 드래프트**: 그들은 트리에서 안 뻗는다. 우리는 거기서 더 가서 **게더도 안 한다**(§31).
- **문법 종료 = 요청 종료**: 그들은 `FINISH_MATCHED_TOKEN` 으로 그 자리에서 끝낸다. 우리는 문법이 강제한
  stop 토큰을 실제로 한 번 더 뽑는다(스텝 하나). 토큰이 실제로 나가는 쪽이 계산·스트림과 일관된다.
- **준비 합의**: 그들은 `all_gather_object` 교집합 + 폴 타임아웃. 우리는 문이 결정하고 랭크는 첫 마스크에서
  막힌다 — 메시지 없이 합의하는 우리 방식 그대로.

## 4. 안 가져올 것

**점프 포워드 — SGLang 자신이 안 쓴다.** `try_jump_forward` / `jump_and_retokenize` / `outlines_jump_forward.py`
는 전부 살아 있지만, **0.5.19 에서 `srt/constrained/` 밖의 호출자가 없다.** 스케줄러의 `check_for_jump_forward`
가 사라졌다. 재토큰화 위험(점프한 문자열이 토큰 단위로 다시 갈리는 문제)과 스펙 디코딩·radix 캐시와의 상호작용이
값보다 비쌌다는 뜻으로 읽힌다. 우리에게는 더욱 비싸다 — D9 균질 스텝이라 디코드 행을 프리필로 되돌려야 하고,
KDA 상태와 드래프터 링까지 그 토큰들을 지나야 한다.

> **대신 우리 모양의 후보**: 문법이 한 토큰만 허용하는 자리는 **드래프트로 쓰면 된다**(`find_jump_forward_string`
> 을 토큰화해 드래프터의 제안을 대체). 타깃이 어차피 검증하므로 **틀려도 정확성 위험이 0** 이고, 스케줄러는
> 하나도 안 바뀐다. 측정 없이 넣지 않는다 — 후보 목록으로.

**트리 드래프트 마스크** — 우리 드래프터는 체인이다(`GrammarTree.from_linear_chain` 이 그들의 같은 경우).

**기능 셋은 지금 없다, 필요해지면 그때**: `structural_tag`(툴 호출의 인자를 스키마로 강제), `regex`/`ebnf`
문법 종류, strict thinking(사고 예산이 끝나면 `</think>` 를 강제로 넣는 토큰 필터), `any_whitespace=False`
(JSON 공백을 못 늘리게 해 토큰 낭비를 막는다). 넷 다 문의 표면이지 엔진의 구조가 아니다.
