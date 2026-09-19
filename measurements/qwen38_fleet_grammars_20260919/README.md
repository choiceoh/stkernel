# Qwen3.8 플릿의 문법 컴파일러 — `tools` 요청이 전부 400 이던 것, 그리고 xgrammar 가 문의 토크나이저를 읽어도 되는가 (2026-09-19)

## 무엇이었나

2026-09-19 Qwen3.8 플릿의 MTP 데이터 창(트리 main `a8e3c4de`)에서 `tools` 가 실린 `/v1/chat/completions` 요청이 **전부**
HTTP 400 `structured output (response_format) is not served: no grammar compiler is bound` 였다.

- 문(`base/serve`)은 `tools` 가 있으면 도구 호출 EBNF 를 싣고 `<tool_call>` 에서 켜지게 한다(`grammar_after`). 문의 정책이다.
- 플릿 부팅(`engine/profiles/qwen38/fleet.py` `build` → `adapter.build_model`)은 `grammars=` 를 넘기지 않았다. 그래서 서빙
  모델(`base/composed.ComposedModel`)의 `validate_options` 가 문법이 실린 요청을 모두 거절했다.
- CPU 부팅(`qwen38/boot.py`)과 GLM-5.3 플릿(`glm53/boot.py`)은 컴파일러를 묶는다. Qwen3.8 플릿만 빠져 있었고, Deneb 의
  에이전트 트래픽(도구 호출)을 Qwen3.8 이 하나도 서빙하지 못했다.

## 고친 것

- **프렐류드.** `door_host_half`(부팅 프렐류드 스레드)가 컴파일러를 **모든 랭크에서** 만든다. 렌더러는 랭크 0 만 갖지만
  문법 행의 매처는 랭크마다 돈다. 스레드는 장치를 건드리지 않는다.
- **`qualify grammar` 행.** `build` 가 프렐류드를 합류한 바로 뒤, 캡처 전에 만든다.
  - 마스크 커널을 장치에서 증명한다(`Grammars.qualify`).
  - 서빙 모델에 묶는다(`bind_grammars`).
  - 스레드가 체크포인트에서 다시 읽은 어휘·종료 토큰이 엔진의 것과 다르면 부팅이 죽는다(GLM-5.3 `Prelude.take` 의 대조).
- **바뀌지 않는 것.** 문의 도구 문법 정책, 서빙 검증 경로. 문법 행은 이미 rich 행이다. `_draft_sampling`·`_draw_ahead` 가 비껴가고,
  draft-ahead 검증(`_ahead_ready`, 장치의 argmax — 마스크가 없다, #1273)도 그 행이 있는 스텝은 받지 않는다. 블록 검증 없이 위치마다
  마스크를 거쳐 고른다. CPU 테스트가 그것을 지킨다.

## 토크나이저 경로 — 백엔드에서 읽어도 같은가

GLM-5.3 은 PR #1000 이후 xgrammar 의 `TokenizerInfo` 를 문의 `tokenizers.Tokenizer` 에서 읽는다(`base/grammar.tokenizer_info`).
그 전제는 "tokenizer.json 이 transformers 가 줄 토큰을 모두 담는다" 이고, GLM 메타에서만 확인됐었다. Qwen3.8 서빙 메타로 다시 쟀다.

- **어디서.** srv4, 서빙 이미지 `st-engine:qwen38`(`sha256:edf87125abe4…`, 2026-09-19 18:07 KST 빌드), CPU, `--network none`.
  xgrammar 0.2.3, transformers 5.15.1, tokenizers 0.22.2.
- **메타.** `/home/choiceoh/models/st-qwen38-tep4` — `tokenizer.json` sha256 `0997f410c57a…`,
  `tokenizer_config.json` `b11349aafa7c…`, `generation_config.json` `e70c136c1b78…`.
- **스크립트와 로그.** [tokenizer_info.py](tokenizer_info.py), [tokenizer_info.log](tokenizer_info.log). 프로세스 두 번 돌렸다.

| 대조 | 결과 |
|---|---|
| vocab dict — transformers `get_vocab()` 대 백엔드 `get_vocab(with_added_tokens=True)` | 248,077 개, 같다 |
| decoded vocab — 헤드 폭 248,320 | 다른 것 0 개 |
| vocab type / prefix space | `BYTE_LEVEL` / `False`, 같다 |
| stop id(엔진 종료 토큰 248044·248046) / special id 243 개 | 같다 |
| `dump_metadata()` | 같다 |
| `<tool_call>` | 한 토큰(248058), 두 경로 모두 `<tool_call>` 로 디코드 |
| 도구 호출 하나(`get_weather`, 도시 "서울")를 lazy 도구 문법으로 걸은 25 위치 | 마스크 모두 같다, 종료 토큰에서 끝난다 |
| JSON 하나를 builtin JSON 문법으로 걸은 22 위치 | 마스크 모두 같다 |

| 경로 | 초 (두 번) |
|---|---:|
| 백엔드 — `Tokenizer.from_file` + `tokenizer_info` | 0.61 + 1.32 = 1.93 / 0.73 + 1.56 = 2.29 |
| transformers — `AutoTokenizer` + `from_huggingface` | 5.16 + 1.09 = 6.25 / 5.56 + 1.09 = 6.65 |
| `import xgrammar` (transformers 를 끌어온다) | 4.85 / 4.20 |

srv4 는 프로덕션 옆이라 시간은 경합 아래 값이다. 판정에 쓴 것은 대조다. 플릿은 백엔드 경로(GLM-5.3 과 같은 길)를 쓴다.

## 재지 않은 것 — 다음 운영자 창

- **플릿 부팅.** rank 0 표에서 볼 것:
  - `qualify grammar` 행. GLM-5.3 의 따뜻한 부팅에서는 0.044 s 였다.
  - `prelude_s` 와 `wait for the prelude`. 컴파일러(약 2 s)가 모든 랭크의 프렐류드에 더해진다. 랭크 1-3 은 xgrammar 가 끌어오는
    transformers import(약 4 s)도 더해진다. 09-19 부팅에서 프렐류드가 숨는 창(`arena` ~ `runner`)은 약 12 s 였다. 숨는지는 그 행이 말한다.
  - 부팅 줄 `structured output: on`.
- **도구 요청 끝까지.** `tools` 가 실린 채팅 요청 하나가 200 으로 오는지, 호출이 선언된 도구로 나오는지 본다.
- **도구 요청의 속도.** 문법 행은 호출 전(문법이 아직 잠든 동안)에도 rich 행이다. 드래프트는 헤드의 argmax, 검증은 위치마다 한 번씩
  고른다. 그래서 `tools` 요청의 tok/s 는 문법 없는 요청보다 낮을 수 있다. 잰 적 없다.
