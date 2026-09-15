# 문법 자격 검사 — xgrammar 가 토크나이저를 transformers 로 한 번 더 파싱하지 않는다 (2026-09-15)

## 무엇인가

`qualify grammar` 는 main `3acae017` 의 두 따뜻한 부팅에서 rank 0 기준 10.18 s(03:38 UTC 프로덕션)와 6.75 s(03:51 UTC 큐 hold)였다. 네 랭크 모두 같은 일을 하고, 문은 가장 느린 랭크를 기다린다.

`grammar.for_checkpoint` 는 세 가지를 한다.
- transformers `AutoTokenizer` 로 20.2 MB 짜리 `tokenizer.json` 을 다시 읽는다.
- `xgr.TokenizerInfo.from_huggingface` 를 부른다.
- 장치에서 마스크 커널을 검사한다(`qualify`).

엔진은 같은 파일을 이미 `tokenizers.Tokenizer` 로 읽는다(`boot.tokenizer`, 문과 빌드가 쓴다). 그러니 transformers 객체는 xgrammar 에 입력을 넘겨주려고만 만들어졌다.

**CPU 로 잰 단계별 시간** — `st-engine:glm53`/`bracket-9c45086a0622`, CUDA 숨김, 프로세스 두 번([grammar_steps.log](grammar_steps.log)):

| 단계 | 초 |
|---|---:|
| `import xgrammar`(transformers 를 끌어온다) | 2.59 / 2.55 |
| `AutoTokenizer.from_pretrained` | 2.25 / 2.30 |
| `TokenizerInfo.from_huggingface` | 1.13 / 1.13 |
| builtin JSON 문법 컴파일 | 0.03 / 0.03 |

## 같은 입력을 백엔드에서 읽는다

fast 토크나이저에 대한 `from_huggingface` 는 세 가지를 읽는다.
- `get_vocab()`: `PreTrainedTokenizerFast` 에서는 백엔드의 `get_vocab(with_added_tokens=True)` 다.
- `backend_tokenizer.to_str()`: vocab type 과 prefix space 를 감지하는 데 쓴다.
- stop id 들.

그런 뒤 `TokenizerInfo(encoded_vocab, vocab_type, vocab_size, stop_token_ids, add_prefix_space)` 를 만든다.

GLM-5.3 의 실제 메타(`/home/choiceoh/st-engine/st-glm53-meta`)에서 두 경로를 대조했다([tokenizers_info.py](tokenizers_info.py), [tokenizers_info.log](tokenizers_info.log)). 백엔드에는 `boot.tokenizer` 처럼 `no_truncation`·`no_padding` 을 적용했다.

| 대조 항목 | 결과 |
|---|---|
| vocab dict | 같다 |
| 감지된 metadata(`BYTE_LEVEL`, `add_prefix_space=False`) | 같다 |
| decoded vocab 154,880 개 | 모두 같다 |
| stop id / special id | 같다 |
| `dump_metadata()` | 같다 |

| 경로 | 초 |
|---|---:|
| `AutoTokenizer` + `from_huggingface` | 2.23 + 1.13 = 3.36 |
| 백엔드: `get_vocab` + `to_str` 감지 + 생성자 | 0.18 + 0.78 + 0.21 = 1.17 (+ 파일 로드 0.84, 문 단계에서 옮겨 온다) |

## 버린 방법 — `TokenizerInfo.serialize_json` 캐시

xgrammar 0.2.3 의 JSON 직렬화는 왕복하면 NUL 바이트만으로 된 토큰 다섯 개(id 188, 102858, 105749, 110532, 119174)가 빈 바이트열이 된다([decoded_vocab_diff.log](decoded_vocab_diff.log)). 세 문법으로 걸어 본 마스크는 같았지만, 디스크 캐시로 쓰기에는 decoded vocab 이 바이트 단위로 같지 않다. 그래서 넣지 않았다.

## 바꾼 것

- **`base/grammar.tokenizer_info(tokenizer, vocab_size, stop_token_ids)`**
  - `from_huggingface` 의 fast 경로와 같은 순서·같은 입력으로 `TokenizerInfo` 를 만든다.
  - stop id 가 없으면 거절한다. transformers 의 `eos_token` 을 대신할 수 없다.
- **`for_checkpoint(..., tokenizer=None)`** 과 `Grammars(..., info=None)`
  - 토크나이저를 받으면 백엔드 경로를 쓰고, 안 받으면 예전 그대로다. qwen38 과 테스트는 바뀌지 않는다.
- **GLM 부팅**
  - `qualify grammar` 단계에서 문의 토크나이저를 먼저 로드하고, 그 토크나이저로 문법을 만든다.
  - 문 단계는 그 객체를 그대로 쓴다. 로드는 옮겨졌을 뿐 늘지 않았다.
  - GLM-5.3 의 `tokenizer.json` 이 transformers 가 줄 모든 토큰을 담는다는 것은 위 대조로 확인했다.
- **바뀌지 않는 것.** 마스크·수치·노브.

## 검증

- **CPU 테스트.** `tests.test_engine_grammar tests.test_engine_bootpaths`: 44 OK(1 스킵은 xgrammar 없는 환경 전용). 로그는 [cpu-tests.log](cpu-tests.log) 에 있다.
- **새 `TokenizerInfoTests`.**
  - byte-level BPE 와 Metaspace BPE 토크나이저를 특수 토큰과 함께 학습시켰다. 헤드가 토크나이저보다 넓은 경우도 넣었다.
  - 각 경우에서 `from_huggingface(PreTrainedTokenizerFast)` 와 `tokenizer_info` 를 비교한다: decoded vocab, vocab type, prefix space, stop id, special id, metadata.
  - JSON object 와 JSON schema 문법을 토큰마다 걸으며, 초안 세 개씩 두 경로의 마스크가 같은지 본다.
- **줄어들 값(추정).** 랭크마다 CPU 로 약 2.2 s 다(3.36 s → 1.17 s). 함대의 `qualify grammar` 는 6.75–10.18 s 로 호스트 측정보다 크고, 어디서 늘었는지는 기록이 없다.

## 재지 않은 것

- **GPU 부팅.** 다음 부팅 rank 0 표에서 볼 것:
  - `qualify grammar`: 기준 6.75 / 10.18 s.
  - `door`: 토크나이저 로드가 빠져 더 짧아야 한다.
- 문 뒤 구조화 출력 요청의 실제 응답은 재지 않았다. 마스크 동일성은 위 CPU 대조가 전부다.
