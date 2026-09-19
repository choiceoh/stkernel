# DSv4.1 의 조합 — 무엇이 있고 무엇이 없나

> 살아 있는 참조 — **한 줄이 닫히면(모듈이 생기면·오라클이 붙으면) 이 표의 상태 칸을 고친다.** 여기가 틀리면 그건 버그다.

DSv4.1 이 범위로 돌아왔다(CHARTER [D5](CHARTER.md), 2026-09-19). 그 모델을 서빙하려면 **조합**이 있어야 하고
(`base/composition`: 층 계획 + 잔차 형식 + 특징), 조합은 **가족에 그 형이 있을 때만** 선언된다.
이 문서는 그 목록이다 — 커널 쪽의 `cells.admission()` 에 해당하는, 모듈 가족 쪽의 작업표.

**지금 상태**: 계획은 선언됐다(`profiles/dsv41/composition.py`, CPU 판정 `tests/test_engine_dsv41_composition.py`).
특징은 아직이다 — 아래 표의 "없다" 가 닫히기 전에는 `build()` 가 없다.

## 계획 (선언됨)

```
40 layers: encoder 20 (compress [0, 2]), decoder 20 (compress [1]);
kv sources [2, 8, 14, 20]; indexers [2, 8, 14, 20, 24, 28, 32, 36]; engram [1, 14]
```

CED — 인코더 20 층, 디코더 20 층. 경계(20)를 설정이 세 가지로 같게 말한다(`compress_ratios` 의 2→1 계단,
`candidate_source_layer_id`, 경계를 넘지 않는 `kv_source_layer_ids`). 층 0·1 은 압축을 아예 안 하고(비율 0)
슬라이딩 윈도로 붙으며 **다른 로터리 표**를 쓴다(기본 `rope_theta`, YaRN 끔 — 나머지는 `compress_rope_theta` + YaRN).
모든 층이 라우팅한다(`intermediate_size` 가 null: dense MLP 는 공유 전문가다).

유도와 그 검사들은 물러난 오버레이의 `dsv41_layers.py`(git history, #1152 가 지움) 것이다. 그 프로브는 이 계획이
체크포인트의 96,085 텐서를 양방향으로 예측하게 했다.

## 특징 — 가족에 있나

| # | 필요한 것 | 가족 | 무엇이 판정하나 | 상태 |
|---|---|---|---|---|
| 1 | 잔차: mHC(split-sinkhorn) | `residual.HyperStreams(head="weighted")` | transformers `deepseek_v4` HyperConnection, CPU (`tests/test_engine_residual_family.py`) | **있다** |
| 2 | engram 주입 | `ngram_embedding.NGramInjection` — `VARIANTS["engram"]` · `SCHEMES["dsv41"]` · `NGramHash.rng` | 벤더 `inference/engram.py`·`model.py` (`tests/test_engine_ngram_family.py`; 오라클 사이트가 있어야 돈다) | **있다** |
| 3 | 전문가 양자화 MXFP4 | `quant.fp4_gemm` (`profiles/dsv41/kernels.py` 가 이미 꽂는다) | 벤더 참조 · 커널 쪽은 `cells.py` 의 세 길 | **있다**(모듈), 레인은 D5 의 두 번째 축이 정한다 |
| 4 | MoE 나머지 축 (noaux_tc·norm_topk·routed_scaling 1.5·swiglu_limit 10·공유 1) | `moe.MoE` / `moe.route` | `tests/test_engine_moe_family.py` | **있다** |
| 5 | 로터리 표 둘 | `Attention` 인스턴스 둘(층마다 `theta`·`scale`) — 계획이 어느 층이 어느 표인지 말한다(`LayerPlan.rope`) | 가족의 회전 테스트 | **있다**(표현 가능) |
| 6 | 어텐션 sink | `Attention(sink=True)` | `sparse_attention.sparse_attn` | **있다** |
| 7 | MLA 폭들(latent·nope·v_dim) | `Attention(form="mla", latent=…, nope=…, v_dim=…)` | — | **빈칸** — 설정에 `head_dim 512`·`qk_rope_head_dim 64` 는 있으나 GLM 설정의 `kv_lora_rank`·`qk_nope_head_dim`·`v_head_dim` 에 해당하는 키가 없다. 벤더 `model.py` 가 정한다 |
| 8 | 출력 저랭크 (`o_lora_rank` 1024, `o_groups` 8) | 없음 — MLA 이름표는 `o` 하나다 | — | **없다** — 가족의 축이 아니다 |
| 9 | 선택: CED 키 압축 + 후보 블록 | 없음 — `QSA`·`DSAKpool`·`MSA`·`Window` 뿐 | — | **없다** — torch 형은 git history(`overlay/modules/dsv41_model/dsv41_compressor.py`·`dsv41_indexer.py`·`dsv41_packed_index.py`), Triton 둘이 그 옆에 있었다. `cells.py` 의 인덱서 레시피가 같은 자리를 가리킨다 |
| 10 | 라우터 점수 `sqrtsoftplus` | 없음 — `moe.route` 는 `softmax`\|`sigmoid` | — | **없다** — **수식이 이 트리에 없다.** 설정 문자열과 "MegaMoE 는 sqrtsoftplus 만"이라는 거절 메시지뿐(git history). 벤더 `model.py` 가 정한다 |
| 11 | 층 간 KV·인덱스 소싱 | 없음 — `Layer` 는 (mixer, mlp, inject) 뿐 | — | **없다** — 계획은 `kv_source`·`indexer` 를 알지만 조합이 "이 층은 저 층의 KV 를 읽는다"를 표현하지 못한다 |
| 12 | engram 이 층 안 **어디에** 쓰나 | — | — | **빈칸** — 어느 층이 표를 갖는지는 설정이 말하지만(1·14), 층 안의 자리를 이 트리의 어떤 파일도 말하지 않는다. 계획은 조합의 유일한 주입 자리(층 앞)에 뒀다 |
| 13 | MTP 헤드 3 의 fuse 형 | `mtp.fuse_concat` 가 DeepSeek-V3 의 형 | — | **빈칸** — `modules/mtp` 는 그 형을 쓰는 모델로 GLM-5.3·Kimi K3·Ling-3.0·MiniMax-M3 를 적고 DSv4.1 을 적지 않는다. 오라클도 없다("not held to an oracle") |
| 14 | 비전 타워 · DSpark 드래프터 | — | — | **범위 밖(이 문서의)** — 체크포인트의 `vision_config` 와 `dspark_*` 는 텍스트 조합의 층 계획 밖이다 |

**빈칸의 규칙**: 위의 "빈칸"은 모르는 것이지 없는 것이 아니다. 전부 벤더 `inference/model.py`(sha 는
`profiles/dsv41/caches.py` 가 핀, srv4 에 있다)가 정하고, 이 컨테이너에는 그 트리가 없다. 추측해서 채우면
조합이 **돌면서 틀린다** — 그래서 비워 둔다.

## 다음 한 걸음

싼 것부터, 그리고 각 줄이 무엇으로 닫히는지:

1. **9번(CED 선택)** — git history 의 torch 형을 `modules/attention` 의 선택으로 되살린다. 판정은 그 시절 프로브가
   쓰던 기준(벤더 압축기와 비트 동일)이고, 오라클 사이트가 필요하다.
2. **10번(sqrtsoftplus)** — 벤더 `model.py` 한 줄을 읽어 `moe.route` 의 점수 축에 넣는다. 그 전에는 못 쓴다.
3. **7·8번(MLA 폭·출력 저랭크)** — 같은 파일이 정한다. 8 번은 가족에 축을 하나 더 들이는 일이다.
4. **11번(층 간 소싱)** — `base/composition` 의 계획 표현을 넓히는 일이라 가장 크다. GLM·Qwen 에는 없던 축이다.
5. **12·13번** — 오라클을 한 번 돌리면 닫히는 확인들이다.

커널 쪽 목록은 따로다 — `cells.plan()` 이 세우고 [`DSV41_CARRY_20260919.md`](DSV41_CARRY_20260919.md) §7 에 있다.
