# ST 엔진

> 살아 있는 참조 — **ST 엔진이 지금 무엇인지. 코드가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

stkernel 의 자체 추론 엔진. 네 가지를 옵션이 아니라 **형태**로 가진다:

- **TP=4** — 스파크 네 대가 유일한 world. 랭크-로컬 형상은 사실(`profiles/*/facts.py`)이고 코드에 `// world` 가 없다.
  한 노드 검증도 `base/comm.LocalTP` 로 네 랭크를 스레드로 돌려 진짜 all-reduce 의미를 쓴다.
- **DGX Spark(GB10)** — 장치 하나, 통합 메모리, SM121. 부팅·검증 때 단언(`base/box.check_box`, 프로필마다 `facts.check_box` 로 부른다).
- **NVFP4 가 기본형** — packed 바이트 그대로 상주, packed 위에서 TP, 전역 스케일은 곱셈자, 서빙 커널이 레인. 유일형은
  아니다: 체크포인트가 bf16 으로 가진 것은 bf16 으로 쥔다.
- **ModelOpt dense 안전장치** — 엔비디아 체크포인트의 첫 3개 dense MLP는 긴 prefill(기본 4,096행)에서
  b12x W4A16으로 내려 activation-side FP4 오차를 줄인다. decode와 짧은 prefill은 NVFP4를 유지하며,
  `STK_GLM53_DENSE_W4A16_GUARD_ROWS=0`은 비교 실험에서만 guard를 끈다.
- **KDA 상태 저장 정밀도** — 기본 FP32. 활성 링·prefix 스냅샷·경계 스테이지와 계산·게이트·프리필 중간 상태를
  FP32로 유지한다. 운영자가 답변 길이 증가와 draft 수용률 하락을 관측해 FP16 실험을 잠정 폐기하고 기본값을 복원했다.
  `STK_kda_state_dtype=fp16`은 비프로덕션 재현용으로만 남긴다. 이 실험은 KV 블록과 스냅샷 개수를
  같은 FP32 예산의 개수로 고정하며, TP4/C=4/K=6/스냅샷 48개에서 선언된 절감량은 노드당 1,496 MiB다.
  FP32 복원 시 그 메모리가 다시 필요하다. NVMe 상태 형식을 구분하므로 FP32와
  FP16 부팅은 서로의 저장 상태를 이어받지 않는다. 실제 정밀도는 `st:lane_info`와 onepass의
  `kda_state_dtype`에 기록된다. 품질·속도 판정은 [실험 기록](../measurements/kda_fp16_20260913/README.md)을 따른다.
- **레거시 없음** — 범용 레거시 폴백·플랫폼 디스패치·전방 컨텍스트·가중치 로더 추상은 없다. 수치 안전장치처럼
  프로필 계약에 속한 명시적 guard만 둔다.
- **모델은 형태가 아니다** (2026-09-19, CHARTER D5) — 위의 것들은 **하드웨어의** 형태이고, 모델은 형태가 아니라
  **앞문으로 붙는다.** 프로필 패키지가 자기를 선언하면(`SHAPES`·`MODEL_TYPES`) 마법사가 목록 없이 발견하고
  (`base/kernel_shape.profiles`·`claims`), 프로필이 아직 없는 체크포인트도 `tools/onboard.py` 가 설정만으로 읽어
  형상·레인 표·작업 목록을 낸다. **설정이 정하지 못한 것은 빈칸으로 나오지 추측으로 채워지지 않는다**
  (`base/onboard`): 빈칸 하나가 곧 부팅되면서 틀리는 모델이기 때문이다.

설계 원칙은 `CHARTER.md`(D1~D17). 세 조합 계층(D15)과 실행 커널:

    base/       모델 이름이 없는 것: 아레나, 로더(사전샤딩된 랭크 파일의 범위 읽기), KV 블록/슬롯, NVMe 티어, 스케줄러,
                스텝 메타, 러너, 기록/사망 덤프, 설정(사실+만료 노브), 증명·판정, 그래프, comm(플릿 / LocalTP),
                조합 틀(composition: 층 계획 + 잔차 형식 + 특징, 한 step 루프)
    modules/    특징 모듈: 선형 순환 가족(GDN·KDA 한 특징, 여섯 축), 어텐션 가족(GQA|MLA × 회전 × 노름 × 게이트 × 선택 QSA|DSA|MSA|윈도 × 싱크·상대 편향), MoE 가족(라우터 softmax|sigmoid × 보정 편향 × 그룹 × 활성 silu|clamped|swigluoai × 공유 전문가 plain|sigmoid|sink), 잔차 형식 가족(pre-norm(+출력 conv)·게이트 스트림·mHC 스트림·AttnRes), 해시 n-gram 메모리 가족(Qwen PLE·DSv4.1 engram), 희소 커널 참조(kpool·QSA·MLA·GQA), NVFP4 선형·MoE(공유 전문가
                게이트 포함)·양자화, 하이퍼커넥션(mhc·split-sinkhorn·게이트 잔차), 노름, 회전, 로짓
    profiles/   모델별: 사실·가중치 지도(specs)·사전샤딩·레인 표·조합(net)·검증(check). glm53 이 첫 대상.
                qwen38 은 base/composition 위에 계획과 가중치 이름만 선언한다(composition.py).
                dsv41 은 계획 계층(shapes·budget·placement·caches·engram·dist_run)과 층 계획(composition)·
                핀된 레퍼런스뿐이다 — 레인 표도 조합의 특징도 아직 없다. 모델은 한정하지 않는다(CHARTER D5, 2026-09-19): 새 모델을
                거절하는 것은 목록이 아니라 `cells.admission()` 과 `modules/` 가족의 축이다.
    kernels/    ST가 소유하는 Triton·TileLang·CuTe DSL·CUDA 커널과 필요한 보조 코드

**조합 틀(base/composition, 2026-09-13).** 모델은 파일이 아니라 세 가지 선언이다: 층마다 어떤 토큰 믹서·채널
믹서·잔차 주입을 돌리는지(계획), 서브층이 잔차를 읽고 쓰는 방식(잔차 형식: 평범한 pre-norm, GLM 의 mhc, Qwen3.8 의
게이트 잔차 스트림, DeepSeek-V4.1 의 split-sinkhorn), 그리고 계획이 부르는 특징(modules, 이름은 모델이 아니라 특징).
루프·세그먼트 step·상태 계약·캐시 명세 집계만 base 에 있고 모델 이름은 없다(CHARTER D6 정정판). Qwen3.8 은
`profiles/qwen38/composition.py` 가 config 에서 계획을 유도하고 체크포인트 이름으로 특징에 가중치를 묶는다.
`tests/test_engine_composition.py` 가 그 조립을 transformers 5.16.1 의 `Qwen4ExpForCausalLM`(plan.py 가 sha 로 핀한
오라클)과 CPU 에서 대조한다: prefill 전 토큰·증분 디코드·청크 prefill·EOS 가 섞인 두 시퀀스 한 step 모두 FP32 에서
최대 5e-8, BF16 상대오차 5e-3. 캐시 명세 합은 `qwen38/plan.state_bytes` 와 같다(QSA 키만 원시 키라 압축 비율배).
**특징은 가족이다(modules/linear_attention, 2026-09-13).** 새 모델의 조립 시간을 줄이는 쪽은 모델별 구현을 나란히 두는 게 아니라
가족 하나에 변형 축을 다는 것이다. Qwen3.8(GDN)·GLM-5.3(KDA)·Kimi K3·Ling-3.0-flash 의 선형 어텐션은 되풀이 하나(`gated_delta_rule`)에
여섯 축이다: decay 가 헤드별이냐 채널별이냐, safe gate 의 lower_bound 냐 softplus 냐, decay·게이트 투영이 저랭크 쌍이냐 한 행렬이냐,
게이트 활성(silu|sigmoid), 노름의 반올림 위치(qwen4_exp 는 가중치 전에, glm5_next 는 끝에 한 번). `GatedDeltaNet` 이 그 특징이고
`VARIANTS` 가 네 모델을 축의 값으로 이름 짓는다(gdn·kda·kda_full_gate·kda_full); fused/separate 투영과 conv 는 축이 아니라 가중치 배치라
`named(scheme, source)` 가 이어 붙인다. GLM 의 레인이 임포트하는 `kda_gate`·`kda_output_norm` 은 이 위의 래퍼로 그대로다.
`tests/test_engine_linear_family.py` 가 KDA 변형을 transformers 5.16.1 의 `Glm5NextTextLinearAttention` 에 두 decay 형 모두 FP32 2e-6 으로
붙잡고(GDN 변형은 조립 테스트가 qwen4_exp 에), 조각 prefill·디코드 == 통짜, 분리 가중치 == fused, full-rank == 저랭크 쌍을 본다.
**어텐션도 가족이다(modules/attention).** 일곱 모델의 어텐션은 계산 하나 — 선택이 허용한 위치들에 softmax(q·k·scale + bias) v — 에
축이다: 형식(GQA | MLA 잠재), 회전(없음 | 헤드 앞부분 | MLA 의 rope 부분, neox | interleaved), q/k 노름(없음 | T5 | 1+w), 출력 게이트(없음 |
채널 | 헤드), scale, 선택(Causal | Window | QSA | DSAKpool | MSA — 인덱서는 자기 키 행을 따로 든다), 싱크, Inkling 의 상대 편향·log
scaling·k/v 짧은 conv. `Attention` 이 그 특징이고 `named(scheme)` 이 여섯 체크포인트 이름을 잇는다(Qwen 의 q_proj 는 헤드마다 게이트를
옆에 두어 binder 가 가른다). `tests/test_engine_attention_family.py` 가 각 형식을 그것을 정의한 transformers 구현에 CPU 에서 붙잡는다:
glm5_next(MLA + DSA k-pool, 회전 없음), deepseek_v3(dense MLA, 두 회전, q_lora 유무 — K3·Ling 의 형), minimax_m3_vl(GQA + MSA),
inkling(윈도 + 상대 편향 + k/v conv, 전역 층의 log scaling), 싱크는 `sparse_attention.sparse_attn`(DSv4.1 커널 의미)에; Qwen 의
GQA + QSA + 게이트는 조립 테스트가 qwen4_exp 에. 참조가 커널을 따르는 곳 둘: 허용 위치가 없는 행은 0, 인덱서의 동점은 앞 풀/블록으로.
**MoE 도 가족이다(modules/moe).** 일곱 모델의 채널 믹서는 라우터 하나 + 전문가 루프 하나에 축이다: 점수(softmax | sigmoid), 선택용 보정
편향(가중치엔 안 들어간다), 그룹 선택(noaux_tc: n_group·topk_group), 정규화, routed scaling 이 가중치에 붙느냐 출력에 붙느냐, 라우터가 fp32 냐,
활성(silu | GLM 의 clamped swiglu | M3 의 swigluoai), 공유 전문가의 결합(plain | Qwen 의 sigmoid 게이트 | Inkling 의 라우터 sink — 공유 전문가의
로짓이 선택된 전문가들과 함께 정규화된다). `MoE`·`Dense` 가 특징이고 `named`/`experts_of`/`shared_of` 가 다섯 체크포인트의 이름과 공유
전문가 배치(분리 | fused | 쌓인 [S,…])를 잇는다. `tests/test_engine_moe_family.py` 가 각 블록을 그것을 정의한 transformers 구현에 붙잡는다:
glm5_next(그룹 없음/있음, clamped), deepseek_v3(grouped), minimax_m3_vl(swigluoai, 출력 scaling), inkling(sink, route_scale × global_scale),
각 모델의 dense MLP; Qwen 의 softmax + sigmoid 공유는 조립 테스트가 qwen4_exp 에. 전문가의 양자화 형식(NVFP4·GPTQ·FP8·MXFP4)은 로더 쪽
`expert(layer, e)` 의 일이라 특징의 축이 아니다.
**잔차 형식도 가족이다(modules/residual).** 서브층이 잔차를 읽고 쓰는 방식은 다섯: `PreNorm`(x = norm(h), h += out; Inkling 은 out 에 fp32 짧은
conv 를 더한 뒤 — 잔차 형식이 시퀀스별 상태를 들고 `cache_specs` 로 선언한다), Qwen 의 `GatedResidualStreams`(hyper_connection), `HyperStreams`
(mHC: hc 스트림, 서브층마다 노름된 스트림의 선형 하나가 pre·post·comb 를 주고 comb 는 Sinkhorn — GLM-5.3 은 헤드가 평균, DeepSeek-V4 는 가중
collapse), `AttnRes`(Kimi K3: 잔차는 현재 블록의 합, 블록 경계마다 저장, 서브층 입력은 깊이 방향 softmax 혼합 — modeling 코드 인용, 로컬
오라클 없음). Residual 프로토콜의 enter/leave 가 step·state 를 받는다. `tests/test_engine_residual_family.py`: HyperStreams 를 glm5_next 디코더
층(서브층은 HF 모듈 그대로, 잔차 형식만 우리 것)과 deepseek_v4 헤드에, PreNorm 을 deepseek_v3 층과 inkling 층(출력 conv 포함)에, 조각 == 통짜.
**해시 n-gram 메모리도 가족이다(modules/ngram_embedding).** Qwen3.8 의 PLE 와 DeepSeek-V4.1 의 engram 은 해시 하나와 게이트-쓰기 하나다.
해시(`NGramHash`): 창의 규칙(시퀀스 시작·dead(이미지) 토큰에서 멈춤, Qwen 은 한 칸 이상 뒤의 EOS 에서도), 키(토큰 id | 정규화해 겹치는
토큰 맵 — `normalized_token_map`), 곱수(splitmix | numpy rng), 버킷(둘 다 base 이상의 연속 소수, 표 t·차수 o·헤드 h 순). 게이트-쓰기
(`NGramInjection`): key·value 투영(둘 | wkv 하나), 노름 순서(separate: 따로 노름해 반올림 뒤 내적 | joint: fp32 곱 × rsqrt 두 개의 곱),
(1+w) | w, 부호 붙은 sqrt 의 0 처리(sign | copysign), 팽창 conv(Qwen) 유무. 표의 역양자화는 `table` 호출의 일(DSv4.1 은
`block_fp8_rows`). `tests/test_engine_ngram_family.py`: PLE 를 qwen4_exp 의 해시 id·층에, engram 을 **벤더 DeepSeek-V4.1 추론 코드**
(srv4 체크포인트의 inference/engram.py·model.py, MIT, model.py 는 profiles/dsv41/caches.py 가 핀한 sha; git 제외 .oracle-site/dsv41) 에 —
실제 config 의 소수·곱수(합 == 표 높이 384,006,168 / 384,016,682), 토큰 맵, 이미지 스팬과 청크를 넘는 해시 id, 조회는 torch.equal,
fp32 쓰기도 비트 동일; bf16 서빙 경로는 반올림 두 번 거리 안.
**조립이 서빙된다(base/composed).** `PositionStore` 는 같은 State 계약을 엔진의 메모리 위에서 답한다: 특징의 토큰별 행
(`put_rows`/`rows`)은 BlockPool 의 블록에 **위치**로 산다 — 블록은 위치 // 블록 토큰, 행은 위치 % 블록 토큰 — 그래서 시퀀스의
이력은 그 블록표이고 캐시된 prefix 의 블록은 복사 없이 입양된다(base/prefix). 시퀀스별 값(`get`/`put`)은 고정 슬롯에 살고, 슬롯의
바이트는 하나의 연속 영역이라 티어가 옮기고(base/tiered_kv) prefix 경계의 상태는 그 영역을 스냅샷에 한 번 복사한 것이다. 특징이
선언한 캐시 명세(base/cache_spec: key·dtype·shape)가 그 배치를 정한다. `ComposedModel` 은 어떤 조립이든 러너의 Model 계약과
도어(base/serve)의 엔진 면에 답한다: 행별 토큰과 한도, prefix 캐시의 표시에서 나누는 prefill, 디코드 스텝당 토큰 하나(드래프터
없음, horizon = context + 1), base/sampler 와 base/draws(GLM 과 같은 키)로 샘플링, 슬롯 바이트 옆의 호스트 기록으로 파킹.
`tests/test_engine_composed.py`: 저장소 == 참조 State(같은 로짓), 경계 checkpoint/restore 와 블록 입양, 러너를 지난 탐욕 생성 ==
참조 루프, 끝 토큰과 min_tokens, 두 번째 턴, prefix 재사용(8토큰 입양), 파킹 기록·슬롯 바이트로 재개, world 1 도어로 요청 하나.

**드래프터는 조립 위에서 검증된다(base/composed, 2026-09-13).** `ComposedModel(drafter=...)` 는 GLM-5.3 의 위치 검증을 어떤 조립에든
준다: 디코드 스텝이 [마지막 토큰] + 행의 드래프트를 verify 세그먼트로 넣고, 각 위치를 드래프트가 없었을 때 같은 생성 번호가 뽑았을
균등수로 샘플해, 샘플이 드래프트와 같은 동안 수락하고, 수락 + 1 개를 붙인다 — 드래프터는 한 스텝이 내는 토큰 수만 바꾸고 어떤 토큰인지는
못 바꾼다. 틀(base/composition)의 계약: verify 세그먼트에서 시퀀스별 값을 드는 특징(GDN 순환·conv, n-gram 문맥·conv, 잔차·k/v conv)은
각 토큰 뒤의 값을 `put(..., at=j)`(`put_state`) 로 남기고, `accept(seq, n)` 이 토큰 n-1 뒤의 값을 현재로 만든다. `PositionStore` 는 그
값들을 슬롯의 링(`ring` 부)에 두고 수락 때 복사해 온다; 거절된 위치의 행은 다음 스텝이 덮어쓴다. 되감기는 위치를 고르는 일이지 다시 계산이
아니다. 디코드 스텝이 블록 경계를 넘으면 러너의 prefix 체크포인트는 그 경계의 값을 링에서 읽는다. 드래프터 프로토콜은 `observe(seq, ctx,
next_ids, hidden)`(타깃이 닫는 믹스 전의 잔차 상태, `forward(hidden=True)`) 와 `propose(seqs)`. `tests/test_engine_speculative.py`: n 개
수락 == n 개 공급(참조 State·PositionStore), 러너를 지난 실행이 드래프트 없는 실행과 토큰 단위로 같다 — 완벽·일부·무작위 드래프트, 탐욕·샘플,
두 행, 수락 구간 안의 끝 토큰과 min_tokens, 두 번째 턴, prefix 입양, 경계 체크포인트.

**MTP 헤드도 조립이다(modules/mtp).** 일곱 모델이 모두 싣는 드래프터 — 타깃의 한 위치 상태와 그 다음 토큰을 읽어 그 다음을 맞히는
작은 헤드 — 는 같은 특징으로 된 `Composition` 이다: 층은 타깃의 층 뒤 `offset` 에 두어 행이 같은 블록에 살고(`store_for(also=...)`,
`merged_specs`), 헤드는 대개 타깃의 것, 드래프트 헤드를 만드는 건 `fuse` — (다음 토큰의 임베딩, 타깃의 상태)에서 여는 법이다. Qwen3.8 은
`fuse_streams`(vLLM 이미지의 qwen3_8_flash_next MTP: 최종 믹서 전의 멀티 스트림을 hc·H 전체로 (1+w) 노름해 스트림마다 한 행렬로 투영,
노름·투영한 임베딩을 모든 스트림에 더함; 층은 QSA + MoE 한 층, 믹서로 닫고 lm_head 공유), DeepSeek 계열은 `fuse_concat`(검증 안 됨).
`MTPDrafter` 가 base/composed 의 Drafter 다: 타깃이 확정한 위치들을 같은 위치에서 다음 토큰과 함께 한 번 돌려(`observe`) 행을 쓰고, 마지막
예측이 드래프트 1, 이후는 헤드 자신의 믹서 전 상태와 직전 드래프트로 한 칸씩 — 이 사슬의 행은 잠정이라 저장소의 lane(`PositionStore.lane`,
같은 블록·자기 문맥)을 타깃 문맥으로 되돌린다(`place`). `tests/test_engine_mtp.py`: fuse == vLLM 전방 계산 전사, 제안 == 참조 State
위에서 처음부터 다시 굴린 결과(프리필 조각·수락된 드래프트의 여러 위치 관측 후에도), 헤드를 붙인 실행 == 붙이지 않은 실행, 층 두 개 사슬,
prefix 입양. `boot.py --mtp K` 가 체크포인트의 헤드로 드래프트한다.

Qwen3.8 은 이 길로 실제로 돈다. `engine/profiles/qwen38/weights.py` 가 srv2 의 체크포인트(206 샤드, `model.language_model.` 이름)를
조립의 `tensor(name)` 으로 읽는다 — bf16 은 한 번 읽어 쥐고, NVFP4 전문가(modelopt 네 텐서)는 `modules/moe.dequant_nvfp4` 로 요구 시
역양자화해 유계 캐시에, PLE 표(128 샤드 × [2,500,012, 160] e4m3)는 행 번호로 샤드에서 바로 모아 표의 스칼라 `weight_scale` 을 곱한다(그 스칼라를
빼먹으면 행이 수십 배 커져 답이 헛소리가 된다 — 실가중치가 찾은 버그, `tests/test_engine_qwen38_weights.py` 가 합성 체크포인트로 지킨다);
safetensors 라이브러리 없이 헤더와 numpy memmap 뿐이다. `python3 -m engine.profiles.qwen38.boot --ckpt DIR --chat --prompt ... --max-new N` 이 토크나이저·챗 템플릿·조립·
저장소·러너·도어를 잇고(`--tiny` 는 합성 체크포인트로 배관만, `--serve` 는 문을 연 채로), 요청이 문으로 들어가 같은 러너·스케줄러·
블록 풀·슬롯 풀·prefix 캐시를 지나 토큰이 나온다. 참조 레인이다: 특징은 torch 수식을 부르고 저장소는 커널 대신 행을 모아 준다. 서빙
레인(커널·글루·캡처 그래프)을 같은 특징 뒤에 묶는 것과 MTP 가 다음이다. GLM 의 net.py 는 그대로다.

서빙 레이아웃 v3(`st-qwen38-tep4-modelopt-v3`, 2026-09-18, 운영자 지시): `profiles/qwen38/preshard.py` 가 NVIDIA 허브 체크포인트
(`nvidia/Qwen3.8-Flash-Next-NVFP4` @ fc694b54, `quant_algo MIXED_PRECISION`; srv2 `~/models/qwen38-flash-next-nvidia-nvfp4`, sha256 검증)를
네 랭크 파일과 네 PLE 표 파일로 자른다. PLE 표(47.68 GiB, 랭크당 11.92 GiB)는 랭크 파일에 들어가지 않고 `ple-r{r}of4.weight`(그 랭크의
32 샤드를 그대로 이어 쓴 e4m3 행; `ple-r{r}of4.json` 이 행 수·폭·샤드·스케일·sha256)로 옆에 놓이며, 서빙 넷은 행을 번호로 SSD 에서
읽는다(`profiles/qwen38/ple_table.py`: 스레드 pread — srv2 NVMe 실측 128행 1.0 ms·20,000행 71 ms; 즉시 스텝은 호스트에서 해시해
모으고, 캡처 스텝은 재생 전에 `net.stage_ple` 가 그래프의 정적 스테이징 행을 채운다). MTP 헤드의 전문가는 허브 체크포인트에서 FP8
블록스케일(`weight_scale_inv` 를 곱한다 — BF16 복사본 대비 2.66%)이라 역양자화 뒤 NVFP4 로 인코딩한다(드래프터는 수용률만 바꾼다);
예전 복사본(NVFP4 만, BF16 MTP)도 `facts.load` 가 읽는다. 전문가 그룹은 랭크별이라 프리샤드 상주 메모리는 약 2 GiB 다. 첫 서빙
(2026-09-18, `launchers/start-st-qwen38.sh`, 세션 창): 즉시 프리필은 b12x dynamic 커널로 간다(expert-local 한 행 한 라우트, 8 행
초과 — 정적 커널은 행 수마다 아티팩트를 만들어 첫 창에서 토큰을 내지 못했다), one-shot 집합통신은 기본 그대로(스톨 0; `--no-oneshot`
/`ST_ONESHOT=0` 은 옵션). 네 랭크 53.5 s ready, 디코드 35.3 ms/스텝·1.74 토큰/스텝(MTP K=1 수용 72%), 450 토큰 생성 39.4 tok/s,
웜 프리필 약 3.5K tok/s(`measurements/qwen38_fleet_boot_20260918`). D17 의 onepass 기록은 아직 없다. MTP 체인 K>1 은 `--spec-k K`
(launcher `ST_SPEC_K`; 기본은 체크포인트의 1): 드래프트 재생 하나 안에서 헤드를 K 번(`decode_graphs.draft_chain` — 관측 스텝 뒤 행마다
한 위치씩 K−1 런치, 직전 pick 이 토큰·헤드의 streams 가 상태), 검증 스텝은 행당 K+1 토큰, 고정 링(QSA raw key·PLE id 8)은 K ≤ 4
(`caches.check_rings`). 두 행 C=1 실측(같은 날 창 3): 450 토큰 K=3 2.43 토큰/스텝·44.3 ms/스텝·46.2 tok/s vs K=1 1.69·34.7·39.5;
4 행은 (3 행 × 4) 12 토큰이 micro 상한 8 을 넘어 정적 MoE 커널로 가고 그 첫 런치가 캡처에서 죽어(illegal access) 격리 전이다.

빠른 확인(GLM-5.3, 실가중치, 한 노드, TP=4 스레드; 랭크 파일은 `profiles/glm53/preshard.py` 가 한 번 자른다):

    PYTHONPATH=. python3 engine/profiles/glm53/check.py --layers 0-4              # 참조 레인: 랭크 동일 + 청크/verify/롤백 판정
    bash engine/runtime/build.sh                                                # vLLM이 제거된 ST 이미지
    bash probes/run_engine_check.sh --layers 0-4                                  # ST 서빙 커널 레인
    PYTHONPATH=. python3 engine/profiles/glm53/boot.py --local --layers 0-4       # 러너가 돈다 (+ --drafter DFlash2, --park NVMe 파킹, --serve HTTP 문)

플릿(스파크 4대, 각 노드에 ST 이미지를 빌드한 뒤 노드당 컨테이너):

    bash launchers/fanout-st-ranks.sh            # 랭크 r 파일을 노드 r 로
    bash launchers/start-st-glm53.sh             # 부팅; glm53*/q38* 컨테이너가 있으면 거부
    bash bench/fleet.sh run --gpu st 30 "decode graph" -- bash probes/run_engine_probe.sh probes/engine_decode_graph_check.py
                                                 # GPU 검사는 벤치 큐에 줄을 선다(2026-09-12): 창이 없으면 미루지 말고 예약한다.
                                                 # 큐는 st-* 컨테이너가 떠 있으면 허가하지 않고, 런처는 큐에 holder 가 있으면 거부한다.
                                                 # GPU 하나면 되는 검사(--distributed 없는 ST 검사)는 플릿을 잡지 않고 srv4 한 대에서 프로덕션 옆에 돈다:
                                                 # 단일 GPU 레인(holder-single). 증거는 그 박스의 여유 메모리(--test 부팅과 같은 16 GiB 바닥)이고,
                                                 # 플릿 부팅과는 한 박스를 나누지 않는다. 랭크 파일 넷이 다 필요한 검사는 run --gpu --fleet.
    curl -s http://10.10.10.2:8000/v1/engine/completions -d '{"prompt": "...", "max_tokens": 64}'                    # 엔진 방언: ids/text
    curl -s http://10.10.10.2:8000/v1/engine/completions -d '{"conversation": 0, "prompt": "...", "max_tokens": 64}'   # 파킹된 대화 이어가기
    curl -s http://10.10.10.2:8000/v1/completions -d '{"prompt": "...", "max_tokens": 64, "n": 2, "logprobs": 3}'    # OpenAI completions
    curl -s http://10.10.10.2:8000/tokenize -d '{"prompt": "..."}'; curl -s http://10.10.10.2:8000/detokenize -d '{"tokens": [1, 2]}'
    STK_context_ceiling=131072 bash launchers/start-st-glm53.sh   # 선언된 D11 노브는 STK_* 로 부팅에 들어간다(미선언·만료 = 사망)
    bash launchers/start-st-glm53.sh stop        # 컨테이너 제거 + 잠금 해제. start 는 glm53*/q38*/vllm*/st-* 컨테이너나 srv2 의 `st-fleet.lock` 이 있으면 거부한다
    bash launchers/start-st-glm53.sh held         # 누가 쥐고 있고 무엇을 하는 중인지(엔진이 리스에 계속 쓴다)
    bash launchers/start-st-glm53.sh yield "이유"  # 죽이지 말고 넘겨받기: 엔진이 받던 요청을 끝내고
                                                 # 대화를 NVMe 로 파킹한 뒤 리스를 놓는다. 다음 보유자가 그 대화를 이어받는다.
                                                 # (플릿을 쓰는 세션은 모두 이 잠금을 지킨다: 09-11 19:42 두 세션의 플릿이 같은 노드에서 충돌해 둘 다 죽었다)
    curl -s http://10.10.10.2:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"..."}],"max_tokens":64,"stream":true}'   # OpenAI 방언(SSE), bench/onepass.py 가 쓰는 것
    curl -s http://10.10.10.2:8000/v1/models; curl -s http://10.10.10.2:8000/metrics                                  # 모델 이름, 벤치 이름의 카운터

`/metrics`(프로메테우스 텍스트, HELP·TYPE 포함): 벤치 방언(`vllm:request_success_total`·`num_requests_{running,waiting}`·`prompt/generation_tokens_total`·`spec_decode_*`·`iteration_tokens_total_count`)은 이름과 의미 그대로 유지하고, 그 위에 **지연 히스토그램 셋**(`vllm:time_to_first_token_seconds`·`time_per_output_token_seconds`·`e2e_request_latency_seconds`, 요청 도착 시각 기준), **포화도**(`vllm:gpu_cache_usage_perc`·`st:kv_blocks_{total,used,free}`·`st:state_slots_{total,free}`), **재사용**(`vllm:prefix_cache_{queries,hits}_total`·`st:prefix_cache_*`), **스텝 종류**(`st:steps_{prefill,decode}_total`, D9), **티어**(`st:conversations_parked`·`st:tier_bytes_*`), **취소·타임아웃**(`st:requests_{cancelled,timed_out}_total`)을 낸다. 비동기 경로를 판단할 수 있도록 `st:async_decode_steps_total`, `st:sync_drain_steps_total`, `st:decode_row_steps_total`, `st:decode_batch_capacity`도 낸다.
vLLM 이 낼 수 없는 것(이 엔진에만 있는 부품이라): **어느 캡처 그래프가 돌았나**(`st:decode_steps_by_sequences_total{sequences}` = 스케줄러가 실제로 채운 배치, `st:decode_capacity_bucket_total{capacity}` = `STK_context_ceiling` 을 자를 유일한 프로덕션 증거), **스텝 벽시계**(`st:step_seconds{kind}`, 호스트 관측 종단 — 두 종류 모두 샘플 읽기로 끝나므로 발사 시간이 아니라 스텝 전체다), **수용 분포**(`st:spec_accepted_per_step_total{accepted}` — 평균이 아니라 모양이 `spec_k` 를 정한다), **무엇이 실제로 묶였나**(`st:lane_info{lanes,moe_static,mla_prefill,spec_k,context_ceiling,dense_w4a16_guard_rows}` — "무장 ≠ 서빙"을 부팅 로그가 아니라 스크레이프로 판정).
비용(실측): 렌더 0.096 ms·11 KB·190줄(스크레이프당 1회), 관측 0.96 µs(디코드 스텝 최악 24회 = 46 ms 스텝의 0.05%). 디바이스 읽기·동기화 없음.

문(`base/serve.py`): 엔진 방언(`POST /v1/completions` ids|prompt, `conversation` 으로 이어가기)과 OpenAI chat 방언(`POST /v1/chat/completions`,
`stream` 이면 토큰 단위 SSE, `chat_template_kwargs` 통과, `</think>` 앞은 `reasoning_content` 뒤는 `content`; `GET /v1/models`, `/metrics`, `/health`).
프로필이 템플릿(`chat_template_mm_v2.jinja`, 프로덕션과 같은 것)과 `</think>` id 를 넘긴다.
GLM-5.3-Flash의 `reasoning_effort`는 생략하거나 `null`이면 `high`다. `low`·`high`를 허용하며,
OpenAI 호환 값 `medium`과 `max`는 `high`로 매핑한다. 최상위 필드와 템플릿 옵션을 먼저 정규화하므로
`max`와 `high`를 함께 보내도 같은 값으로 처리한다. 체크포인트에 이전 템플릿이 있어도 같은 정책을 적용한다.
요청은 `stop`(문자열 ≤4, 내용 채널에서 잘라 조기 종료), `min_tokens`(그 전엔 끝 토큰 불가), `tools`(템플릿이 렌더, 답의
`<tool_call>` 은 `tool_calls` 로 파싱, finish `tool_calls`), `n`=1 만, `logprobs` 는 400. 클라이언트가 끊으면(소켓 EOF·broken pipe) 요청을
취소해 행을 돌려주고, `REQUEST_TIMEOUT_S`(3600) 를 넘긴 요청은 504 로 취소한다 — 취소는 rank 0 이 도착과 같은 브로드캐스트로 실어 네 랭크가
같은 반복에서 같은 행을 버린다(`/metrics` 의 `st:requests_cancelled_total`).
그림(45차 §23 A7, `profiles/glm53/vision.py`): chat 의 `content` 배열에 `image_url`(≤4)·`video_url`(≤1, 프로덕션 PR #431 의 MM_LIMIT) 파트를
받는다 — `data:` 또는 http(s) URL. rank 0 의 문이 프로덕션 프로세서와 같은 규칙으로 캔버스를 만들고(28 정렬·pad·bicubic, 영상은 32 프레임
균등 → GLM 샘플러 → 프레임 쌍마다 `<|begin_of_image|>…<|end_of_image|>N.N seconds`), 템플릿의 자리표시자 하나를 토큰 수만큼 늘려 보낸다;
캔버스는 요청과 함께 네 랭크로 가고 각 랭크가 같은 비전 타워(`vision.safetensors`, 랭크 파일 옆; `preshard.py --vision` 으로 한 번)를 돌려
자리표시자 위치의 임베딩을 바꿔 넣는다(D3: 네 랭크의 결과 합이 다르면 죽는다). 같은 자리표시자에 다른 그림은 다른 프롬프트다 — 이어가기(B1)와
prefix 캐시 둘 다 그림의 digest 를 본다. 플릿 부팅은 `vision.safetensors` 가 없으면 서지 않는다(프로덕션이 그림을 서빙하므로).
prefix 재사용의 단위는 풀의 **블록 768**(`facts.BLOCK`; 프리필 청크의 정렬 2,304 = `facts.CHUNK_ALIGN` 는 그대로, 6,912 법칙 불변)이다
(`base/prefix.py`): 청크(6,912) 안의 여덟 경계는 스텝 전에 `marks` 로 이름 붙여 스텝이 도는 동안 받는다 — KDA 되풀이를 그 자리에서
끊어(정확: 커널 청크가 64 정렬 경계에서 다시 시작) 조각의 끝 상태와 conv 입력을 스냅샷에 쓰고, 드래프터 링은 스텝 전 링 + 경계 전
위치를 스냅샷에 관측한다(`adapter.prefill`). **생성 중에 넘은 경계**도 들어간다: 동기 스텝 뒤엔 링에서, 앞서 도는 스텝은 장치의
"경계 스테이지"(슬롯당 KDA 상태 + conv 탭, `caches.stage_boundaries`; 넘은 스텝만 쓴다)에 놓아 두고 호스트가 결과를 읽을 때
스냅샷으로 옮긴다(드래프터 링은 그때의 산 링: 넘은 뒤 몇 자리가 창의 가장 오래된 칸에 얹힐 뿐). 스냅샷 96개(경계당 ~45 MiB,
`boot.PREFIX_SNAPSHOT_GIB` 가 선언한 바이트에서 형상이 개수를 정한다 — native 4.25 GiB ≈ 45.3 MiB × 96); 자리가 모자라면 **한 번도 채택되지 않은 경계부터** 나간다(`prefix._victim`) — 긴 프롬프트 하나가 모두가
공유하는 시스템 프롬프트를 밀어내지 못한다. 밀려나는 **잎 경계**(다른 경계가 잇지 않는 것)는 NVMe **prefix 티어**에 남는다: 스냅샷이
`runner.spill_low_water`(8) 아래로 줄면 러너가 미리 잎을 써 두고(블록 + 스냅샷 45 MiB + 기록, 대화 티어 옆 `prefix/` 디렉터리, 키는
해시 56비트), 메모리에 없는 경계를 티어가 들고 있으면 요청 행에 읽어 들여(`restore_begin/finish`, 네 랭크 투표 뒤) 그 뒤부터 프리필한다
— 32K 프롬프트 재적중 ≈ 280 MB 읽기 vs 16 s 프리필. 같은 프롬프트가 동시에 오면 둘째는 **첫째의 프리필이 그 경계를 캐시할 때까지
기다렸다 채택**한다(`runner.shared_ahead`, 대기 중 다른 요청을 먼저 들여보내고 경계가 오면 맨 앞으로). `POST /v1/prefix/warm`
(`messages`/`prompt`/`ids`, `pin: true`)로 알려진 시스템 프롬프트를 미리 넣고 고정, `POST /v1/prefix/unpin` 으로 해제
(`probes/st_prefix_warm.py prompts.jsonl --pin`); `/metrics` 의 `st:prefix_{reused_tokens_total,entries,pinned_entries,tier_entries,
tier_spills_total,tier_restores_total,dedup_waits_total}`. 이어가기(B1)는 히스토리가 끝 토큰(`<|endoftext|>` 등, 템플릿이 되그리지 않는)으로 끝났으면
그 토큰 앞까지 맞아도 이어간다 — 그 토큰은 뽑혔지만 먹인 적이 없어 캐시가 정확히 그 앞에 서 있다(`extend(drop_unfed=True)`).
이어갈 대화를 찾는 훑기(`_continuation`)는 rank 0 의 요청 스레드가 락 밖에서 한다(긴 기록 비교가 입장을 막지 않게). 그 사이 루프가
행을 비우거나 파킹·재개한 후보는 건너뛰고(2026-09-19: `history_ref` 의 `KeyError` 로 채팅 ~530건 중 2건이 응답 없이 끊겼다), 입장은
루프에서 **모든 랭크가 같이 가진 장부**로 그 대화가 아직 같은 기록인지 다시 비교해 아니면 새 프롬프트로 넣는다 — 사라짐·바뀜·다른 요청이
잇는 중(n>1 선택지는 모두 같은 대화를 가리킨다)·그림이 자름선에 걸침(`st:continuation_fallbacks_total{reason="gone|changed|busy|picture"}`).
파킹이 착지 중인 대화와, 턴이 취소되어 되읽힌 뒤 그대로 다시 파킹될 대화만 기다린다(게이트웨이의 재시도가 긴 기록을 새로 프리필하지
않게): 티어는 랭크마다 다른 순간에 착지하므로 그것만으로 가르면 랭크가 갈라진다(`_stale_hint`). 훑기가 티어에서 읽어 온
파킹 기록·다이제스트는 runner 가 자기 락(`Runner._book`) 아래에 두고, 읽는 사이 루프가 그 대화를 파킹·재개·잊었으면 버린다. 파킹됐는지
(`is_parked`)는 티어에만 묻는다: 그 캐시는 rank 0 만 채우고, 거기서 답하면 rank 0 혼자 다음 파킹을 "already parked"로 거절해 투표가
그 대화를 모든 랭크에서 버렸다. 디코드는
**호스트보다 앞서 돈다**(`profiles/glm53/pipeline.py`, vLLM 의 비동기 스케줄링): 타깃
그래프 → 샘플러 → 커밋(`base/sampler.commit_batch`) → 마스크 관측 → 제안 → 다음 스텝 ids 가 장치에 남고, 결과만 핀 버퍼로 건너와
다음 스텝이 이미 도는 동안 읽힌다(`runner.inflight`, 깊이 2). 장치에서 끝난 행은 상태 슬롯을 null 슬롯으로 돌려 유령 스텝이 링에 아무
것도 못 쓰고, 러너는 한 스텝 늦게 끝을 알아 그 유령의 결과를 버린다. 온도/top_k/top_p·요청별 seed·페널티·logit_bias·logprobs·min_tokens
미충족 행도 이 비동기 경로를 사용한다. 요청별 seed는 기존 난수 키와 같은 값을 만드는 nonce로 넘기며, 페널티 이력은 장치에서 실제로 커밋된
토큰만 센다. 검증 블록의 각 위치에 draft 접두사를 반영하고, min_tokens에 도달하기 전 위치에서는 EOS와 stop_token_ids를 금지한다.
greedy 옵션 행은 로컬 어휘 조각을 FP32로 처리한 뒤 기존 MAX 집단통신으로 고르고, 확률 샘플링·logprobs 행은 전체 어휘를 모은다.
logprobs는 선택 토큰과 상위 후보만 연속 핀 버퍼로 옮겨 기존 결과 이벤트 뒤에서 공개하며, 거절된 draft·생성 한도 뒤 토큰·유령 스텝은 기록하지 않는다.

페널티·logit_bias·logprobs·미충족 min_tokens는 깊이 2의 일반 비동기 체인을 사용한다. 기존 4회 greedy 버스트는 기본 채팅과 중립 옵션
(예: penalty=0, repetition_penalty=1, greedy seed)에 유지된다. 문법 matcher와 reasoning_budget 경계는 호스트 판정이 필요해 여전히
먼저 체인을 비운다. 문법 행도 위치별 로짓 변환을 한 GPU 블록으로 처리하고 logprobs를 일괄 읽으며, matcher의 draft 롤백·커밋 규칙은 유지한다.
새 옵션 커널과 FP32 샘플러 변형은 부팅 때 준비한다. 로컬 CUDA 수치·그래프·상태 전이 검증과 처리 단계 측정은
[요청 옵션 비동기화 기록](../measurements/st_async_options_20260915/README.md)에 있다. 엔진 전체 TP4 속도는 별도 측정 대상이다.
루프의 도착 브로드캐스트와 투표는 gloo 제어 그룹(`Comm.control`)으로 간다 — NCCL 그룹의 객체 브로드캐스트는 스텝의 커널 뒤에
줄 서고 읽기 위해 장치를 기다린다.

GB10 고정 구성의 실행 순서 실험은 [측정 계약](../measurements/st_execution_plans_20260913/README.md)에 정리했다.
C=4 통신·연산 겹치기, 타깃 후반부와 DFlash2 문맥 투영 겹치기, 레이어별 프리필 창을 각각 선택할 수 있다.
기본값은 모두 꺼져 있으며 KDA 상태는 FP32를 유지한다. 실제 원패스 품질·수용률·속도 증거가 있어야 채택한다.

GLM-5.3 기본 동시 요청 상한은 `MAX_SEQS=2`이며 C=1·2만 캡처한다(K=7 기준 검증 8·16행). 세 번째 요청부터는 빈 행을 기다린다.
텐서 병렬 구성은 GB10 네 대(TP4)다. 빈 디코드 행은 대기 상한을 기다리지 않고 다음 프리필 경계에서 입장하며,
기존 디코더가 있으면 프리필 예산을 2,304토큰으로 줄여 프리필 한 청크 뒤에 디코드 한 스텝을 실행한다. 프리필·디코드는 여전히 혼합하지 않는다.
새 행이 합류하거나 행 순서가 바뀌어도 살아 있는 행의 장치 컨텍스트·draft·샘플링 분포를 보존해 다시 제안하지 않고, 실제로 바뀐 행에만 호스트 무효화를
표시한다. 요청 취소·재사용·이어가기는 해당 행을 참조하는 pending prefix만 수거하므로 다른 행의 앞선 디코드를 불필요하게 비우지 않는다.

커밋과 로짓 선택에는 작은 장치 경계를 줄이는 경로가 있다. `decode_commit`은 수용 토큰·EOS·생성 한도·null 슬롯 전환·상태 갱신을 한 Triton 행 프로그램으로 처리하고,
`vocab_candidates`는 전체 FP32 어휘 임시 버퍼 없이 1,024개 부분 최댓값을 만든다. TP4의 1~64개 int64 후보 MAX는 one-shot transport를 사용하고 그 밖의 크기·형상은 NCCL로 남긴다.
`Comm.all_gather`는 rank별 리스트와 `cat` 대신 직접 rank-major 출력 버퍼에 수집한다. 이 변경은 CUDA Graph를 하나의 영속 transformer 커널로 합치지 않으며,
레이어별 mHC/KDA/MLA/MoE 경계와 row-parallel 집단통신은 모델 의미상 남아 있다.

운영(`launchers/st-glm53-supervisor.sh` + `st-glm53.service`, 헤드 srv2 의 사용자 유닛): 30 s 마다 진짜 4 토큰 chat 으로 건강을 재고(문이
열려 있어도 링은 죽어 있을 수 있다), 3 회 연속 실패면 포렌식(네 랭크 로그·free·nvidia-smi·metrics → `~/glm53-logs/st-forensics/`) → stop →
start. 재시작 간격은 60 s 부터 두 배씩 30 분까지, 5 회 실패 뒤엔 멈추고 사람을 부른다. 프로덕션 vLLM·q38 컨테이너가 보이면 절대 띄우지 않는다.
`ST_SUPERVISOR_ONCE=1` 로 한 사이클만 판정할 수 있다. 한 노드는 **자기 자신에게 ssh 하지 못하므로**(srv2 가 자기 키를 거부한다) 런처와
슈퍼바이저는 대상 IP 가 자기 것이면 로컬 셸로 돌린다 — 그래서 헤드에서 도는 슈퍼바이저가 rank 0 의 컨테이너·로그·잠금을 본다.

프로덕션 모델 선택(2026-09-19, 운영자: 데네브에서 엔진의 모델을 고른다): 프로덕션이 어느 모델을 서빙하는지는 한 파일
`~/glm53-logs/st-production.json` 이 정하고, 슈퍼바이저·deploy-watch·prebuild 가 모두 `launchers/st_production.py` 로 그것을 읽는다.
파일이 없으면 glm53 — 고른 적 없는 박스는 예전 그대로다. 선택이 바뀌면 슈퍼바이저가 문이 조용해지길(최대 `ST_SWITCH_QUIET_S`, 120 s)
기다렸다가 돌던 플릿을 내리고 고른 프로필을 같은 production 리스로 띄우며, 단계마다 `st-production-state.json` 에 적는다(데네브가 ssh
로 읽는 것). 창(티켓·세션)이 플릿을 쥐고 있으면 아무것도 내리지 않고, 창이 끝난 뒤의 프로덕션 부팅이 새 모델이 된다. 고른 모델이
`LAUNCH_HOLD_AFTER` 번 연속 못 뜨면 HELD 대신 glm53 으로 되돌리고 이유를 선택 파일에 남긴다 — 프로덕션이 못 띄우는 모델은 프로덕션이
아니다. 프로덕션의 트리·이미지는 모델과 무관하게 프로덕션의 것이고(ST 이미지에는 모델이 없고 릴리스는 엔진 트리 전체다), 모델마다 다른
것은 런처·컨테이너 이름·문이 답하는 모델 id·실행 환경뿐이다. 실행 환경은 프로필마다 `~/.config/st-<profile>.env` 이고, 다른 모델의
부팅은 어느 프로필이든 정하는 키(예: st-glm53.env 의 `RANKS_DIR`)를 전부 지운 뒤 자기 것만 얹는다 — `ST_REPO`·`ST_ENGINE_DIR`·
`ST_IMAGE` 는 프로덕션의 것이라 유지된다. D17 표본(onepass 프로브)은 GLM-5.3 의 계열이라 프로덕션이 다른 모델일 때는 걸지 않는다.

    python3 launchers/st_production.py show                            # 선택·상태·서빙 가능한 프로필
    python3 launchers/st_production.py select qwen38 --note "why"      # 다음 사이클(≤30 s)에 전환

프로덕션 전환: 프로덕션 vLLM 을 되살리는 경로는 `fleet-idle-recovery.timer`(5 분 유휴 뒤 복구) 하나뿐이다. ST 가 프로덕션이 되는 동안은
그 타이머를 끄고(`st-glm53.service` 의 `Conflicts=`가 같은 일을 한다) 슈퍼바이저 유닛을 켠다. 되돌리기는 그 반대 순서다:

    systemctl --user disable --now st-glm53          # (헤드)
    bash launchers/start-st-glm53.sh stop            # 네 노드 컨테이너 + 잠금 해제
    systemctl --user start fleet-idle-recovery.timer # 5 분 유휴 뒤 vLLM 복귀

프로덕션은 `ST_PRODUCTION=1`로 실행한다. `boot.py --production`은 네이티브 dense·one-shot AR·TP4 GPTQ 드래프터·프리필 SP, `t,r,sf6,batch,q0` MoE, 검증된 tile32 MLA 대형 프리필과 전체 컨텍스트,
served 레인, 캡처 decode를 고정한다. 실험 노브를 선언하지 않아 실험 만료일이 지난 뒤에도 같은 릴리스로 재시작할 수 있고,
`STK_*`를 섞으면 부팅을 거절한다. `ST_KV_GIB`는 명시적인 KV 바이트 예산을 `--kv-gib`로 전달하며, 미지정 시 프로필 기본값 24GiB를 사용한다. 실험은 기존 기본 실행 모드와 만료 규칙을 사용한다.

`st-glm53.service`는 `~/.config/st-glm53.env`를 읽는다. `ST_REPO`와 `ST_ENGINE_DIR`를 동일한
`/home/choiceoh/st-releases/<commit>`으로, `ST_IMAGE`를 `st-engine:prod-<commit>`으로 고정하고,
유닛의 `ExecStart`도 그 릴리스의 supervisor를 가리키는 drop-in으로 설치한다. 이렇게 하면 실험용
`~/st-engine`의 변경이 실행 중인 프로덕션의 소스에 반영되지 않는다. 기존 컨테이너의 자동 재시작은 끄고
헤드의 supervisor가 네 랭크를 함께 복구한다. 헤드 사용자에 linger가 필요하다.

런처는 잠금을 원자적으로 획득하고 준비 실패 시 자기 잠금만 해제한다. `stop`은 다른 실험의 잠금을 거절한다.
supervisor는 다른 `st-*` 컨테이너·외부 잠금·접속 불가 노드를 보면 재시작을 보류하며 실패 횟수도 소모하지 않는다.
전환 전 설정과 이미지 태그를 보존하고, vLLM 복구 timer는 `disable --now`로 재부팅 후에도 비활성화한다.

Prefix 재사용(`base/prefix.py`): 프롬프트를 블록(768 토큰, 6,912 청크당 9개) 단위로 해시 사슬을 만들고, 블록 경계마다 모델의 위치 링 상태
(KDA conv 탭 3개 + 재귀 상태 1개 × 34층, 드래프터 문맥 링; 인덱서 꼬리는 경계에서 비어 있어 제외)를 아레나의 스냅샷 슬롯(96개,
랭크당 ~45 MiB 씩; 상한은 `boot.PREFIX_SNAPSHOT_GIB`)에 두고 그 앞 블록들을 고정한다. 새 프롬프트는 자기 길이보다 짧은 가장 긴 캐시 경계를 **입양**(읽기 전용 공유
블록 + 슬롯에 상태 복원)하고 거기서부터 프리필한다 — 토큰 하나는 반드시 계산한다. 블록 소유는 개수(행 참조 + 캐시 핀)로, 예약이 모자라면
풀이 캐시에 LRU 회수를 요청한다. 랭크 넷이 같은 승인을 같은 순서로 하므로 메시지 없이 같은 캐시 상태다. 판정: `tests/test_engine_prefix.py`
(가짜 모델), GPU 는 `probes/engine_prefix_check.py`(같은 두 청크를 공유하는 두 프롬프트가 같은 토큰을 내야 한다).

GLM의 `served()`는 `engine/kernels`를 직접 호출한다. KDA·conv·mHC·kpool·MLA·b12x는
이 패키지 안에 있고, 인덱서와 mHC prenorm GEMM은 독립 `deep_gemm` 라이브러리를 사용한다.
vLLM 설치나 overlay 마운트는 필요하지 않다. 이식 출처, 라이브러리 버전과 빌드 계약은
[`kernels/README.md`](kernels/README.md), [`runtime/dependencies.json`](runtime/dependencies.json)에 있다.

가중치 없이 전체 커널 패키지와 GPU 수치 계약을 검사한다:

    ST_PROBE_NO_GPU=1 bash probes/run_engine_probe.sh probes/engine_kernel_check.py --imports-only
    bash probes/run_engine_probe.sh probes/engine_kernel_check.py

검사는 vLLM 임포트를 차단한다. GPU 검사는 KDA의 시작 상태·매 토큰 상태, conv 상태,
mHC pre/post, 유효 인덱서 로짓, kpool 바이트, MLA 부팅 판정·그래프 재생, b12x 출력을 확인한다.
b12x는 이식 전 FlashInfer 커널과 직접 비교하며, PyTorch 참조와 남은 오차는 별도로
기록한다. `--lanes moe --moe-experts 288`은 실제 TP4 전문가 형상을 검사한다.
커널 수치 검사는 실제 모델의 품질·처리량·ITL 판정과 별개다.

**이전 GLM 랭크 파일은 사전 샤딩을 다시 실행해야 한다.** b12x가 읽는 routed FC1은
`up | gate` 순서다. packed 가중치와 접힌 스케일을 이 순서로 저장하며, 파일 메타데이터의
`weight_layout=st-glm53-b12x-up-gate-v1`을 부팅·실가중치 검사에서 아레나 할당 전에 확인한다.
이전 `gate | up` 파일을 그대로 읽어 다른 모델을 실행하는 일은 허용하지 않는다.

커널 이식 검증은
[`st_engine_native_kernels_20260911`](../measurements/st_engine_native_kernels_20260911/README.md)에 있다.
이전 측정과 판정은 `MEASUREMENTS.md` 44~45차.

공통 실행부의 CPU 회귀 검증. **판정은 한 줄이다**:

    python3 tools/check.py                 # tests/test_engine_*.py 전체(현재 263), ok / FAILED / CANNOT RUN / skipped
    python3 tools/regress.py               # origin/main 과의 **차이**만 (절대 개수는 못 믿는다)
    python3 tools/mutate.py --tests tests.test_engine_prefix    # 내가 더한 줄 중 아무도 안 보는 것

`check.py` 가 **FAILED 와 CANNOT RUN 을 나눈다**. 전자만 종료 코드를 세운다 — 모듈이 임포트조차 안 되는 것은
테스트 결과가 아니라 환경 문제이고, 그 둘을 못 나누면 없는 버그를 쫓거나 있는 버그를 내보낸다. 스킵 수도 같이
나온다(파일·테스트·스킵 수는 설치된 휠과 장치에 따라 달라지므로 `check.py` 출력을 그대로 읽는다). 커널을 쓰기 전에는 `engine/INVENTORY.md` — 이미지 안에 이미 있는 것.

실행 소유권은 커널 표 → LocalTP → 개별 `run` 순서로 명시한다. `lanes.served(tp=tp)`가
만든 표는 해당 실행기에 고정되며, 다른 표의 생성이나 실행이 이 연결을 바꾸지 않는다.
플릿과 직접 워밍업은 `lanes.served()`로 호출한다. 전역 `bind_tp` 상태는 없다.

- 각 `run`이 배리어·통신 버퍼·결과·커널 대기열을 새로 소유하고, 끝날 때 참조를 반납한다.
- 같은 LocalTP의 중첩·중복 실행, 외부 스레드의 디스패치, 종료된 랭크 핸들의 재사용은 즉시 실패한다.
- 독립 LocalTP 둘은 각자의 호출 스레드에서 커널을 실행하며 통신 상태를 공유하지 않는다.
- 랭크·커널·스레드 시작 실패 시 대기 커널을 취소하고 시작된 랭크를 합류시킨 뒤 원인을 전달한다.
  다음 명시적 실행은 새 제어 상태를 받는다. 모델/KV나 CUDA 문맥의 복구·자동 재시도는 별도 책임이다.

부팅과 체인 검사도 이 소유권 경계를 따른다. 검증은
[`measurements/st_engine_execution_20260911`](../measurements/st_engine_execution_20260911/README.md)에 있다.

요청과 캐시의 소유권은 다음 경계에서 확정한다:

- `Runner.submit`은 입력 검증 → KV·상태 슬롯 예약 → `Model.open` → 스케줄러 등록 순서다.
  중간 실패 시 확보한 자원을 반납한다. `Model.close`는 실패한 `open`의 부분 상태도 정리해야 한다.
- 디코드는 `BlockPool.reserve_to`로 요청별 쓰기 끝 위치를 받아 배치 전체의 증가분을 먼저 검증·예약한다. 자원이 부족하면
  어느 행의 토큰 수도 바뀌지 않는다. 거절된 드래프트가 쓰던 공간은 다시 예약하지 않는다.
  블록 풀과 러너 상태는 러너를 소유한 스레드에서 갱신한다.
- 대기 상한은 디코드 폭에 여유가 있을 때 프리필을 허용한다. `max_running`이 꽉 차면 기존 요청이
  끝날 때까지 기다린다. 이미 실행 폭을 넘긴 상태는 오류이며, 앞쪽 요청만 잘라 실행하지 않는다.
  입장한 긴 프리필은 청크 사이에 실행 중인 디코드를 한 번씩 처리한다. 프리필 중인 요청의
  디코드 자리도 예약으로 취급하므로 유휴 대화 깨우기가 그 자리를 가져갈 수 없다.
- 파킹의 단위는 **대화**이지 행이 아니다. `Runner.park_begin(row, key)`는 유휴 행의 모델 호스트 기록(`Model.park`: 토큰 이력·문맥,
  `"context"`·`"pending"` 필수)을 떼고 블록과 상태 슬롯 바이트를 티어 스레드에 넘긴다; 행은 `retiring`(유휴도 빈 것도 아님)이고
  `park_finish(row)`가 쓰기 완료 뒤 **행과 슬롯을 반납**한다. `resume_begin(row, key)`는 빈 행 아무 곳에 블록을 예약하고 빈 슬롯
  아무 곳을 잡아 읽기를 넘기고, `resume_finish(row)`가 커밋한다(`Model.resume`은 슬롯을 지우지 않는다: 바이트는 티어가 되돌렸다).
  `transfer_done(row)`는 기다리지 않는다(D10). 쓰기 실패는 행을 유휴·상주로 되돌리고, 읽기 실패는 받은 행·슬롯·블록을 반납하고
  디스크 사본을 보존한다. `park`/`resume`은 두 반쪽을 이어 붙인 것(기다려도 되는 호출자용). 보존 대화 수의 상한은 `max_seqs`가 아니라 디스크다.
- `NvmeTier`는 파일 하나에 [블록들][슬롯 바이트, 섹터 패딩]을, 옆에 JSON 기록을 두고 manifest 에 셋을 게시한다. 파일시스템
  여유(기본 1 GiB 예비)나 선언한 `capacity_bytes`를 넘길 demote 는 한 바이트도 쓰기 전에 `TierFull`을 낸다.
- 티어는 노드마다 따로라 네 랭크의 티어가 어긋날 수 있다(다른 실행의 잔여물, 스텝 중에 갈라진 플릿에서 턴을 끝낸 랭크만 파킹한
  대화). 부팅은 **키별로 재조정**한다(`Server._reconcile_parked`, 캡처 전에 한 번·서버 생성자에서 한 번, 대화와 접두사 경계 둘 다):
  각 랭크가 (키, digest = 호스트 기록 + 필요한 블록 수)를 호스트 그룹으로 교환하고, 네 랭크가 같은 digest 로 가진 키만 남기고 나머지는
  가진 랭크에서 지운다. 표가 모든 랭크에서 같으니 결정도 같고, 남은 대화의 번호가 같다. 읽을 수 없는 기록은 어느 피어와도 맞지
  않는 digest 를 받아 지워진다. 예전에는 캡처 전 검사가 어긋나면 부팅을 죽였고(2026-09-13 13:04~13:20 프로덕션 재기동 여섯 번),
  생성자 검사는 어긋나면 전부 버렸다.
- `NvmeTier.run_async`는 `Future`를 반환한다. `.done()`으로 완료를 확인하고 `.result()`로 결과·오류를
  받는다. 전송끼리는 공유 스테이징 버퍼 사용을 직렬화한다. 디코더는 이 잠금이나 Future를 기다리지 않는다.
- 러너의 계측은 프리필·디코드별 누적 시간과 호출 수를 보존하고, 개별 스텝 기록은 고정 크기 링에 남긴다.
  부팅·진단 계측은 기존처럼 단계별로 보존한다.

이 검증은 스케줄링·자원 예약·실패 복구 계약을 확인한다. GLM 수치 일치, CUDA 전송 정확성,
NVMe 트래픽 중 디코더 ITL은 위 실가중치 검사와 `probes/kv_tier_interference.py`로 별도 판정한다.

GLM 조합을 실제 요청 실행에 연결하는 경로는 `profiles/glm53/runtime.py`의 `Glm53Runtime`이다.
`Glm53Net`과 `Glm53Caches`를 바인딩한 뒤 `submit(seq, ids, max_new_tokens)` →
`step()` → `take_result(seq)` 순서로 사용한다. 마지막 프리필에서 첫 토큰을 샘플링하고,
EOS나 생성 한도에 도달하면 그 스텝에서 KV와 상태 슬롯을 반납한다. 이 간단한 검증 경로는
`draft_slots=0` 계약을 사용한다. PR #534의 `Glm53Engine`·`boot.py`는 같은 캐시와 러너 위에서
DFlash2를 연결하며, 드래프터 문맥 링도 같은 아레나의 상태 슬롯 예산에 포함한다.

`Glm53Engine`은 전체 행의 temperature가 0인 스텝에서 rank별 유효 어휘 최댓값과 토큰 ID를
하나의 int64 후보로 부호화하고, 후보가 1~64개인 TP4 디코드에서는 one-shot MAX, 그 밖에는 NCCL MAX로 선택한다. 전체 로짓을 모으지 않으며, 동점은
가장 작은 전역 토큰 ID로 결정한다. 타깃 그래프는 rank별 로짓만 보관하고, 확률·혼합
샘플링 그래프가 필요할 때 전체 로짓을 모은다. DFlash의 후보 top-k는 기존 경로다.
이 스텝은 난수를 뽑지 않는다. 확률·혼합 스텝도 뽑지 않는다: **모든 균일난수는 입력**이다(`base/draws`,
45차 2026-09-13). 행 키는 (부팅 시드, 행의 입장 순번 nonce — 시드를 가진 요청은 그 시드만 —, 스텝 시작 시점의 생성 토큰 수)의
해시(splitmix64)이고, 한 난수는 그 키에 (용도, 위치) 워드를 섞은 것이다. 용도는 드래프트 walk(`DRAFT`)·타깃 선택(`PICK`)·검증
위치(`VERIFY`)·보정/보너스 뽑기(`FRESH`)·rich 샘플러(`RICH`). 호스트 정수와 int64 텐서 두 구현이 비트 단위로 같아서, 같은 뽑기는 네
랭크에서, 동기 경로에서든 장치 체인에서든, eager 든 캡처 그래프든 같은 수이고 그 전에 무엇을 뽑았는지에 의존하지 않는다 — 엔진에
스트림이 없다. 이전에는 랭크마다 Philox 스트림 하나를 문법 span·경로마다 다른 횟수로 전진시켜, 한 번의 차이가 플릿이 서 있는 내내 두
스트림으로 남았다(§97 의 미해결 용의자). 체인은 한 스텝이 한 행에 쓰는 전부(walk K, 검증 K, 보정 1)를 **드래프트 캡처 그래프 안에서**
해시 한 번으로 계산한다(`step_block`, 입력은 `b["nonce"]`·`b["generated"]`): walk 가 앞의 K 를 쓰고, 뒤의 K+1 은 `b["draws"]` 로 남아
다음 스텝의 검증이 쓴다(그 사이 생성 수는 그대로다). 동기 경로는 같은 워드를 호스트에서 계산해 캡처 샘플러에는 온도처럼 pinned
입력으로(`SamplingGraphs.run(..., uniforms)`), rich 행·walk 에는 텐서로 준다. 시드를 가진 요청은 여전히 동기 경로다(`CHAIN_BLOCKERS`).
디코드 입력은 모든 시퀀스와 draft를 평탄화해 한 번 업로드하고, 생성 한도 판정은 토큰 버퍼의
길이 차이로 계산한다. 결과를 수거할 때만 생성 이력의 독립 사본을 만든다.
구성요소 전후 측정과 회귀검사는
[`measurements/st_engine_decode_20260911`](../measurements/st_engine_decode_20260911/README.md)에 있다.

캡처 그래프는 커널 작업 버퍼의 **소유자도 보관**한다. MoE의 eager 캐시는 더 큰 배치나
프리필을 만날 때 버퍼를 교체하므로, 작은 배치 그래프가 참조하는 이전 버퍼를 그 캐시에만
맡길 수 없다. `DecodeGraphs(resources=...)`는 매 캡처 직후 소유자를 보관하고 모든 그래프를
reset한 뒤 반납한다. 전체 모델의 첫 디코드 정지 재현과 수정 검증은
[`engine_decode_replay_20260912`](../measurements/engine_decode_replay_20260912/README.md)에 있다.

HTTP 요청 번호는 내부 KV 행 번호와 분리한다. `Server`는 기본 64개의 미완료·미수거 요청까지
보관하고, 빈 행과 각 요청의 최대 생성 길이를 담을 블록 예산이 있을 때 FIFO 순서로 입장시킨다.
완료 결과를 복사한 뒤 일반 요청의 버퍼와 행을 반납한다. `keep_idle` 모드에서는 대화 ID와
요청 ID를 분리해 문맥을 보존한다. 단, 아무도 이어가지 않을 턴은 끝나면 **보존하지 않고 반납**한다: 요청이
`"retain": false`(ST 확장, 헬스체크·프로브용; 불리언이 아니면 400)라고 하거나, 문맥이 `park_min_tokens`(GLM 프로덕션 부팅 128,
`boot.PARK_MIN_TOKENS`; 서버 기본 0 = 끄기)보다 짧을 때다. 짧은 기록은 다시 프리필하는 편이 슬롯 상태(랭크당 약 256 MiB)를
티어에 쓰고 되읽는 것보다 싸다. 표시는 옵션(`_transient`)으로 브로드캐스트를 타고 문맥은 행의 것이라 네 랭크가 같은 턴을 반납한다.
반납한 대화 ID 로 이어가면 409, 채팅은 새 프롬프트로 처리된다. `st:turns_not_retained_total{reason="asked|short"}`.
슈퍼바이저의 헬스 ping 은 `retain: false` 를 보낸다. 티어가 없으면 유휴 대화가 행에 상주하고 새 요청에 공간이 필요하면
가장 오래된 유휴 대화를 정리한다. 티어가 있으면 끝난 턴의 파킹과 이어가기의 복원은 **티어 스레드**에서 돌고, 스텝 루프는
매 스텝 완료 여부만 묻는다(디코더는 디스크를 기다리지 않는다, D10). 완료·성패는 `all_reduce` 한 번으로 **네 랭크가 합의**한 뒤에
적용한다(락스텝): 모두 성공이면 행이 비거나 턴이 입장하고, 한 랭크라도 실패면 그 대화는 모든 랭크에서 버린다(복원 실패는 503),
전송이 한 랭크에서 **시작조차 못 해도**(`park_begin`·`resume_begin`·`restore_begin` 의 예외) 그 행은 서버의 장부에 남아 그 랭크가
"done, not ok" 로 투표한다 — 혼자 폴백하거나 혼자 죽는 랭크는 없다(45차: 그 분기가 나머지 랭크를 원샷 all-reduce 의 스톨 트랩으로 보냈다),
입장의 세 가지 랭크-지역 입력(접두사 캐시의 경계와 진행 중 판정, `kv.available`)도 한 번의 교환으로 합의한 뒤 분기한다(`admit:prefix`·`admit:fits`·`admit:reorder`),
모두 `TierFull` 이면 가장 오래 전에 파킹된 대화부터 잊고 다시 쓴다(잊을 것이 없으면 보존하지 않음). 파킹이 진행 중인 대화의
이어가기는 그 파킹이 끝날 때까지 줄에서 기다린다. 대화 ID는 부팅을 넘겨 유효하다: 새 서버의 요청 번호는 티어에 남은 가장 큰
대화 ID 위에서 시작하고, 엔진이 죽어도 파킹된 대화는 디스크에 남는다(정지 때 진행 중이던 전송은 기다려서 마무리한다).
알 수 없거나 실행 중·정리된 대화는 기존 요청을 건드리지 않고 409로 응답하고, 모델의 학습 위치(`facts.max_position`)를 넘는
문맥은 400 이다. 누적 요청 수와 보존 대화 수는 KV 행 수에 제한되지 않는다.
잘못된 입력은 400, 대기열 초과와 종료된 엔진은 503으로 응답한다. 문 안에서 예상 못 한 예외는 연결을 끊지 않고 500(JSON
`{"error": …}`)으로, 스트림이 이미 시작됐으면 마지막 이벤트로 알리고 트레이스백은 로그에 남긴다(`st:http_internal_errors_total`).
종료 신호는 모든 랭크로
전달하며, 실행·대기 중 요청의 자원을 정리하고 기다리는 HTTP 호출을 깨운다.

**갈라진 랭크는 기다리지 않고 말하며 죽는다**(`base/tripwire`, `base/stall`, 45차 2026-09-13). 호스트 투표와 교환은 전부 고정 길이(80 int64)
벡터에 랭크마다 (순번, 지점, 개수, 방식) 꼬리표를 싣고 간다: 지점이 다른 두 랭크도 같은 all-reduce 를 완주하고, 네 랭크가 같은 표를
읽어 **같은** `CollectiveDivergence` 를 던진다("rank2: #5518 'admit:fits' 대 rank0: #5518 'settle:done'"). 스텝 브로드캐스트도 rank 0 이
같은 꼬리표를 찍고 나머지가 대조한다. 지점: `settle:done`·`settle:outcome`·`admit:restore`·`admit:fits`·`admit:prefix`·`admit:reorder`·
`gather:rows`·`gather:detail`·`boot:seed`. 부팅 단계의 투표(`RuntimeMemory.checkpoint`)는 읽기가 실패해도 그 실패를 표로 던지고, 캡처·
자격 단계에서 죽는 랭크는 나머지가 기다리는 다음 표에 `failed` 를 던지며, `weights-loaded` 랑데부 전에 죽는 랭크는 `failed:` 단계로
랑데부에 합류해 1800 s 대신 지금 모두를 세운다. 원샷 레인 선택은 포인터 정렬이 아니라 dtype·형상으로 정한다(`Comm._settled`). 살아서
멎은 스텝은 옆의 스레드가 60 s 에 기록하고 300 s 에 링을 쓴 뒤 SIGKILL 한다(`ST_STEP_STALL_NOTE_S`/`ST_STEP_STALL_TRAP_S`). 어떤 죽음이든
덤프 디렉터리에 `death-rank{r}-*.json`(divergence / peer-left / local)과 `stall-rank{r}-*.json` 이 남고, 브래킷은 네 랭크의 `docker logs`
를 정지 전에 남긴다(`st-bracket-dumps/<session>-<arm>/rank{r}-<ip>.log`).

`NvmeTier`는 완성된 새 파일의 이름을 manifest에 원자적으로 게시한다. 이전 세대는 그때까지
보존하며, 삭제 실패는 manifest의 `retired` 또는 `deleting` 기록으로 남긴다. 재시작 후 또는
디코드 경로 밖에서 `tier.cleanup()`을 호출하면 미완료 삭제와 게시되지 않은 세대 파일을 정리한다.
`deleting` 상태는 재승격할 수 없으며, 같은 저장 디렉터리는 하나의 `NvmeTier`가 소유한다.

`Glm53Caches`는 하나의 아레나에 다음 영역을 선언한다:

- 물리 KV 블록마다 모든 DSA 층의 fp8 latent와 압축 키·스케일을 함께 저장한다.
  블록 전체가 NVMe 전송 단위이며, 레인에는 층별 절대 슬롯 주소를 전달한다.
- 시퀀스 상태 슬롯마다 KDA conv·recurrent 링과 인덱서 꼬리 링을 둔다.
  인덱서 링은 `kpool - 1 + spec_k`개 위치를 보존해 드래프트 거절 후에도 앞선 풀을 복원한다.
  긴 프리필은 최신 창만 한 번씩 기록하여 CUDA의 중복 scatter 순서에 의존하지 않는다.
- 블록표도 아레나에 상주하며 `prepare(step)`이 매번 예약 문맥과 상태 슬롯 소유권을 확인한다.
  블록 매핑이 같으면 전송을 생략하고, 늘어나면 새 구간만 게시한다. `BlockPool.release`가
  행의 세대 번호를 바꾸므로, 행 재사용이나 NVMe 복원은 블록 개수가 같아도 다시 게시한다.
  이전보다 짧은 행의 남은 항목은 같은 전송에서 `-1`로 지우며, 캐시 전체 reset도 게시 상태를 초기화한다.

`BlockPool.table`, `row(seq)`, `epochs`는 읽기 전용 뷰다. 매핑 변경은 `reserve*`와 `release`를
통해야 하며, 이를 통해 같은 세대의 블록표가 뒤에만 늘어난다는 조건을 지킨다. 2차 최적화의
회귀검사와 구성요소 측정은
[`measurements/st_engine_cache_20260911`](../measurements/st_engine_cache_20260911/README.md)에 있다.

예산은 `profiles/glm53/budget.py`가 선언한다(D1): OS 예비 = 이 박스의 earlyoom SIGTERM 선 + 1 GiB(`budget.os_reserve_gib`, 여기선 7.00 GiB), 런타임 바닥(원장, 그리고 그중 우리 기동분이 아닌 몫은 "already on this box before us" 로 따로 선다), 가중치·드래프터(랭크 파일), 상태 슬롯·prefix 스냅샷(레이아웃),
아레나 밖 작업공간 상한 12 GiB(`base/runtime_memory`가 강제; 부팅 원장 `memory-rankN.json`을 주면 실측 피크를 줄에 적는다),
NVMe 스테이징 — 남는 것이 KV 자리이고, 전체 모델 부팅은 rank 0 에서 그 표와 "선언한 KV 가 남기는 양"을 찍는다
(`python3 engine/profiles/glm53/budget.py [--ledger …]`). 프리필 인덱서 선택은 질의 1,024 행씩 나눠(`net.SELECT_ROWS`) 로짓 과도를
행×후보×4 B 로 묶고, 디코드 그래프의 용량 사다리는 `facts.max_position`에서 끝난다.

프리필 메모리 검증은 KV 용량과 실제 서빙 문맥 상한(`engine.max_context`) 중 작은 범위의 양 끝을 실행한다.
프리필·커널 워밍업과 최종 준비 완료 시점에는 가드 검사 전에 비활성 CUDA 예약 블록을 반환한다.
다른 부팅 단계도 즉시 여유가 OS 예비 아래면 반환 후 다시 검사한다. 생존 텐서·그래프 소유 메모리와 누적 피크는 유지한다.
각 `memory-rankN.json` 행에는 반환 전 예약량·즉시 여유, 실제 반환량(`allocator_reclaimed_bytes`),
반환 후 host/device free와 실패 이유가 남는다. 호스트 종료 기준(6/4.5 GiB)과 작업공간 상한(12 GiB)은 별도다.

로더의 읽기는 **O_DIRECT** 다(티어가 자기 읽기에 대해 대는 것과 같은 이유: 페이지 캐시는 이 상자에서 아레나와 같은 풀이다).
44.5 GiB 랭크 파일을 버퍼드로 읽으면 엔진이 드라이버에 아레나를 달라고 하는 바로 그 순간 44.5 GiB 의 클린 페이지가 남는다 —
첫 45층 부팅의 55.4 GiB 할당 실패에 연루된 압력이다. 런 하나는 자기를 담는 섹터 창으로 읽고 런은 그 안의 슬라이스이며,
스테이징 버퍼는 페이지 정렬(CUDA 대상이면 핀)이다. O_DIRECT 를 거부하는 파일시스템(overlayfs·tmpfs)에서는 버퍼드로 내려가고
그 경로만 `fadvise` 를 쓴다. 사전샤딩이 체크포인트를 읽는 쪽(`base/checkpoint`)도 같은 범위 읽기를 쓴다 — 텐서마다
`safe_open` 하던 경로가 아니다. 실측(srv4, 랭크 파일 4.14 GiB, cpu 대상, 3회): **O_DIRECT 1.92~2.09 vs 버퍼드 0.70~0.88 GiB/s**;
체크포인트 층 하나(4.08 GiB, 2,619 텐서)의 읽기는 **2.71 → 1.56 s**(mmap 은 게을러서 페이지를 만져야 공정하게 잰다).

아레나는 가상 범위 하나를 20 MiB 물리 조각으로 매핑한다(`expandable_segments`, 이 상자가 vLLM 을 띄우는
방식). 첫 45층 부팅은 55.4 GiB `cudaMalloc` 한 번에서 로드 전에 OOM 이었다. 입장 관문 `prepare_allocation`은 랭크·
드래프터 파일의 캐시 페이지를 버리고, 그래도 `MemFree` 가 아레나 + 16 GiB 에 못 미치면 `MemAvailable` 이 허락하는 만큼
익명 페이지를 잠깐 잡았다 놓아(`touch_pages`) 캐시를 회수한 뒤 다시 잰다. 회수한 양과 아레나 매핑 방식은 부팅 표에
`boot_reclaimed_GiB`·`arena_expandable` 로 찍힌다. 할당 모양의 실측은 `probes/engine_alloc_shape_check.py`.

그 익명 회수는 컨테이너 안에서 할 수 있는 마지막 수단일 뿐이다. 2026-09-13 19:29~19:48 의 프로덕션 부팅은 전부 입장에서
멈췄다. srv2 는 `overcommit_memory=2` 에 CommitLimit 75.8 GiB 라 아레나 + 헤드룸 76.47 GiB 의 회수 매핑 자체를 거절했고,
srv4 는 캐시 12 GiB 를 비우려는 폴트가 SIGTERM 선을 넘는다고 거절했다. 캐시는 엔진 것이 아니었다(로더와 티어는 O_DIRECT).
그래서 런처가 노드마다 **컨테이너를 띄우기 직전에 호스트에서** 깨끗한 파일 캐시를 버린다
(`launchers/st-return-file-cache.sh`, rsync·이미지 빌드 뒤, 리스를 쥐고 노드가 비어 있을 때; `ST_RECLAIM_FILE_CACHE=0` 이면 끈다).
할당도 커밋 차지도 없다. `sudo -n` 이 안 되는 노드는 그렇다고 말하고 그대로 띄운다 — 판정은 여전히 입장이 한다.
입장이 시작될 때의 파일 캐시와 `MemAvailable` 은 `boot_file_cache_GiB`·`boot_available_GiB` 로 찍혀, 호스트의 반납이 랭크에
닿았는지 다음 부팅이 스스로 말한다.

컨테이너가 뜬 뒤 부팅이 읽는 것이 캐시를 다시 채우므로, 런처는 랭크마다 호스트에 **회수 브로커**도 띄운다
(`launchers/st-reclaim-broker.sh`, 디렉터리 `glm53-logs/st-reclaim/rank<r>` 를 컨테이너에 `ST_RECLAIM_DIR` 로 넘긴다). 입장과 준비
단계의 회수는 즉시 가용이 모자라면 익명 폴트 전에 브로커에 묻고(`arena.host_reclaim`), 브로커가 호스트에서 캐시를 버린 뒤 답한다.
할당이 없으니 커밋 한도도 쓰지 않는다 — srv2 는 `overcommit_memory=2`·비율 50(CommitLimit 75.8 GiB)이라 엔진의 익명 회수가
들어갈 자리가 없었다. 브로커가 없거나(하트비트가 5 s 넘게 멈춤) 버려도 모자라면 예전의 폴트가 그대로 돈다. 브로커는 컨테이너가
사라지면, 컨테이너가 끝내 안 뜨면, 또는 두 시간 뒤 스스로 끝나고 `stop` 이 함께 끝낸다. 결과는 랭크 로그 한 줄과
`boot_host_reclaim`(0 안 물음, 1 못 버림, 2 버림).

사전샤딩은 `RankWriter`가 모든 실제 가중치의 데이터 오프셋을 256바이트 경계에 맞춘다.
작은 스케일 뒤의 행렬도 TMA 정렬을 유지하도록 safetensors의 명시적 U8 패딩 텐서를 사용한다.
표준 safetensors 리더로 읽을 수 있고, 로더는 패딩을 포함한 연속 범위를 한 번 업로드한 뒤 뷰를 만든다.
**기존 PR #532 랭크 파일을 서빙 레인에서 사용할 때는 수정된 preshard로 다시 생성해야 한다.**
로딩 시 잘린 파일과 진행하지 않는 쓰기는 즉시 오류가 되며 무한 재시도하지 않는다.

서빙 KDA 어댑터는 엔진의 `[H,K,V]` 상태와 커널의 `[H,V,K]` 상태를 경계에서 변환한다.
recurrent 레인도 연결되어 검증 토큰마다 상태를 반환하고, 시작 상태를 보존한다.
MLA 참조 레인은 선택된 fp8 행만 변환하며, 패딩 슬롯이 가리키는 미사용 블록의 NaN을 마스킹한다.

인덱서의 Hadamard-128 변환·FP8 양자화는 `indexer_quant` 레인으로 실행한다.
키 풀링은 `kpool_compress` 레인이 `compress_pool_keys`를 직접 호출한다. GB10에서 128차원
풀 하나를 1워프로 처리하며, XOR 짝을 이용한 Hadamard 회전은 공유 메모리를 사용하지 않는다.
입력 스트라이드를 직접 받아 복사 없이 읽고, 결과 FP8 키와 스케일만 할당한다. 반환 전용
경로에서 쓰지 않던 임시 캐시·위치 배열·쓰기 마스크를 만들지 않는다. BF16 반올림 경계와
FP8 양자화 수식은 유지한다. 정확성 및 성능 근거는
[`measurements/st_engine_warp_pooling_20260911`](../measurements/st_engine_warp_pooling_20260911/README.md)에 있다.

선택한 풀의 최종 주소 생성은 `pool_slots` 레인 하나가 맡는다. GB10에서 토큰 2,051개를
먼저 펼쳐 정렬하던 경로를, 풀 ID 512개를 정렬한 뒤 토큰을 생성하는 구조로 바꿨다.
서빙 커널은 4워프 프로그램 하나가 한 행을 처리하며, 중간 GPU 텐서를 할당하지 않는다.

KV 블록은 풀 크기의 정수 배수여야 한다. 이 계약 덕분에 풀마다 블록 주소를 한 번 읽고
네 토큰의 주소를 레지스터에서 계산한다. 미완성 꼬리는 최신 토큰부터 앞에 배치하고,
중복 풀은 토큰별 중복 횟수를 유지하며, 패딩과 유효 개수까지 같은 커널에서 쓴다.
참조 레인은 기존 토큰 확장·정렬·매핑 수식을 유지해 정확한 정수 비교의 기준으로 쓴다.

캐시는 `token_map(layer, seq)`로 블록 행과 잠재 벡터 행 단위의 블록 크기·간격·레이어
오프셋을 제공한다. 연속 캐시 검사의 `None` 블록 행은 위치와 슬롯이 같은 매핑이다.
LocalTP 소유권과 오류 전파 규칙은 다른 레인과 같다. 이전 `expand_pools`, `indexer_slots`
서빙 레인은 제거했으며, 기존 함수는 구성요소 검사와 명시적인 과거 커밋 비교에만 사용한다.
검증 범위와 변경 전후 측정은
[`measurements/st_engine_pool_slots_20260911`](../measurements/st_engine_pool_slots_20260911/README.md)에 있다.

네 노드 검증은 각 노드에서 같은 인자로 `check.py --distributed`를 실행한다.
기본 노드 순서는 **rank 0=srv2, rank 1=srv1, rank 2=srv3, rank 3=srv4**다.
`MASTER_ADDR`은 rank 0 서버를 가리켜야 한다. `RANK`, `WORLD_SIZE=4`, 격리된 `MASTER_PORT`,
RoCE 인터페이스·GID는 실행기가 설정하고, `--ranks`에는 해당 랭크의 정렬된 파일을 둔다.
`probes/engine_comm_check.py`로 가중치 없이 통신을 먼저 검증할 수 있다.

추가 장치 검사:

    bash probes/run_engine_probe.sh probes/engine_kda_check.py
    python3 probes/engine_cuda_io_check.py

검사 결과와 실제 사용한 네 노드 실행기는
[`measurements/st_engine_runtime_20260911`](../measurements/st_engine_runtime_20260911/README.md)에 보관한다.
이는 2개 실제 층(KDA+dense, DSA+NVFP4 MoE)의 실행·캐시 계약 검사다. 통합 전 로그는
참조 전문가를 사용했고, PR #534 통합 후 `check.py --lanes served`는 b12x 전문가를 포함한다.
전체 45층 onepass 품질, 실제 DFlash2 수용률, 그래프 기반 요청 실행 및 처리량·ITL 판정은 별도다.


디코드 그래프 연결과 실가중치 수치 안정성 후속 작업은
[`measurements/st_engine_completion_20260911`](../measurements/st_engine_completion_20260911/README.md)에 기록했다.
TP4의 두 층 검사에서 일반 실행·그래프 출력과 캐시가 일치하며, MoE BF16 누적의 반복 편차를
FP32 합산으로 줄였다. 전체 모델 첫 부팅은 CUDA 아레나 할당 실패로 중단됐다. 따라서 전체
45층 품질·DFlash2 수용률·NVMe 간섭 ITL과 최종 플릿 릴리스는 아직 통과한 상태가 아니다.
독립 이미지의 버전·소스 식별 방법은 [`runtime/README.md`](runtime/README.md)를 따른다.

SM121a·TP4 후속 변경은 [`st_engine_four_optimizations_20260911`](../measurements/st_engine_four_optimizations_20260911/README.md)에 있다.
타깃 그래프는 물리 슬롯 ID를 Triton 상태 커널에 전달한다. KDA는 이전 recurrent 상태 하나와
conv 이력만 읽고 새 토큰 위치만 쓴다. 작은 인덱서 꼬리의 읽기 버퍼는 유지하며, 상태 링 전체의
gather/commit은 하지 않는다. 전체 모델 부팅은 별도의 작업공간 상한과 OS 여유를 선언하고,
최대 프리필·모든 그래프 형상의 준비 중 관측한 메모리 최대치를 저장한다.

네이티브 DFlash의 캐시와 프리픽스 스냅샷은 각 랭크가 계산하는 KV 2헤드만 보관한다. BF16 가중치의 원래 아레나 예약은 팩 저장 공간으로 재사용하며 그대로 남는다.
UMA 입장 검사는 공간이 부족하면 지정된 모델 보관 경로의 `.safetensors`와 다운로드 임시 파일에서 깨끗한 파일 캐시를 반환한다. 파일 내용·진행 중인 쓰기·다운로드 프로세스는 바꾸지 않는다. 최종 입장 조건은 아레나 + 작업 공간 상한 12GiB + OS 여유(SIGTERM 선 + 1GiB) + 접두사 티어 호스트 캐시이며, 준비 중에도 실제 여유와 할당 최고치를 검사한다. 작업 공간 상한은 2026-09-14 실측 최고치 9.67GiB(서빙 문맥 끝에서 돈 가장 큰 프리필 청크)에 여유를 더한 값이고, 더 쓰는 형상은 `--workspace-gib`(런처 `ST_WORKSPACE_GIB`)로 올린다.

## 새 체크포인트 붙이기

```bash
python3 tools/onboard.py --ckpt ~/models/<checkpoint> --placement ep     # 사람이 읽는 표
python3 tools/onboard.py --config config.json --json                     # 기계가 읽는 문서
python3 tools/onboard.py --config config.json --placement ep \
    --state attention.kind=mla --state indexer.compress=ced              # 레퍼런스가 말한 사실을 채워서
```

세 가지 답 중 하나가 나온다.

| 답 | 뜻 | 다음 |
|---|---|---|
| **profile \<name\>** | 그 `model_type` 을 선언한 프로필이 있다 | 프로필의 유도로 형상·레인 표가 바로 나온다 |
| **generic + 형상** | 프로필은 없지만 설정이 필요한 것을 다 말했다 | 레인 표와 `cells.plan()` 의 작업 목록이 그대로 할 일이다 |
| **generic + 빈칸** | 설정이 정하지 못한 것이 있다 | 빈칸마다 **무엇이 그것을 정하는지**가 같이 나온다 — 체크포인트의 레퍼런스 구현, `engine/profiles/` 의 프로필, `--placement`, 또는 `--state <필드>=<값>` |

**라우팅 전문가가 없는 체크포인트**(평범한 dense LLM)도 형상이 선다: 모든 토큰이 지나는 MLP 하나를 이 엔진이 실제로
서빙하는 **E=1 셀**로 읽는다 — b12x 의 게이트가 라우팅 튜플 옆에 `(1, hidden, dense_inter_local, 1)` 을 승인한다
(`kernels/b12x/moe_dispatch._glm_tp_scatter_shape`). 그 MLP 는 여기 모든 dense·공유 MLP 처럼 TP 로 쪼개지므로 **고를
배치가 없고 `--placement` 를 묻지 않는다.** 반대로 설정이 **전문가를 부르는데** 이 문이 못 읽는 철자라면(예: Mixtral 의
`num_local_experts`) 그것은 **빈칸**이다 — dense 로 읽는 순간 모든 토큰을 MLP 하나로 보내는 모델이 된다.

셋 다 **읽은 것을 먼저 표로 낸다**(필드 · 값 · 그 값이 온 키). 설정이 **이름을 대는** 축은 추측이 아니라 읽기다 —
`index_kpool_compress` 는 인덱서의 압축을, `kda_layers` 는 선형 어텐션의 채널별 감쇠를, `mhc: true` 는 잔차 혼합을
그 키가 말한다. 설정이 끝내 말하지 않는 축(`STATEABLE`: 어텐션 종류 · sink · 인덱서 압축 · 감쇠 · 혼합기 · 전문가
양자화 · 게이트)은 운영자가 `--state` 로 **댈 수 있다**. 이것은 노브가 아니다(D11: 입력은 사실뿐) — **빈칸만
채우고**, 설정이 이미 정한 필드를 대면 덮어쓰지 않고 **거부한다**. 세 프로필의 유도와 이 앞문이 필드 단위로
**같다**는 것이 `tests/test_engine_onboard.py` 의 판정이고, 각 모델이 레퍼런스에서 가져오는 사실은 셋 · 다섯 ·
여섯 개가 전부다.

빈칸은 실패가 아니라 **작업 지시**다. 예: 인덱서를 선언한 설정은 키 압축 방식(kpool·ced·qsa)을 말하지 않고,
KV 헤드가 하나인 설정은 MLA 인지 GQA 인지 말하지 않는다 — 둘 다 추측하면 **부팅되면서 틀린다.**
