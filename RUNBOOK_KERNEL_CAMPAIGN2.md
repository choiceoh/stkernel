# Kernel Campaign II 런북 — GLM-5.3, srv4 TP=4 (2026-08-31)

`STEP_KERNEL_MAP.md`(#108)가 문서화한 열린 축 4개를 측정하는 순서다. 하나의
노브/재기동, **base→cand→base 브래킷** 원칙, 그리고 기존 판정 원장의 게이트를
그대로 따른다. apply 명령은 사람이 실행한다(자동화는 읽기 전용).

공통 게이트(모든 암 공통):

| 게이트 | 기준 |
|---|---|
| 품질 | `bench/check-quality.py` 9/9 |
| 한국어 | `bench/korean-corruption.py` 0/16 (U+FFFD·혼자모음 0) |
| 판정 채널 | **C=1 step/s** (`BENCH_MODEL=glm-5.3-flash bench/bench-dec.py`), 부팅 CV 1.7% |
| 수용률 보호 | pos-1 acceptance ±2pct (수용률을 움직이는 변경은 tok/s + tokens/step 함께 기록) |
| 판정값 | engine-confirmed 값 (요청한 값 아님 — #116 교훈) |
| 원장 | 결과와 기각 포함 전부 `MEASUREMENTS.md`에 기록 |

배포: `launchers/deploy-overlays.sh glm53` — `tests/test_logic.py` 게이트 →
4노드 배포 → SHA-256·preimage 검증. 재기동은 `--enable-expert-parallel` 등
플래그는 전부 런처가 만드니 노브만 바꿔 재기동한다.

---

## EXP-1 — b12x EP 첫 실측 (`ENABLE_EP=1`)

보드에서 가장 큰 숫자: 전문가 읽기 41ms = 66ms 스텝의 62%(대역폭 91%). EP=4로
전문가가 샤딩되면 그 읽기가 줄어들 여지가 있다. #115(EP 위장)·#120~122(프리필
용량 고침)로 구현은 끝났고 **부팅 측정만 남았다.** 부팅 전 데이터: pairs=65536
프리필에서 최다 전문가 = 더미 48,377행 = static map의 74% — 여유 있음.

```bash
# base (통제): 현재 프로필 그대로 (ENABLE_EP=0)
ENABLE_EP=1 bash launchers/start-glm53-nvfp4-tp4.sh     # cand
```

- 런처가 GRAPH_CAP=32(≤80)를 확인하고 `--enable-expert-parallel`을 건다.
- 부팅 로그에서 EP 리맵·용량 프로브 경고 확인. 용량 프로브가 "no workspace fits"
  로 올리면 그 조언(`FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0`)을 따라
  시도하기 전에 **먼저 기록**하고, 노브를 하나씩만 추가한다.
- 중단 기준: 부팅 실패, 출력 파손(인사말 공백·한국어 U+FFFD), 품질 게이트 미달.
  롤백 = env 하나 (`ENABLE_EP=0`, 현행 프로필이 곧 base).
- 판정: C=1 step/s 브래킷. 프리필은 `bench-tp4.py`(75K)도 함께.

기대값(정직): 전문가 읽기가 4분의 1로 줄어드는 게 아니라 **더미 전문가 1개 분량의
패딩과 리맵 오버헤드를 지불하면서** 샤딩 이득을 받는다. 디코드 그래프는 고정
형상이라 리맵이 그래프 안에 있다. 승부는 실측이 결정한다.

## EXP-2 — custom_ops 융합 A/B (`CUSTOM_OPS_AXIS=none`)

elementwise 글루 483개/스텝(25.6%)의 원인 검증. `custom_ops`가 인듀서의 융합
장벽이라 residual add·scale·mask가 독립 커널로 남는다. 첫 A/B 부팅은 #116에서
노브가 전달되지 않은 채 **무효**로 판정됐고, 잠금은 고쳐졌지만 유효 A/B는 아직
없었다. **코드 변경 0.**

```bash
CUSTOM_OPS_AXIS=none bash launchers/start-glm53-nvfp4-tp4.sh   # cand
```

- 주의: 빈 값(`CUSTOM_OPS_AXIS=`)은 caller-env 보존 목록에서 걸러진다(비어있지
  않은 값만 보존). 융합 암은 `none`으로 보낸다.
- 부팅 로그에서 engine-confirmed `custom_ops`가 `["none"]`인지 **반드시** 확인
  (#116 실패 모드).
- 상한: 483 × 5.4µs = 2.6ms = 3.9%. 단 5.4µs는 상치(上치)이고 약화 중 — 실제는
  그보다 작을 공산이 크다.
- 실패해도 축이 닫히는 게 이득이다: "장벽은 필요했다"를 원장에 남긴다.

## EXP-3 — MHC small-M 스윕 → `VLLM_GLM53_MHC_SMALLM`

MHC TileLang 185 커널/스텝(9.8%)의 디코드 런치 휴리스틱. 이미지 원본이 직접
남긴 `TODO(gnovack): investigate autotuning` 자리고, dsv4 선례(R1)에서 같은
자리 (6,4)가 small-M +16.3%를 냈다. 이번에 `glm53_mhc_tilelang` 모듈로
오버레이 가능해졌다(프로필에 이미 포함 — 노브 unset은 불활성).

1단계 — 프로브 (엔진 죽이기 전에, 새 컨테이너에서; 서빙 컨테이너 CUDA 금지):

```bash
# srv4의 ~/stkernel 체크아웃을 배포 커밋으로 pull한 뒤 (srv4 뒤처진 클론 사례 있음)
ssh srv4 'cd /home/choiceoh/stkernel && \
  probes/run_mhc_glm53_bench.sh --quick'

# 같은 창에서 함께 돌릴 프로브 두 개:
#  --onepass : 융합 단일 런치(#133) vs 스톡 쌍 수치(rel<=1e-4)+시간 비교
ssh srv4 'cd /home/choiceoh/stkernel && \
  probes/run_mhc_glm53_bench.sh --onepass'
#  --prefill : 프리필 big_fuse h_blk 1024/2048/4096 스윕(#136, dsv4 R3 선례)
ssh srv4 'cd /home/choiceoh/stkernel && \
  probes/run_mhc_glm53_bench.sh --prefill'
```

- wrapper는 `compose-overlays.sh glm53`를 먼저 실행하고 `tilelang.py`와
  `tilelang_kernels.py`를 이미지의 실제 target에 bind한다. 프로브도 import된 두
  파일의 SHA-256이 합성 산출물과 같은지 확인하므로 stock 이미지 경로면 즉시 중단한다.
- `--quick`이 JIT 컴파일 10개(≈수 분), 전체 스윕은 20~28개(십수 분).
- 표의 `!`는 재현 오차 >1e-4 — 그 셀은 시간이 빨라도 채택 후보에서 뺀다.
- `--onepass`는 채택 전제: rel_err <=1e-4 유지 + 브래킷 종단 확인.
- 승자 없음 → 축 기록하고 종료. 승자 → 2단계.

2단계 — 브래킷:

```bash
EXTRA_ENV="VLLM_GLM53_MHC_SMALLM=<tile_n>,<n_splits>" \
  bash launchers/start-glm53-nvfp4-tp4.sh   # cand
```

- 노브 unset(=base)은 스톡과 바이트 단까지 동일하므로, base 부팅은 직전 브래킷
  base를 재활용해도 된다(오버레이 변경이 없었다면).
- 파서가 계약 위반값을 스톡으로 되돌리는지는 `tests/test_logic.py`
  `test_mhc_smallm_knob`가 검증한다.

## EXP-4 — b_proj/indexer fp8 (`VLLM_GLM53_FP8_DENSE_BPROJ=1`)

W8A8 이후 남은 bf16 GEMM 145개/스텝(7.7%) 중 가드가 통과시키는 세 사영:
`q_b_proj [4096,1536]`·`kv_b_proj [4096,512]`·`indexer.wq_b [4096,1536]`(복제).
바이트 −160MB/스텝 ≈ **천장 0.9%** — CV 1.7% 아래라 단일 부팅으론 안 보이고,
브래킷 3회로 겨우 판정 가능한 크기다. `f_b_proj/g_b_proj`(min dim 128)와
`wk_weights_proj`(로더가 융합을 위해 bf16 강제)는 의도적으로 제외했다.

```bash
EXTRA_ENV="VLLM_GLM53_FP8_DENSE_BPROJ=1" \
  bash launchers/start-glm53-nvfp4-tp4.sh   # cand
```

- 전제: `VLLM_GLM53_FP8_DENSE=1`(프로필 기본) — 없으면 이 노브는 무효.
- 부팅 지문에서 linears 수 증가 확인(기존 대비 +33 근처; 인덱서 wq_b가 체크포인트
  에서 이미 fp8이면 그만큼 적다 — skipped 목록이 알려준다).
- #110 교훈 반영: 이 축은 전적이 0이므로 **기본화 금지.** 채택돼도 다음 캠페인
  창에서 별도 판정 후 프로필에 올린다.
- 인덱서 GEMM은 aux 스트림 경합으로 늘어난 시간이라(08-10 분해) 대역폭 이득이
  그대로 안 나올 수 있다 — 그러면 기각 사유가 명확한 셈이다.

---

## EXP-5 — 프리필: 캡처 → KDA 청크 스윕 (2026-09-01 추가)

프리필은 FLOP 바운드(추정 MFU 30~45%, 전문가 FLOP 6할)라 디코드의 런치-절감
프레임이 안 통한다. 오버레이가 접수 가능한 프리필 커널 축은 (a) mhc BIGFUSE
(#136, 준비 완료) (b) **KDA 청크 커널 launch-config** (이 항목) 둘.

GLM의 실제 프리필 엔트리는 Kimi-K3 벤더드 포크가 아니라 generic FLA
`vllm/third_party/flash_linear_attention/ops/kda.py`의
`chunk_kda_with_fused_gate`다. 이 경로는 kda.py의 KKT inter/intra,
recompute, GLA output, gate+cumsum과 외부 `ops/chunk_delta_h.py`, 합계 6개
Autotuner를 실행한다. plain/spec decode는 별도 `fused_recurrent_kda`라 이
청크 스윕 대상이 아니다.

6개 커널 모두 T가 `do_not_specialize`이고 기존 cache key에 raw T가 없다.
`glm53_kda_prefill_regime`은 exact GLM/TP4/SM121 계약에서만 한 호출의 packed
평균 길이가 1024 이상인지 한 번 계산해 동일한 0/1 runtime key를 6개에
전달한다. raw T별 cache 폭증이나 코드 specialization은 만들지 않는다.
1024 경계는 아직 미측정 가설이므로 프로필 기본값은 0이다.

1단계 — 캡처 (서빙 중, 엔진이 죽을 수 있으므로 창의 마지막에):

```bash
# 헤드에서: 긴 프롬프트(75K 벤치) 진행 중에
curl -X POST localhost:8000/start_profile; sleep 20; \
curl -X POST localhost:8000/stop_profile
# 트레이스는 헤드 ~/vllm-prof/ 에 착지. srv4로 옮겨 인구조사:
scp <head>:~/vllm-prof/*rank0*.pt.trace.json.gz srv4:/tmp/prefill_trace.gz
ssh srv4 'docker run --rm --entrypoint python3 \
  -v /home/choiceoh/stkernel-c2:/repo:ro \
  -v /tmp/prefill_trace.gz:/tmp/trace.gz:ro glm53:v13-b12x \
  /repo/census.py /tmp/trace.gz'
```

- census에 "KDA/FLA 청크" 그룹 추가됨 — chunk/recurrent/conv가 기타와 norm에서
  분리 집계된다. 합계를 청크 수(T/8192)로 나눠 청크당으로 환산할 것.
- 주의: glm53 이미지의 docker ENTRYPOINT는 `vllm` CLI — 반드시
  `--entrypoint python3` 를 명시한다.

2단계 — KDA 청크 스윕 (새 컨테이너, GPU):

```bash
ssh srv4 'docker run --rm --gpus all --entrypoint python3 \
  -v /home/choiceoh/stkernel-c2:/repo:ro glm53:v13-b12x \
  /repo/probes/kda_prefill_bench.py --T 8192'
```

- 좌표하강으로 core-six config를 하나씩 갈아끼우고 실측 엔트리
  `chunk_kda_with_fused_gate(output_final_state=True)` 시간을 재며, 스톡의
  출력과 final recurrent state 모두에 rel err 게이트(>1e-2 플래그)를 건다.
- env=1 첫 long-prefill은 새 bucket을 채우며 KKT inter 24개, GLA output
  36개 등을 다시 autotune한다. 타이밍/수락률 브래킷 전에 같은 형상을 먼저
  prewarm하고, env=0은 레짐만 끌 뿐 byte-identical rollback이 아님을 기록한다.
- 승자 → kda config 테이블 인수(env 게이트) → 프리필 tok/s 브래킷.
- 승자 없으면 축 기록으로 종료 (FLA 공동 튜닝 선례상 가능성 있음).

3단계 — engine-down bracket arm (헤드 checkout):

```bash
VLLM_GLM53_KDA_PREFILL_REGIME=1 \
  launchers/start-glm53-nvfp4-tp4.sh
```

이 키는 profile-owned라 `EXTRA_ENV`로 넘기지 않는다. launcher 명령의 caller
env로 주어야 profile 기본 0을 덮은 값이 모든 rank에 전달된다. 별도 fresh-cache
arm에서 `short -> long`과 `long -> short` 순서를 모두 맞춰 prewarm한다. 첫 long
요청은 autotune 전용이고 두 번째부터 timed/수락률 bracket을 시작한다. 위
standalone probe는 base-image config inventory를 스윕할 뿐 full-engine init latch와
0/1 dispatch 자체를 증명하지 않는다.

---

## EXP-6 — 메가커널 세그먼트 (`glm53_megakernel`, 2026-09-01 추가)

콜렉티브 사이 구간을 **persistent 48블록 런치 하나**로 흡수하는 신규 모듈.
sm_121a 계약(mma.sync e4m3 · 클러스터/WGMMA 금지 · 48블록 고정)과 그래프
안전 바리어(단조 티켓, osar done_ctr 방식)를 따른다. 세그먼트:

| 세그먼트 | 흡수 | 스톡→MK (런치/스텝) |
|---|---|---|
| MK-MHC | hc post+pre (수학은 소유 TileLang 소스의 비트 충실 포팅) | 179 → 45 |
| MK-GEMM | per-token quant + W8A8 GEMM, M≤32 | ~360 → ~180 |
| MK-KDA | KDA 블록 전체(in_proj→conv→recurrent→norm→o_proj) | ~510 → 34 |

천장: 5.4µs/커널(약화 중인 상한) × 약 900개 ≈ **4.9 ms = 스텝의 7.4%**. 실측은
그 이하일 공산이고, **측정 전까지 어떤 수치도 원장에 들어가지 않는다.**

사다리(순서대로, 건너뛰기 없음):

```bash
# 1. 순수 논리 (이 리포, GPU 불필요)
python3 tests/test_logic.py          # test_glm53_megakernel_contracts 포함

# 2. srv4 새 컨테이너(서빙 컨테이너 CUDA 금지): 수치 + 타이밍 프로브
#    rel 게이트: MHC 1e-3 · GEMM 2e-2 · KDA out+상태 2e-2. '!' 셀 = 실격
#    (래퍼가 합성 오버레이를 실제 이미지 경로에 바인드한다)
bash probes/run_megakernel_bench.sh

# 3. KDA 섀도 부팅 (상태-인덱스 계약이 열린 항목 — 섀도 통과 전에 암 금지)
VLLM_GLM53_MEGAKERNEL=1 VLLM_GLM53_MK_KDA_SHADOW=1 \
  bash launchers/start-glm53-nvfp4-tp4.sh
#    bench-tp4 1회 내내 [megakernel] kda shadow 로그에 DRIFT 0 확인

# 4. 브래킷 — 세그먼트별 개별 암(MHC와 GEMM은 별도 부팅으로 분리)
VLLM_GLM53_MEGAKERNEL=1 VLLM_GLM53_MK_MHC=1 \
  bash launchers/start-glm53-nvfp4-tp4.sh   # cand A
VLLM_GLM53_MEGAKERNEL=1 VLLM_GLM53_MK_GEMM=1 \
  bash launchers/start-glm53-nvfp4-tp4.sh   # cand B (MHC 합침은 그 다음)
```

주의:

- 부팅 로그의 `[megakernel] armed=...` 지문과 `selftest ... -> ARM/DISARM` 이
  판정 근거다 — 노브가 1이어도 셀프테스트 탈락은 자동 해제(스톡 경로)다.
- **MK-GEMM 암은 fp8 가중치 바이트 중복(~+4 GB/랭크)**: KV 라인 확인,
  모자라면 GMU 한 단계 하향(README의 memfree-preflight 계산).
- 첫 암 부팅은 확장 컴파일(~1분, `/root/.mk_build`)만큼 느려진다.
- MK-KDA는 3단계 섀도 로그가 깨끗하기 전에 4단계에 올리지 않는다.

## EXP-7 — 준비 커널 통합 (`glm53_prep_fused`, 2026-09-02 추가)

디코드 스텝의 **호스트 쪽** 입력 준비를 Triton 발사 하나로 접는 모듈. 트레이스
(2026-09-01, rank 3, 229 스텝)에서 스텝의 GPU 유휴가 8.9 ms/72 ms(12%) 이고,
그중 5.7 ms 가 드래프터 그래프와 타깃 그래프 사이의 eager 준비 구간이다 —
`prepare_inputs` + `prepare_attn` + KV 그룹 7개의 메타데이터 빌더가 스텝마다
aten 호출 ~1,000개, memcpy ~45개, 1~3 us 짜리 커널 ~100개를 내고, dflash 는
스케줄러가 동기라 그 시간 내내 GPU 가 빈다. 프로파일러 없는 앵커는 원장의
"디코드 중 GPU 93%" (같은 정의) → 실제 4~5 ms/스텝. 천장: 그 구간 자체,
스텝의 4~6%. 읽는 바이트는 0.

| 스텝의 준비 구간 | stock | fused |
|---|---|---|
| H2D memcpy | ~45 | 1 (idx_mapping, pinned) |
| 커널 발사 | ~100 | 1 (+ deep_gemm 스케줄 메타 1 + 복사 1) |
| aten 호출 | ~1,000 | ~15 |

적용 조건(하나라도 아니면 stock): 전 요청이 spec-verify 이고 드래프트가 꽉 참
(`decode_query_len - 1`), FULL cudagraph 디스패치, 요청 패딩 없음, 프리필 없음,
적응형 검증·DCP·PCP·PP·LoRA 없음. 설치 시 러너/빌더 파일 17개의 preimage 를
고정하고(드리프트 → DISARM), 첫 적격 스텝에서 live 러너로 plan 을 만들며
기하가 다르면 그 부팅은 stock.

사다리(순서대로, 건너뛰기 없음):

```bash
# 1. 순수 논리
python3 tests/test_logic.py                      # test_glm53_prep_fused_contracts

# 2. 프로필 이미지(glm53:v13-b12x, srv2)의 새 컨테이너: stock 빌딩블록 vs fused
#    커널, 무작위 배치 60회 bit-exact. 래퍼가 행마다 preimage 를 검증하므로
#    srv4 의 sm121-fi618 은 거부된다(마운트 파일 5개가 다른 빌드).
bash probes/run_prep_fused_check.sh --trials 60

# 3. 섀도 부팅: fused 뒤에 stock 사슬 전체를 같은 버퍼에 돌려 diff 하고, 깨끗하면
#    fused 배치를 그대로 흘려 armed 분기까지 실행한다. 프로필 선언 키라
#    EXTRA_ENV 가 아니라 caller env 로 넘긴다(런처가 EXTRA_ENV 재정의를 거부).
VLLM_GLM53_PREP_FUSED=shadow bash launchers/start-glm53-nvfp4-tp4.sh
#    bench-tp4 1회 동안 [prep-fused] shadow ... drift=0 확인

# 4. 브래킷: base -> cand -> base, C=1 step/s. armed 모드는 64 fused 스텝마다
#    stock 사슬을 다시 돌려 self-check 하고 drift 면 DISARM 한다
#    (VLLM_GLM53_PREP_FUSED_SELFCHECK_EVERY, 0 이면 끔).
VLLM_GLM53_PREP_FUSED=1 bash launchers/start-glm53-nvfp4-tp4.sh
```

주의:

- 부팅 로그의 `[prep-fused] installed mode=...` 와 `[prep-fused] plan built:` 가
  판정 근거다. `preimage drift -> DISARM` 이나 `plan build failed` 가 보이면 그
  부팅은 stock 이고, 그 브래킷 셀은 무효다.
- 물리 기전 확인: 켠 부팅의 트레이스에서 준비 구간의 `Memcpy HtoD/DtoD` 와
  `at::native::*` 커널이 사라지고 `_glm53_prep_fused_kernel` 하나가 남아야 한다.
- 수치는 stock 과 bit-exact 가 계약이라 품질 게이트는 형식상 통과해야 하지만,
  섀도 drift=0 없이는 arm 하지 않는다.
- 이 모듈은 kpool tail 의 원형 슬롯 매핑을 **건드리지 않는다**(러너가 빌더에
  `positions` 를 안 넘겨 그 매핑은 이 이미지에서 잠들어 있고, fused 는 현행
  generic 매핑을 그대로 재현). 그 활성화는 C>=2 수치 변경이라 별도 브래킷.

## EXP-8 — dflash 에 async scheduling (`glm53_async_dflash`, 2026-09-02 추가)

이미지의 `config/vllm.py` 는 `async_scheduling=None`(런처는 플래그를 안 준다)을
speculative method **이름** 허용 목록으로 판정한다: eagle 계열·ngram GPU·
`draft_model`·`dspark`. `dflash` 는 없어서 모든 dflash 부팅이 "Async scheduling
not supported with dflash-based speculative decoding and will be disabled" 로
동기 스케줄러를 쓴다. 업스트림 main 도 같은 목록(2026-09-02 확인).

기전으로는 이름 문제다: `DSparkSpeculator` 는 `DFlashSpeculator` 의 서브클래스로
`propose()` 를 상속하고(async 가 건드리는 유일한 드래프터 흐름), dspark 는 dsv4 에서
8월부터 `--async-scheduling` 으로 서빙 중이다. V2 러너에는 method 별 async 분기가
없고(요청 상태 미러가 설계상 낙관적 상한), 스케줄러 쪽은 `AsyncScheduler` 의
`[-1]` placeholder 를 워커가 `combine_sampled_and_draft_tokens` 커널로 덮어쓴다.
마운트된 glm53 오버레이 중 스케줄러 쪽 드래프트 id 를 읽는 것은 없다.

**기대값**: 9월 1일 트레이스의 GPU 유휴 8.9 ms/72 ms(프로파일러 없이 ~7%) —
입력 준비 ~5.7, 그래프 제출 1.43, 스텝 전환 — 가 동기 스케줄러 때문에 임계경로에
있다. async 는 N+1 스텝의 스케줄·준비·그래프 제출을 N 스텝의 GPU 실행과 겹치므로
천장은 유휴 몫 전부, **스텝의 7~12%**. 이 캠페인에서 가장 큰 단일 레버다.
`glm53_prep_fused`(EXP-7)와 독립이며 같이 켤 수 있다.

```bash
# 부팅 로그에서 "[async-dflash] ... whitelisted" 가 있고 "Async scheduling not
# supported" 가 없어야 켜진 것이다 (engine-confirmed 원칙).
VLLM_GLM53_ASYNC_DFLASH=1 bash launchers/start-glm53-nvfp4-tp4.sh   # cand (프로필 선언 키: caller env)
```

- 게이트: 품질 9/9, 한국어 0/16, **pos-1 수용률 ±2pct(움직이면 안 된다 — 언제 도는지만
  바뀌고 무엇을 뽑는지는 안 바뀐다)**, C=1 step/s 브래킷 base→cand→base.
- 첫 부팅에서 볼 것: V2 + async 는 `max_concurrent_batches` 가 2 라 KV 캐시의 in-flight
  예약이 두 배 — KV 라인과 memfree preflight 를 먼저 읽고 step/s 를 비교한다.
- 구조화 출력 요청은 드래프트 id 를 스케줄러로 되돌리는 경로(`DraftTokensHandler`)를
  탄다 — 일반 생성은 안 탄다. 벤치에 구조화 출력이 섞이면 따로 본다.
- 롤백 = env 한 줄. `ASYNC_SCHED=0` 은 여전히 동기 강제.
- 원장의 2026-08 한국어 손상 조사에서 async 는 용의자로 기각됐다(SPEC=0 의 async on 과
  dflash 의 async off 모두 손상, 원인은 LibertAIDAI 가중치). 이 실험은 그 판정을
  재론하지 않는다.

---

## 순서와 근거

1. **EXP-1 (EP)** — 기대값 최대. 실패해도 부팅 하나로 원장에 정리된다.
2. **EXP-2 (custom_ops)** — 코드 0, 부팅 1회, 축 자체를 닫거나 열어준다.
3. **EXP-3 (MHC)** — 프로브가 먼저 승자를 가려내 부팅을 아낀다.
4. **EXP-4 (bproj)** — 천장 0.9%, 최우선순위 아님. 다음 창으로 미뤄도 됨.
5. **EXP-5 (프리필)** — 캡처 1회가 관문. KDA 스윕은 그 다음.
6. **EXP-6 (메가커널)** — 프로브와 섀도가 먼저 수치·계약을 닫고 브래킷.
7. **EXP-7 (준비 커널 통합)** — bit-exact 프로브 → 섀도 부팅 → 브래킷. 호스트 유휴 4~5 ms 가 표적이라 GPU 커널 축과 독립.
8. **EXP-8 (dflash async scheduling)** — 부팅 하나, 코드 변경은 허용 목록 한 조건. 천장이 가장 크다(7~12%); EXP-7 보다 먼저 돌려도 된다.

## 금지 (기존 판정 유지 — 재조사하지 않는다)

W4A8 기본화(#110), drafter 측 비용(#104), FP4 CUTLASS 패딩(출력 파손),
업스트림 b12x EP(#3383), ROCm 전용 융합 3종, `fuse_allreduce_rms`(sm_121 키
부재), `enable_qk_norm_rope_fusion`(NoPE), 클럭 상향(운영자 거부권).
