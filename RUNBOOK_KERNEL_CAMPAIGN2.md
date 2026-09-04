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

## EXP-8 — dflash async scheduling: **이미 켜져 있었다 (기각, 2026-09-03 부팅으로 확정)**

전제가 틀렸다. 이미지의 `config/speculative.py` 는

```python
DFlashModelTypes = Literal["dflash"]
EagleModelTypes  = Literal["eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes]
```

이고 `get_args` 는 중첩 Literal 을 평탄화하므로 **`dflash` 가 이미 허용 목록 안에 있다**
(`get_args(EagleModelTypes)` 마지막 원소가 `'dflash'`). 따라서 `config/vllm.py` 의 async
비활성화 조건은 **첫 항 `method not in get_args(EagleModelTypes)` 에서 단락**되고, 체인의
나머지 분기(모두 경고를 남긴다)도 걸리지 않아 끝의 `else: async_scheduling = True` 가 실행된다.

**부팅 증거(2026-09-03, 전 팔 부팅)**: 헤드·워커 로그에 "Async scheduling not supported"
도, 우리 화이트리스트 줄도, 다른 비활성화 경고도 하나도 없다. 조용한 경로는 pooling(디버그
로그)과 `else`(무로그) 둘뿐이고 이 모델은 pooling 이 아니다 → async 는 켜진 상태로 서빙 중.

**따라서 `glm53_async_dflash` 모듈은 죽은 코드였다** — 헬퍼는 두 분기 모두에서 도달 불가.
제거했다(노브 `VLLM_GLM53_ASYNC_DFLASH` 포함).

**천장 7~12% 주장도 함께 철회한다.** 9월 1일 트레이스의 호스트 준비 유휴는 async 가 켜진
상태에서 찍힌 것이다. async 스케줄링은 스케줄러 파이썬 작업을 겹치게 할 뿐, 워커의
`execute_model` 안에서 도는 입력 준비는 여전히 그래프 리플레이 사이에 직렬로 남는다.
그 구간은 **EXP-7(`glm53_prep_fused`)이 유일한 레버**다.

## EXP-9 — 인덱서 fp32 head-gate 를 split-K 로 (`glm53_indexer_gate_splitk`, 2026-09-02 추가, 09-03 정정)

`Indexer.forward` 의 `weights = torch.mm(hidden_states.float(), self._wp_fp32)`
([M,4096]×[4096,**32**] fp32 — 플릿 체크포인트의 `index_n_heads=32`; 첫 판은 N<=16 만
받아 플릿에서 한 번도 돌지 않았고 수치도 합성 N=16 이었다, 리뷰 #1 로 정정) 을 cuBLAS 가
2블록 `gemmSN` 커널로 답한다: 콜드 88 us, 9월 1일 트레이스 88 us, 층당 1회 × 11.
K 를 32개 프로그램으로 나눠 가중치 슬라이스를 한 번씩만 읽고 부분합을 **고정 순서**로
더하는 두 커널이 같은 곱을 12.7 us(M=8) 에 낸다 — atomic 없음, 리플레이·랭크 간 비트 동일.
**M<=16(디코드) 만** 이 경로, 나머지(프리필, C>=3)는 stock `torch.mm`(M=24 부터 17 us).

**수치**: 양쪽 다 fp32 누적, 합산 순서만 다르다 — bit-exact 가 아니다(서빙 수치 변경).
표와 오프라인 검증(순위 뒤집힘 0, 50회 비트 동일, stride·K 불일치·M=0 라우팅)은 모듈
README 한 곳에만 둔다; `probes/indexer_gate_check.py --config <ckpt>/config.json` 이
실패 시 비-0 으로 종료한다.

**천장**: 11 × (89.5 − 12.7) us ≈ 0.84 ms/스텝 = C=1 71 ms 스텝의 **~1.2%**. 단독 부팅은
하지 않고 **EXP-7 부팅에 얹는다** (EXP-7 은 bit-exact 라 그 부팅의 수치 축은 이것 하나 —
"부팅당 수치 축 하나" 규칙 유지).

```bash
# 프로필 선언 키: caller env. 부팅 로그의 "[indexer-gate] ... w(4096, 32) -> split-K" 가
# 켜진 증거; "shape not admitted" 가 보이면 노브는 켜졌으나 커널이 안 도는 것이다.
VLLM_GLM53_INDEXER_GATE_SPLITK=1 bash launchers/start-glm53-nvfp4-tp4.sh
```

- 게이트: 품질 9/9, 한국어 0/16, **pos-1 수용률 ±2pct**, C=1 step/s 브래킷 base→cand→base.
- 롤백 = env 한 줄. 기본 0 = stock 과 동일한 `torch.mm` 호출.

---

## EXP-10 — 드래프터 GEMM 을 MK W4 레인으로 (2026-09-04 개정: 프로브가 돌았고, 숫자가 나왔다)

**전제가 바뀌었다.** 09-03 무장 트레이스(STEP_KERNEL_MAP 보충 분해 4)를 러너의 준비
커널에서 잘라 보면 디코드 스텝 = 타깃 forward 59~62 ms + **꼬리 7.5~8.5 ms**(타깃 헤드
836 us + AllGather 0.4, 드래프터 fc 792 us, 5층 × ~660 us, 드래프트 헤드 812 us) 이고
GPU 는 꼬리 내내 95% 바쁘다. "호스트 유휴 뒤에 숨어 D≈0" 은 9/1 프로파일러 부팅의
사실이지 무장 부팅의 사실이 아니다 — 꼬리의 매 us 가 스텝 시간이다. 그리고 fc 는
deep_gemm fp4 가 아니라 **bf16 cutlass** 였다(168 MB, 212 GB/s; 9/1 메모의 809 us
fp4 귀속은 오귀속). 드래프터는 그 부팅에서 **TP=4** 로 돈다(o_proj/down_proj 뒤 AR,
12 MB qkv 스트림) — 프로필의 `DRAFT_TP=1` 은 서빙에 닿지 않았고, 대역폭으로는 TP=4 가
맞는 쪽이다(랭크 0 단독이면 층당 400 MB 를 혼자 읽는다).

**1단계 프로브 — 돌았다(2026-09-04, srv2, 실형상, TP=4 샤딩, DRAM-cold 순환, M=7)**:
`probes/drafter_fc_check.py`. 이전 판은 한 번도 실행된 적이 없었다(첫 팔에서 참조
matmul 전치 오류 + 그래프 헬퍼 인자 불일치로 즉사) — 프로브는 쓸 때가 아니라 돌 때
측정된다.

| 형상 (랭크당) | bf16 | fp8 pair | MK W4 |
|---|---|---|---|
| fc [4096×20480] | 682 | 489 | **301** (5 K-chunk) |
| qkv [1536×4096] ×5 | 60.5 | 33.5 | 22.2 |
| o [4096×1024] ×5 | 39.4 | 25.8 | 16.7 |
| gate_up [6144×4096] ×5 | 212 | 175 | 73.6 |
| down [4096×3072] ×5 | 104 | 83 | 40.5 |
| kernel_projection [1024×4096] ×10 | 46.8 | 26.1 | 18.7 |
| **스텝 합** | **3.23 ms** | 2.34 (−0.9, −1.3%) | **1.25 (−2.0 ms, −3.0%)** |

deep_gemm fp8×fp4 팔은 이 형상들에서 assertion 으로 SKIP. 헤드 형상 [38720×4096]
참고치: bf16 1306 / fp8 918 / MK W4 418 us — 서빙 헤드 둘(W8A16, 812·836 us)을 W4 로
바꾸면 각 −400 us 지만 타깃 헤드는 서빙 로짓(운영자 결정), 드래프트 헤드는 수용률 게이트.

**2단계 — 배선(랜딩, 부팅 없음)**: `VLLM_DFLASH2_FP8_DENSE=1` 이 드래프터의 dense 선형
전부를 덮는다(전엔 base 패턴이 o/gate_up/down 만 맞아 바이트의 43% 가 bf16 으로
남았다 — fc 만 23%). 드래프터 forward 는 torch.compile 되므로 GEMM 마다 **불투명
커스텀 op 하나**(`glm53_fp8_dense::gemm_mk_or_fp8`)가 런타임에 MK-or-fp8 을 고른다;
fc 는 K=20480 이라 레인의 K 계약(4096) 을 K-chunk 5개로 넘는다(`gemm_w4a8` 이 fp32 로
합산). `w8` 값은 fp8 pair 만(팩 없음) — 부팅당 수치 축 하나.
오프라인 게이트: `probes/drafter_dense_path_check.py`(빌드 패스 → 레인 비트 동일 서빙
→ fullgraph/dynamic compile → CUDA-graph 캡처 → w8 팔).

**3단계 — 브래킷(운영자 승인, 다운타임 규칙)**: 기준 → `VLLM_DFLASH2_FP8_DENSE=1`
(MK-GEMM 무장 부팅) → 기준. 판정: pos-1 수용률 ±2 pct(드래프터의 게이트), 품질 9/9,
한국어 0/16, C=1 step/s, **같은 부팅에서 프리필 2048/8192 cold·warm**. 기대값(정직):
W4 −2.0~2.4 ms/스텝 = C=1 −3~3.5%(수용률이 버틸 때), 대안 팔 `w8` −0.9~1.2 ms.
수용률이 2 pct 넘게 빠지면 `w8` 로 재브래킷.

**결과(2026-09-04, MEASUREMENTS 28차) — 기본값으로 올렸다.** 첫 브래킷은 Δ=0 이었는데
후보가 서빙되지 않은 측정이었다: vLLM 의 torch.compile/AOT 캐시는 vllm.envs 에 등록된
env·설정·forward 소스로 키하고 로드 후 바꿔 끼운 quant_method 는 키가 아니라서, 09-03
에 컴파일된 bf16 드래프터 그래프가 노브를 켠 부팅마다 그대로 로드됐다(eager 인 fc 만
MK). 노브를 캐시 키에 등록하고 forward 첫 호출들에서 opaque op 를 세는 증명 줄
(`drafter lane serving: 30 of 31`)을 넣은 뒤의 브래킷(두 팔 MK-MLA off): **C=1 step/s
15.95 → 16.235(+1.8%)**, pos-1 64.5% vs 61.6%, 품질 9/9, 한국어 0/16, 프리필 동일. →
`VLLM_DFLASH2_FP8_DENSE=1` 기본, `w8` 스킴 삭제(운영자 규칙). 롤백 = env 0.

## EXP-11 — dsv4 에 MK_SEG_MHC (2026-09-03 추가, 1단계만 랜딩)

메가커널 코어를 모델 무관 모듈로 쪼개고 `profiles/dsv4.env` 가 그것을 마운트한다.
**노브 전부 0, 부팅 0.** V4-Flash 에서 닿는 세그먼트는 MHC 하나뿐이다 — 두 모델의
하이퍼커넥션 기하가 같고(hc_mult 4 · sinkhorn 20 · hc_eps 1e-6 · hidden 4096)
`mhc_fused_post_pre_tilelang` 시그니처가 인자 순서까지 같아서, GLM 이 쓰는 훅을
`dsv4_mhc_tilelang` 에 같은 자리로 넣은 것이 전부다. KDA(선형어텐션 층 없음)·
MLA(rope 64 · topk 512 · 압축기 · 슬라이딩 윈도)·GEMM(dense 가 블록-fp8, W4 팩을
만들 bf16 원본 없음)은 해당 없음.

- **1단계 (완료, 이 커밋)**: `glm5next_kda.py` 행과 `requires` 를
  `glm53_mk_kda_wiring` 으로 분리 → 코어는 상대경로 + `absent` 2행만 남는다.
  glm53 컴포즈 결과는 전후 **바이트 동일**(매니페스트 행 순서만 이동).
  dsv4 런처에 프로필 키 EXTRA_ENV 가드 이식(여기선 `$COMMON` 이 `$ENVV` 보다
  먼저 렌더돼 프로필 값이 무조건 이긴다 — 스윕이 조용히 0 을 측정하는 자리).
- **2단계 도구 (완료, 같은 커밋)**: 프로브를 프로필 인자화했다. 프로필이
  이미지·컴포즈 트리·마운트 목록·패키지 루트를 정하고(`MK_PKG_PATH`: GLM 은
  dist-packages, dsv4 는 venv site-packages), `--segments`/`--stock` 이 세그먼트와
  기준 팔을 고른다. dsv4 기본값은 `--segments mhc --stock both`.

  ```bash
  bash probes/run_megakernel_bench.sh --profile dsv4 --iters 20
  ```

  `--stock raw` 는 기존 기준(직접 호출한 tile_n=2/n_splits=4 페어) — 기록된 수치와
  비교 가능한 **커널** 계측. `--stock dispatch` 는 래퍼를 MK 해제/무장 두 번 불러
  **부팅이 실제로 타는 팔**을 재고, MK 가 그 호출을 실제로 가져갔는지 `hit` 열로
  말한다. dsv4 에서 이게 중요한 이유: 이 레인의 stock 페어는 이미 스윕돼 있어서
  (R1/R2/R3) 스윕 안 된 기준에 대고 재면 스윕의 이득을 두 번 세게 된다.
- **★기준 변경 (2026-09-03, 리뷰 결과)**: `--sinkhorn` 기본값이 **20** 이다 —
  두 모델이 서빙에서 넘기는 `hc_sinkhorn_iters` 값. 프로브는 그동안 4 로
  하드코딩돼 있었고 MEASUREMENTS 의 MHC 행(4·9·10차)은 전부 그 값으로 잰
  것이다. 싱크혼은 양쪽 팔 모두 런타임 루프(`.cu` 의 `it < repeat-1`,
  TileLang 의 `T.serial`)이고 두 구현이 같은 비율로 늘지 않으므로 **4 에서의
  비율은 서빙으로 이전되지 않는다**. 옛 행 재현은 `--sinkhorn 4`, 실제 값은
  헤더에 찍힌다.

  **부팅 자가검증도 같은 값으로 옮겼다**(`SINKHORN_SERVED = 20`, 드라이버의
  단일 상수 — 프로브 기본값도 여기서 읽는다). 무장 여부를 정하는 게이트가
  루프 3회만 돌리고 서빙이 19회를 돌던 상태였다: 3회 뒤에 벌어지는 차이는
  게이트를 통과하고 서빙에서 틀린다. **부수 효과 예고**: 이 게이트가 이제
  더 엄해졌으므로 MHC 세그먼트가 DISARM 될 수 있다. 그러면 그건 회귀가
  아니라 **발견**이다 — stock 으로 떨어져 안전하고, 원인은 싱크혼 19회에서의
  MK↔TileLang 발산이다.
- **2단계 실행 (미착수, 부팅 없음)**: srv2 의 `production-hybrid-1.6` 이미지로
  위 명령. 판정: rel 1e-3 게이트 + T=8/16 의 dispatch 시간(그 위는 훅이 안
  걸린다). 훅이 한 번도 안 걸린 런은 **FAIL** 로 끝난다(전 행이 stock↔stock
  이라 rel 0 으로 통과해버리던 자리를 막았다).
- **전제 하나**: dsv4 는 `VLLM_USE_B12X_MHC` 가 꺼져 있을 때만 이 래퍼에
  도달한다(model.py 의 `_should_run_b12x_mhc` 가 먼저 cute 경로로 빠진다).
  그 노브는 이 플릿에서 비가용 판정이지만, 켜지는 날 훅은 도달 불가가 되고
  부팅 로그는 여전히 armed 라고 말한다.
- **3단계**: 2단계가 이기면 그때만. 프로덕션이고 리듀스 순서가 바뀌므로
  품질 9/9 + 한국어 0/16 + C=1/2/4 브래킷, 부팅당 수치 축 하나.
- **정직한 기대값**: 43층 × 페어 하나 = 0.2~0.3 ms / 43.6 ms 스텝 ≈ 0.5% 급.
  GLM 의 −16%/−41% 는 **스윕 안 된** stock 상대의 숫자다. 이 레인의 페어는 이미
  R1/R2/R3 로 스윕돼 있어(per-call 15.6 → 13.1 us) 격차가 더 작을 수 있다.
- **창 (2026-09-03 정정)**: 커널 게이트는 T ≤ 32 지만 **서빙 훅은 래퍼의
  `use_small_fma`(T ≤ 16) 안에 있다 — 두 모델 다.** dsv4 는 C×6 이라 **C ≤ 2**,
  GLM 은 C×8 이라 역시 C ≤ 2. 16 < T ≤ 32 는 stock 이 post+big_fuse 로 가는
  구간이라 MK 는 제안조차 못 받는다(문은 열려 있고 미측정). 기록된 T=32 MK-MHC
  수치는 전부 **커널 측정**(`_mhc_call` 직접 호출)이지 서빙 형상이 아니다 —
  프로브의 `--stock dispatch` 가 부팅이 실제로 타는 팔을 재고 `hit` 열로 그걸
  말한다.

## EXP-12 — 서빙 PDL (`VLLM_GLM53_MK_PDL=1`, 2026-09-04 추가)

메가커널 발사는 PDL(programmatic dependent launch)로 튜닝돼 있다 — 다음 MK 커널이
앞 커널이 비운 SM 에서 시작해 꼬리 동안 첫 W 타일을 당긴다(연속 2발 발사당
−17~19%, 2차). 그런데 드라이버는 env 를 읽고 **프로브만 그것을 켰다**: 프로필에도
`ab-glm53.sh` 의 cand 팔에도 없었으므로 지금까지의 무장 부팅은 전부 PDL 없이
돌았다(173발/스텝). EXP-6 종단 무효과의 용의자 1번.

- 수치 불변: 모든 MK 커널이 앞 커널 출력을 읽기 전에 `griddepcontrol.wait` 하고,
  wait 앞의 채움은 가중치뿐(mk_gemm_phase 의 hoist). 그래프 캡처 안 체인 형태는
  `probes/mk_pdl_graph_check.py` 가 판정한다(gemm→gemm→gemm, mhc→gemm 리플레이
  비트 동일 + PDL on/off 발사당 µs).
- 배선: `profiles/glm53.env` 기본 1, `ab-glm53.sh` cand 팔에 명시. 세그먼트가
  하나도 무장되지 않으면 무효(inert).
- 상한: 173발 × 5~10 µs = −0.9~1.7 ms/스텝. 단독 부팅 금지 — EXP-6 브래킷의 cand
  팔에 얹는다(수치 축 하나 규칙: PDL 은 속도만 바꾼다).

```bash
bash probes/run_mk_probe.sh probes/mk_pdl_graph_check.py                     # PDL=1
VLLM_GLM53_MK_PDL=0 bash probes/run_mk_probe.sh probes/mk_pdl_graph_check.py # 대조
```

**결과(2026-09-04, srv2, 27차)**: 체인 리플레이 비트 동일(on/off 모두), 24발 그래프 off 58.0 →
on 53.6 µs/발사(−7.6%). 프로필 기본 1. 남은 것 = EXP-6 브래킷의 cand 팔.

## EXP-13 — AR 대기 중 다음 커널 가중치 L2 프리페치 (`VLLM_GLM53_AR_PREFETCH`, 2026-09-04 추가)

`k_oneshot` 의 대기(회당 38.7/45.5 µs, 스텝당 ~100회)는 모든 랭크에서 DRAM 이
노는 시간이다. 커널이 `HintArgs`(최대 8개 (포인터, 바이트))를 받아 워프 1~7 이
`prefetch.global.L2` 로 걷고 스레드 0 은 그대로 플래그를 돈다. 힌트는 **학습**:
MK 드라이버의 발사(`_gemm_call`·`_mhc_call`·`_kda_launch`)가 읽을 가중치를
`note_consumer` 로 알리고, shim 이 "타깃 forward 의 몇 번째 콜렉티브 뒤인가" 로
파일해 두었다가 캡처 시 발사에 굳힌다. forward 경계는 컴파일 영역 위의
`Glm5NextForConditionalGeneration.forward`. MK-GEMM 이 무장돼야 배울 것이 있다.

- 게이트(순서대로): (1) `probes/osar_build_check.py` — 새 명령의 ptxas 통과(실패면
  부팅이 NCCL 로 조용히 떨어진다), (2) `probes/oneshot_ar_disttest.py` 4랭크 —
  12 MB 힌트/무힌트의 `t_wait` 와 maxerr 0(힌트가 NIC 쓰기를 밀어 대기를 늘리면
  손해), (3) `moe_decode_stream_probe.py` 의 `gemm cold` vs `gemm L2-warm` 행 —
  소비자 쪽 이득의 단위, (4) 플릿 브래킷: EXP-6+12 위에 `VLLM_GLM53_AR_PREFETCH=1`
  (caller env). 수치 불변 → step/s 만. 부팅 로그에 `[osar] prefetch hints learned:
  N collectives, X MB` 가 없으면 무장이 아니다.
- 상한: 임계 랭크의 대기 ≈ 전송 ~20 µs = 4.6 MB → −1.5~2.5 ms/스텝(전략 문서).
- 예산 노브: 1 = 12 MB/콜렉티브, N = N MB(1..20; L2 24 MB).
- **결과(2026-09-04, 27차)**: 게이트 1 PASS(빌드), 게이트 3 = n=6416 W4 GEMM cold 86.0 → L2-warm
  79.9 µs(−7%): 소비자가 DRAM 바운드가 아니라 상한은 **−0.4~0.6 ms/스텝**으로 내려갔다(전략 문서의
  −1.5~2.5 는 철회). 단독 판정 불가 — EXP-6+12 위에 얹어서만. 게이트 2(4랭크 disttest)는 부팅.

## EXP-14 — MK_SEG_MOE go/no-go (2026-09-04 추가, 프로브만)

21차의 "MoE 는 대역폭 바닥" 은 190 GB/s 를 **추정한 고유 전문가 수 ~40** 에서
역산한 값이라 순환이다. `probes/moe_decode_stream_probe.py` 가 서빙이 만드는
b12x 래퍼(같은 기하, 같은 디스패치 오버레이; C=1 은 64 pairs 라 static 백엔드)를
디코드 형상(8토큰·top-8)에서 고유 전문가 U=8..64 별로, 가중치 DRAM-cold(8 세트
순환), 그래프 리플레이로 재고 같은 바이트를 MK W4 레인이 스트리밍하는 속도(팩
12개 연속, PDL)와 견준다.

- 판정 규칙: b12x 가 레인의 90% 이상이면 축을 닫는다(원장에 기록). 아래면
  (레인 − b12x) 비율 × 31 ms 가 세그먼트의 상한이고, 설계는 전략 문서 4장(48블록
  persistent, FC1 (전문가, n타일) 유닛 → 전문가별 완료 카운터로 열리는 FC2 동적
  큐, 공유 전문가 = 41번째 전문가, b12x nvfp4 레이아웃 제자리 읽기, A4→A8).
- 실제 서빙 U 는 다음 부팅에서 로그 한 줄로 확정한다(프로브는 U 별 곡선만 준다).
- **결과(2026-09-04, 27차)**: b12x static U=40 = 197 GB/s, 레인 n=6416 = 196 GB/s → **103%, 닫힘**.
  읽기 전용 참조(torch sum) 245 GB/s 와의 20% 는 두 커널 공통의 발사 안 구조 몫이라, 재개 조건은
  "persistent 전문가 타일 스트림 마이크로커널이 240 에 닿는다" 는 2~3일 프로브의 양성이다. 보류.

## EXP-15 — 드래프터 fc 를 타깃 헤드·샘플러 아래로 (`glm53_dflash_early_fc`, 2026-09-04 추가)

디코드 꼬리는 타깃 헤드 → 로짓 AllGather → 거부 샘플러 → 드래프터 순인데, 드래프터의 첫
GEMM `fc`(aux 은닉 [토큰, 5×4096] → 4096; 레인에서 301 µs)는 타깃 forward 가 끝나는 순간
입력이 다 있다. stock 은 `propose()` 안에서 샘플러 뒤에 계산한다. 그 사이 구간(AllGather 는
패브릭, 샘플러는 소형 커널)은 DRAM 이 놀아 fc 가 공짜로 흐른다. 생산자 = `GPUModelRunner.
execute_model` 래퍼(forward 뒤 side stream 에서 cat + fc, 영속 버퍼 + 이벤트), 소비자 =
드래프터 오버레이의 `combine_hidden_states`(같은 토큰 수의 대기 결과만, 이벤트 대기 뒤). 소비가
`precompute_and_store_context_kv` 와 드래프터 그래프보다 앞이라 MK 발사끼리 겹치지 않는다.

- 수치 동일(같은 커널·같은 입력). 노브 `VLLM_GLM53_DFLASH_EARLY_FC=1`, 기본 0, 생산자 실패 시
  부팅 동안 자동 해제.
- 상한 ~0.3 ms/스텝(27차 인구조사: fc 만 그 시점에 입력이 준비된 꼬리 GEMM). EXP-10 위에서만
  의미(fc 가 레인에 있어야 함). 단독 판정 불가 — EXP-10 브래킷의 cand 에 얹는다.

## EXP-16 — 드래프터 메가커널 (제안, 착수 전 승인; 2026-09-04)

27차 인구조사(깨끗한 디코드 스텝): 꼬리 136 커널 중 GEMM 33(EXP-10 뒤 MK 발사 ~31), AR 11,
`kernel_mha` 5, 글루 ~50개 0.33 ms, 발사 간극 ~0.35 ms. 융합으로 없앨 수 있는 것은 글루 + 간극 +
MK 발사 고정비(30 × ~10 µs) ≈ **−0.8~1.0 ms(1.2~1.5%)**; AR 0.79 ms 와 mha 는 남는다. 브래킷
해상도(CV 1.7%) 아래라 단독으로는 판정할 수 없고, 공사는 MK-KDA 급(2주). 설계는 층당 두 발사:
[norm → conv.prepare → qkv GEMM] 과 [conv.finish → post-norm → mlp_conv.prepare → gate_up →
act → mlp_conv.finish → down], 어텐션(`kernel_mha`)은 stock 유지. **운영자 승인 뒤 착수** — 정정된
상한을 본 뒤의 결정이어야 한다.

## 브래킷 자동화 — `bench/bracket.py` (도구, 판정 아님)

`leg`(살아있는 서버에 rep 기록) + `judge`(기록 판정) 2중 명령. 원장 규율을 코드로
박은 것: 판정 채널 C=1 step/s (`tok/s ÷ (1 + k×raw_acc)`), 유의 문턱은 **base 두
다리의 드리프트**(같은 설정 재부팅 차이). 재기동은 여전히 사람이 한다 — 이 도구는
읽기 전용이고 다리 사이 env 스냅샷을 기록해 #116 부류(노브 미전달 부팅)를 judge 가
보게 한다. C≠1 기록은 참고용으로만 남긴다.

---

## 순서와 근거

1. **EXP-1 (EP)** — 기대값 최대. 실패해도 부팅 하나로 원장에 정리된다.
2. **EXP-2 (custom_ops)** — 코드 0, 부팅 1회, 축 자체를 닫거나 열어준다.
3. **EXP-3 (MHC)** — 프로브가 먼저 승자를 가려내 부팅을 아낀다.
4. **EXP-4 (bproj)** — 천장 0.9%, 최우선순위 아님. 다음 창으로 미뤄도 됨.
5. **EXP-5 (프리필)** — 캡처 1회가 관문. KDA 스윕은 그 다음.
6. **EXP-6 (메가커널)** — 프로브와 섀도가 먼저 수치·계약을 닫고 브래킷.
7. **EXP-7 (준비 커널 통합)** — bit-exact 프로브 → 섀도 부팅 → 브래킷. 호스트 유휴 4~5 ms 가 표적이라 GPU 커널 축과 독립.
8. ~~EXP-8 (dflash async scheduling)~~ — **기각**: 이미 켜져 있었다(`dflash` ∈ `EagleModelTypes`). 모듈 제거, 천장 주장 철회.
9. **EXP-9 (head-gate split-K)** — 단독 부팅 금지, EXP-7 부팅에 얹는다.
10. **EXP-7 이 붙은 뒤 드래프터 D 를 다시 잰다** — 9월 1일 트레이스에서 드래프터 ~4.3 ms 는 다음 스텝의 호스트 준비 유휴 뒤에 숨어 있었다(그래서 D≈0). 은신처가 사라지면 임계경로에 올라온다(천장 ~6%; fc GEMM 809 us 는 K=20480 직렬 스케줄이라 split-K 후보). #104 를 지금 재론하는 것이 아니라 조건이 바뀐 뒤의 재측정이다. 오프라인으로 닫을 >1% 레버는 더 없다(STEP_KERNEL_MAP 보충 분해 3).
11. **EXP-10 (드래프터 GEMM → MK W4)** — **닫힘, 기본값(2026-09-04, 28차)**: 서빙된 브래킷 C=1 step/s 15.95 → 16.235(+1.8%), 수용률·품질·한국어·프리필 게이트 통과. 첫 브래킷의 0 은 컴파일 캐시가 옛 bf16 그래프를 서빙한 탓 — 노브는 이제 캐시 키이고 부팅 로그가 `drafter lane serving: 30 of 31` 로 증명한다. MK-MLA 서빙 사망(스크래치 재할당)도 같은 항목에서 수정.
12. **EXP-12 (서빙 PDL)** — 프로브(그래프 체인) 뒤 EXP-6 브래킷의 cand 팔에 얹는다. 단독 부팅 없음.
13. **EXP-13 (AR 프리페치)** — 컴파일 → 4랭크 disttest → 브래킷(EXP-6+12 위). 수치 불변.
14. **EXP-14 (MK_SEG_MOE go/no-go)** — 프로브 하나가 착수 여부를 정한다. 90% 규칙.

12. **EXP-11 (dsv4 에 MK_SEG_MHC)** — 2단계는 **부팅 없음**: srv2 에서 서빙 컨테이너가
    비었을 때(`docker ps`) `bash probes/run_megakernel_bench.sh --profile dsv4 --iters 20`.
    판정은 rel 1e-3 + T=8/16 의 dispatch 행(`hit=yes` 인 행만 의미가 있다). 이기면
    3단계 브래킷인데 그건 **프로덕션 dsv4 의 다운타임 창**이고 EXP-10 의 GLM 브래킷과
    다른 스택이라 창을 다투지 않는다 — 다만 창은 C ≤ 2 뿐이라(래퍼의 T ≤ 16) C=4
    다리는 통제군과 같아야 정상이다. 2단계에서 지면 여기서 닫고 러너북에 숫자만 남긴다.

## 금지 (기존 판정 유지 — 재조사하지 않는다)

W4A8 기본화(#110), drafter 측 비용(#104), FP4 CUTLASS 패딩(출력 파손),
업스트림 b12x EP(#3383), ROCm 전용 융합 3종, `fuse_allreduce_rms`(sm_121 키
부재), `enable_qk_norm_rope_fusion`(NoPE), 클럭 상향(운영자 거부권).

## 부팅 전 확인 (2026-09-03, 하루에 네 번 같은 모양으로 틀린 뒤)

넷 다 "확인 가능한 것을 확인하지 않고 넘어간" 경우다. 부팅을 걸기 전에 이 넷을
본다 — 넷 다 1분 안에 끝나고, 넷 다 실제로 반나절을 먹었다.

1. **배포가 통과했나.** `deploy-overlays.sh` 는 HEAD 가 현재 origin/main 위에
   있지 않으면 거부한다(낡은 오버레이 롤백 방지). 그 ABORT 는 세 줄이고,
   부팅 스크립트가 `| tail -1` 로 받으면 마지막 줄(`origin/main <sha>`)만 보여서
   **평범한 상태 줄처럼 읽힌다.** 이걸로 세 번 부팅이 여섯 커밋 낡은 오버레이로
   떴고 로그는 멀쩡해 보였다. 파이프하지 말고 종료 코드를 볼 것.
   증상: 새 로그 줄이 안 나오는데 예외도 없다 → 그 코드는 컨테이너에 없다.

2. **그 모듈이 조합에 들어 있나.** `grep <file> build/glm53/manifest.tsv`.
   `glm53_drop_audit`(gpu_model_runner.py)과 `glm53_tail_slot_persistent` 는
   **어떤 조합에도 없는 고아**다. 거기 코드를 넣으면 조용히 죽은 코드가 된다.

3. **런처에 넘기는 값이 이미지에 실제로 있나.** 문자열 검증은 검증이 아니다.
   `LOAD_FORMAT=instanttensor` 는 case 문을 통과하고 75초 뒤 전 랭크가
   `No module named 'instanttensor'` 로 죽었다.

4. **GPU 수치를 다른 것이 GPU 를 쓰는 동안 재지 않았나.** 같은 프로브가 부팅 중
   43.5, 유휴 104.5 를 냈다 — 천장 대비 28% 냐 67% 냐를 가르는 크기다.

그리고 ISA 발행률은 명령 하나를 어셈블해 재고 끝내지 않는다. 같은 형식에 빠른
경로와 느린 호환 경로가 둘 다 있으면 느린 쪽만 재고 "이득 0"이라 적게 된다
(fp4: 155 로 적었는데 실제는 310). 벤더 스펙과 클럭 환산으로 교차 검증할 것.
