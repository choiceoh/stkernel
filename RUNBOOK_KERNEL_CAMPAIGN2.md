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

## 현재 상태 한눈에 (2026-09-05)

각 항목의 근거는 아래 해당 절과 원장에 있다. **부팅** 열이 자원이다 — 부팅 창이
열리면 이 표의 "부팅 대기" 부터 고른다.

| EXP | 무엇 | 상태 | 부팅 | 지금 값 · 근거 |
|---|---|---|---|---|
| 6 | 메가커널 세그먼트 | **채택 · 프로덕션 기본값** | — | `MEGAKERNEL`·`MK_MHC`·`MK_GEMM`·`MK_MLA`=1 (28차 §8) |
| 10 | 드래프터 GEMM → MK W4 | **채택 · 기본값** | — | C=1 15.95 → **16.235**(+1.8%), 28차 §4 |
| 12 | 서빙 PDL | **채택 · 기본값** | — | 발사당 58.0 → 53.6 µs(27차 오프라인). **종단 수치는 아직 없다** |
| 8 | dflash async scheduling | **기각** | — | 이미 켜져 있었다(`dflash` ∈ `EagleModelTypes`), 모듈 제거 |
| 14 | MK_SEG_MOE go/no-go | **닫힘** | — | b12x static 197 vs MK 레인 196 GB/s = 103%(27차). 재개 조건은 마이크로커널 240 |
| 7 | 준비 커널 통합 (prep_fused) | 부팅 대기 | ✅ | `PREP_FUSED=0` · 표적 = 유휴 8.18 ms(지도 보충 분해 5) |
| 1 | b12x EP | 부팅 대기 | ✅ | `ENABLE_EP=0` · 구현 완료, 측정만 남음 |
| 2 | custom_ops 융합 | 부팅 대기 | ✅ | `CUSTOM_OPS_AXIS=all` · 기대값 하향: 글루 481발 전부 합쳐 1.00 ms |
| 13 | AR 프리페치 | 브래킷 대기 | ✅ | `AR_PREFETCH=0` · 상한 −0.4~0.6 ms(27차 정정), EXP-6+12 위에만 |
| 18 | osar 벡터화 + 캐시정책 | 브래킷 대기 | ✅ | 코드 반영 완료. 직렬 절감 단독은 ~0.2~0.3 ms 로 CV 미달 — **EXP-13 팔과 같은 부팅**에서 L2 위생 시너지로 판정(#303 정정) |
| 9 | 인덱서 head-gate split-K | 얹기 대기 | ✅ | `INDEXER_GATE_SPLITK=0` · 무장 트레이스에도 `gemmSN` 11발 그대로 |
| 15 | 드래프터 early-fc | 얹기 대기 | ✅ | `DFLASH_EARLY_FC=0` · 상한 ~0.3 ms |
| 4 | b_proj/indexer fp8 | 대기 | ✅ | `FP8_DENSE_BPROJ` 프로필 미설정(= 모듈 기본 off) · 표적은 무장 뒤에도 bf16 201발 중 ~112발 |
| 3 | MHC small-M 스윕 | 프로브 먼저 | — | `MHC_SMALLM` unset = 불활성 |
| 17 | MHC TileLang 패스설정 | 프로브 먼저 · **표적 축소** | — | `MHC_PASSES` 기본 off (#301). 무장 뒤 디코드 몫은 잔여 mhc 14발뿐이고 주종은 **프리필 big_fuse**(#303 정정) |
| 11 | dsv4 에 MK_SEG_MHC | 프로브 먼저 | — | 2단계는 무부팅, 이기면 dsv4 다운타임 창 |
| 19 | hc 가중치 bf16 | **사실상 초월** · 참조 측정 | — | mk_mhc 가 pair 를 대체해 ONEPASS 전제가 사라졌다 — 폴백 경로의 커널급 참조로만 유효(#303) |
| 5 | 프리필: 캡처 → KDA 스윕 | 부분 완료 | — | 캡처·프로파일은 09-03 원장에 있고 KDA 청크 스윕이 남음 |
| 16 | 드래프터 메가커널 | 제안 · 착수 승인 대기 | — | 상한 −0.8~1.0 ms, 공사 2주 |

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

| 세그먼트 | 흡수 | 예측 (런치/스텝) | **실측** (09-04 무장 트레이스) |
|---|---|---|---|
| MK-MHC | hc post+pre (수학은 소유 TileLang 소스의 비트 충실 포팅) | 179 → 45 | **179 → 89** + stock 잔여 7 · 2.54 → 2.07 ms |
| MK-GEMM | per-token quant + W8A8 GEMM, M≤32 | ~360 → ~180 | **376 → 187** (deep_gemm 197 + quant 179 → mk_gemm 185 + lm_head 2) · 밀집 GEMM 시간 14.06 → 14.13 ms(제자리), 양자화 몫 −1.5 ms |
| MK-KDA | KDA 블록 전체(in_proj→conv→recurrent→norm→o_proj) | ~510 → 34 | **미무장** (`MK_SEG_KDA=0`) — KDA 102 발 그대로 |
| MK-MLA | sparse MLA 디코드 (NoPE, fp8 KV) | — | 28차부터 기본 on · 디코드 +1.0%, 프리필 +15~18% |

**세트는 2026-09-04 20:35 부터 프로덕션 기본값**이다(28차 §8: MEGAKERNEL·MK_MHC·
MK_GEMM·MK_MLA=1, KDA 0). 남은 세그먼트는 KDA 하나이고, conv 상태 dtype 계약을 여는
`MAMBA_CACHE_DTYPE` 노브가 붙어 있다.

천장이었던 "5.4µs/커널 × 약 900개 ≈ 4.9 ms" 는 **낙관으로 확인됐다**: 세트가 커널을
304 발 지웠는데 스텝에서 시간이 준 자리는 MHC(−0.5 ms)와 양자화(−1.5 ms) 둘뿐이고,
밀집 GEMM 자체는 제자리였다(STEP_KERNEL_MAP 보충 분해 5 · ⑥). **개수가 아니라 합쳐진
일이 시간을 준다** — 다음 세그먼트 후보를 발사 수로 정렬하지 말 것.

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

## EXP-22 — 작은 형상 GEMM 의 배리어 없는 로컬 양자화 경로 (`VLLM_GLM53_MK_LOCALQ`, 2026-09-04 추가)

28차의 30~45 µs 클래스 — 공유 전문가의 gate_up `[1024 × 4096]` 과 down `[4096 × 512]`, 스텝당
86발, 3.5 ms, 그중 2.4 ms 가 다른 스트림에 덮이지 않는 임계경로 — 는 바이트로는 1~2 MB(레인 속도로
5~12 µs)인데 발사당 30~45 µs 다. 남는 것은 발사 고정비: grid-wide A 양자화(블록당 k-블록 하나를
`g_mk_aq` 로) + 공표 배리어와 그 스큐(x 로드가 호이스트된 W 채움 뒤에 줄을 선다, 4차 스탬프
프롤로그 중앙값 8~10 µs) + `sxs` 왕복 + 그 전부를 기다리는 유휴 16 블록. 게다가 서빙은 이 두
발사를 라우팅 MoE 커널 옆 aux 스트림에 띄우므로 48블록 persistent 발사는 MoE 블록이 물러나는
대로 SM 을 얻고, 배리어는 마지막 블록이 들어올 때까지 상주 블록 전부를 잡아둔다.

- 변경: 호출 측이 bg 로 표시한 발사(fp8-dense 훅의 `mlp.shared_experts.*`; 노브 1) — 또는 벤치의
  노브 2 로 유닛 ≤ grid 인 모든 발사 — 는 `mk_gemm_lq_kernel` 을 **유닛 수만큼의 블록**으로 띄운다:
  각 블록이 자기 유닛의 A k-블록을 x(L2)에서 직접 양자화해 smem 에 넣고(두 k-블록 앞서 로드, mma
  앞에서 축약), 배리어·전역 타일 없음. 산술이 `quant_store` 와 같은 헬퍼(`mk_pack4`·`mk_warp_amax`·
  `mk_pow2_scale`)라 출력은 같은 split 에서 **비트 동일**(자가진단이 두 커널을 exact + 비트 동일로
  검사, 벤치 `same` 열). 큰 형상과 KDA 인라인 phase 는 전역 커널(`mk_gemm_kernel`, 80 레지스터).
- 게이트(순서대로): (1) `tests/test_logic.py`, (2) srv2 빈 창에서 아래 셋 — 스윕의 `same` 전부 yes
  + exact 게이트 PASS + 리플레이 안정, 표준 표에서 큰 형상(n=6416/4096, 전역 경로)이 main 과 같을 것,
  (3) MoE 아래 동시 실행 프로브의 `exposed per layer` 가 lq=1 에서 줄 것(스텝에 매핑되는 수치),
  (4) 플릿 브래킷은 EXP-6 위에 얹는다(수치 불변 → step/s 만; PDL 전례).
- 상한(29차로 정정): 단독으론 없다 — 전역 프롤로그(x 로드+양자화+배리어)는 정상 상태 스탬프로
  5.5 µs 뿐이고 로컬 경로의 루프 안 양자화가 그만큼 되돌려준다(첫 판 +4~12 µs, v2 32블록 쌍 +8 µs,
  행 술어화 판 미실측). 남는 것은 MoE 아래 **노출분**: 동시 실행 프로브가 정한다(전역 47.4 → 로컬
  32블록 31.8 µs/층, ×42 층 ≈ −0.66 ms/스텝의 *투영*; 프로브와 스텝의 형태 차이는 프로브 docstring).
  단독 부팅 금지.
- 스윕 노브: `VLLM_GLM53_MK_KSR`(0 = 비용 모델; 비용 모델은 경로와 무관하게 같다 — 29차 스윕에서
  로컬 경로도 모델의 선택(n=1024 ksr 4)이 최선이었다). 발사 그리드(로컬 32/48/16, 전역 32 대조군)는
  벤치 전용 `set_probe` 로만 바꾼다 — env 표면 없음.

```bash
bash probes/run_megakernel_bench.sh --segments gemm,exact --iters 20      # 표준 표 + exact 게이트
bash probes/run_megakernel_bench.sh --gemm-sweep --iters 20               # local × split, same 열
VLLM_GLM53_MK_PHASE_TS=1 bash probes/run_megakernel_bench.sh --gemm-sweep --stamps --iters 10
bash probes/run_mk_probe.sh probes/mk_gemm_concurrent_probe.py            # MoE 아래 동시 실행
```

**결과(2026-09-04 23:36~42, srv2 빈 창, 29차)**: 첫 판(48블록 발사, 양자화가 sync 구간 안)은 **단독으로 손해**
— 같은 split 에서 lq0 → lq1 이 m=8 n=1024 24.7 → 28.7, m=32 n=1024 32.6 → 45.0 µs; 스탬프로 보면 전역
경로의 프롤로그(x 로드+양자화+배리어)는 5.5 µs 뿐이고 로컬 경로는 k-블록마다 ~0.85 µs 를 루프에
더한다(4차의 "8~10 µs" 는 콜드 첫 발사 값). 비트 동일·exact·리플레이는 전부 통과. **MoE 아래 동시
실행**(`mk_gemm_concurrent_probe.py`, U=40 740 µs): 전역 경로는 두 순서 모두 쌍 전체가 노출(43.8/44.9 µs
= 쌍 단독 40.5 + 발사), 로컬 경로는 moe-first 17.3 / pair-first 41.8 — 이득의 정체는 양자화가 아니라
"배리어 없음"이고, pair-first(서빙의 aux 스트림 순서)에선 48 블록이 SM 을 다 잡아 MoE 가 기다린다.
→ v2: 로컬 커널을 유닛 수(32)만큼만, 또는 더 적게(`VLLM_GLM53_MK_LQ_GRID`) 띄우고 양자화를 wait 앞으로;
호출 측이 공유 전문가 선형을 bg 로 표시(`VLLM_GLM53_MK_LOCALQ=1` = bg 만, 2 = 유닛 ≤ grid 전부).
v2 의 동시 실행 수치는 원장 29차.

## EXP-21 — MK-GEMM v2: 비상주 GEMM 레인 (`VLLM_GLM53_MK_GEMM2`, 2026-09-05 추가)

**근거**: 원장 30차. 무장 트레이스(09-04 18:42)의 mk_gemm 185발 14.13 ms 중 5.7 ms 는
공유 전문가 down [4096×512] 42발(단독 18 µs → 서빙 135 µs): 상주 48블록(smem 69.6 KB)이
MoE 커널(90 KB)과 SM 을 못 나눠 aux 스트림에서 직렬화되고 배리어가 꼬리를 품는다.
deep_gemm 은 독립 블록이라 같은 GEMM 을 MoE 꼬리 안에서 끝냈다(스톡 트레이스 36 µs).
메인 스트림 99발은 발사당 고정비 15~20 µs + 정적 배분 꼬리(dense gate_up 119 µs/바닥 62).

**팔**: `VLLM_GLM53_MK_GEMM2=1`(기본 0). 커널 `mk_gemm2_kernel` — grid = (n/128)×ksr
독립 블록, 배리어·공유 A 없음, W4 레지스터 전개, smem 37 KB(SM 당 2블록), 결정적 폴드.
수치: 비분할 형상은 v1 과 비트 동일, 분할 형상은 fp32 슬라이스 합 순서만 다름(exact 게이트
두 레인 × ksr 5종).

**오프라인 게이트(부팅 전, srv2 GPU 창)**: `run_megakernel_bench.sh --segments exact,gemm
--gemm2 both`(기본 `--gemm2 env` 는 서빙 레인만 판정한다) 전 행 PASS + v2 열이 v1 열보다
형상마다 같거나 빠름 + `mk_gemm_moe_overlap_probe.py` 의 v2 노출이 v1 의 28 µs/층보다 작음.
상태(09-05 01:39, 원장 30차 §4~6): **오프라인 게이트 전부 통과** — exact PASS(두 레인 × ksr
5종), 규칙판 v2 x2 가 m ≤ 16 전 형상에서 v1 보다 빠르고 m = 32 동률, 노출 프로브 47.3 → 28.6
µs/층(쌍 먼저 순서). 브래킷 착수 가능(기대 −1.3~−1.8 ms/스텝). 상한: 스텝 −2.5 ms
(메인 −1.5, aux 노출 −1.0), 트레이스 mk 합 14.1 → ~7.

**부팅 게이트**: base(기본값) → cand(`VLLM_GLM53_MK_GEMM2=1`) 브래킷, step/s(acc 정규화)
+ 프리필 동반 + 품질 9/9 + 한국어 0/16 + pos-1 ±2 pct. 통과하면 프로필 기본 1 로 올리고
v1 상주 커널은 KDA 내장 phase 로만 남긴다(운영자 규칙: 이득 확인된 개선은 기본값, 반대쪽 삭제).
**주의**: SMLP(EXP-20 계열, 상주+배리어)를 같은 자리에 켜면 이 진단의 직렬화가 되살아난다 —
공유 전문가 융합은 v2 구조(독립 블록 + 타일 도착 의존) 위에 다시 지어야 한다.


## EXP-20 — 자체 소유 미세 융합 묶음 2 (2026-09-04 추가, "소소한 이익 묶음" 방식)

개별로는 브래킷 해상도(CV 1.7%) 아래라 판정 불가한 네 축을 R2 선례(MEASUREMENTS
"R2 — mhc 라운치 구성 2차 묶음")대로 **한 캠페인 창에 묶어** 채택 판정한다:
오프라인 수치 게이트 + 인그래프 물리확인(트레이스에서 대상 커널 소멸) + e2e 무회귀.
09-04 18:42 rank3 트레이스(무장 세트, 스텝 64.8 ms, 커널 1,548개) 기준 대상:

| 축 | 노브 | 제거 런치/스텝 | 스텝 Δ(31차 측정, 조용한 GPU) |
|---|---|---|---|
| ~~게이트 체인 완결~~ — epilogue top-k 는 bit-exact 였지만 한 CTA 의 직렬 선택이 대체한 커널보다 느렸다(M=8 28.2→31.3 us/층). **오프라인 기각, 트리에 없음**(31차) | — | (42) | — |
| f_b+g_b 듀얼 GEMM — 같은 병합 행의 인접 128열 두 슬라이스, 한 런치에 dot 둘 | `VLLM_GLM53_KDA_DUAL_GEMM=1` | 34 (wmma 4.7~6 us) | −0.12 ms (M=8, BLOCK_N=32) |
| KDA 원패스 — conv + q/k/v/beta 복사 4 + recurrent + gated norm 을 순수 spec-verify 스텝에서 한 런치로 | `VLLM_GLM53_KDA_ONEPASS=1` | 204 (층당 7→1) | −0.04 ms (C=1) / −0.23 ms (C=4) |
| kpool 갱신 — int32 캐스트 복사 없이 int64 positions 직접 | `VLLM_GLM53_KPOOL_UPDATE_DIRECT_POS=1` | 11 (`direct_copy` 1.7 us) | −0.014 ms |

수치 등급: KDA 상태(conv·recurrent)와 출력은 프로브의 플릿 형상 8 케이스(bf16/fp32 conv 상태,
2스텝 사슬 포함)와 부팅 자가진단 3 케이스에서 **bit-exact**
(gated norm 의 128-합 트리가 stock 커널과 달라 등급은 reduce-order 로 선언) → 묶음 부팅은
품질 9/9 + 한국어 0/16 + 수용률 ±2 pct 게이트를 진다. 듀얼 GEMM 은 M≤16 bit-exact(M=32 는
cuBLAS 가 다른 커널을 골라 1 ulp 소수). kpool 은 값 동일.

MK_SEG_KDA(`VLLM_GLM53_MK_KDA=1`)가 서빙되면 KDA 두 축은 무효(메가커널 분기가 먼저).
"kpool 원패스 11층→1" 은 층간 데이터 의존(층 L 의 갱신은 층 L 의 K 를 먹고 같은 층의
logits 가 그 캐시를 읽음) 때문에 플래그 대기 persistent 커널 없이는 불가 — 스텝 내내
SM 을 점유하는 그 형태는 채택 대상이 아니라서 캐스트 복사 제거로 축소했다.

```bash
bash probes/run_micro_fusion_check.sh            # 오프라인 게이트 (fresh container)
VLLM_GLM53_KDA_DUAL_GEMM=1 VLLM_GLM53_KDA_ONEPASS=1 VLLM_GLM53_KPOOL_UPDATE_DIRECT_POS=1 \
  bash launchers/start-glm53-nvfp4-tp4.sh   # cand
```

- 부팅 로그 앵커: `[kda-onepass] self-test PASS (...) -> one-pass ARMED`(자가진단, 첫 eager
  forward), `[kda-onepass] dual gate GEMM serving`, `[kda-onepass] one-pass KDA serving`.
  경고 줄 `decode shape not admitted -> stock two GEMMs` / `tensors not admitted -> stock chain` /
  `self-test FAIL ... DISARMED` 가 있으면 그 축은 안 돈 것이다. 프리필·프로필 런(M>32)은 설계상
  stock 이고 로그되지 않는다. conv 상태 dtype 은 bf16·fp32 둘 다 받으므로 MK-KDA 의
  `MAMBA_CACHE_DTYPE=float32` 부팅에서도 돈다(MK 가 서빙하면 MK 분기가 먼저 반환해 무효).
- 물리확인: cand 트레이스에서 `_causal_conv1d_update_kernel` 0, `layer_norm_gated_fwd_kernel` 0,
  KDA 층당 `_kda_onepass_spec_kernel` 1, `_dual_gate_gemm_kernel` 1, 인덱서 층당 `direct_copy`
  −1. 스텝의 GDN 창 시간(트레이스 도구)으로 ms 를 잰다.
- 브래킷: base → cand(네 노브) → base, C=1 step/s + 프리필 2048/8192 동반. 채택 뒤
  프로필 기본값으로 올린다(운영자 규칙: 이득이 확인된 개선은 기본값).


## 브래킷 자동화 — `bench/bracket.py` (도구, 판정 아님)

`leg`(살아있는 서버에 rep 기록) + `judge`(기록 판정) 2중 명령. 원장 규율을 코드로
박은 것: 판정 채널 C=1 step/s (`tok/s ÷ (1 + k×raw_acc)`), 유의 문턱은 **base 두
다리의 드리프트**(같은 설정 재부팅 차이). 재기동은 여전히 사람이 한다 — 이 도구는
읽기 전용이고 다리 사이 env 스냅샷을 기록해 #116 부류(노브 미전달 부팅)를 judge 가
보게 한다. C≠1 기록은 참고용으로만 남긴다.

---

---

## EXP-17 — MHC TileLang 패스설정 A/B: TMA lowering · warp specialization (2026-09-04 추가)

이미지 원본이 모든 mhc 커널에서 **둘 다 꺜 두었다** (`tilelang_kernels.py`
의 `TL_DISABLE_TMA_LOWER: True` · `TL_DISABLE_WARP_SPECIALIZED: True`).
GB10엔 TMA가 있고(deep_gemm sm120이 실사용) 비활성 사유의 기록이 없다.

**표적 축소(2026-09-05)**: 무장 디코드가 mk_mhc 로 TileLang mhc 179발을
대체했다(185→14발, 지도 ④). 이 노브의 디코드 몫은 잔여 14발 + **프리필
big_fuse** 가 주종이다 — 프로브의 pair/onepass 시간은 여전히 유효한
커널급 A/B 지만 스텝 환산 천장은 mk 이전 수치로 쓰면 안 된다.
vllm-mhc 쪽 PDL은 다르다 — 이미지가 SM12x에서 "unvalidated + KDA state
kernel 경합"으로 명시적으로 봉인했다(신규 컨테이너 실측 False, 원장
2026-09-04). **vllm-mhc PDL 축은 재개 금지.** (MK 자체 PDL=EXP-12 는 별개.)

1단계 — 프로브 (srv4 새 컨테이너):

```bash
ssh srv4 'cd /home/choiceoh/stkernel && probes/run_mhc_glm53_bench.sh --passes'
```

- 4콤보(none/tma/ws/tma,ws) 각각 **독립 컨테이너**(패스설정은 import 시
  컴파일에 굳는다). stock 조합이 참조를 저장(`--ref-save`)하고 나머지가
  대조(`--ref-load`, 게이트 rel≤1e-4). ONEPASS도 함께 켜 각 조합에서
  pair+onepass 둘 다 잰다. 조합이 컴파일에 실패하면 그 자체가 그 조합의
  판정으로 기록되고 스윕은 계속된다.
- 시간 승자가 없거나 rel 게이트 깨지면 "비활성은 필요했다"로 축 종결.

2단계 — 브래킷 (승자가 있는 경우만):

```bash
EXTRA_ENV="VLLM_GLM53_MHC_PASSES=<승자조합>" \
  bash launchers/start-glm53-nvfp4-tp4.sh   # cand
```

- 부팅 로그의 `[deneb] VLLM_GLM53_MHC_PASSES=... TL_DISABLE_...=...` 라인으로
  engine-confirmed 값을 확인한다. 게이트: 9/9 + 한국어 0/16 + C=1 step/s.

## EXP-18 — osar copy/reduce 16B 벡터화 + 캐시정책 (코드 반영 완료, 브래킷 대기)

#99 위상 실측의 직렬 성분(copy 6.6µs + reduce 5.2µs/콜렉티브)을 2라운드로
절감했다. **R1(벡터화)**: copy/reduce를 16B(8×bf16) 렌으로 — 발행 명령 8배
감소, reduce의 `src` 재독기를 레지스터 스태시로 제거, warp 요청 64B→128B.
**R2(캐시정책·팩 변환)**: `tx` 저장 `__stwt`(쓰기-스루), 피어 `rx` 로드
`__ldcs`(evict-first), 변환 `__bfloat1622float2`/`__float22bfloat162_rn` 팩.
요소별 연산 순서와 rn 반올림이 불변이라 **출력은 스칼라 원조까지 비트
동일**. 펜스·done_ctr·48블록 그리드·EXP-13 프리페치 기계장치는 불변.
16B 미정렬 입력은 shim `_eligible`이 NCCL로 돌린다.

- **기대치는 정직하게(2026-09-05 무장 시대 수치로 정정)**: #99 시대의
  copy+reduce 11.8µs/콜 전제는 낡았다 — 현재 osar 는 45.5µs/콜·4.70ms/스텝
  (14.5%)이고 고정비 ~22.7µs(two-shot 기각 판정)가 지배하므로 **R1 의 직렬
  절감 천장은 ~0.2-0.3ms(0.3-0.45%)로 단독 CV 미달**. R2 의 본체는 L2 위생이고
  **EXP-13 프리페치와의 시너지가 주 판정 근거다**: osar 의 L2 churn(~100콜 ×
  ~256KB ≈ 26MB/스텝)은 프리페치 예산(12MB, L2 24MB)과 직접 경쟁하는데,
  `__stwt`/`__ldcs` 로 그 간섭을 제거하면 프리페치가 예열한 라인이 살아남는다.
  `__stwt` 는 L2 쓰기-결합을 포기해 copy 위상 자체를 늦출 수 있다 — 위상 로그는
  진단, **판정은 C=1 브래킷(EXP-13 팔과 같은 부팅에서 관찰)**.
- 검증 순서(귀속 주의): 부팅 osar self-test PASS → 부팅 로그
  `[osar] source md5=b0275622 kernels=1` 지문 확인 → `[osar] phase` 로그에서 R1은 copy/reduce 감소가 기대치,
  R2의 ldcs+팩 변환은 reduce 추가 감소, stwt는 copy에 비용 가능 —
  → C=1 step/s 브래킷 + 9/9 + 한국어.
- 게이트 없는 코드 반영이므로 다음 배포부터 dsv4·glm53 양 프로필 모두에
  적용된다. 롤백 = 모듈 리버트 후 재배포(manifest SHA가 게이트한다).

## EXP-19 — hc 가중치 bf16 (`--hcweight` 프로브 — 수치 축, 후순위)

**사실상 초월(2026-09-05)**: ONEPASS 는 한 번도 채택되지 않았고 무장
디코드는 mk_mhc 로 pair 를 대체했다(ONEPASS 가 노리던 "두 발을 한 발로"를
메가커널이 더 깊게 수행). 이 프로브는 이제 **폴백 경로의 커널급 참조 측정**일
뿐이고, "fp32 weight fp32 플로어"라는 같은 질문을 프로덕션 경로에 하려면
mk_mhc 쪽이 별도 축이다. (아래 수치는 스톡 체인 기준의 역사 참고.)

mhc onepass 1회 호출의 지배 비용이 `weight_t` fp32 ~1.57MB 읽기(DRAM
플로어 5.8µs × 90호출 = 0.52ms)다. bf16이면 절반이지만 "hc weights are
fp32 by design"이라 품질 축이다. 프로브는 오류 두 종류를 분리한다:
대조군(같은 반올림 가중치를 fp32로 먹인 스톡 onepass) 대조 ≤1e-4는 **전위
오류 게이트**, 스톡-fp32 대조 오차는 양자화 비용으로 **보고만** 한다.

```bash
ssh srv4 'cd /home/choiceoh/stkernel && probes/run_mhc_glm53_bench.sh --hcweight'
```

- 시간 승자 + 양자화 오차가 수용 가능해 보여도 채택은 별도 작업이다:
  바인딩 시점 가중치 캐스팅 배선 + 풀 품질 게이트(9/9 + 한국어 + 브래킷).

---

## 순서와 근거

채택·기각·닫힘은 위 상태 표에 있다. 여기는 **남은 것만**, 무엇을 먼저 할지의 이유와
함께 적는다. (번호는 EXP 번호이고 순서가 아니다.)

**무부팅으로 먼저 — 부팅 창을 아끼는 것들**

- **EXP-3 (MHC small-M)** · **EXP-17 (TileLang 패스설정)** — 프로브가 승자를 가려
  브래킷 부팅을 한 번으로 줄인다. 둘 다 같은 커널이라 한 창에서 함께 재고,
  이기는 쪽만 부팅에 올린다.
- **EXP-11 (dsv4 에 MK_SEG_MHC)** — 2단계는 부팅 없음: 서빙 컨테이너가 비었을 때
  srv2 에서 `bash probes/run_megakernel_bench.sh --profile dsv4 --iters 20`.
  판정은 rel 1e-3 + T=8/16 의 dispatch 행(`hit=yes` 인 행만 의미가 있다). 이기면
  3단계는 **프로덕션 dsv4 의 다운타임 창**이라 GLM 브래킷과 창을 다투지 않는다 —
  다만 래퍼가 T ≤ 16 이므로 C=4 다리는 통제군과 같아야 정상이다. 지면 여기서 닫는다.
- **EXP-19 (hc bf16)** — 수치 축이라 승자여도 배선·품질 게이트가 별도. 후순위.

**다음 부팅 창에서 — 한 부팅에 얹을 수 있는 조합**

1. **EXP-7 (준비 커널 통합)** — 표적이 GPU 유휴 8.18 ms(무장 트레이스)로 남은 것 중
   가장 크고, GPU 커널 축과 독립이라 다른 암과 섞이지 않는다. bit-exact 프로브 →
   섀도 부팅 → 브래킷 순서를 건너뛰지 않는다.
2. **EXP-9 (head-gate split-K)** · **EXP-15 (드래프터 early-fc)** — 단독 부팅 금지,
   EXP-7 부팅에 얹는다. 각각 ~1.2% · ~0.3 ms 급이라 단독으로는 CV 아래다.
3. **EXP-13 (AR 프리페치)** · **EXP-18 (osar 벡터화)** — 수치 불변이라 cand 팔에
   얹어서만 판정한다(EXP-6+12 위). EXP-13 의 상한은 −0.4~0.6 ms 로 이미 정정됐다.
4. **EXP-1 (EP)** — 부팅 하나로 축 하나가 원장에 정리된다. 실패해도 기록이 남는다.
5. **EXP-2 (custom_ops)** — 코드 변경 0. 다만 기대값은 낮아졌다: 무장 스텝의
   elementwise 481 발을 **전부** 없애도 1.00 ms(1.4%)다.
6. **EXP-4 (b_proj/indexer fp8)** — 천장 0.9%. 표적(저랭크 뒤쪽 절반 ~112 발)은
   무장 뒤에도 그대로 남아 있으니 창이 남을 때 얹는다.

7. **EXP-22 (공유 전문가 GEMM 의 로컬 양자화 커널)** — 29차: 단독 손해, MoE 아래 노출
   v1 48블록 17.3(moe-first)/41.8(pair-first) → v2 32블록 36.2/**31.8**(전역 47.4).
   EXP-21(비상주 v2 레인)과 같은 진단의 다른 처방이라 **같은 창에서 나란히 잰다**: 전역
   커널 32블록 대조군(이득이 "배리어 없음"인지 "블록 수"인지)과 행 술어화 판의 단독
   손해가 먼저. 이기면 EXP-6 브래킷의 cand 팔에 `VLLM_GLM53_MK_LOCALQ=1`(투영 −0.66
   ms/스텝 ≈ 1%, 브래킷 해상도 안쪽이라 세트로만). 단독 부팅 없음.
8. **EXP-20 (미세 융합 묶음 2)** — 오프라인 게이트 뒤 세 노브를 한 cand 로 브래킷. 개별 판정 없음(R2 방식). 게이트 epilogue 축은 프로브에서 기각.

8. **EXP-20 (미세 융합 묶음 2)** — 오프라인 게이트 뒤 세 노브를 한 cand 로 브래킷.
   개별 판정 없음(R2 방식). 게이트 epilogue 축은 프로브에서 기각.

**승인이 필요한 것**

- **EXP-16 (드래프터 메가커널)** — 상한 −0.8~1.0 ms(1.2~1.5%)에 공사 2주. 브래킷
  해상도(CV 1.7%) 아래라 단독 판정이 불가능하다는 것을 착수 전에 합의해야 한다.

**조건부 재측정**

- **드래프터 D** — "D≈0" 은 호스트가 병목이던 시절의 값이었고, 09-03 무장 트레이스가
  꼬리 7.4~8.5 ms 를 GPU 95% 바쁜 임계경로로 확정했다(EXP-10 이 그중 −1.1 ms).
  EXP-7 이 붙어 준비 유휴가 사라지면 **꼬리를 다시 잰다**.

## 금지 (기존 판정 유지 — 재조사하지 않는다)

추가(2026-09-04): **vllm-mhc 체인의 PDL 재활성** — 이미지가 "SM12x
unvalidated + KDA state kernel 경합"으로 봉인(원장 2026-09-04). MK 자체
PDL(EXP-12)은 별개 축.

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
