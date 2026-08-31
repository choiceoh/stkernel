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

KDA 프리필 커널은 벤더드 트리톤 패키지
`vllm/models/kimi_k3/nvidia/ops/third_party/kda/` (chunk/intra/
chunk_intra_token_parallel/fused_recurrent). autotune 캐시 키에 **T가 없어서**
(do_not_specialize) 프리필 T=8192와 디코드 T=8이 같은 (warps, stages, BK/BV)를
공유 — 먼저 오토튠한 레짐이 나머지를 지배한다. 레짐 분리가 레버.

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

- 좌표하강으로 커널별 config를 하나씩 갈아끼우고 실측 엔트리
  `chunk_kda_with_fused_gate` 시간을 재고, 스톡 출력과 rel err 게이트(>1e-2 플래그).
- 승자 → kda config 테이블 인수(env 게이트) → 프리필 tok/s 브래킷.
- 승자 없으면 축 기록으로 종료 (FLA 공동 튜닝 선례상 가능성 있음).

---

## 순서와 근거

1. **EXP-1 (EP)** — 기대값 최대. 실패해도 부팅 하나로 원장에 정리된다.
2. **EXP-2 (custom_ops)** — 코드 0, 부팅 1회, 축 자체를 닫거나 열어준다.
3. **EXP-3 (MHC)** — 프로브가 먼저 승자를 가려내 부팅을 아낀다.
4. **EXP-4 (bproj)** — 천장 0.9%, 최우선순위 아님. 다음 창으로 미뤄도 됨.
5. **EXP-5 (프리필)** — 캡처 1회가 관문. KDA 스윕은 그 다음.

## 금지 (기존 판정 유지 — 재조사하지 않는다)

W4A8 기본화(#110), drafter 측 비용(#104), FP4 CUTLASS 패딩(출력 파손),
업스트림 b12x EP(#3383), ROCm 전용 융합 3종, `fuse_allreduce_rms`(sm_121 키
부재), `enable_qk_norm_rope_fusion`(NoPE), 클럭 상향(운영자 거부권).
