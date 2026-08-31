# 사후 리뷰 — 2026-08-31 병합 배치 (#120–#137)

Date: 2026-08-31 (main 기준 커밋 `b549c46`, #137)
Scope: `f63c084`(#119) 이후 오늘 main에 병합된 18개 PR — b12x EP 축(#120–#127, #130),
fp8 lm-head 축(#129–#132, #135), 캠페인 II 접수(#128), mhc tilelang 축(#133, #136, #137),
docs(#134). 커널/디스패처/호스트 가드 전부, docs와 런북은 정합성만.

PR #138 후속 검토에서 최신 main의 #139–#141까지 병합해 충돌·재합성을 다시 확인했고,
GPU 프로브가 overlay를 실제로 import하지 않거나 ONEPASS 게이트를 너무 늦게 켜는
거짓 판정 경로를 추가로 찾아 닫았다.

방법: ① 병합 최종 상태(HEAD)의 overlay 소스를 diff 정독 ② mhc 커널은 스톡 preimage
두 커널(`mhc_fused_tilelang` + `mhc_pre_big_fuse_with_norm_tilelang`)과 줄 단위 대조
③ `python3 tests/test_logic.py` 전체 green 확인 ④ `launchers/compose-overlays.sh glm53`
재생성 산출물과 커밋된 `build/glm53/` 스냅샷 대조.

## 판정 요약

| # | 표면 (PR) | 판정 | 조치 |
|---|---|---|---|
| 1 | `mhc_onepass_tilelang` 커널 (#133) | **전사 충실** — phase1 FMA(`cm[k,j]` 순서)·sqr/warp 리덕션·mixes 핸드오프·sinkhorn `<32` 분기·norm 분기 전부 스톡 2커널과 일치. 디스패처 계약(n_out≤32, h%256, norm_weight 필수) 재검증됨 | GPU 검증 대기 (게이트 default off) |
| 2 | ONEPASS/SMALLM/BIGFUSE 게이트 (#133, #136, #137) | import 동결 + 호출별 계약 재검증, parse 무효값 = 스톡 | — |
| 3 | fp8_lm_head UE8M0 호스트 게이트 + one-shot 폴백 (#129–#132, #135) | 정상 — bitmask/값 계약, 시도 래치, weight 마지막 공개 순서 확인 | — |
| 4 | b12x EP 고정 쌍 계획·용량 가드·스크래치 청킹 (#120–#127, #130) | 정상 — stable argsort·사이클 반복·가중치 0, `bincount` minlength가 더미 포함, capture 안전 | — |
| 5 | bproj fp8 암 (#128) | 정상 — 게이트 기본 off, min(shape)≥512 가드 유지 | — |
| 6 | vocab orphan 마스크 이관 (#130, #132) | 정상 — 캐시·오버라이드·샤드 경계 매핑 확인 | — |
| 7 | `build/<profile>/` 스냅샷 ↔ module 소스 | **이격 6파일** — GLM53 3개(fp8_lm_head/tilelang 2개), 후속 parity 검사에서 DSV4 3개(oneshot CU/shim, moe gate)도 옛 코드로 남은 것을 확인 | 두 profile 재생성 + 현재 inventory 37파일 byte parity 회귀 게이트 |
| 8 | `probes/mhc_glm53_bench.py --onepass` 출력 비교 | **첫 실행 크래시** — comb_mix (m,HC,HC) vs (m,HC·HC) 비교 불가, post는 (m,HC,1) 브로드캐스트로 norm 2배 과대 | `reshape_as`로 stock buffer shape에 정렬 + CPU 회귀 테스트 |
| 9 | GPU 프로브 실행 경로 | **거짓 판정 가능** — 기존 Docker 명령은 repo만 `/repo`에 mount해 이미지 stock MHC를 import했고, kernels import가 dispatcher를 먼저 eager-import하므로 함수 안의 ONEPASS env 설정도 늦었음 | 합성 소스 target bind wrapper + import SHA-256 검증 + import 전 게이트 설정 |
| 10 | 일반 sweep/stock 기준 | **첫 다른 n_splits에서 shape 크래시** — config별 gemm 중간 버퍼를 비교했고, M≥8도 실제 stock `(3,4)` 대신 `(2,8)`을 기준으로 사용 | 최종 출력 4개만 비교 + M별 stock config 사용 |
| 11 | `tests/test_logic.py` | 1156 checks green — UE8M0 섹션은 torch 없는 호스트에서 소리내어 스킵(#125 설계대로) | snapshot/probe 계약 회귀 게이트 추가 |

## 수정 내역 (이 PR)

### 1. `probes/mhc_glm53_bench.py::onepass_check` — 판정 전 크래시

`rel_err` zip 비교가 op 반환값을 그대로 대조했다:
`comb_mix_cur`는 (m, HC, HC) 뷰인데 스톡 참조는 (m, HC·HC) — 4 vs 16은
브로드캐스트 불가라 첫 비교(m=1)에서 RuntimeError. GPU 창의 첫 `--onepass`
실행이 판정을 남기지 못하고 죽는 경로였다. `post_mix_cur` (m, HC, 1) vs (m, HC)는
(m, HC, HC)로 확장돼 norm이 √HC=2배 과대계상되어 1e-4 임계값 판정이 왜곡됐다.
수정: 각 출력을 참조와 같은 shape로 reshape 후 비교.

### 2. `build/<profile>/` 스냅샷 재동기화

`compose-overlays.sh glm53`를 HEAD overlay 소스로 재생성해 산출물과 커밋된
스냅샷을 대조 — 정확히 3파일이 이격되어 있었고 나머지(manifest 포함)는 일치.
- `fp8_lm_head.py`(97줄): #135 이전 동작이 스냅샷에 남아 있었음 — ImportError 시
  `use_e8m0=True` 무검증 폴백(장치측 trap 경로 재진입 가능) + one-shot 래치 없음.
- `tilelang.py`(18줄)/`tilelang_kernels.py`(6줄): #137(게이트 동결·fragment 정렬) 미반영.
실배포는 deploy 단계에서 재생성하므로 영향 없음 — 저장소 스냅샷이 배포 산출물과
다른 코드를 보여주는 정합성 문제.

후속으로 #139–#141을 최신 main에서 병합한 뒤 다시 compose했고, 새 BIGFUSE/V2/EP
변경과 이 PR의 ONEPASS/FP8 가드를 모두 보존한 상태로 16개 GLM53 build 파일이 각
module source와 byte-identical임을 확인했다. 같은 검사를 DSV4에 적용하자
`dsv4_oneshot_ar.cu`, `dsv4_oneshot_shim.py`, `moe_gate_sm121.py`도 stale임이 드러나
재합성했다. `tests/test_logic.py`가 두 profile 현재 37파일의 manifest/inventory/bytes를
매번 검사하므로 오래된 브랜치의 build 재생성이 다시 섞이면 배포 전 실패한다.

### 3. GPU 프로브가 실제 overlay를 검증하도록 fail-closed

기존 런북은 repo를 `/repo`로만 mount했지만 probe의 import는 이미지
`site-packages/vllm/...`를 읽었다. 더구나 `mhc.tilelang_kernels` import는 parent
`mhc/__init__.py`를 거쳐 dispatcher를 eager-import하므로 `onepass_check()` 안에서
env를 켜도 import 동결값은 이미 off였다. 결과적으로 `--onepass`가 stock 2-launch를
stock pair와 비교해 깨끗하다고 말할 수 있었다.

`probes/run_mhc_glm53_bench.sh`가 compose 후 두 MHC 파일을 manifest의 실제 container
target에 bind한다. Python probe는 인자 파싱 직후, 어떤 vLLM import보다 먼저 ONEPASS를
켜고, import된 두 파일의 SHA-256이 `/repo/build/glm53`와 같은지 및 dispatcher의
`_DENEB_ONEPASS is True`/onepass 커널 존재를 확인한다. 직접 Docker 실행이나 stock
source는 판정 전에 중단한다. 일반 sweep도 n_splits 크기가 다른 gemm 중간 버퍼 대신
residual/post/comb/layer 최종 출력만 비교하고 M≥8의 stock `(3,4)`를 사용한다.

## 잔여 (수정하지 않음)

- **onepass GPU 검증 미실행**: 게이트 default off. GPU 창에서
  `probes/run_mhc_glm53_bench.sh --onepass` 실행이 채택 전제다.
- `test_mhc_onepass_math`의 post 경로는 bf16 반올림을 시뮬레이션하지만 실제 커널은
  fp32(스톡 big_fuse 전사). post_mult=2.0(2의 거듭제곱)이라 결과 일치 — 기록만.
- `fp8_lm_head._repair_ue8m0_scales`: #131 이후 호출처 없음(dead). 제거 보류.
- vocab 마스크의 `shard_indices.org_vocab_start_index` 의존: 이미지 vLLM 버전에서
  attr 실존 확인 필요. 실패 시 무해 방향(valid_end=local_width → 마스크 없음).
