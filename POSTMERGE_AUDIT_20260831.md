# 사후 리뷰 — 2026-08-31 병합 배치 (#120–#137)

Date: 2026-08-31 (main 기준 커밋 `b549c46`, #137)
Scope: `f63c084`(#119) 이후 오늘 main에 병합된 18개 PR — b12x EP 축(#120–#127, #130),
fp8 lm-head 축(#129–#132, #135), 캠페인 II 접수(#128), mhc tilelang 축(#133, #136, #137),
docs(#134). 커널/디스패처/호스트 가드 전부, docs와 런북은 정합성만.

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
| 7 | `build/glm53/` 스냅샷 ↔ overlay 소스 | **이격 3파일** — #135/#137이 compose를 다시 돌리지 않아 fp8_lm_head.py(97줄)·tilelang.py(18줄)·tilelang_kernels.py(6줄)가 스냅샷에 옛 코드로 남음 | 수정 (재생성 동기화) |
| 8 | `probes/mhc_glm53_bench.py --onepass` | **첫 실행 크래시** — comb_mix (m,HC,HC) vs (m,HC·HC) 비교 불가, post는 (m,HC,1) 브로드캐스트로 norm 2배 과대 | 수정 |
| 9 | `tests/test_logic.py` | 904 checks green — UE8M0 섹션은 torch 없는 호스트에서 소리내어 스킵(#125 설계대로) | — |

## 수정 내역 (이 PR)

### 1. `probes/mhc_glm53_bench.py::onepass_check` — 판정 전 크래시

`rel_err` zip 비교가 op 반환값을 그대로 대조했다:
`comb_mix_cur`는 (m, HC, HC) 뷰인데 스톡 참조는 (m, HC·HC) — 4 vs 16은
브로드캐스트 불가라 첫 비교(m=1)에서 RuntimeError. GPU 창의 첫 `--onepass`
실행이 판정을 남기지 못하고 죽는 경로였다. `post_mix_cur` (m, HC, 1) vs (m, HC)는
(m, HC, HC)로 확장돼 norm이 √HC=2배 과대계상되어 1e-4 임계값 판정이 왜곡됐다.
수정: 각 출력을 참조와 같은 shape로 reshape 후 비교.

### 2. `build/glm53/` 스냅샷 재동기화

`compose-overlays.sh glm53`를 HEAD overlay 소스로 재생성해 산출물과 커밋된
스냅샷을 대조 — 정확히 3파일이 이격되어 있었고 나머지(manifest 포함)는 일치.
- `fp8_lm_head.py`(97줄): #135 이전 동작이 스냅샷에 남아 있었음 — ImportError 시
  `use_e8m0=True` 무검증 폴백(장치측 trap 경로 재진입 가능) + one-shot 래치 없음.
- `tilelang.py`(18줄)/`tilelang_kernels.py`(6줄): #137(게이트 동결·fragment 정렬) 미반영.
실배포는 deploy 단계에서 재생성하므로 영향 없음 — 저장소 스냅샷이 배포 산출물과
다른 코드를 보여주는 정합성 문제.

## 잔여 (수정하지 않음)

- **onepass GPU 검증 미실행**: 게이트 default off. GPU 창에서 `probes --onepass`
  실행이 채택 전제 — 이번 수정으로 첫 실행이 완주 가능해짐.
- **probe가 m≥8에서도 기준 config를 (2,8)로 고정**: m=8/16의 타이밍 baseline은
  디스패처 실제 (3,4)와 다름. 수학 동등성 판정엔 무해하고 채택 판정은 bracket이
  결정하므로 유지.
- `test_mhc_onepass_math`의 post 경로는 bf16 반올림을 시뮬레이션하지만 실제 커널은
  fp32(스톡 big_fuse 전사). post_mult=2.0(2의 거듭제곱)이라 결과 일치 — 기록만.
- `fp8_lm_head._repair_ue8m0_scales`: #131 이후 호출처 없음(dead). 제거 보류.
- vocab 마스크의 `shard_indices.org_vocab_start_index` 의존: 이미지 vLLM 버전에서
  attr 실존 확인 필요. 실패 시 무해 방향(valid_end=local_width → 마스크 없음).
