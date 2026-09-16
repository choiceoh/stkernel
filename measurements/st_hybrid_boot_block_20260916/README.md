# 하이브리드 팔은 지금 부팅되지 않는다 — m=16 동반 레인이 직접 스캐터를 sf6 없이 요구받는다

2026-09-16. #911 하이브리드의 head NLL 을 강한 채널로 다시 재려고 격리 부팅을 잡았다.
팔 A(프로덕션 랭크 + 오늘 재교정한 블롭)는 떴고 133 문서를 받았다. 팔 B(하이브리드 랭크)는
**문을 열지 못했다.** 여기 적는 것은 그 실패 하나이고, 그 이상은 적지 않는다.

## 잰 것

두 팔 모두 같은 sha `a0e9221cb` (= `origin/main` + `EXPERT_CAPTURE = True`), 같은 레인
`t,r,sf6,batch,q0` (`lanes.MOE_STATIC_PRODUCTION`), 같은 플릿, 같은 창.

| 팔 | 랭크 | 결과 |
|---|---|---|
| A | `st-glm53-9391-up-gate-full` (`st-glm53-b12x-up-gate-v1`) | 15:20:21 문 열림, 133 문서 캡처 |
| B | `st-glm53-hybrid-gptq-v1` (`st-glm53-modelopt-up-gate-bf16-dense-v1`) | 15:48 부팅 중 죽음 |

팔 B 의 죽음 (`rank0_boot_death.log`):

```
[serve] death rank=0 kind=local phase='boot': ValueError: private scatter requires packed FP32 output without split work
  engine/profiles/glm53/net.py:272 warmup_decode_experts
  engine/kernels/b12x/moe_dispatch.py:2461 _get_static_kernel_v2
  engine/kernels/b12x/moe_static_kernel_v4.py:159 __init__
```

## 어디서 갈라지는가

디코드 그래프 캡처는 행마다 레인을 **두 개씩** 세운다 — sf6 를 쓰는 것과 쓰지 않는 동반 레인
(`q0`, `lanes.parse_moe_static` 의 "TP SF6 Q0 flag"). 로그의 순서가 그대로 보여준다:

```
m=1   ...sf6v1fc1sepwordfc2worda2u64a1reusecompactsfregssyncc256     OK
m=1   ...a2u64c256                                                   OK   <- 동반
m=8   ...sf6v1...                                                    OK
m=8   ...a2u64c256                                                   OK   <- 동반
m=64  ...sf6v1c256                                                   OK
m=64  ...c256                                                        OK   <- 동반
m=16  ...sf6v1...c2scatterreuseprefetch3syncc256                     OK
                                                                     죽음 <- m=16 의 동반
```

m=16 만 다른 이유는 `batch` 셀이다. `moe_dispatch._static_v2_config_for` 에서

```python
direct_scatter = bool(reform and m == 16 and config.get("batch_reform")
                      and config.get("c2_direct_scatter", True))
```

이라 **m=16 에서만** 직접 레지스터 스캐터가 켜진다. 그런데 커널 생성자는

```python
if (self.route_scatter or self.direct_scatter) and not (
        scatter_fp32 and reform_sf_pack and not split):
    raise ValueError("private scatter requires packed FP32 output without split work")
```

라서, sf6 없는 동반 레인(= `reform_sf_pack=False`)이 m=16 에서 만들어지는 순간 거부한다.
m=1/8/64 의 동반 레인은 `direct_scatter` 가 꺼져 있어 이 검사를 지나간다.

## 아직 안 가린 것 (읽는 사람이 속지 않도록)

**팔 B 는 팔 A 와 두 가지가 다르다.** 하이브리드 랭크인 것, 그리고 **교정 부팅**인 것:

```
calibration: rank 0 not filed -- collecting 0/131072 rows over 176 blobs, 58 tiles deferred
memory gate: ... differs at config.lane_info.calibration, ...
```

#1045 가드가 하이브리드(`modelopt-up-gate-bf16-dense-v1`)에게 프로덕션 블롭
(`b12x-up-gate-v1` 도장)을 미교정으로 읽히므로, 하이브리드는 **반드시** 교정 부팅으로 시작한다.
그래서 이 관측 하나로는 원인이 하이브리드 기하인지 교정 모드인지 가릴 수 없다.
`calibration` 은 레인 아이덴티티의 일부이고(위 memory gate 줄), 이 sha 에서 프로덕션 랭크로
교정 부팅을 한 기록은 없다 — 오늘의 재교정은 #1053/#1054 이전 빌드였다.

가리는 실험은 한 부팅이다: **프로덕션 랭크 + 교정 부팅**을 이 sha 로 띄운다.
죽으면 원인은 교정 모드이고, 뜨면 하이브리드 기하다. 창을 하나 더 써야 해서 오늘은 안 했다.

## 결과

- 팔 A 의 `head.jsonl` 은 남겼다: `~/glm53-logs/headnll2/armA.head.jsonl` (133 문서).
- 팔 B 는 없다. 따라서 #911 의 하이브리드 판정(2026-09-14, "속도는 올랐고 점수는 떨어졌다")은
  **오늘 갱신되지 않았다.** 재교정한 블롭 위에서 다시 재려면 위 결함을 먼저 넘어야 한다.
- 프로덕션 블롭은 팔 B 를 위해 네 랭크 모두 `mkcalib/rank$R.prod-0916` 으로 치웠다가
  되돌렸다 (476개씩, 오늘 재교정본). 팔 B 는 교정까지 못 갔으므로 아무것도 덮지 않았다.

## 남은 것

`mk_use_compact_m8` 때(#1054)와 같은 모양이다 — 두 계약이 한 셀에서 어긋난다. 고치려면
동반 레인에서 `direct_scatter` 를 끄거나(`reform_sf_pack` 이 없으면 m=16 이어도 켜지 않는다),
동반 레인을 m=16 에서 세우지 않으면 된다. 어느 쪽이든 **측정된 셀**
(`st_c2_dense_cells_20260915` 가 재놓은 m=16 직접 스캐터)을 건드리지 않는지 부팅으로 보여야 한다.
