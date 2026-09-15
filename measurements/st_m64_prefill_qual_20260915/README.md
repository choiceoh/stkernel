# M64 프리필 MoE 자격검사 — 네 타일까지 정확하고, 다섯 번째에서 깨진다 (2026-09-15)

> 그날의 조사 — **2026-09-15 의 조사다.** 그날 참이었던 것이고 유지되지 않는다 — 이후 무엇이 바뀌었는지는 `MEASUREMENTS.md` 가 안다.

**판정: M64 후보는 자격을 얻지 못했다.** 257 행부터 **같은 입력에 대해 자기 출력과 어긋난다.**
M128 대조군은 모든 폭에서 퍼짐 0.0 이다.

#876 이 M64 프리필 후보를 CPU 오라클까지 붙여 넣은 뒤 **장치에서 한 번도 돌지 않았다.** 이것이 그 첫 실행이다.

## 어디서 돌았나

srv4 의 GB10(sm_121a, compute 12.1), **프로덕션 `st-glm53` 옆**, 큐의 단일 GPU 레인
(`fleet.sh run --gpu`, holder-single, 8 GiB 예산, 플릿 리스 없음). 네 판: `st-m64-qual0915`(배선 실패),
`…b`(같은 이유), `…c`(레인 첫 진입), `…d`·`bisect`·`final`.

- 이미지 `st-engine:bracket-b39f2bda8014`, torch 2.13.0+cu132, CUDA 13.2
- 랭크 `~/models/st-glm53-9391-up-gate-full/rank3of4.safetensors`, `L3.moe.*`, folded(Red Hat) 스케일
  - `w13` `f75862ef…`, `w13_sf` `729cb186…`, `w2` `6acdfea4…`, `w2_sf` `c5b07a16…`
- 셀: E=288 local, hidden 4096, inter_local 512, topk 8, nvfp4, swigluoai_uninterleave, limit 10

## 판정

두 팔은 서빙 진입점 인자 **하나**로만 갈린다(`launch_sm120_moe(_prefill_tile64=)`). 프로브가
`workspace.tile_m` 을 확인하므로 레인에 닿지 않고 통과할 수는 없다.

| 행 | M128 자기 퍼짐 | **M64 자기 퍼짐** | 팔 사이 차이 | |
|---:|---:|---:|---:|---|
| 129 | 0.0 | **0.0** | **0.0** | M128 과 비트 동일 |
| 193 | 0.0 | **0.0** | 0.0 | |
| 201·209·217·225·233·241·249 | 0.0 | **0.0** | | |
| **256** | 0.0 | **0.0** | **0.0** | 마지막으로 정확한 폭 |
| **257** | 0.0 | **1.10 ~ 1.34** | 2.39 | **깨진다** |
| 385 | 0.0 | 1.128 | 2.06 | |
| 513 | 0.0 | 1.153 | 2.41 | |
| 1024 | 0.0 | 1.139 | 1.84 | |
| 2304 | 0.0 | 1.187 | 1.85 | |

**경계는 256 → 257 이고, 256 = 4 × 64 다.** M64 타일 넉 장까지는 정확하고 결정적이며, 다섯 장째가
필요해지는 순간 결과가 실행마다 달라진다. 퍼짐 1.1~1.3 은 반올림이 아니다 — 경쟁이다.
`_prefill_m64_bodies` 의 "16-row-per-warp scatter strips"(타일당 넉 줄)와 #876 의
"give M64 tasks independent activation-scale atoms" 가 같은 자리를 가리킨다.

129 행에서 두 팔 모두 토치 참조 레인과 **0.0885** 떨어져 있다 — 같은 거리다. NVFP4 대 참조의 거리이지
팔 사이의 차이가 아니다.

## 속도는 주장하지 않는다

프로덕션 옆에서 잰 eager B/A/A/B 라 대조군의 브래킷 드리프트가 **0.04 ~ 0.73** 이다. 2304 행의
`ratio 0.81` 같은 값은 그 잡음 안에 있고, 게다가 그 폭의 M64 출력은 틀렸다. **−20% 는 이 기록으로
확인되지도 반박되지도 않았다.** 경쟁을 먼저 고쳐야 물어볼 수 있는 질문이다.

## 게이트가 자기 자신에 대해 잡은 것

첫 판정은 `passed: true` 였다. 규칙이 `across <= floor × factor` 였고 floor 가 후보 자신의 퍼짐이라,
**망가진 팔이 자기가 재어질 기준을 스스로 올렸다.** 자기 일치를 먼저 본다:

    reproducible = candidate_spread <= max(control_spread × factor, 1e-3)

FP32 원자합 재정렬이 BF16 출력에 남기는 것은 그 천장 한참 아래다. `tests/test_engine_prefill_m64_probe.py`
가 실측 m=1024 행을 그대로 넣어 옛 규칙은 통과시키고 새 규칙은 거절하는 것을 박는다.

## 실행이 잡은 배선 결함 셋 (전부 첫 실행 전에는 안 보였다)

1. **큐가 프로브를 접수하지 않았다** — `probes/engine_moe_prefill_m64.py is not a canonical ST check`.
   `bench/fleet_onepass.ST_PROBES` 는 허용 목록이다.
2. **`cell` 이 dict 였다** — `eligibility()` 가 리포트용 dict 를 돌려줘 `cell.topk` 가 죽었다.
3. **플래그가 지워지고 있었다** — `b12x_fused_moe` 가 `_prefill_tile64=None` 을 명시적으로 넘기는데
   호출 시점 키워드가 `functools.partial` 을 덮는다. 프로브의 `True` 가 `None` 이 되어 **두 팔이 모두
   M128 로 돌았고**, 게이트가 `reached (128, 128)` 로 잡았다. m=65 에서 예외가 안 난 것이 단서였다.

## 65 행은 이 레인의 창이 아니다

자격 창은 `64 < m` 이지만 디스패처는 routed-pair 컷오버를 넘어야 dynamic 백엔드에 닿는다. m=65 는
static 계열이 서빙한다(`[b12x static v2] lane serving: static2_m65…`). 프로브가 이제 행마다 백엔드를
물어 `skipped_static_rows` 에 적는다 — 통과로 읽히지 않게.

## 다음

M64 는 **넉 장 초과에서 경쟁하는 것을 고치기 전에는** 프로덕션 프리필 청크(2304·4608·6912)에 쓸 수 없다.
고친 뒤 이 프로브를 그대로 다시 돌리면 같은 사다리가 판정한다. 기본값 전환은 그 다음이고, tok/s 는
D17 대로 플릿 onepass 몫이다.

## 재현

```sh
git -C ~/stkernel worktree add --detach ~/st-worktrees/m64-90d0e567 90d0e567
cd ~/st-worktrees/m64-90d0e567
FLEET_SINGLE_GPU_HOST=srv4 ST_PROBE_TREE=st-m64-90d0e567 ST_IMAGE=st-engine:bracket-b39f2bda8014 \
  bash bench/fleet.sh run --gpu st-m64 25 'M64 prefill MoE' -- \
  bash probes/run_engine_probe.sh probes/engine_moe_prefill_m64.py \
    --ranks ~/models/st-glm53-9391-up-gate-full/rank3of4.safetensors \
    --rows 129,256,257,2304 --output /cache/st-m64.json
```

장치 없이 배선만:

```sh
docker run --rm -e CUDA_VISIBLE_DEVICES= -e CUTE_DSL_ARCH=sm_121a -e PYTHONPATH=/repo \
  -v "$PWD":/repo:ro -w /repo --entrypoint python3 st-engine:bracket-b39f2bda8014 \
  probes/engine_moe_prefill_m64.py --cpu
```

행별 원시 JSON 은 `rows.jsonl`. CPU 검사 `tests/test_engine_prefill_m64_probe.py` 15 개.

---

# 추적 2차 — 어디가 아닌지 (2026-09-15, 같은 레인 3판 추가)

경계가 256 → 257 이라는 것만으로는 원인이 안 나온다. **가설 셋을 세워 셋 다 실측으로 지웠다.**

## 1. 타일 넉 장이 아니다

256 = 4 × 64 라 "M64 타일 넉 장짜리 자원"을 의심했다. 틀렸다 — 타일은 전문가마다 선다.
부팅 로그의 `tiles` 는 m=129 에서 이미 304 이고 m=2304 에서 575 다. 넉 장인 적이 없다.

## 2. split-task 창이 아니다

`_prefill_m64_bodies.py` 에 정확히 그 경계가 있다. 세 갈래다:

    num_tokens <= 256     -> publish_uniform_deferred_tasks
    256 < num_tokens <= 4096 -> publish_variable_deferred_tasks   (split_tile_count)
    num_tokens > 4096     -> publish_uniform_deferred_tasks

깨진 폭이 전부 가운데 창 안이었으니 그럴듯했다. **예측: 4096 초과는 멀쩡할 것.** 재 보니
m=4097·4608·6912·8192 **전부 깨진다**(퍼짐 1.05~1.25). 창이 아니다.

| 행 | M64 자기 퍼짐 |
|---:|---:|
| 4096 (창 안) | 1.046 |
| 4097 (창 밖) | 1.250 |
| 4608 · 6912 · 8192 (창 밖) | 1.092 · 1.048 · 1.077 |

## 3. 워크스페이스가 아니다

`_prefill_m64_workspace` 의 용량은 `1 << (num_tokens-1).bit_length()` 라 256 → 257 에서
**정확히 두 배가 된다**(256 → 512). 그럴듯했다. 사다리를 **내림차순**으로 돌려 갈랐다:

    --rows 2304,256,129,256

m=2304 가 먼저 깨지고 큰 워크스페이스가 자리잡은 뒤에도 **256 과 129 는 퍼짐 0.0, 팔 사이
불일치 0 행**이다. 같은 프로세스, 같은(이미 커진) 워크스페이스. 용량이 아니라 `num_tokens` 다.

## 어느 행이 틀리나 — 꼬리가 아니라 전부다

프로브에 행 단위 진단을 더했다(두 M64 반복 사이, 그리고 두 팔 사이).

| 행 | 틀린 행 수 | 첫 행 | 마지막 행 |
|---:|---:|---:|---:|
| 256 | **0** | — | — |
| 257 | 257 / 257 | 0 | 256 |
| 320 | 320 / 320 | 0 | 319 |
| 513 | 513 / 513 | 0 | 512 |
| 2304 | 2304 / 2304 | 0 | 2303 |

**0 행부터 마지막까지 전부**다. 용량 끝을 넘어가는 주소 오류라면 꼬리만 틀려야 한다.
런치 전체가 오염된다.

## 남긴 관찰

`producer_batch_tokens = tile_m × tile_n / hidden` 이라 **M128 은 4, M64 는 2** 다
(`_prefill_m64_bodies.py:1304`). Q0 스테이징 복사는 여전히 4 에서 쪼개고 둘째 조각을
`+4 × cols × 2` 에 쓴다 — 2 에서는 둘째 조각이 0 바이트라 무해하지만, 그 비대칭은 여기 적어 둔다.
M64 는 CTA 당 한 번에 토큰 둘만 집으므로 프로듀서 반복이 M128 의 두 배다.

## 다음에 볼 자리

전 행 오염 + `num_tokens` 의존 + 워크스페이스 무관 → 런치 전체가 공유하는 무언가가
토큰 수에 따라 어긋난다. 후보는 그리드 폭/잔여 CTA 의 일 분배(`resident_grid_barrier`,
`pair_head` 클레임 루프)와 커널이 직접 0 으로 채우는 scatter 출력(`scatter_vecs` 루프)이
그 뒤의 원자합과 제대로 순서 지어지는지다. 여기서부터는 커널을 계측해야 하고, 이 조사는
컴파일 없이 갈 수 있는 데까지 왔다.
