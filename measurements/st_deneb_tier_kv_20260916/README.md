# 접두사 캐시 적중이 낮은 이유는 용량이 아니었다 — 둘 곳이 없었다 (2026-09-16, srv2 프로덕션)

운영자: "데네브 캐시 적중률이 너무 낮은데."

## 1. 실측 — 기준선 (티어 off, `ST_KV_GIB=7.0`)

라이브 프로덕션의 카운터:

| | 값 |
|---|---|
| `prefix_cache_queries_total` | 100 |
| `prefix_cache_hits_total` | **17** |
| `prefix_cache_evictions_total` | **1,144** |
| `prefix_tier_entries` | 0 |
| `gpu_cache_usage_perc` | **3.0 %** |
| 그때까지 처리한 요청 | 91 |

**용량이 아니다.** 블록의 97 % 가 비어 있는데 축출이 1,144 번 일어났다. 요청당
1,144 ÷ 91 ≈ **12.6 개의 경계**가 버려졌다.

## 2. 원인 — 티어가 없으면 끝난 턴은 대화로 등록되지 않는다

`engine/base/serve.py:2711` 의 유휴 처리:

```python
if self.runner.tiered is None:
    self._idle_order[row] = None
    return
```

티어가 없으면 그 행은 유휴 목록에 들어가지 않고, 회수될 때 그 대화의 블록 경계가 같이
사라진다. 계산은 다 해 놓고 **둘 곳이 없어서** 버리는 것이다. 티어는 2026-09-15 에
랭크 간 티어 발산으로 꺼졌고(그날의 원장 항목), 그 발산의 원인인 key 단위 화해
(`Server._reconcile_parked`, #837)는 그 뒤 main 에 들어와 있다.

**주의: 적중은 768 토큰 블록 경계에서만 난다.** 한 블록보다 짧은 공유 접두사는 티어가
있든 없든 적중하지 않는다. 이걸 모르고 잰 첫 시험은 0 적중이 나왔고, 그건 시험이 틀린
것이었다.

## 3. 조치 1 — 티어를 켜고 KV 를 두 배로 (11:04 부팅)

`~/.config/st-glm53.env`: `ST_TIER_DIR` 를 주고 `ST_KV_GIB` 7.0 → **14.0**
(백업 `~/.config/st-glm53.env.bak-20260916-tier`).

| | 전 | 후 |
|---|---|---|
| `kv_blocks_total` | 1,398 | **2,987** |
| declared paged KV | 6.16 GiB | **13.16 GiB** |
| unassigned | +27.10 GiB | +19.59 GiB |
| prefix 스냅샷 예산 | 4.25 GiB (untiered) | 2.12 GiB (tiered) |

부팅은 한 번에 붙었다 — `launch: door up after 135s`, `healthy after 135s (a chat
answered)`. 티어도 살아서 보고했다: `0 conversations parked from before, 0.0 GiB of
64 GiB` / `0 prefix boundaries parked from before, 0.0 GiB of 16 GiB`.

재사용도 확인했다. 약 3,000 토큰(= 블록 여러 개)짜리 공유 접두사를 두 번 보내
`prefix_cache_hits_total` 0 → **1**, `evictions` 0.

## 4. 그런데 그 티어는 컨테이너 안에 있었다

`ST_TIER_DIR=/home/choiceoh/st-tier` 로 줬는데, 랭크 컨테이너가 바인드하는 호스트
디렉터리는 **`/home/choiceoh/glm53-logs` 하나뿐이다**(`launchers/start-st-glm53.sh` 의
`docker run`). 그 밖의 경로는 컨테이너의 writable layer 에 생긴다.

증거 (`raw/srv2-state.txt`, 11:14):

```
=== ~/st-tier (off the mount) ===
absent on the host -- it only ever existed inside the container
```

네 노드 어디에도 `~/st-tier` 가 없다. 진짜 티어 루트는 예전부터
`~/glm53-logs/st-tier` 이고(09-15 이전 런처 기본값도 그것이었다), 지금 네 노드 모두
**비어 있다**(4.0K). 발산했던 사본은 옆에 `st-tier.diverged-0915`(769M)로 남아 있다.

즉 3 절의 부팅은 **부팅도 성공했고, 티어도 살아 있다고 보고했고, 재사용도 됐다** —
그리고 컨테이너와 함께 전부 버려진다. 실패가 아니라 조용한 무효다. 플릿 리스도
`~/st-fleet.lock` 에서 같은 것을 한 번 겪었다(런처 주석).

## 5. 조치 2 — 리포에 고정한다 (이 PR)

운영자: "고정해야지."

프로덕션의 모양이 **상자 위에만** 있으면 살아남지 못한다. 오늘 하루에 세 번 그랬다 —
09-15 에 손으로 고친 배포 트리의 런처, 6 절의 낡은 env 를 든 배포 사이클, 그리고 4 절의
잘못된 경로. 배포는 트리에서 다시 올리고, 트리는 이 리포다.

- **티어.** `launchers/start-st-glm53.sh` 기본값 `off` → **`$MOUNTED_ROOT/st-tier`**.
  `off` 는 이제 끄는 말이지 기본값이 아니다.
- **마운트 가드.** `MOUNTED_ROOT=/home/choiceoh/glm53-logs` 를 두고, `off` 도 아니고 그
  아래도 아닌 `ST_TIER_DIR` 는 **거부한다**(exit 2). 4 절을 두 번 겪지 않기 위해서다.
- **KV.** `KV_GIB=${ST_KV_GIB:-14.0}`. 전에는 env 에 값이 없으면 `--kv-gib` 를 아예 안
  넘겨 boot.py 의 24.0(vLLM 동수 비교용 단일 상자 기본값)으로 갔다. 프로덕션이 쓰던
  7.0 은 **원장 항목 없이 env 에 손으로 박혀 있던 값**이다. 14.0 은 3 절에서 실제로 잰
  값이다. 숫자가 아닌 `ST_KV_GIB` 는 이제 거부한다(전에는 빈 값과 구분이 없었다).
- `bench/st_bracket.sh` 의 팔별 티어는 `$LOGD/st-bracket-tier/...` 로 이미 그 아래다.

Git Bash 로 직접 확인한 경우들:

| `ST_TIER_DIR` | | `ST_KV_GIB` | |
|---|---|---|---|
| (없음) | `--tier-dir $LOGD/st-tier` | (없음/빈 값) | `--kv-gib 14.0` |
| `off` | `--tier-dir=` | `7.0` / `24` | 그대로 통과 |
| `$LOGD/st-bracket-tier/x-base` | 통과 | `lots` / `-3` | exit 2 |
| `~/st-tier`, `$LOGD` 자신, `/tmp/tier`, `off2`, `relative/tier` | exit 2 | | |

## 6. 지금 프로덕션의 상태 — 다시 티어 off 다

11:09:34 에 `st-deploy-watch` 가 4347fe9d 를 배포하면서 플릿을 다시 올렸고, 그 사이클은
**env 를 고치기 전에 시작**했으므로 배포 트리의 런처 기본값으로 갔다:

```
--kv-gib 7.0 ... --tier-dir=
```

그래서 11:09:52 부터 프로덕션은 **KV 7.0 · 티어 off** 다. 3 절의 상태는 약 3 분
살았다.

env 파일의 두 줄은 이 PR 이 배포되면 **없어도 되는 값**이다. 그리고 `ST_TIER_DIR` 는
아직 4 절의 잘못된 경로라 **그대로 두면 새 가드에 걸려 런치가 거부된다.** 지우는 것이
맞다:

```bash
cp ~/.config/st-glm53.env ~/.config/st-glm53.env.bak-20260916-pin   && sed -i '/^ST_TIER_DIR=/d; /^ST_KV_GIB=/d' ~/.config/st-glm53.env
```

지우는 순간부터, 배포된 트리가 아직 옛 런처라면 티어 off · KV 24.0 이고(옛 런처는
`--kv-gib` 를 안 넘긴다), 이 PR 이 배포된 뒤라면 티어 on · KV 14.0 이다. 그러니 **이
PR 이 배포된 뒤에** 지우는 편이 낫다.

## 7. 미측정

- **진짜 데네브 트래픽에서의 적중률.** 티어를 켠 상태로 100 질의를 받아 본 적이 없다.
  3 절의 확인은 내가 만든 공유 접두사 한 쌍이고, 1 절의 기준선은 티어가 꺼진 상태다.
  기준선과 비교하려면 티어를 켠 부팅이 살아서 트래픽을 받아야 한다.
- **KV 두 배의 효과.** 축출의 원인이 용량이 아니었으므로(97 % 여유) KV 증설 자체가
  적중률을 올린다는 근거는 없다. 티어가 켜져 경계가 쌓이기 시작한 뒤에야 의미가 있다.
