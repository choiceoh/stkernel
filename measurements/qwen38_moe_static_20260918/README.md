# Qwen3.8 MoE 디코드 — 정적 커널의 모양 판정과 16행 패딩 (2026-09-18, srv4 단일 GPU, 세션 창 18:14~18:24)

> 그날의 조사 — 운영자 "8행으로 가자". K=3 네 행 부팅(PR #1182)이 12 토큰 정적 MoE 커널의 첫 런치에서 죽어, 8행 부팅이 밟는 정적
> 모양 전부를 격리해서 판정한 기록. 코드는 PR #1192(프로브 확장, 레인 승인, `fleet_prepare` 수정, `lanes.static_pad`).

## 왜 단일 GPU 레인이 아니라 세션 창인가

`bash bench/fleet.sh run --gpu --detach qwen38-moe-static …` 이 두 번 거절됐다.
1. `probes/engine_qwen38_moe.py` 가 `bench/fleet_onepass.ST_PROBES` 에 없었다("not a canonical ST check") — 네 Qwen 프로브의 docstring 은
   모두 이 레인을 적고 있는데 승인 목록에는 하나도 없었다. 승인했다.
2. 승인 뒤: `bench/fleet_prepare.py` 의 `main()` 이 `prepare(…, approve_deploy=…)` 를 넘기는데 #1152(06:12)가 `prepare()` 에서 그 인자를
   뺐다 — 오늘 아침부터 모든 `fleet.sh run` 이 `TypeError` 로 죽는다. 호출자를 고쳤다.
3. 남은 것: 공유 진입점 `~/glm53-logs/fleet.sh`(06:34)가 main 의 최근 여섯 판 어느 것과도 일치하지 않는다 — 공유 진입점 검사가 다음에 걸릴
   수 있다. 이 PR 이 손대지 않는다.

그래서 `fleet_lease.py yield --kind session` 으로 창을 받아 srv4 에서 `run_engine_probe.sh` 로컬 모드(`ST_PROBE_CHECKS_ONLY=1`)로 돌렸다.
프로덕션은 18:24 복귀.

## 판정 (합성 128 전문가, 서빙 라우터의 라우트, 오라클 = `engine/modules/moe.expert_gemm` 데이터플로, 문턱 2%)

첫 실행은 사다리 전체(micro 8→2, 정적 10→32 오름차순): micro 넷 통과, **정적 10 에서 illegal memory access**(캡처 워밍업의 첫 런치;
`torch.cuda.synchronize` 에서 드러남) — CUDA 컨텍스트가 죽어 나머지는 모양마다 한 프로세스씩(`ST_PROBE_DECODE_TOKENS=<m>`).

| 토큰(행 × K+1) | 커널 | 결과 | 오라클 상대오차 | 재생 |
|---|---|---|---|---|
| 2 / 4 / 6 / 8 | micro (zero-weight skip) | 통과 | 0.0064 / 0.0056 / 0.0063 / 0.0053 | 바이트 동일 |
| **10** | static_m10 | **illegal memory access** | — | — |
| **12** | static_m12 | **illegal memory access** (네 행 K=3 부팅과 같다; 거기서는 r160 용량) | — | — |
| 14 | static_m14 | 통과 | 0.0053 | 바이트 동일 |
| 16 | static_m16 | 통과 | 0.0053 | 바이트 동일 |
| 20 | static_m20 | 통과 | 0.0040 | 바이트 동일 |
| 24 | static_m24 | 통과 | 0.0040 | 바이트 동일 |
| 28 | static_m28 | 통과 | 0.0069 | 바이트 동일 |
| 32 | static_m32 | 통과 | 0.0069 | 바이트 동일 |

모양마다 한 프로세스 약 12~15 s(컨테이너·컴파일 1~3 s·검사). 원인은 커널 안(모양 10·12 에 특정된 codegen)이고 이 기록은 격리하지 않는다 —
14 는 통과하므로 "16 미만" 규칙도, 20·28 이 통과하므로 "8의 배수" 규칙도 아니다.

## 고친 것: `engine/profiles/qwen38/lanes.static_pad`

캡처 런치가 micro 상한(8) 위·정적 한 타일(16) 아래면 16행으로 채운다 — 0 행을 이 랭크의 전문가 0 에 가중치 0 으로(다른 랭크 라우트와 같은
약속). 8행 사다리에서 걸리는 모양은 K=3 의 12(3행)와 K=1 의 10·12·14 뿐이고, 채운 런치는 네 행 K=3 이 이미 지난 `static_m16` 이다.

| 토큰 | 채운 뒤 커널 | 결과 | 오라클 |
|---|---|---|---|
| 10 | static_m16 | 통과 | 0.0053 |
| 12 | static_m16 | 통과 | 0.0063 |
| 14 | static_m16 | 통과 | 0.0052 |

비용: 더미 행 2~6 개 × top-10 라우트가 전문가 0 에 가중치 0 으로 — 전문가 0 을 한 번 더 읽는 정도. 즉시(compact) 프리필 경로는 그대로.

## 곁들인 발견

- 창 3 부팅(17:35)과 프로덕션 복귀(17:48)가 공유 `/cache` 의 MoE 모듈을 서로 지웠다(#1187 이 모듈 이름을 키 파일 내용 해시로 바꿔 고쳤다).
  이 창의 컴파일은 그 새 이름(`st_b12x_moe_e96383066039_…`) 아래 처음이라 micro 커널도 다시 컴파일됐다(각 3~5 s).
- 원시 기록 `decode-checks-srv4.jsonl`(각 실행의 `device`·`decode_check`·`checks_only` 행과 fault 행).

## 안 한 것

정적 커널 안의 원인(m=10·12 codegen), 9·11·13·15 와 16 위 4의 배수 아닌 모양(8행 사다리에 없음), 8행 K=3 플릿 부팅과 C=1~8 짝 측정(다음
창), 정적 모양의 tile/MAC 스윕(레인 티켓; 레인은 위 3 이 남아 있다).
