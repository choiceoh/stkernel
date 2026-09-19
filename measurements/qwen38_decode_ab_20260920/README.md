# Qwen3.8 C1·C4 전체 서비스 비교 — 진행 중

사용자 목표: C=2~4 전체 처리량 우선, 품질 유지, C1 하락 최대 5%. 추가 지시: C1 자체도 크게 개선해야 한다. 커널 결과와 실제 출력 tok/s를 구분한다.

## 고정 팔

- A: `632d1c7ccfa439af853dd469486f802df2aa5bd4` — 두 후보 끔.
- B: `c861b72b8b36c1e40da106310bf2c5fbc6a4580c` — A 위에서 fleet CLI의 `hc-w8a16`, `moe-decode-chunks` 두 기본값만 True로 고정한 실험 전용 커밋. 이 커밋은 제품 브랜치에 합치지 않는다.
- 제품 브랜치 `codex/qwen38-concurrent-decode-0920`는 기본값을 꺼 둔다. [믹서 증거](../qwen38_mix_w8_20260920/README.md), [MoE 증거](../qwen38_moe_chunks_20260920/README.md).
- 컨트롤러 checkout: srv2 `/home/choiceoh/st-worktrees/qwen-c4be-moe-0920`. 런처는 각 팔의 고정 릴리스에서 가져온다.
- `ST_BRACKET_PROFILE=qwen38`, `PROFILE=qwen38`, `ST_PRODUCTION_ENV=/dev/null`, port 8001, KV 16GiB, max_seqs 4, K3, MTP dense/expert BF16, tuned head 없음, MTP 입력 기록 끔.
- `ST_SELF_CALIBRATE=0`, private `/home/choiceoh/qwen-c4be-cache-0920`. 두 팔 모두 빈 `mkcalib`에서 동일 RTN dense pack을 사용한다. 공유 `cu132` 컴파일 캐시만 연결했다. 기존 calibration/운영 캐시의 모델 분포를 변경하지 않는다. 현재 별도 GPTQ/튜닝 작업의 결과와 직접 비교하는 실험이 아니다.
- 기존 MTP 세션이 스스로 임대를 반납한 뒤 제출했다. 운영 서비스의 quiet gate 및 정규 플릿 큐를 이용한다.

## 실행 계획과 상태

먼저 A B A short screen(C1/C4, 2K)을 실행한다. screen은 품질 관찰과 실제 지연만 기록하며 채택 근거가 아니다. 후보가 유효하면 동일 팔의 `ST_BRACKET_VALIDATION=full ONEPASS_PROFILE=extended`로 두 onepass/boot, 2K/32K/128K 품질, C4, tokens/step, 실제 출력 tok/s와 출력 hash를 확인한다. C2는 별도 실제 요청 증거가 필요하다.

`qwen-decode-screen-0920a`는 준비 검사 직후 `fleet_prepare.py`의 삭제된 `validate_targets` 호출 때문에 **예약 전 실패**했다. 프로덕션을 멈추지 않았으며 측정값은 없다. ST signed preparation과 canonical consumer를 재검증하는 현행 경계로 복구하고 회귀 검사를 추가했다. 후속 결과는 이 기록에 덧붙인다.

`qwen-decode-screen-0920b`는 컨트롤러 `c04f76f7`에서 정상 접수됐다. ticket `1789833653179195`, revision 1. 01:02 KST 확인 당시 `q38gptq-330k-0920c`(약 150분 창)를 기다리는 큐 3번이었다. 다른 예약은 건드리지 않았다.

- 런 로그: srv2 `/home/choiceoh/glm53-logs/fleet/run-logs/47607a72e0ad54ea4d32b11d09d6bddbd805e53a1ccb0aac9f3fe4fa733ec03a.log`
- 결과 원장: srv2 `/home/choiceoh/glm53-logs/qwen-c4be-screen-0920b.jsonl`
- 이 작업의 자동 재개 `qwen`(10분 주기)을 등록했다. 대기 중 변화가 없으면 조용히 기다리며, 완료/실패 때 결과를 회수하고 남은 full 검증을 이어간다. 실험 결과를 확인하기 전 자동 채택하지 않는다.

검사: bracket/profile 59개 CPU 검사 통과. ST admission 복구 후 fleet bracket/pause/pending 62개 검사 통과. 도구 실행기가 넣은 PYTHONPATH는 테스트 제출 argv에서 명시적으로 해제하며, 제품의 코드 주입 거부는 유지한다. signed receipt 실패, 소스 변경, custom GPU 명령 거부도 검사했다.

최종 관련 엔진/프로브 검사는 8파일 77개 중 54개 실행 통과, GPU 전용 23개 건너뜀이었다. `git diff --check`도 통과했다. GPU 증거는 앞의 실제 단일 GPU 실행들로 별도 기록한다.

접수한 명령(컨트롤러 checkout에서):

```bash
REPO=$PWD bash bench/fleet.sh run --gpu --detach qwen-decode-screen-0920b 40 \
  'Qwen C1/C4 A-B-A screen: W8 mixer plus MoE chunks; same RTN cache and K3; no adoption verdict' -- \
  env ST_BRACKET_PROFILE=qwen38 PROFILE=qwen38 ST_SOURCE="$PWD" \
  ST_PRODUCTION_ENV=/dev/null ST_BRACKET_VALIDATION=screen \
  CACHE_DIR=/home/choiceoh/qwen-c4be-cache-0920 ST_KV_GIB=16 ST_MAX_SEQS=4 \
  ST_SELF_CALIBRATE=0 ST_TAP_MTP_INPUTS=0 ST_SPEC_K=3 \
  ST_MTP_PRECISION=bf16 ST_MTP_EXPERTS=bf16 ST_MTP_TUNED= \
  ONEPASS_JSONL=/home/choiceoh/glm53-logs/qwen-c4be-screen-0920b.jsonl \
  bash bench/st_bracket.sh chain \
  A=632d1c7ccfa439af853dd469486f802df2aa5bd4 \
  B=c861b72b8b36c1e40da106310bf2c5fbc6a4580c A
```

재실행은 새 세션·출력 파일로 제출한다. 기존 실행이 대기 중이면 중복 제출하지 않는다.

표본이 없는 현재 속도 개선·품질 통과·배포 완료를 주장하지 않는다.
