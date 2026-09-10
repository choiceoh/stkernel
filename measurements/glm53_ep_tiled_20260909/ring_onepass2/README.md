# EP SF6 A-ring onepass 2: 아티팩트 선택 검사에서 종료

Frozen source `5663b1beb13e97116e6244af02a22ecc5074a263`, session `eptiledring0909v2`. 정상 chain은 TP+SF6 B → EP+SF6 A이며, A startup에서 종료됐다. B의 canonical 원패스 원본은 보존했지만 A의 serving tok/s·TTFT 기록은 없다. **67 tok/s 달성 또는 실패로 판정할 수 없다.**

4rank 모두 첫 `mixed6 / initial-C1-eager` 수치 비교를 실제 실행했고 `bad_rows=0`을 기록했다. 이어 `_cache_evidence()`의 `native EP tiled decode artifact was not selected/warmed` 검사에서 실패했다. frozen static source는 M<=8 SF6에 새 `glm53_ep_static_sf6_a_ring_v1` suffix를 붙이고, frozen canary는 suffix 없는 기존 키를 조회한다. 이 source 불일치와 최초 오류를 보존하며 원래 FAIL verdict를 PASS로 고쳐 쓰지 않았다.

| rank | 첫 C1 max relative abs | 첫 C1 max relative L2 | bad rows |
|---|---:|---:|---:|
| 0 / head | 0.0112359552 | 0.0054609962 | 0 |
| 1 / .1 | 0.0093167704 | 0.0055711013 | 0 |
| 2 / .3 | 0.0133333337 | 0.0054369578 | 0 |
| 3 / .4 | 0.0109890113 | 0.0052958508 | 0 |

rank당 baseline reference 비교 기록 54개와 candidate 최초 eager 비교 **1개**가 완료됐다. candidate 전체 54개 비교가 통과한 것이 아니다. 나머지 candidate 53개 비교, graph/side replay와 full canary acceptance는 미완료다. cache dictionary 전체가 receipt에 없으므로 키 불일치 설명은 함께 보존한 frozen source에 근거한다.

A FAIL 완료 시각은 .1 22:52:25.679, head 22:52:39.725, .4 22:52:40.504, .3 22:52:44.913 KST다. 정상 fleet terminal은 22:52:59.669, payload/outer rc1이며 수집 시 해당 holder가 없었다. readiness 이전 실패로 A ready 4node 스냅샷은 없다.

B ready 스냅샷은 22:44:28.775 KST에 4rank 모두 같은 frozen source 71 mounts·이미지·TP4/EP0·컨테이너 ID/start 전후 일치, TP Q0 canary PASS·graph capture·trim COMPLETE를 확인했다. `B-ready/`는 허용된 필드와 원본해시만 담는다. **raw Docker inspect/Env/HostConfig는 private 원본에 남기고 이 공개 아카이브에서 제외했다.**

`onepass.jsonl`, `submission.json`은 정확한 원본 bytes다. `failure/`의 4로그와 `run.log.gz`는 원본을 mtime=0 gzip으로 보존하며 전체 trace가 포함된다. `canary/`의 FAIL JSON은 원본 marker의 JSON substring 전체다. `terminal.json`은 원본 pending의 허용된 종료 필드 요약이며 원본 SHA를 연결한다. `source/`는 수집 시 exact clean frozen HEAD를 확인한 build bytes다. A의 4rank receipt source hash는 서로 같고 frozen B mounts와 대조했다.

수집: `/tmp/glm53_collect_ep_ring_failure2.py`가 정확한 원격 파일을 두 번 읽어 stat·크기·SHA 일치 후 전송했다. `/tmp/glm53_archive_ep_ring_failure2_local.py`는 그 로컬 private 원본만 사용해 이 아카이브를 만들었다. `provenance.json`과 `manifest.json`은 원본/저장 SHA 및 추출 근거를 구분한다. credential 패턴 검사와 파일 allowlist 검토를 했으며 `privacy-scan.json`에 범위를 기록했다. GPU·추가 HTTP·서비스·큐·소스 변경 없이 수집했다. 이전 B/private 증거는 수정하지 않았다.
