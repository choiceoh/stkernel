# EP SF6 A-ring onepass 3: 67 tok/s 목표 미달, 품질 gate FAIL

Frozen `57914a3f8bb01a76a20099b9c2605be3ea15b7f4`, session `eptiledring0909v3`. 동일 소스의 TP+SF6 B → EP+SF6 A canonical 원패스다. A pooled **61.35529851 tok/s**로 목표 67에 미달했다. 사실 품질은 양쪽 18/18이지만 A 한국어 gate가 1/8로 실패하여 canonical verdict는 **invalid / GATE FAIL: korean 1/8**, 종료 rc4다. 기본값을 채택하거나 품질 기준을 바꾸지 않는다.

| fixed1024 | B TP+SF6 tok/s | A EP+SF6 a_ring tok/s |
|---|---:|---:|
| rep0 | 70.11283 | 60.05171 |
| rep1 | 71.64968 | 63.47653 |
| rep2 | 72.28344 | 60.64516 |
| pooled | 71.33694298 | 61.35529851 |
| engine step/s (fixed intervals) | 20.45511677 | 18.20276249 |

Pooled는 3회 각각 첫 토큰을 제외한 `sum(completion_tokens-1)/sum(decode_s)` = `3069/sum(decode_s)`다. 산술 평균이나 유리한 rep 선택이 아니다. A decode는 B보다 13.99% 낮고 engine step/s도 11.01% 낮다. 이 비율은 실패한 A의 서술용 관측값이며 acceptance가 아니다.

A fixed rep2의 `reasoning` 채널에서 `Halvorsen博士`의 cjk_mixed 2개가 검출됐다. 나머지 gated 종류는 0이며 B 한국어는 0/8이다. 두 arm 모두 고정 길이 응답의 content 채널은 비어 있어 '최종 content 품질이 검증됐다'고 해석할 수 없다. 기존 combined-text classifier/gate 및 원본 verdict는 그대로 보존했다.

| prefill raw observation | actual prompt tokens | B TTFT s / tok/s | A TTFT s / tok/s |
|---|---:|---:|---:|
| 2K: within-leg 최소 TTFT | 2128 | 0.876882 / 2426.78033 | 0.713152 / 2983.93709 |
| 32K: 단일 요청 | 32545 | 10.636055 / 3059.87524 | 10.039151 / 3241.80789 |
| 128K: 단일 요청 | 128559 | 40.970631 / 3137.83307 | 39.445337 / 3259.16848 |

**B는 `cold_compile:true`, A는 해당 필드가 없다. 따라서 matched-warm prefill 이득을 주장하지 않는다.** 2K TTFT 전체 표본은 B `[2.374799,0.878133,0.876882]`, A `[2.232832,0.713152,0.757185]`이며 32K/128K에는 독립 warm 반복이 없다. 정확한 필드와 표본은 `comparison.json`/원본 JSONL에 있다.

이번 A는 이전 ring_onepass2의 cache-key 검사 누락을 고친 source다. 4rank 각각 첫 eligible actual-weight layer의 9cases/54candidate+54reference 수치 비교가 bad_rows0으로 완료됐고, mixed6 a_ring cache key, packed SF6 전후 불변, graph/side replay·실제 graph capture·trim·FINALIZED가 확인됐다. 이는 bounded startup canary이며 전 모델 모든 layer 또는 full sanitizer acceptance가 아니다. 숫자 검증 통과가 serving 품질/속도 통과를 대신하지 않는다.

B/A ready private snapshot은 각 23:09:05.333/23:16:10.201 KST에 저장했다. 4rank의 이미지·71개 frozen source mount가 같고 컨테이너 ID/start와 source가 캡처 전후 안정적이다. 전체 Docker Env 차이는 TP_Q0 1→0, EP_TILED 0→1, launcher EP_COMPACT absent→1뿐이다. raw Docker inspect/Env/HostConfig는 private에만 유지하고 공개본은 allowlist identity·해시를 담는다.

B와 A 모두 동일한 `[prep-fused] preimage drift -> DISARM (stock path)`를 기록했다(`v1/worker/utils.py fd27... != 3dcd...`). 공통 실행 조건으로 남기며 A만의 회귀 원인으로 단정하지 않는다. 동일 request hashes와 실행 workload는 `comparison.json`, 원본 내용은 `onepass.jsonl`, `verdicts.jsonl`, `submission.json`, `run.log.gz`, `boot/`에 있다. `A-canary/`는 4rank 전체 PASS JSON 원문, `A-ready/`는 그 원본 serving log와 source/identity 증거를 담는다.

수집은 정상 terminal rc4·해제된 holder와 exact clean frozen source를 읽기 전용으로 확인한 뒤 원본을 두 번 읽어 stat/크기/SHA를 검사했다. `/tmp/glm53_collect_ep_ring_completed3.py` → `/tmp/glm53_archive_ep_ring_completed3_local.py`. `provenance.json`과 `manifest.json`은 원본/저장 SHA를 구분하며 gzip mtime=0이다. `privacy-scan.json`의 credential 패턴 검사와 명시적 allowlist 검토를 거쳤다. GPU·추가 HTTP·서비스·큐 변경, 추가 테스트, 원본 verdict 수정 없이 보존했다.
