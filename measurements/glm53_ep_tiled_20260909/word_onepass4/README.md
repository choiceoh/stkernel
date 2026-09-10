# EP SF6 word-unpack onepass 4: 67 tok/s 목표 미달, 품질 gate FAIL

Frozen `549fd55319c3379438c3cb99694df9f28c01aefb`, session `eptiledword0909v4`. 동일 소스의 TP+SF6 B → EP+SF6 word-unpack A canonical 원패스다. A pooled **66.30368965 tok/s < 67**이다. 사실 품질은 양쪽 18/18, 한국어는 B 0/8·A 1/8이므로 canonical 결과는 **invalid / GATE FAIL: korean 1/8**, 종료 rc4다. 기본값 또는 속도 개선을 채택한 결과가 아니다.

| fixed1024 | B TP+SF6 tok/s | A EP+SF6 word-unpack tok/s |
|---|---:|---:|
| rep0 | 67.58975 | 62.81452 |
| rep1 | 70.52315 | 65.62780 |
| rep2 | 71.43685 | 70.97726 |
| pooled | 69.81085060 | 66.30368965 |
| engine step/s | 20.47366557 | 18.21260891 |

Pooled는 각 1024토큰 응답에서 첫 토큰을 제외한 `sum(completion_tokens-1)/sum(decode_s)` = `3069/sum(decode_s)`다. A는 B보다 5.02% 낮다. A engine **18.21260891 step/s**는 이전 [ring_onepass3](../ring_onepass3/README.md)의 **18.20276249**와 약 0.05% 차이여서 engine 이득은 확립되지 않았다. 이전 실험과의 출력 tok/s 차이는 matched 반복 효과 검증을 대신하지 않는다.

A **fixed rep0**의 `reasoning` 채널에서 `Halvorsen博士`의 cjk_mixed 2개가 검출됐다. snippet은 `Francis Crick biography - Halvorsen博士 signing tundra report` 주변이며 원본 offset·선택 채널별 문자 수·gated 종류별 counts·bounded snippet을 `comparison.json`과 원본 JSONL에 보존했다. 그 외 gated 종류는 0이다. 두 arm의 fixed 응답은 content/reasoning_content 채널이 비어 있으므로 'content 품질 통과'로 해석하지 않는다. 기존 combined-text classifier와 gate/verdict는 바꾸지 않았다.

| prefill raw observation | actual prompt tokens | B TTFT s / tok/s | A TTFT s / tok/s |
|---|---:|---:|---:|
| 2K: within-leg 최소 TTFT | 2128 | 0.832163 / 2557.19225 | 0.758776 / 2804.51586 |
| 32K: 단일 요청 | 32545 | 10.906610 / 2983.97032 | 9.890686 / 3290.46955 |
| 128K: 단일 요청 | 128559 | 41.012409 / 3134.63663 | 39.413085 / 3261.83549 |

**B는 `cold_compile:true`, A는 필드가 없다. matched-warm prefill 개선 판정은 불가하다.** 2K 전체 TTFT 표본은 B `[2.402552,0.832163,0.874055]`, A `[1.873835,0.766209,0.758776]`다. 32K/128K는 독립 warm 반복이 없다. 품질 실패까지 포함한 원본 관측치를 보존하며 유리한 표본만 골라 판정하지 않는다.

[CPU word4 evidence](../word_cpu4/README.md)는 같은 소스의 107 CPU tests와 6 실제 no-device CuTe lowerings를 별도 증명한다. M6 SF6 복원 PTX의 명령 감소는 CPU 정적 결과이며 GPU 속도 증명이 아니다. 이번 A는 **4rank 각각 첫 eligible actual-weight layer**의 9cases/54candidate+54reference 비교가 bad_rows0으로 완료됐다. M6 key18의 ring→word suffix, M12/24/32 key16, packed SF6 전후 불변, graph/side replay·실제 graph capture·trim·FINALIZED를 보존했다. 전체 모델 모든 layer 또는 full sanitizer acceptance를 의미하지 않는다.

B/A ready snapshot은 각각 23:36:39.510 / 23:46:44.859 KST에 저장했다. 4rank 이미지·71개 frozen source mount가 동일하고 각 컨테이너 ID/start·source는 캡처 전후 안정적이다. private 전체 Env 비교 차이는 TP_Q0 1→0, EP_TILED 0→1, launcher EP_COMPACT absent→1뿐이었다. raw Docker inspect/Env/HostConfig는 공개본에 넣지 않았다. `A-canary/`에는 4rank PASS JSON 원문, `A-ready/`에는 원본 serving log와 allowlisted identity가 있다.

정상 wrapper는 **2026-09-09 23:49:46.206 KST rc4로 종료**했고 exact owned holder는 해제됐다. 이는 public 모델 복구 완료 증거와 별개다. exact frozen source clean 및 terminal을 읽기 전용으로 검사한 뒤 원본 파일을 두 번 읽어 stat·크기·SHA를 검증했다. 수집 명령은 `python3 /tmp/glm53_collect_ep_word_completed4.py` → `python3 /tmp/glm53_archive_ep_word_completed4_local.py`다. 원본 `onepass.jsonl`, `verdicts.jsonl`, `submission.json`, `run.log.gz`, `boot/`와 source binding을 보존했다. `provenance.json`/`manifest.json`은 원본과 저장 SHA를 구분하며 gzip mtime=0이다. 명시적 allowlist와 credential 패턴 검사를 거쳤다. 추가 GPU·HTTP·서비스·큐 조작, 테스트, 원본 verdict 변경 없이 수집했다.
