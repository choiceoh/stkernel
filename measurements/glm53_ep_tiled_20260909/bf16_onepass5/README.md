# EP SF6 BF16 scatter onepass 5: 목표 67 tok/s 미달, 후보 채택 거부

Frozen `e88f5fd368ef8c895000496101fc7fcf8e7fb344`, session `eptiledbf160910v5`. Canonical **B0 → B1 → A**를 동일 소스에서 모두 실행했다. B0는 사전 지정한 컴파일 준비 arm이고, 주 비교는 **warm B1 TP+SF6 대 A EP+SF6**다. A의 fixed1024×3 pooled **59.88724076 tok/s < 67**로 절대 목표에 실패했다. **후보는 거부하며 default-off를 유지한다.**

| fixed1024 | B0 준비 tok/s | B1 TP+SF6 tok/s | A EP+SF6 BF16 tok/s |
|---|---:|---:|---:|
| rep0 | 80.93133 | 70.52436 | 59.98393 |
| rep1 | 77.94746 | 78.98315 | 59.94231 |
| rep2 | 78.38747 | 69.93944 | 59.73607 |
| pooled | 79.06711270 | 72.92437199 | 59.88724076 |
| engine step/s | 20.41960295 | 20.52757511 | 18.44468953 |

Pooled는 3회 각각 첫 토큰을 제외한 `sum(completion_tokens-1)/sum(decode_s)` = `3069/sum(decode_s)`다. B1 대비 A decode는 **-17.88%**, engine step/s는 **-10.15%**다. 산술 평균·유리한 rep 선택·B0 값으로 목표 이동을 하지 않는다. 67 tok/s는 측정 전 정한 절대 기준이다.

모든 arm은 사실 품질 **18/18**, 한국어 **0/8**, traffic issues `[]`다. 채널별 문자 수·gated counts·원본 요청/출력 hash를 JSONL과 `comparison.json`에 그대로 보존했다. **정상 rc0는 실행·품질·proof 완료이며 성능 채택을 뜻하지 않는다.** Canonical 원본은 gates `[]`, proof `1/1`, delta `-17.9%`, floor_n `2`, status `incomplete`, decision `unresolved`를 두 번 기록했다. 필요한 baseline noise 표본 수를 낮추거나 원본 verdict를 재작성하지 않았다. 이 canonical 통계 판정과 별개로 절대 목표 67 미달이 확인되어 후보는 거부됐다.

| warm B1/A prefill observation | 실제 prompt tokens | B1 TTFT s / tok/s | A TTFT s / tok/s |
|---|---:|---:|---:|
| 2K: within-leg 최소 TTFT | 2128 | 0.875940 / 2429.39092 | 0.706143 / 3013.55579 |
| 32K: 단일 요청 | 32545 | 10.583262 / 3075.13880 | 10.385828 / 3133.59716 |
| 128K: 단일 요청 | 128559 | 40.913601 / 3142.20688 | 39.595434 / 3246.81377 |

B0만 `cold_compile:true`, **B1/A는 양쪽 필드가 없어 동일 warm class**다. 따라서 이전 two-arm의 cold mismatch는 없다. 이 쌍에서 prefill TTFT는 낮아졌으나, 2K 값은 within-leg 최소 표본이고 32K/128K는 독립 warm 반복이 없는 단일 요청이다. 2K 전체 TTFT는 B1 `[1.917767,0.887016,0.875940]`, A `[2.228683,0.707598,0.706143]`이며 A의 첫 요청은 더 느렸다. 이 관측이 decode 실패를 상쇄하거나 통계적 개선·기본값 채택을 증명하지 않는다. B0 원본도 모두 유지했다.

[CPU bf16_cpu5](../bf16_cpu5/README.md)는 같은 소스의 109 CPU tests, 6 실제 no-device CuTe lowerings와 실제 CPU 프로세스가 import한 `fp4_common` helper의 path/전체 SHA·동일 함수 객체를 별도 증명한다. GPU canary는 그 helper의 별도 파일 SHA를 직접 기록하지 않으므로 CPU helper 증거를 GPU import 측정으로 바꾸지 않는다. Pinned serving image와 13개 source identity는 4rank 원문에 보존했다.

A **4rank 각각 첫 eligible actual-weight layer**의 9cases/54candidate+54reference 비교가 bad_rows0으로 완료됐다. M6의 key19는 index15 `bf16_scatter`, 뒤에 ring→word→BF16 suffix이며 M12/24/32는 key16 FP32다. Packed SF6 before/after 불변·graph/side replay·실제 graph capture·trim·FINALIZED42를 확인했다. 각 rank는 raw scale 4,756,340,736 bytes를 해제하고 packed 3,604,414,464 bytes를 보유한다. 수치 canary가 전 모델 모든 layer·full sanitizer 또는 bitwise FP32/BF16 합산 동일성을 보장하지 않는다.

B0/B1/A ready snapshot은 각각 **00:16:20.792 / 00:23:26.102 / 00:33:33.190 KST**다. 각4rank ID/start·이미지·71개 source mount를 캡처 전후 검증했고 B1/A의 source/image는 동일하다. Private 전체 Env 차이는 TP_Q0 1→0, EP_TILED 0→1, launcher COMPACT absent→1뿐이었다. Raw Docker inspect/Env/HostConfig는 private에만 유지하고 공개본은 allowlisted identity·해시를 담는다.

정상 wrapper 종료는 **2026-09-10 00:36:25.811 KST rc0**, owned holder는 해제됐다. Public 복구는 별도 idle-controller로 넘겨졌으며 이 아카이브의 terminal이 복구 완료를 뜻하지 않는다. Exact frozen source clean/terminal 확인 후 원본을 두 번 읽어 stat·크기·SHA를 검사했다. 수집 명령: `python3 /tmp/glm53_collect_ep_bf16_completed5.py` → `python3 /tmp/glm53_archive_ep_bf16_completed5_local.py`. `onepass.jsonl` 3행·`verdicts.jsonl` 2행은 원본 바이트 그대로다. `provenance.json`/`manifest.json`은 원본·저장 SHA를 구분하고 gzip mtime=0, allowlist·credential 패턴 검사로 raw Env 노출을 막았다. 추가 GPU·HTTP 추론·서비스·큐·테스트·소스 변경은 없다.
