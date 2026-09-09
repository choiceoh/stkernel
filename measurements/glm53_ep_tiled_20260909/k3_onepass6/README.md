# EP SF6 K3 onepass 6: 원본 측정과 사전 지정 목표

Frozen `6977199cf699f82696f925f77bc930d500135532`, session `eptiledk30910v6`. Canonical **B0 → B1 → A → ABASE**를 같은 소스에서 실행했다. B0는 컴파일 준비 arm이고, 주 비교는 **B1 TP+SF6/K5 대 A EP+SF6/K3**다. 요청·품질·한국어 gate 및 fixed1024×3 조건은 유지했다.

| fixed1024 | B0 준비 tok/s | B1 TP+SF6 K5 tok/s | A EP+SF6 K3 tok/s |
|---|---:|---:|---:|
| rep0 | 68.72487 | 77.27785 | 60.36829 |
| rep1 | 78.59247 | 72.60268 | 60.31594 |
| rep2 | 73.10229 | 71.57161 | 61.90889 |
| pooled | 73.25274207 | 73.73555378 | 60.85547982 |
| engine step/s | 20.53069649 | 20.38948240 | 21.03647277 |

Pooled는 `sum(completion_tokens-1)/sum(decode_s)` = `3069/sum(decode_s)`다. A의 사전 지정 **67 tok/s 목표 충족: False**. B1 대비 decode **-17.47%**, engine step/s **3.17%**. 유리한 rep·B0를 대신 선택하지 않는다. 이 아카이브는 default adoption이나 통계적 개선을 승인하지 않는다.

| arm | facts | Korean dirty/total | cold_compile 원문 | traffic issues |
|---|---|---|---|---|
| B0 | 18/18 | 1/8 | True | [] |
| B1 | 18/18 | 1/8 | 필드 없음 | [] |
| A | 18/18 | 0/8 | 필드 없음 | [] |
| ABASE | 18/18 | 0/8 | 필드 없음 | [] |

B1/A 품질·한국어·traffic 조건을 모두 통과한 비교인지: **False**. 실패한 baseline은 채택 근거로 쓰지 않는다. B0의 품질 실패도 보존했다. 요청/출력 hash와 채널별 gated counts·bounded snippet은 원본 `onepass.jsonl` 및 `comparison.json`에 있다. 종결 rc0와 canonical verdict는 별개다. 원본 `verdicts.jsonl`의 incomplete/invalid/noise-floor 판단을 완화하거나 재작성하지 않았다.

| prefill 관측 | 실제 prompt tokens | B1 TTFT s / tok/s | A TTFT s / tok/s |
|---|---:|---:|---:|
| 2000 | 2128 | 0.872080 / 2440.14385 | 0.745598 / 2854.08356 |
| 32000 | 32545 | 10.616468 / 3065.52060 | 9.948743 / 3271.26745 |
| 128000 | 128559 | 40.930475 / 3140.91154 | 40.332436 / 3187.48413 |

B1/A warm cache class 호환: True. 2K 표는 within-leg 최소 TTFT이며 전체 요청은 원문에 유지한다. 32K/128K는 독립 warm 반복이 없는 단일 요청이다. 품질 실패가 있거나 cold class가 다르면 prefill 개선 판정을 내리지 않는다.

A 4rank 각각 **첫 eligible actual-weight layer**의 12cases/72candidate+72reference 비교, M4/6/8 BF16 key19와 M12/16/24/32 FP32 key16을 보존했다. Packed SF6 before/after, graph/side replay 및 FINALIZED42 원문도 포함한다. rank별 raw scale 4,756,340,736 bytes 해제, packed 3,604,414,464 bytes 보유 계약이다. 모든 layer의 수치 또는 full sanitizer 증거로 확대하지 않는다. [같은 소스 CPU gate](../k3_cpu6/README.md)는 별도 증거이며 GPU import 측정으로 바꾸지 않는다.

각 arm 4rank의 실제 argv/env K(B0/B1=5,A=3), pinned image·71 source mounts·container ID/start를 전후 확인했다. Speculation config/command는 safe digest만 공개한다. 실제 draft 실행은 A 원본 `speculation`의 기존 전체 onepass metrics-counter 증거에 따르며, 이를 fixed1024 구간별 acceptance로 재해석하지 않는다. Raw Docker Env/inspect/HostConfig는 private에만 두었다.

원본 canonical JSONL·submission·run/boot logs·terminal 및 source를 이중 stat/SHA로 캡처했다. `provenance.json`은 원본/저장 해시를 구분하며 `manifest.json`은 전체 저장 파일을 검증한다. 수집 명령은 `python3 /tmp/glm53_collect_ep_k3_completed6.py` → `python3 /tmp/glm53_archive_ep_k3_completed6_local.py`다. Owned holder 해제는 public 복구 완료를 뜻하지 않는다. 추가 추론·GPU 실행·서비스·큐 변경은 없다.

## 자동 보충 기준선과 읽기 범위

B0/B1이 한국어 게이트를 실패하여 정상 chain의 유효 기준선 수가 0/2였다. 기존 `chain.sh:53–62`가 A 뒤에 **ABASE**를 자동 실행했다. 별도 수동 예약이나 후보 재실행은 없었다. A는 01:26:37 판정까지 완료됐고 ABASE 뒤 전체 chain은 01:34:00에 끝났다. ABASE는 74.31471111 tok/s, facts18/18, Korean0/8, engine 20.53829872 step/s였다. 원판정은 ABASE 대비 −18.1%, 유효 baseline n1로 incomplete/unresolved다. 사전 지정 B1의 실패를 ABASE로 지우지 않으며, 절대67 실패도 변하지 않는다.

ABASE 원본 row와 head boot log는 보존하지만 해당 팔의 독립 4rank ready snapshot은 없다. 관찰기의 `valid_expected_prefix:false`는 3팔만 예상했던 파서가 4번째 자동 기준선을 만난 결과다. 원문 이벤트를 수정하지 않았으며 최종 archive는 정확한 4팔 순서와 세 기본 계획 팔의 기존 snapshot을 각각 검증한다.

K3 whole-onepass counter delta는 drafts2164, drafted6492, accepted4018, positions1759/1273/986이다. 실제 K3 실행 proof2/2가 통과했지만 이 수치를 fixed1024 구간 수용률로 해석하지 않는다. A 첫 2K TTFT는2.211582s로 B1의1.921149s보다 느렸다. 표의 2K warm 최소값만으로 첫 요청의 개선을 주장하지 않는다.
