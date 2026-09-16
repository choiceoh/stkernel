# 실제 draft replay 캡처

**8개 C=1 greedy 요청에서 16개 사례 수집 완료.** 4개 rank 각각에 실제 prepared state, proposal 직전 context ring·embedding, baseline draft 및 실제 greedy continuation 정답을 보관했다. 측정용 서버는 종료했고 fleet lease는 반납했다.

| 문맥 구간 | 실제 입력 토큰 | 요청 | 고유 사례 |
|---|---|---:|---:|
| 짧은 문맥 | 3,429 / 3,468 / 3,461 / 3,455 | 4 | 8 |
| 32K 문서 | 34,138 / 34,549 | 2 | 4 |
| 128K 문서 | 133,044 / 133,196 | 2 | 4 |

문서 크기와 실제 prompt 길이는 다르다. 템플릿·질문·출력 스키마를 포함한 입력 길이를 위에 기록했다. 각 요청은 `thinking=false`로 렌더링한 입력 ID를 engine endpoint에 전달했고 temperature=0, max_tokens=512, retain=false를 사용했다. 각 요청에서 첫 eligible step과 16 step 뒤를 캡처했다. 모두 512토큰까지 생성했으므로 완결 답변 품질 평가용 기록으로 취급하지 않는다.

## 검증

- 4개 rank 모두 state 1개, snapshot 16개, 정답 16개이며 SHA-256 검증 통과.
- 모든 사례의 baseline draft, 정답, position, request key가 rank 간 일치.
- 16개 정답 모두 실제 응답의 해당 위치에서 나온 7개 토큰과 정확히 일치. 짧게 잘린 정답 없음.
- context ring과 embedding에 NaN/Inf 없음. rank별 dense reader 30개 보존.
- CPU 전용 컨테이너에서 파일을 검사했고 CUDA를 초기화하지 않음.

검증 도중 HTTP request ID와 엔진의 재사용되는 resident seq를 같은 것으로 연결하면 잘못된 대응이 생긴다는 점을 확인했다. 최종 검증은 순차 요청별 `cases_added`로 연결하고 admission nonce를 포함하는 `request_key` 및 실제 출력 위치를 추가 확인한다. 요청 0~7은 `0:1`~`0:8`에 대응한다. 캡처 파일이나 정답을 수정하지 않았다.

[종합 검증 JSON](audit.json), [요청별 기록](live-capture/requests.json), [수집 로그](collection.log), [fleet 실행 및 반납](fleet-launch.log).

재검증:

```sh
python3 measurements/st_draft_sensitivity_20260916/audit_capture.py measurements/st_draft_sensitivity_20260916/live
```

## 실행 및 보관

- 코드: `4e6698a1a47ffe447c4293d3cfc3122065bc1c69`, 브랜치 `codex/draft-replay-capture-0916`.
- fleet 세션: `draftreplay4-0916`, 측정용 포트 8001. [실행 이미지 및 release](runtime.txt).
- rank 0/1/2/3은 각각 srv2/srv1/srv3/srv4에 있다.
- 각 노드의 보관 경로: `/home/choiceoh/expert-capture/draft-sensitivity-0916-7e62/captured/rankN/`.
- 4개 rank 합계 prepared state 1,919,969,708바이트, snapshots 675,446,528바이트: 약 **2.42 GiB**. 작은 JSON 메타데이터는 별도다.
- 원본은 각 노드의 `/home/choiceoh/glm53-logs/st-bracket-dumps/draftreplay4-0916-hold-4e6698a1a47f/draft-replay/rankN/`에도 보존했다.

총 요청 시간은 158.07초다. 비동기 decode 중단과 파일 저장을 포함하므로 **정상 서빙의 tok/s 또는 캡처 오버헤드 실측으로 해석하지 않는다.** native GPU replay와 층별 precision 교체 비교는 아직 실행하지 않았다. 실제 순위·수용률 개선·성능 개선 결론은 없다.

## 부팅 실패와 조치

첫 GPU 실행 `draftreplay3-0916`은 최신 main에서 자동 활성화된 별도 FC pair collector가 `consume_weights` 이후 남아 있지 않은 BF16 source를 요구해 종료됐다. replay prepared state 저장은 성공했지만 요청 사례는 없었다. [rank 0 오류](failed-boot-fc-rank0.log), [실패 세션 기록](failed-boot-fleet.log).

이번 측정 arm에서만 `DRAFT_FC_CAPTURE_WHEN_MISSING=False`로 설정해 재실행했다. main이나 운영 배포 설정을 변경하지 않았고 FC bias를 적용하지 않았다. 이전 실패의 state와 이번 성공 데이터는 다른 dump 경로에 있다.
