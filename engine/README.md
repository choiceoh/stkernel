# ST 엔진

stkernel 의 자체 추론 엔진. 네 가지를 옵션이 아니라 **형태**로 가진다:

- **TP=4** — 스파크 네 대가 유일한 world. 랭크-로컬 형상은 사실(`profiles/*/facts.py`)이고 코드에 `// world` 가 없다.
  한 노드 검증도 `base/comm.LocalTP` 로 네 랭크를 스레드로 돌려 진짜 all-reduce 의미를 쓴다.
- **DGX Spark(GB10)** — 장치 하나, 통합 메모리, SM121. 부팅·검증 때 단언(`facts.check_box`).
- **NVFP4 가 기본형** — packed 바이트 그대로 상주, packed 위에서 TP, 전역 스케일은 곱셈자, 서빙 커널이 레인. 유일형은
  아니다: 체크포인트가 bf16 으로 가진 것은 bf16 으로 쥔다.
- **레거시 없음** — 폴백·옵션·플랫폼 디스패치·전방 컨텍스트·가중치 로더 추상이 없다. 사실과 만료 노브만(D11).

설계 원칙은 `CHARTER.md`(D1~D16). 세 계층(D15):

    base/       모델 이름이 없는 것: 아레나, 로더(사전샤딩된 랭크 파일의 범위 읽기), KV 블록/슬롯, NVMe 티어, 스케줄러,
                스텝 메타, 러너, 기록/사망 덤프, 설정(사실+만료 노브), 증명·판정, 그래프, comm(플릿 / LocalTP)
    modules/    특징 모듈: 선형 어텐션(KDA), 희소 인덱서·희소 MLA, NVFP4 선형·MoE·양자화, 하이퍼커넥션, 노름, 회전, 로짓
    profiles/   모델별: 사실·가중치 지도(specs)·사전샤딩·레인 표·조합(net)·검증(check). glm53 이 첫 대상.

빠른 확인(GLM-5.3, 실가중치, 한 노드, TP=4 스레드):

    PYTHONPATH=. python3 engine/profiles/glm53/check.py --layers 0-4          # 참조 레인
    bash probes/run_engine_check.sh --layers 0-4                              # 서빙 커널 레인, 판정 이미지 안

측정과 판정은 `MEASUREMENTS.md` 44~45차.
