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

공통 실행부의 CPU 회귀 검증(PyTorch·GPU·체크포인트 없이 실행):

    python3 -m unittest discover -s tests -p 'test_engine_*.py' -v

요청과 캐시의 소유권은 다음 경계에서 확정한다:

- `Runner.submit`은 입력 검증 → KV·상태 슬롯 예약 → `Model.open` → 스케줄러 등록 순서다.
  중간 실패 시 확보한 자원을 반납한다. `Model.close`는 실패한 `open`의 부분 상태도 정리해야 한다.
- 디코드는 `BlockPool.reserve_to`로 요청별 쓰기 끝 위치를 받아 배치 전체의 증가분을 먼저 검증·예약한다. 자원이 부족하면
  어느 행의 토큰 수도 바뀌지 않는다. 거절된 드래프트가 쓰던 공간은 다시 예약하지 않는다.
  블록 풀과 러너 상태는 러너를 소유한 스레드에서 갱신한다.
- 대기 상한은 디코드 폭에 여유가 있을 때 프리필을 허용한다. `max_running`이 꽉 차면 기존 요청이
  끝날 때까지 기다린다. 이미 실행 폭을 넘긴 상태는 오류이며, 앞쪽 요청만 잘라 실행하지 않는다.
- `TieredKV.resume`은 빈 행으로만 복귀한다. 읽기 실패 시 새 블록을 반납하고 디스크 사본을 보존한다.
  읽기 완료 후 디스크 사본 정리에 실패하면 복귀한 메모리는 유지한다.
- `NvmeTier.run_async`는 `Future`를 반환한다. `.done()`으로 완료를 확인하고 `.result()`로 결과·오류를
  받는다. 전송끼리는 공유 스테이징 버퍼 사용을 직렬화한다. 디코더는 이 잠금이나 Future를 기다리지 않는다.
- 러너의 계측은 프리필·디코드별 누적 시간과 호출 수를 보존하고, 개별 스텝 기록은 고정 크기 링에 남긴다.
  부팅·진단 계측은 기존처럼 단계별로 보존한다.

이 검증은 스케줄링·자원 예약·실패 복구 계약을 확인한다. GLM 수치 일치, CUDA 전송 정확성,
NVMe 트래픽 중 디코더 ITL은 위 실가중치 검사와 `probes/kv_tier_interference.py`로 별도 판정한다.

GLM 조합을 실제 요청 실행에 연결하는 경로는 `profiles/glm53/runtime.py`의 `Glm53Runtime`이다.
`Glm53Net`과 `Glm53Caches`를 바인딩한 뒤 `submit(seq, ids, max_new_tokens)` →
`step()` → `take_result(seq)` 순서로 사용한다. 마지막 프리필에서 첫 토큰을 샘플링하고,
EOS나 생성 한도에 도달하면 그 스텝에서 KV와 상태 슬롯을 반납한다. 이 간단한 검증 경로는
`draft_slots=0` 계약을 사용한다. PR #534의 `Glm53Engine`·`boot.py`는 같은 캐시와 러너 위에서
DFlash2를 연결하며, 드래프터 문맥 링도 같은 아레나의 상태 슬롯 예산에 포함한다.

`Glm53Caches`는 하나의 아레나에 다음 영역을 선언한다:

- 물리 KV 블록마다 모든 DSA 층의 fp8 latent와 압축 키·스케일을 함께 저장한다.
  블록 전체가 NVMe 전송 단위이며, 레인에는 층별 절대 슬롯 주소를 전달한다.
- 시퀀스 상태 슬롯마다 KDA conv·recurrent 링과 인덱서 꼬리 링을 둔다.
  인덱서 링은 `kpool - 1 + spec_k`개 위치를 보존해 드래프트 거절 후에도 앞선 풀을 복원한다.
  긴 프리필은 최신 창만 한 번씩 기록하여 CUDA의 중복 scatter 순서에 의존하지 않는다.
- 블록표도 아레나에 상주하며 `prepare(step)`이 예약된 문맥과 상태 슬롯 소유권을 확인하고 갱신한다.

사전샤딩은 `RankWriter`가 모든 실제 가중치의 데이터 오프셋을 256바이트 경계에 맞춘다.
작은 스케일 뒤의 행렬도 TMA 정렬을 유지하도록 safetensors의 명시적 U8 패딩 텐서를 사용한다.
표준 safetensors 리더로 읽을 수 있고, 로더는 패딩을 포함한 연속 범위를 한 번 업로드한 뒤 뷰를 만든다.
**기존 PR #532 랭크 파일을 서빙 레인에서 사용할 때는 수정된 preshard로 다시 생성해야 한다.**
로딩 시 잘린 파일과 진행하지 않는 쓰기는 즉시 오류가 되며 무한 재시도하지 않는다.

서빙 KDA 어댑터는 엔진의 `[H,K,V]` 상태와 커널의 `[H,V,K]` 상태를 경계에서 변환한다.
recurrent 레인도 연결되어 검증 토큰마다 상태를 반환하고, 시작 상태를 보존한다.
MLA 참조 레인은 선택된 fp8 행만 변환하며, 패딩 슬롯이 가리키는 미사용 블록의 NaN을 마스킹한다.

네 노드 검증은 각 노드에서 같은 인자로 `check.py --distributed`를 실행한다.
기본 노드 순서는 **rank 0=srv2, rank 1=srv1, rank 2=srv3, rank 3=srv4**다.
`MASTER_ADDR`은 rank 0 서버를 가리켜야 한다. `RANK`, `WORLD_SIZE=4`, 격리된 `MASTER_PORT`,
RoCE 인터페이스·GID는 실행기가 설정하고, `--ranks`에는 해당 랭크의 정렬된 파일을 둔다.
`probes/engine_comm_check.py`로 가중치 없이 통신을 먼저 검증할 수 있다.

추가 장치 검사:

    bash probes/run_mk_probe.sh probes/engine_kda_check.py
    python3 probes/engine_cuda_io_check.py

검사 결과와 실제 사용한 네 노드 실행기는
[`measurements/st_engine_runtime_20260911`](../measurements/st_engine_runtime_20260911/README.md)에 보관한다.
이는 2개 실제 층(KDA+dense, DSA+NVFP4 MoE)의 실행·캐시 계약 검사다. 통합 전 로그는
참조 전문가를 사용했고, PR #534 통합 후 `check.py --lanes served`는 b12x 전문가를 포함한다.
전체 45층 onepass 품질, 실제 DFlash2 수용률, 그래프 기반 요청 실행 및 처리량·ITL 판정은 별도다.
