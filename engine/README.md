# ST 엔진

stkernel 의 자체 추론 엔진. 네 가지를 옵션이 아니라 **형태**로 가진다:

- **TP=4** — 스파크 네 대가 유일한 world. 랭크-로컬 형상은 사실(`profiles/*/facts.py`)이고 코드에 `// world` 가 없다.
  한 노드 검증도 `base/comm.LocalTP` 로 네 랭크를 스레드로 돌려 진짜 all-reduce 의미를 쓴다.
- **DGX Spark(GB10)** — 장치 하나, 통합 메모리, SM121. 부팅·검증 때 단언(`facts.check_box`).
- **NVFP4 가 기본형** — packed 바이트 그대로 상주, packed 위에서 TP, 전역 스케일은 곱셈자, 서빙 커널이 레인. 유일형은
  아니다: 체크포인트가 bf16 으로 가진 것은 bf16 으로 쥔다.
- **레거시 없음** — 폴백·옵션·플랫폼 디스패치·전방 컨텍스트·가중치 로더 추상이 없다. 사실과 만료 노브만(D11).

설계 원칙은 `CHARTER.md`(D1~D16). 세 조합 계층(D15)과 실행 커널:

    base/       모델 이름이 없는 것: 아레나, 로더(사전샤딩된 랭크 파일의 범위 읽기), KV 블록/슬롯, NVMe 티어, 스케줄러,
                스텝 메타, 러너, 기록/사망 덤프, 설정(사실+만료 노브), 증명·판정, 그래프, comm(플릿 / LocalTP)
    modules/    특징 모듈: 선형 어텐션(KDA), 희소 인덱서·희소 MLA, NVFP4 선형·MoE·양자화, 하이퍼커넥션, 노름, 회전, 로짓
    profiles/   모델별: 사실·가중치 지도(specs)·사전샤딩·레인 표·조합(net)·검증(check). glm53 이 첫 대상.
    kernels/    ST가 소유하는 Triton·TileLang·CuTe DSL·CUDA 커널과 필요한 보조 코드

빠른 확인(GLM-5.3, 실가중치, 한 노드, TP=4 스레드; 랭크 파일은 `profiles/glm53/preshard.py` 가 한 번 자른다):

    PYTHONPATH=. python3 engine/profiles/glm53/check.py --layers 0-4              # 참조 레인: 랭크 동일 + 청크/verify/롤백 판정
    bash engine/runtime/build.sh                                                # vLLM이 제거된 ST 이미지
    bash probes/run_engine_check.sh --layers 0-4                                  # ST 서빙 커널 레인
    PYTHONPATH=. python3 engine/profiles/glm53/boot.py --local --layers 0-4       # 러너가 돈다 (+ --drafter DFlash2, --park NVMe 파킹, --serve HTTP 문)

플릿(스파크 4대, 각 노드에 ST 이미지를 빌드한 뒤 노드당 컨테이너):

    bash launchers/fanout-st-ranks.sh            # 랭크 r 파일을 노드 r 로
    bash launchers/start-st-glm53.sh             # 부팅; glm53*/q38* 컨테이너가 있으면 거부
    curl -s http://10.10.10.2:8000/v1/completions -d '{"prompt": "...", "max_tokens": 64}'
    curl -s http://10.10.10.2:8000/v1/completions -d '{"conversation": 0, "prompt": "...", "max_tokens": 64}'   # 파킹된 대화 이어가기

GLM의 `served()`는 `engine/kernels`를 직접 호출한다. KDA·conv·mHC·kpool·MLA·b12x는
이 패키지 안에 있고, 인덱서와 mHC prenorm GEMM은 독립 `deep_gemm` 라이브러리를 사용한다.
vLLM 설치나 overlay 마운트는 필요하지 않다. 이식 출처, 라이브러리 버전과 빌드 계약은
[`kernels/README.md`](kernels/README.md), [`runtime/dependencies.json`](runtime/dependencies.json)에 있다.

가중치 없이 전체 커널 패키지와 GPU 수치 계약을 검사한다:

    ST_PROBE_NO_GPU=1 bash probes/run_engine_probe.sh probes/engine_kernel_check.py --imports-only
    bash probes/run_engine_probe.sh probes/engine_kernel_check.py

검사는 vLLM 임포트를 차단한다. GPU 검사는 KDA의 시작 상태·매 토큰 상태, conv 상태,
mHC pre/post, 유효 인덱서 로짓, kpool 바이트, MLA 부팅 판정·그래프 재생, b12x 출력을 확인한다.
b12x는 이식 전 FlashInfer 커널과 직접 비교하며, PyTorch 참조와 남은 오차는 별도로
기록한다. `--lanes moe --moe-experts 288`은 실제 TP4 전문가 형상을 검사한다.
커널 수치 검사는 실제 모델의 품질·처리량·ITL 판정과 별개다.

**이전 GLM 랭크 파일은 사전 샤딩을 다시 실행해야 한다.** b12x가 읽는 routed FC1은
`up | gate` 순서다. packed 가중치와 접힌 스케일을 이 순서로 저장하며, 파일 메타데이터의
`weight_layout=st-glm53-b12x-up-gate-v1`을 부팅·실가중치 검사에서 아레나 할당 전에 확인한다.
이전 `gate | up` 파일을 그대로 읽어 다른 모델을 실행하는 일은 허용하지 않는다.

커널 이식 검증은
[`st_engine_native_kernels_20260911`](../measurements/st_engine_native_kernels_20260911/README.md)에 있다.
이전 측정과 판정은 `MEASUREMENTS.md` 44~45차.

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

`Glm53Engine`은 전체 행의 temperature가 0인 스텝에서 유효 어휘 뷰의 argmax만 실행한다.
이 스텝은 RNG를 소비하지 않는다. 확률·혼합 스텝은 기존 base sampler를 사용하며, 같은
시작 RNG 상태에서 토큰과 종료 RNG 상태를 보존한다. 이전 버전의 greedy 스텝은 버릴 난수도
소비했으므로, greedy 이후 확률 생성까지 포함한 버전 간 출력 일치는 보장하지 않는다.
디코드 입력은 모든 시퀀스와 draft를 평탄화해 한 번 업로드하고, 생성 한도 판정은 토큰 버퍼의
길이 차이로 계산한다. 결과를 수거할 때만 생성 이력의 독립 사본을 만든다.
구성요소 전후 측정과 회귀검사는
[`measurements/st_engine_decode_20260911`](../measurements/st_engine_decode_20260911/README.md)에 있다.

HTTP 요청 번호는 내부 KV 행 번호와 분리한다. `Server`는 기본 64개의 미완료·미수거 요청까지
보관하고, 빈 행과 각 요청의 최대 생성 길이를 담을 블록 예산이 있을 때 FIFO 순서로 입장시킨다.
완료 결과를 복사한 뒤 일반 요청의 버퍼와 행을 반납한다. `keep_idle` 모드에서는 대화 ID와
요청 ID를 분리해 문맥을 보존하고, 새 요청에 공간이 필요하면 가장 오래된 유휴 대화를 정리한다.
이어가기 요청은 전체 문맥 예산을 확인한 뒤 NVMe에서 복원하며, 알 수 없거나 실행 중·정리된
대화는 기존 요청을 건드리지 않고 409로 응답한다. 누적 요청 수는 KV 행 수에 제한되지 않는다.
잘못된 입력은 400, 대기열 초과와 종료된 엔진은 503으로 응답한다. 종료 신호는 모든 랭크로
전달하며, 실행·대기 중 요청의 자원을 정리하고 기다리는 HTTP 호출을 깨운다.

`NvmeTier`는 완성된 새 파일의 이름을 manifest에 원자적으로 게시한다. 이전 세대는 그때까지
보존하며, 삭제 실패는 manifest의 `retired` 또는 `deleting` 기록으로 남긴다. 재시작 후 또는
디코드 경로 밖에서 `tier.cleanup()`을 호출하면 미완료 삭제와 게시되지 않은 세대 파일을 정리한다.
`deleting` 상태는 재승격할 수 없으며, 같은 저장 디렉터리는 하나의 `NvmeTier`가 소유한다.

`Glm53Caches`는 하나의 아레나에 다음 영역을 선언한다:

- 물리 KV 블록마다 모든 DSA 층의 fp8 latent와 압축 키·스케일을 함께 저장한다.
  블록 전체가 NVMe 전송 단위이며, 레인에는 층별 절대 슬롯 주소를 전달한다.
- 시퀀스 상태 슬롯마다 KDA conv·recurrent 링과 인덱서 꼬리 링을 둔다.
  인덱서 링은 `kpool - 1 + spec_k`개 위치를 보존해 드래프트 거절 후에도 앞선 풀을 복원한다.
  긴 프리필은 최신 창만 한 번씩 기록하여 CUDA의 중복 scatter 순서에 의존하지 않는다.
- 블록표도 아레나에 상주하며 `prepare(step)`이 매번 예약 문맥과 상태 슬롯 소유권을 확인한다.
  블록 매핑이 같으면 전송을 생략하고, 늘어나면 새 구간만 게시한다. `BlockPool.release`가
  행의 세대 번호를 바꾸므로, 행 재사용이나 NVMe 복원은 블록 개수가 같아도 다시 게시한다.
  이전보다 짧은 행의 남은 항목은 같은 전송에서 `-1`로 지우며, 캐시 전체 reset도 게시 상태를 초기화한다.

`BlockPool.table`, `row(seq)`, `epochs`는 읽기 전용 뷰다. 매핑 변경은 `reserve*`와 `release`를
통해야 하며, 이를 통해 같은 세대의 블록표가 뒤에만 늘어난다는 조건을 지킨다. 2차 최적화의
회귀검사와 구성요소 측정은
[`measurements/st_engine_cache_20260911`](../measurements/st_engine_cache_20260911/README.md)에 있다.

사전샤딩은 `RankWriter`가 모든 실제 가중치의 데이터 오프셋을 256바이트 경계에 맞춘다.
작은 스케일 뒤의 행렬도 TMA 정렬을 유지하도록 safetensors의 명시적 U8 패딩 텐서를 사용한다.
표준 safetensors 리더로 읽을 수 있고, 로더는 패딩을 포함한 연속 범위를 한 번 업로드한 뒤 뷰를 만든다.
**기존 PR #532 랭크 파일을 서빙 레인에서 사용할 때는 수정된 preshard로 다시 생성해야 한다.**
로딩 시 잘린 파일과 진행하지 않는 쓰기는 즉시 오류가 되며 무한 재시도하지 않는다.

서빙 KDA 어댑터는 엔진의 `[H,K,V]` 상태와 커널의 `[H,V,K]` 상태를 경계에서 변환한다.
recurrent 레인도 연결되어 검증 토큰마다 상태를 반환하고, 시작 상태를 보존한다.
MLA 참조 레인은 선택된 fp8 행만 변환하며, 패딩 슬롯이 가리키는 미사용 블록의 NaN을 마스킹한다.

인덱서의 Hadamard-128 변환·FP8 양자화와 pool→토큰 확장은 `indexer_quant`, `expand_pools`
레인으로 실행한다. 서빙 표는 `engine.kernels.kpool`의 융합 Triton 커널에 연결하고, 참조 표는 기존 PyTorch
수식을 유지한다. 두 레인도 LocalTP의 메인 스레드 경유 규칙을 따르며, 바인딩이나 실행 실패는
그대로 전파한다. 실제 가중치 인덱서의 결과·캐시 일치와 구성요소 성능은
[`measurements/st_engine_indexer_20260911`](../measurements/st_engine_indexer_20260911/README.md)에 기록했다.

선택 토큰의 최종 주소 변환은 `indexer_slots` 레인이 담당한다. PyTorch의 내림차순 정렬을
유지하고, 유효 개수 집계·블록 주소 변환·패딩·출력 쓰기를 한 Triton 커널로 실행한다.
캐시는 `token_map(layer, seq)`로 블록 행과 잠재 벡터 행 단위의 블록 크기·간격·레이어 오프셋을
제공한다. 연속 캐시 검사의 `None` 블록 행은 위치와 슬롯이 같은 매핑이다. 모든 출력 칸을
덮어쓰므로 별도의 초기화 커널이 필요 없고, LocalTP와 오류 전파 규칙은 다른 레인과 같다.
검증과 변경 전후 측정은
[`measurements/st_engine_slots_20260911`](../measurements/st_engine_slots_20260911/README.md)에 있다.

네 노드 검증은 각 노드에서 같은 인자로 `check.py --distributed`를 실행한다.
기본 노드 순서는 **rank 0=srv2, rank 1=srv1, rank 2=srv3, rank 3=srv4**다.
`MASTER_ADDR`은 rank 0 서버를 가리켜야 한다. `RANK`, `WORLD_SIZE=4`, 격리된 `MASTER_PORT`,
RoCE 인터페이스·GID는 실행기가 설정하고, `--ranks`에는 해당 랭크의 정렬된 파일을 둔다.
`probes/engine_comm_check.py`로 가중치 없이 통신을 먼저 검증할 수 있다.

추가 장치 검사:

    bash probes/run_engine_probe.sh probes/engine_kda_check.py
    python3 probes/engine_cuda_io_check.py

검사 결과와 실제 사용한 네 노드 실행기는
[`measurements/st_engine_runtime_20260911`](../measurements/st_engine_runtime_20260911/README.md)에 보관한다.
이는 2개 실제 층(KDA+dense, DSA+NVFP4 MoE)의 실행·캐시 계약 검사다. 통합 전 로그는
참조 전문가를 사용했고, PR #534 통합 후 `check.py --lanes served`는 b12x 전문가를 포함한다.
전체 45층 onepass 품질, 실제 DFlash2 수용률, 그래프 기반 요청 실행 및 처리량·ITL 판정은 별도다.
