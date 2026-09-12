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
    bash bench/fleet.sh run --gpu st 30 "decode graph" -- bash probes/run_engine_probe.sh probes/engine_decode_graph_check.py
                                                 # GPU 검사는 벤치 큐에 줄을 선다(2026-09-12): 창이 없으면 미루지 말고 예약한다.
                                                 # 큐는 st-* 컨테이너가 떠 있으면 허가하지 않고, 런처는 큐에 holder 가 있으면 거부한다.
    curl -s http://10.10.10.2:8000/v1/engine/completions -d '{"prompt": "...", "max_tokens": 64}'                    # 엔진 방언: ids/text
    curl -s http://10.10.10.2:8000/v1/engine/completions -d '{"conversation": 0, "prompt": "...", "max_tokens": 64}'   # 파킹된 대화 이어가기
    curl -s http://10.10.10.2:8000/v1/completions -d '{"prompt": "...", "max_tokens": 64, "n": 2, "logprobs": 3}'    # OpenAI completions
    curl -s http://10.10.10.2:8000/tokenize -d '{"prompt": "..."}'; curl -s http://10.10.10.2:8000/detokenize -d '{"tokens": [1, 2]}'
    STK_moe_static=t,r,sf6 STK_context_ceiling=131072 bash launchers/start-st-glm53.sh   # 선언된 D11 노브는 STK_* 로 부팅에 들어간다(미선언·만료 = 사망)
    bash launchers/start-st-glm53.sh stop        # 컨테이너 제거 + 잠금 해제. start 는 glm53*/q38*/vllm*/st-* 컨테이너나 srv2 의 `st-fleet.lock` 이 있으면 거부한다
                                                 # (플릿을 쓰는 세션은 모두 이 잠금을 지킨다: 09-11 19:42 두 세션의 플릿이 같은 노드에서 충돌해 둘 다 죽었다)
    curl -s http://10.10.10.2:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"..."}],"max_tokens":64,"stream":true}'   # OpenAI 방언(SSE), bench/onepass.py 가 쓰는 것
    curl -s http://10.10.10.2:8000/v1/models; curl -s http://10.10.10.2:8000/metrics                                  # 모델 이름, 벤치 이름의 카운터

`/metrics`(프로메테우스 텍스트, HELP·TYPE 포함): 벤치 방언(`vllm:request_success_total`·`num_requests_{running,waiting}`·`prompt/generation_tokens_total`·`spec_decode_*`·`iteration_tokens_total_count`)은 이름과 의미 그대로 유지하고, 그 위에 **지연 히스토그램 셋**(`vllm:time_to_first_token_seconds`·`time_per_output_token_seconds`·`e2e_request_latency_seconds`, 요청 도착 시각 기준), **포화도**(`vllm:gpu_cache_usage_perc`·`st:kv_blocks_{total,used,free}`·`st:state_slots_{total,free}`), **재사용**(`vllm:prefix_cache_{queries,hits}_total`·`st:prefix_cache_*`), **스텝 종류**(`st:steps_{prefill,decode}_total`, D9), **티어**(`st:conversations_parked`·`st:tier_bytes_*`), **취소·타임아웃**(`st:requests_{cancelled,timed_out}_total`)을 낸다.
vLLM 이 낼 수 없는 것(이 엔진에만 있는 부품이라): **어느 캡처 그래프가 돌았나**(`st:decode_steps_by_sequences_total{sequences}` = 스케줄러가 실제로 채운 배치, `st:decode_capacity_bucket_total{capacity}` = `STK_context_ceiling` 을 자를 유일한 프로덕션 증거), **스텝 벽시계**(`st:step_seconds{kind}`, 호스트 관측 종단 — 두 종류 모두 샘플 읽기로 끝나므로 발사 시간이 아니라 스텝 전체다), **수용 분포**(`st:spec_accepted_per_step_total{accepted}` — 평균이 아니라 모양이 `spec_k` 를 정한다), **무엇이 실제로 묶였나**(`st:lane_info{lanes,moe_static,mla_prefill,spec_k,context_ceiling}` — "무장 ≠ 서빙"을 부팅 로그가 아니라 스크레이프로 판정).
비용(실측): 렌더 0.096 ms·11 KB·190줄(스크레이프당 1회), 관측 0.96 µs(디코드 스텝 최악 24회 = 46 ms 스텝의 0.05%). 디바이스 읽기·동기화 없음.

문(`base/serve.py`): 엔진 방언(`POST /v1/completions` ids|prompt, `conversation` 으로 이어가기)과 OpenAI chat 방언(`POST /v1/chat/completions`,
`stream` 이면 토큰 단위 SSE, `chat_template_kwargs` 통과, `</think>` 앞은 `reasoning_content` 뒤는 `content`; `GET /v1/models`, `/metrics`, `/health`).
프로필이 템플릿(`chat_template_mm_v2.jinja`, 프로덕션과 같은 것)과 `</think>` id 를 넘긴다.
요청은 `stop`(문자열 ≤4, 내용 채널에서 잘라 조기 종료), `min_tokens`(그 전엔 끝 토큰 불가), `tools`(템플릿이 렌더, 답의
`<tool_call>` 은 `tool_calls` 로 파싱, finish `tool_calls`), `n`=1 만, `logprobs` 는 400. 클라이언트가 끊으면(소켓 EOF·broken pipe) 요청을
취소해 행을 돌려주고, `REQUEST_TIMEOUT_S`(3600) 를 넘긴 요청은 504 로 취소한다 — 취소는 rank 0 이 도착과 같은 브로드캐스트로 실어 네 랭크가
같은 반복에서 같은 행을 버린다(`/metrics` 의 `st:requests_cancelled_total`).
그림(45차 §23 A7, `profiles/glm53/vision.py`): chat 의 `content` 배열에 `image_url`(≤4)·`video_url`(≤1, 프로덕션 PR #431 의 MM_LIMIT) 파트를
받는다 — `data:` 또는 http(s) URL. rank 0 의 문이 프로덕션 프로세서와 같은 규칙으로 캔버스를 만들고(28 정렬·pad·bicubic, 영상은 32 프레임
균등 → GLM 샘플러 → 프레임 쌍마다 `<|begin_of_image|>…<|end_of_image|>N.N seconds`), 템플릿의 자리표시자 하나를 토큰 수만큼 늘려 보낸다;
캔버스는 요청과 함께 네 랭크로 가고 각 랭크가 같은 비전 타워(`vision.safetensors`, 랭크 파일 옆; `preshard.py --vision` 으로 한 번)를 돌려
자리표시자 위치의 임베딩을 바꿔 넣는다(D3: 네 랭크의 결과 합이 다르면 죽는다). 같은 자리표시자에 다른 그림은 다른 프롬프트다 — 이어가기(B1)와
prefix 캐시 둘 다 그림의 digest 를 본다. 플릿 부팅은 `vision.safetensors` 가 없으면 서지 않는다(프로덕션이 그림을 서빙하므로).
prefix 재사용의 단위는 풀의 **블록 768**(`facts.BLOCK`; 프리필 청크의 정렬 2,304 = `facts.CHUNK_ALIGN` 는 그대로, 6,912 법칙 불변)이다
(`base/prefix.py`): 청크(6,912) 안의 여덟 경계는 스텝 전에 `marks` 로 이름 붙여 스텝이 도는 동안 받는다 — KDA 되풀이를 그 자리에서
끊어(정확: 커널 청크가 64 정렬 경계에서 다시 시작) 조각의 끝 상태와 conv 입력을 스냅샷에 쓰고, 드래프터 링은 스텝 전 링 + 경계 전
위치를 스냅샷에 관측한다(`adapter.prefill`). **생성 중에 넘은 경계**도 들어간다: 동기 스텝 뒤엔 링에서, 앞서 도는 스텝은 장치의
"경계 스테이지"(슬롯당 KDA 상태 + conv 탭, `caches.stage_boundaries`; 넘은 스텝만 쓴다)에 놓아 두고 호스트가 결과를 읽을 때
스냅샷으로 옮긴다(드래프터 링은 그때의 산 링: 넘은 뒤 몇 자리가 창의 가장 오래된 칸에 얹힐 뿐). 스냅샷 96개(경계당 ~77 MiB,
`boot.PREFIX_SNAPSHOTS`); 자리가 모자라면 **한 번도 채택되지 않은 경계부터** 나간다(`prefix._victim`) — 긴 프롬프트 하나가 모두가
공유하는 시스템 프롬프트를 밀어내지 못한다. 이어가기(B1)는 히스토리가 끝 토큰(`<|endoftext|>` 등, 템플릿이 되그리지 않는)으로 끝났으면
그 토큰 앞까지 맞아도 이어간다 — 그 토큰은 뽑혔지만 먹인 적이 없어 캐시가 정확히 그 앞에 서 있다(`extend(drop_unfed=True)`). 디코드는
**호스트보다 앞서 돈다**(`profiles/glm53/pipeline.py`, vLLM 의 비동기 스케줄링): 타깃
그래프 → 샘플러 → 커밋(`base/sampler.commit_batch`) → 마스크 관측 → 제안 → 다음 스텝 ids 가 장치에 남고, 결과만 핀 버퍼로 건너와
다음 스텝이 이미 도는 동안 읽힌다(`runner.inflight`, 깊이 2). 장치에서 끝난 행은 상태 슬롯을 null 슬롯으로 돌려 유령 스텝이 링에 아무
것도 못 쓰고, 러너는 한 스텝 늦게 끝을 알아 그 유령의 결과를 버린다. 온도/top_p 만 있는 행은 장치의 기각 샘플링
(`speculative_pick_batch`)으로 앞서 돌고, 페널티·logit_bias·seed·logprobs·문법·min_tokens 미충족 행은 동기 경로(러너가 먼저 비운다).
루프의 도착 브로드캐스트와 투표는 gloo 제어 그룹(`Comm.control`)으로 간다 — NCCL 그룹의 객체 브로드캐스트는 스텝의 커널 뒤에
줄 서고 읽기 위해 장치를 기다린다.

운영(`launchers/st-glm53-supervisor.sh` + `st-glm53.service`, 헤드 srv2 의 사용자 유닛): 30 s 마다 진짜 4 토큰 chat 으로 건강을 재고(문이
열려 있어도 링은 죽어 있을 수 있다), 3 회 연속 실패면 포렌식(네 랭크 로그·free·nvidia-smi·metrics → `~/glm53-logs/st-forensics/`) → stop →
start. 재시작 간격은 60 s 부터 두 배씩 30 분까지, 5 회 실패 뒤엔 멈추고 사람을 부른다. 프로덕션 vLLM·q38 컨테이너가 보이면 절대 띄우지 않는다.
`ST_SUPERVISOR_ONCE=1` 로 한 사이클만 판정할 수 있다. 한 노드는 **자기 자신에게 ssh 하지 못하므로**(srv2 가 자기 키를 거부한다) 런처와
슈퍼바이저는 대상 IP 가 자기 것이면 로컬 셸로 돌린다 — 그래서 헤드에서 도는 슈퍼바이저가 rank 0 의 컨테이너·로그·잠금을 본다.

프로덕션 전환: 프로덕션 vLLM 을 되살리는 경로는 `fleet-idle-recovery.timer`(5 분 유휴 뒤 복구) 하나뿐이다. ST 가 프로덕션이 되는 동안은
그 타이머를 끄고(`st-glm53.service` 의 `Conflicts=`가 같은 일을 한다) 슈퍼바이저 유닛을 켠다. 되돌리기는 그 반대 순서다:

    systemctl --user disable --now st-glm53          # (헤드)
    bash launchers/start-st-glm53.sh stop            # 네 노드 컨테이너 + 잠금 해제
    systemctl --user start fleet-idle-recovery.timer # 5 분 유휴 뒤 vLLM 복귀

Prefix 재사용(`base/prefix.py`): 프롬프트를 프리필 청크(6,912 = 블록 3개) 단위로 해시 사슬을 만들고, 청크 경계마다 모델의 위치 링 상태
(KDA conv 탭 3개 + 재귀 상태 1개 × 34층, 드래프터 문맥 링; 인덱서 꼬리는 경계에서 비어 있어 제외)를 아레나의 스냅샷 슬롯(`PREFIX_SNAPSHOTS`
= 8, 랭크당 ~77 MiB 씩)에 두고 그 앞 블록들을 고정한다. 새 프롬프트는 자기 길이보다 짧은 가장 긴 캐시 경계를 **입양**(읽기 전용 공유
블록 + 슬롯에 상태 복원)하고 거기서부터 프리필한다 — 토큰 하나는 반드시 계산한다. 블록 소유는 개수(행 참조 + 캐시 핀)로, 예약이 모자라면
풀이 캐시에 LRU 회수를 요청한다. 랭크 넷이 같은 승인을 같은 순서로 하므로 메시지 없이 같은 캐시 상태다. 판정: `tests/test_engine_prefix.py`
(가짜 모델), GPU 는 `probes/engine_prefix_check.py`(같은 두 청크를 공유하는 두 프롬프트가 같은 토큰을 내야 한다).

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

실행 소유권은 커널 표 → LocalTP → 개별 `run` 순서로 명시한다. `lanes.served(tp=tp)`가
만든 표는 해당 실행기에 고정되며, 다른 표의 생성이나 실행이 이 연결을 바꾸지 않는다.
플릿과 직접 워밍업은 `lanes.served()`로 호출한다. 전역 `bind_tp` 상태는 없다.

- 각 `run`이 배리어·통신 버퍼·결과·커널 대기열을 새로 소유하고, 끝날 때 참조를 반납한다.
- 같은 LocalTP의 중첩·중복 실행, 외부 스레드의 디스패치, 종료된 랭크 핸들의 재사용은 즉시 실패한다.
- 독립 LocalTP 둘은 각자의 호출 스레드에서 커널을 실행하며 통신 상태를 공유하지 않는다.
- 랭크·커널·스레드 시작 실패 시 대기 커널을 취소하고 시작된 랭크를 합류시킨 뒤 원인을 전달한다.
  다음 명시적 실행은 새 제어 상태를 받는다. 모델/KV나 CUDA 문맥의 복구·자동 재시도는 별도 책임이다.

부팅과 체인 검사도 이 소유권 경계를 따른다. 검증은
[`measurements/st_engine_execution_20260911`](../measurements/st_engine_execution_20260911/README.md)에 있다.

요청과 캐시의 소유권은 다음 경계에서 확정한다:

- `Runner.submit`은 입력 검증 → KV·상태 슬롯 예약 → `Model.open` → 스케줄러 등록 순서다.
  중간 실패 시 확보한 자원을 반납한다. `Model.close`는 실패한 `open`의 부분 상태도 정리해야 한다.
- 디코드는 `BlockPool.reserve_to`로 요청별 쓰기 끝 위치를 받아 배치 전체의 증가분을 먼저 검증·예약한다. 자원이 부족하면
  어느 행의 토큰 수도 바뀌지 않는다. 거절된 드래프트가 쓰던 공간은 다시 예약하지 않는다.
  블록 풀과 러너 상태는 러너를 소유한 스레드에서 갱신한다.
- 대기 상한은 디코드 폭에 여유가 있을 때 프리필을 허용한다. `max_running`이 꽉 차면 기존 요청이
  끝날 때까지 기다린다. 이미 실행 폭을 넘긴 상태는 오류이며, 앞쪽 요청만 잘라 실행하지 않는다.
  입장한 긴 프리필은 청크 사이에 실행 중인 디코드를 한 번씩 처리한다. 프리필 중인 요청의
  디코드 자리도 예약으로 취급하므로 유휴 대화 깨우기가 그 자리를 가져갈 수 없다.
- 파킹의 단위는 **대화**이지 행이 아니다. `Runner.park_begin(row, key)`는 유휴 행의 모델 호스트 기록(`Model.park`: 토큰 이력·문맥,
  `"context"`·`"pending"` 필수)을 떼고 블록과 상태 슬롯 바이트를 티어 스레드에 넘긴다; 행은 `retiring`(유휴도 빈 것도 아님)이고
  `park_finish(row)`가 쓰기 완료 뒤 **행과 슬롯을 반납**한다. `resume_begin(row, key)`는 빈 행 아무 곳에 블록을 예약하고 빈 슬롯
  아무 곳을 잡아 읽기를 넘기고, `resume_finish(row)`가 커밋한다(`Model.resume`은 슬롯을 지우지 않는다: 바이트는 티어가 되돌렸다).
  `transfer_done(row)`는 기다리지 않는다(D10). 쓰기 실패는 행을 유휴·상주로 되돌리고, 읽기 실패는 받은 행·슬롯·블록을 반납하고
  디스크 사본을 보존한다. `park`/`resume`은 두 반쪽을 이어 붙인 것(기다려도 되는 호출자용). 보존 대화 수의 상한은 `max_seqs`가 아니라 디스크다.
- `NvmeTier`는 파일 하나에 [블록들][슬롯 바이트, 섹터 패딩]을, 옆에 JSON 기록을 두고 manifest 에 셋을 게시한다. 파일시스템
  여유(기본 1 GiB 예비)나 선언한 `capacity_bytes`를 넘길 demote 는 한 바이트도 쓰기 전에 `TierFull`을 낸다.
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

`Glm53Engine`은 전체 행의 temperature가 0인 스텝에서 rank별 유효 어휘 최댓값과 토큰 ID를
하나의 int64 후보로 부호화하고 NCCL MAX로 선택한다. 전체 로짓을 모으지 않으며, 동점은
가장 작은 전역 토큰 ID로 결정한다. 타깃 그래프는 rank별 로짓만 보관하고, 확률·혼합
샘플링 그래프가 필요할 때 전체 로짓을 모은다. DFlash의 후보 top-k는 기존 경로다.
이 스텝은 RNG를 소비하지 않는다. 확률·혼합 스텝은 기존 base sampler를 사용하며, 같은
시작 RNG 상태에서 토큰과 종료 RNG 상태를 보존한다. 이전 버전의 greedy 스텝은 버릴 난수도
소비했으므로, greedy 이후 확률 생성까지 포함한 버전 간 출력 일치는 보장하지 않는다.
디코드 입력은 모든 시퀀스와 draft를 평탄화해 한 번 업로드하고, 생성 한도 판정은 토큰 버퍼의
길이 차이로 계산한다. 결과를 수거할 때만 생성 이력의 독립 사본을 만든다.
구성요소 전후 측정과 회귀검사는
[`measurements/st_engine_decode_20260911`](../measurements/st_engine_decode_20260911/README.md)에 있다.

캡처 그래프는 커널 작업 버퍼의 **소유자도 보관**한다. MoE의 eager 캐시는 더 큰 배치나
프리필을 만날 때 버퍼를 교체하므로, 작은 배치 그래프가 참조하는 이전 버퍼를 그 캐시에만
맡길 수 없다. `DecodeGraphs(resources=...)`는 매 캡처 직후 소유자를 보관하고 모든 그래프를
reset한 뒤 반납한다. 전체 모델의 첫 디코드 정지 재현과 수정 검증은
[`engine_decode_replay_20260912`](../measurements/engine_decode_replay_20260912/README.md)에 있다.

HTTP 요청 번호는 내부 KV 행 번호와 분리한다. `Server`는 기본 64개의 미완료·미수거 요청까지
보관하고, 빈 행과 각 요청의 최대 생성 길이를 담을 블록 예산이 있을 때 FIFO 순서로 입장시킨다.
완료 결과를 복사한 뒤 일반 요청의 버퍼와 행을 반납한다. `keep_idle` 모드에서는 대화 ID와
요청 ID를 분리해 문맥을 보존한다. 티어가 없으면 유휴 대화가 행에 상주하고 새 요청에 공간이 필요하면
가장 오래된 유휴 대화를 정리한다. 티어가 있으면 끝난 턴의 파킹과 이어가기의 복원은 **티어 스레드**에서 돌고, 스텝 루프는
매 스텝 완료 여부만 묻는다(디코더는 디스크를 기다리지 않는다, D10). 완료·성패는 `all_reduce` 한 번으로 **네 랭크가 합의**한 뒤에
적용한다(락스텝): 모두 성공이면 행이 비거나 턴이 입장하고, 한 랭크라도 실패면 그 대화는 모든 랭크에서 버린다(복원 실패는 503),
모두 `TierFull` 이면 가장 오래 전에 파킹된 대화부터 잊고 다시 쓴다(잊을 것이 없으면 보존하지 않음). 파킹이 진행 중인 대화의
이어가기는 그 파킹이 끝날 때까지 줄에서 기다린다. 대화 ID는 부팅을 넘겨 유효하다: 새 서버의 요청 번호는 티어에 남은 가장 큰
대화 ID 위에서 시작하고, 엔진이 죽어도 파킹된 대화는 디스크에 남는다(정지 때 진행 중이던 전송은 기다려서 마무리한다).
알 수 없거나 실행 중·정리된 대화는 기존 요청을 건드리지 않고 409로 응답하고, 모델의 학습 위치(`facts.max_position`)를 넘는
문맥은 400 이다. 누적 요청 수와 보존 대화 수는 KV 행 수에 제한되지 않는다.
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

예산은 `profiles/glm53/budget.py`가 선언한다(D1): OS 예비 2×5%, 런타임 바닥(원장), 가중치·드래프터(랭크 파일), 상태 슬롯·prefix 스냅샷(레이아웃),
아레나 밖 작업공간 상한 12 GiB(`base/runtime_memory`가 강제; 부팅 원장 `memory-rankN.json`을 주면 실측 피크를 줄에 적는다),
NVMe 스테이징 — 남는 것이 KV 자리이고, 전체 모델 부팅은 rank 0 에서 그 표와 "선언한 KV 가 남기는 양"을 찍는다
(`python3 engine/profiles/glm53/budget.py [--ledger …]`). 프리필 인덱서 선택은 질의 1,024 행씩 나눠(`net.SELECT_ROWS`) 로짓 과도를
행×후보×4 B 로 묶고, 디코드 그래프의 용량 사다리는 `facts.max_position`에서 끝난다.

로더의 읽기는 **O_DIRECT** 다(티어가 자기 읽기에 대해 대는 것과 같은 이유: 페이지 캐시는 이 상자에서 아레나와 같은 풀이다).
44.5 GiB 랭크 파일을 버퍼드로 읽으면 엔진이 드라이버에 아레나를 달라고 하는 바로 그 순간 44.5 GiB 의 클린 페이지가 남는다 —
첫 45층 부팅의 55.4 GiB 할당 실패에 연루된 압력이다. 런 하나는 자기를 담는 섹터 창으로 읽고 런은 그 안의 슬라이스이며,
스테이징 버퍼는 페이지 정렬(CUDA 대상이면 핀)이다. O_DIRECT 를 거부하는 파일시스템(overlayfs·tmpfs)에서는 버퍼드로 내려가고
그 경로만 `fadvise` 를 쓴다. 사전샤딩이 체크포인트를 읽는 쪽(`base/checkpoint`)도 같은 범위 읽기를 쓴다 — 텐서마다
`safe_open` 하던 경로가 아니다. 실측(srv4, 랭크 파일 4.14 GiB, cpu 대상, 3회): **O_DIRECT 1.92~2.09 vs 버퍼드 0.70~0.88 GiB/s**;
체크포인트 층 하나(4.08 GiB, 2,619 텐서)의 읽기는 **2.71 → 1.56 s**(mmap 은 게을러서 페이지를 만져야 공정하게 잰다).

아레나는 가상 범위 하나를 20 MiB 물리 조각으로 매핑한다(`expandable_segments`, 이 상자가 vLLM 을 띄우는
방식). 첫 45층 부팅은 55.4 GiB `cudaMalloc` 한 번에서 로드 전에 OOM 이었다. 입장 관문 `prepare_allocation`은 랭크·
드래프터 파일의 캐시 페이지를 버리고, 그래도 `MemFree` 가 아레나 + 16 GiB 에 못 미치면 `MemAvailable` 이 허락하는 만큼
익명 페이지를 잠깐 잡았다 놓아(`touch_pages`) 캐시를 회수한 뒤 다시 잰다. 회수한 양과 아레나 매핑 방식은 부팅 표에
`boot_reclaimed_GiB`·`arena_expandable` 로 찍힌다. 할당 모양의 실측은 `probes/engine_alloc_shape_check.py`.

사전샤딩은 `RankWriter`가 모든 실제 가중치의 데이터 오프셋을 256바이트 경계에 맞춘다.
작은 스케일 뒤의 행렬도 TMA 정렬을 유지하도록 safetensors의 명시적 U8 패딩 텐서를 사용한다.
표준 safetensors 리더로 읽을 수 있고, 로더는 패딩을 포함한 연속 범위를 한 번 업로드한 뒤 뷰를 만든다.
**기존 PR #532 랭크 파일을 서빙 레인에서 사용할 때는 수정된 preshard로 다시 생성해야 한다.**
로딩 시 잘린 파일과 진행하지 않는 쓰기는 즉시 오류가 되며 무한 재시도하지 않는다.

서빙 KDA 어댑터는 엔진의 `[H,K,V]` 상태와 커널의 `[H,V,K]` 상태를 경계에서 변환한다.
recurrent 레인도 연결되어 검증 토큰마다 상태를 반환하고, 시작 상태를 보존한다.
MLA 참조 레인은 선택된 fp8 행만 변환하며, 패딩 슬롯이 가리키는 미사용 블록의 NaN을 마스킹한다.

인덱서의 Hadamard-128 변환·FP8 양자화는 `indexer_quant` 레인으로 실행한다.
키 풀링은 `kpool_compress` 레인이 `compress_pool_keys`를 직접 호출한다. GB10에서 128차원
풀 하나를 1워프로 처리하며, XOR 짝을 이용한 Hadamard 회전은 공유 메모리를 사용하지 않는다.
입력 스트라이드를 직접 받아 복사 없이 읽고, 결과 FP8 키와 스케일만 할당한다. 반환 전용
경로에서 쓰지 않던 임시 캐시·위치 배열·쓰기 마스크를 만들지 않는다. BF16 반올림 경계와
FP8 양자화 수식은 유지한다. 정확성 및 성능 근거는
[`measurements/st_engine_warp_pooling_20260911`](../measurements/st_engine_warp_pooling_20260911/README.md)에 있다.

선택한 풀의 최종 주소 생성은 `pool_slots` 레인 하나가 맡는다. GB10에서 토큰 2,051개를
먼저 펼쳐 정렬하던 경로를, 풀 ID 512개를 정렬한 뒤 토큰을 생성하는 구조로 바꿨다.
서빙 커널은 4워프 프로그램 하나가 한 행을 처리하며, 중간 GPU 텐서를 할당하지 않는다.

KV 블록은 풀 크기의 정수 배수여야 한다. 이 계약 덕분에 풀마다 블록 주소를 한 번 읽고
네 토큰의 주소를 레지스터에서 계산한다. 미완성 꼬리는 최신 토큰부터 앞에 배치하고,
중복 풀은 토큰별 중복 횟수를 유지하며, 패딩과 유효 개수까지 같은 커널에서 쓴다.
참조 레인은 기존 토큰 확장·정렬·매핑 수식을 유지해 정확한 정수 비교의 기준으로 쓴다.

캐시는 `token_map(layer, seq)`로 블록 행과 잠재 벡터 행 단위의 블록 크기·간격·레이어
오프셋을 제공한다. 연속 캐시 검사의 `None` 블록 행은 위치와 슬롯이 같은 매핑이다.
LocalTP 소유권과 오류 전파 규칙은 다른 레인과 같다. 이전 `expand_pools`, `indexer_slots`
서빙 레인은 제거했으며, 기존 함수는 구성요소 검사와 명시적인 과거 커밋 비교에만 사용한다.
검증 범위와 변경 전후 측정은
[`measurements/st_engine_pool_slots_20260911`](../measurements/st_engine_pool_slots_20260911/README.md)에 있다.

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


디코드 그래프 연결과 실가중치 수치 안정성 후속 작업은
[`measurements/st_engine_completion_20260911`](../measurements/st_engine_completion_20260911/README.md)에 기록했다.
TP4의 두 층 검사에서 일반 실행·그래프 출력과 캐시가 일치하며, MoE BF16 누적의 반복 편차를
FP32 합산으로 줄였다. 전체 모델 첫 부팅은 CUDA 아레나 할당 실패로 중단됐다. 따라서 전체
45층 품질·DFlash2 수용률·NVMe 간섭 ITL과 최종 플릿 릴리스는 아직 통과한 상태가 아니다.
독립 이미지의 버전·소스 식별 방법은 [`runtime/README.md`](runtime/README.md)를 따른다.

SM121a·TP4 후속 변경은 [`st_engine_four_optimizations_20260911`](../measurements/st_engine_four_optimizations_20260911/README.md)에 있다.
타깃 그래프는 물리 슬롯 ID를 Triton 상태 커널에 전달한다. KDA는 이전 recurrent 상태 하나와
conv 이력만 읽고 새 토큰 위치만 쓴다. 작은 인덱서 꼬리의 읽기 버퍼는 유지하며, 상태 링 전체의
gather/commit은 하지 않는다. 전체 모델 부팅은 별도의 작업공간 상한과 OS 여유를 선언하고,
최대 프리필·모든 그래프 형상의 준비 중 관측한 메모리 최대치를 저장한다.
