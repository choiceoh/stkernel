# ST 디코드 파이프라인 개선 근거

이 기록은 디코드 배치가 작게 유지되거나 새 요청이 들어올 때 앞선 디코드가 불필요하게 비워지는 경로를 줄인 변경의 검증 자료다. 프로덕션 기본값은 `MAX_SEQS=8`, `MAX_WAIT_S=0`, 드래프트 `K=5`가 있는 동안 `decode_token_budget=2,304`다. 빈 행에는 다음 프리필 경계에서 요청을 넣고, 디코더가 있는 동안에는 2,304토큰 프리필 청크와 디코드 스텝을 번갈아 실행한다. 프리필과 디코드는 여전히 한 스텝에 섞지 않는다.

## 구현 범위

- `AsyncDecode`가 생존 행의 컨텍스트·draft·분포를 장치에서 보존한 채 새 행을 합치고, 바뀐 행만 무효화한다.
- 러너의 취소·재사용·이어가기 수거는 해당 행을 참조하는 pending prefix까지만 기다린다.
- `decode_commit`이 수용 토큰, EOS, 생성 한도, 상태 슬롯과 null 슬롯 전환을 행 단위 Triton 커널에서 처리한다.
- `vocab_candidates`가 1,024개 부분 최댓값으로 greedy 후보를 만들며, 전체 FP32 vocabulary 임시 버퍼를 만들지 않는다.
- `Comm.all_gather`가 rank-major 출력 버퍼에 직접 수집하고, 1~64개 int64 후보의 TP4 MAX는 one-shot transport를 선택한다. 지원하지 않는 크기는 NCCL 경로를 유지한다.
- 실제 배치 폭에 맞춰 mHC Python 게이트, one-shot BF16/정수 self-test, GPTQ 보정기의 디코드 행 경계를 64행/48행까지 확장했다.

## 검증

`validation.json`은 재현 가능한 요약과 소스 해시를 담는다.

- CPU/가짜 모델 회귀: 커널 provenance, 스케줄러, 러너, pending merge, 행 재사용, stochastic join, comm layout, calibration을 포함한 73개 테스트가 통과했다. 2-process Gloo all-gather도 strided tensor와 음수 차원을 포함해 통과했다.
- 최신 `main`과의 통합 회귀: graph contract, prefix, serving, native execution, vocab, knobs, sampling과 budget을 포함한 266개 테스트가 통과했다(30개는 CUDA 또는 선택적 런타임 부재로 skip).
- Triton CPU interpreter: 커밋 576개, vocabulary 선택 52개에서 기존 참조 결과와 일치했다. NaN, signed zero, 동점, EOS, 생성 한도, 유령 행, vocabulary 경계를 포함한다.
- SM121 device compilation: `decode_commit` 8개(`K=0/1/5/8`, greedy/stochastic)와 vocabulary 4개 변형 등 12개 cubin을 생성했다. 이 단계는 장치를 실행하지 않는다.
- Torch/CUDA one-shot 확장: Torch `2.13.0+cu130`과 CUDA `13.0`으로 컴파일·로드를 통과했다. `MAX_ELEMENTS=262144`와 signed int64 MAX specialization을 포함한다. CUDA 장치 실행은 하지 않았다.
- 메모리 산수: `box=121.6 GiB`, `KV=12 GiB`, `PREFIX_SNAPSHOTS=96` 기준으로 4행과 8행을 다시 계산했다. 8행은 상태 슬롯 1.909 GiB, snapshot 4.237 GiB, 생성 경계 staging 0.309 GiB, paged KV 10.090 GiB다.

예시 명령:

```bash
/tmp/st-hw-d36a/compile-venv/bin/python -m unittest \
  tests.test_engine_kernels \
  tests.test_engine_runtime tests.test_engine_runner_async tests.test_engine_pipeline \
  tests.test_engine_comm_layout tests.test_engine_dense_calibration \
  tests.test_engine_oneshot_integer
python3 probes/engine_decode_fusion_host_check.py --output /tmp/st-throughput-host-20260912
```

## 해석 범위

이 자료는 correctness와 컴파일 근거다. 전체 GLM forward의 GPU latency, TP4 NCCL/one-shot 실측, 실제 트래픽에서의 batch-width 분포는 GPU 검증 큐에서 별도로 확인해야 한다. CUDA Graph를 하나의 persistent transformer kernel로 합친 변경은 아니며, mHC/KDA/MLA/MoE 경계와 row-parallel 의미상 필요한 collective는 남아 있다. 따라서 이 변경으로 vLLM과의 처리량 동률이나 특정 속도 향상을 주장하지 않는다.
