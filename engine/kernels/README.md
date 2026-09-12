# ST 커널 패키지

> 살아 있는 참조 — **서빙 레인이 어느 커널을 쓰는지. 레인이 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

`engine.profiles.glm53.lanes.served()`가 사용하는 실행 코드를 이 패키지가 소유한다.
vLLM의 임포트, `torch.ops.vllm` 등록, FlashInfer 패키지 내부로의 커널 마운트는 없다.
라이브러리·컴파일러가 없거나 MLA 수치 판정이 실패하면 오류를 전달한다.

| 레인 | ST 구현 | 외부 라이브러리 |
| --- | --- | --- |
| KDA chunk / recurrent | `kda/`의 커널과 FLA 보조 파일; recurrent는 Q/K/V·gate·beta stride를 직접 읽음 | PyTorch, Triton |
| KDA 출력 정규화 | `kda/output.py`에서 FP32 RMS norm·weight·sigmoid gate를 합치고 BF16으로 한 번 저장 | PyTorch, Triton |
| causal conv | `causal_conv_single.py`의 단일 시퀀스 conv·상태 반환 커널; 범용 prefill / update는 `causal_conv.py` | PyTorch, Triton (범용 커널은 NumPy 추가) |
| 상태 링 | `state.py`에서 물리 슬롯·위치로 필요한 이력을 읽고 변경된 위치만 쓰기 | PyTorch, Triton |
| mHC pre / post | `mhc/`의 TileLang 혼합, hidden 512 단위 TMA post, 작은 M의 prenorm 패딩 | PyTorch, TileLang, Triton, DeepGEMM |
| 인덱서 로짓 | `deep_gemm.py`에서 `deep_gemm.fp8_fp4_mqa_logits` 직접 호출 | DeepGEMM |
| 인덱서 query 양자화 | `kpool.py`의 Hadamard-128·FP8 커널, GB10 행 수별 1/8/32행 tile | PyTorch, Triton |
| kpool | `kpool.py`의 1워프 반환 전용 압축·회전·FP8 변환, 별도 캐시 쓰기 진입점 | PyTorch, Triton |
| 인덱서 슬롯 | `indexer.py`의 풀 ID 정렬·토큰 확장·페이지 주소 변환·유효 개수·출력 쓰기를 한 커널에서 처리 | PyTorch, Triton |
| MLA | `mla/`의 전용 Python 드라이버, FP8/BF16 `ldmatrix`, warp max reduction과 DSMEM split 병합 | PyTorch, CUDA 13 nvcc |
| b12x MoE | `b12x/`의 API·디스패치·CuTe 커널·내부 보조 모듈 | PyTorch, CUTLASS DSL, CUDA bindings, FlashInfer 유틸/JIT |

`SOURCES.json`은 이식 전 파일의 경로와 SHA256을 기록한다. 저장소의 기존 overlay가
소유하는 구현을 우선했고, 없는 FLA/b12x 보조 파일과 conv/kpool은 같은 플릿 이미지에서
가져왔다. b12x의 stock `_moe_dynamic/gated.py`는 바이트를 유지해 기존 소스 검증도
그대로 유효하다. KDA의 L2norm 소스 검증 값은 임포트가 바뀐 로컬 파일의 SHA256이다.
MLA CUDA의 `sha256`은 이식 원본을 보존하고, `local_sha256`과 `local_modifications`는
ST의 하드웨어 최적화가 반영된 파일과 변경 내용을 검증한다.
라이선스와 출처는 `THIRD_PARTY_NOTICES.md`에 있다.

mHC는 GLM 레인이 호출하는 pre/post를 직접 제공한다. vLLM의 CustomOp 등록, 모델 후크,
플랫폼 디스패치와 DeepGEMM 미설치 대체 경로는 포함하지 않는다. KDA의 보조 RMSNorm은
일반 `torch.nn.Module`이다. MLA의 Python 드라이버는 sparse MLA에 필요한 빌드·작업공간·
수치 판정만 보존하며, 다른 모델의 GEMM/가중치 포장 후크를 가져오지 않는다.

GB10 플릿 이미지의 PDL 정책을 유지한다: conv·TileLang·DeepGEMM에서 끈다(KDA 상태 커널 경합).
메가커널 발사는 프로덕션 채택값대로 PDL 을 켠다(`mk_pdl_enabled()` 가 코드 기본값 true, 27차 발사당 58.0→53.6 µs).
DeepGEMM 설정은 첫 실행에 한 번 적용해 CPU에서의 패키지 검사와 장치 배정 전 임포트가
CUDA 문맥을 만들지 않게 한다. MLA는 첫 eager 호출에 JIT와 수치 판정을 끝내야 한다.
그래프 캡처 중 처음 부르면 명시적으로 실패한다.

MLA는 GB10에서 실측한 thread-block cluster와 distributed shared memory(DSMEM)를
사용한다. 기본 split 계획과 합산 순서를 유지하면서 `32 <= T <= 64`, `1 <= W <= 2176`,
split 2 또는 3인 호출만 cluster로 실행한다. 각 split의 기존 shared memory를 재사용해
FP32 부분값을 합치므로 이 경로는 전역 partial 버퍼와 grid 전체 ticket barrier를 사용하지
않는다. 다른 형상은 기존 경로를 사용한다. A/B 는 `maybe_arm()` 전에 모듈 속성 `mla.ENABLE_MLA_CLUSTER=False` 로 끄고 비교하며(env 아님), 부팅 시 실제 커널의 cluster 수용량과 수치 결과를 확인한다.
측정에서 느렸던 4~8-block cluster는 기본 디스패치에 포함하지 않았다.
결과와 재현 절차는 [GB10 MLA 측정](../../measurements/st_gb10_mla_20260911/README.md)에 있다.

mHC post는 hidden 512개마다 별도 CTA를 실행하고, residual과 layer output을 TMA로
읽어 하나의 transaction barrier로 완료를 기다린다. MLA의 일반·클러스터 커널은
BF16 Q/P를 `ldmatrix.x4`, FP8 PV 조각을 SM121의 byte `ldmatrix.trans`로 읽는다.
이 변경은 벤치마크를 도입 조건에서 제외하라는 운영자 지시에 따라 코드 기본값에
반영했다. GPU 수치 검증과 성능 수치는 아직 남아 있으며, 서버 배포 완료를 뜻하지 않는다.
[도입·컴파일 기록](../../measurements/st_gb10_tma_cluster_20260912/README.md)에 상태와 재현 절차를 기록한다.

MLA의 Q 복사는 L1 캐시 힌트를 유지하는 `cp.async.ca`로 첫 KV 전송 그룹에 합친다.
기존 wait가 Q와 첫 KV 타일을 함께 기다리므로 추가 배리어가 필요 없고, 빈 split은 Q를 읽지 않는다.
3-CTA 클러스터의 병합 head 배분은 8/8/0에서 6/5/5로 바꾸며, 일반 경로의 전역 partial 병합도
워프 내 연속 주소를 사용한다. split 합산 순서와 출력 정밀도는 유지한다.
[추가 개선 기록](../../measurements/st_gb10_stream_20260912/README.md)에 GPU 실행 전 검증 범위를 기록한다.

인덱서 query 양자화는 회전·BF16 반올림·FP8 scale 계산을 유지하면서 launch 크기를
선택한다. 1,024행 이하는 1행·1 warp, 1,025~65,536행은 8행·1 warp를 사용하며,
더 큰 입력은 기존 32행·2 warp를 사용한다. GLM의 인덱서 head는 32개이므로 행 수는
토큰 수의 32배다. [GB10 양자화 측정](../../measurements/st_gb10_indexer_quant_20260911/README.md)에
레지스터·shared memory, 실제 가중치 검사와 형상별 시간을 기록했다.

KDA recurrent는 엔진의 `[H,K,V]` 상태를 직접 읽고 모든 토큰의 상태를 같은 배치로 쓴다.
`state_layout="kv"`는 dense 단일 sequence·별도 출력 상태 계약이며, 기존 `vk` 상태 테이블
API와 구분한다. GLM TP4의 16 heads·128×128·1~6토큰에서는 BV16·1 warp를 사용한다.
초기 상태와 draft 롤백 위치를 보존하면서 두 번의 전치 복사를 제거한다. context 0의 영 상태
입력은 그래프와 일반 실행의 합산 순서를 맞추기 위해 유지한다. 구형 경로와의 FP32 합산 순서는
달라질 수 있으며, 측정된 수치 차이·전체 KDA 블록 시간·재현 절차는
[GB10 KDA 상태 측정](../../measurements/st_gb10_kda_state_20260911/README.md)에 있다.

KDA 출력 정규화는 128차원 head마다 한 warp가 FP32 reduction과 gate를 계산한다.
FP32 임시 tensor를 없애며, sigmoid가 매우 작을 때도 BF16 subnormal을 보존하도록
`div.rn.f32`를 사용한다. 일반 실행과 그래프에서 같은 커널을 호출하고 LocalTP의 main-thread
dispatch도 다른 레인과 같이 적용한다. 실제 가중치·반올림·메모리 검사는
[GB10 KDA 출력 정규화 측정](../../measurements/st_gb10_kda_norm_20260911/README.md)에 있다.

## 노브 (D11, 2026-09-12 정리)

이 패키지는 환경 변수를 읽지 않는다(예외는 `ST_MLA_BUILD_ROOT`, `ST_DENSE_BUILD_ROOT`,
`ST_ONESHOT_BUILD_ROOT` 캐시 경로, `TRITON_CACHE_DIR` 와 같은 부류).
`tests/test_engine_kernels.py` 가 AST 로 강제한다. 이식 때 남았던 43개 환경 노브는 셋으로 갈랐다.

- **코드에 박은 프로덕션 채택값**: 메가커널 PDL on, GEMM 입력 모드(`MK_INPUT_CTA=4`, `MK_INPUT_REUSE=1`),
  KDA strided Q/K norm on(`VLLM_GLM53_KDA_PREFILL_QK_NORM=1`, 2026-09-06), 정적 컴팩트 컷오버 640,
  micro 입력 공유 on, 경계 검사 on, FLA 라이브러리 기본값(ieee tril, 정확 exp/log, TMA·그래프 off, kernel2 norm).
- **프로필이 선언하는 만료 노브** (`engine/profiles/glm53/boot.declared`, `STK_*`; 미선언·만료는 부팅 사망):
  `STK_moe_static` — b12x 정적 레인 사양, 프로덕션 `t,r,sf6,q0`(09-09 채택값과 TP 레시피), 비교용 `stock` 후보도 유지.
  `lanes.served()` 가 `moe_dispatch.configure_static_v2()`/`configure_tp_sf6_q0()` 로 한 번 적용하고, 바인딩 때
  `Lanes.moe_prepare` 가 층마다 뷰를 만든다(셀 `t` 는 아레나 바이트를 제자리 타일 우선으로, `sf6` 는 packed-only 스케일 소유자).
  참조 레인은 `engine/modules/expert_layout.py` 로 같은 바이트를 행 우선으로 읽는다.
  `STK_mla_prefill` — 큰 M 프리필 후보 `stock | tile32 | pair | pair4`(tile32 프로덕션 기본값), `mla.configure_prefill()` 이 무장 전에 적용.
- **프로브 훅으로 남긴 것**(env 가 아니라 인자·모듈 속성; 서빙은 안 건드림): MLA 분할 강제 `mla_decode(splits=)`, MLA 루프라인 모드
  `mla_decode(probe=)`(`.cu` `run_mla` 의 넷째 int), 쌍 프리필 겹침 통계 `mla.PAIR_STATS`, 동적 tile_m 고정
  `moe_dispatch._DYNAMIC_TILE_M_OVERRIDE`(새 형상 셀 측정용), 백엔드·컷오버·MAC 사다리 `moe_dispatch._GLM53_B12X_*`(직접 대입),
  GEMM v2 k-슬라이스 `mk_set_gemm2`(pybind), mHC 공유 패스 `engine.kernels.configure_mhc_passes()`(mhc 임포트 전; post의 고정 TMA 정책은 유지).
- **버린 것(측정돼서 진 것)**: EP 타일 계열 5파일(E=72 전문가 병렬; 프로덕션은 절대 목표로 채택했으나 디코드 TP 보다 9.8% 느림, ST 는 TP=4 형태),
  강제 W4A16(API 인자 `activation_precision="bf16"` 는 그대로), prefill reuse(39차 NEUTRAL) 와 FC1 N128(기각, −71%), KDA regime(NEUTRAL),
  mHC big-fuse(GLM in-graph +0.1%). 코드는 git 에 있고 되살리기는 `git checkout` 한 줄이다.

MLA는 필수 레인이므로 이전 `VLLM_GLM53_MEGAKERNEL`/`VLLM_GLM53_MK_MLA` 활성화 변수가 필요하지 않다.
`SOURCES.json` 의 `sha256` 은 이식 전 바이트, `local_sha256`/`local_modifications` 가 서빙되는 ST 사본과 그 편집 내역이다.

## 런타임 이미지

저장소 루트에서 `bash engine/runtime/build.sh`로 `st-engine:glm53`을 만든다.
빌드 스크립트가 seed 이미지 ID를 확인하고 그 ID에 고정한 로컬 태그를 사용한다.
`ST_IMAGE`로 결과 이미지 이름을 지정할 수 있다.

seed는 기존 플릿과 같은 PyTorch/CUDA/CuTe ABI를 제공한다. 빌드 중 포함되어 있던
DeepGEMM의 확장 바이너리와 JIT 헤더 879개 파일을 `deep_gemm` 독립 경로로 복사하고
모든 SHA256을 확인한 다음, vLLM 패키지와 남은 overlay 파일을 제거한다. 결과 이미지에서
`find_spec('vllm') is None`과 두 DeepGEMM API를 검사한다. 실행 시 seed의 프레임워크
패키지에 접근하지 않는다. DeepGEMM 추출 기록은 설치 경로의 `ST_SOURCE.json`에 있다.

새 라이브러리 빌드를 채택할 때는 `runtime/dependencies.json`의 버전과 두 DeepGEMM
API, 전체 수치 검사를 다시 검증한다. 이미지 빌드·프로브는 기존 서빙 컨테이너를 변경하지 않는다.

    bash engine/runtime/build.sh
    ST_PROBE_NO_GPU=1 bash probes/run_engine_probe.sh probes/engine_kernel_check.py --imports-only
    bash probes/run_engine_probe.sh probes/engine_kernel_check.py
    bash probes/run_engine_check.sh --layers 0-4

GPU 검사는 사용 가능한 GB10에서 실행한다. JIT 캐시는 기본 `$HOME/.cache/st`에 두며
`ST_CACHE`로 변경한다. JIT 캐시 지도(2026-09-12 실측): Triton `/cache/triton`, TileLang `/cache/tilelang`,
DeepGEMM `/cache/deep_gemm`, nvcc 빌드는 MLA `/cache/mla`, dense `/cache/st-dense`,
one-shot `/cache/st-oneshot`이다. 세 네이티브 확장은 소스·로컬 헤더의 내용과 명시적 빌드 옵션,
Torch/CUDA 버전으로 캐시를 나누고, 그 안의 `src/`를 Ninja 입력으로 사용한다. 같은 내용의
체크아웃·rsync·touch는 입력 경로나 수정 시각을 바꾸지 않는다. 실제 변경은 새 캐시를 만들며,
Torch/Ninja의 빌드·헤더 의존성 검사·잠금은 그대로 사용한다. 이 방식으로 전환할 때 기존
캐시는 한 번 새로 빌드한다. 컨테이너를 다시 만들어도 재사용하려면 `/cache`를 영속 볼륨으로
연결해야 한다. CPU 전용 NVCC 재현은 `probes/engine_native_cache_check.py`와
[네이티브 빌드 캐시 측정](../../measurements/st_native_cache_20260913/README.md)에 있다.

one-shot은 전체 Torch C++ 프런트엔드 대신 Tensor·pybind 헤더만 읽어 첫 NVCC 빌드의
파싱 비용도 줄인다. [콜드 컴파일 비교](../../measurements/st_native_compile_20260913/README.md)는
매번 새 캐시로 세 쌍을 실행하고 생성된 GPU 코드·상수·실행 메타데이터를 대조한다.

b12x 는 flashinfer 래퍼(`build_and_load_cute_dsl_kernel`)가
`/cache/.cache/flashinfer/<버전>/121a/cached_ops/st_b12x_moe_sm121a_cute_dsl/*.o` 로 내보내고 적중 시 DSL 컴파일 없이 로드한다
(키 = DSL 스택 버전 + `_kernel_source_files()` 해시, `moe_dispatch.py` 포함). CuTe DSL 자체 파일 캐시(`CUTE_DSL_CACHE_DIR`)는
`cute.compile` 에서 꺼지므로(`compile_only` → `no_cache`) ST 에는 무효다. direct micro 커널도 같은 래퍼를 탄다(모듈
`st_b12x_direct_micro_sm121a_cute_dsl`, TVM-FFI 형태: 포인터는 정수 주소, 스트림은 env 스트림). 디스크에서 다시 읽은 `.o` 로는
block-dim 프로브(레지스터 압력이 512 스레드 CTA 를 막는지)를 못 돌리므로, 빌드 때 판정을 `<커널>.blockdim.json` 사이드카로 `.o` 옆에
남기고 적중 때 읽는다. 사이드카가 없거나 낡은 `.o` 는 다시 빌드한다. `--lanes conv,kda,mhc`처럼 일부 레인을 골라 재현할 수 있다.
b12x는 `--lanes moe --moe-experts 288`로 실제 TP4 형상(288 experts, top-k 8,
hidden 4096, rank intermediate 512)을 추가 검사한다. 이 검사에서만 seed의 원본
FlashInfer API를 호출해 이식 전후를 비교한다. 엔진은 항상 자체 b12x를 호출한다.
일반 이식 형상은 원본 커널 대비 상대 오차 2%로 검사한다. GLM TP4 고정 레인은 아래의
실가중치 반복 검사와 독립 반올림 참조를 추가로 사용한다.
원본과 ST의 JIT 캐시는 서로 다른 모듈 이름으로 저장한다.

사전 샤딩의 routed FC1은 b12x의 **up | gate** 순서이며 스케일도 같은 순서다.
이전 ST의 gate | up 파일은 사용할 수 없다. 새 preshard는
`weight_layout=st-glm53-b12x-up-gate-v1` 메타데이터를 기록하고, 부팅과 실가중치 검사는
이 값을 아레나 할당 전에 확인한다. 기존 파일은 다시 생성해야 한다.
참조의 FC1 누적 정밀도와 FP4 중간값 반올림도 FP32 및 ties-to-even으로 수정했다.
실가중치 분석에서는 PTX 근사 역수·FP8 반올림으로 생기는 FP4 활성값 차이와 BF16 합산의
실행 간 차이를 분리했다. GLM TP4의 micro/static 디코드는 BF16으로 반올림한 전문가 기여분을
미리 할당한 FP32 버퍼에 더한 뒤 한 번만 BF16으로 변환한다. 36개 조건·각 64회 반복에서
일반 실행과 그래프의 반복 차이는 0%, 독립 역수 반올림 참조 대비 최대 차이는 0.663%였다.
큰 프리필의 dynamic 커널은 유지되며, 전체 모델 품질·성능 판정은 아직 완료되지 않았다.
실험 원본과 한계는 `measurements/st_engine_completion_20260911/README.md`에 있다.

GPU가 없는 개발 환경에서는 아래 검사로 임포트 경로와 전이 의존 파일, 소스 보존을 확인한다.

    python3 -m unittest discover -s tests -p 'test_engine_*.py'
