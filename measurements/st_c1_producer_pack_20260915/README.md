# KDA o_proj C1 입력 pack 을 출력 정규화가 쓴다 — 기본 적용, GPU 미검증 (2026-09-15)

운영자 "o proj는 검증하지 말고 바로 기본값 pr머지". 이 기록은 무엇을 확인했고 무엇을 **확인하지 않았는지**를 남긴다.

## 무엇이 바뀌었나

- **전.** 8행(C=1) KDA o_proj 는 bound C1 셀이다. `run_gemm_bound_input` 이 `mk_input_pack_kernel` 로 입력 FP8 pack
  을 따로 실행하고(PDL 한 단계), 이어 ordered in-CTA 커널이 그 pack 을 읽어 TX 슬롯에 쓴다.
- **지금.** `engine/kernels/kda/output.py` `_output_norm_pack` 이 정규화를 기존 `_output_norm` 과 문장 단위로 같게
  계산해 Y 를 쓰고, 이어 같은 프로그램에서 셀의 pack 을 Y 의 BF16 바이트로부터 쓴다.
  - **왜 한 프로그램에서 되나.** D=128 이라 프로그램 하나 = (토큰, 헤드) = o_proj 입력의 128열 K-block 하나다.
    그래서 그 블록의 amax 가 프로그램 자기 행에서 바로 나온다.
  - **pack 산술.** `kernels.cu` `mk_act_scale`·`mk_act_rcp`·`mk_f32x4_to_e4m3` 와 같다.
    - 스케일은 amax·(1/448)(FP32 상수), 하한 1e-30 이다.
    - 역수는 `div.rn.f32` 로 구한다.
    - 변환은 네이티브 `cvt.rn.satfinite.e4m3x2.f32` 이고, x0 가 낮은 바이트다.
    - 워드 오프셋 `kb*1024+ks*256+(row*4+q)*8+(word&1)*4` 와 행 스케일 `[kb*8+row]` 는 pack 커널의 식이다.
- **셀 쪽.** `run_gemm_bound_input(..., producer_pack=)` 를 받으면 pack 실행을 건너뛰고 그 버퍼를 읽는다.
  bound C1 셀만 받고, 다른 경우는 거부한다.
- **켜지는 조건.** 8행 스텝, bound direct writer(`DenseLinear.producer_pack_rows`), 운영 Triton 정규화
  (`producer_pack` 속성)가 모두 맞을 때만 쓴다. `Net.producer_packs=False` 가 같은 빌드의 대조군이다.

## 확인한 것 (GPU 없음)

- **Triton 컴파일.** SM121 오프라인 컴파일(`GPUTarget("cuda", 121, 32)`, CUDA 초기화 없음)이 네 변형 모두 통과했다.

  | 변형 | 레지스터 | 공유 메모리 |
  |---|---|---|
  | 기존 `_output_norm` (bf16 / fp32 weight) | 35 / 33 | 0 |
  | `_output_norm_pack` (bf16 / fp32 weight) | 38 / 35 | 1024 B |

  두 커널 모두 stack/local 은 0 이다.
- **네이티브 빌드.** 운영 플래그 dense 확장이 CUDA 를 숨긴 채 빌드됐다(69 s). 장치 프로브는 드라이버가 없어 예상대로 거부했다.
- **CPU 테스트.** `test_engine_direct_mhc`, `decode_fastpaths`, `kda_norm`, `kernel_glue`, `linear_family`,
  `decode_seven` 에서 53개가 통과했다(GPU 전용 22개 건너뜀).

## 확인하지 않은 것

- **GPU 정확도.** 정규화 출력·TX 슬롯·pack 바이트가 `mk_input_pack_kernel` 결과와 비트 동일한지는 GPU 에서 한 번도
  돌리지 않았다. 게이트는 준비돼 있다: `probes/engine_producer_pack.py` (`--lanes producer_pack`), 큐 티켓
  `c1packs-oproj0915`(소스 59b2ad56).
- **속도.** 컴포넌트 속도와 소비자 tok/s·수락률·품질은 재지 않았다. pack 몫 크기는 `c1packs-share0915` 에 들어 있다.
- **만약 틀렸다면.** pack 바이트가 어긋나면 o_proj 출력이 조용히 달라진다. 되돌리는 법은 `Net.producer_packs=False`
  또는 이 PR 을 revert 하는 것이다.
