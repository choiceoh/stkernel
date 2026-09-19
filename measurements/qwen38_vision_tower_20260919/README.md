# Qwen3.8 비전 타워(1단계) — 전처리 비트 일치, 타워는 transformers bf16 과 같은 오차 (2026-09-19)

srv2 CPU, 프로덕션 옆(컨테이너 `--memory` 12~16 GB, GPU 없음). **서빙 전이다** — 타워·전처리기·preshard 만. 엔진 연결(mRoPE·프리필 행 교체·부팅)은
다음 단계다.

## 참조

`probes/qwen38_vision_reference.py` 를 서빙 이미지 `vllm/vllm-openai:qwen38-flash-next`(transformers 5.15.1, torch 2.13, torchvision 0.28,
PIL 12.3 — ST 이미지와 같은 판) 안에서 돌려 `tests/fixtures/qwen38_vision_reference.json` 을 썼다. 생성기는 이번엔 저장소에 있다.

- **전처리기**: vLLM 의 로더(`ImageMediaIO`: EXIF 회전, 투명 → 흰 바탕) + 체크포인트의 `Qwen2VLImageProcessor`. 합성 그림 아홉 가지
  (32 배수 그대로, 반올림, 최소 픽셀 아래 확대, 최대 픽셀 위 축소 5000×4000 → 65,208 패치, 3000×20 극단 비율, RGBA, 회색조, 팔레트+투명,
  EXIF 6 JPEG). **ST 이미지에서 pixel_values sha256 아홉 개 모두 일치.**
- **mRoPE**: vLLM `Qwen3VLForConditionalGeneration._get_mrope_input_positions` 세 프롬프트(글-그림-글, 그림 둘, 그림 먼저) — 위치 [3, N]·delta
  모두 일치.
- **타워(장난감 설정)**: transformers `Qwen3VLVisionModel` fp32 대비 우리 bf16 — 상대 오차 0.0051, 행 코사인 최소 0.99998.

## 실가중치 타워 (`q38vis-real.json`)

체크포인트의 333 텐서, 같은 bf16 입력. 기준은 transformers fp32.

| 그림 (grid) | 토큰 | 우리 bf16 vs fp32 | transformers bf16 vs fp32 | 우리 vs transformers bf16 |
|---|---:|---|---|---|
| 640×480 (1,30,40) | 300 | 5.78% / cos min 0.983 | 5.86% / 0.987 | 6.46% / 0.988 |
| 517×333 (1,20,32) | 160 | 5.57% / 0.983 | 5.36% / 0.978 | 5.47% / 0.983 |

상대 오차(프로베니우스) / 행 코사인 최소. 우리 타워가 fp32 에서 벗어난 만큼은 transformers 자신을 bf16 으로 돌린 만큼과 같다 — 27 층 bf16 잔차의
몫이지 구현 차이가 아니다. 두 bf16 경로끼리의 차이(5.5~6.5%)도 독립된 두 bf16 경로에서 기대하는 크기다.
