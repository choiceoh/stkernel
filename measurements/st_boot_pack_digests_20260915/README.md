# 팩 스토어가 가중치 하나를 레인마다 따로 해시하고 있었다 (2026-09-15)

## 무엇인가

- 부팅의 `loaded` 행은 main `3acae017` 프로덕션 두 번째 부팅(2026-09-15 03:38 UTC)에서 53.5 s 였다.
  - rank 0 의 recorder 표: `arena` 2.37 s, `load` 11.10 s, **`prepare native execution` 32.16 s**, `wait for weight preparation` 6.17 s.
  - 네 랭크가 `weights-loaded` 에 도착한 시각은 arena admission(03:38:11.0) 뒤 47.3 / 49.4 / 48.5 / **53.4 s**(rank 0/1/2/3)다. 가장 느린 rank 3(srv4)이 행을 정한다.
  - SF6 준비 줄이 끝난 뒤 `weights-loaded` 까지가 랭크마다 33.7–36.2 s 다. 그 대부분이 `prepare native execution` 이다.
- 그 안에서 dense 가중치 하나(대상 202개, 헤드, 드래프터 fc 와 블록 30개)마다 캐시 키를 두 레인이 따로 계산했다.
  - W4 레인 `PackStore.pack` 과 FP8 레인 `PackStore.pack_fp8` 이 각각 이 일을 했다.
    - 가중치 바이트를 호스트로 복사해 sha256 한다.
    - 교정 blob 을 mmap 으로 열고, `isfinite` 로 검사하고, 스무딩한 뒤 sha256 한다.
  - 같은 바이트를 두 번 한 셈이다. rank 3 의 대상 교정 blob 은 203개 7.25 GiB 이고, 여기에 드래프터 fc blob 2개(각 1.56 GiB)와 블록 30개가 더해진다.

## 바꾼 것

- **`PackStore.weight_digest(weight)` → `WeightDigest`.**
  - 호스트 사본은 호출 스레드에서 뜬다(항상 사본이라 뒤의 쓰기가 해시에 닿지 않는다). sha256 은 스토어의 워커 스레드 하나가 한다.
  - 그 사이 호출 스레드는 교정 blob 을 읽고 검사·스무딩·해시한다.
  - digest 는 자기가 해시한 바이트에만 답한다: 저장소 포인터, 오프셋, 모양, stride, 쓰기 횟수(`_version`)가 같아야 한다. 다른 텐서의 해시로 팩을 찾는 일은 `ValueError` 다.
- **교정 해시도 digest 안에서 한 번.** 키는 blob 파일(경로·장치·inode·크기·mtime), `k`, 스무딩 해시다.
  - 다음 가중치는 새 digest 를 쓴다. 스토어 전역에 남는 기억은 없다. 그래서 다시 쓴 blob 을 나중에 물으면 다시 읽는다.
- **캐시 미스(빌드) 경로.** Hessian 을 다시 읽고, 식별자에 넣은 해시와 같은지 확인한 뒤 GPTQ 를 돈다. 다르면 `calibration changed while packing` 이다.
- **`DenseLinear`.** 레인을 만들기 전에 digest 를 연다. 폭이 TILE 을 넘어 타일 사본을 만드는 가중치는 사본이 다른 바이트이므로 digest 를 넘기지 않는다.
- **`release_pages()`.** 워커 스레드를 닫는다.
- **식별자 dict 의 내용과 키 순서는 그대로다.**
  - 그래서 이미 있는 `st-dense-packs` 파일을 같은 이름으로 찾는다. 팩 바이트도 서빙 수치도 바뀌지 않는다.
  - 새 테스트 `test_both_lanes_find_the_filed_packs_and_read_the_calibration_once` 가 옛 코드(main `bb72154d`)의 식별자 계산을 그대로 옮겨 파일한 팩을 새 코드가 찾는지 본다.
- 테스트 가짜 스토어 두 개(`tests/test_engine_draft_acceptance.py`)에 `weight_digest` 를 더했다.

## CPU 측정 — rank 3 의 실제 교정 blob, 차가운 페이지 캐시

```bash
bash measurements/st_boot_pack_digests_20260915/bench_store_digests.sh <이 브랜치 트리> <스크래치> <로그>
```

- **환경.** 이미지 `st-engine:bracket-9c45086a0622`, CUDA 숨김, srv4(Grace 20코어).
  - `/home/choiceoh/glm53-cache/mkcalib`(교정 blob) 과 rank 3 / 드래프터 safetensors 헤더는 읽기 전용으로 붙였다.
  - 플릿의 `/cache/st-dense-packs` 에는 쓰지 않는다.
- **가중치.** rank 파일 헤더의 실제 모양을 가진 합성 BF16 이다. 대상 202 + 헤드 + fc + 드래프터 블록 30 = 234개. 교정 blob 은 실제 파일이다.
- **캐시 적중 경로.**
  - 먼저 스토어 자신의 식별자로 서빙 레이아웃의 스텁 팩을 파일했다(W4 232, FP8 205).
  - 두 변형 모두 같은 437개 파일을 찾아야 하고, 빌드가 하나라도 있으면 실패로 끝난다.
- **순서와 캐시.** old → new → new → old 순서로 돌렸다. 변형마다 새 프로세스를 띄웠고, 매번 시작 전에 모든 blob 과 팩에 `POSIX_FADV_DONTNEED` 를 걸었다.
- **GPU 와의 차이.** 부팅의 가중치 복사는 장치→호스트이지만 여기서는 호스트 복사다. 옛 코드는 CPU 텐서에서 `.cpu()` 가 복사를 하지 않으므로 이 차이는 새 코드에 불리하게 작용한다.
- 원시 출력은 [bench_store_digests.log](bench_store_digests.log) 에 있다.

| 순서 | 변형 | 가중치 | 교정 blob 읽기(`_hessian`) | 찾은 팩 W4 / FP8 | 빌드 | 초 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | old (`bb72154d`) | 234 | 437 | 232 / 205 | 0 | 25.88 |
| 2 | new | 234 | 235 | 232 / 205 | 0 | 13.85 |
| 3 | new | 234 | 235 | 232 / 205 | 0 | 14.86 |
| 4 | old (`bb72154d`) | 234 | 437 | 232 / 205 | 0 | 24.64 |

- 두 쌍 모두 새 코드가 빠르다: 평균 **25.26 → 14.36 s, 한 랭크에 약 −10.9 s(−43%)**.
- 같은 코드로 스크립트 경로만 달랐던 앞선 한 벌도 같은 모양이었다: old 25.29 / 25.08 s, new 14.48 / 14.34 s.
- 줄어든 blob 읽기 202회는 대상 가중치 202개의 두 번째 레인이다. 헤드는 FP8 레인만 있고, 드래프터 블록은 W4 레인만 있으며, fc 의 두 레인은 서로 다른 blob 을 읽는다.

## CPU 테스트

```bash
python3 -m unittest tests.test_engine_dense_store_digest tests.test_engine_dense_calibration tests.test_engine_draft_tuning \
  tests.test_engine_draft_acceptance tests.test_engine_dense_smoothing tests.test_engine_dense tests.test_engine_prefill_record
```

- 같은 이미지, CUDA 숨김, main `34751523` 위로 리베이스한 트리: 85 테스트 OK, 7 스킵(CUDA 전용). 로그는 [cpu-tests.log](cpu-tests.log) 에 있다.
  - 리베이스 전 트리에서는 앞의 여섯 모듈이 61 테스트 OK(7 스킵)였다.
- 새 모듈 `tests/test_engine_dense_store_digest.py` 의 여섯 테스트:
  - 옛 식별자로 파일된 팩을 두 레인이 찾고, blob 은 한 번만 읽는다.
  - digest 없이 따로 부른 두 레인도 같은 팩을 찾는다(blob 두 번).
  - digest 는 해시한 바이트에만 답한다. 쓰기 뒤, 부분 타일, 복제본은 거절한다.
  - 다시 쓴 blob 은 다시 읽고 새 팩을 파일한다.
  - 빌드 중 교정이 바뀌면 거절한다.
  - `release_pages` 가 워커를 닫고, 뒤의 digest 는 새 워커를 쓴다.

## 재지 않은 것

- **GPU 부팅.** 다음 부팅의 두 값을 이번 기준과 비교할 것.
  - rank 0 표의 `prepare native execution`: 기준 32.16 s.
  - 원장의 `loaded` 행: 기준 53.5 s.
- **추정.** 랭크마다 약 −10 s 다. CPU 측정의 −10.9 s 에서 왔고, GPU 에서는 장치→호스트 복사 시간이 다르다. 네 랭크 모두에 적용되므로 가장 느린 랭크의 `loaded` 가 그만큼 줄어야 한다. 추정이지 측정이 아니다.
- **남은 비용.** 새 코드도 가중치마다 blob 한 번의 로드·`isfinite`·스무딩·sha256, 가중치 sha256, 팩 로드를 한다. CPU 에서 14.4 s 다. 가중치를 가로지르는 병렬화나 파일 식별자로 해시를 부팅 사이에 기억하는 방법은 넣지 않았다.
