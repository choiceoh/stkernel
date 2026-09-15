# 부팅 실측 — main `7a26c9a1` 이전 배포본, 2026-09-16 06:49:46 KST 프로덕션 (srv2 rank 0)

> 그날의 조사 — **2026-09-16 의 부팅 하나다.** 그날 참이었던 것이고 유지되지 않는다 — 이후 무엇이 바뀌었는지는 `MEASUREMENTS.md` 가 안다.

컨테이너 시작 `2026-09-15T21:49:45.93Z` → `serving on :8000` `21:51:35.04Z` = **벽시계 109.1 s**.
런타임 메모리 원장(아레나 선언부터 `production/ready` 까지) **94.29 s** — 같은 자를 쓴
main `3acae017` 프로덕션 부팅(2026-09-15 12:38 KST)의 **168.87 s** 와 대조된다.

띄운 것은 내가 아니다. 배포 감시가 올린 프로덕션 부팅을 로그와 덤프로 읽었다.

## 1. rank 0 표

```
phase                                      seconds   dev GiB  counters
comm                                         2.116     +0.04
native builds                                0.078     +0.01  (아홉 확장 전부 캐시 적중)
prepare one-shot                             1.852     +1.42
lanes                                        3.224     +0.23
arena                                        2.600    +58.82
load                                        13.205     +3.44  blocks=54 bytes=4.77863e+10
prepare native execution                    23.501    +19.28
  load drafter                               0.200     +0.00
load vision                                  0.149     +0.00
caches                                       0.004     +0.00
engine                                       0.002     +0.00
wait for weight preparation                  3.468     -0.11
runner                                       0.000     +0.00
prefill record                               0.062     +0.00
wait for the prelude                         0.000     +0.00  prelude_s=13.141
capture decode                              45.184     +5.55
warmup shapes                                0.564     +0.00
qualify vision                               3.751     +3.55
qualify grammar                              0.044     +0.00
release warmup cache                         0.060     -3.40
```

합 **99.86 s**(`load drafter` 는 `prepare native execution` 안에 중첩). 벽시계와의 차 **9.2 s** 가
표 밖의 앞부분이고, 그 행은 다음 부팅부터 나온다(#1029, 이 부팅에는 아직 없다).

## 2. 어제 머지된 넷이 각각 얼마였나

| 항목 | 기준 (`3acae017`) | 이 부팅 | 차 |
|---|---:|---:|---:|
| 먼 끝 프리필 패스 (게이트 재사용) | 47.40 | **0.007** | −47.4 |
| `qualify grammar` (프리루드) | 10.18 | **0.044** | −10.1 |
| `prepare native execution` (팩 digest) | 32.16 | **23.50** | −8.7 |
| `wait for weight preparation` (스큐) | 9.69 | **3.47** | −6.2 |
| `warmup shapes` (프리필 여섯 제거) | 5.25 | **0.564** | −4.7 |
| `native builds` (빌드 루트 `/cache`) | 부팅마다 재컴파일 | **0.078** | 캐시 적중 |
| `load` | 9.63 | **13.21** | **+3.6** |
| 원장 총계 | 168.87 | **94.29** | **−74.6** |

`prelude_s=13.141`, `wait for the prelude 0.000` — 문의 호스트 절반 **13.14 s 가 통째로 로드 밑에
숨었다.** 추정은 −7 이었고 실제는 그 두 배다(`door` 행의 렌더러까지 같이 옮겨간 몫).

`load` 만 반대로 갔다(+3.6 s, 47.8 GB / 13.205 s = **3.62 GiB/s**). 이 부팅에는 로더 게이지가 없어
(#1029) 페이지 캐시였는지 O_DIRECT 였는지 말할 수 없다 — 다음 부팅이 답한다.

## 3. 게이트의 15.6 초 — 컴파일도, 스큐도, 회수도 아니다

행마다의 스탬프(#1023)가 처음 말했다. **rank 0 과 rank 3 이 바이트까지 같다.**

| 패스 | forward | vote | observe | 행 |
|---|---:|---:|---:|---:|
| `prefill/128/0` | **15.628** | 0.0003 | 0.0495 | 15.686 |
| `prefill/1024/0` | 1.579 | 0.0003 | 0.0164 | 1.604 |
| `prefill/32256/0` | 10.159 | 0.0009 | 0.0777 | 10.339 |
| `prefill/32256/1016320` | — | — | — | 0.007 (재사용) |
| `prefill/1024/32256` | 1.458 | 0.0003 | 0.0036 | 1.469 |

- **합의가 아니다.** 네 패스의 `vote` 를 다 더해도 **1.8 ms** 다. 랭크는 어긋나 있지 않다.
- **회수가 아니다.** 행과 forward 의 차가 곧 회수이고, 128 에서 **58 ms** 다.
- **컴파일이 아니다.** rank 3 의 walk 이 끝까지 돌고 말했다:
  `jit writes: rank 3 none -- 0 artifacts newer than the first window over 44785 files in 0.83 s`.
  게이트·커널 워밍업·캡처 **전 구간에 새 아티팩트가 0 개**다.
  (rank 0 은 `stopped at the 2 s cap` 이라 37,451 개까지만 봤다 — 부분 답이다. 상한을 10 s 로 올렸다.)

**그래서 남은 것은 첫 포워드 그 자체다.** 1,024 와 32,256 의 두 점이 토큰당 **0.2747 ms** 의 직선을
주고, 그 직선이 128 에 주는 값은 **1.33 s** 다. 실제 15.63 → **일회성 +14.3 s**.

두 랭크가 같은 값이라는 것이 후보를 좁힌다: 경쟁도 아니고 어느 한 노드의 사정도 아니다.
`CUDA_MODULE_LOADING` 은 컨테이너에서 **비어 있다**(CUDA 13.2 의 기본값 = LAZY). 그러니 남은 1순위
가설은 **첫 런치마다의 cuModuleLoad** 이고, 그것은 캐시에 쓰지 않으므로 walk 이 못 보는 종류다.
다음 부팅 하나가 가른다: `CUDA_MODULE_LOADING=EAGER` 로 띄우면 그 14.3 초는 컨텍스트 생성 쪽으로
옮겨가거나(가설 성립), 그대로 있거나(가설 기각) 한다.

## 4. 남은 자리

| | 초 | 다음 |
|---|---:|---|
| `capture decode` 45.18 | 게이트 29.1 + 커널 워밍업 2.27 + 그래프 ~11.8 | 게이트의 14.3 이 위 §3 |
| `prepare native execution` | 23.50 | 팩 digest 뒤에 남은 것의 분해가 없다 |
| `load` | 13.21 | #1029 의 `direct`·`wait_s`·`copy_s` 가 다음 부팅에 답한다 |
| 앞부분 | ~9.2 | #1029 의 `front` 행과 `import_s` |
| `qualify vision` | 3.75 | GPU 패스. 옮길 곳이 없다 |
| `lanes` | 3.22 | 커널 모듈 임포트. 프리루드와 같은 종류의 후보 |

## 5. 파일

- `memory-rank0.json`, `memory-rank3.json` — 그 부팅의 런타임 메모리 원장 두 벌(`st-dumps/`).
- `rank0-boot-lines.txt` — 컨테이너 시작 시각, rank 0 표, 양 랭크의 `jit writes`·`memory gate`·커널 워밍업 줄.
