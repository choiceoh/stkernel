# Qwen3.8 부팅 시간 — 공유 캐시의 MoE 커널 교대 삭제, 문 준비의 직렬 7 s (2026-09-18, 부팅 0회)

> 그대로 두는 기록 — 이 날 플릿에서 다른 세션들이 띄운 Qwen3.8 창 세 번과 그 사이 프로덕션 부팅의 로그에서 읽은 것이다.
> 이 세션은 부팅을 띄우지 않았다. 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

## 1. 공유 `/cache` 에서 프로덕션과 Qwen3.8 창이 서로의 MoE 커널을 지운다

네 노드의 `/cache` (= `~/glm53-cache`) 는 프로덕션 릴리스와 그 옆에서 창을 여는 트리가 함께 쓴다(`start-st-qwen38.sh` 의
`CACHE_DIR` 기본값이 프로덕션과 같다). b12x 의 CuTe-DSL MoE 커널은 flashinfer 모듈 `st_b12x_moe_sm121a_cute_dsl` 에 `.o` 로 남고,
flashinfer 는 모듈의 `meta.json`(키 파일 **내용**의 해시)이 지금 빌드하는 커널과 다르면 **모듈 디렉터리를 통째로 지운다**.
모듈 이름은 키 파일의 **이름**만으로 정해졌으므로(`moe_dispatch._cute_dsl_module`), `moe_dispatch.py` 등 키 파일 내용이 다른 두 트리는
같은 디렉터리를 번갈아 지웠다. Qwen3.8 과 GLM-5.3 의 커널 형상은 겹치지 않는다(k2560·n640 대 k4096·n512) — 공유로 얻는 것은 없다.

| 시각 (KST) | 누가 | 트리 | 문 열림 | `Invalidating stale ... st_b12x_moe` |
|---|---|---|---|---|
| 15:26 | 프로덕션 | 4bf1bf76 | 105 s | (로그 없음) |
| 16:14~16:47 | Qwen3.8 창 1 (첫 플릿 부팅) | 창 트리 | 랭크 0 "ready in 107.4 s", 재부팅 40.1 s | (로그 없음) |
| 16:48 | 프로덕션 | 4bf1bf76 | 105 s | (로그 없음) |
| 17:01~17:08 | Qwen3.8 창 2 (C4 프로브 + 부팅, #1180 트리) | 1ade438e | — | (로그 없음) |
| 17:09 | 프로덕션 | 6ad27304 | **150 s** | srv4 랭크 3: 있음, 뒤이어 정적 커널 6개 컴파일(앞 5개 연달아 약 9.5 s 씩) — [원문](srv4-st-glm53-rank3-1709-cute.log) |
| 17:34 | Qwen3.8 창 3 (`qwen38-k3`, spec-k 3) | 창 트리 | 캡처 중 IMA 로 사망(그 세션의 일) | srv4 랭크 3: 있음 (08:35:21Z), 뒤이어 `static_m16`·`static_m12` 컴파일 4.3 s·2.4 s — [원문](srv4-st-qwen38-rank3-1734.log) |
| 17:48 | 프로덕션 | 6ad27304 | **150 s** | srv4 랭크 3: 있음 (08:50:22Z), 17:09 와 같은 커널을 다시 컴파일 — [원문](srv4-st-glm53-rank3-1748-cute.log) |

- 두 부팅의 랭크 0 단계 표: `capture decode` 90.8 s / 91.3 s, 합계 152.3 s / 152.5 s — [17:09](srv2-st-glm53-boot-rank0-1709.json),
  [17:48](srv2-st-glm53-boot-rank0-1748.json). 로그가 남지 않은 105 s 부팅(15:26·16:48)의 표는 덮어써져 없다 — 150 대 105 의 차는
  문 열림 시각의 비교이지 단계별 차가 아니다. 감독기 기록은 [srv2-st-supervisor-0918.log](srv2-st-supervisor-0918.log).
- 무효화는 **다른 트리**가 모듈을 쓴 뒤에만 일어난다(같은 트리면 `meta.json` 이 같다). 17:48 의 직전 작성자는 17:35 의 Qwen3.8 창 3 이다 —
  그 부팅이 프로덕션의 17:09 모듈을 지우고 `static_m16`·`static_m12` 를 썼고, 17:35~17:48 에 다른 작성자는 없다. 17:09 의 직전 작성자는
  창 2(1ade438e) 또는 16:48 의 4bf1bf76 인데(16:59 배포 부팅이 17:01 에 죽기 전 커널을 썼는지 로그가 없다), 어느 쪽이든 다른 트리다.
- srv4 캐시 디렉터리 크기: 모듈 하나 360~760 KiB.

## 2. Qwen3.8 부팅의 앞부분 — 랭크 3, 창 3 (17:34), 컨테이너 타임스탬프

`fleet.py` 는 단계 표(Recorder)를 만들되 출력도 덤프도 하지 않았다. 그래서 로그 타임스탬프로만 읽는다.

| 구간 | 초 | 무엇 |
|---|---:|---|
| 컨테이너 시작 → `box:` | 2.87 | 파이썬·torch·facts·박스 검사 |
| `kernel shape` → `collectives` | 5.53 | `Comm.init`(랑데부) + one-shot 준비 |
| `collectives` → `lanes qualified` | **3.96** | `lanes.served()`(커널 패키지 임포트) + `lanes.qualify` |
| `arena-admission` → `weights-loaded` 랑데부 | 12.42 | 아레나·로드·PLE 표·dense 팩·캐시·엔진 + 가장 느린 랭크 대기 |
| 캡처 시작 → 첫 무효화·컴파일 | — | 이 부팅은 캡처 중 죽었다 |

## 3. 문 준비와 커널 임포트의 호스트 비용 — ST 이미지, CPU 전용 컨테이너, 새 프로세스

`st-engine:qwen38`, `--cpus 4 --memory 6g`, `CUDA_VISIBLE_DEVICES=`, 메타데이터는 `~/models/st-qwen38-tep4`.

| 일 | 초 |
|---|---:|
| `door_host_half(renderer=True)` — 토크나이저 + transformers 챗 템플릿 + 생각 블록·도구 형식·effort 판독 (랭크 0), 첫 호출 | **7.04** |
| 같은 일, 같은 프로세스 두 번째 호출 | 1.17 |
| `door_host_half(renderer=False)` — 토크나이저만 (랭크 1~3) | 0.60 |
| `lanes.import_kernels()` — triton·flashinfer·CuTe DSL(b12x) | 1.76 |

이전 `fleet.py` 는 이 7 s 를 캡처가 **끝난 뒤** 랭크 0 에서 직렬로 썼다. 문은 랭크 0 이 연다.

## 4. 이 기록을 근거로 한 변경 (PR)

- `moe_dispatch._cute_dsl_module`: 모듈 이름에 키 파일 내용 해시를 넣는다(direct micro 모듈도). 트리마다 자기 디렉터리를 쓰므로
  사이에 무엇이 돌았든 자기 `.o` 를 찾는다.
- `qwen38/fleet.py`: 커널 임포트를 랑데부 밑 스레드로, 문 준비(§3)를 로드·팩 밑 스레드로(캡처 전 합류), 단계 행(front·comm·
  prepare one-shot·lanes·qualify lanes·wait for the prelude·door), 랭크 0 표 출력, 랭크마다 `boot-rank{r}.json`·`memory-rank{r}.json` 을
  `--dump-dir`(기본 `~/glm53-logs/st-qwen38-dumps`)에.
- GLM-5.3 의 `Background` 는 `engine/base/background.py` 로 옮겼다(동작 동일, GLM 부팅 테스트 통과).

## 5. 안 잰 것

- 이 변경이 들어간 부팅은 없다. 효과는 **추정**이다: 공유 캐시 쪽은 창 사이에 프로덕션이 돌아도 Qwen3.8 부팅이 웜 캐시로
  시작하고(창 1 의 콜드 107.4 s 대 웜 40.1 s 차이 중 b12x 몫은 갈리지 않았다), 프로덕션도 창 뒤에 제 커널을 찾는다(17:09·17:48 의 150 s 대
  15:26·16:48 의 105 s). 문 준비는 랭크 0 에서 최대 약 7 s, 임포트는 약 2 s 가 가려질 수 있으나 로드 창(약 12 s)의 GIL 경합이 얼마를 되돌려 받는지는
  모른다. 다음 창의 `boot-rank*.json` 이 판정한다.
- 이 트리가 처음 배포될 때 b12x 커널은 새 모듈 이름으로 한 번 전부 다시 컴파일된다(`moe_dispatch.py` 가 바뀌는 모든 배포와 같다).
  이전 모듈 디렉터리는 지워지지 않고 남는다(하나 1 MiB 미만).
