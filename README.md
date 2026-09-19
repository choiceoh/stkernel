# stkernel

> 살아 있는 참조 — **이 스택이 무엇인지. 형태가 바뀌면 여기부터 고친다.** 여기가 틀리면 그건 버그다.

**ST 엔진 전용 리포** — 4× NVIDIA DGX Spark (GB10, sm_121a, 48 SM), CRS812 스위치드
패브릭, TP=4. 자체 추론 엔진(`engine/`, ST 엔진)이 GLM-5.3-Flash를 프로덕션으로
서빙한다. vLLM 포크 이미지에 얹던 오버레이 스택은 **2026-09-18 폐기**됐고 —
오버레이 모듈·vLLM 런처·프로필·pair 실험 레인 전부 — 그 시절의 측정과 비교는
`measurements/` 와 최상위 기록 문서가 그대로 간직한다.

## 문서는 세 가지다

제목 아래 한 줄이 그 문서가 **아직 참이어야 하는지**를 말한다. 셋은 장식이 아니라 **사실이 움직였을 때 무엇을
할지**다 — 오늘 하나를 고치면 나머지 둘은 손대면 안 되는 것들이다.

| 배너 | 뜻 | 사실이 바뀌면 |
|---|---|---|
| **살아 있는 참조** | 코드를 따라가야 한다 | **여기를 고친다.** 틀린 채 남아 있으면 그건 버그다 |
| **그날의 조사** | 그날 참이었다 | **그대로 둔다.** 이후 바뀐 것은 `MEASUREMENTS.md` 가 안다 |
| **그대로 두는 기록** | 당시의 산출물 | **고치지 않는다.** 고치면 기록이 거짓이 된다 |

배너가 없으면 `engine/`·`bench/`·`docs/` 에서 `tests/test_docs_status.py` 가 거절한다
(`probes/*.md` 는 코드 옆에서 늙어가는 것을 전제로 면제된다).

## 무엇을 물으면 어디를 여나

| 질문 | 문서 |
|---|---|
| ST 엔진이 지금 무엇인가 | [`engine/README.md`](engine/README.md) — 형태 4가지·조합 계층·커널 · 설계 원칙은 [`engine/CHARTER.md`](engine/CHARTER.md)(D1~D17) |
| 독립 런타임 이미지를 어떻게 짓나 | [`engine/runtime/README.md`](engine/runtime/README.md) — CUDA 13.2 시드·빌드·검증 (vLLM 없음을 verify가 강제한다) |
| 이 수치가 실측인가 | [`MEASUREMENTS.md`](MEASUREMENTS.md) — **여기 없으면 미실측**. 맨 앞에 판정 규율과 찾아보기 |
| 무엇이 기본으로 켜져 있나 | [`engine/SERVING_DEFAULTS.md`](engine/SERVING_DEFAULTS.md) — 레버마다 기본값 · 켬/끔 · 그렇게 둔 근거(없으면 "원장 항목 없음") |
| 플릿 큐에 GPU 실험을 어떻게 건나 | [`bench/EXPERIMENTS.md`](bench/EXPERIMENTS.md) — 제출·브래킷·증거 계약 |
| 새 체크포인트를 붙이려면 무엇이 필요한가 | `python3 tools/onboard.py --ckpt <경로>` — 읽은 것(필드·값·키)·형상·레인 표·작업 목록, 설정이 못 정한 것은 빈칸으로(레퍼런스가 말하는 사실은 `--state <필드>=<값>`) |
| 최신 논문 중 무엇이 이 스택에 붙나 | [`docs/PAPER_MAP_20260917.html`](docs/PAPER_MAP_20260917.html) — 그날의 조사 |
| 무엇으로 재나, 어디에 함정이 있나 | 이 문서의 `bench/` · `probes/` · `tools/` 절 |

## 개발환경 자가진단

에이전트는 **`./dev` 한 명령에서 시작**한다. 도구 검색, 용도·인자 확인, 환경 관리,
실행 결과가 JSON으로 연결된다. Python 셸 환경이 달라도 준비된 개발 Python을 선택한다.

```bash
./dev                         # 도구와 작업 순서
./dev search '검사'            # 목적에 맞는 도구 + 새로 생긴 스크립트
./dev status                  # 설치·버전·누락·복구 명령
./dev describe feedback       # 실행 조건과 인자 예시
./dev run feedback -- engine/kernels/mhc_contract.py
./dev audit                   # 도구 경로와 문서가 아직 있는지
```

[에이전트 도구 계약](docs/DEVTOOLS.md) · [세션 시작 규칙](AGENTS.md).
아래 기존 명령도 그대로 사용할 수 있다.

설치나 GPU를 변경하지 않고 현재 checkout이 어떤 작업을 실행할 수 있는지
확인한다. 기본 모드는 선택적 GPU 의존성 부족을 경고만 하고, CPU 기준 누락만
`--strict`에서 실패한다.

```bash
python3 tools/dev_doctor.py
python3 tools/dev_doctor.py --strict
python3 tools/dev_doctor.py --gpu --strict
python3 tools/dev_doctor.py --json
```

`OK`는 실행 가능, `WARN`은 선택적 경로가 비활성화된 상태, `FAIL`은 strict
모드에서 막히는 조건이다. 진단 도구는 패키지를 설치하거나 GPU 메모리를
할당하지 않는다.

### 노드들과 Mac 을 같은 환경으로 (`tools/devenv/`)

버전의 원본은 저장소의 `tools/devenv/versions.env` 다. 버전을 바꾸려면 이 파일을 고치는 PR 하나면 된다. 노드는 GB10
네 대(srv1~srv4, aarch64)와 RTX 5050 PC `ost-97x`(WSL2 Ubuntu, x86_64 — 테일넷 이름, Windows 본체와 별개)다.

| 파일 | 하는 일 |
|---|---|
| `node.sh` | 노드 하나를 목록에 맞춘다: `~/.local/bin`(uv·gh·mergiraf·wt·node·codex·claude, 아키텍처에 맞는 릴리스), 시스템 python3(3.12) 사용자 영역의 torch(CUDA 13.0)·triton·`PY_PACKAGES`, graphify, git 전역 설정, `~/stkernel` 체크아웃. 없거나 버전이 다른 것만 설치하고(바꾸는 파일은 `~/.local/bin/.pre-devenv/` 로 옮겨 둔다), 이미 있는 git 설정과 추적 파일이 바뀐 체크아웃은 건드리지 않는다. 릴리스 압축 파일은 `versions.env` 에 아키텍처별로 적힌 SHA-256 과 같을 때만 푼다 — 버전을 올리면 `bash tools/devenv/node.sh --digests` 가 적을 줄을 뽑아 준다. 판정은 GPU 를 가린 `dev_doctor.py --strict` |
| `sync.sh` | 모든 노드에 `node.sh` 를 한꺼번에 적용하고 노드마다 판정을 찍는다(`--verify` 는 CPU 시험 몇 개까지). Windows 가 잠들 수 있는 `ost-97x` 만 닿지 않아도 건너뛰고, 서버가 닿지 않으면 나머지를 맞춘 뒤 실패로 끝난다. 로그는 `~/.local/state/devenv-sync/` |
| `devenv-sync` + `.service`/`.timer` | srv4 의 사용자 타이머가 매일 05:10 에 **main 의** `tools/devenv` 로 `sync.sh` 를 돌린다(작업 트리가 아니라 `origin/main` 에서 읽는다). 설치는 srv4 에서 `bash tools/devenv/sync.sh --install` |
| `mac.sh` | Mac: `~/.venvs/stkernel`(`PYTHON_VERSION` 3.12, torch 의 macOS 휠)을 만들고 zsh 의 PATH 맨 앞에 둔다. 다른 Python 으로 만든 venv 는 옆으로 옮겨 두고 다시 만든다. graphify 는 `GRAPHIFY_VERSION` 그대로 자기 venv(`~/.venvs/stkernel-graphify`)에 두고 핀이 바뀌면 다시 설치해서, 위 venv 의 bin 에 링크한다(pipx 같은 다른 설치는 건드리지 않는다). macOS 용 triton 은 없어서 Triton 시험은 `stk-test` 컨테이너에서 돈다 |

사람이 할 일은 로그인뿐이다 — `gh auth login`(git 의 github.com 자격 증명도 gh 가 답한다), codex·claude 의 첫 로그인.

## 구성

| 디렉터리 | 내용 |
|---|---|
| `engine/` | **ST 엔진** — `base/`(아레나·로더·KV·스케줄러·리스·설정), `modules/`(특징 모듈 가족), `kernels/`(NVFP4·b12x·MLA·KDA·MHC·oneshot AR), `profiles/`(glm53·qwen38 모델 형상과 부팅, dsv41 계획 계층 — 모델은 한정하지 않는다, CHARTER D5), `runtime/`(독립 이미지 빌드·검증) |
| `launchers/` | ST 런처·슈퍼바이저·systemd 유닛(`st-glm53.service`, `st-deploy-watch`), 플릿 리스·가드(`docker-fleet-guard.sh`, `lib/fleet-lease.sh`), 4노드 RoCE GID 사전(`lib/common-tp4.sh`) |
| `bench/` | 플릿 큐(`fleet.sh` 와 `fleet_*.py`), 측정 하네스(`onepass.py`, `st_bracket.sh`, `step_*`/`storacle`), CPU 게이트(`cpu_checks.py`) |
| `probes/` | 엔진 커널·그래프·드래프터 오프라인 프로브(`engine_*.py`, `run_engine_probe.sh`)와 GB10 하드웨어 상한 마이크로벤치 |
| `tools/` · `census.py` | 트레이스 분석 — 커널 인구조사·소유 판정. 이미지 인벤토리(`inventory.py`), 테스트 러너(`check.py`) |
| `tests/` | GPU 없이 도는 검증 — `python3 tools/check.py` 가 전체 판정을 내고, CI는 `.github/workflows/` 가 대표 게이트를 돌린다 |
| `MEASUREMENTS.md` | **실측 원장** — 모든 판정과 수치 (여기 없는 주장은 미실측) |

## 서빙

- 기동: `bash launchers/start-st-glm53.sh` (4노드, 랭크 0=srv2) · Qwen3.8 은
  `start-st-qwen38.sh`. 이미지는 `engine/runtime/build.sh` 가 노드마다 만든
  `st-engine:glm53` — 엔진 트리를 `/repo` 에 마운트한다.
- 상시 운영: `st-glm53.service` → `st-glm53-supervisor.sh` — 부팅 시 자동 기동,
  헬스 프롬프트 실패 시 재기동, 프로덕션 리스 보유. 배포 후 자동 검증은
  `st-deploy-watch.timer` → `st-deploy-watch.py`.
- **플릿 리스 하나**(`engine/base/fleet_lease.py`): 프로덕션은 `production` 리스,
  큐 티켓은 GO 때 `queue/<session>` 으로 잡고 페이로드는 검증만 한다. 누가 4노드를
  쥐는지의 유일한 기록이며, `docker-fleet-guard.sh` 가 리스 없는 컨테이너 격상을
  막는다.
- 롤백 경로였던 vLLM 오버레이 플릿은 폐기됐다 — 프로덕션 복구는 슈퍼바이저의
  재기동 루프가 담당하고, `fleet.sh restore-needed` 는 항상 no다.

## bench/ — 측정 도구와 함정

에이전트의 공유 플릿 실험은 [비동기 실험 제출 안내](bench/EXPERIMENTS.md)를 따른다.
GPU 하나면 되는 ST 검사는 플릿(스파크 넷)을 잡지 않고 srv4 한 대에서 프로덕션 옆에
돈다 — 큐의 단일 GPU 레인(`holder-single`). 증거는 여유 메모리(16 GiB 바닥)다.
세션은 즉시 반납하고, 커밋 하나를 팔로 삼는 비교는 `fleet.sh st-pair/st-chain`
(ST 브래킷, [`bench/EXPERIMENTS.md`](bench/EXPERIMENTS.md) 참고)으로 건다.

**플릿 없이 쓰는 측정 경로** — 4박스를 잡지 못한 상태에서도 스텝/레이턴시 숫자를
얻는 세 도구가 있다: `step_sim.py`(엔진 스텝 루프의 숙주 비용), `step_peek.py`
(살아있는 부팅의 `/metrics` 관측), `step_replay.py`(저장된 링·jsonl 증거 재계산).
이 숫자들은 숙주 비용·관측·재분석이지, 속도 주장의 판정 채널(플릿 onepass 두 번)이
아니다.

`python3 bench/storacle.py acceptance peek.jsonl`은 **각 위치까지의 누적 수락률**을
행-스텝 수와 함께 보여준다. K=6이면 마지막 행은 `6개 모두 수락한 행 / 전체 행`.
엔진 수락 카운터 기준이며 보너스 토큰은 제외한다. 빈 창, 카운터 리셋, 엔진/K 변경
기록은 수치를 내지 않는다.

| 도구 | 용도 | 함정 |
|---|---|---|
| `onepass.py` | **표준 측정** — 품질·한국어 손상·성능을 한 부팅에서 두 판(콜드·웜) | 판정은 웜끼리; 콜드 열에는 컴파일 꼬리가 섞인다 |
| `st_bracket.sh` · `st_judge.py` | **ST 엔진 브래킷** — 커밋 sha 하나가 팔, 프로덕션 형상, 부팅당 onepass 두 판; `fleet.sh st-pair/st-chain/st-hold` 로 큐에 건다; `st-probe` 는 부팅 없이 라이브 문에서 한 판 | 팔은 #770 이후의 sha 여야 한다(릴리스의 런처가 티켓 리스를 검증) · 판정은 웜끼리 |
| `korean-corruption.py` | **채택 게이트** — 한국어 출력 손상을 세어 "간헐적"을 비율로 만든다 | n=16 의 잡음이 1/16 급이다 |
| `check-quality.py` · `needle-256k.py` | **채택 게이트** — 리트리벨 9/9 · 256K needle | 인덱스 stride 버그는 산문 열화가 아니라 검색 실패로 드러난다 |
| `streamgap.py` | 동시 수용 매끄러움 — 디코드 스트리밍 중에 프리필을 끼얹는다 | 서빙 품질 축이라 step/s 와 다른 신호다 |
| `step_sim.py` | **플릿 없는 스텝 루프·형상 시뮬레이션** — 실측 상수 모형으로 TTFT·e2e·TPOT 예측. `--against <onepass jsonl...>` 로 상수 폴딩 | 장치 시간이 모형; 판정은 플릿 onepass |
| `step_peek.py` | **관측 전용 /metrics 스크랩** — 플릿을 잡지 않고 살아있는 부팅의 step/s·수용률 | 지금 돌고 있는 누군가의 부팅을 보고 있다 — 독점성도 판정도 없다 |
| `step_replay.py` | **저장 증거 재계산** — `steps-*.ring`, onepass jsonl, peek 샘플 | 재분석이지 재측정이 아니다 |
| `bench_common.py` · `window_metrics.py` | 공용 하네스(엔드포인트·프롬프트·메트릭 파서) | |

## probes/ — 오프라인 프로브

캠페인 항목의 대부분은 **부팅 전에 프로브가 먼저 답한다**. 엔진 프로브는
`probes/run_engine_probe.sh <probe>` (또는 `run_engine_check.sh`)로 돈다 — 단일
GPU 레인이면 프로덕션 옆 한 장에서, `--distributed` 면 4노드를 쓴다.

| 축 | 프로브 | 무엇을 답하나 |
|---|---|---|
| 커널 정합 | `engine_kernel_check.py` · `engine_full_check.py` · `engine_decode_graph_check.py` | 레인별 정합·그래프 캡처 아래 실제 동작 |
| MoE/NVFP4 | `engine_moe_prefill_m64.py` · `engine_fp4_instructions.py` · `engine_sparse_nvfp4.py` | 타일·명령 자격·희소 자격 |
| KDA/MLA/MHC | `engine_kda_*` · `engine_mla_*` · `engine_mhc_*` | 상태 정밀도·커널 계약·발사 설정 |
| 드래프터 | `engine_drafter_*` · `engine_draft_*` · `draft_fc_fp8_error.py` | 드래프트 품질·FC 캡처·정밀도 |
| 통신 | `oneshot_ar*.cu` · `uma_datapath.cu` | 원샷 AR 프로토타입 · UMA 데이터패스 게이트 |
| 하드웨어 상한 | `gb10_mma_rates.cu` · `gb10_gather_roof.cu` | sm_121a 텐서코어 발행량 · LPDDR gather 천장 |

**규율**: 프로브 숫자는 원장에 그대로 들어가지 않는다 — 오프라인 값은 상한이거나
형상이 다르다. 프로브가 하는 일은 **부팅할 가치가 있는지**를 먼저 가르는 것이다.

## tools/ · census.py — 트레이스 분석

프로파일 캡처로 뜬 torch 트레이스에서 **커널이 몇 발 돌고 어디에 시간이 가는지**를
뽑는다. 규칙은 하나다 — **개수는 정본, 시간은 같은 트레이스 안에서만 상대 비교**
(CUPTI 가 GPU 바쁜 시간을 부풀린다).

```bash
python3 census.py <trace.json.gz>
python3 census.py --after REGEX [--depth N]   # 인접성: 그 커널 직후 같은 스트림에서 무엇이 도나
```

새 커널을 만들면 `tools/trace_common.py` 의 `OURS` 에 심볼을 넣어야 지도가 우리
것으로 센다. **플릿 규율**: 이 분석은 랭크 노드의 CPU 를 쓴다 — 디코드 레그가
도는 동안 돌리지 말고 부팅 창에서 `nice -n 19 taskset -c 19` 로.

## 라이선스

vLLM 프로젝트에서 포팅된 커널(`engine/kernels/`)은 원본 SPDX 저작권 헤더 그대로
**Apache-2.0**이며, 리포 전체가 같은 라이선스를 따른다 (`LICENSE`).
