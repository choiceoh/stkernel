#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ST 오라클(ST Oracle) — 하나의 문.

플릿 없이 스텝과 레이턴시를 답하는 이 리포의 시뮬레이션 도구군의 이름이다. 엔진의 실제
스텝 루프를 돌리고(step_sim), 스텝을 바이트로 조립하며(step_kernels), 살아있는 부팅을
관측하고(step_peek), 저장된 증거를 다시 판다(step_replay). 이 파일은 얇은 문: 하위
명령을 각 도구로 건네고, 모델 레지스트리의 상태를 한 장으로 보여준다.

    python3 bench/storacle.py models                 # 모델별 바이트 상태(측정/구조/결측)
    python3 bench/storacle.py predict --model glm53 # 마법사 판정 + 예상 속도 + 신뢰도
    python3 bench/storacle.py sim --compose ...      # step_sim 으로
    python3 bench/storacle.py kernels --model ...    # step_kernels 로
    python3 bench/storacle.py peek|replay ...        # 관측·재분석으로

D17 그대로: 오라클은 부팅 수를 줄이지, 판정을 대신하지 않는다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = {"sim": "step_sim.py", "kernels": "step_kernels.py",
         "peek": "step_peek.py", "replay": "step_replay.py"}


def cmd_models() -> int:
    sys.path.insert(0, str(HERE))
    import step_kernels as kern
    print("ST-Oracle · 모델 레지스트리 — 측정됨/구조/결측:")
    for name in sorted(kern.MODEL_REGISTRY):
        b = kern.EngineBytes.for_model(name)
        miss = b.missing()
        have = []
        if b.kv_bytes_per_token:
            have.append(f"KV {b.kv_bytes_per_token / 1024:.2f} KiB/토큰")
        if b.state_ring_bytes:
            have.append(f"상태 {b.state_ring_bytes / (1 << 20):.1f} MiB/행")
        if b.experts:
            have.append(f"전문가 {b.experts}×top{b.topk}")
        status = "조립 가능" if not miss else "결측: " + ", ".join(miss)
        print(f"  {name:<8} {'·'.join(have) or '(레지스트리 항목 없음)'} → {status}")
    print("\n결측은 한 번의 체크포인트 읽기(config.json·랭크 파일 크기) 또는 프로브가 채운다 —")
    print("채워지면 --compose 가 그 모델의 사다리를 같은 검증 경로로 만든다.")
    return 0


def cmd_predict(model: str, partial: bool, ctx: int = 32000, generic: "list[str] | None" = None) -> int:
    """새 모델 예측: 마법사가 어떤 커널 레인을 확보하는지 판정하고, ST 오라클이 예상
    속도를 신뢰도와 함께 뽑는다 — 한 장으로. 레인이 거부되면 그 모델은 이 엔진에
    커널이 없는 것이다: 속도보다 레시피가 답이다."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent))
    import step_kernels as kern

    print(f"ST-Oracle · predict · 모델: {model}")
    shape, geo_note = None, ""
    if model == "glm53":
        from engine.base import kernel_shape as ks
        shape = ks.MEASURED                       # 판정된 형상(부팅 옆에 기록된 것)
    else:
        # 마법사는 형상 전체(어텐션 셀·인덱서·MoE 치수·통신…)를 묻는다 — 레지스트리가
        # 아직 그만큕 가지고 있지 않다. config.json 한 번이 채운다.
        geo_note = "형상 미완(레지스트리) — 마법사 판정은 config.json 읽기가 채우면 붙는다"

    if shape is not None:
        from engine.kernels import cells
        verdicts = cells.admission(shape)
        counts = cells.counts(verdicts)
        print(cells.table(verdicts))
        if counts["refused"] or counts["unmeasured"]:
            print(cells.work_table(verdicts))
            if counts["refused"]:
                print(f"\n!! 거부 레인 {counts['refused']}개 — 이 형상은 이 엔진에 커널이 없다: "
                      "예상 속도보다 위 레시피가 답이다.")
                return 1
        # 커널별 실측 속도(격리표, #838 아티팩트)를 레인 옆에
        tl = HERE.parent / "measurements/c4_scaling_20260913/decode-timeline-rank3.json"
        table = kern.fold_kernels_from_timeline(tl) if tl.exists() else []
        if table:
            print("\n커널별 실측(격리표 · #838 아티팩트):")
            for e in sorted(table, key=lambda x: -x["median_us"])[:6]:
                if e["rows"] == 28:
                    print(f"  {e['kernel']:<26} {e['median_us']:>7.1f}us {e['mb']:>6.1f}MB {e['gbps']:>5.0f}GB/s")
    else:
        print(f"  [마법사] {geo_note}")

    b = kern.EngineBytes.for_model(model)
    miss = b.missing()
    if miss and not partial:
        print(f"\n[속도] 결측 — {', '.join(miss)} · --partial 로 구간·신뢰도와 함께")
        return 1
    rng = kern.decode_range(b, model, ctx, generic_lanes=tuple(generic))
    step_s = 1000.0 / rng["mid_ms"]
    tps = 1 + b.spec_k * 0.45
    print(f"\n[예상 속도] ctx {ctx//1000}K C=1: 스텝 {rng['lo_ms']:.1f}~{rng['hi_ms']:.1f} ms"
          f" ({step_s:.1f} step/s) · 클라이언트 ~{step_s*tps:.0f} tok/s (k={b.spec_k}, acc 45% 가정)"
          f" · 신뢰도 {rng['confidence']:.0f}%")
    if rng["generic"]:
        print(f"  범용 서빙 성분: {', '.join(rng['generic'])} ×{kern.GENERIC_FACTOR[0]:g}~{kern.GENERIC_FACTOR[1]:g}"
              " (전용 실측 대비 — 범용 격차 실측이 다음 프로브)")
    if miss:
        print(f"  가정: {', '.join(rng['assumed'])}"
              + ("" if model == "glm53" else " · 이관: 비MoE·통신 상수·k 는 glm53 실측"))
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    sub = sys.argv[1]
    if sub == "models":
        return cmd_models()
    if sub == "predict":
        import argparse
        ap = argparse.ArgumentParser(prog="storacle predict")
        ap.add_argument("--model", default="glm53")
        ap.add_argument("--partial", action="store_true")
        ap.add_argument("--ctx", type=int, default=32000)
        ap.add_argument("--generic", default="",
                        help="what-if: 이 레인들을 범용 커널로 서빙한다고 보고 속도를 잰다(쉼표: mhc_decode,kda_recurrent)")
        a = ap.parse_args(sys.argv[2:])
        return cmd_predict(a.model, a.partial, a.ctx,
                           [x.strip() for x in a.generic.split(",") if x.strip()])
    if sub not in TOOLS:
        print(f"알 수 없는 하위 명령 {sub!r} — models | sim | kernels | peek | replay", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, str(HERE / TOOLS[sub]), *sys.argv[2:]]).returncode


if __name__ == "__main__":
    sys.exit(main())
