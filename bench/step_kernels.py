#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""디코드 스텝을 바이트로 분해한다 — 시뮬레이터의 장치 상수가 *왜* 그 값인지.

step_sim 은 장치 시간을 실측 상수로 폴딩한다(한 판이 상수의 원천). 이 모듈은 그 밑의
한 층을 흉내 낸다: **스텝이 움직이는 바이트**를 모델 형상에서 계산하고, 대역폭·효율로
나눠 스텝 시간을 *구조에서* 유도한다. 새 빌드의 구조 변경(복사 하나를 없애는 PR)이
스텝 시간을 얼마나 움직이는지 — 부팅 없이, 델타로 — 말할 수 있게 하는 것이 목적이다.

**바이트의 출처는 전부 이 리포의 실측/계산 문서다** (engine/MEMORY_REVIEW_20260911
아레나 표, STEP_KERNEL_MAP 의 커널 수, engine/profiles/glm53/facts.py):

  랭크당(TP=4)   MoE 전문가 packed+스케일 39.87 GiB / (288 전문가 × 42 층)
                 KDA·MLA·dense·shared·embed·인덱서·mHC·라우터 4.63 GiB (bf16)
                 드래프터(DFlash2, 랭크마다 통째) 2.18 GiB — GEMM 쪽 2.03 GiB
  행당           paged KV 5.90 KiB/토큰 (블록 13.28 MiB = 2304 토큰 × 11 DSA 층)
                 상태 슬롯 247.2 MiB (recurrent 204 + 드래프터 링 40 + conv 3.2)
  스텝당         all-reduce(k_oneshot) 102 회 — 실측 확정(STEP_KERNEL_MAP)

검증 스텝은 1+k 토큰을 한 번에 본다(k=6 → 7 토큰): 전문가 읽기는 그 7 토큰이 뽑는
*서로 다른* 전문가 수(기댓값 288×(1−(1−8/288)⁷) ≈ 51.6 + 공유 1)로 정해지고, 어텐션
KV 는 컨텍스트에, KDA 상태는 행 수에 정확히 비례한다. GB10 통합메모리 공칭 대역폭
273 GB/s 에 도달률 하나(`--eff`, 기본 0.85)와 집합통신 지연 하나(`--ar-ms`)가 이 모형의
전부다 — 둘 다 한 기록의 decode ms 에 맞춰 폼하고(calibrate), 나머지는 예측이다.

    python3 bench/step_kernels.py                    # 분해표 + 기록 대비
    python3 bench/step_kernels.py --eff 0.8 --ar-ms 0.08
    python3 bench/step_kernels.py --ctx 2000,32000,128000 --width 1,4

정직한 자리: 커널 미시구조(점유율·캐시)는 흉내 내지 않는다 — 바이트 예산과 도달률이다.
원장 규칙 6 이 그대로 적용된다: 이 숫자는 상한·델타 예측이지 판정이 아니다(D17).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GIB = 1 << 30
MIB = 1 << 20
KIB = 1 << 10


@dataclass
class EngineBytes:
    """랭크당·행당·스텝당 바이트 — 전부 리포 문서의 실측/계산 값이고, 각주가 출처다."""
    # engine/MEMORY_REVIEW_20260911 아레나 표 (랭크당, TP=4)
    expert_bytes_all: float = 39.87 * GIB      # 전문가 packed 35.44 + 스케일 4.43 GiB
    static_weights: float = 4.63 * GIB         # KDA 2.23 + MLA .73 + dense·shared .70 + embed·head .59 + 인덱서 .16 + mHC .13 + 라우터 .09
    drafter_weights: float = 2.03 * GIB        # DFlash2 GEMM 쪽(랭크마다 통째)
    # 같은 표, 행당
    kv_bytes_per_token: float = 13_922_304 / 2304      # 블록 13.28 MiB = 2304 토큰 × 11 DSA 층
    state_ring_bytes: float = (204.0 + 40.0 + 3.2) * MIB   # recurrent(K+1=6×34층) + 드래프터 + conv
    # STEP_KERNEL_MAP (k_oneshot 실측)
    ar_per_step: int = 102
    # facts.py / STEP_KERNEL_MAP (형상)
    experts: int = 288
    topk: int = 8
    moe_layers: int = 42
    spec_k: int = 6
    tp: int = 4
    bw_bytes_s: float = 273e9                  # GB10 통합메모리 공칭(도달률은 --eff)


def distinct_experts(tokens: int, experts: int = 288, topk: int = 8) -> float:
    """검증 토큰 `tokens` 개가 무작위 라우팅으로 뽑는 서로 다른 전문가 수의 기댓값.
    라우팅이 몰리면(실제 문서는 그렇다) 이보다 작아진다 — 기댓값은 상한 쪽이다."""
    if tokens <= 0:
        return 0.0
    p = min(1.0, topk / experts)
    return experts * (1.0 - (1.0 - p) ** tokens) + 1.0   # + 공유 전문가 1


@dataclass
class StepBudget:
    ms: dict = field(default_factory=dict)

    def total(self) -> float:
        return sum(self.ms.values())

    def rows(self) -> str:
        total = self.total()
        lines = [f"{'구성':<28}{'ms':>9}  {'몫':>6}  바이트 근거"]
        notes = {
            "MoE 전문가": "서로 다른 전문가 × 3.55 MB/랭크 × 42 층",
            "정적 가중치(dense·KDA·MLA·head)": "M=7 행에서는 가중치 읽기가 바닥",
            "드래프터(DFlash2)": "2.03 GiB GEMM 가중치, 랭크마다 통째",
            "어텐션 KV": "ctx × 5.90 KiB/토큰 — 컨텍스트 항 (평탄성의 이유)",
            "KDA 상태": "247.2 MiB 읽기+쓰기 × 행",
            "집합통신(AR 102회)": "지연 × 102 — 바이트는 무시할 만큼 작다",
        }
        for k, v in self.ms.items():
            share = f"{100 * v / total:5.1f}%" if total else "  -  "
            lines.append(f"{k:<28}{v:>9.2f}  {share}  {notes.get(k, '')}")
        lines.append(f"{'합계':<28}{total:>9.2f}")
        return "\n".join(lines)


def decode_step(b: EngineBytes, ctx: int, width: int = 1, k: int | None = None,
                eff: float = 0.85, ar_ms: float = 0.05) -> StepBudget:
    """검증 스텝 하나의 바이트 예산 → ms. 순수 함수."""
    k = b.spec_k if k is None else k
    tokens = width * (1 + k)                    # 검증은 폭×(1+k) 토큰을 한 번에 본다
    expert_bytes = b.expert_bytes_all / (b.experts * b.moe_layers)
    out = StepBudget()
    out.ms["MoE 전문가"] = distinct_experts(tokens, b.experts, b.topk) * expert_bytes * b.moe_layers / (b.bw_bytes_s * eff) * 1e3
    out.ms["정적 가중치(dense·KDA·MLA·head)"] = b.static_weights / (b.bw_bytes_s * eff) * 1e3
    out.ms["드래프터(DFlash2)"] = b.drafter_weights / (b.bw_bytes_s * eff) * 1e3
    out.ms["어텐션 KV"] = width * ctx * b.kv_bytes_per_token / (b.bw_bytes_s * eff) * 1e3
    out.ms["KDA 상태"] = width * b.state_ring_bytes * 2 / (b.bw_bytes_s * eff) * 1e3
    out.ms["집합통신(AR 102회)"] = b.ar_per_step * ar_ms
    return out


def calibrate(b: EngineBytes, target_ms: float, ctx: int = 32000, width: int = 1,
              eff: float = 0.85) -> float:
    """AR 지연 하나를 target_ms 에 맞춘다(도달률은 주어진 값으로 굳는다) —
    지연이 음수로 나오면 도달률이 낮다는 뜻이다."""
    fixed = decode_step(b, ctx, width, eff=eff, ar_ms=0.0).total()
    return max(0.0, (target_ms - fixed) / b.ar_per_step)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", default="2000,32000,128000")
    ap.add_argument("--width", default="1", help="행 수(폭) — C=4 예측은 --width 1,4")
    ap.add_argument("--k", type=int, help="드래프트 k (기본 6)")
    ap.add_argument("--drafter-gib", type=float, dest="drafter_gib",
                    help="드래프터 GEMM 가중치 GiB(기본 2.03 bf16) — W4 드래프터 부팅은 ~0.55")
    ap.add_argument("--eff", type=float, default=0.85, help="대역폭 도달률 (기본 0.85)")
    ap.add_argument("--ar-ms", type=float, help="집합통신 1회 지연 — 주지 않으면 한 기록에 폼판다")
    ap.add_argument("--calibrate-to", type=float, default=71.7,
                    help="AR 지연 폴딩의 기준 스텝 ms (기본: K-steady 32K 실측 71.7)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # AR 지연은 기본 바이트로 *한 번* 폼한다 — 노브를 바꿔 볼 때 지연이 그것을 다시
    # 흡수하면 순환이 된다(드래프터 W4 를 줄였는데 합계가 그대로인 잘못을 막는다).
    ar_ms = args.ar_ms if args.ar_ms is not None else calibrate(EngineBytes(), args.calibrate_to, eff=args.eff)
    b = EngineBytes()
    if args.drafter_gib is not None:
        b.drafter_weights = args.drafter_gib * GIB
    ctxs = [int(x) for x in args.ctx.split(",")]
    widths = [int(x) for x in args.width.split(",")]

    rows = []
    for width in widths:
        for ctx in ctxs:
            budget = decode_step(b, ctx, width, k=args.k, eff=args.eff, ar_ms=ar_ms)
            rows.append((width, ctx, budget))
            if not args.json:
                print(f"== decode 스텝 — ctx {ctx//1000}K, 폭 {width}, k={args.k or b.spec_k}, "
                      f"eff={args.eff:g}, AR {ar_ms*1e3:.0f}us:")
                print(budget.rows())
                print()
    if len(ctxs) > 1 and not args.json:
        base = next(t for w, _c, t in rows if w == widths[0])
        flat = [t.ms["어텐션 KV"] for w, _c, t in rows if w == widths[0]]
        print(f"평탄성 검증: 어텐션 KV 항이 {ctxs[0]//1000}K→{ctxs[-1]//1000}K 에서 "
              f"{flat[0]:.2f}→{flat[-1]:.2f} ms — 스텝의 {100*flat[-1]/base.total():.1f}% "
              f"(실측: 대부분 부팅에서 컨텍스트에 평탄)")
        w1 = next(t for w, _c, t in rows if w == 1 and True) if 1 in widths else None
        if w1 is not None and max(widths) > 1:
            wN = next(t for w, _c, t in rows if w == max(widths))
            print(f"폭 예측: 1 → {max(widths)} 행에서 {w1.total():.1f} → {wN.total():.1f} ms "
                  f"({wN.total()/w1.total():.2f}x) — MoE·드래프터는 스텝에 한 번, KV·상태만 행마다")
    if args.json:
        print(json.dumps([{"width": w, "ctx": c, "ms": t.ms, "total_ms": round(t.total(), 3)}
                          for w, c, t in rows], ensure_ascii=False))
    print("D17: 바이트 예산의 상한·델타 예측이다 — 판정은 플릿 onepass 두 판이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
