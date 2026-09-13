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
    # ---- PR #838(c4_scaling_20260913) 실측 정정: 바이트는 맞았으나 셋이 틀렸다 ----
    expert_mb: float = 2.10 + 1.05 + 0.30      # w13 2.10 + w2 1.05 + SF6 0.30 MB/전문가/랭크 (3.44; 아레나 표 평균 3.54 와 3%)
    moe_bw: float = 207e9                      # 정적 MoE 커널 실측 207 GB/s = 공칭의 76% (§6 CUPTI)
    routing_gamma: float = 0.763               # 실측 역산 U(7)=33, U(28)=95 → U(t)=a·t^γ (§6; 균등 가정 51.5/157 은 우연히만 맞았다)
    routing_scale: float = 33.0 / 7 ** 0.763
    nonmoe_flat_ms: float = 22.0               # 비MoE(정적·dense·KDA·글루): 1행 22 ms (§3: MoE 60~65% + 비MoE 30% + 집단통신 5~10%)
    nonmoe_per_row_ms: float = 4.33            # 4행에서 35 ms (KDA 링 쓰기 + elementwise + FP32 SGEMM 라우터)
    ctx_slope_ms_per_token: float = 3.17e-5    # 1행 2K→128K +4.0 ms (인덱서 logits 풀 32K × 질의 + 희소 MLA top-2048)
    comms_flat_ms: float = 1.0                 # 집단통신 잔여(테이블 대비): ~1 + 1.9×행 ms
    comms_per_row_ms: float = 1.9
    drafter_ms: float = 3.4                    # propose+observe 실측(§3 표; forward 와 별개) — W4 드래프터


def distinct_experts(tokens: int, experts: int = 288, topk: int = 8,
                     gamma: float = 0.0, scale: float = 0.0) -> float:
    """검증 토큰 `tokens` 개가 뽑는 서로 다른 전문가 수.

    gamma>0 면 실측 멱법칙 U(t)=min(288, scale·t^gamma) — PR #838 §6 이 플릿 forward
    에서 역산한 U(7)=33, U(28)=95 (합성 라우팅 프로브조차 28.7/61.5: 라우터 가중치가
    실제로 몰린다). gamma=0 면 균등 가정(상한 쪽): 288×(1−(1−8/288)^t)+1."""
    if tokens <= 0:
        return 0.0
    if gamma > 0:
        return min(experts, scale * tokens ** gamma)
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
            "MoE 전문가": "고유 전문가 U(토큰) × 3.44 MB × 42 층 ÷ 207 GB/s (정적 커널 실측)",
            "비MoE(정적·dense·KDA·글루)": "22 ms + 4.33 ms/행 — #838 §3/§6 (KDA 링 쓰기·elementwise·FP32 라우터)",
            "어텐션·인덱서(컨텍스트)": "31.7 µs/1K-토큰/행 — 인덱서 logits 풀 32K × 질의 + 희소 MLA top-2048 (평탄성의 이유)",
            "집합통신": "≈ 1 + 1.9×행 ms (AR 102회/스텝, 스텝의 5~10%)",
            "드래프터(W4)": "propose+observe 실측 3.4 ms — forward 와 별개",
        }
        for k, v in self.ms.items():
            share = f"{100 * v / total:5.1f}%" if total else "  -  "
            lines.append(f"{k:<28}{v:>9.2f}  {share}  {notes.get(k, '')}")
        lines.append(f"{'합계':<28}{total:>9.2f}")
        return "\n".join(lines)


def decode_step(b: EngineBytes, ctx: int, width: int = 1, k: int | None = None,
                routing: str = "measured") -> StepBudget:
    """검증 스텝 하나의 예산 → ms. 순수 함수.

    구조는 바이트에서, 상수는 PR #838 의 실측에서: MoE 는 고유 전문가 수 × 3.44 MB
    를 정적 커널의 207 GB/s 로, 비MoE 는 22 ms + 4.33 ms/행(정적·dense·KDA 링 쓰기·
    글루), 컨텍스트 항은 31.7 µs/1K-토큰/행(인덱서 logits + 희소 MLA), 집단통신은
    잔여 1 + 1.9×행 ms. routing="uniform" 은 라우팅 몰림을 무시한 상한 쪽 값이다."""
    k = b.spec_k if k is None else k
    tokens = width * (1 + k)                    # 검증은 폭×(1+k) 토큰을 한 번에 본다
    gamma, scale = (b.routing_gamma, b.routing_scale) if routing == "measured" else (0.0, 0.0)
    out = StepBudget()
    out.ms["MoE 전문가"] = (distinct_experts(tokens, b.experts, b.topk, gamma, scale)
                            * b.expert_mb * 1e6 * b.moe_layers / b.moe_bw * 1e3)
    out.ms["비MoE(정적·dense·KDA·글루)"] = b.nonmoe_flat_ms + b.nonmoe_per_row_ms * (width - 1)
    out.ms["어텐션·인덱서(컨텍스트)"] = max(0, ctx - 2000) * b.ctx_slope_ms_per_token * width
    out.ms["집합통신"] = b.comms_flat_ms + b.comms_per_row_ms * width
    out.ms["드래프터(W4)"] = b.drafter_ms
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", default="2000,32000,128000")
    ap.add_argument("--width", default="1", help="행 수(폭) — C=4 예측은 --width 1,4")
    ap.add_argument("--k", type=int, help="드래프트 k (기본 6)")
    ap.add_argument("--routing", choices=("measured", "uniform"), default="measured",
                    help="고유 전문가 수: measured=#838 역산 멱법칙 U(7)=33,U(28)=95 (기본) / uniform=균등 가정(상한)")
    ap.add_argument("--drafter-ms", type=float, dest="drafter_ms",
                    help="드래프터(propose+observe) ms — bf16 복원은 ~10.6 (2.03 GiB ÷ 207 GB/s... 실측 W4 기본 3.4)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    b = EngineBytes()
    if args.drafter_ms is not None:
        b.drafter_ms = args.drafter_ms
    ctxs = [int(x) for x in args.ctx.split(",")]
    widths = [int(x) for x in args.width.split(",")]

    rows = []
    for width in widths:
        for ctx in ctxs:
            budget = decode_step(b, ctx, width, k=args.k, routing=args.routing)
            rows.append((width, ctx, budget))
            if not args.json:
                print(f"== decode 스텝 — ctx {ctx//1000}K, 폭 {width}, k={args.k or b.spec_k}, "
                      f"routing={args.routing}:")
                print(budget.rows())
                print()
    if len(ctxs) > 1 and not args.json:
        base = next(t for w, _c, t in rows if w == widths[0])
        flat = [t.ms["어텐션·인덱서(컨텍스트)"] for w, _c, t in rows if w == widths[0]]
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
