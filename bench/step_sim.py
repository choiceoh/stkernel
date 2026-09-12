#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""스텝 루프의 숙주 비용을, 플릿 없이 잰다.

D17 은 "속도 주장은 플릿 onepass 두 번으로 끝난다"이고 이 도구는 그것을 대신하지
않는다. 4박스를 잡지 못한 상태에서도 답할 수 있는, 하나 더 좁은 질문이 있다:
**이 엔진의 스텝 루프가 숙주(host)에서 한 스텝에 몇 ms 쓰는가.** 45차의 50→77 ms
회귀 중 숙주 편은 부팅 없이 여기서 잡혔을 것이고, 그 반대편(커널)만 플릿을 기다린다.

실제 코드를 그대로 돌린다 — `engine.base.runner.Runner` 와 스케줄러, BlockPool/
SlotPool, Recorder, Ring. 모델만 장치가 없는 NullModel: prefill/decode 가 장치 시간을
0으로 만들면 Ring 에 쌓이는 스텝별 wall 이 곧 숙주 비용이다(launch glue + 예약 +
기록 + 계측). `--device-ms` 로 장치 시간을 주면 decode_async 가 스레드에서 그 시간을
잠 자고, 호스트는 depth 큐가 강제할 때까지 readback 을 미룬다 — 즉 **겹침이 실제로
모델링된다**: device 50 ms, 숙주 3 ms 면 측정 캐던스는 ~20 step/s 에서 호스트 여유
(headroom)가 같이 나온다.

    python3 bench/step_sim.py                       # 숙주 비용만 (기본 2K/32K/128K)
    python3 bench/step_sim.py --device-ms 50        # 50 ms 장치 스텝 위의 캐던스
    python3 bench/step_sim.py --out-dir /tmp/sim    # steps-sim-*.ring 덤프 (step_replay 가 읽는다)

StepMeta(`--meta`)는 별도 보고다: 엔진이 아직 단계마다 build 를 부르는 곳이 없어서,
루프 비용에 섞지 않고 "커널 호출자가 스텝마다 지불할 flat array 구축 비용"으로 따로
잰다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.base import scheduler as sched
from engine.base import step_meta
from engine.base.instruments import Recorder
from engine.base.kv import BlockPool, SlotPool
from engine.base.record import DeathDump, Ring
from engine.base.runner import KIND, STEP_RECORD, Runner


class NullModel:
    """Model 프로토콜, 장치 없음. decode_async 의 장치 시간은 조인되기 전까지
    호스트와 겹치는 스레드 위에서 흐른다 — 박스에서와 같은 구조, 같은 depth."""

    def __init__(self, gen_tokens: int, device_s: float = 0.0, can_async: bool = False):
        self.gen, self.device_s, self.can_async = gen_tokens, device_s, can_async
        self.ctx: dict = {}
        self.target: dict = {}
        self._dev = threading.Lock()                     # 장치는 하나: 스텝은 백그랜드에서도 줄을 선다

    def open(self, seq, slot):
        pass

    def close(self, seq):
        self.ctx.pop(seq, None)
        self.target.pop(seq, None)

    def horizon(self, seq):
        return self.ctx[seq] + 1

    def context(self, seq):
        return self.ctx[seq]

    def prefill(self, seq, start, tokens, blocks, slot):
        self.ctx[seq] = start + tokens
        return None

    def decode(self, seqs, blocks, slots):
        time.sleep(self.device_s)                        # 동기 모드: 장치 시간이 스텝 안에 있다
        return [self._advance(s) for s in seqs]

    def async_ready(self, seqs):
        return self.can_async

    def decode_async(self, seqs, blocks, slots):
        def serve():
            with self._dev:                              # 박스처럼: 장치 스텝은 한 줄로 실행된다
                time.sleep(self.device_s)
        th = threading.Thread(target=serve, daemon=True)
        th.start()
        return _Pending(self, seqs, th)

    def _advance(self, seq):
        if seq not in self.ctx:                          # 고스트: 이전 readback 에서 끝난 행 — 인에르트 스텝
            return True
        self.ctx[seq] += 1
        return self.ctx[seq] >= self.target[seq]


class _Pending:
    def __init__(self, model, seqs, th):
        self.model, self.seqs, self.th = model, seqs, th

    def resolve(self):
        self.th.join()                                   # readback: depth 큐가 강제할 때만 호스트가 기다린다
        return [self.model._advance(s) for s in self.seqs]


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def _pools(prompts, gen, block=16):
    """submit 은 프롬프트 전체를 예약한다(D3) — 풀은 모든 프롬프트를 한꺼번에 담는다.
    seq id 는 1부터 쓰므로 행 수는 프롬프트 수 + 1 (SlotPool 의 null slot 과 같은 규칙)."""
    need = [(p + gen) // block + 2 for p in prompts]
    return (BlockPool(sum(need) + 16, block, len(prompts) + 1, max(need) + 2),
            SlotPool(len(prompts) + 1))


def run_once(prompts, gen, contract, device_s=0.0, can_async=False, with_meta=False,
             ring_capacity=8192, host_med_ms=None) -> dict:
    """한 워크로드를 끝까지 돌리고 Ring 기록으로 스텝 통계를 낸다. 순수 로직 —
    파일도 네트워크도 없다(테스트가 이 함수를 직접 부른다). `host_med_ms`: 장치 0
    캘리브레이션의 숙주 med — 장치 시간 위의 여유분 계산에만 쓴다."""
    kv, slots = _pools(prompts, gen, contract.chunk_align)
    model = NullModel(gen, device_s, can_async)
    rec = Recorder("sim")
    ring = Ring(ring_capacity, STEP_RECORD.size)
    r = Runner(model, contract, kv, slots, ring, recorder=rec)
    meta_us = []
    for i, p in enumerate(prompts, 1):
        model.target[i] = p + gen
        r.submit(i, p, now=time.monotonic())
    t0 = time.monotonic()
    kinds = {"prefill": 0, "decode": 0}
    while True:
        if with_meta:
            if contract.max_wait_s != 0.0:
                raise ValueError("--meta 는 max_wait_s=0 계약에서만: plan 이 now 에 의존해 미리 빌드하기 때문")
            # 스텝 전 상태로 빌드해야 한다(prefill 의 context 는 청크 이전 토큰수).
            # plan 은 순수 함수고 max_wait_s=0 이면 now 에 무관하니 r.step() 이 같은 스텝을 고른다.
            planned = sched.plan(r.state, contract, time.monotonic())
            if planned is None:
                break
            m0 = time.perf_counter()
            step_meta.build(planned, r.state, kv, r.slot_of, draft_slots=contract.draft_slots)
            meta_us.append((time.perf_counter() - m0) * 1e6)
        s = r.step()
        if s is None:
            break
        kinds[s.kind] += 1
    r.drain()
    wall = time.monotonic() - t0
    by_kind = {"prefill": [], "decode": []}
    tokens = {"prefill": [], "decode": []}
    for raw in ring.ordered():
        _, step_s, kind, _n, ntok, _seq = STEP_RECORD.unpack(raw)
        by_kind[sched.PREFILL if kind == KIND[sched.PREFILL] else sched.DECODE].append(step_s * 1e3)
        tokens[sched.PREFILL if kind == KIND[sched.PREFILL] else sched.DECODE].append(ntok)
    stats = {}
    for kind, walls in by_kind.items():
        if not walls:
            continue
        stats[kind] = {"steps": len(walls), "med_ms": round(statistics.median(walls), 3),
                       "p95_ms": round(_pct(walls, 0.95), 3), "max_ms": round(max(walls), 3),
                       "tokens_med": int(statistics.median(tokens[kind]))}
    cadence = round(sum(kinds.values()) / wall, 2) if wall > 0 else None
    decode_rate = round(kinds["decode"] / wall, 2) if wall > 0 and kinds["decode"] else None
    out = {"steps": kinds, "wall_s": round(wall, 3), "step_s": cadence,
           "decode_step_s": decode_rate, "by_kind": stats,
           "host_med_decode_ms": stats.get("decode", {}).get("med_ms"),
           "device_ms": round(device_s * 1e3, 3),
           # 숙주 여유: 한 스텝 주기 중 숙주가 바쁘지 않은 비율(장치 0 캘리브레이션 기준).
           # 0 이하면 숙주가 병목이라는 뜻이다.
           "headroom": (round(1 - (host_med_ms / 1e3) * decode_rate, 3)
                        if decode_rate and device_s > 0 and host_med_ms is not None else None),
           "async": r.async_steps, "sync_drains": r.sync_drain_steps,
           "recorder": rec.as_dict()["root"]["children"],
           "meta_med_us": round(statistics.median(meta_us), 1) if meta_us else None}
    out["ring"] = ring                                                    # 호출자이 쓸 때만 덤프
    return out


def _drop_ring(out: dict) -> dict:
    o = dict(out)
    o.pop("ring", None)
    return o


def _fmt(out: dict, meta: bool) -> str:
    lines = [f"steps: {out['steps']['prefill']} prefill, {out['steps']['decode']} decode"
             + (f", async {out['async']} (동기 drain {out['sync_drains']})" if out["async"] else "")]
    for kind, s in out["by_kind"].items():
        lines.append(f"  {kind:<7} n={s['steps']:<5} med {s['med_ms']:>8.3f} ms  "
                     f"p95 {s['p95_ms']:>8.3f}  max {s['max_ms']:>8.3f}  (in-flight, launch→readback)")
    if out["device_ms"] == 0:
        host = out.get("host_med_decode_ms")
        if host:
            lines.append(f"숙주 상한: {1000 / host:.0f} step/s (1000 / decode med {host:.3f} ms) — "
                         "장치 시간 0에서 잰 숙주 비용만의 역수")
    else:
        lines.append(f"장치 {out['device_ms']:.1f} ms 위의 측정 캐던스: {out['decode_step_s']} decode step/s, "
                     f"숙주 여유 {out['headroom']:+.0%}" if out["headroom"] is not None
                     else f"장치 {out['device_ms']:.1f} ms 위의 측정 캐던스: {out['decode_step_s']} decode step/s")
    if meta and out.get("meta_med_us") is not None:
        lines.append(f"StepMeta build: med {out['meta_med_us']:.1f} us/스텝 (커널 호출자의 flat array 비용, 루프와 별도)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", default="2000,32000,128000",
                    help="쉼표 프롬프트 토큰수들 (기본: onepass 의 세 컨텍스트)")
    ap.add_argument("--gen", type=int, default=256, help="프롬프트당 생성 토큰")
    ap.add_argument("--device-ms", type=float, default=0.0,
                    help="스텝당 시뮬레이션 장치 시간. >0 이면 decode_async 겹침 경로로 잰다")
    ap.add_argument("--sync-device", action="store_true",
                    help="장치 시간을 동기 스텝 안에서 흐르게 한다 (겹침 없음 — 비교용)")
    ap.add_argument("--chunk-align", type=int, default=16)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--draft-slots", type=int, default=3)
    ap.add_argument("--max-running", type=int, default=8)
    ap.add_argument("--meta", action="store_true", help="StepMeta 구축 비용을 별도로 같이 잰다")
    ap.add_argument("--no-calib", action="store_true", help="장치 0 캘리브레이션 스텝을 건너뛴다")
    ap.add_argument("--out-dir", help="steps-sim-*.ring 덤프를 이 디렉터리에 쓴다 (step_replay 가 읽는다)")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로")
    args = ap.parse_args()

    prompts = [int(x) for x in args.prompts.split(",")]
    contract = sched.Contract(chunk_align=args.chunk_align, token_budget=args.token_budget,
                              draft_slots=args.draft_slots, max_wait_s=0.0,
                              max_running=args.max_running)
    device_s = args.device_ms / 1e3
    calib = None
    if not args.no_calib:
        calib = run_once([512], 300, contract)            # 장치 0, 동기 — 숙주 전용 캘리브레이션
    out = run_once(prompts, args.gen, contract, device_s=device_s,
                   can_async=device_s > 0 and not args.sync_device, with_meta=args.meta,
                   host_med_ms=(calib or {}).get("host_med_decode_ms"))

    if args.out_dir:
        dump = DeathDump(args.out_dir, out["ring"],
                         boot_id=f"sim-{time.strftime('%Y%m%d-%H%M%S')}-{prompts[0]}k{args.gen}"
                         f"-d{args.device_ms:g}", signals=())
        dump.write_now()
        dump.close()
        out["ring_path"] = str(dump.path)

    if args.json:
        print(json.dumps({"calib": _drop_ring(calib), "run": _drop_ring(out)}, ensure_ascii=False))
        return 0
    print(f"== step_sim: prompts {prompts} gen {args.gen}, device {args.device_ms:g} ms"
          f"{' (동기)' if args.sync_device else ''}, rows {args.max_running}")
    if calib:
        print(f"캘리브레이션(장치 0, C=1): decode med {calib['by_kind']['decode']['med_ms']:.3f} ms/스텝"
              f" → 숙주 상한 {1000 / calib['by_kind']['decode']['med_ms']:.0f} step/s")
    print(_fmt(out, args.meta))
    if out.get("ring_path"):
        print(f"ring: {out['ring_path']}")
    print("D17: 이 숫자는 숙주 비용/시뮬레이션 캐던스다 — 속도 주장은 플릿 onepass 두 번으로 끝난다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
