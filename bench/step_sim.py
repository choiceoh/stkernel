#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""스텝 루프와 서빙 형상을, 플릿 없이 시뮬레이션한다.

D17 은 "속도 주장은 플릿 onepass 두 번"이고 이 도구는 그것을 대신하지 않는다.
대신 4박스를 잡지 못한 상태에서 답할 수 있는 질문의 범위를 최대한 넓힌다:

  1. **숙주 비용** — 장치 시간 0 으로 돌려 스텝 루프(Runner·스케줄러·BlockPool·Ring)
     의 숙주 ms 를 종류별로 잰다(45차 50→77 ms 회귀의 숙주 편은 여기서 잡힌다).
  2. **형상 시뮬레이션** — 장치 시간을 상수 sleep 이 아니라 `CostModel` 로 준다:
     prefill 은 컨텍스트별 실측 처리량(tok/s 테이블), decode 는 스텝당 ms, 스펙
     디코드는 k 와 수용률로 스텝마다 1+accepted 토큰을 뽑는다. 그러면 러너의
     진짜 스텝 수·TTFT·큐잉이 모형에 반응한다 — 도착 시각(`--arrive-ms`)과 기아
     밸브(`--max-wait-s`, D10)까지 실험된다.
  3. **정확도 검증** — `--against <onepass result.jsonl>`: 기록된 플릿 숫자와
     시뮬레이션 결과를 나란히 놓고 델타를 낸다. 상수를 그 기록에서 폈으면 일치는
     자명하다 — 검증의 의미는 **스케줄러·러너 층이 실측 형상을 재현한다**는 것과,
     어느 계수가 아직 미계수(평탄 가정)인지 드러내는 것이다.

기본 비용 상수는 실측 증거에서 왔다(measurements/st_onepass_20260912_0746,
2026-09-12 ST 엔진 onepass): decode 창 중앙값 10.963 step/s → 스텝당 91.2 ms,
tokens/step 3.309·수용률 46.2% → k=5, prefill 2K 2009 / 32K 12150 / 128K 1996
tok/s. **decode 의 폭·컨텍스트 의존은 아직 미계수다**(평탄) — 플릿이 C=2/4 창을
더 주면 `--cost-json` 으로 폴딩한다.

    python3 bench/step_sim.py                                  # 실측 상수, 2K/32K/128K
    python3 bench/step_sim.py --prompts 2000 --gen 64          # 빠른 한 수
    python3 bench/step_sim.py --against measurements/st_onepass_20260912_0746/result.jsonl \
                              measurements/glm53_ep_tiled_20260909/word_onepass4/onepass.jsonl
    python3 bench/step_sim.py --against ... --fit-channel client   # 클라이언트 실측에 폼 맞춤
    python3 bench/step_sim.py --arrive-ms 0,30000 --prompts 2000,2000   # 기아 밸브 실험

`--device-ms`/`--prefill-device-ms` 는 모형을 무시하는 수동 상수(비교용)로 남는다.
StepMeta(`--meta`)는 별도 보고다: 엔진이 아직 단계마다 build 를 부르는 곳이 없어서,
루프 비용에 섞지 않고 "커널 호출자의 flat array 비용"으로 따로 잰다.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.base import scheduler as sched
from engine.base import step_meta
from engine.base.instruments import Recorder
from engine.base.kv import BlockPool, SlotPool
from engine.base.record import DeathDump, Ring
from engine.base.runner import KIND, STEP_RECORD, Runner


@dataclass
class CostModel:
    """장치 비용의 전부. 어느 숫자가 실측이고 어느 것이 가정인지가 이 모형의 정직함이다.

    측정 출처는 name 이 말한다. prefill_tok_s 는 컨텍스트 길이→tok/s 계단이고 사이는
    선형 보간한다. decode_ms_per_row / decode_ms_per_1k_ctx 는 미계수(0 = 평탄) — 플릿이
    C=2/4 창을 주면 채워지는 자리다."""
    name: str = "st-20260912"
    k: int = 5                        # 스펙 디코드 드래프트 토큰 수
    acc: float = 0.462                # raw 수용률 (1 + k×acc = tokens/step 기댓값)
    decode_ms: float = 91.2           # decode 스텝 장치 시간 (1000/10.963)
    decode_ms_per_row: float = 0.0    # 미계수: 스텝당 행 수 의존
    decode_ms_per_1k_ctx: float = 0.0 # 미계수: 컨텍스트 의존
    prefill_tok_s: dict = field(default_factory=lambda: {2000: 2009.0, 32000: 12150.0,
                                                         128000: 1996.0})
    prefill_flat_ms: float = 0.0      # >0 이면 처리량 테이블 대신 청크당 상수(수동 비교용)
    seed: int = 7                     # 수용률 추첨의 시드 — 재현 가능해야 시뮬레이션이다

    def __post_init__(self):
        # JSON 에서 오면 키가 문자열이다("512") — 계단은 항상 int:float 로 정규화한다
        self.prefill_tok_s = {int(c): float(v) for c, v in self.prefill_tok_s.items()}

    def tokens_per_step_mean(self) -> float:
        return 1.0 + self.k * self.acc

    def per_position_acc(self) -> float:
        """원장의 raw_acc(=accepted/drafted) 를 위치별 수용 확률 q 로 바꾼다.

        스펙 디코드 한 스텝에서 각 행은 첫 실패까지의 성공 수(상한 k)를 얻는다:
        E[accepted](q) = Σ n·qⁿ(1-q) + k·q^k. 원장의 tokens/step = 1 + k×raw_acc 이고
        이것이 1 + E[accepted](q) 와 같은 q 를 이등분법으로 찾는다. k=5, acc=46.2% →
        q≈0.753. 이 매핑을 건너뛰면 tokens/step 이 절반쯤으로 나간다(검증에서 발견)."""
        target = self.k * self.acc
        lo, hi = 0.0, 1.0 - 1e-9
        for _ in range(60):
            q = (lo + hi) / 2
            ea = sum(n * q ** n * (1 - q) for n in range(self.k)) + self.k * q ** self.k
            if ea < target:
                lo = q
            else:
                hi = q
        return (lo + hi) / 2

    def decode_delay(self, width: int, max_ctx: int) -> float:
        ms = (self.decode_ms + self.decode_ms_per_row * max(0, width - 1)
              + self.decode_ms_per_1k_ctx * max_ctx / 1000.0)
        return ms / 1e3

    def prefill_delay(self, prompt_len: int, chunk_tokens: int) -> float:
        if self.prefill_flat_ms > 0:
            return self.prefill_flat_ms / 1e3
        table = sorted((int(c), float(v)) for c, v in self.prefill_tok_s.items())
        if not table:
            return 0.0
        ctx = min(max(prompt_len, table[0][0]), table[-1][0])
        for (c0, v0), (c1, v1) in zip(table, table[1:]):
            if c0 <= ctx <= c1:
                tok_s = v0 if c1 == c0 else v0 + (v1 - v0) * (ctx - c0) / (c1 - c0)
                break
        else:
            tok_s = table[-1][1]
        return chunk_tokens / tok_s

    def summary(self) -> str:
        fitted = [f"decode {self.decode_ms:g} ms", f"k={self.k}", f"acc={self.acc:.1%}",
                  "prefill " + " ".join(f"{c//1000}K:{v:.0f}" for c, v in sorted(self.prefill_tok_s.items()))]
        unfitted = []
        if not self.decode_ms_per_row:
            unfitted.append("폭")
        if not self.decode_ms_per_1k_ctx:
            unfitted.append("컨텍스트")
        note = ("미계수(평탄 가정): " + ", ".join(unfitted)) if unfitted else "전 계수 폴딩됨"
        return f"{self.name} [{', '.join(fitted)}] — {note}"


def _delay(seconds: float) -> None:
    """장치 시간 흉내: 2 ms 이상은 sleep 하고 나머지는 스핀 — sleep 의 ~1 ms 해상도가
    짧은 장치 시간을 물먹지 않게(정확도의 한 축은 계기 자체다)."""
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    if seconds > 0.002:
        time.sleep(seconds - 0.0015)
    while time.perf_counter() < end:
        pass


class NullModel:
    """Model 프로토콜, 장치 없음. 비용은 CostModel 이 말한다.

    prefill 은 동기 스텝이다(러너가 async 로 내보내는 것은 decode 뿐) — 청크의 장치
    시간이 스텝 안에서 흐른다. decode_async 는 장치 시간을 직렬 단일 서버(락) 위의
    스레드에서 흘려 보내고, resolve() 가 depth 큐가 강제할 때만 호스트가 기다리게
    한다 — 박스에서와 같은 구조. 스펙 디코드: 스텝마다 각 행이 1+accepted 토큰을
    얻고 accepted 는 첫 실패까지의 성공 수(기댓값 k×acc)."""

    def __init__(self, cost: CostModel, can_async: bool = False):
        self.cost, self.can_async = cost, can_async
        self.rng = random.Random(cost.seed)
        self.q = cost.per_position_acc()                 # 스텝마다의 추첨은 위치별 q 로 한다
        self.ctx: dict = {}
        self.target: dict = {}
        self.prompt: dict = {}
        self.arrived: dict = {}
        self.prefill_done: dict = {}
        self.done: dict = {}
        self.gen_tokens = 0
        self._dev = threading.Lock()                     # 장치는 하나: 스텝은 백그라운드에서도 줄을 선다

    def open(self, seq, slot):
        pass

    def close(self, seq):
        self.ctx.pop(seq, None)
        self.target.pop(seq, None)

    def horizon(self, seq):
        # 이 스텝이 쓸 수 있는 최대 토큰수만큼 미리 잡는다(러너 규약: 예약은 결과에 앞선다)
        return self.ctx[seq] + 1 + self.cost.k

    def context(self, seq):
        return self.ctx[seq]

    def prefill(self, seq, start, tokens, blocks, slot):
        _delay(self.cost.prefill_delay(self.prompt[seq], tokens))
        self.ctx[seq] = start + tokens
        if start + tokens >= self.prompt[seq]:
            # 첫 토큰은 프리필 끝에서 뽑힌다 — TTFT 는 여기서 잰다(원장/onepass 규약)
            self.prefill_done[seq] = time.monotonic()
            self.ctx[seq] += 1
        return None

    def decode(self, seqs, blocks, slots):
        _delay(self.cost.decode_delay(len(seqs), max(self.ctx[s] for s in seqs)))
        return [self._advance(s) for s in seqs]

    def async_ready(self, seqs):
        return self.can_async

    def decode_async(self, seqs, blocks, slots):
        delay = self.cost.decode_delay(len(seqs), max(self.ctx[s] for s in seqs))

        def serve():
            with self._dev:                              # 박스처럼: 장치 스텝은 한 줄로 실행된다
                _delay(delay)
        th = threading.Thread(target=serve, daemon=True)
        th.start()
        return _Pending(self, seqs, th)

    def _accepted(self) -> int:
        n = 0
        while n < self.cost.k and self.rng.random() < self.q:
            n += 1
        return n

    def _advance(self, seq):
        if seq not in self.ctx:                          # 고스트: 이전 readback 에서 끝난 행
            return True
        made = 1 + self._accepted()
        self.ctx[seq] += made
        self.gen_tokens += made
        if self.ctx[seq] >= self.target[seq]:
            self.done[seq] = time.monotonic()
            return True
        return False


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


def _pools(prompts, gens, block, k):
    """submit 은 프롬프트 전체를 예약한다(D3) — 풀은 모든 프롬프트를 한꺼번에 담는다.
    seq id 는 1부터 쓰므로 행 수는 프롬프트 수 + 1 (SlotPool 의 null slot 과 같은 규칙).
    horizon 이 k 만큼 앞서 예약하므로 블록 여유도 k 만큼 더 둔다."""
    need = [(p + g) // block + k + 2 for p, g in zip(prompts, gens)]
    return (BlockPool(sum(need) + 16, block, len(prompts) + 1, max(need) + 2),
            SlotPool(len(prompts) + 1))


def run_once(prompts, gen, contract, cost=None, arrive_ms=None, can_async=True,
             with_meta=False, ring_capacity=8192, host_med_ms=None,
             device_s=0.0, prefill_device_s=0.0, labels=None, closed_loop=False) -> dict:
    """한 워크로드를 끝까지 돌리고 Ring 기록으로 스텝 통계와 요청별 결과를 낸다.

    순수한 계기 — 네트워크는 없다(장치 시간은 스레드 위의 지연이다). `cost` 가
    없으면 `device_s`/`prefill_device_s` 수동 상수로 모형을 만든다(k=0: 스텝당
    1 토큰, 예전 동작). `gen`: 프롬프트당 생성 토큰(스칼라 또는 프롬프트별 리스트).
    `arrive_ms`: 프롬프트별 도착 오프셋(ms) — D10 기아 밸브 실험용. `closed_loop`:
    onepass 하네스의 의미 그대로 **앞 요청이 끝나야 다음을 보낸다**(검증 모드) —
    기록 시각에서 도착을 누적 유도하면 시뮬이 살짝 느릴 때 다음 프리필이 겹쳐
    간섭이 증폭되니, 순차 실행은 시뮬 자기 시계로 닫는다. `labels`: 요청별 ctx
    라벨. `host_med_ms`: 장치 0 캘리브레이션의 숙주 med(여유분 계산에만 쓴다)."""
    if cost is None:
        cost = CostModel(decode_ms=device_s * 1e3, prefill_flat_ms=prefill_device_s * 1e3,
                         k=0, acc=0.0, prefill_tok_s={}, name="manual")
    if arrive_ms is None or closed_loop:
        arrive_ms = [0.0] * len(prompts)                 # closed loop 의 도착은 엔진 시계가 정한다
    if len(arrive_ms) != len(prompts):
        raise ValueError("--arrive-ms 는 --prompts 와 길이가 같아야 한다")
    gens = list(gen) if isinstance(gen, (list, tuple)) else [gen] * len(prompts)
    kv, slots = _pools(prompts, gens, contract.chunk_align, cost.k)
    model = NullModel(cost, can_async and cost.decode_ms > 0)
    rec = Recorder("sim")
    ring = Ring(ring_capacity, STEP_RECORD.size)
    r = Runner(model, contract, kv, slots, ring, recorder=rec)
    meta_us = []
    arrivals = sorted(zip((m / 1e3 for m in arrive_ms), range(1, len(prompts) + 1), prompts))
    pending = list(arrivals)
    t0 = time.monotonic()

    def admit(force=False):
        now = time.monotonic()
        while pending and pending[0][0] <= now - t0 + 1e-9:
            if closed_loop and not force and (r.state.running or r.state.in_prefill is not None
                                              or r.inflight):
                return                                      # 앞 요청이 끝나야 다음(하네스 의미)
            _off, seq, p = pending.pop(0)
            model.prompt[seq] = p
            model.target[seq] = p + gens[seq - 1]
            model.arrived[seq] = time.monotonic() if closed_loop else t0 + _off
            r.submit(seq, p, now=model.arrived[seq])
            if closed_loop:
                return                                      # 한 번에 하나: 하네스는 폭 1 이다
    admit(force=True)
    kinds = {"prefill": 0, "decode": 0}
    while True:
        if with_meta:
            if contract.max_wait_s != 0.0:
                raise ValueError("--meta 는 max_wait_s=0 계약에서만: plan 이 now 에 의존해 미리 빌드하기 때문")
            # 스텝 전 상태로 빌드해야 한다(prefill 의 context 는 청크 이전 토큰수).
            # plan 은 순수 함수고 max_wait_s=0 이면 now 에 무관하니 r.step() 이 같은 스텝을 고른다.
            planned = sched.plan(r.state, contract, time.monotonic())
            if planned is None and not pending:
                break
            if planned is not None:
                m0 = time.perf_counter()
                step_meta.build(planned, r.state, kv, r.slot_of, draft_slots=contract.draft_slots)
                meta_us.append((time.perf_counter() - m0) * 1e6)
        s = r.step()
        if s is None:
            if pending:
                _delay(max(0.0, t0 + pending[0][0] - time.monotonic()))
                admit()
                continue
            break
        kinds[s.kind] += 1
        admit()
    r.drain()
    wall = time.monotonic() - t0
    by_kind = {"prefill": [], "decode": []}
    tokens = {"prefill": [], "decode": []}
    for raw in ring.ordered():
        _c, step_s, kind, _n, ntok, _seq = STEP_RECORD.unpack(raw)
        k = sched.PREFILL if kind == KIND[sched.PREFILL] else sched.DECODE
        by_kind[k].append(step_s * 1e3)
        tokens[k].append(ntok)
    stats = {}
    for k, walls in by_kind.items():
        if not walls:
            continue
        stats[k] = {"steps": len(walls), "med_ms": round(statistics.median(walls), 3),
                    "p95_ms": round(_pct(walls, 0.95), 3), "max_ms": round(max(walls), 3),
                    "tokens_med": int(statistics.median(tokens[k]))}
    # 요청별 결과 — TTFT 는 prefill 종점, e2e 는 마지막 토큰(모형이 직접 잰 시계)
    requests = []
    for seq, p in sorted(model.prompt.items()):
        if seq not in model.done:
            continue                                      # 끝내지 못한 요청은 결과가 아니다
        gen_i = gens[seq - 1]
        ttft = model.prefill_done[seq] - model.arrived[seq]
        e2e = model.done[seq] - model.arrived[seq]
        requests.append({"seq": seq, "prompt": p,
                         "ctx": labels[seq - 1] if labels else p,
                         "ttft_s": round(ttft, 3),
                         "e2e_s": round(e2e, 3),
                         "decode_s": round(e2e - ttft, 3),
                         "tok_s": round((gen_i - 1) / (e2e - ttft), 2) if e2e > ttft else None})
    # 토큰/스텝 은 행-스텝 기준(원장의 tokens/step = C=1 토큰/스텝) — 폭 w 스텝은
    # w 개의 행-스텝을 실으니 스텝 수로 나누면 w 배 부풀린다(검증에서 발견).
    row_steps = sum(n * c for n, c in enumerate(r.decode_batches))
    cadence = round(sum(kinds.values()) / wall, 2) if wall > 0 else None
    decode_rate = round(kinds["decode"] / wall, 2) if wall > 0 and kinds["decode"] else None
    out = {"cost": {f: getattr(cost, f) for f in
                    ("name", "k", "acc", "decode_ms", "decode_ms_per_row", "decode_ms_per_1k_ctx",
                     "prefill_tok_s", "prefill_flat_ms")},
           "steps": kinds, "wall_s": round(wall, 3), "step_s": cadence,
           "decode_step_s_wall": decode_rate,
           "decode_step_s_phase": round(1.0 / cost.decode_delay(1, 0), 2) if cost.decode_ms > 0 else None,
           "tokens_per_step": round(model.gen_tokens / row_steps, 3) if row_steps else None,
           "tokens_per_wall_step": round(model.gen_tokens / kinds["decode"], 3) if kinds["decode"] else None,
           "by_kind": stats,
           "host_med_decode_ms": stats.get("decode", {}).get("med_ms"),
           # 숙주 여유: 캘리브레이션(장치 0)의 숙주 med 로만 계산 — ring 의 in-flight 은
           # 장치 시간과 depth 대기를 포함하니 여유분에 쓰면 안 된다.
           "headroom": (round(1 - (host_med_ms / 1e3) * decode_rate, 3)
                        if decode_rate and cost.decode_ms > 0 and host_med_ms is not None else None),
           "async": r.async_steps, "sync_drains": r.sync_drain_steps,
           "decode_widths": {str(n): c for n, c in enumerate(r.decode_batches) if n and c},
           "requests": requests,
           "recorder": rec.as_dict()["root"]["children"],
           "meta_med_us": round(statistics.median(meta_us), 1) if meta_us else None}
    out["ring"] = ring                                    # 호출자가 쓸 때만 덤프
    return out


def _drop_ring(out: dict) -> dict:
    o = dict(out)
    o.pop("ring", None)
    return o


def _ttft_by_ctx(requests: list) -> dict:
    by = {}
    for q in requests:
        by.setdefault(q.get("ctx", q["prompt"]), []).append(q["ttft_s"])
    return {p: {"n": len(v), "med_s": round(statistics.median(v), 3),
                "p95_s": round(_pct(v, 0.95), 3)} for p, v in sorted(by.items())}


def _med(values, default=None):
    return statistics.median(values) if values else default


def _fmt(out: dict, meta: bool) -> str:
    lines = [f"cost {out['cost']['name']}: k={out['cost']['k']} acc={out['cost']['acc']:.1%}"
             f" decode {out['cost']['decode_ms']:g} ms"
             + (f" prefill flat {out['cost']['prefill_flat_ms']:g} ms" if out['cost']['prefill_flat_ms']
                else " prefill " + " ".join(f"{c // 1000}K:{v:.0f}" for c, v in sorted(out['cost']['prefill_tok_s'].items())))
             + (" [폭·컨텍스트 미계수: 평탄]" if not (out['cost']['decode_ms_per_row'] or out['cost']['decode_ms_per_1k_ctx']) else "")]
    lines.append(f"steps: {out['steps']['prefill']} prefill, {out['steps']['decode']} decode"
                 + (f", async {out['async']} (동기 drain {out['sync_drains']})" if out["async"] else "")
                 + (f", tokens/step {out['tokens_per_step']}" if out.get("tokens_per_step") else ""))
    for kind, s in out["by_kind"].items():
        lines.append(f"  {kind:<7} n={s['steps']:<5} med {s['med_ms']:>8.3f} ms  "
                     f"p95 {s['p95_ms']:>8.3f}  max {s['max_ms']:>8.3f}  (in-flight, launch→readback)")
    for p, s in _ttft_by_ctx(out["requests"]).items():
        lines.append(f"  TTFT ctx{p // 1000}K: n={s['n']} med {s['med_s']:.3f}s p95 {s['p95_s']:.3f}s")
    toks = [q["tok_s"] for q in out["requests"] if q.get("tok_s")]
    if toks:
        lines.append(f"  클라이언트 decode: med {statistics.median(toks):.1f} tok/s"
                     f" [{min(toks):.1f}, {max(toks):.1f}] (첫 토큰 이후)")
    if out["cost"]["decode_ms"] == 0:
        host = out.get("host_med_decode_ms")
        if host:
            lines.append(f"숙주 상한: {1000 / host:.0f} step/s (1000 / decode med {host:.3f} ms) — "
                         "장치 시간 0에서 잰 숙주 비용만의 역수")
    else:
        lines.append(f"decode 스텝: 벽 기준 {out['decode_step_s_wall']} step/s (prefill 포함), "
                     f"모형 내재 {out['decode_step_s_phase']} step/s (1000/decode_ms)")
    if out.get("headroom") is not None:
        lines.append(f"숙주 여유 {out['headroom']:+.0%}")
    if meta and out.get("meta_med_us") is not None:
        lines.append(f"StepMeta build: med {out['meta_med_us']:.1f} us/스텝 (커널 호출자의 flat array 비용, 루프와 별도)")
    if out.get("decode_widths"):
        widths = " ".join(f"{n}×{c}" for n, c in out["decode_widths"].items())
        lines.append(f"디코드 폭: {widths} (스텝이 실은 행 수 × 스텝 수)")
    return "\n".join(lines)


def arrivals_from_record(record: dict):
    """onepass 기록에서 (요청별 도착 오프셋 ms, 요청별 생성 토큰, 요청별 프롬프트 토큰,
    요청별 ctx 라벨) 을 유도한다.

    onepass 하네스는 요청을 순차적으로 보낸다(한 컨텍스트의 세 질문도 한 줄씩) —
    기록의 ttft/decode 는 각자 자기 도착에서 잰 것이므로, 시뮬레이션도 그 도착을
    재현해야 같은 질문이 된다. 도착 오프셋 = 앞 요청들의 (ttft+decode) 합."""
    reqs = record.get("requests") or []
    offsets, gens, toks, ctxs, acc = [], [], [], [], 0.0
    prefill = {row.get("ctx"): row for row in record.get("prefill", [])}
    for q in reqs:
        ctx = q.get("ctx")
        if not isinstance(ctx, int):
            continue
        offsets.append(round(acc * 1e3, 1))
        gens.append(q.get("completion_tokens") if isinstance(q.get("completion_tokens"), int) else 256)
        toks.append(prefill.get(ctx, {}).get("tok") or ctx)
        ctxs.append(ctx)
        acc += (q.get("ttft_s") or 0.0) + (q.get("decode_s") or 0.0)
    if not offsets:
        return None
    return offsets, gens, toks, ctxs


def fit_cost(record: dict, channel: str = "windows") -> "CostModel | None":
    """기록 한 줄에서 비용 상수를 폴딩한다. 순수 함수.

    decode 스텝 ms 는 `channel` 이 말한다: "windows"(기본) 는 판정 채널(fixed pooled
    → 창 중앙값)의 역수 — 원장 규칙 2. "client" 는 요청별 실측(decode_tok_s 중앙값 ÷
    tokens/step) — 두 채널은 기록 안에서도 2~5% 차이가 나고, 어느 쪽에 폼을 맞추면
    반대쪽이 예측이 된다. k·수용률은 스펙 카운터, prefill 계단은 컨텍스트별 실제
    토큰수(warm_s: cold 는 JIT 꼬리를 실어 steady 처리량이 아니다)에서. 이렇게 폼
    상수로 맞춘 값의 일치는 자명하고, **그 위에 얹힌 나머지 측정값**(클라이언트
    tok/s·TPOT·e2e·TTFT 의 큐잉)이 따라 오는 것이 검증의 주장이다."""
    dec = record.get("decode") or {}
    tps = dec.get("tokens_per_step")
    rate = None
    if channel == "client" and tps:
        client = [q.get("decode_tok_s") for q in record.get("requests") or [] if q.get("decode_tok_s")]
        if client:
            rate = statistics.median(client) / tps
    if not rate:
        rate = dec.get("fixed_pooled_step_s") or dec.get("windows_med")
    if not rate:
        return None
    table = {}
    for row in record.get("prefill", []):
        tok, warm = row.get("tok"), row.get("warm_s")
        # warm_s 를 쓴다(cold 는 JIT 꼬리). 결합(1요청) 행은 그 컨텍스트의 유일한
        # 실측이라 warm==cold 여도 받는다 — 스킵하면 그 컨텍스트의 계단이 사라진다.
        if isinstance(tok, (int, float)) and isinstance(warm, (int, float)) and warm > 0:
            table[int(tok)] = tok / warm
    name = record.get("name") or "fitted"
    return CostModel(name=f"fit:{name}",
                     k=int(dec.get("num_spec") or 5),
                     acc=dec.get("acc_raw") or 0.45,
                     decode_ms=1000.0 / rate,
                     prefill_tok_s=table)


def validate_against(record: dict, sim: dict) -> list:
    """onepass 기록 한 줄과 시뮬레이션 결과의 나란히 비교. 순수 함수.

    각 행이 어느 쪽인지 표시한다: [입력] 폼 상수를 그 값에서 폈으니 일치는 자명,
    [예측] 폼에 얹히지 않은 값 — 클라이언트 tok/s·TPOT·e2e·TTFT(warm, 큐잉 포함)이
    따라 오는 것이 검증의 주장이다. TTFT cold 는 비교에서 뺀다(JIT 꼬리는 모형의
    범위가 아니다)."""
    dec = record.get("decode") or {}
    reqs = record.get("requests") or []
    simreq = sim["requests"]
    rows = []

    def add(label, kind, rec_v, sim_v):
        delta = (sim_v - rec_v) / rec_v if isinstance(rec_v, (int, float)) and rec_v and isinstance(sim_v, (int, float)) else None
        rows.append((label, kind, rec_v, sim_v, delta))

    add("decode step/s", "입력", dec.get("fixed_pooled_step_s") or dec.get("windows_med"),
        sim.get("decode_step_s_phase") or sim.get("decode_step_s_wall"))
    add("tokens/step", "예측", dec.get("tokens_per_step"), sim.get("tokens_per_step"))
    # 기록側 요청 단위 값들
    by_ctx = {}
    for q in reqs:
        by_ctx.setdefault(q.get("ctx"), []).append(q)
    for ctx, group in sorted(by_ctx.items()):
        warm = [q["ttft_s"] for q in group[1:]] or [group[0]["ttft_s"]]
        s_group = [q for q in simreq if q.get("ctx") == ctx]
        s = _med([q["ttft_s"] for q in s_group])
        add(f"TTFT(warm) ctx{ctx // 1000}K", "입력+큐", _med(warm), s)
        add(f"e2e med ctx{ctx // 1000}K", "예측",
            _med([q["ttft_s"] + q["decode_s"] for q in group]),
            _med([q["e2e_s"] for q in s_group]))
    add("클라이언트 tok/s", "예측", _med([q.get("decode_tok_s") for q in reqs if q.get("decode_tok_s")]),
        _med([q.get("tok_s") for q in simreq if q.get("tok_s")]))
    add("TPOT ms", "예측", _med([q.get("tpot_ms") for q in reqs if q.get("tpot_ms")]),
        _med([round(1e3 / q["tok_s"], 2) for q in simreq if q.get("tok_s")]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", default="2000,32000,128000",
                    help="쉼표 프롬프트 토큰수들 (기본: onepass 의 세 컨텍스트)")
    ap.add_argument("--gen", type=int, default=256, help="프롬프트당 생성 토큰")
    ap.add_argument("--arrive-ms", help="프롬프트별 도착 오프셋 ms (쉼표; 기본 전부 0) — D10 기아 밸브 실험")
    ap.add_argument("--max-wait-s", type=float, default=0.0, dest="max_wait_s",
                    help="D10 기아 밸브. >0 이면 살아있는 디코더를 그 초만큼 보호한다")
    ap.add_argument("--k", type=int, help="수동 스펙 k (비용 모형 덮어쓰기)")
    ap.add_argument("--acc", type=float, help="수동 수용률")
    ap.add_argument("--decode-ms", type=float, help="수동 decode 스텝 ms")
    ap.add_argument("--device-ms", type=float, default=0.0,
                    help="옛 방식: 평탄 decode 상수(모형 무시, k=0)")
    ap.add_argument("--prefill-device-ms", type=float, default=0.0, dest="prefill_device_ms",
                    help="옛 방식: prefill 청크당 평탄 상수(모형 무시)")
    ap.add_argument("--cost-json", help="CostModel JSON (k/acc/decode_ms/prefill_tok_s/...)")
    ap.add_argument("--against", type=Path, nargs="+",
                    help="onepass result.jsonl 한 개 이상 — 각 기록에서 상수를 폴드하고 나머지 측정값과 나란히 델타")
    ap.add_argument("--fit-channel", choices=("windows", "client"), default="windows",
                    dest="fit_channel",
                    help="폴딩이 맞출 decode 채널: windows=판정 채널(기본), client=요청별 실측")
    ap.add_argument("--chunk-align", type=int, default=16)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--draft-slots", type=int, default=3)
    ap.add_argument("--max-running", type=int, default=8)
    ap.add_argument("--meta", action="store_true", help="StepMeta 구축 비용을 별도로 같이 잰다")
    ap.add_argument("--no-calib", action="store_true", help="장치 0 캘리브레이션 스텝을 건너뛴다")
    ap.add_argument("--out-dir", help="steps-sim-*.ring 덤프를 이 디렉터리에 쓴다 (step_replay 가 읽는다)")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로")
    args = ap.parse_args()

    args = ap.parse_args()

    def overrides(cost):
        for flag, attr in ((args.k, "k"), (args.acc, "acc"), (args.decode_ms, "decode_ms")):
            if flag is not None:
                cost = replace(cost, **{attr: flag})
        return cost

    contract = sched.Contract(chunk_align=args.chunk_align, token_budget=args.token_budget,
                              draft_slots=args.draft_slots, max_wait_s=args.max_wait_s,
                              max_running=args.max_running)
    calib = None
    if not args.no_calib:
        calib = run_once([512], 300, contract)            # 장치 0, 동기 — 숙주 전용 캘리브레이션

    if args.against:
        # 검증 모드: 기록마다 상수를 폴딩해 재현하고, 폼에 얹히지 않은 나머지
        # 측정값들(클라이언트 tok/s·TPOT·e2e)과 나란히 놓는다. 수동 --k/--acc/
        # --decode-ms 를 주면 폴딩을 덮어쓴다(한 기록의 상수로 다른 기록을 보는
        # 폴드아웃도 이렇게 한다).
        worst = []
        for path in args.against:
            with open(path, encoding="utf-8") as f:
                record = json.loads(next(line for line in f if line.strip()))
            cost = fit_cost(record, channel=args.fit_channel)
            if cost is None:
                print(f"!! {path.name}: 판정 채널(step/s)이 없어 폴딩 불가 — 건너뛴다")
                continue
            cost = overrides(cost)
            derived = arrivals_from_record(record)
            if derived is None:
                print(f"!! {path.name}: 요청 기록이 없다 — 건너뛴다")
                continue
            arrive, gen, toks, labels = derived
            out = run_once(toks, gen, contract, cost=cost, arrive_ms=arrive,
                           can_async=True, host_med_ms=(calib or {}).get("host_med_decode_ms"),
                           labels=labels, closed_loop=True)
            print(f"== {path.name} · {record.get('name')} · git {record.get('git')}"
                  f" · 요청 {len(toks)}개")
            print(cost.summary())
            print(_fmt(out, False))
            rows = validate_against(record, out)
            print(f"-- 검증 ({path.name}): [입력] 폼 상수를 그 값에서 폈으니 일치는 자명,")
            print("   [예측] 폼에 얹히지 않은 값이 따라 오는 것이 주장이다:")
            for label, kind, rec_v, sim_v, delta in rows:
                def _f(v):
                    return f"{v:.4g}" if isinstance(v, (int, float)) else "-"
                move = f" ({delta:+.1%})" if isinstance(delta, float) else ""
                print(f"   [{kind:<4}] {label:<24} 기록 {_f(rec_v):>10}  시뮬 {_f(sim_v):>10}{move}")
                if kind == "예측" and isinstance(delta, float):
                    worst.append((abs(delta), path.name, label, delta))
            print()
        if worst:
            worst.sort(reverse=True)
            print(f"-- 예측 행 최대 잔여: {worst[0][2]} {worst[0][3]:+.1%} ({worst[0][1]}),"
                  f" 예측 {len(worst)}행 중 |잔여|>10% 는 {sum(1 for w in worst if w[0] > 0.10)}행")
        print("D17: 이 숫자는 시뮬레이션이다 — 속도 주장은 플릿 onepass 두 번으로 끝난다.")
        return 0

    prompts = [int(x) for x in args.prompts.split(",")]
    arrive = [float(x) for x in args.arrive_ms.split(",")] if args.arrive_ms else None
    gen: "int | list" = args.gen
    if args.device_ms or args.prefill_device_ms:
        cost = CostModel(decode_ms=args.device_ms, prefill_flat_ms=args.prefill_device_ms,
                         k=0, acc=0.0, prefill_tok_s={}, name="manual")
    else:
        cost = CostModel()
        if args.cost_json:
            loaded = json.loads(Path(args.cost_json).read_text(encoding="utf-8"))
            cost = replace(cost, **loaded)
            cost.name = loaded.get("name", cost.name + "+json")
        cost = overrides(cost)
    out = run_once(prompts, gen, contract, cost=cost, arrive_ms=arrive,
                   can_async=True, with_meta=args.meta,
                   host_med_ms=(calib or {}).get("host_med_decode_ms"))

    if args.out_dir:
        dump = DeathDump(args.out_dir, out["ring"],
                         boot_id=f"sim-{time.strftime('%Y%m%d-%H%M%S')}-{prompts[0]}k{args.gen}"
                         f"-d{cost.decode_ms:g}", signals=())
        dump.write_now()
        dump.close()
        out["ring_path"] = str(dump.path)

    if args.json:
        print(json.dumps({"calib": _drop_ring(calib), "run": _drop_ring(out)}, ensure_ascii=False))
        return 0
    print(f"== step_sim: prompts {prompts} gen {gen if isinstance(gen, list) else args.gen},"
          f" rows {args.max_running}"
          + (f", 도착 {arrive}ms" if arrive else ""))
    if calib:
        print(f"캘리브레이션(장치 0, C=1): decode med {calib['by_kind']['decode']['med_ms']:.3f} ms/스텝"
              f" → 숙주 상한 {1000 / calib['by_kind']['decode']['med_ms']:.0f} step/s")
    print(cost.summary())
    print(_fmt(out, args.meta))
    if out.get("ring_path"):
        print(f"ring: {out['ring_path']}")
    print("D17: 이 숫자는 숙주 비용/시뮬레이션이다 — 속도 주장은 플릿 onepass 두 번으로 끝난다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
