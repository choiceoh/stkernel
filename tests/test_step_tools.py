"""노플릿 측정 도구의 순수 로직 회귀 — GPU 도 네트워크도 없다.

대상: bench/step_sim.py (스텝 루프 숙주 비용), bench/step_peek.py (/metrics 관측
파싱·창 계산), bench/step_replay.py (저장된 증거 재계산). D17 은 안 바뀐다 — 이 도구들의
숫자는 숙주 비용/관측/재분석이지 플릿 판정이 아니다.

실행: python3 -m unittest discover -s tests -p 'test_step_tools.py' -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import step_peek as peek                     # noqa: E402
import step_replay as replay                 # noqa: E402
import step_sim as sim                       # noqa: E402
from engine.base import scheduler as sched   # noqa: E402
from engine.base.record import DeathDump     # noqa: E402

CONTRACT = sched.Contract(chunk_align=16, token_budget=1024, draft_slots=3,
                          max_wait_s=0.0, max_running=8)


class CostModelTest(unittest.TestCase):
    FAST = sim.CostModel(k=2, acc=0.5, decode_ms=2.0, prefill_tok_s={512: 5120.0, 2048: 5120.0},
                         name="fast-test")

    def test_per_position_acc_solves_the_ledger_mapping(self):
        # 원장 tokens/step = 1 + k×raw_acc 를 만드는 위치별 q: k=5, acc=46.2% → ~0.75.
        # 이 매핑을 건너뛰면 스텝당 토큰이 절반쯤으로 나간다(검증에서 발견한 버그).
        cost = sim.CostModel(k=5, acc=0.462)
        q = cost.per_position_acc()
        self.assertTrue(0.70 < q < 0.80, q)
        self.assertAlmostEqual(cost.tokens_per_step_mean(), 1 + 5 * 0.462, places=6)

    def test_tokens_per_step_is_row_basis(self):
        out = sim.run_once([512, 512], 96, CONTRACT, cost=self.FAST, can_async=False)
        # 행-스텝 기준: 폭 2 스텝이 많아도 토큰/행-스텝 은 기댓값 1+k×acc=2.0 근처
        self.assertAlmostEqual(out["tokens_per_step"], 2.0, delta=0.5)
        if out["steps"]["decode"]:
            self.assertGreaterEqual(out["tokens_per_wall_step"], out["tokens_per_step"])

    def test_seed_makes_runs_reproducible(self):
        a = sim.run_once([512], 64, CONTRACT, cost=self.FAST, can_async=False)
        b = sim.run_once([512], 64, CONTRACT, cost=self.FAST, can_async=False)
        self.assertEqual(a["steps"], b["steps"])
        self.assertEqual(a["tokens_per_step"], b["tokens_per_step"])

    def test_ttft_is_prefill_throughput(self):
        out = sim.run_once([512], 64, CONTRACT, cost=self.FAST, can_async=False)
        (q,) = out["requests"]
        self.assertAlmostEqual(q["ttft_s"], 512 / 5120.0, delta=0.06)

    def test_late_arrival_does_not_queue_behind_earlier_request(self):
        out = sim.run_once([512, 512], 96, CONTRACT, cost=self.FAST, can_async=False,
                           arrive_ms=[0.0, 900.0])
        first, second = out["requests"]
        self.assertAlmostEqual(first["ttft_s"], 512 / 5120.0, delta=0.06)
        # 두번째는 자기 도착에서 잰다: 첫 요청의 디코드가 이미 끝났으니 큐잉이 없다
        self.assertAlmostEqual(second["ttft_s"], 512 / 5120.0, delta=0.06)

    def test_closed_loop_admits_next_only_after_completion(self):
        # 하네스 의미(검증 모드): 앞 요청이 끝나야 다음이 간다 — 겹침이 없으니
        # 두번째 TTFT 는 첫 요청 e2e 뒤에도 프리필 값 그대로다.
        out = sim.run_once([512, 512], 96, CONTRACT, cost=self.FAST, can_async=False,
                           closed_loop=True)
        first, second = out["requests"]
        self.assertAlmostEqual(second["ttft_s"], 512 / 5120.0, delta=0.06)
        # 순차 실행: 두번째의 시작은 첫 e2e 뒤(도착=끝남 시각의 차로 확인)
        self.assertGreater(second["e2e_s"], 0.0)
        widths = out["decode_widths"]
        self.assertTrue(all(int(n) == 1 for n in widths), widths)   # 폭 1 만 허용

    def test_arrivals_from_record_and_validation(self):
        record = {"requests": [
            {"ctx": 2000, "ttft_s": 1.0, "decode_s": 4.0, "completion_tokens": 120,
             "decode_tok_s": 30.0, "tpot_ms": 33.3},
            {"ctx": 2000, "ttft_s": 1.0, "decode_s": 4.0, "completion_tokens": 120,
             "decode_tok_s": 30.0, "tpot_ms": 33.3},
            {"ctx": 32000, "ttft_s": 2.5, "decode_s": 4.0, "completion_tokens": 200,
             "decode_tok_s": 50.0, "tpot_ms": 20.0}],
            "prefill": [{"ctx": 2000, "tok": 2128, "cold_s": 2.4, "warm_s": 1.0},
                        {"ctx": 32000, "tok": 32660, "cold_s": 10.0, "warm_s": 9.5}],
            "decode": {"tokens_per_step": 3.0, "windows_med": 10.0, "num_spec": 5,
                       "acc_raw": 0.4, "fixed_pooled_step_s": 10.0}}
        derived = sim.arrivals_from_record(record)
        self.assertEqual(derived[0], [0.0, 5000.0, 10000.0])   # 요청마다 앞 요청의 e2e 뒤
        self.assertEqual(derived[1], [120, 120, 200])
        self.assertEqual(derived[2], [2128, 2128, 32660])      # 실제 토큰수, ctx 아님
        self.assertEqual(derived[3], [2000, 2000, 32000])
        # 폴딩: decode_ms 는 판정 채널의 역수, prefill 계단은 warm 실측 처리량
        cost = sim.fit_cost(record)
        self.assertAlmostEqual(cost.decode_ms, 100.0)
        self.assertAlmostEqual(cost.prefill_tok_s[2128], 2128.0)
        self.assertAlmostEqual(cost.prefill_tok_s[32660], 32660 / 9.5, places=1)
        # 검증 행: [예측] 표시가 붙은 값들이 기록에서 온 비교값과 짝을 이룬다
        sim_out = {"tokens_per_step": 3.02, "decode_step_s_phase": 10.1, "requests": [
            {"ctx": 2000, "ttft_s": 1.01, "e2e_s": 5.0, "tok_s": 29.8},
            {"ctx": 2000, "ttft_s": 1.00, "e2e_s": 5.1, "tok_s": 30.2},
            {"ctx": 32000, "ttft_s": 9.6, "e2e_s": 13.5, "tok_s": 49.5}]}
        rows = {label: (kind, delta) for label, kind, _r, _s, delta in
                sim.validate_against(record, sim_out)}
        self.assertEqual(rows["decode step/s"][0], "입력")
        self.assertEqual(rows["tokens/step"][0], "예측")
        self.assertEqual(rows["클라이언트 tok/s"][0], "예측")
        self.assertEqual(rows["TPOT ms"][0], "예측")
        self.assertAlmostEqual(rows["클라이언트 tok/s"][1], 0.2 / 30.0, places=3)
        self.assertAlmostEqual(rows["TPOT ms"][1], (1000 / 30.2 - 33.3) / 33.3, places=3)
        self.assertAlmostEqual(rows["e2e med ctx2K"][1], (5.05 - 5.0) / 5.0, places=2)


class StepSimTest(unittest.TestCase):
    def test_host_cost_by_kind(self):
        out = sim.run_once([200, 400], 32, CONTRACT)
        self.assertGreater(out["steps"]["prefill"], 0)
        self.assertGreater(out["steps"]["decode"], 0)
        for kind in ("prefill", "decode"):
            s = out["by_kind"][kind]
            self.assertGreater(s["med_ms"], 0.0)
            self.assertGreaterEqual(s["p95_ms"], s["med_ms"])
            self.assertGreaterEqual(s["max_ms"], s["p95_ms"])
        self.assertIsNone(out["headroom"])          # 장치 0에서는 여유분이 정의되지 않는다

    def test_async_cadence_tracks_serial_device(self):
        # 장치 50 ms 가 한 줄로 실행되면 캐던스는 1000/50 = 20 을 넘을 수 없고,
        # 숙주 비용(~0.1 ms)이 병목일 이유도 없다.
        out = sim.run_once([256], 40, CONTRACT, device_s=0.05, can_async=True, host_med_ms=0.1)
        self.assertGreater(out["async"], 0)
        self.assertGreater(out["decode_step_s_wall"], 12.0)
        self.assertLess(out["decode_step_s_wall"], 21.0)
        self.assertGreater(out["headroom"], 0.5)
        # async 스텝의 in-flight wall 은 depth 대기를 포함해 장치 시간보다 길다
        self.assertGreater(out["by_kind"]["decode"]["med_ms"], 50.0)

    def test_meta_is_measured_separately(self):
        out = sim.run_once([200], 16, CONTRACT, with_meta=True)
        self.assertIsNotNone(out["meta_med_us"])
        self.assertGreater(out["meta_med_us"], 0.0)

    def test_ring_dump_replays_identically(self):
        out = sim.run_once([200, 400], 24, CONTRACT)
        with tempfile.TemporaryDirectory() as d:
            dump = DeathDump(d, out["ring"], boot_id="roundtrip", signals=())
            dump.write_now()
            dump.close()
            stats = replay.ring_stats(Path(dump.path))
        self.assertEqual(stats["pushed"], out["steps"]["prefill"] + out["steps"]["decode"])
        for kind in ("prefill", "decode"):
            self.assertEqual(stats["by_kind"][kind]["steps"], out["by_kind"][kind]["steps"])
            self.assertAlmostEqual(stats["by_kind"][kind]["med_ms"],
                                   out["by_kind"][kind]["med_ms"], delta=0.05)


# 엔진의 실제 /metrics 노출 형식(engine/base/serve.metrics)을 축약한 fixture.
def _rescrape(text: str, **changes) -> str:
    """일부 계열의 값만 바꾼 스크랩. changes 키는 계열 이름의 끝 부분(첫 일치만)."""
    lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            for frag, val in changes.items():
                if frag in line:
                    line = f"{line.rsplit(' ', 1)[0]} {val}"
                    break
        lines.append(line)
    return "\n".join(lines)


METRICS_A = """
# HELP vllm:iteration_tokens_total_count model steps
# TYPE vllm:iteration_tokens_total_count counter
vllm:iteration_tokens_total_count{engine="st"} 1000
vllm:generation_tokens_total{engine="st"} 3309
vllm:spec_decode_num_accepted_tokens_total{engine="st"} 1356
vllm:spec_decode_num_draft_tokens_total{engine="st"} 3660
vllm:num_requests_running{engine="st"} 1
vllm:num_requests_waiting{engine="st"} 0
st:step_seconds_bucket{engine="st",kind="prefill",le="0.25"} 3
st:step_seconds_bucket{engine="st",kind="prefill",le="+Inf"} 3
st:step_seconds_sum{engine="st",kind="prefill"} 0.6
st:step_seconds_count{engine="st",kind="prefill"} 3
st:step_seconds_bucket{engine="st",kind="decode",le="0.05"} 90
st:step_seconds_bucket{engine="st",kind="decode",le="0.1"} 100
st:step_seconds_bucket{engine="st",kind="decode",le="+Inf"} 100
st:step_seconds_sum{engine="st",kind="decode"} 7.5
st:step_seconds_count{engine="st",kind="decode"} 100
vllm:time_to_first_token_seconds_bucket{engine="st",le="1.0"} 2
vllm:time_to_first_token_seconds_bucket{engine="st",le="+Inf"} 2
vllm:time_to_first_token_seconds_sum{engine="st"} 1.4
vllm:time_to_first_token_seconds_count{engine="st"} 2
"""
METRICS_B = _rescrape(METRICS_A,
                      iteration_tokens_total_count=1020,
                      generation_tokens_total=3379,
                      num_accepted_tokens_total=1376,
                      num_draft_tokens_total=3710,
                      **{'kind="decode",le="0.05"': 130,
                         'kind="decode",le="0.1"': 150,
                         'kind="decode",le="+Inf"': 150,
                         'kind="decode"} 7.5': 9.5,
                         'kind="decode"} 100': 150})


class StepPeekTest(unittest.TestCase):
    def setUp(self):
        self.a = peek.parse_metrics(METRICS_A)
        self.b = peek.parse_metrics(METRICS_B)

    def test_parse_and_series(self):
        self.assertEqual(peek.series(self.a, "vllm:iteration_tokens_total_count"), 1000.0)
        self.assertIsNone(peek.series(self.a, "vllm:spec_decode_num_drafts_total"))
        # 없는 카운터는 0이 아니다 — None
        self.assertIsNone(peek.series(self.a, "st:steps_prefill_total"))

    def test_hist_delta_math(self):
        h = peek.hist_delta(self.a, self.b, "st:step_seconds", kind="decode")
        self.assertEqual(h["count"], 50)
        self.assertAlmostEqual(h["mean"], 0.04)                  # (9.5-7.5)/50
        self.assertAlmostEqual(h["p50"], 0.05 * 25 / 40)         # 버킷 [0,0.05) 안 보간
        self.assertAlmostEqual(h["p95"], 0.05 + (7.5 / 10) * 0.05)
        # 라벨이 다른 계열(prefill)은 증분이 없다
        self.assertIsNone(peek.hist_delta(self.a, self.b, "st:step_seconds", kind="prefill"))
        # 계열이 아예 없으면 None
        self.assertIsNone(peek.hist_delta(self.a, self.b, "vllm:inter_token_latency_seconds"))

    def test_window_rates(self):
        w = peek.window(self.a, self.b, 2.0)
        self.assertAlmostEqual(w["step_s"], 10.0)                # 20 스텝 / 2 s
        self.assertAlmostEqual(w["gen_tok_s"], 35.0)
        self.assertAlmostEqual(w["acc_raw"], 20 / 50)
        self.assertEqual(w["step_ms"]["count"], 50)
        self.assertEqual(w["gauges"]["vllm:num_requests_running"], 1.0)
        # 증분이 없는 ttft: 요청이 그 창에 끝나지 않았다 → None (0이 아니다)
        self.assertIsNone(w["ttft"])

    def test_decode_quantiles_exclude_prefill_buckets_with_the_same_bounds(self):
        expected = peek.hist_delta(self.a, self.b, "st:step_seconds", kind="decode")
        a, b = dict(self.a), dict(self.b)
        for le in ("0.01", "0.05", "0.1", "+Inf"):
            key = f'st:step_seconds_bucket{{engine="st",kind="prefill",le="{le}"}}'
            a[key], b[key] = 0, 1000
        actual = peek.hist_delta(a, b, "st:step_seconds", kind="decode")
        self.assertEqual(actual, expected)

    def test_missing_step_counter_degrades(self):
        stripped = {k: v for k, v in self.b.items() if "iteration_tokens" not in k}
        w = peek.window(self.a, stripped, 2.0)
        self.assertIsNone(w["step_s"])
        self.assertAlmostEqual(w["gen_tok_s"], 35.0)             # 나머지는 잰다

    def test_summarize_and_track_roundtrip(self):
        s = peek.summarize([(0.0, self.a), (1.0, self.b)])
        self.assertEqual(s["windows"], 1)
        self.assertAlmostEqual(s["step_s_med"], 20.0)
        # 저장용 subset 은 json 직렬화가 되고, 다시 summarize 했을 때 같은 중앙값
        line = json.dumps({"monotonic": 1.0, "series": peek.track(self.b)})
        back = json.loads(line)
        s2 = peek.summarize([(0.0, peek.track(self.a)),
                             (back["monotonic"], back["series"])])
        self.assertEqual(s2["step_s_med"], s["step_s_med"])


def _onepass(name, step_s, warm_s, warm_tok_s, acc):
    return {"name": name,
            "prefill": [{"ctx": 2000, "cold_s": warm_s * 2, "warm_s": warm_s,
                         "cold_tok_s": warm_tok_s / 2, "warm_tok_s": warm_tok_s}],
            "decode": {"windows_med": step_s, "windows": [step_s - 0.5, step_s + 0.5],
                       "acc_raw": acc, "tokens_per_step": 1 + 5 * acc},
            "requests": [{"ttft_s": 1.0, "tpot_ms": 25.0}, {"ttft_s": 3.0, "tpot_ms": 35.0}],
            "quality": {"ok": 9, "total": 9}, "korean": {"dirty": 0, "n": 3}}


class StepReplayTest(unittest.TestCase):
    def test_onepass_summary(self):
        s = replay.onepass_summary(_onepass("A", 10.0, 1.0, 2000.0, 0.5))
        self.assertEqual(s["step_s"], 10.0)
        self.assertEqual(s["windows"], 2)
        self.assertEqual(s["ttft_s_med"], 2.0)
        self.assertEqual(s["tpot_ms_med"], 30.0)
        self.assertEqual(s["tpot_p95"], 35.0)
        self.assertEqual(s["quality"]["ok"], 9)
        self.assertIsNone(s["issues"])

    def test_onepass_delta(self):
        base = replay.onepass_summary(_onepass("A", 10.0, 1.0, 2000.0, 0.50))
        cand = replay.onepass_summary(_onepass("B", 11.0, 0.9, 2100.0, 0.52))
        rows = {label: ch for label, _v0, _v1, ch in replay.onepass_delta(base, cand)}
        self.assertAlmostEqual(rows["decode step/s"], 0.10)
        self.assertAlmostEqual(rows["ctx2K warm TTFT"], -0.10)
        self.assertAlmostEqual(rows["ctx2K warm tok/s"], 0.05)
        self.assertAlmostEqual(rows["raw 수락률"], 0.02)         # 포인트 차이

    def test_bracket_summary(self):
        rec = {"name": "E", "tag": "base", "conc": 1,
               "reps": [{"tok_s": 70.0, "win_step_s": [19.0, 21.0]},
                        {"tok_s": 72.0, "win_step_s": [20.0]}]}
        s = replay.bracket_summary(rec)
        self.assertEqual(s["win_step_s_med"], 20.0)
        self.assertEqual(s["windows"], 3)
        self.assertEqual(s["tok_s_med"], 71.0)

    def test_load_detects_kinds(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "onepass.jsonl"
            p.write_text(json.dumps(_onepass("A", 10.0, 1.0, 2000.0, 0.5)) + "\n",
                         encoding="utf-8")
            kind, payload = replay._load(p)
            self.assertEqual(kind, "onepass")
            self.assertEqual(payload[0]["name"], "A")
            q = Path(d) / "peek.jsonl"
            q.write_text(json.dumps({"monotonic": 0.0, "series": peek.track(peek.parse_metrics(METRICS_A))}) + "\n"
                         + json.dumps({"monotonic": 1.0, "series": peek.track(peek.parse_metrics(METRICS_B))}) + "\n",
                         encoding="utf-8")
            kind2, payload2 = replay._load(q)
            self.assertEqual(kind2, "peek")
            self.assertAlmostEqual(payload2["step_s_med"], 20.0)


if __name__ == "__main__":
    unittest.main()
