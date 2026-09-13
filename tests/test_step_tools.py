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
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import step_peek as peek                     # noqa: E402
import step_replay as replay                 # noqa: E402
import step_kernels as kern                 # noqa: E402
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

    def test_pick_last_takes_the_tail_and_zero_takes_all(self):
        # 원장은 append 이므로 '최근 기록'은 끝의 줄 — 대기 작업 재검증의 선택 규칙
        records = ["a", "b", "c", "d"]
        self.assertEqual(sim.pick_last(records, 1), ["d"])
        self.assertEqual(sim.pick_last(records, 3), ["b", "c", "d"])
        self.assertEqual(sim.pick_last(records, 0), records)

    def test_front_is_charged_per_request_not_queue_wait(self):
        # 문의 앞면(입장·토큰화)은 요청마다 한 번 — TTFT 에는 들지만 큐 대기는 아니다
        cost = replace(self.FAST, front_ms=60.0)
        out = sim.run_once([512], 64, CONTRACT, cost=cost, can_async=False)
        (q,) = out["requests"]
        self.assertAlmostEqual(q["ttft_s"], 0.06 + 512 / 5120.0, delta=0.05)
        self.assertAlmostEqual(q["queue_wait_s"], 0.0, delta=0.05)

    def test_cold_tail_is_charged_once_per_prompt_length(self):
        # JIT 꼬리는 그 프롬프트 길이의 첫 요청만 낸다 — 둘째는 warm
        cost = replace(self.FAST, cold_extra_s={512: 0.4})
        out = sim.run_once([512, 512], 64, CONTRACT, cost=cost, can_async=False,
                           closed_loop=True)
        first, second = out["requests"]
        self.assertAlmostEqual(first["ttft_s"], 0.4 + 512 / 5120.0, delta=0.06)
        self.assertAlmostEqual(second["ttft_s"], 512 / 5120.0, delta=0.06)

    def test_decode_ladder_follows_context(self):
        # windows_by_ctx 폴딩: 컨텍스트별로 다른 decode 스텝 시간을 계단이 담는다
        cost = sim.CostModel(k=0, acc=0.0, decode_ms=5.0,
                             decode_ms_by_ctx={512: 5.0, 4096: 20.0},
                             prefill_tok_s={512: 5120.0, 4096: 5120.0}, name="ladder")
        out = sim.run_once([512, 4096], 64, CONTRACT, cost=cost, can_async=False,
                           closed_loop=True)
        short, long_ = out["requests"]
        # 클라이언트 tok/s = 1 토큰/스텝 ÷ 스텝 시간 — 5ms 대 20ms 면 4배
        self.assertAlmostEqual(short["tok_s"] / long_["tok_s"], 4.0, delta=0.5)

    def test_acc_hist_samples_the_measured_shape(self):
        # 실측 accepted 분포(쌍봉 0/k)를 주면 쌍봉으로 나온다 — 기하 추첨은 평균만 맞춘다
        cost = sim.CostModel(k=5, acc=0.5, decode_ms=1.0, prefill_tok_s={512: 5120},
                             acc_hist=[1, 0, 0, 0, 0, 1], name="hist")
        self.assertAlmostEqual(cost.tokens_per_step_mean(), 3.5)
        out = sim.run_once([512], 400, CONTRACT, cost=cost, can_async=False)
        self.assertAlmostEqual(out["tokens_per_step"], 3.5, delta=0.3)

    def test_fit_cost_folds_front_cold_and_ctx_ladder(self):
        record = {"name": "F", "requests": [
            {"ctx": 2000, "ttft_s": 1.5, "decode_s": 4.0, "completion_tokens": 120},
            {"ctx": 2000, "ttft_s": 1.1, "decode_s": 4.0, "completion_tokens": 120},
            {"ctx": 2000, "ttft_s": 1.1, "decode_s": 4.0, "completion_tokens": 120}],
            "prefill": [{"ctx": 2000, "tok": 1000, "cold_s": 1.5, "warm_s": 1.0}],
            "decode": {"tokens_per_step": 3.0, "windows_med": 20.0, "num_spec": 5,
                       "acc_raw": 0.4, "fixed_pooled_step_s": 20.0,
                       "windows_by_ctx": {"2000": [20, 20], "32000": [10, 10]}}}
        cost = sim.fit_cost(record)
        self.assertAlmostEqual(cost.decode_ms, 50.0)
        self.assertAlmostEqual(cost.decode_ms_by_ctx[2000], 50.0)
        self.assertAlmostEqual(cost.decode_ms_by_ctx[32000], 100.0)
        # 앞면: warm 중앙값 1.1 − 프리필(1000 tok / 1000 tok/s = 1.0s) = 0.1s
        self.assertAlmostEqual(cost.front_ms, 100.0, delta=5.0)
        self.assertAlmostEqual(cost.cold_extra_s[1000], 0.5, places=2)

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
        # 줄이 없으니 큐 대기도 없다
        self.assertEqual(out["queue_wait"]["max_s"], 0.0)

    def test_queue_wait_behind_a_running_prefill(self):
        # 프리필은 한 번에 하나(D10 직렬): 첫 청크(2s) 도중 도착한 요청은 그 청크가
        # 끝나기를 기다린다 — 1.5s 대기가 잡혀야 계산이라고 할 수 있다.
        cost = sim.CostModel(k=0, acc=0.0, decode_ms=1.0, prefill_tok_s={4096: 2048.0}, name="q")
        c2 = sched.Contract(16, 1024, 0, 0.0, 8)
        out = sim.run_once([4096, 4096], 16, c2, cost=cost, can_async=False,
                           arrive_ms=[0.0, 500.0])
        first, second = out["requests"]
        self.assertLess(first["queue_wait_s"], 0.05)
        self.assertAlmostEqual(second["queue_wait_s"], 1.5, delta=0.4)
        self.assertAlmostEqual(second["ttft_s"],
                               second["queue_wait_s"] + second["prefill_s"], delta=0.02)
        self.assertEqual(out["queue_wait"]["max_s"], second["queue_wait_s"])

    def test_queue_wait_is_the_starvation_valve(self):
        # D10: 살아있는 디코더는 max_wait_s 만큼 보호된다 — 두번째 요청의 큐 대기가
        # 곧 밸브 시간(2s)이 된다.
        cost = sim.CostModel(k=2, acc=0.5, decode_ms=10.0, prefill_tok_s={512: 5120.0}, name="v")
        c2 = sched.Contract(16, 1024, 3, 2.0, 8)
        out = sim.run_once([512, 512], 600, c2, cost=cost, can_async=False,
                           arrive_ms=[0.0, 200.0])
        first, second = out["requests"]
        self.assertLess(first["queue_wait_s"], 0.1)
        self.assertAlmostEqual(second["queue_wait_s"], 2.0, delta=0.4)
        self.assertGreater(second["queue_wait_s"], first["queue_wait_s"])

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
    def test_live_token_counter_is_distinct_from_completed_requests(self):
        names = ('st:generation_tokens_committed_total', 'st:decode_row_steps_total', 'st:steps_decode_total')
        a, b = dict(self.a), dict(self.a)
        for name, first, second in zip(names, (100, 40, 10), (220, 72, 18)):
            a[name+'{engine="st"}'], b[name+'{engine="st"}'] = first, second
        result = peek.window(peek.track(a), peek.track(b), 1.)
        self.assertEqual(result['gen_tok_s'], 0.)
        self.assertEqual(result['committed_tok_s'], 120.)
        self.assertEqual(result['mean_decode_rows'], 4.)
        self.assertIsNone(peek.window(self.a, self.b, 1.)['committed_tok_s'])
        self.assertIsNone(peek.window(b, a, 1.)['committed_tok_s'])

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

    def test_decode_quantiles_exclude_active_prefill_with_shared_bounds(self):
        a, b = dict(self.a), dict(self.b)
        for bound in ("0.05", "0.1"):
            key = f'st:step_seconds_bucket{{engine="st",kind="prefill",le="{bound}"}}'
            a[key], b[key] = 0, 1000
        h = peek.hist_delta(a, b, "st:step_seconds", kind="decode")
        self.assertAlmostEqual(h["p50"], 0.05 * 25 / 40)
        self.assertAlmostEqual(h["p95"], 0.05 + (7.5 / 10) * 0.05)

    def test_changed_bucket_bounds_do_not_invent_quantiles(self):
        b = dict(self.b)
        old = 'st:step_seconds_bucket{engine="st",kind="decode",le="0.1"}'
        b[old.replace('"0.1"', '"0.2"')] = b.pop(old)
        h = peek.hist_delta(self.a, b, "st:step_seconds", kind="decode")
        self.assertEqual(h["count"], 50)
        self.assertAlmostEqual(h["mean"], .04)
        self.assertIsNone(h["p50"])
        self.assertIsNone(h["p95"])

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


class KernelBudgetTests(unittest.TestCase):
    """바이트 예산 조립(step_kernels) — 상수는 PR #838(c4_scaling_20260913) 실측."""

    def test_distinct_experts_uniform_and_measured(self):
        # 균등 가정(상한 쪽): 7 토큰 ≈ 52.6 / 28 토큰 ≈ 157
        self.assertAlmostEqual(kern.distinct_experts(7), 52.6, delta=0.5)
        self.assertLess(kern.distinct_experts(7), 7 * 8)
        self.assertEqual(kern.distinct_experts(0), 0.0)
        # 실측 멱법칙(라우팅 몰림): U(7)=33, U(28)=95 — #838 §6 역산
        b = kern.EngineBytes()
        self.assertAlmostEqual(kern.distinct_experts(7, gamma=b.routing_gamma, scale=b.routing_scale), 33.0, delta=0.3)
        self.assertAlmostEqual(kern.distinct_experts(28, gamma=b.routing_gamma, scale=b.routing_scale), 95.0, delta=0.5)

    def test_composition_reproduces_the_measured_step_table(self):
        # #838 §3 의 단계 합(forward + propose + observe) 열두 지점을 ±10% 안에.
        b = kern.EngineBytes()
        measured = {(2000, 1): 51.1, (2000, 2): 77.5, (2000, 3): 99.7, (2000, 4): 115.6,
                    (32000, 1): 50.1, (32000, 2): 83.4, (32000, 3): 105.6, (32000, 4): 125.0,
                    (128000, 1): 55.1, (128000, 2): 86.0, (128000, 3): 104.9, (128000, 4): 131.9}
        worst = 0.0
        for (ctx, w), m in measured.items():
            got = kern.decode_step(b, ctx, w).total()
            worst = max(worst, abs(got / m - 1))
        self.assertLess(worst, 0.10, f"worst residual {worst:.1%}")

    def test_composition_explains_measured_flatness(self):
        b = kern.EngineBytes()
        lo = kern.decode_step(b, 2000)
        hi = kern.decode_step(b, 128000)
        # 컨텍스트에 자라는 항은 인덱서·MLA 뿐: 2K→128K 잔여가 스텝의 ~8% 이내 (실측: 평탄)
        self.assertLess(hi.total() - lo.total(), 0.08 * hi.total())

    def test_knob_moves_the_step_by_its_bytes(self):
        b = kern.EngineBytes()
        base = kern.decode_step(b, 32000).total()
        # 드래프터 W4→bf16 복원: 2.03 GiB ÷ 207 GB/s ≈ 10.6 ms (실측 W4 는 3.4)
        got = kern.decode_step(replace(b, drafter_ms=10.6), 32000).total()
        self.assertAlmostEqual(base - got, 3.4 - 10.6, delta=0.01)
        self.assertAlmostEqual((2.03 - 0.55) * kern.GIB / 207e9 * 1e3, 7.68, delta=0.1)

    def test_width_prediction_is_sublinear(self):
        b = kern.EngineBytes()
        one = kern.decode_step(b, 32000, width=1).total()
        four = kern.decode_step(b, 32000, width=4).total()
        # 전문가·드래프터는 스텝에 한 번, 컨텍스트·비MoE 행 성분만 행마다 — 실측 2.3~2.6×
        self.assertLess(four / one, 4.0)
        self.assertGreater(four / one, 2.0)

    def test_reads_engine_facts_module(self):
        # 시뮬레이터가 엔진 자신의 사실 원천을 읽는다 — 엔진이 바뀌면 따라간다
        facts = kern.load_engine_facts()
        self.assertEqual(facts.get("tp"), 4)
        self.assertEqual(facts.get("spec_k"), 6)
        self.assertEqual(facts.get("chunk_align"), 2304)
        self.assertIn("facts.py", facts.get("source", ""))

    def test_folds_routing_from_timeline_artifact(self):
        # 합성 아티팩트에서 멱법칙 회복
        with tempfile.TemporaryDirectory() as d:
            art = Path(d) / "tl.json"
            art.write_text(json.dumps({"decode": [
                {"tokens": 7, "unique_experts_mean": 29.0},
                {"tokens": 28, "unique_experts_mean": 62.0}]}), encoding="utf-8")
            folded = kern.fold_routing_from_timeline(art)
            self.assertAlmostEqual(folded["routing_gamma"], 0.5608, delta=0.02)
            u7 = kern.distinct_experts(7, gamma=folded["routing_gamma"], scale=folded["routing_scale"])
            self.assertAlmostEqual(u7, 29.0, delta=0.2)
        # 실물 #838 아티팩트: 층별 실측 (7,28.9)...(28,62.0)
        real = Path("measurements/c4_scaling_20260913/decode-timeline-rank3.json")
        if real.exists():
            folded = kern.fold_routing_from_timeline(real)
            self.assertEqual([p[0] for p in folded["points"]], [7, 14, 21, 28])
            self.assertAlmostEqual(folded["points"][0][1], 28.9, delta=0.1)

    def test_kernel_table_and_prefill_fold_from_artifacts(self):
        tl = Path("measurements/c4_scaling_20260913/decode-timeline-rank3.json")
        if tl.exists():
            table = kern.fold_kernels_from_timeline(tl)
            self.assertEqual(len(table), 32)
            moe7 = next(e for e in table if "static" in e["kernel"] and e["rows"] == 7)
            self.assertAlmostEqual(moe7["median_us"], 868.0, delta=1.0)
            cls = kern.kernel_classes(table)
            self.assertGreater(len(cls["launch"]), len(cls["bytes"]))   # 대다수는 런치 바닥 급
        cp = Path("measurements/c4_scaling_20260913/chunk-profile-rank3-sf6.json")
        if cp.exists():
            fold = kern.fold_prefill_from_profile(cp)
            self.assertAlmostEqual(fold["ms_per_token"], 0.28795, delta=0.0002)
            self.assertAlmostEqual(fold["fixed_ms_per_chunk"], 269.5, delta=0.5)
            # 플릿 32K(9216 청크) 예측이 측정 모형의 10.89s 를 재현
            self.assertAlmostEqual(kern.prefill_ms(32545, 9216, fold, fleet=True) / 1000, 10.89, delta=0.05)

    def test_acc_hist_roundtrip_from_peek_scrapes(self):
        a = peek.parse_metrics('st:spec_accepted_per_step_total{engine="st",accepted="0"} 100\n'
                               'st:spec_accepted_per_step_total{engine="st",accepted="6"} 10\n')
        b = peek.parse_metrics('st:spec_accepted_per_step_total{engine="st",accepted="0"} 110\n'
                               'st:spec_accepted_per_step_total{engine="st",accepted="6"} 18\n')
        hist = peek.acc_hist_from_scrapes(a, b)
        self.assertEqual(hist, [10.0, 0, 0, 0, 0, 0, 8.0])
        self.assertIsNone(peek.acc_hist_from_scrapes({}, {}))
        with tempfile.TemporaryDirectory() as d:
            q = Path(d) / "p.jsonl"
            q.write_text("\n".join(json.dumps({"monotonic": i, "series": s}) for i, s in enumerate((a, b))) + "\n",
                         encoding="utf-8")
            self.assertEqual(sim.acc_hist_from_peek(q), hist)

    def test_fold_width_waits_for_a_completed_c4_pair(self):
        c1 = {"requests": [{"ctx": 2000, "concurrency": 1}], "decode": {"fixed_pooled_step_s": 19.5}}
        c4 = {"requests": [{"ctx": 2000, "concurrency": 4}], "decode": {"fixed_pooled_step_s": 8.8}}
        folded = sim.fold_width_from_records([c1, c4])
        self.assertAlmostEqual(folded["decode_ms"], 51.3, delta=0.1)
        self.assertAlmostEqual(folded["decode_ms_per_row"], (113.6 - 51.3) / 3, delta=0.1)
        # 짝이 없으면 None — 지금 리포의 정답(완결 C=4 대기)
        self.assertIsNone(sim.fold_width_from_records([c1]))
        mixed = {"requests": [{"ctx": 2000, "concurrency": 1}, {"ctx": 2000, "concurrency": 4}],
                 "decode": {"windows_med": 10.0}}
        self.assertIsNone(sim.fold_width_from_records([mixed]))

    def test_chunk_structural_prefill(self):
        # 청크 모형(#838 §4): total = v·토큰 + F·청크 — 청크가 작을수록 토큰당 비싸다
        cost = sim.CostModel(k=0, acc=0.0, decode_ms=1.0,
                             prefill_ms_per_token=0.2879, prefill_fixed_ms_per_chunk=379.5,
                             prefill_tok_s={}, name="chunk")
        self.assertAlmostEqual(cost.prefill_delay(2128, 2128) * 1e3,
                               0.2879 * 2128 + 379.5, delta=1.0)
        self.assertAlmostEqual(cost.prefill_delay(9216, 2304) * 1e3,
                               0.2879 * 2304 + 379.5, delta=1.0)

    def test_composed_cost_has_every_coefficient(self):
        cost = sim.composed_cost(routing="measured")
        self.assertAlmostEqual(cost.decode_ms_by_ctx[2000], 51.4, delta=0.2)
        self.assertAlmostEqual(cost.decode_ms_by_ctx[128000], 55.4, delta=0.2)
        # 폭 계수는 플릿 실측(#838 §3)으로 못박고 조립값과 5% 안에서 일치한다
        self.assertAlmostEqual(cost.decode_ms_per_row, 21.03, delta=0.1)
        self.assertLess(abs(cost._composed_row_crosscheck - cost.decode_ms_per_row)
                        / cost.decode_ms_per_row, 0.05)
        self.assertAlmostEqual(cost.prefill_ms_per_token, 0.2879, delta=0.0005)
        self.assertAlmostEqual(cost.prefill_fixed_ms_per_chunk, 269.5 + 110.0, delta=1.0)
        art = sim.composed_cost(routing="artifact")
        self.assertLess(art.decode_ms_by_ctx[2000], cost.decode_ms_by_ctx[2000])  # U(7)=29.3 < 33

    def test_conc_mode_runs_rows_together(self):
        fast = sim.CostModel(k=2, acc=0.5, decode_ms=2.0,
                             prefill_tok_s={512: 5120.0, 2048: 5120.0}, name="conc")
        out = sim.run_once([512] * 3, 64, CONTRACT, cost=fast, can_async=False)
        widths = {int(n) for n in out["decode_widths"]}
        self.assertIn(3, widths)                       # 세 행이 한 스텝에 함께 디코드
        waits = [q["queue_wait_s"] for q in out["requests"]]
        self.assertGreater(max(waits), 0.0)            # 동시 도착: 직렬 프리필 뒤에 줄이 선다

    def test_coexist_penalty_only_beside_live_decoders(self):
        # 혼자 프리필(하네스 순차)에는 벌이 없다 — 11개 기록 폴드아웃이 지키던 불변
        cost = sim.CostModel(k=0, acc=0.0, decode_ms=5.0,
                             prefill_ms_per_token=0.1, prefill_fixed_ms_per_chunk=100.0,
                             prefill_tok_s={}, name="co")
        out = sim.run_once([512, 512], 64, CONTRACT, cost=cost, can_async=False,
                           closed_loop=True)
        first, second = out["requests"]
        self.assertAlmostEqual(second["ttft_s"], (0.1 * 512 + 100.0) / 1e3, delta=0.03)
        # 동시 도착: 둘째의 프리필은 디코더 옆 — 공존 벌 1.27× 이상
        out2 = sim.run_once([512, 512], 256, CONTRACT, cost=cost, can_async=False)
        a, b = sorted(out2["requests"], key=lambda q: q["seq"])
        solo = (0.1 * 512 + 100.0) / 1e3
        self.assertGreaterEqual(b["ttft_s"], solo * 1.2)

    def test_width_from_fleet_stage_table(self):
        # #838 §3 CUDA 이벤트 스테이지 표(2K): rows1 47.7 → rows4 110.8 → per_row 21.0
        folded = sim.fold_width_from_stage(sim.STAGE_WIDTH_2K)
        self.assertAlmostEqual(folded["decode_ms_per_row"], (110.8 - 47.7) / 3, delta=0.01)
        # 조립(composed)의 폭은 플릿 실측으로 못박고, 조립값과의 차가 교차검증이다
        cost = sim.composed_cost(routing="measured")
        self.assertAlmostEqual(cost.decode_ms_per_row, 21.03, delta=0.05)
        self.assertAlmostEqual(cost._composed_row_crosscheck, 20.7, delta=0.6)
        self.assertLess(abs(cost._composed_row_crosscheck - cost.decode_ms_per_row)
                        / cost.decode_ms_per_row, 0.05)
        self.assertEqual(folded["decode_ms"], 47.7)
