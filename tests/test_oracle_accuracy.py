"""Oracle accuracy contracts: known token counts, measured shapes and independent records."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bench import oracle_records as records
from bench import step_sim as sim
from engine.base import instruments
from engine.base.scheduler import Contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Contract(16, 1024, 7, 0.0, 8)


def request(ctx=512, prompt=512, gen=8, width=1, duration=.02):
    return dict(ctx=ctx, prompt_tokens=prompt, completion_tokens=gen, concurrency=width,
                ttft_s=.004, elapsed_s=duration, decode_s=duration-.004,
                decode_tok_s=(gen-1)/(duration-.004), tpot_ms=1000*(duration-.004)/(gen-1))


def record(name="training", width=1):
    return dict(name=name, git="same-build", image="same-image", requests=[request(width=width) for _ in range(width)],
                prefill=[dict(ctx=512, tok=512, cold_s=.004, warm_s=.004)],
                decode=dict(windows_med=1000., tokens_per_step=1., num_spec=0, acc_raw=0.))


class AccountingTests(unittest.TestCase):
    def cost(self, **kw):
        return sim.CostModel(**dict(dict(k=7, acc=1., decode_ms=1., prefill_tok_s={512: 512000.}), **kw))

    def test_one_token_finishes_at_prefill_without_decode(self):
        out = sim.run_once([512,512], 1, CONTRACT, cost=self.cost(), closed_loop=True)
        self.assertEqual(out["steps"]["decode"], 0)
        self.assertEqual(out["committed_decode_tokens"], 0)
        self.assertIsNone(out["tokens_per_step"])
        self.assertEqual(len(out["requests"]), 2)
        for q in out["requests"]:
            self.assertEqual(q["decode_s"], 0.)
            self.assertEqual(q["completion_tokens"], 1)

    def test_tail_clips_tokens_and_ghost_rows_do_not_change_acceptance(self):
        for asynchronous in (False, True):
            out = sim.run_once([512,512], [2,10], CONTRACT, cost=self.cost(), can_async=asynchronous)
            self.assertEqual(out["committed_decode_tokens"], 10)  # (2-1)+(10-1)
            self.assertEqual(out["active_decode_row_steps"], 3)
            self.assertAlmostEqual(out["tokens_per_step"], 10/3, places=3)
            self.assertGreaterEqual(out["launched_decode_row_steps"], 3)

    def test_phase_rate_uses_actual_width_and_context_service_sum(self):
        cost = self.cost(k=0, acc=0., decode_ms=1., decode_ms_per_row=3., decode_ms_by_ctx={1:2., 512:4.})
        for asynchronous in (False, True):
            out = sim.run_once([512]*4, 12, CONTRACT, cost=cost, can_async=asynchronous)
            widths = {int(w): n for w,n in out["decode_widths"].items()}
            self.assertIn(4, widths)
            service = sum(n * (4.+3.*(w-1))/1000 for w,n in widths.items())
            self.assertAlmostEqual(out["modeled_decode_service_s"], service)
            self.assertAlmostEqual(out["decode_step_s_phase"], sum(widths.values())/service, delta=.01)
            self.assertLess(out["decode_step_s_phase"], 250.)

    def test_concurrent_waves_retire_before_the_next_wave(self):
        out = sim.run_once([512]*8, 16, CONTRACT, cost=self.cost(k=0, acc=0.),
                           groups=[0]*4+[1]*4, can_async=True)
        self.assertEqual(len(out["requests"]), 8)
        self.assertIn("4", out["decode_widths"])
        self.assertTrue(all(int(w) <= 4 for w in out["decode_widths"]))
        self.assertEqual([q["group"] for q in out["requests"]], [0]*4+[1]*4)
        self.assertLessEqual(max(q["completed_s"] for q in out["requests"][:4]),
                             min(q["arrival_s"] for q in out["requests"][4:]))

    def test_cpu_simulation_never_probes_device_memory(self):
        with patch.object(instruments, "_dev_free_bytes", side_effect=AssertionError("GPU query")):
            out = sim.run_once([512], 2, CONTRACT, cost=self.cost(), can_async=False)
        self.assertEqual(out["committed_decode_tokens"], 1)
        self.assertFalse(instruments.Recorder(memory_sampling=False).as_dict()["memory_sampling"])
        with patch.object(instruments, "_dev_free_bytes", side_effect=[100, 80]) as memory:
            default = instruments.Recorder()
            with default.phase("normal"):
                pass
            self.assertEqual(memory.call_count, 2)
            self.assertEqual(default.root.children[0].dev_bytes, 20)

    def test_shared_cold_key_with_different_real_prompt_lengths(self):
        cost = self.cost(cold_extra_s={512:.1})
        model = sim.NullModel(cost)
        model.prompt, model.target = {1:500, 2:512}, {1:501, 2:513}
        model.cold_key = {1:512, 2:512}
        with patch.object(sim, "_delay") as delay:
            model.prefill(1,0,500,[],0)
            model.prefill(2,0,512,[],0)
        self.assertEqual(sum(call.args == (.1,) for call in delay.call_args_list), 1)


class EvidenceTests(unittest.TestCase):
    def test_zero_acceptance_and_zero_k_are_values_not_missing(self):
        self.assertEqual(sim.fit_cost(record()).k, 0)
        self.assertEqual(sim.fit_cost(record()).acc, 0.)
        r = record()
        r["decode"]["num_spec"] = 7
        self.assertEqual(sim.fit_cost(r).acc, 0.)
        self.assertEqual(sim.fit_cost(r).tokens_per_step_mean(), 1.)

    def test_client_fit_cannot_be_shadowed_by_windows_context_ladder(self):
        r = record()
        r["requests"] = [request(ctx=512), request(ctx=1024)]
        r["requests"][0]["decode_tok_s"] = 100.
        r["requests"][1]["decode_tok_s"] = 50.
        r["decode"]["windows_by_ctx"] = {"512":[1000.], "1024":[500.]}
        cost = sim.fit_cost(r, channel="client")
        self.assertAlmostEqual(cost.decode_delay(1,512), .010)
        self.assertAlmostEqual(cost.decode_delay(1,1024), .020)
        for q in r["requests"]:
            q.pop("decode_tok_s")
        self.assertIsNone(sim.fit_cost(r, channel="client"))

    def test_fitted_c4_rate_is_anchored_at_c4_not_reused_as_c1(self):
        from dataclasses import replace
        r = record(width=4)
        r["decode"]["windows_med"] = 100.
        cost = replace(sim.fit_cost(r), decode_ms_per_row=2.)
        self.assertEqual(cost.decode_reference_width, 4)
        self.assertAlmostEqual(cost.decode_delay(4,512), .010)
        self.assertAlmostEqual(cost.decode_delay(1,512), .004)

    def test_invalid_evidence_and_nonfinite_rates_cannot_calibrate(self):
        for mutation in (dict(valid=False), dict(evidence_issues=["compile in measured window"])):
            self.assertIsNone(sim.fit_cost(dict(record(), **mutation)))
        for value in (0., -1., float("nan"), float("inf")):
            r = record()
            r["decode"]["windows_med"] = value
            self.assertIsNone(sim.fit_cost(r))
        r = record()
        r["decode"]["windows_by_ctx"] = {"512":[0., -1., float("nan"), 1000.]}
        self.assertIsNotNone(sim.fit_cost(r))

    def test_width_fit_requires_matching_runtime_shape_k_and_complete_waves(self):
        c1, c4 = record(), record(width=4)
        c4["decode"]["windows_med"] = 250.
        self.assertIsNotNone(sim.fold_width_from_records([c1,c4]))
        for change in (dict(git="other"), dict(image="other"), dict(valid=False), dict(git=None)):
            self.assertIsNone(sim.fold_width_from_records([c1,dict(c4,**change)]))
        bad = copy.deepcopy(c4)
        bad["decode"]["num_spec"] = 7
        self.assertIsNone(sim.fold_width_from_records([c1,bad]))
        bad = copy.deepcopy(c4)
        bad["requests"][0]["ctx"] = 1024
        self.assertIsNone(sim.fold_width_from_records([c1,bad]))
        bad = dict(c4, requests=c4["requests"][:3])
        self.assertIsNone(sim.fold_width_from_records([c1,bad]))

    def test_actual_prompt_tokens_and_concurrent_offsets(self):
        r = record(width=4)
        r["requests"] += [request(prompt=600, duration=.03)]
        r["requests"][0]["prompt_tokens"] = 500
        plan = records.workload(r)
        self.assertEqual(plan["prompts"], [500,512,512,512,600])
        self.assertEqual(plan["arrive_ms"], [0.,0.,0.,0.,20.])
        self.assertEqual(plan["groups"], [0,0,0,0,1])
        self.assertEqual(plan["cold_keys"], [512]*5)
        r["requests"] = r["requests"][:3]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            records.workload(r)

    def test_missing_completion_count_is_not_invented_as_256(self):
        r = record()
        del r["requests"][0]["completion_tokens"]
        with self.assertRaisesRegex(ValueError, "completion_tokens"):
            records.workload(r)

    def test_new_and_historical_c4_artifacts_are_visible(self):
        r = record()
        r["c4"] = [dict(ctx=512, concurrency=4, valid=True, requests=[request() for _ in range(4)])]
        c1, c4 = list(records.views(r))
        self.assertEqual(c1["decode"]["windows_med"], 1000.)
        self.assertEqual(c4["decode"], {})  # no invented C4 rate from the C1 counters
        self.assertEqual(records.workload(c4)["groups"], [0]*4)
        path = ROOT / "measurements/c4_scaling_20260913/c4-20260912T232937.json"
        historical = records.load_records(path)[0]
        views = list(records.views(historical))
        self.assertEqual(len(views), 4)
        for view in views[1:]:
            plan = records.workload(view)
            self.assertEqual(plan["groups"], [0]*4)
            self.assertTrue(all(source == "request" for source in plan["prompt_sources"]))

    def test_warm_comparison_excludes_cold_on_both_sides(self):
        r = record()
        r["requests"] = [request(), request()]
        r["requests"][0]["ttft_s"], r["requests"][1]["ttft_s"] = 10., 2.
        output = dict(requests=[dict(ctx=512,ttft_s=10.,e2e_s=11.,tok_s=100.),
                               dict(ctx=512,ttft_s=2.,e2e_s=3.,tok_s=100.)])
        rows = {label:(kind,delta) for label,kind,_,_,delta in sim.validate_against(r,output)}
        self.assertEqual(rows["TTFT(warm) ctx0K"][1], 0.)
        self.assertEqual(rows["TTFT(cold) ctx0K"][0], "입력파생")
        r["prefill"] = []
        self.assertTrue(sim.validate_against(r, output))  # no None.get() crash
        self.assertTrue(all(row[1] == "검증" for row in sim.validate_against(r,output,mode="holdout")))

    def test_aggregate_uses_client_span_not_async_drain_wall(self):
        r = record(width=4)
        r["aggregate_output_tok_s"] = 32.
        output = dict(requests=[dict(ctx=512,ttft_s=.1,e2e_s=1.,tok_s=8.,completion_tokens=8) for _ in range(4)],
                      wall_s=2., request_span_s=1.)
        row = next(row for row in sim.validate_against(r,output) if row[0].startswith("전체 출력"))
        self.assertEqual(row[3:], (32.,0.))


class ValidationCliTests(unittest.TestCase):
    def test_empty_input_is_a_json_failure_not_a_successful_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"empty.jsonl"
            path.write_text("")
            command = [sys.executable,str(ROOT/"bench/step_sim.py"),"--against",str(path),"--no-calib","--json"]
            run = subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertEqual(run.returncode,2)
            self.assertEqual(json.loads(run.stdout)["skipped"][0]["reason"],"no records")

    def test_holdout_freezes_training_and_writes_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train, target, output = [directory / n for n in ("train.json", "target.jsonl", "report.json")]
            train.write_text(json.dumps(record(), indent=2))
            validation = record("holdout")
            validation["decode"].update(windows_med=100., num_spec=7, acc_raw=.9)
            validation["requests"][0]["elapsed_s"] = .03
            target.write_text(json.dumps(validation)+"\n")
            command = [sys.executable, str(ROOT/"bench/step_sim.py"), "--fit-from", str(train),
                       "--against", str(target), "--no-calib", "--json", "--validation-output", str(output)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report, json.loads(output.read_text()))
            self.assertEqual(report["mode"], "holdout")
            cost = report["evaluations"][0]["simulation"]["cost"]
            self.assertEqual((cost["k"], cost["acc"], cost["decode_ms"]), (0, 0., 1.))
            renamed = dict(record(), name="renamed copy")
            target.write_text(json.dumps(renamed))
            run = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("calibration observations", run.stderr)

    def test_c4_requires_width_evidence_and_is_replayed_with_explicit_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"c4.json"
            r = record(width=4)
            r["decode"]["windows_med"] = 250.
            path.write_text(json.dumps(r))
            command = [sys.executable, str(ROOT/"bench/step_sim.py"), "--against", str(path), "--no-calib", "--json"]
            refused = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("uncalibrated", json.loads(refused.stdout)["skipped"][0]["reason"])
            bad_training = subprocess.run(command + ["--fit-from", str(path)], capture_output=True,
                                          text=True, timeout=15)
            self.assertEqual(bad_training.returncode, 2)
            self.assertIn("no valid C1 calibration record", bad_training.stderr)
            run = subprocess.run(command + ["--decode-ms-per-row", "1"], capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            out = json.loads(run.stdout)["evaluations"][0]
            self.assertEqual(out["workload"]["groups"], [0]*4)
            self.assertIn("4", out["simulation"]["decode_widths"])


if __name__ == "__main__":
    unittest.main()
