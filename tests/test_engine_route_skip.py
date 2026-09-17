"""The routed-slot skip of a capture arm (ACE, arXiv:2609.05228): off by default, exact weights when on, per pass.

measurements/st_ace_routes_20260917 sized the bytes a skip removes from real C=1 routes; this is the quality arm's
switch. Served steps never see it: `Glm53Net.route_skip` is None unless a capture pass sets it around an eager prefill.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from engine.profiles.glm53.net import Glm53Net, skip_route_weights

ROOT = Path(__file__).resolve().parents[1]


class SkipWeightsTests(unittest.TestCase):
    def weights(self):
        raw = torch.tensor([[0.30, 0.20, 0.15, 0.12, 0.09, 0.06, 0.05, 0.03],
                            [0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12]], dtype=torch.float32)
        return raw / raw.sum(-1, keepdim=True) * 2.5

    def test_slots_below_the_threshold_drop_and_the_rest_renormalise_to_the_scale(self):
        w = self.weights()
        out = skip_route_weights(w, 0.07, 2.5)
        self.assertEqual(out[0, 5:].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(torch.all(out[0, :5] > 0))
        self.assertTrue(torch.allclose(out.sum(-1), torch.full((2,), 2.5)))
        self.assertTrue(torch.allclose(out[0, :5] / out[0, :5].sum(), w[0, :5] / w[0, :5].sum()))
        self.assertTrue(torch.allclose(out[1], w[1]), "a flat row above the threshold is untouched")

    def test_the_top_one_survives_any_threshold(self):
        w = self.weights()
        out = skip_route_weights(w, 0.99, 2.5)
        self.assertEqual((out > 0).sum(-1).tolist(), [1, 1])
        self.assertAlmostEqual(float(out[0, 0]), 2.5, places=5)

    def test_a_threshold_outside_a_gate_is_refused(self):
        with self.assertRaises(ValueError):
            skip_route_weights(self.weights(), 1.0, 2.5)


class SelectRoutesTests(unittest.TestCase):
    def net(self, skip):
        routes = (torch.tensor([[5, 1, 2, 3, 4, 6, 7, 8]], dtype=torch.int32),
                  torch.tensor([[0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.06, 0.04]], dtype=torch.float32))
        return SimpleNamespace(F=SimpleNamespace(topk_experts=8, routed_scale=2.5), p={"L3.moe.bias": None},
                               lanes=SimpleNamespace(route_weights=lambda *a: routes), route_skip=skip), routes

    def test_no_skip_returns_the_served_routes_untouched(self):
        net, routes = self.net(None)
        ids, w = Glm53Net._select_routes(net, 3, None)
        self.assertIs(ids, routes[0])
        self.assertIs(w, routes[1])

    def test_a_layer_table_applies_its_own_threshold_and_skips_layers_it_does_not_name(self):
        net, routes = self.net({3: 0.05})
        ids, w = Glm53Net._select_routes(net, 3, None)
        self.assertIs(ids, routes[0], "ids stay: a skipped slot reads its expert at weight 0")
        self.assertEqual(int((w > 0).sum()), 5)
        net, routes = self.net({4: 0.05})
        self.assertIs(Glm53Net._select_routes(net, 3, None)[1], routes[1])

    def test_a_global_threshold_applies_to_every_layer(self):
        net, _ = self.net(0.05)
        self.assertEqual(int((Glm53Net._select_routes(net, 3, None)[1] > 0).sum()), 5)


class ScheduleContractTests(unittest.TestCase):
    def test_boot_serves_no_skip_by_default_and_hands_the_schedule_only_to_a_capture(self):
        tree = ast.parse((ROOT / "engine/profiles/glm53/boot.py").read_text())
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id in ("ROUTE_SKIP_PASSES", "ROUTE_SKIP_PASS_DOCUMENTS", "EXPERT_CAPTURE")}
        self.assertEqual(values, {"EXPERT_CAPTURE": False, "ROUTE_SKIP_PASSES": (), "ROUTE_SKIP_PASS_DOCUMENTS": 0})
        attach = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "attach"
                  and any(k.arg == "skip_passes" for k in c.keywords)]
        self.assertEqual(len(attach), 1)

    def test_the_capture_sets_the_skip_around_the_prefill_and_always_clears_it(self):
        source = (ROOT / "engine/profiles/glm53/capture.py").read_text()
        body = source.split("def prefill_forward(step):")[1].split("def route_hook")[0]
        self.assertIn("net.route_skip = self.skip_passes[", body)
        finally_part = body.split("finally:")[1]
        self.assertIn("net.route_skip = None", finally_part)
        self.assertIn("self.doc % self.pass_documents", source, "a pass scores the same positions as the first")


if __name__ == "__main__":
    unittest.main()
