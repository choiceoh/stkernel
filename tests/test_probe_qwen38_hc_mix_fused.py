"""The Qwen3.8 mixer-fold probe (engine/QWEN38_CARRY.md H1/H2) measures the served site: its widths are the checkpoint's
(probes/qwen38_config.json), its rows are a captured step's, and what it holds the prototype to is the lane itself.
CPU only: the probe's measurement needs a CUDA device and says so."""
import ast
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "probes/engine_qwen38_hc_mix_fused.py"
TRITON = importlib.util.find_spec("triton") is not None and importlib.util.find_spec("torch") is not None


def constants():
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], (ast.Name, ast.Tuple)):
            names = [node.targets[0]] if isinstance(node.targets[0], ast.Name) else node.targets[0].elts
            if all(isinstance(n, ast.Name) and n.id.isupper() for n in names):
                values = ast.literal_eval(node.value)
                found.update(zip([n.id for n in names], values if isinstance(node.targets[0], ast.Tuple) else [values]))
    return found


class ProbeShapeTests(unittest.TestCase):
    def test_the_widths_are_the_checkpoints(self):
        text = json.loads((ROOT / "probes/qwen38_config.json").read_text(encoding="utf-8"))
        text = text.get("text_config", text)
        c = constants()
        self.assertEqual((c["HC"], c["HIDDEN"], c["RANK"]), (text["hc_count"], text["hidden_size"], text["hc_lowrank"]))

    def test_the_rows_are_a_captured_steps(self):
        from engine.profiles.qwen38 import fleet
        rows = constants()["ROWS"]
        self.assertEqual(max(rows), fleet.MAX_SEQS * 2)             # SPEC_K = 1: two tokens a row
        self.assertTrue({2, 4, 6, 8} <= set(rows))

    def test_it_is_held_to_the_lane_and_reports_through_the_contract(self):
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn("hcr.mix(normed, down, up, HC, inject=False)", source)
        self.assertIn("torch.equal(got, want)", source)
        self.assertIn("write_report(metrics, proof, len(ROWS), torch.cuda.get_device_name())", source)
        # the number a GB10 run is for: what a launch costs there is what any fold could recover
        self.assertIn('metrics[f"headroom_us_rows{rows}"] = round(lane - gemms, 2)', source)

    def test_it_imports_only_what_the_lane_ships(self):
        """probes/run_engine_probe.sh rsyncs engine/ and probes/ to the lane's box: a probe that needs bench/ at import
        dies there in seconds (the first headroom ticket did, 2026-09-19). The report's writer is under probes/ for
        that reason, so it is imported plainly: behind a guard the report was dropped on the one lane it is for.
        tests/test_engine_lane_promises.py holds the same rule for every check the lane admits."""
        tree = ast.parse(PROBE.read_text(encoding="utf-8"))
        needed = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertTrue(all(m.split(".")[0] in ("__future__", "engine", "probes", "pathlib") for m in needed), needed)
        self.assertIn("probes.probe_report", needed)
        self.assertEqual([n for n in tree.body if isinstance(n, ast.Try)], [], "no guard: what it imports is shipped")

    @unittest.skipUnless(TRITON, "the probe imports Triton")
    def test_without_a_device_it_refuses_rather_than_report(self):
        import torch
        if torch.cuda.is_available():
            self.skipTest("a CUDA device is present")
        spec = importlib.util.spec_from_file_location("hc_mix_fused_probe", PROBE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.main([]), 2)


if __name__ == "__main__":
    unittest.main()
