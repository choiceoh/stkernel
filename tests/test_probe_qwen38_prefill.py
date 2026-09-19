"""The prefill census's arithmetic, its lane in the kernel check, and that it imports only what the lane ships.

The run needs a GB10 and a rank file (probes/engine_qwen38_prefill.py). Held here: four layer sets solve the four
unknowns exactly and add up to the 48-layer chunk, a profile's kernels fall into the step probe's families, and the
host's share is the wall less the device.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probes import engine_qwen38_prefill as probe  # noqa: E402

# what a layer set holds (engine_qwen38_step.counts at the served layer types): the probe's four sets
COUNTS = {"4,5,6,7": {"fixed": 1, "gdn": 3, "qsa": 1, "ple": 0}, "4,5": {"fixed": 1, "gdn": 2, "qsa": 0, "ple": 0},
          "7": {"fixed": 1, "gdn": 0, "qsa": 1, "ple": 0}, "1": {"fixed": 1, "gdn": 1, "qsa": 0, "ple": 1}}
TRUE = {"wall_ms": {"fixed": 40.0, "gdn": 9.0, "qsa": 30.0, "ple": 2.0},
        "device_ms": {"fixed": 30.0, "gdn": 8.0, "qsa": 25.0, "ple": 1.0},
        "launches": {"fixed": 200, "gdn": 60, "qsa": 90, "ple": 12}}
FAMILIES = {"moe b12x": {"fixed": 5.0, "gdn": 4.0, "qsa": 4.0, "ple": 0.0},
            "qsa": {"fixed": 3.0, "gdn": 0.0, "qsa": 15.0, "ple": 0.0}}


def chunk_row(counts: dict, context: int) -> dict:
    total = lambda parts: sum(parts[u] * n for u, n in counts.items())  # noqa: E731
    row = {"context": context, "tokens": 4096, **{m: total(parts) for m, parts in TRUE.items()},
           "families": {f: {"launches": 1, "ms": total(parts)} for f, parts in FAMILIES.items()}}
    row["host_ms"] = row["wall_ms"] - row["device_ms"]
    return row


class AssembleTests(unittest.TestCase):
    def setUp(self):
        self.builds = {name: {"counts": c, "chunks": [chunk_row(c, 0), chunk_row(c, 4096)]} for name, c in COUNTS.items()}

    def test_four_sets_solve_the_four_unknowns_and_add_up_to_48_layers(self):
        out = probe.assemble(self.builds)
        self.assertEqual(sorted(out), ["context 0", "context 4096"])
        entry = out["context 0"]
        for metric, parts in TRUE.items():
            self.assertEqual(entry[metric]["parts"], {u: float(v) for u, v in parts.items()})
            self.assertAlmostEqual(entry[metric]["chunk_48"], parts["fixed"] + 36 * parts["gdn"] + 12 * parts["qsa"] + parts["ple"])
        self.assertAlmostEqual(entry["host_ms"]["chunk_48"], entry["wall_ms"]["chunk_48"] - entry["device_ms"]["chunk_48"])
        self.assertAlmostEqual(entry["tokens_a_second_one_rank"], round(4096 / entry["wall_ms"]["chunk_48"] * 1e3, 1))

    def test_families_come_largest_first_with_their_parts(self):
        fams = probe.assemble(self.builds)["context 0"]["families"]
        self.assertEqual(list(fams), ["moe b12x", "qsa"])                  # 5 + 36*4 + 12*4 = 197 over 3 + 12*15 = 183
        self.assertAlmostEqual(fams["moe b12x"]["chunk_48_ms"], 197.0)
        self.assertAlmostEqual(fams["qsa"]["chunk_48_ms"], 183.0)
        self.assertEqual(fams["qsa"]["parts_ms"], {"fixed": 3.0, "gdn": 0.0, "qsa": 15.0, "ple": 0.0})

    def test_a_missing_layer_set_leaves_the_chunk_unsolved(self):
        del self.builds["7"]
        self.assertEqual(probe.assemble(self.builds), {})


class FamiliesTests(unittest.TestCase):
    def test_a_profiles_kernels_fall_into_the_step_probes_families(self):
        events = [SimpleNamespace(key="_qsa_mqa_paged_group_kernel", count=12, self_device_time_total=3000.0),
                  SimpleNamespace(key="chunk_gla_fwd_kernel_o", count=3, self_device_time_total=500.0),
                  SimpleNamespace(key="chunk_gla_fwd_kernel_o", count=3, self_device_time_total=250.0),
                  SimpleNamespace(key="cpu only", count=4, self_device_time_total=0.0)]
        fams, kernels = probe.families_of(SimpleNamespace(key_averages=lambda: events))
        self.assertEqual(fams, {"qsa": [12, 3000.0], "gdn / kda": [6, 750.0]})
        self.assertEqual(kernels["chunk_gla_fwd_kernel_o"], [6, 750.0])
        self.assertNotIn("cpu only", kernels)

    def test_an_older_torchs_attribute_name_is_read_too(self):
        events = [SimpleNamespace(key="gemm_x", count=2, self_cuda_time_total=10.0)]
        fams, _ = probe.families_of(SimpleNamespace(key_averages=lambda: events))
        self.assertEqual(fams, {"gemm (cublas/cutlass)": [2, 10.0]})


class LaneTests(unittest.TestCase):
    def test_the_kernel_check_runs_it_with_the_chunk_after_the_colon(self):
        source = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("args.lanes == 'qwen38_prefill' or args.lanes.startswith('qwen38_prefill:')", source)
        self.assertIn("qwen38_prefill(args.output, args.ranks, chunk=int((args.lanes.split(':')[1:] or [CHUNK])[0]))", source)
        self.assertIn("--lanes qwen38_prefill --ranks", probe.__doc__)

    def test_it_measures_the_served_prefill_and_never_samples(self):
        source = (ROOT / "probes/engine_qwen38_prefill.py").read_text(encoding="utf-8")
        self.assertIn('model.prefill(seq, row["context"], chunk, None, slot)', source)
        self.assertIn("total = chunk * chunks + F.block", source)          # the prompt outlasts what is prefilled
        self.assertEqual(probe.LAYER_SETS, ((4, 5, 6, 7), (4, 5), (7,), (1,)))

    def test_the_memory_ceiling_is_the_tickets_budget(self):
        self.assertEqual(probe.ceiling_gib({"ST_PROBE_GIB": "8"}), 7.0)      # less the context the allocator does not see
        self.assertEqual(probe.ceiling_gib({"ST_PROBE_GIB": "1.5"}), 1.0)
        self.assertEqual(probe.ceiling_gib({}), probe.MAX_GIB)
        self.assertEqual(probe.ceiling_gib({"ST_PROBE_GIB": ""}), probe.MAX_GIB)

    def test_it_imports_only_what_the_lane_ships(self):
        tree = ast.parse((ROOT / "probes/engine_qwen38_prefill.py").read_text(encoding="utf-8"))
        standard = ("__future__", "argparse", "json", "os", "pathlib", "subprocess", "sys", "tempfile", "time", "torch")
        for node in ast.walk(tree):
            names = ([node.module] if isinstance(node, ast.ImportFrom) else
                     [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                self.assertIn(name.split(".")[0], standard + ("engine", "probes", "tests"), name)


if __name__ == "__main__":
    unittest.main()
