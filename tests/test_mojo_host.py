"""CPU-only differential tests; set ST_MOJO_HOST_DIRECTORY to REQUIRE the real extension."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

from bench import mojo_host as host


class ArtifactTests(unittest.TestCase):
    def test_rejects_wrong_runtime_source_and_binary_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            binary = directory / f"{host.MODULE}.so"
            binary.write_bytes(b"not loaded in this test")
            manifest = dict(schema=1, runtime=host.runtime_identity(), compiler="Mojo 1.0.0 (test)",
                            source_sha256=host.sha256(host.SOURCE), library_sha256=host.sha256(binary))
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(host.validate_artifact(directory), manifest)
            for key in ("runtime", "source_sha256", "library_sha256", "compiler", "schema"):
                with self.subTest(key=key):
                    path.write_text(json.dumps(dict(manifest, **{key: "stale"})))
                    with self.assertRaisesRegex(RuntimeError, "mismatch"):
                        host.validate_artifact(directory)

    def test_failed_build_does_not_replace_existing_artifact(self):
        from unittest.mock import patch
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / f"{host.MODULE}.so"
            binary.write_bytes(b"previous complete build")
            with patch.object(host.shutil, "which", return_value="/missing/mojo"), \
                 patch.object(host.subprocess, "run", side_effect=[
                     subprocess.CompletedProcess([], 0, stdout="Mojo 1.0.0 (test)"),
                     subprocess.CalledProcessError(1, "mojo")]):
                with self.assertRaises(subprocess.CalledProcessError):
                    host.build(directory)
            self.assertEqual(binary.read_bytes(), b"previous complete build")
            self.assertFalse((Path(directory) / "manifest.json").exists())

    def test_cpu_oracle_runs_without_engine_imports_and_detects_drift(self):
        oracle = host.load_oracle()
        p, pending = host.fixture(4, context=60)
        result = host.trace(4, 60, 1)[0][0]
        oracle(p, pending, result)
        self.assertEqual(p.e.ctx, {seq: 60 + n for seq, n in zip(pending.seqs, result["count"])})
        self.assertEqual(p.e.accepted_total, sum(result["accepted"]))
        self.assertEqual(pending.outcomes, [result])
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.py"
            changed.write_text(host.ORACLE.read_text().replace("def _apply_outcome(", "def _renamed_outcome("))
            with self.assertRaises(ValueError):
                host.load_oracle(changed)

    def test_benchmark_checks_every_sample_and_records_both_orders(self):
        functions = dict(python=host.load_oracle(), mojo=host.load_oracle())
        case = host.measure_case(functions, rows=4, context=60, steps=9, samples=2, warmup=1)
        self.assertTrue(case["all_samples_state_equal"])
        self.assertEqual([p["order"] for p in case["samples"]], [("python", "mojo"), ("mojo", "python")])
        self.assertEqual(len(case["host_state_sha256"]), 64)
        functions["mojo"] = lambda *args: None
        with self.assertRaisesRegex(AssertionError, "host state mismatch"):
            host.measure_case(functions, rows=1, context=0, steps=1, samples=2, warmup=1)


@unittest.skipUnless(os.environ.get("ST_MOJO_HOST_DIRECTORY"),
                     "optional Mojo extension; dedicated CPU CI builds and requires it")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An explicitly requested missing/stale/unloadable artifact is a FAILURE, not a skip.
        module, _ = host.load_native(os.environ["ST_MOJO_HOST_DIRECTORY"])
        cls.native = staticmethod(host.native_wrapper(module))
        cls.oracle = staticmethod(host.load_oracle())

    def compare(self, p, pending, result, error=None):
        candidates = [copy.deepcopy((p, pending)) for _ in range(2)]
        before = copy.deepcopy(host.snapshot(p, pending))
        for fn, (state, batch) in zip((self.oracle, self.native), candidates):
            if error:
                with self.assertRaisesRegex(RuntimeError, error):
                    fn(state, batch, copy.deepcopy(result))
                self.assertEqual(host.snapshot(state, batch), before)
            else:
                fn(state, batch, copy.deepcopy(result))
        self.assertEqual(host.snapshot(*candidates[0]), host.snapshot(*candidates[1]))
        return candidates[0]

    def test_randomized_row_counts_k_boundaries_and_histogram(self):
        rng = random.Random(103)
        for rows in (1, 2, 3, 4):
            for k in (1, 3, 5, 7):
                for context in (0, 60, 32760, 131064):
                    with self.subTest(rows=rows, k=k, context=context):
                        # Keep a short host history here; the benchmark times full histories.
                        p, pending = host.fixture(rows, k=k)
                        p.e.ctx = dict.fromkeys(pending.seqs, context)
                        for step in range(12):
                            counts = [rng.randrange(1, k + 2) for _ in range(rows)]
                            result = dict(count=counts, done=[False] * rows,
                                          before=[p.e.ctx[s] for s in pending.seqs],
                                          accepted=[rng.randrange(n) for n in counts],
                                          tokens=[[rng.randrange(10000) for _ in range(k + 1)] for _ in range(rows)])
                            p, pending = self.compare(p, pending, result)
                            pending.outcomes.clear()

    def test_released_rows_zero_count_done_and_already_finished(self):
        for released in ((), (1,), (0, 1, 2, 3)):
            p, pending = host.fixture(4, context=63)
            pending.finished[0] = True
            for row in released:
                del p.e.tokens[pending.seqs[row]]
                del p.e.ctx[pending.seqs[row]]
            result = dict(count=[0, 1, 2, 8], done=[True, False, True, False],
                          before=[63] * 4, accepted=[0, 0, 1, 7], tokens=[list(range(8)) for _ in range(4)])
            self.compare(p, pending, result)
        # All active rows may stop with zero tokens; this is progress via done.
        p, pending = host.fixture(1)
        self.compare(p, pending, dict(count=[0], done=[True], before=[0], accepted=[0], tokens=[[]]))

    def test_no_progress_count_and_stale_context_fail_before_mutation(self):
        p, pending = host.fixture(4, context=63)
        valid = dict(count=[1, 2, 3, 4], done=[False] * 4, before=[63] * 4,
                     accepted=[0, 1, 2, 3], tokens=[list(range(8)) for _ in range(4)])
        for field, value in (("count", -1), ("count", 9), ("before", 62)):
            result = copy.deepcopy(valid)
            result[field][-1] = value
            self.compare(p, pending, result, error="lost row/context order")
        # Preserve no-progress error priority even if the context is also stale.
        result = dict(valid, count=[0] * 4, before=[62] * 4)
        self.compare(p, pending, result, error="made no progress")

    def test_agreement_precedes_every_write_and_keeps_iteration(self):
        for fn in (self.oracle, self.native):
            p, pending = host.fixture(1)
            result = host.trace(1, 0, 1)[0][0]
            pending.outcomes.append({"previous": True})
            before = copy.deepcopy(host.snapshot(p, pending))
            calls = []
            def reject(seqs, value, *, iteration):
                calls.append((seqs, value, iteration))
                raise RuntimeError("rank outcome mismatch")
            p._agree_outcome = reject
            with self.assertRaisesRegex(RuntimeError, "rank outcome mismatch"):
                fn(p, pending, result)
            self.assertEqual(host.snapshot(p, pending), before)
            self.assertEqual(calls, [(pending.seqs, result, 1)])

    def test_duplicate_and_opaque_sequence_ids_follow_python_updates(self):
        p, pending = host.fixture(1, context=63)
        p.e.tokens = {"request": [0]}
        p.e.ctx = {"request": 63}
        pending.seqs, pending.finished = ("request", "request"), [False, False]
        self.compare(p, pending, dict(count=[1, 2], done=[False, False], before=[63, 63],
                                      accepted=[0, 1], tokens=[[1], [2, 3]]))

    def test_native_array_bounds_are_checked(self):
        for rows in (0, 5):
            p, pending = host.fixture(rows)
            result = dict(count=[1] * rows, done=[False] * rows, before=[0] * rows,
                          accepted=[0] * rows, tokens=[[1]] * rows)
            before = copy.deepcopy(host.snapshot(p, pending))
            with self.assertRaisesRegex(RuntimeError, "requires 1..4 rows"):
                self.native(p, pending, result)
            self.assertEqual(host.snapshot(p, pending), before)


if __name__ == "__main__":
    unittest.main()
