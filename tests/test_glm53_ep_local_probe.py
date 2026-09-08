"""CPU tests of evidence rejection and the actual numerical comparison code."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"probes"))
import glm53_ep_local_check as probe
import glm53_ep_local_evidence as binding
import run_glm53_ep_local_offline as runner


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch; pinned CPU image covers this")
class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_small_stock_noise_preserves_existing_candidate_floor(self):
        b = self.torch.ones(3, 8)
        self.assertEqual(probe.compare(b+.006, b, b+.001)["bad_rows"], 0)

    def test_large_stock_noise_cannot_authorize_a_bad_candidate(self):
        b = self.torch.ones(3, 8)
        # Old relative-to-noise limits accepted this 20% candidate error.
        with self.assertRaisesRegex(AssertionError, "UNSTABLE_STOCK_CONTROL"):
            probe.compare(b+.2, b, b+.1)

    def test_corrupt_small_row_cannot_hide_behind_a_large_row(self):
        b = self.torch.tensor([[1e6]*8, [1.]*8])
        a = b.clone()
        a[1, 0] = 1.25
        with self.assertRaisesRegex(AssertionError, "CANDIDATE_NUMERICS_FAIL"):
            probe.compare(a, b, b.clone())

    def test_nonfinite_and_broadcastable_shapes_are_rejected(self):
        b = self.torch.ones(2, 8)
        with self.assertRaisesRegex(AssertionError, "shape mismatch"):
            probe.compare(b[:1], b, b)
        a = b.clone()
        a[0, 0] = float("nan")
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            probe.compare(a, b, b)


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for name in ("moe_dispatch.py", "moe_dynamic_ep_local.py", "glm53_ep_route_remap.py", "b12x_moe.py",
                     "flashinfer_b12x_moe.py"):
            target = "/installed/flashinfer/"+name
            source = root/"build/glm53"/name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("content of "+name)
            rows.append(name+"\t"+target+"\tunused")
        (root/"build/glm53/manifest.tsv").write_text("\n".join(rows)+"\n")
        for name in binding.CONTRACT_PATHS:
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("contract "+name)
        sources = {target: binding.digest(path) for target, path in binding.mounted_sources(root).items()}
        evidence = dict(arm="local", cuda_initialized=False,
                        cache_key=["glm53_ep_prefill_local_v1"], artifacts=["ptx"], resources=["cubin"],
                        sources=sources, mounted_sources=sources.copy(),
                        remap_compilation=[dict(case, ptx_sha256="ptx", cubin_sha256="cubin")
                                           for case in binding.compile_cases()],
                        contracts=dict(tests_run=1, failures=0, errors=0, skips=0,
                                       files={name: binding.digest(root/name) for name in binding.CONTRACT_PATHS}))
        path = root/"compile.json"
        path.write_text(json.dumps(evidence))
        return path, evidence

    def test_matching_evidence_passes_and_changed_overlay_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.fixture(root)
            binding.validate_compile_evidence(root, path)
            (root/"build/glm53/moe_dynamic_ep_local.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "compiled overlay source changed"):
                binding.validate_compile_evidence(root, path)

    def test_missing_mount_skips_and_changed_test_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, evidence = self.fixture(root)
            for key, value in (("skips", 1), ("tests_run", 0), ("errors", 1)):
                altered = dict(evidence, contracts=dict(evidence["contracts"], **{key:value}))
                path.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    binding.validate_compile_evidence(root, path)
            path.write_text(json.dumps(dict(evidence, mounted_sources={})))
            with self.assertRaisesRegex(ValueError, "file sets differ"):
                binding.validate_compile_evidence(root, path)
            path.write_text(json.dumps(evidence))
            (root/binding.CONTRACT_PATHS[0]).write_text("different test")
            with self.assertRaisesRegex(ValueError, "contract source changed"):
                binding.validate_compile_evidence(root, path)

    def test_incomplete_remap_compile_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, evidence = self.fixture(root)
            evidence["remap_compilation"].pop()
            path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "remap compilation proof"):
                binding.validate_compile_evidence(root, path)

    def test_stale_receipt_is_rejected_before_service_inventory_or_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["runner", "--revision", "a"*40, "--out", str(Path(directory)/"capture")]
            with (patch.object(sys, "argv", args),
                  patch.object(runner.signal, "signal"),
                  patch.object(runner.lifecycle, "check_holder"),
                  patch.object(runner.lifecycle, "pinned"),
                  patch.object(runner.lifecycle, "snapshot") as snapshot,
                  patch.object(runner.lifecycle, "with_paused") as pause,
                  patch.object(runner, "validate_compile_evidence", side_effect=ValueError("stale receipt")),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 1)
            snapshot.assert_not_called()
            pause.assert_not_called()

    def test_failed_probe_preserves_phase_and_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            args = ["probe", "--case", "balanced4096", "--compile-evidence", "unused", "--output", str(path)]
            def fail(args, result):
                result.update(phase="changed-control", controls=[{"bad_rows": 1}])
                raise ValueError("unstable control")
            with (patch.object(sys, "argv", args), patch.object(probe, "run_case", fail),
                  patch.dict(probe.os.environ), redirect_stdout(io.StringIO()),
                  self.assertRaisesRegex(ValueError, "unstable control")):
                probe.main()
            got = json.loads(path.read_text())
            self.assertEqual((got["verdict"], got["phase"]), ("FAIL", "changed-control"))
            self.assertEqual(got["controls"], [{"bad_rows":1}])
            self.assertFalse(got["performance_acceptance"])

    def test_stopped_handoff_reaches_failed_cell_and_restores_original_running_flags(self):
        before = {node:dict(id=node, running=False, auto_remove=False, image=runner.lifecycle.IMAGE,
                           overlays={'source':'sha'}, manifest='sha', port=8000)
                  for node in runner.lifecycle.NODES}
        actions = []
        def transition(states, action):
            self.assertEqual(states, before)
            actions.append(action)
            self.assertEqual(action, 'stop')
            return before
        def process(command, **kwargs):
            if command[:2] == ['docker', 'run']:
                return subprocess.CompletedProcess(command, 42)
            self.assertEqual(command[:2], ['docker', 'inspect'])
            return subprocess.CompletedProcess(command, 1, '', '')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'repo'
            (root/'build/glm53').mkdir(parents=True)
            (root/'build/glm53/manifest.tsv').write_text('')
            output = Path(directory)/'capture'
            with (patch.object(sys, 'argv', ['runner', '--revision', 'a'*40, '--out', str(output)]),
                  patch.object(runner, '__file__', str(root/'probes/run_glm53_ep_local_offline.py')),
                  patch.dict(runner.os.environ, FLEET_SESSION='unit-stopped'),
                  patch.object(runner.signal, 'signal'), patch.object(runner.lifecycle, 'check_holder'),
                  patch.object(runner.lifecycle, 'pinned'), patch.object(runner, 'validate_compile_evidence'),
                  patch.object(runner.sanitizer_support, 'preflight', return_value={'verdict':'PASS'}),
                  patch.object(runner, 'resources', return_value={}),
                  patch.object(runner.lifecycle, 'snapshot', return_value=before),
                  patch.object(runner.lifecycle, 'transition_all', side_effect=transition),
                  patch.object(runner.lifecycle, 'healthy') as health,
                  patch.object(runner.lifecycle, 'idle') as idle,
                  patch.object(runner.lifecycle, 'restore_public') as public_restore,
                  patch.object(runner.subprocess, 'run', side_effect=process),
                  patch.object(runner.subprocess, 'check_output', return_value=''),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 1)
            result = json.loads((output/'completion.json').read_text())
            self.assertEqual(result['incoming_mode'], 'stopped')
            self.assertEqual(result['cells'][0]['exit_code'], 42)
            self.assertEqual(actions, ['stop'])
            self.assertTrue(result['restored_original'])
            self.assertEqual(json.loads((output/'restored.json').read_text()), before)
            health.assert_not_called(); idle.assert_not_called(); public_restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
