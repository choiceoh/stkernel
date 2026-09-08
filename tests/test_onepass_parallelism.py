"""Configured launch metadata remains separate from EP execution evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import onepass
import proof


class OnepassParallelismTests(unittest.TestCase):
    def collect(self, launch_command, *, inspect_fails=False):
        calls = []
        container_id = "a" * 64

        def run(argv, **kwargs):
            calls.append(argv)
            if argv[1] == "ps":
                text, rc = "glm53\n", 0
            elif argv[3] == "{{.Id}}|{{.State.StartedAt}}":
                text, rc = container_id + "|2026-09-08T08:00:00Z\n", 0
            elif argv[3] == "{{json .Config.Env}}":
                self.assertEqual(argv[-1], container_id)
                text, rc = json.dumps(["VLLM_GLM53_EP_PREFILL_LOCAL=1"]), 0
            elif argv[3] == "{{json .Config.Cmd}}":
                self.assertEqual(argv[-1], container_id)
                text, rc = json.dumps(launch_command), 1 if inspect_fails else 0
            else:
                raise AssertionError(argv)
            return subprocess.CompletedProcess(argv, rc, stdout=text, stderr="")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profiles").mkdir()
            (root / "profiles/glm53.env").write_text("VLLM_GLM53_EP_PREFILL_LOCAL=0\n")
            (root / "stamp").write_text("b" * 64)
            with patch.dict(os.environ, {"MK_OVERLAY_STAMP": str(root / "stamp")}), \
                    patch.object(subprocess, "run", side_effect=run):
                result = onepass._served_build(str(root))
        return result, calls

    def test_metadata_is_attached_to_observed_container_without_runtime_env(self):
        # The collector consumes the parser's facts, not the caller's ENABLE_EP.
        parsed = dict(schema=1, source="container-launch-script", enabled=True,
                      tensor_parallel_size=4, nnodes=4, node_rank=0)
        with patch("glm53_launch_metadata.launch_parallelism", return_value=parsed), \
                patch.dict(os.environ, {"ENABLE_EP": "0"}):
            result, _ = self.collect(["-c", "configured command"])
        self.assertEqual(result["parallelism"], dict(parsed, container_id="a" * 64, issues=[]))
        self.assertEqual(result["knobs"], {"VLLM_GLM53_EP_PREFILL_LOCAL": "1"})
        self.assertNotIn("proof", result)

    def test_bad_or_missing_launch_metadata_is_unknown_and_retains_existing_fields(self):
        for inspect_fails in (False, True):
            with self.subTest(inspect_fails=inspect_fails):
                result, _ = self.collect(["-c", "unrecognized secret command"], inspect_fails=inspect_fails)
                self.assertIsNone(result["parallelism"]["enabled"])
                self.assertEqual(result["parallelism"]["issues"], ["launch metadata unavailable"])
                self.assertEqual(result["overlay"], "b" * 12)
                self.assertEqual(result["knobs"], {"VLLM_GLM53_EP_PREFILL_LOCAL": "1"})
                self.assertNotIn("secret", json.dumps(result))

    def test_ep_marker_requires_the_actual_full_token_launch_line(self):
        knob = "VLLM_GLM53_EP_PREFILL_LOCAL"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "head.log"
            log.write_text("[prefill-sp] sequence-parallel prefill armed\n")
            self.assertFalse(proof.check([knob], str(log))["proof"][knob])
            log.write_text("[ep-prefill-local] LAUNCHED full-token E72/I2048/top8 T=8192\n")
            self.assertTrue(proof.check([knob], str(log))["proof"][knob])


if __name__ == "__main__":
    unittest.main()
