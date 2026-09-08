"""Exercise failure handling without Docker, CUDA or fleet access."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / 'probes/run_ar_consumer_gpu.py'
spec = importlib.util.spec_from_file_location('ar_consumer_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ContainerEvidence(unittest.TestCase):
    def exercise(self, launch_error=None, inspect_error=None, cleanup_error=None):
        calls = []
        error = None
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / 'container.json'

            def remote(node, argv, **kwargs):
                calls.append((node, argv))
                if argv[1] == 'run':
                    if launch_error:
                        raise launch_error
                elif argv[1] == 'inspect':
                    if inspect_error:
                        raise inspect_error
                    return subprocess.CompletedProcess(argv, 0, json.dumps({
                        'state': {'OOMKilled': bool(launch_error), 'ExitCode': 9 if launch_error else 0}}))
                elif argv[1] == 'rm':
                    if cleanup_error:
                        raise cleanup_error
                else:
                    self.fail(argv)

            with patch.object(runner, 'remote', side_effect=remote):
                try:
                    runner.run_container('10.10.10.1', ['docker', 'run'],
                                         'owned-probe', evidence, None)
                except BaseException as exc:
                    error = exc
            self.assertEqual([argv[1] for _, argv in calls], ['run', 'inspect', 'rm'])
            self.assertTrue(all(node == '10.10.10.1' for node, _ in calls))
            self.assertEqual(calls[-1][1], ['docker', 'rm', '-f', 'owned-probe'])
            return error, json.loads(evidence.read_text())

    def test_oom_is_not_accepted_as_a_successful_probe(self):
        failure = subprocess.CalledProcessError(9, ['docker', 'run'])
        error, evidence = self.exercise(launch_error=failure)
        self.assertIs(error, failure)
        self.assertTrue(evidence['state']['OOMKilled'])

    def test_missing_container_does_not_hide_launch_error(self):
        failure = subprocess.CalledProcessError(125, ['docker', 'run'])
        error, evidence = self.exercise(launch_error=failure,
            inspect_error=subprocess.CalledProcessError(1, ['docker', 'inspect']),
            cleanup_error=subprocess.CalledProcessError(1, ['docker', 'rm']))
        self.assertIs(error, failure)
        self.assertIn('inspection_error', evidence)

    def test_cleanup_failure_blocks_an_otherwise_successful_probe(self):
        error, evidence = self.exercise(
            cleanup_error=subprocess.TimeoutExpired(['docker', 'rm'], 15))
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(evidence['state']['OOMKilled'])

    def test_success_retains_exit_state_and_removes_owned_container(self):
        error, evidence = self.exercise()
        self.assertIsNone(error)
        self.assertEqual(evidence['state']['ExitCode'], 0)


if __name__ == '__main__':
    unittest.main()
