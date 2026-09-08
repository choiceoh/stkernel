"""Classifier explanations preserve the existing shell policy and source lines."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import fleet_classify as classify


class ClassifierExplanationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = (ROOT / 'bench/fleet.sh').read_text()
        self.shell_function = source[source.index('classify_cmd() {'):source.index('production_line() {')]
        self.env = dict(os.environ, REPO=str(ROOT))
        self.env.pop('FLEET_REHEARSE', None)

    def explain(self, command, **environment):
        env = dict(self.env, **environment)
        # Only the extracted classification function runs; commands supplied to
        # it are inspected as strings and never executed.
        actual = subprocess.check_output(['bash', '-c', self.shell_function + '\nclassify_cmd "$@"',
                                          'classification-fixture', *command], env=env, text=True).strip()
        answer = classify.explain(actual, command, env)
        self.assertEqual(answer['classification'], actual)
        return answer

    def test_gpu_priority_points_to_exact_file_line_and_filters_comment_lines(self):
        script = self.root / 'worker.sh'
        script.write_text('# nvidia-smi is just a comment\nexport MK_PROBE_NO_GPU=1\ndocker run --gpus all image\n')
        value = self.explain(['bash', str(script)])
        self.assertEqual(value['classification'], 'gpu')
        self.assertEqual(value['evidence'][0]['path'], str(script.resolve()))
        self.assertEqual(value['evidence'][0]['line'], 3)
        self.assertEqual(value['evidence'][0]['match'], 'docker run')
        self.assertEqual(value['evidence'][0]['snippet'], 'docker run --gpus all image')
        self.assertTrue(all(reason.get('line') != 1 for reason in value['evidence']))

    def test_argv_match_spanning_arguments_and_cpu_hint_are_explained(self):
        value = self.explain(['docker', 'run', 'image'])
        self.assertEqual(value['classification'], 'gpu')
        self.assertEqual(value['evidence'][0]['argv_indices'], [0, 1])
        self.assertEqual(value['evidence'][0]['match'], 'docker run')
        value = self.explain(['bash', '-n', '/missing/input.sh'])
        self.assertEqual(value['classification'], 'nogpu')
        self.assertEqual(value['evidence'][0]['argv_indices'], [0, 1])

    def test_python_extraction_uses_only_first_three_tokens_then_final_gpu_regex(self):
        script = self.root / 'candidate.py'
        script.write_text('# torch.cuda ignored\nx.cuda()\ny.cuda()\nz.cuda()\ntorch.cuda.synchronize()\n')
        value = self.explain(['python3', str(script)])
        self.assertEqual(value['classification'], 'unknown')
        self.assertEqual(value['evidence'], [])
        script.write_text('# torch.cuda ignored\nx.cuda()\ntorch.cuda.synchronize()\n')
        value = self.explain(['python3', str(script)])
        self.assertEqual(value['classification'], 'gpu')
        self.assertEqual(value['evidence'][0]['line'], 3)
        script.write_text('"""torch.cuda in a docstring is not a comment-only line"""\n')
        value = self.explain(['python3', str(script)])
        self.assertEqual(value['classification'], 'gpu')
        self.assertEqual(value['evidence'][0]['line'], 1)

    def test_rehearsal_and_reviewed_compile_entrypoint_explain_early_exceptions(self):
        value = self.explain(['nvidia-smi'], FLEET_REHEARSE='1')
        self.assertEqual(value['classification'], 'nogpu')
        self.assertEqual(value['evidence'][0]['name'], 'FLEET_REHEARSE')
        value = self.explain(['python3', 'bench/cpu_compile.py', '/tmp/kernel.cu'])
        self.assertEqual(value['classification'], 'nogpu')
        self.assertIn('reviewed', value['reason'])
        value = self.explain(['python3', '/unreviewed/cpu_compile.py', '/tmp/kernel.cu'])
        self.assertEqual(value['classification'], 'gpu')
        self.assertEqual(value['evidence'][0]['match'], '.cu')

    def test_bounded_file_reads_do_not_override_authoritative_classification(self):
        script = self.root / 'large.sh'
        script.write_text('#' + 'x' * classify.MAX_FILE_BYTES + '\nnvidia-smi\n')
        value = self.explain(['bash', str(script)])
        self.assertEqual(value['classification'], 'gpu')
        self.assertEqual(value['evidence'], [])
        self.assertEqual(value['truncated_files'], [str(script)])
        self.assertIn('remains authoritative', value['reason'])

    def test_reasons_and_snippets_are_bounded_and_cli_is_json(self):
        script = self.root / 'many.sh'
        script.write_text('nvidia-smi ' + 'x' * 500 + '\n' + 'nvidia-smi\n' * 20)
        value = self.explain(['bash', str(script)])
        self.assertEqual(len(value['evidence']), classify.MAX_REASONS)
        self.assertTrue(all(len(item['snippet']) <= 240 for item in value['evidence']))
        result = subprocess.check_output([sys.executable, str(ROOT / 'bench/fleet_classify.py'),
                                          '--classification', 'gpu', '--', 'docker', 'run', 'image'],
                                         env=self.env, text=True)
        self.assertEqual(json.loads(result)['evidence'][0]['argv_indices'], [0, 1])


if __name__ == '__main__':
    unittest.main()
