"""What the single-GPU lane promises, held on a CPU: a check it admits can start there, a command a probe documents is
one the queue takes, and what a check measured comes back to the controller as a report bound to its ticket.

Each of these was found the expensive way first. #1192 was a pull request because a ticket had been refused on srv4 for
a probe whose docstring named the lane; engine_qwen38_hc_mix_fused.py named the lane too, and would have died there on
`from bench.probe_report import ...`, because the lane's host is sent engine/, probes/ and tests/ and nothing else.
"""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench'))
import feedback
import fleet_onepass
import fleet_single
from probes import probe_report

# Documented, and refused: each is a decision about what may run beside production, which is the operator's.
# This list only shrinks -- an entry the queue now takes fails the test until it is removed.
KNOWN_REFUSED = {
    'probes/b12x_lane_semantics.py': 'loads a rank file\'s expert weights: its budget beside production is not a kernel check\'s',
    'probes/engine_draw_contract.py': 'a few rows of integers, but its documented command passes --k, which ST_FLAGS does not carry',
    'probes/engine_grammar_gpu.py': 'loads the checkpoint\'s tokenizer through --ckpt, a flag and a path the lane does not carry',
    'probes/engine_prefix_check.py': 'boots layers of the real checkpoint with a KV arena: a full-model budget, not the default',
    'probes/glm53_arena_bytes_check.py': 'reads a whole rank file, and --rank is not a lane flag',
}


def shipped():
    """The trees the runner sends the lane's host, read off its rsync line rather than repeated here."""
    runner = (ROOT / 'probes/run_engine_probe.sh').read_text(encoding='utf-8')
    line = next(l for l in runner.splitlines() if l.lstrip().startswith('rsync ') and '$probe_host:' in l)
    return tuple(name + '/' for name in re.findall(r'"\$repo/([A-Za-z_]+)"', line))


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = feedback.Graph(ROOT)

    def test_the_runner_sends_three_trees(self):
        self.assertEqual(shipped(), ('engine/', 'probes/', 'tests/'))

    def test_an_admitted_check_can_start_on_what_the_lanes_host_is_sent(self):
        trees = shipped()
        for probe in fleet_onepass.ST_PROBES:
            outside = sorted(self.graph.rel(p) for p in self.graph.eager(ROOT / probe) if not self.graph.rel(p).startswith(trees))
            self.assertEqual(outside, [], f'{probe} imports these at module level, and the lane\'s host never gets them')

    def test_a_lazy_import_is_not_a_start_dependency_but_a_module_level_one_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for relative, text in {
                    'bench/fleet_onepass.py': "ST_PROBES = ()\nST_PROBE_BUDGET_GIB = {}\n",
                    'bench/helper.py': 'X = 1\n', 'bench/late.py': 'Y = 2\n',
                    'engine/__init__.py': '', 'engine/core.py': 'def f():\n    from bench import late\n',
                    'probes/good.py': 'from engine import core\n',
                    'probes/bad.py': 'try:\n    from bench.helper import X\nexcept ImportError:\n    X = 0\n'}.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            graph = feedback.Graph(root)
            self.assertEqual({graph.rel(p) for p in graph.eager(root / 'probes/good.py')}, {'engine/__init__.py', 'engine/core.py'})
            self.assertIn('bench/helper.py', {graph.rel(p) for p in graph.eager(root / 'probes/bad.py')})
            self.assertIn(root / 'bench/late.py', graph.reach(root / 'probes/good.py'), 'reach still follows it: the check is about it')

    def test_a_command_a_probe_documents_is_one_the_queue_takes(self):
        rows = feedback.lane_audit(self.graph)
        self.assertGreater(len(rows), 10, 'the docstrings stopped spelling out runner commands, or the parser stopped reading them')
        refused = {probe: why for probe, argv, why in rows if why}
        surprise = {probe: why for probe, why in refused.items() if probe not in KNOWN_REFUSED}
        self.assertEqual(surprise, {}, 'a docstring tells the reader to run this through the queue, and the queue refuses it: '
                                       'admit it in bench/fleet_onepass.py ST_PROBES (with its budget), or correct the docstring')
        stale = sorted(set(KNOWN_REFUSED) - set(refused))
        self.assertEqual(stale, [], 'the queue takes these now (or they no longer document a command): drop them from KNOWN_REFUSED')

    def test_the_mixer_fold_probe_is_admitted_with_a_kernel_checks_budget(self):
        probe = 'probes/engine_qwen38_hc_mix_fused.py'
        contract = fleet_onepass.validate(['bash', 'probes/run_engine_probe.sh', probe], ROOT, ROOT, {}, kind=fleet_onepass.SINGLE)
        self.assertEqual((contract['gpus'], contract['budget_gib']), (1, 8))
        self.assertIn('from probes.probe_report import write_report', (ROOT / probe).read_text(encoding='utf-8'))


class InstructionParserTests(unittest.TestCase):
    def commands(self, doc):
        return [argv[1:] for argv in feedback.instructed(doc)]

    def test_continuations_placeholders_and_optional_parts(self):
        doc = '''Run it:

            bash bench/fleet.sh run --gpu s 5 'note' -- \\
                bash probes/run_engine_probe.sh probes/engine_a.py --k 7

            bash probes/run_engine_probe.sh probes/engine_b.py [--layers 0-4] [--tail 300]
            bash probes/run_engine_probe.sh probes/engine_c.py \\
                --ranks <rank file> --output /cache/st-m64.json
            bash probes/run_engine_probe.sh probes/engine_d.py --seqs 1,2 --output /cache/<name>.jsonl  # then read it
            Inside the ST image: bash probes/run_engine_check.sh --layers 0-4.
        '''
        self.assertEqual(self.commands(doc), [
            ['probes/run_engine_probe.sh', 'probes/engine_a.py', '--k', '7'],
            ['probes/run_engine_probe.sh', 'probes/engine_b.py'],
            ['probes/run_engine_probe.sh', 'probes/engine_c.py', '--output', '/cache/st-m64.json'],
            ['probes/run_engine_probe.sh', 'probes/engine_d.py', '--seqs', '1,2'],
            ['probes/run_engine_check.sh', '--layers', '0-4']])

    def test_prose_that_only_mentions_the_runner_is_not_a_command_with_arguments(self):
        self.assertEqual(self.commands('Run through run_engine_probe.sh in the ST image.'), [])
        self.assertEqual(self.commands(None), [])


class ReportTests(unittest.TestCase):
    METRICS, PROOF = {'headroom_us_rows2': 17.5, 'rows_where_fused_wins': 0}, {'byte_equal_rows2': True, 'byte_equal_rows4': True}

    def test_nobody_asked_so_nothing_is_written(self):
        self.assertIsNone(probe_report.write_report(self.METRICS, self.PROOF, 4, 'NVIDIA GB10', environ={}))

    def test_a_tickets_report_comes_back_bound_to_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = fleet_single.report_name('exp/mixer fold#1')
            self.assertEqual(name, 'probe-report-exp_mixer_fold_1.json')
            self.assertRegex('.cache/st/' + name, r'\.cache/st/[A-Za-z0-9][A-Za-z0-9_.,=-]*$', 'a name `collect` copies back')
            env = {'ST_PROBE_REPORT': str(Path(tmp) / 'cache' / name), 'ST_PROBE_SESSION': 'exp/mixer fold#1'}
            path = probe_report.write_report(self.METRICS, self.PROOF, 4, ' NVIDIA GB10 ', environ=env)
            report = json.loads(path.read_text())
            self.assertEqual((report['schema'], report['session'], report['device'], report['samples'], report['passed']),
                             (2, 'exp/mixer fold#1', 'NVIDIA GB10', 4, True))
            self.assertIn('not a speed verdict', report['scope'])
            self.assertEqual(list(path.parent.iterdir()), [path], 'no temporary file is left beside it')
            self.assertEqual(fleet_single.read_report(path, 'exp/mixer fold#1'),
                             ('passed', '4 sample(s) on NVIDIA GB10, 2 metric(s)'))
            state, why = fleet_single.read_report(path, 'another-ticket')
            self.assertEqual(state, 'unreadable', 'the lane host\'s cache outlives a ticket: an older report is not this one\'s')
            self.assertIn("written for ticket 'exp/mixer fold#1'", why)

    def test_a_marker_that_did_not_hold_fails_whatever_the_file_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {'ST_PROBE_REPORT': str(Path(tmp) / 'r.json'), 'ST_PROBE_SESSION': 's'}
            path = probe_report.write_report(self.METRICS, dict(self.PROOF, byte_equal_rows4=False), 4, 'NVIDIA GB10', environ=env)
            self.assertEqual(json.loads(path.read_text())['failed'], ['byte_equal_rows4'])
            forged = dict(json.loads(path.read_text()), passed=True)
            path.write_text(json.dumps(forged))
            self.assertEqual(fleet_single.read_report(path, 's'),
                             ('failed', '4 sample(s) on NVIDIA GB10, 2 metric(s); proof not held: byte_equal_rows4'))

    def test_a_malformed_report_is_refused_where_it_is_written(self):
        env = {'ST_PROBE_REPORT': str(Path(tempfile.gettempdir()) / 'never-written.json')}
        for metrics, proof, samples, device in (({}, self.PROOF, 4, 'GB10'), ({'x': float('nan')}, self.PROOF, 4, 'GB10'),
                                                ({'x': True}, self.PROOF, 4, 'GB10'), (self.METRICS, {}, 4, 'GB10'),
                                                (self.METRICS, {'ok': 1}, 4, 'GB10'), (self.METRICS, self.PROOF, 0, 'GB10'),
                                                (self.METRICS, self.PROOF, 4, ' ')):
            with self.assertRaises(ValueError):
                probe_report.write_report(metrics, proof, samples, device, environ=env)
        self.assertFalse(Path(env['ST_PROBE_REPORT']).exists())
        for text in ('not json', '[]', json.dumps(dict(schema=1)), json.dumps(dict(schema=2, session='s', proof={}, metrics={}))):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'r.json'
                path.write_text(text)
                self.assertEqual(fleet_single.read_report(path, 's')[0], 'unreadable', text)

    def test_the_supervisor_names_the_report_and_the_runner_carries_it_into_the_container(self):
        boot = (ROOT / 'bench/fleet_boot.py').read_text(encoding='utf-8')
        self.assertIn("environment['ST_PROBE_SESSION'] = self.session", boot)
        self.assertIn("environment['ST_PROBE_REPORT'] = '/cache/' + fleet_single.report_name(self.session)", boot)
        self.assertLess(boot.index("environment['ST_PROBE_HOST'] = host"), boot.index("environment['ST_PROBE_REPORT']"),
                        'set with the lane\'s host, after the payload\'s own env prefix: a command cannot move its report')
        runner = (ROOT / 'probes/run_engine_probe.sh').read_text(encoding='utf-8')
        self.assertIn('for name in $(compgen -v ST_ || true)', runner, 'ST_* is what crosses into the container')
        self.assertIn('type=bind,src=$home/.cache/st,dst=/cache', runner, '/cache there is ~/.cache/st, which collect reads')
        self.assertIn("-name '*.json'", fleet_single.RESULT_FIND)

    def test_the_queues_log_line_says_what_the_report_says(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = fleet_single.report_name('t1')
            probe_report.write_report(self.METRICS, self.PROOF, 4, 'NVIDIA GB10',
                                      environ={'ST_PROBE_REPORT': str(Path(tmp) / name), 'ST_PROBE_SESSION': 't1'})
            argv = ['collect', '--host', 'srv4', '--since', '0', '--into', tmp]
            for copied, session, expected in (([name, 'kernel.log'], 't1', '2 file(s), report passed (4 sample(s) on NVIDIA GB10, 2 metric(s))'),
                                              (['kernel.log'], 't1', '1 file(s)'),          # most checks leave no report
                                              ([name], None, 1)):                           # the bare count an older caller reads
                with mock.patch.object(fleet_single, 'collect', return_value=copied), mock.patch('builtins.print') as out:
                    self.assertEqual(fleet_single.main(argv + (['--session', session] if session else [])), 0)
                out.assert_called_once_with(expected)
        fleet = (ROOT / 'bench/fleet.sh').read_text(encoding='utf-8')
        self.assertIn('--since "$2" --into "$into" --session "$1"', fleet)
        self.assertIn('logit "results of $1: $n in $into"', fleet)


if __name__ == '__main__':
    unittest.main()
