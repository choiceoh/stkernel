"""bench/feedback.py names, per file, the checks the queue already admits -- and never one it refuses."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import feedback
import fleet_onepass


def at(*parts):
    """A repo path built from parts: a literal one here would make THIS file a check that 'names' it."""
    return '/'.join(parts)


def write(root, relative, text=''):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class LaneTests(unittest.TestCase):
    """The lanes are the queue's own tables, read not copied."""

    @classmethod
    def setUpClass(cls):
        cls.graph = feedback.Graph(ROOT)
        cls.sources = cls.graph.sources()

    def test_lane_tables_are_the_modules_own_literals(self):
        self.assertEqual(self.graph.st_probes, fleet_onepass.ST_PROBES)
        self.assertEqual(self.graph.budgets, fleet_onepass.ST_PROBE_BUDGET_GIB)

    def test_every_single_rung_is_a_command_the_queue_admits_to_one_gpu(self):
        single = [s for s in self.sources if s['lane'] == 'single']
        self.assertEqual({s['path'] for s in single}, set(fleet_onepass.ST_PROBES), 'a named ST check is missing from the tree')
        for source in single:
            argv = shlex.split(source['command'].split(' -- ', 1)[1])
            contract = fleet_onepass.validate(argv, ROOT, ROOT, {}, kind=fleet_onepass.SINGLE)
            self.assertEqual(contract['gpus'], 1, source['path'])
            self.assertEqual(contract['budget_gib'], source['budget_gib'], source['path'])

    def test_no_unadmitted_probe_would_pass_the_queue(self):
        unadmitted = [s['path'] for s in self.sources if s['lane'] == 'unadmitted']
        self.assertTrue(unadmitted, 'the probes the queue refuses are the finding this lane exists to show')
        for path in unadmitted[:12]:
            with self.assertRaisesRegex(ValueError, 'not a canonical ST check'):
                fleet_onepass.validate(['bash', 'probes/run_engine_probe.sh', path], ROOT, ROOT, {}, kind=fleet_onepass.SINGLE)

    def test_cpu_rung_is_the_one_file_tools_check_would_run(self):
        cpu = [s for s in self.sources if s['lane'] == 'cpu']
        self.assertGreater(len(cpu), 300)
        for source in cpu:
            argv = shlex.split(source['command'])
            self.assertEqual(argv[:3], ['python3', 'tools/check.py', '--pattern'], source['path'])
            # tools/check.py: sorted((ROOT / "tests").glob(pattern + ".py"))
            self.assertEqual([p.relative_to(ROOT).as_posix() for p in (ROOT / 'tests').glob(argv[3] + '.py')], [source['path']])

    def test_a_kernel_file_is_answered_by_its_own_test_then_its_own_st_check(self):
        ladder = self.graph.rungs(at('engine', 'kernels', 'mhc_contract.py'))
        by_lane = {}
        for rung in ladder['rungs']:
            by_lane.setdefault(rung['lane'], rung)
        self.assertEqual(by_lane['cpu']['path'], at('tests', 'test_engine_mhc_contract.py'))
        self.assertEqual(by_lane['single']['path'], at('probes', 'engine_mhc_contract_check.py'))
        self.assertEqual((by_lane['cpu']['distance'], by_lane['single']['distance']), (1, 1))
        self.assertEqual([r['lane'] for r in ladder['rungs']], sorted((r['lane'] for r in ladder['rungs']), key=feedback.LANES.index))
        self.assertEqual(ladder['verdict'], feedback.ST_PAIR)

    def test_a_shell_script_is_answered_by_the_tests_that_name_it(self):
        ladder = self.graph.rungs(at('bench', 'fleet.sh'), limit=None)
        self.assertIn(at('tests', 'test_fleet_single.py'), [r['path'] for r in ladder['rungs'] if r['distance'] == 1])
        self.assertIsNone(ladder['verdict'], 'the queue does not ship into serving')


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name).resolve()
        write(root, 'bench/fleet_onepass.py', "ST_PROBES = ('probes/engine_a_check.py',)\n"
                                              "ST_PROBE_BUDGET_GIB = {'probes/engine_a_check.py': 24}\n")
        write(root, 'bench/tool.py', 'import json\n')
        write(root, 'bench/queue.sh', 'exit 0\n')
        write(root, at('engine', '__init__.py'))
        write(root, at('engine', 'kernels', '__init__.py'))
        write(root, 'engine/kernels/common.py', 'X = 1\n')
        write(root, 'engine/kernels/a.py', 'from . import common\nfrom .common import X\n')
        write(root, 'engine/kernels/b.py', 'from engine.kernels import a\n')
        write(root, 'engine/kernels/lonely.py', 'Y = 2\n')
        write(root, 'tests/test_a.py', '"""The a kernel, alone."""\nimport unittest\nfrom engine.kernels import a\n')
        write(root, 'tests/test_far.py', 'import unittest\nimport engine.kernels.b\n')
        write(root, 'tests/test_queue.py', 'import unittest\nSCRIPT = "bench/queue.sh"\nOTHER = "queue.sh"\n')
        write(root, 'tests/test_tool.py', 'import unittest\nimport tool\n')
        write(root, 'probes/engine_a_check.py', '"""Judge a on the device."""\nfrom engine.kernels import a\n')
        write(root, 'probes/engine_b_bench.py', 'from engine.kernels import a\n')
        self.graph = feedback.Graph(root)

    def test_nearest_first_within_a_lane_and_cheapest_lane_first(self):
        ladder = self.graph.rungs('engine/kernels/a.py')
        self.assertEqual([(r['lane'], r['path'], r['distance']) for r in ladder['rungs']],
                         [('cpu', 'tests/test_a.py', 1), ('cpu', 'tests/test_far.py', 2),
                          ('single', 'probes/engine_a_check.py', 1)])
        self.assertEqual(ladder['rungs'][0]['answers'], 'The a kernel, alone.')
        self.assertEqual(ladder['rungs'][0]['command'], 'python3 tools/check.py --pattern test_a')
        self.assertEqual(ladder['rungs'][2]['budget_gib'], 24)
        self.assertEqual(ladder['unadmitted'], ['probes/engine_b_bench.py'])
        self.assertEqual(ladder['verdict'], feedback.ST_PAIR)
        self.assertTrue(ladder['local'])

    def test_relative_imports_and_package_inits_are_followed(self):
        ladder = self.graph.rungs('engine/kernels/common.py')
        self.assertEqual([(r['path'], r['distance']) for r in ladder['rungs'] if r['lane'] == 'cpu'][0], ('tests/test_a.py', 2))
        self.assertFalse(ladder['local'], 'nothing imports it directly: every rung tests it through a')
        init = self.graph.rungs(at('engine', 'kernels', '__init__.py'))
        self.assertEqual(init['rungs'][0]['distance'], 1, 'importing a submodule runs its package')

    def test_a_named_repo_path_is_an_edge_and_a_bare_filename_is_not(self):
        ladder = self.graph.rungs('bench/queue.sh')
        self.assertEqual([(r['path'], r['distance']) for r in ladder['rungs']], [('tests/test_queue.py', 1)])
        self.assertEqual(self.graph.direct(self.root / 'tests/test_queue.py')[1], {self.root / 'bench/queue.sh'})

    def test_tooling_has_no_speed_verdict_and_an_unreached_file_says_so(self):
        tool = self.graph.rungs('bench/tool.py')
        self.assertEqual([r['path'] for r in tool['rungs']], ['tests/test_tool.py'])
        self.assertIsNone(tool['verdict'])
        self.assertIn('does not ship into serving', feedback.render(tool))
        lonely = self.graph.rungs('engine/kernels/lonely.py')
        self.assertEqual(lonely['rungs'], [])
        self.assertIn('no CPU test or admitted one-GPU check reaches this file', feedback.render(lonely))

    def test_an_edited_check_is_its_own_nearest_answer(self):
        test = self.graph.rungs('tests/test_a.py')
        self.assertEqual([(r['path'], r['distance']) for r in test['rungs']], [('tests/test_a.py', 0)])
        probe = self.graph.rungs('probes/engine_a_check.py')
        self.assertEqual([(r['lane'], r['distance']) for r in probe['rungs']], [('single', 0)])
        refused = self.graph.rungs('probes/engine_b_bench.py')
        self.assertEqual((refused['rungs'], refused['unadmitted']), ([], ['probes/engine_b_bench.py']))

    def test_the_limit_counts_what_it_leaves_out(self):
        ladder = self.graph.rungs('engine/kernels/a.py', limit=1)
        self.assertEqual([r['path'] for r in ladder['rungs']], ['tests/test_a.py', 'probes/engine_a_check.py'])
        self.assertEqual(ladder['more'], {'cpu': 1})
        self.assertIn('+1 farther', feedback.render(ladder))

    def test_a_lane_table_that_stops_being_a_literal_is_an_error_not_an_empty_lane(self):
        write(self.root, 'bench/fleet_onepass.py', 'ST_PROBES = tuple(load())\nST_PROBE_BUDGET_GIB = {}\n')
        with self.assertRaisesRegex(ValueError, 'ST_PROBES is no longer a top-level literal'):
            feedback.Graph(self.root)

    def test_index_and_changed_files_from_the_command_line(self):
        run = lambda *args: subprocess.run([sys.executable, str(ROOT / 'bench/feedback.py'), '--root', str(self.root), *args],
                                           capture_output=True, text=True, timeout=60)
        index = json.loads(run('--index').stdout)
        self.assertEqual((index['schema'], index['lanes']), (1, ['cpu', 'single', 'verdict']))
        checks = {c['path']: c for c in index['checks']}
        self.assertEqual(checks['probes/engine_a_check.py']['lane'], 'single')
        self.assertEqual(checks['probes/engine_b_bench.py']['lane'], 'unadmitted')
        self.assertEqual(checks['tests/test_far.py']['reaches']['engine/kernels/a.py'], 2)
        self.assertNotIn('engine/kernels/common.py', checks['tests/test_far.py']['reaches'], 'three imports away: past --depth 2')
        git = lambda *args: subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=T', '-c', 'user.email=t@invalid', *args],
                                           check=True, capture_output=True)
        git('init', '-q')
        git('add', '-A')
        git('commit', '-qm', 'base')
        write(self.root, 'engine/kernels/a.py', 'from . import common\nZ = 3\n')
        git('commit', '-qam', 'change')
        ladders = json.loads(run('--base', 'HEAD~1', '--json').stdout)
        self.assertEqual([l['file'] for l in ladders], ['engine/kernels/a.py'])
        nothing = run('--base', 'HEAD')
        self.assertEqual((nothing.returncode, nothing.stdout.strip()), (0, 'HEAD changed no file since HEAD'))
        missing = run('engine/kernels/nothing.py')
        self.assertEqual(missing.returncode, 2)


if __name__ == '__main__':
    unittest.main()
