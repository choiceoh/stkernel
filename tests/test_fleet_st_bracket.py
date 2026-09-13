#!/usr/bin/env python3
"""The ST engine's bracket through the queue: admission, pins, the release cut, the judge, and a
rehearsal end to end -- no GPU, no docker, no fleet.

One committed sha per arm in production shape, two onepass runs per boot (D17), judged warm
against warm; the queue takes the fleet lease at GO and the release's own launcher verifies it.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
sys.path.insert(0, str(ROOT / 'launchers'))
import fleet_onepass as policy      # noqa: E402
import fleet_pin                    # noqa: E402
import fleet_prepare                # noqa: E402
import st_judge                     # noqa: E402
import st_release                   # noqa: E402

CAND, BASE = '0123abcdef0123abcdef0123abcdef0123abcdef', 'deadbeef00deadbeef00deadbeef00deadbeef00'


def bash_major():
    try:
        return int(subprocess.run(['bash', '-c', 'echo ${BASH_VERSINFO[0]}'], capture_output=True, text=True).stdout.strip())
    except ValueError:
        return 0


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.controller, self.repo = self.root / 'controller', self.root / 'candidate'
        for relative in (*policy.SHELL_ENTRIES, *policy.PYTHON_ENTRIES, *policy.ST_BRACKET_DEPENDENCIES,
                         'bench/fleet.sh', 'bench/onepass_deploy.py'):
            for base in (self.controller, self.repo):
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# reviewed fixture ' + relative + '\n')

    def validate(self, command, **kwargs):
        return policy.validate(command, self.repo, self.controller, {'REPO': str(self.repo)}, **kwargs)

    def test_the_bracket_is_admitted_with_shas_and_literal_names(self):
        for command in (['bash', 'bench/st_bracket.sh', 'pair', CAND],
                        ['bash', 'bench/st_bracket.sh', 'pair', CAND[:8], '--base', BASE],
                        ['bash', 'bench/st_bracket.sh', 'chain', 'A=' + BASE, 'B=' + CAND, 'A', 'B'],
                        ['bash', 'bench/st_bracket.sh', 'hold', CAND],
                        ['bash', 'bench/st_bracket.sh', 'hold', CAND, '45']):
            with self.subTest(command=command):
                contract = self.validate(command)
                self.assertEqual((contract['entry'], contract['gpus']), (policy.ST_BRACKET, 4))

    def test_the_grammar_refuses_what_is_not_a_sha_or_a_name(self):
        for tail in ([], ['pair'], ['pair', 'main'], ['pair', CAND, '--base'], ['pair', CAND, 'extra'],
                     ['chain'], ['chain', 'A'], ['chain', 'A=main'], ['chain', 'A=' + CAND, 'B'],
                     ['chain', 'a name=' + CAND], ['hold'], ['hold', CAND, '0'], ['hold', CAND, '1000'],
                     ['boot', CAND], ['pair', CAND + '; rm -rf /']):
            with self.subTest(tail=tail):
                with self.assertRaises(ValueError):
                    self.validate(['bash', 'bench/st_bracket.sh', *tail])

    def test_the_single_lane_refuses_a_bracket(self):
        with self.assertRaisesRegex(ValueError, 'needs the four Sparks'):
            self.validate(['bash', 'bench/st_bracket.sh', 'pair', CAND], kind='single')

    def test_what_the_controller_runs_is_pinned(self):
        for relative in policy.ST_BRACKET_DEPENDENCIES + (policy.ST_BRACKET,):
            with self.subTest(relative=relative):
                path = self.repo / relative
                original = path.read_text()
                path.write_text(original + 'echo tampered\n')
                try:
                    with self.assertRaisesRegex(ValueError, 'differs from the current canonical'):
                        self.validate(['bash', 'bench/st_bracket.sh', 'pair', CAND])
                finally:
                    path.write_text(original)

    def test_the_bracket_boots_the_release_s_own_launcher_which_is_not_pinned(self):
        """The launcher under test comes from the arm's release, like the engine it boots."""
        self.assertNotIn('launchers/start-st-glm53.sh', policy.ST_BRACKET_DEPENDENCIES)
        runner = (ROOT / 'bench/st_bracket.sh').read_text()
        self.assertIn('bash "$RELEASE/launchers/start-st-glm53.sh" start', runner)
        self.assertIn('bash "$RELEASE/launchers/start-st-glm53.sh" stop', runner)

    def test_a_rehearsal_is_allowed_and_a_rehearsal_takes_no_gpu(self):
        contract = self.validate(['env', 'FLEET_REHEARSE=1', 'bash', 'bench/st_bracket.sh', 'pair', CAND], rehearsal_only=True)
        self.assertEqual(contract['entry'], policy.ST_BRACKET)

    def test_the_probe_verb_belongs_to_the_live_lane(self):
        """Two onepass runs on the live door, beside production: a probe ticket, never a boot."""
        for tail in (['probe'], ['probe', CAND]):
            with self.subTest(tail=tail):
                self.assertEqual(self.validate(['bash', 'bench/st_bracket.sh', *tail], kind='probe')['kind'], 'probe')
                with self.assertRaisesRegex(ValueError, 'belongs to the live-serving lane'):
                    self.validate(['bash', 'bench/st_bracket.sh', *tail])
        for tail in (['probe', 'main'], ['probe', CAND, '1']):
            with self.assertRaises(ValueError):
                self.validate(['bash', 'bench/st_bracket.sh', *tail], kind='probe')
        with self.assertRaisesRegex(ValueError, 'live-serving lane accepts only'):
            self.validate(['bash', 'bench/st_bracket.sh', 'pair', CAND], kind='probe')
        fleet = (ROOT / 'bench/fleet.sh').read_text()
        self.assertIn('  st-probe)', fleet)
        self.assertIn('run --gpu --probe ${detach[@]+"${detach[@]}"} "$s" "$est" "$note" -- bash "$REPO/bench/st_bracket.sh" probe', fleet)

    def test_the_probe_measures_only_the_release_the_door_serves(self):
        """A probe ticket queued before a deploy and run after it would label the next engine's
        numbers with the old commit: the probe reads the door's ST_RELEASE and refuses a mismatch."""
        text = (ROOT / 'bench/st_bracket.sh').read_text()
        body = text[text.index('probe() {'):]
        self.assertIn('docker exec st-glm53 printenv ST_RELEASE', body)
        self.assertIn('ABORT: the door serves release $served, not ${ARM_SHA:0:12}', body)
        self.assertLess(body.index('printenv ST_RELEASE'), body.index('for run in $(seq 1 "$runs")'), 'before any run')
        self.assertIn('[ "$REHEARSE" != 1 ]', body[:body.index('printenv ST_RELEASE')], 'a rehearsal has no door to ask')

    def test_the_runner_snapshot_carries_what_the_bracket_needs(self):
        pinned = set(fleet_pin.source_files(ROOT))
        for relative in ('bench/st_bracket.sh', 'bench/st_judge.py', 'bench/onepass.py', 'launchers/st_release.py'):
            self.assertIn(relative, pinned, relative)

    def test_fleet_sh_dispatches_the_three_verbs_as_boot_tickets(self):
        fleet = (ROOT / 'bench/fleet.sh').read_text()
        for verb in ('st-pair)', 'st-chain)', 'st-hold)'):
            self.assertIn('  ' + verb, fleet)
        self.assertIn('bash "$REPO/bench/st_bracket.sh" pair "$sha"', fleet)
        self.assertIn('bash "$REPO/bench/st_bracket.sh" chain "$@"', fleet)
        self.assertIn('bash "$REPO/bench/st_bracket.sh" hold "$sha" "$est"', fleet)
        self.assertIn('fleet.sh st-pair s <sha> [--base <sha>] [est] [note]', fleet)


class PrepareTests(unittest.TestCase):
    def test_a_bracket_ticket_pins_its_arms_not_the_checkout_head(self):
        """A moving controller checkout paused an ST ticket with "queued checkout revision changed"
        (45차 §95). The bracket's revision is the sha it names, not HEAD."""
        self.assertTrue(fleet_prepare.pins_its_own_revision(['bash', 'bench/st_bracket.sh', 'pair', CAND]))
        self.assertTrue(fleet_prepare.pins_its_own_revision(['env', 'FLEET_REHEARSE=1', 'bash', 'bench/st_bracket.sh', 'hold', CAND]))
        self.assertFalse(fleet_prepare.pins_its_own_revision(['bash', 'bench/pair.sh', 'A']))
        self.assertFalse(fleet_prepare.pins_its_own_revision([]))
        source = (ROOT / 'bench/fleet_prepare.py').read_text()
        create = source[source.index('def prepare('):]
        self.assertIn('elif pins_its_own_revision(command):', create)
        self.assertLess(create.index('elif pins_its_own_revision(command):'), create.index("value['head'] = [repo"))


class ReleaseCutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for relative in ('engine/base/x.py', 'launchers/start.sh', 'tests/test_x.py', 'probes/p.py', 'README.md'):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# ' + relative + '\n')
        env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@x', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@x')
        for cmd in (['git', 'init', '-q'], ['git', 'add', '.'], ['git', 'commit', '-q', '-m', 'one']):
            subprocess.run(cmd, cwd=self.source, check=True, env=env, capture_output=True)
        self.sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=self.source, capture_output=True, text=True, check=True).stdout.strip()

    def test_cut_archives_the_carried_parts_from_git_once(self):
        self.assertEqual(st_release.resolve(self.sha[:8], source=self.source), self.sha)
        logs = []
        target = st_release.cut(self.sha, source=self.source, releases=self.root / 'releases', log=logs.append, meta=None)
        self.assertEqual(target, self.root / 'releases' / self.sha[:12])
        for relative in ('engine/base/x.py', 'launchers/start.sh', 'tests/test_x.py', 'probes/p.py'):
            self.assertTrue((target / relative).is_file(), relative)
        self.assertFalse((target / 'README.md').exists(), 'only the carried parts')
        self.assertFalse((target.with_suffix('.partial')).exists())
        again = st_release.cut(self.sha, source=self.source, releases=self.root / 'releases', log=logs.append, meta=None)
        self.assertEqual(again, target)
        self.assertTrue(any('already cut' in line for line in logs))

    def test_an_unknown_commit_is_refused_not_guessed(self):
        with self.assertRaisesRegex(ValueError, 'not a commit'):
            st_release.resolve('0123abcdef', source=self.source)
        with self.assertRaisesRegex(ValueError, 'not a commit id'):
            st_release.resolve('main', source=self.source)

    def test_deployed_reads_what_deploy_watch_recorded(self):
        state = self.root / 'deploy-state.json'
        self.assertEqual(st_release.deployed(state), '')
        state.write_text(json.dumps({'deployed': BASE, 'release': '/x'}))
        self.assertEqual(st_release.deployed(state), BASE)

    def test_deploy_watch_and_the_bracket_cut_through_one_module(self):
        watch = (ROOT / 'launchers/st-deploy-watch.py').read_text()
        self.assertIn('st_release.cut(sha, source=SOURCE, releases=RELEASES', watch)
        module = (ROOT / 'launchers/st_release.py').read_text()
        self.assertIn('set -o pipefail', module)              # a half-written archive is not a release
        self.assertIn('| tar -x', module)
        runner = (ROOT / 'bench/st_bracket.sh').read_text()
        self.assertIn('"$REPO/launchers/st_release.py" cut', runner)


def record(sha, name, windows, *, run_index=2, boot='b', quality=(9, 9), dirty=0, issues=(), rehearsal=False, key='arm_sha'):
    rec = {'name': name, 'engine': 'st', key: sha, 'run_index': run_index, 'boot_id': boot + '|started',
           'decode': {'windows_med': windows, 'tokens_per_step': 3.5}, 'quality': {'ok': quality[0], 'total': quality[1]},
           'korean': {'dirty': dirty, 'n': 5}, 'traffic': {'issues': list(issues)},
           'prefill': [{'ctx': 2000, 'cold_s': 6.0, 'warm_tok_s': 3000.0}, {'ctx': 32000, 'cold_s': 40.0, 'warm_tok_s': 3100.0}],
           'engine_shape': {'max_concurrent_requests': 4}}
    if rehearsal:
        rec['rehearsal'] = True
    return rec


class JudgeTests(unittest.TestCase):
    def test_warm_against_warm_with_the_base_s_spread_as_the_floor(self):
        rows = [record(BASE, 'B', 12.0, boot='b1'), record(BASE, 'B', 12.0, run_index=1, boot='b1'),
                record(BASE, 'B', 12.4, boot='b2'), record(CAND, 'C', 13.5, boot='c1'), record(CAND, 'C', 9.0, run_index=1, boot='c1')]
        out = st_judge.judge(rows, CAND, BASE)
        self.assertEqual((out['n_cand'], out['n_base']), (1, 2))
        self.assertAlmostEqual(out['delta_pct'], (13.5 - 12.2) / 12.2 * 100, places=6)
        self.assertAlmostEqual(out['floor_pct'], 0.4 / 12.2 * 100, places=6)
        self.assertIn('BEYOND the base floor', out['verdict'])
        self.assertEqual(out['cand_summary']['cold_ttft_2k_s'], 6.0)      # the cold column rides beside
        self.assertEqual(len(st_judge.samples(rows, BASE)), 2)
        table = st_judge.table(out)
        self.assertIn('decode step/s', table)
        self.assertIn('verdict:', table)

    def test_two_runs_on_one_boot_are_one_sample(self):
        rows = [record(BASE, 'B', 12.0, boot='b1'), record(BASE, 'B-again', 12.0, run_index=2, boot='b1')]
        self.assertEqual(len(st_judge.samples(rows, BASE)), 1)

    def test_a_single_base_boot_has_no_floor_and_says_so(self):
        rows = [record(BASE, 'B', 12.0), record(CAND, 'C', 12.6)]
        self.assertIn('NO FLOOR', st_judge.judge(rows, CAND, BASE)['verdict'])

    def test_a_single_base_boot_borrows_the_floor_other_commits_measured(self):
        """Booting the base again only to learn what noise is costs a boot; the noise of this engine on
        these boxes is in the records already. The pooled floor is the median spread of every commit
        that has two boots -- and the verdict says it borrowed."""
        import statistics
        X, Y = 'aaaa' * 10, 'bbbb' * 10
        rows = [record(BASE, 'B', 12.0), record(CAND, 'C', 12.3),
                record(X, 'X', 10.0, boot='x1'), record(X, 'X', 10.2, boot='x2'),          # 0.2 / 10.1 = 1.98% spread
                record(Y, 'Y', 20.0, boot='y1'), record(Y, 'Y', 20.8, boot='y2')]          # 0.8 / 20.4 = 3.92% spread
        out = st_judge.judge(rows, CAND, BASE)
        self.assertEqual(out['floor_source'], 'pooled')
        self.assertAlmostEqual(out['floor_pct'], statistics.median([0.2 / 10.1, 0.8 / 20.4]) * 100, places=6)
        self.assertIn('WITHIN the pooled floor', out['verdict'])                           # +2.5% against a 2.95% floor
        self.assertIn('pooled from 2 commits', out['verdict'])
        rows.append(record(BASE, 'B', 12.5, boot='b2'))
        self.assertEqual(st_judge.judge(rows, CAND, BASE)['floor_source'], 'base', 'the base speaks for itself once it can')

    def test_the_adopted_candidate_s_records_are_the_next_base_s(self):
        """The operator's rule (2026-09-13): a candidate that is adopted brings its own measurement
        along as the next baseline. Identity is the engine tree, so the squash commit main made of
        the candidate, or a fleet-side merge that left engine/ alone, is the same sample."""
        SQUASH = 'cccc' * 10
        rows = [dict(record(CAND, 'C', 13.0), arm_tree='0d3c61aa802d'), record('dddd' * 10, 'N', 14.0)]
        self.assertEqual(len(st_judge.samples(rows, SQUASH)), 0, 'by commit: a stranger')
        self.assertEqual(len(st_judge.samples(rows, SQUASH, tree='0d3c61aa802d')), 1, 'by tree: the adopted candidate')
        out = st_judge.judge(rows, 'dddd' * 10, SQUASH, base_tree='0d3c61aa802d')
        self.assertNotIn('NO BASE', out['verdict'])
        self.assertEqual(out['base_summary']['n'], 1)
        self.assertEqual(len(st_judge.samples(rows, SQUASH, tree='ffffffffffff')), 0, 'another tree is another engine')

    def test_a_probe_s_run_after_a_reset_is_a_warm_sample(self):
        """One run on a live door is a sample: run 1 after a prefix reset is warm (cold=reset), not the
        boot's cold column -- so a probe needs one run, not two."""
        rows = [dict(record(BASE, 'd17', 12.0, run_index=1, boot='live'), cold='reset')]
        self.assertEqual(len(st_judge.samples(rows, BASE)), 1)
        self.assertEqual(st_judge.colds(rows, BASE), [])

    def test_boots_names_the_boot_each_sample_came_from(self):
        import subprocess
        import tempfile
        rows = [record(BASE, 'B', 12.0, boot='b1'), record(BASE, 'B', 12.0, run_index=1, boot='b1'), record(BASE, 'B', 12.2, boot='b2')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'r.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            out = subprocess.run([sys.executable, str(ROOT / 'bench/st_judge.py'), 'boots', '--sha', BASE, '--jsonl', str(path)],
                                 capture_output=True, text=True)
            self.assertEqual(out.stdout.split(), ['b1|started', 'b2|started'])

    def test_gates_make_a_record_no_evidence(self):
        rows = [record(BASE, 'B', 12.0), record(CAND, 'C', 15.0, quality=(8, 9))]
        out = st_judge.judge(rows, CAND, BASE)
        self.assertIn('NO EVIDENCE', out['verdict'])
        self.assertTrue(out['invalid_candidates'])
        rows = [record(BASE, 'B', 12.0), record(CAND, 'C', 15.0, dirty=1)]
        self.assertIn('NO EVIDENCE', st_judge.judge(rows, CAND, BASE)['verdict'])
        rows = [record(BASE, 'B', 12.0), record(CAND, 'C', 15.0, issues=['traffic'])]
        self.assertIn('NO EVIDENCE', st_judge.judge(rows, CAND, BASE)['verdict'])

    def test_identity_is_the_commit_the_bracket_named_or_the_release_the_launcher_stamped(self):
        rows = [record(BASE[:12], 'prod', 12.0, key='release'), record(CAND, 'C', 12.0)]
        self.assertEqual(len(st_judge.samples(rows, BASE)), 1)
        self.assertEqual(st_judge.identity({'release': 'st-engine'}), '')          # not a sha: no identity
        self.assertEqual(st_judge.identity({'engine_source_sha256': 'ab' * 32}), '')

    def test_a_probe_s_first_run_is_not_the_cold_column(self):
        """A probe's run 1 follows a prefix reset, not a boot: TTFT without the compile tail."""
        rows = [record(BASE, 'B', 12.0), record(BASE, 'B', 12.0, run_index=1),
                dict(record(BASE, 'd17', 12.1, run_index=1, boot='live'), cold='reset'),
                dict(record(CAND, 'C', 12.5, run_index=1, boot='c1'), cold='reset'), record(CAND, 'C', 12.5, boot='c1')]
        self.assertEqual([r['name'] for r in st_judge.colds(rows, BASE)], ['B'])
        self.assertEqual(st_judge.colds(rows, CAND), [])
        self.assertIsNone(st_judge.judge(rows, CAND, BASE)['cand_summary']['cold_ttft_2k_s'])

    def test_rehearsal_records_count_only_when_asked(self):
        rows = [record(BASE, 'B', 12.0, rehearsal=True)]
        self.assertEqual(len(st_judge.samples(rows, BASE)), 0)
        self.assertEqual(len(st_judge.samples(rows, BASE, allow_rehearsal=True)), 1)

    def test_the_cli_writes_a_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / 'onepass.jsonl'
            jsonl.write_text('\n'.join(json.dumps(r) for r in (record(BASE, 'B', 12.0), record(CAND, 'C', 12.1))) + '\n')
            out = subprocess.run([sys.executable, str(ROOT / 'bench/st_judge.py'), 'judge', '--cand', CAND, '--base', BASE,
                                  '--jsonl', str(jsonl), '--write'], capture_output=True, text=True,
                                 env=dict(os.environ, ONEPASS_VERDICTS=str(Path(tmp) / 'verdicts.jsonl')))
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('verdict:', out.stdout)
            verdict = json.loads((Path(tmp) / 'verdicts.jsonl').read_text().splitlines()[-1])
            self.assertEqual((verdict['engine'], verdict['cand'], verdict['base']), ('st', CAND[:12], BASE[:12]))
            count = subprocess.run([sys.executable, str(ROOT / 'bench/st_judge.py'), 'samples', '--sha', BASE, '--jsonl', str(jsonl)],
                                   capture_output=True, text=True)
            self.assertEqual(count.stdout.strip(), '1')


@unittest.skipUnless(bash_major() >= 4, 'the bracket runner needs bash 4 (associative arrays); run in the Linux container')
class RehearsalTests(unittest.TestCase):
    """FLEET_REHEARSE=1 boots nothing and fabricates records, so the flow and the judge can be checked without GPUs."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp = Path(self.temporary.name)
        self.jsonl = self.tmp / 'onepass.jsonl'
        self.env = dict(os.environ, FLEET_REHEARSE='1', REPO=str(ROOT), LOGD=str(self.tmp / 'logs'),
                        ONEPASS_JSONL=str(self.jsonl), ONEPASS_VERDICTS=str(self.tmp / 'verdicts.jsonl'),
                        ST_RELEASES=str(self.tmp / 'releases'), ST_SOURCE=str(ROOT), FLEET_SESSION='rehearse',
                        ST_DEPLOY_STATE=str(self.tmp / 'deploy-state.json'))
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        self.cand, self.base = head, CAND        # one commit the source has, one it does not: both rehearse

    def run_bracket(self, *args):
        return subprocess.run(['bash', str(ROOT / 'bench/st_bracket.sh'), *args], env=self.env, capture_output=True, text=True, timeout=120)

    def records(self):
        return [json.loads(line) for line in self.jsonl.read_text().splitlines() if line.strip()]

    def test_pair_rehearses_two_runs_per_arm_then_judges_then_reuses_the_base(self):
        out = self.run_bracket('pair', self.cand, '--base', self.base)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        recs = self.records()
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(r['rehearsal'] and r['engine'] == 'st' for r in recs))
        self.assertEqual(sorted(r['run_index'] for r in recs), [1, 1, 2, 2])
        self.assertEqual({r['arm_sha'][:12] for r in recs}, {self.cand[:12], self.base[:12]})
        self.assertIn('verdict:', out.stdout)
        self.assertTrue((self.tmp / 'verdicts.jsonl').exists())
        again = self.run_bracket('pair', self.cand, '--base', self.base)
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertIn('reused', again.stdout)
        self.assertEqual(len(self.records()), 6, 'the base was not booted again')

    def test_pair_takes_the_deployed_commit_as_the_base_by_default(self):
        (self.tmp / 'deploy-state.json').write_text(json.dumps({'deployed': self.base}))
        out = self.run_bracket('pair', self.cand)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(self.base[:12], out.stdout)
        missing = self.run_bracket('pair', self.cand)
        (self.tmp / 'deploy-state.json').unlink()
        self.assertNotEqual(self.run_bracket('pair', self.cand).returncode, 0, 'no base and nothing deployed: refused')

    def test_chain_alternates_and_judges_every_other_commit_against_the_first(self):
        out = self.run_bracket('chain', 'A=' + self.base, 'B=' + self.cand, 'A', 'B')
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        recs = self.records()
        self.assertEqual([r['name'] for r in recs], ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'])
        self.assertEqual(out.stdout.count('verdict:'), 1)
        self.assertIn('judge B against A', out.stdout)
        self.assertTrue(all('arm_tree' in r for r in recs if r['name'] == 'B'), 'the real commit names its engine tree')
        self.assertTrue(all('arm_tree' not in r for r in recs if r['name'] == 'A'), 'a commit the source lacks has no tree')
        out = self.run_bracket('chain', '--reuse', 'A=' + self.base, 'B=' + self.cand, 'A', 'B')
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(len(self.records()), 8, '--reuse: every arm had a sample already, nothing was booted')
        self.assertEqual(out.stdout.count('reused:'), 4)
        self.assertIn('verdict:', out.stdout)

    def test_probe_rehearses_one_run_on_the_live_door_and_names_the_engine_tree(self):
        out = self.run_bracket('probe', self.cand)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        recs = self.records()
        self.assertEqual([(r['run_index'], r['cold'], r['name']) for r in recs], [(1, 'reset', 'd17-' + self.cand[:12])],
                         'one run: a boot is one sample, and after a reset it is warm')
        self.assertEqual(recs[0]['arm_tree'], subprocess.run(['git', 'rev-parse', self.cand + ':engine'], cwd=ROOT,
                                                             capture_output=True, text=True).stdout.strip()[:12])
        self.assertIn('no boot, no lease', out.stdout)
        self.env['ST_PROBE_RUNS'] = '2'
        out = self.run_bracket('probe', self.cand)
        self.assertEqual(len(self.records()), 3, 'ST_PROBE_RUNS=2 still gives two')
        del self.env['ST_PROBE_RUNS']
        (self.tmp / 'deploy-state.json').write_text(json.dumps({'deployed': self.base}))
        out = self.run_bracket('probe')
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(self.records()[-1]['arm_sha'][:12], self.base[:12], 'no sha: the deployed commit')

    def test_hold_rehearses_a_boot_and_lets_go(self):
        out = self.run_bracket('hold', self.cand, '1')
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn('hold over', out.stdout)
        self.assertIn('fleet.sh cancel rehearse', out.stdout)


if __name__ == '__main__':
    unittest.main()
