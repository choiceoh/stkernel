"""Preparation failures must happen before reservation and preserve source identity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_prepare as prep


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.directory = self.root / 'fleet'
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'fixture')
        (self.repo/'input.py').write_text('print(1)\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'initial')

    def git(self, *args):
        return subprocess.check_output(['git','-C',str(self.repo),*args], text=True).strip()

    def prepare(self, command=None, **kwargs):
        path = prep.prepare(self.directory, 'fixture', command or [sys.executable,'input.py'], self.repo, **kwargs)
        return json.loads(path.read_text())

    def test_unchanged_inputs_validate_without_reexecuting_cpu(self):
        script = self.repo/'check.sh'
        script.write_text('echo run >> count\n')
        classifier = self.repo/'fleet.sh'
        classifier.write_text('echo nogpu\n')
        spec = self.root/'spec.json'
        spec.write_text(json.dumps(dict(cpu_command=['bash','check.sh'])))
        value = self.prepare(spec_path=spec, fleet=classifier)
        for _ in range(3): prep.validate(value)
        self.assertEqual((self.repo/'count').read_text(), 'run\n')

    def test_changed_script_and_head_rejected(self):
        value = self.prepare()
        (self.repo/'input.py').write_text('print(2)\n')
        with self.assertRaisesRegex(ValueError, 'queued input changed'): prep.validate(value)
        (self.repo/'input.py').write_text('print(1)\n')
        self.git('commit','--allow-empty','-qm','new revision')
        with self.assertRaisesRegex(ValueError, 'checkout revision changed'): prep.validate(value)

    def test_missing_input_fresh_output_and_image_fail_early(self):
        spec = self.root/'spec.json'
        spec.write_text(json.dumps(dict(required_paths=['missing'])))
        with self.assertRaisesRegex(ValueError,'required path'): self.prepare(spec_path=spec)
        spec.write_text(json.dumps(dict(absent_paths=['input.py'])))
        with self.assertRaisesRegex(ValueError,'fresh output'): self.prepare(spec_path=spec)
        spec.write_text(json.dumps(dict(images=['fixture-image'])))
        original = prep.run
        def run(argv, *args):
            if argv[0] == 'docker': raise ValueError('image missing')
            return original(argv,*args)
        with mock.patch.object(prep,'run',side_effect=run):
            with self.assertRaisesRegex(ValueError,'image missing'): self.prepare(spec_path=spec)

    def test_actual_campaign_ancestry_guard_before_queue_and_after_wait(self):
        script = self.repo/'campaign.sh'
        script.write_text('''#!/bin/bash
set -eu
git merge-base --is-ancestor origin/main HEAD || { echo 'ABORT: candidate needs current main'; exit 2; }
''')
        self.git('add','.'); self.git('commit','-qm','campaign')
        self.git('update-ref','refs/remotes/origin/main','HEAD')
        value = self.prepare(['bash','campaign.sh'])
        prep.validate(value)
        old = self.git('rev-parse','HEAD')
        self.git('commit','--allow-empty','-qm','new main')
        future = self.git('rev-parse','HEAD')
        self.git('reset','--hard',old)
        self.git('update-ref','refs/remotes/origin/main',future)
        with self.assertRaisesRegex(ValueError,'campaign.sh:3: candidate must include origin/main'):
            prep.validate(value)
        with self.assertRaisesRegex(ValueError,'candidate must include origin/main'):
            self.prepare(['bash','campaign.sh'])
        self.assertFalse((self.directory/'holder').exists())

    def test_external_campaign_guard_uses_execution_checkout(self):
        script=self.root/'external.sh'
        script.write_text('git merge-base --is-ancestor origin/main HEAD || exit 2\n')
        self.git('update-ref','refs/remotes/origin/main','HEAD')
        value=self.prepare(['bash',str(script)])
        self.assertEqual(value['checks'][0]['repo'],str(self.repo.resolve()))

    def test_clean_guard_is_literal_and_comments_not_executed(self):
        script=self.repo/'campaign.sh'
        script.write_text('# git merge-base --is-ancestor invalid HEAD\n[[ -z $(git status --porcelain) ]] || exit 2\n')
        self.git('add','.');self.git('commit','-qm','clean')
        value=self.prepare(['bash','campaign.sh'])
        self.assertEqual([c['kind'] for c in value['checks']],['clean'])
        (self.repo/'dirty').touch()
        with self.assertRaisesRegex(ValueError,'clean checkout'): prep.validate(value)

    def test_missing_entrypoint_is_refused_but_inline_code_is_not_a_filename(self):
        with self.assertRaisesRegex(ValueError,'entrypoint script is missing'):
            self.prepare(['bash','missing.sh'])
        with self.assertRaisesRegex(ValueError,'entrypoint script is missing'):
            self.prepare([sys.executable,'missing.py'])
        self.prepare([sys.executable,'-c','print("example")'])
        self.prepare(['bash','-c','echo missing.sh'])

    def test_empty_argument_and_large_binary_are_valid(self):
        self.prepare([sys.executable,'-c','pass',''])
        binary=self.root/'compiler'
        with binary.open('wb') as stream: stream.truncate(9*1024*1024)
        binary.chmod(0o755)
        value=self.prepare([str(binary),'--version'])
        self.assertNotIn(str(binary),value['files'])
        prep.validate(value)
        binary.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'queued executable changed'): prep.validate(value)

    def test_quoted_clean_guard_is_recognized(self):
        script=self.repo/'campaign.sh'
        script.write_text('[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || exit 2\n')
        self.git('add','.');self.git('commit','-qm','clean')
        value=self.prepare(['bash','campaign.sh'])
        (self.repo/'dirty').touch()
        with self.assertRaisesRegex(ValueError,'clean checkout'): prep.validate(value)

    def test_bad_syntax_and_gpu_cpu_command_refused(self):
        (self.repo/'broken.py').write_text('def :\n')
        with self.assertRaises(SyntaxError): self.prepare([sys.executable,'broken.py'])
        classifier = self.repo/'fleet.sh';classifier.write_text('echo gpu\n')
        spec=self.root/'spec.json';spec.write_text(json.dumps(dict(cpu_command=['nvidia-smi'])))
        with self.assertRaisesRegex(ValueError,'shows GPU use'): self.prepare(spec_path=spec,fleet=classifier)

    def test_failed_old_preparation_cannot_withdraw_a_new_edit(self):
        import fleet_pending
        self.prepare()
        path=next((self.directory/'preparations').glob('*.json'))
        record=dict(state='queued',prepare_manifest=str(path),command=[sys.executable,'input.py'],cwd=str(self.repo.resolve()),
                    ticket='1',pid=123,start='same',revision=1)
        (self.directory/'queue').write_text('1|fixture|1|2|test|boot|123\n')
        replacement=dict(record,revision=2)
        with mock.patch.object(fleet_pending,'read_record',side_effect=[record,replacement]), \
             mock.patch.object(prep,'validate',side_effect=ValueError('old input changed')):
            prep.check_pending(self.directory,'fixture',withdraw_failed=True)
        self.assertIn('fixture',(self.directory/'queue').read_text())
        with mock.patch.object(fleet_pending,'read_record',return_value=record), \
             mock.patch.object(prep,'validate',side_effect=ValueError('old input changed')):
            with self.assertRaisesRegex(ValueError,'old input changed'):
                prep.check_pending(self.directory,'fixture',withdraw_failed=True)
        self.assertEqual((self.directory/'queue').read_text(),'')

    def test_recovery_reclaim_does_not_reapply_payload_preparation(self):
        import fleet_pending
        record=dict(state='finishing',prepare_manifest='/missing/old-input.json')
        with mock.patch.object(fleet_pending,'read_record',return_value=record):
            prep.check_pending(self.directory,'fixture',withdraw_failed=True)

    def test_manifest_matches_edited_command_and_is_private(self):
        import fleet_pending
        value=self.prepare()
        path=next((self.directory/'preparations').glob('*.json'))
        self.assertEqual(path.stat().st_mode & 0o777,0o600)
        record=dict(state='queued',prepare_manifest=str(path),command=[sys.executable,'-c','pass'],cwd=str(self.repo))
        with mock.patch.object(fleet_pending,'read_record',return_value=record):
            with self.assertRaisesRegex(ValueError,'does not match'): prep.check_pending(self.directory,'fixture')


if __name__=='__main__': unittest.main()
