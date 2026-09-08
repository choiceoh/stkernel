"""Deployment identity fixtures use no fleet, Docker, SSH, or model work."""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import onepass_deploy as deploy


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()
        (self.repo / 'profiles').mkdir()
        self.overlay = self.repo / 'mounted'
        self.overlay.mkdir()
        (self.repo / 'profiles/glm53.env').write_text('PROFILE_OVERLAY_DIR=' + str(self.overlay) + '\nMODULES=fixture\nTARGET_PREFIX=/runtime/\n')
        (self.repo / 'overlay/modules/fixture').mkdir(parents=True)
        (self.repo / 'overlay/modules/fixture/kernel.py').write_text('fixture\n')
        (self.repo / 'overlay/modules/fixture/manifest.tsv').write_text('kernel.py\tkernel.py\tabsent\n')
        self.revision = 'a' * 40
        self.env = dict(FLEET_DIR=str(self.repo / 'fleet'), FLEET_SESSION='fixture',
                        MK_OVERLAY_STAMP=str(self.repo / 'stamp'))
        patcher = patch.object(deploy, 'source_revision', return_value=self.revision)
        self.source = patcher.start()
        self.addCleanup(patcher.stop)

    def manifest(self, revision=None):
        data = ('# source_commit=' + (revision or self.revision) + '\n'
                'kernel.py\t/runtime/kernel.py\tabsent\n')
        (self.overlay / 'kernel.py').write_text('fixture\n')
        (self.overlay / 'manifest.tsv').write_text(data)
        return hashlib.sha256(data.encode()).hexdigest()

    def test_matching_source_skips_deploy_and_all_external_runtime_checks(self):
        self.manifest()
        with patch.object(deploy.subprocess, 'run') as run, patch('fleet_idle.boot_authorize') as owner:
            result = deploy.ensure(self.repo, environment=self.env)
        self.assertTrue(result['reused'])
        self.assertEqual(result['source_commit'], self.revision)
        run.assert_not_called()
        owner.assert_not_called()

    def test_different_source_deploys_exact_checkout_once_then_reuses(self):
        self.manifest('b' * 40)
        with patch.object(deploy.subprocess, 'run', side_effect=lambda *a, **k: self.manifest()) as run, \
                patch('fleet_idle.boot_authorize') as owner:
            first = deploy.ensure(self.repo, environment=self.env)
            second = deploy.ensure(self.repo, environment=self.env)
        self.assertFalse(first['reused'])
        self.assertTrue(second['reused'])
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ['bash', str(self.repo / 'launchers/deploy-overlays.sh'), 'glm53'])
        self.assertEqual(run.call_args.kwargs['cwd'], self.repo)
        self.assertEqual(run.call_args.kwargs['env']['PROFILE'], 'glm53')
        owner.assert_called_once_with(self.env['FLEET_DIR'], 'fixture')

    def test_changed_overlay_with_same_commit_is_redeployed_before_boot(self):
        self.manifest()
        (self.overlay / 'kernel.py').write_text('stale kernel bytes\n')
        with patch.object(deploy.subprocess, 'run', side_effect=lambda *a, **k: self.manifest()) as run, \
                patch('fleet_idle.boot_authorize'):
            self.assertFalse(deploy.ensure(self.repo, environment=self.env)['reused'])
        run.assert_called_once()

    def test_live_changed_overlay_with_same_commit_fails_without_deployment(self):
        stamp = self.manifest()
        Path(self.env['MK_OVERLAY_STAMP']).write_text(stamp)
        (self.overlay / 'kernel.py').write_text('stale kernel bytes\n')
        with patch.object(deploy.subprocess, 'run') as run, \
                patch('fleet_entry.inspect') as inspect, self.assertRaisesRegex(ValueError, 'differs'):
            deploy.ensure(self.repo, live=True, environment=self.env)
        run.assert_not_called()
        inspect.assert_not_called()

    def test_missing_required_row_fails_live_and_redeploys_complete_manifest(self):
        self.manifest()
        module = self.repo / 'overlay/modules/fixture'
        (module / 'required.py').write_text('required fixture\n')
        (module / 'manifest.tsv').write_text(
            'kernel.py\tkernel.py\tabsent\nrequired.py\trequired.py\tabsent\n')
        with patch.object(deploy.subprocess, 'run') as run, self.assertRaisesRegex(ValueError, 'differs'):
            deploy.ensure(self.repo, live=True, environment=self.env)
        run.assert_not_called()
        def publish(*args, **kwargs):
            self.manifest()
            (self.overlay / 'required.py').write_text('required fixture\n')
            with (self.overlay / 'manifest.tsv').open('a') as stream:
                stream.write('required.py\t/runtime/required.py\tabsent\n')
        with patch.object(deploy.subprocess, 'run', side_effect=publish) as run, \
                patch('fleet_idle.boot_authorize'):
            self.assertFalse(deploy.ensure(self.repo, environment=self.env)['reused'])
        run.assert_called_once()

    def test_wrong_target_or_base_contract_fails_even_with_matching_source_bytes(self):
        for row in ('kernel.py\t/wrong/kernel.py\tabsent\n',
                    'kernel.py\t/runtime/kernel.py\t' + 'a' * 64 + '\n'):
            with self.subTest(row=row):
                self.manifest()
                (self.overlay / 'manifest.tsv').write_text('# source_commit=' + self.revision + '\n' + row)
                with patch.object(deploy.subprocess, 'run') as run, self.assertRaisesRegex(ValueError, 'differs'):
                    deploy.ensure(self.repo, live=True, environment=self.env)
                run.assert_not_called()

    def test_external_symlink_cannot_replace_committed_module_source(self):
        self.manifest()
        source = self.repo / 'overlay/modules/fixture/kernel.py'
        source.unlink()
        source.symlink_to(self.overlay / 'kernel.py')
        with patch.object(deploy.subprocess, 'run') as run, self.assertRaisesRegex(ValueError, 'differs'):
            deploy.ensure(self.repo, live=True, environment=self.env)
        run.assert_not_called()

    def test_failed_publication_never_marks_wrong_source_reusable(self):
        self.manifest('b' * 40)
        with patch.object(deploy.subprocess, 'run'), patch('fleet_idle.boot_authorize'), \
                self.assertRaisesRegex(ValueError, 'did not attest'):
            deploy.ensure(self.repo, environment=self.env)

    def test_missing_owner_cannot_publish_before_boot(self):
        with patch.object(deploy.subprocess, 'run') as run, \
                patch('fleet_idle.boot_authorize', side_effect=ValueError('not owned')), \
                self.assertRaisesRegex(ValueError, 'not owned'):
            deploy.ensure(self.repo, environment=self.env)
        run.assert_not_called()

    def test_live_mismatch_fails_without_deployment(self):
        self.manifest('b' * 40)
        with patch.object(deploy.subprocess, 'run') as run, self.assertRaisesRegex(ValueError, 'differs'):
            deploy.ensure(self.repo, live=True, environment=self.env)
        run.assert_not_called()

    def live_container(self):
        return {'Id': 'fixture-boot',
                'State': {'Running': True, 'StartedAt': datetime.now(timezone.utc).isoformat()},
                'Mounts': [{'Type': 'bind', 'Source': str(self.overlay / 'kernel.py'),
                            'Destination': '/runtime/kernel.py'}]}

    def test_live_requires_matching_boot_stamp_and_running_server(self):
        stamp = self.manifest()
        Path(self.env['MK_OVERLAY_STAMP']).write_text('old-boot\n')
        with patch('fleet_entry.inspect') as inspect, self.assertRaisesRegex(ValueError, 'not been booted'):
            deploy.ensure(self.repo, live=True, environment=self.env)
        inspect.assert_not_called()
        Path(self.env['MK_OVERLAY_STAMP']).write_text(stamp)
        with patch('fleet_entry.inspect', return_value=self.live_container()), \
                patch.object(deploy.subprocess, 'run') as run:
            self.assertTrue(deploy.ensure(self.repo, live=True, environment=self.env)['reused'])
        run.assert_not_called()
        with patch('fleet_entry.inspect', return_value={'State': {'Running': False}}), \
                self.assertRaisesRegex(ValueError, 'running server'):
            deploy.ensure(self.repo, live=True, environment=self.env)

    def test_live_rejects_republished_source_even_if_disk_stamp_matches(self):
        stamp = self.manifest()
        Path(self.env['MK_OVERLAY_STAMP']).write_text(stamp)
        container = self.live_container()
        # A deploy wrote the same bound path after this engine imported it.
        os.utime(self.overlay / 'kernel.py', (4102444800, 4102444800))
        with patch('fleet_entry.inspect', return_value=container), \
                patch.object(deploy.subprocess, 'run') as run, \
                self.assertRaisesRegex(ValueError, 'changed after'):
            deploy.ensure(self.repo, live=True, environment=self.env)
        run.assert_not_called()

    def test_live_rejects_wrong_container_mount_before_requests(self):
        stamp = self.manifest()
        Path(self.env['MK_OVERLAY_STAMP']).write_text(stamp)
        container = self.live_container()
        container['Mounts'][0]['Source'] = '/another/kernel.py'
        with patch('fleet_entry.inspect', return_value=container), \
                self.assertRaisesRegex(ValueError, 'does not bind'):
            deploy.ensure(self.repo, live=True, environment=self.env)

    def test_source_change_during_deploy_fails_before_measurement(self):
        self.source.side_effect = [self.revision, 'b' * 40]
        with patch.object(deploy.subprocess, 'run', side_effect=lambda *a, **k: self.manifest()), \
                patch('fleet_idle.boot_authorize'), self.assertRaisesRegex(ValueError, 'source changed'):
            deploy.ensure(self.repo, environment=self.env)

    def test_override_cannot_boot_a_different_overlay_directory(self):
        self.manifest()
        with patch.object(deploy.subprocess, 'run') as run, self.assertRaisesRegex(ValueError, 'differs'):
            deploy.ensure(self.repo, environment=dict(self.env, OVERLAY_DIR='/different'))
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
