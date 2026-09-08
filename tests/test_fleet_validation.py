"""CPU-only fixtures for deployment evidence and fixed approved recovery."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_validation as validation


class ValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.store = self.root / 'store'
        self.count = self.root / 'count'
        self.git('init', '-q', '-b', 'main')
        self.git('config', 'user.name', 'CPU fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        files = {
            'launchers/deploy-overlays.sh': '#!/bin/bash\ntrue\n',
            'launchers/compose-overlays.sh': '#!/bin/bash\ntrue\n',
            'launchers/audit-runtime-guards.py': 'print("audit self-test passed")\n',
            'profiles/glm53.env': '# fixture profile\n',
            'tests/test_logic.py': '# fixture logic\n',
            'tests/test_glm53_overlay_sync.py': '# fixture sync\n',
            'bench/cpu_checks.py': '''import json, os, pathlib, sys, time
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not any(k.startswith(('VLLM_', 'FLEET_', 'ONEPASS_')) for k in os.environ)
assert 'PYTHONPATH' not in os.environ and 'BASH_ENV' not in os.environ
with open(COUNT, 'a') as output: output.write('run\\n')
time.sleep(.08)
pathlib.Path(sys.argv[sys.argv.index('--out')+1]).write_text(json.dumps(dict(passed=True, coverage_complete=True, tests_run=7)))
'''.replace('COUNT', repr(str(self.count))),
            '.gitignore': '__pycache__/\n',
        }
        for name, text in files.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')
        # Keep tests independent of installed packages and simulate exact-runtime
        # identity while exercising real clean-checkout, process and file logic.
        def evidence(repo, spec, env):
            tree = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-rz', 'HEAD'])
            payload = tree + json.dumps([spec, env], sort_keys=True).encode()
            return dict(key=hashlib.sha256(payload).hexdigest(), scope='fixture-full-tree', files=7)
        self.evidence = evidence
        self.patch = mock.patch.object(validation.cpu_evidence, 'identity', side_effect=evidence)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.chat_patch = mock.patch.object(validation, 'chat_inputs', return_value=None)
        self.chat_patch.start()
        self.addCleanup(self.chat_patch.stop)
        self.env_patch = mock.patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True, stderr=subprocess.PIPE).strip()

    def validate(self, **kwargs):
        return validation.validate(self.repo, self.store, **kwargs)

    def runs(self):
        return self.count.read_text().count('run\n') if self.count.exists() else 0

    def change(self, name, suffix='\n# change\n'):
        with (self.repo / name).open('a') as output:
            output.write(suffix)
        self.git('add', '.')
        self.git('commit', '-qm', 'change')

    def hold(self):
        fleet = self.root / 'fleet'
        fleet.mkdir(exist_ok=True)
        (fleet / 'holder').write_text('fixture|123|0|boot\n')
        os.environ.update(FLEET_DIR=str(fleet), FLEET_SESSION='fixture')

    def test_identical_source_and_sanitized_environment_reuse(self):
        first = self.validate()
        os.environ.update(VLLM_EXPERIMENT='1', ONEPASS_UNKNOWN='2', PYTHONPATH='/nonexistent',
                          PYTHONHOME='/nonexistent', PYTHONUSERBASE='/nonexistent',
                          PYTHONNOUSERSITE='1', BASH_ENV='/nonexistent', FLEET_POLL='different')
        second = self.validate()
        self.assertFalse(first['reused'])
        self.assertTrue(second['reused'])
        self.assertEqual(first['receipt'], second['receipt'])
        self.assertEqual(self.runs(), 1)

    def test_sanitized_runtime_keeps_user_site_dependencies_and_binds_customization(self):
        # A deliberately isolated base interpreter tests user-site policy;
        # production venvs may correctly have ENABLE_USER_SITE=False.
        base_python = getattr(sys, '_base_executable', sys.executable)
        user = mock.Mock(pw_dir=str(self.root / 'user-home'), pw_name='fixture')
        with mock.patch.object(validation.pwd, 'getpwuid', return_value=user):
            env = validation.environment()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONNOUSERSITE', 'BASH_ENV'):
            self.assertNotIn(key, env)
        directory = Path(subprocess.check_output([base_python, '-c',
            'import site; print(site.getusersitepackages())'], env=env, text=True).strip())
        directory.mkdir(parents=True)
        (directory / 'fixture_cpu_dependency.py').write_text('VALUE = 71\n')
        customization = directory / 'usercustomize.py'
        customization.write_text('FIXTURE = "first"\n')
        (directory / 'fixture.pth').write_text('# fixture startup configuration\n')
        result = subprocess.check_output([base_python, '-c',
            'import fixture_cpu_dependency; print(fixture_cpu_dependency.VALUE)'], env=env, text=True)
        self.assertEqual(result.strip(), '71')
        with mock.patch.object(validation.sys, 'executable', base_python):
            first = validation.python_startup_inputs(self.repo, env)
        self.assertTrue(first['user_site_enabled'])
        self.assertIn(str(customization.resolve()), first['files'])
        self.assertIn(str((directory / 'fixture.pth').resolve()), first['files'])
        customization.write_text('FIXTURE = "second"\n')
        with mock.patch.object(validation.sys, 'executable', base_python):
            second = validation.python_startup_inputs(self.repo, env)
        self.assertNotEqual(first, second)

    def test_changed_tests_never_silently_reuse(self):
        first = self.validate()
        self.change('tests/test_logic.py')
        second = self.validate()
        self.assertNotEqual(first['receipt'], second['receipt'])
        self.assertEqual(self.runs(), 2)
        (self.repo / 'tests/test_logic.py').write_text('dirty test')
        with self.assertRaisesRegex(ValueError, 'clean source'):
            self.validate()

    def test_merge_metadata_preserves_preprimed_same_content_receipt(self):
        first = self.validate()
        before = self.git('rev-parse', 'HEAD^{tree}')
        self.git('branch', 'side')
        self.git('commit', '--allow-empty', '-qm', 'main metadata')
        self.git('switch', '--quiet', 'side')
        self.git('commit', '--allow-empty', '-qm', 'side metadata')
        self.git('switch', '--quiet', 'main')
        self.git('merge', '--no-ff', '--quiet', '-m', 'merge', 'side')
        self.assertEqual(self.git('rev-parse', 'HEAD^{tree}'), before)
        self.assertNotEqual(self.git('rev-parse', 'HEAD'), first['source'])
        self.hold()
        self.assertTrue(self.validate(require_receipt=first['receipt'])['reused'])
        self.assertEqual(self.runs(), 1)

    def test_runtime_change_rejects_old_required_receipt(self):
        first = self.validate()
        def changed(repo, spec, env):
            value = self.evidence(repo, spec, env)
            value['key'] = hashlib.sha256((value['key'] + 'changed dependencies').encode()).hexdigest()
            return value
        with mock.patch.object(validation.cpu_evidence, 'identity', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'exact source/environment'):
                self.validate(require_receipt=first['receipt'])
        self.assertEqual(self.runs(), 1)

    def test_hold_miss_fails_without_running_and_hit_reuses(self):
        self.hold()
        with self.assertRaisesRegex(ValueError, 'before GPU reservation'):
            self.validate()
        self.assertEqual(self.runs(), 0)
        os.environ.pop('FLEET_SESSION')
        first = self.validate()
        self.hold()
        self.assertTrue(self.validate(require_receipt=first['receipt'])['reused'])
        self.assertEqual(self.runs(), 1)

    def test_failed_and_incomplete_coverage_never_create_receipts(self):
        script = self.repo / 'bench/cpu_checks.py'
        script.write_text(script.read_text().replace('passed=True', 'passed=False'))
        self.git('add', '.'); self.git('commit', '-qm', 'incomplete')
        with self.assertRaisesRegex(ValueError, 'coverage is incomplete'):
            self.validate()
        self.assertFalse([p for p in (self.store / 'receipts').glob('*.json') if '.report.' not in p.name])
        script.write_text('raise SystemExit(7)\n')
        self.git('add', '.'); self.git('commit', '-qm', 'failed')
        with self.assertRaisesRegex(ValueError, 'failed with code 7'):
            self.validate()
        self.assertFalse([p for p in (self.store / 'receipts').glob('*.json') if '.report.' not in p.name])

    def test_missing_test_is_not_empty_success(self):
        (self.repo / 'tests/test_logic.py').unlink()
        self.git('add', '.'); self.git('commit', '-qm', 'remove')
        with self.assertRaisesRegex(ValueError, 'prerequisite missing'):
            self.validate()
        self.assertEqual(self.runs(), 0)

    def test_concurrent_preparation_executes_fixed_gate_once(self):
        context = multiprocessing.get_context('fork')
        queue = context.Queue()
        def worker():
            try:
                queue.put(self.validate()['reused'])
            except Exception as exc:
                queue.put(str(exc))
        workers = [context.Process(target=worker) for _ in range(3)]
        for worker in workers: worker.start()
        for worker in workers:
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sorted(queue.get(timeout=2) for _ in workers), [False, True, True])
        self.assertEqual(self.runs(), 1)

    def test_tampered_artifact_cannot_be_consumed_during_hold(self):
        first = self.validate()
        path = Path(first['receipt'])
        report = path.with_name(path.stem + '.report.json')
        report.write_text(report.read_text().replace('7', '700'))
        self.hold()
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.validate(require_receipt=first['receipt'])
        self.assertEqual(self.runs(), 1)

    def recovery(self):
        remote = self.root / 'origin.git'
        subprocess.check_call(['git', 'clone', '--quiet', '--bare', str(self.repo), str(remote)])
        self.git('remote', 'add', 'origin', str(remote))
        return validation.prepare_recovery(self.repo, self.store)

    def test_recovery_keeps_validated_approved_source_when_main_advances(self):
        first = self.recovery()
        original = first['source']
        self.change('tests/test_logic.py')
        self.git('push', '--quiet', 'origin', 'main')
        self.git('fetch', '--quiet', 'origin', 'main')
        self.hold()
        verified = validation.verify_recovery(first['receipt'])
        self.assertEqual(verified['source'], original)
        self.assertEqual(validation.source(Path(verified['repo'])), original)
        self.assertEqual(self.runs(), 1)
        with self.assertRaisesRegex(ValueError, 'before GPU reservation'):
            validation.prepare_recovery(self.repo, self.store)

    def test_recovery_descriptor_change_and_checkout_change_fail_closed(self):
        first = self.recovery()
        path = Path(first['receipt'])
        original = path.read_text()
        path.write_text(original.replace(first['source'], '0' * 40))
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            validation.verify_recovery(path)
        path.write_text(original)
        (Path(first['repo']) / 'tests/test_logic.py').write_text('dirty')
        with self.assertRaisesRegex(ValueError, 'clean source'):
            validation.verify_recovery(path)

    def test_restore_shell_uses_receipt_without_fetch_and_isolates_candidate_environment(self):
        self.hold()
        bindir = self.root / 'bin'
        bindir.mkdir()
        observed = self.root / 'restore-actions'
        shim = bindir / 'python3'
        shim.write_text('#!' + sys.executable + '\n' + '''import os, pathlib, shlex, subprocess, sys
assert not any(k.startswith(('VLLM_', 'ONEPASS_', 'MK_', 'STARTUP_CACHE_', 'PROFILE')) for k in os.environ)
assert 'IMAGE' not in os.environ and 'MODEL_HOST_PATH' not in os.environ
with open(OBSERVED, 'a') as output: output.write(sys.argv[1] + '\\n')
if sys.argv[1].endswith('fleet_validation.py'):
 assert sys.argv[2:] == ['verify-recovery', '--receipt', '/fixture/receipt', '--format', 'shell']
 print('export FLEET_RECOVERY_REPO=' + shlex.quote(RECOVERY_PATH))
 print('export FLEET_RECOVERY_RECEIPT=/fixture/receipt')
elif sys.argv[1].endswith('fleet_entry.py'):
 raise SystemExit(0)
else:
 raise SystemExit('unexpected restore process')
'''.replace('OBSERVED', repr(str(observed))).replace('RECOVERY_PATH', repr(str(self.repo))))
        shim.chmod(0o755)
        git = bindir / 'git'
        git.write_text('#!/bin/sh\necho unexpected-git >&2\nexit 77\n')
        git.chmod(0o755)
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.defpath,
                   FLEET_RUNNER_REPO=str(Path(validation.__file__).resolve().parents[1]),
                   FLEET_VALIDATION_REQUIRED='1', FLEET_RECOVERY_RECEIPT='/fixture/receipt',
                   PROFILE='candidate', PROFILE_IMAGE='candidate-image', IMAGE='candidate',
                   MODEL_HOST_PATH='/candidate/model', VLLM_BAD='1', ONEPASS_BAD='1', MK_BAD='1')
        result = subprocess.run(['bash', str(Path(validation.__file__).with_name('fleet_restore.sh'))],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('no restore boot', result.stdout)
        self.assertEqual(len(observed.read_text().splitlines()), 2)

    def test_new_restore_requires_prepared_receipt(self):
        self.hold()
        env = dict(os.environ, FLEET_VALIDATION_REQUIRED='1')
        result = subprocess.run(['bash', str(Path(validation.__file__).with_name('fleet_restore.sh'))],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('prevalidated approved recovery receipt is missing', result.stdout)

    def test_all_launcher_overrides_are_cleared_before_health_deploy_and_boot(self):
        import re
        import shlex
        self.hold()
        runner = Path(validation.__file__).resolve().parents[1]
        lines = (runner / 'launchers/start-glm53-nvfp4-tp4.sh').read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith('ct_load_profile '))
        arguments = []
        for line in lines[start + 1:]:
            arguments.extend(re.findall(r'\b[A-Z][A-Z0-9_]*\b', line))
            if not line.endswith('\\'):
                break
        overrides = set(arguments) | {'OVERLAY_DIR', 'DRAFT_HOST_PATH', 'DRY_RUN', 'SKIP_PREFLIGHT',
            'PREBUILD', 'PIECEWISE', 'KV_TOKENS', 'KV_HYBRID_BLOCKS', 'MM_ENCODER_ATTN',
            'MM_ENCODER_TP_MODE', 'SKIP_MM_PROFILING', 'SPEC_K_FORCE', 'GRAPH_DEBUG', 'MOE_CUTOVER',
            'CG_MEM_PROFILE', 'CG_UTIL_DELTA', 'GLOO_IFNAME', 'TORCH_LOGS', 'TORCH_CPP_LOG_LEVEL',
            'TORCH_DISTRIBUTED_DEBUG', 'NCCL_ASYNC_ERR', 'FLEET_REHEARSE', 'DEPLOY_PRESERVE_IDENTICAL'}
        self.assertIn('CHAT_TEMPLATE', overrides)
        self.assertIn('DRAFT_TP', overrides)
        observed = self.root / 'restore-phases'
        check = self.root / 'check-environment.py'
        check.write_text('''import os, sys
for key in OVERRIDES:
 if key == 'PREFILL_WARMUP' and sys.argv[1] == 'boot':
  assert os.environ.get(key) == '1'
 else:
  assert key not in os.environ, (sys.argv[1], key, os.environ.get(key))
assert os.environ['FLEET_SESSION'] == 'fixture'
assert os.environ['FLEET_RECOVERY_RECEIPT'] == '/fixture/receipt'
if sys.argv[1] == 'boot':
 assert os.environ['REPO'] == RECOVERY_PATH
 assert os.environ['GLM53_API_HOST'] == '0.0.0.0'
 assert os.environ['GLM53_API_PORT'] == '8000'
 assert os.environ['SKIP_BOOT'] == '0' and os.environ['LEGS'] == 'none'
with open(OBSERVED, 'a') as output: output.write(sys.argv[1] + '\\n')
'''.replace('OVERRIDES', repr(sorted(overrides))).replace('OBSERVED', repr(str(observed)))
                         .replace('RECOVERY_PATH', repr(str(self.repo))))
        bindir = self.root / 'restore-bin'
        bindir.mkdir()
        shim = bindir / 'python3'
        shim.write_text('#!' + sys.executable + '\n' + '''import shlex, subprocess, sys
if sys.argv[1].endswith('fleet_validation.py'):
 print('export FLEET_RECOVERY_REPO=' + shlex.quote(RECOVERY_PATH))
elif sys.argv[1].endswith('fleet_entry.py'):
 subprocess.run([sys.executable, CHECKER, 'health'], check=True)
 raise SystemExit(1)
elif sys.argv[1] == '-':
 raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]], stdin=sys.stdin))
else:
 raise SystemExit('unexpected restore process')
'''.replace('RECOVERY_PATH', repr(str(self.repo))).replace('CHECKER', repr(str(check))))
        shim.chmod(0o755)
        for name, phase in (('launchers/deploy-overlays.sh', 'deploy'), ('bench/ab-lever.sh', 'boot')):
            (self.repo / name).write_text('#!/bin/bash\nexec ' + shlex.join([sys.executable, str(check), phase]) + '\n')
        env = dict(os.environ, **{key: 'candidate' for key in overrides})
        env.update(PATH=str(bindir) + os.pathsep + os.defpath, FLEET_RUNNER_REPO=str(runner),
                   FLEET_RECOVERY_RECEIPT='/fixture/receipt', FLEET_VALIDATION_REQUIRED='1',
                   CUSTOM_OPS_AXIS='')
        result = subprocess.run(['bash', str(runner / 'bench/fleet_restore.sh')],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(observed.read_text().splitlines(), ['health', 'deploy', 'boot'])

    def enable_chat_fixture(self):
        self.chat_patch.stop()
        model = self.root / 'tokenizer'
        model.mkdir()
        (model / 'tokenizer_config.json').write_text('{}')
        (model / 'tokenizer.json').write_text('{"fixture":"v1"}')
        image = self.root / 'image-id'
        image.write_text('sha256:' + 'a' * 64)
        binary = self.root / 'docker-bin'
        binary.mkdir()
        docker = binary / 'docker'
        docker.write_text('#!' + sys.executable + '\n' + '''import pathlib, sys
assert sys.argv[1:5] == ['image', 'inspect', '--format', '{{.Id}}']
print(pathlib.Path(IMAGE_FILE).read_text())
'''.replace('IMAGE_FILE', repr(str(image))))
        docker.chmod(0o755)
        os.environ['PATH'] = str(binary) + os.pathsep + os.defpath
        gate = self.repo / 'launchers/check-glm53-chat.sh'
        args = self.root / 'chat-args'
        import shlex
        gate.write_text('#!/bin/bash\nprintf "%s|%s\\n" "$1" "$2" >> ' + shlex.quote(str(args)) + '\n')
        profile = self.repo / 'profiles/glm53.env'
        profile.write_text('PROFILE_IMAGE="fixture:tag"\nPROFILE_MODEL_PATH=' + shlex.quote(str(model)) + '\n')
        self.git('add', '.'); self.git('commit', '-qm', 'chat fixture')
        audit = mock.patch.object(validation, 'CHAT_SOURCE_AUDIT',
                                  {'launchers/check-glm53-chat.sh': validation.cpu_evidence.sha(gate)})
        audit.start()
        self.addCleanup(audit.stop)
        return model, image, args

    def test_chat_image_change_invalidates_and_executes_immutable_image(self):
        model, image, args = self.enable_chat_fixture()
        first = self.validate()
        self.assertEqual(args.read_text().strip(), str(model.resolve()) + '|sha256:' + 'a' * 64)
        image.write_text('sha256:' + 'b' * 64)
        with self.assertRaisesRegex(ValueError, 'exact source/environment'):
            self.validate(require_receipt=first['receipt'])
        second = self.validate()
        self.assertNotEqual(first['receipt'], second['receipt'])
        self.assertEqual(self.runs(), 2)
        self.assertTrue(args.read_text().splitlines()[-1].endswith('sha256:' + 'b' * 64))

    def test_chat_tokenizer_change_blocks_hold_without_rerunning(self):
        model, _, _ = self.enable_chat_fixture()
        self.validate()
        (model / 'tokenizer.json').write_text('{"fixture":"v2"}')
        self.hold()
        with self.assertRaisesRegex(ValueError, 'before GPU reservation'):
            self.validate()
        self.assertEqual(self.runs(), 1)

    def test_chat_default_recovery_ignores_candidate_paths_and_unknown_inputs_fail(self):
        model, _, _ = self.enable_chat_fixture()
        first = self.validate(use_profile_defaults=True)
        os.environ.update(IMAGE='candidate-image', MODEL_HOST_PATH='/missing/candidate-model')
        self.assertTrue(self.validate(use_profile_defaults=True, require_receipt=first['receipt'])['reused'])
        with self.assertRaisesRegex(ValueError, 'requires local tokenizer'):
            self.validate()
        (model / 'tokenizer_config.json').write_text('{"auto_map":{"AutoTokenizer":"custom.Tokenizer"}}')
        with self.assertRaisesRegex(ValueError, 'custom tokenizer loader'):
            self.validate(use_profile_defaults=True)

    def test_optional_fixed_checkpoint_is_bound_independently_of_selected_model(self):
        model, _, _ = self.enable_chat_fixture()
        fixed = self.root / 'fixed-production-config.json'
        with mock.patch.object(validation, 'LOGIC_CHECKPOINT_CONFIG', fixed):
            first = self.validate(model=str(model))
            self.assertEqual(first['context']['fixed_logic_inputs'][str(fixed)],
                             dict(exists=False, sha256=None))
            fixed.write_text('{"layer_types":["indexer"]}')
            with self.assertRaisesRegex(ValueError, 'exact source/environment'):
                self.validate(model=str(model), require_receipt=first['receipt'])
            present = self.validate(model=str(model))
            fixed.write_text('{"layer_types":[]}')
            with self.assertRaisesRegex(ValueError, 'exact source/environment'):
                self.validate(model=str(model), require_receipt=present['receipt'])
            fixed.unlink()
            self.assertTrue(self.validate(model=str(model), require_receipt=first['receipt'])['reused'])
        self.assertEqual(self.runs(), 2)

    def test_deploy_adopts_source_helper_when_old_runner_has_none(self):
        deploy = Path(validation.__file__).resolve().parents[1] / 'launchers/deploy-overlays.sh'
        text = deploy.read_text()
        setup = text[text.index('REPO=$('):text.index('require_deployable_checkout()')]
        old = self.root / 'old-pinned-runner'
        old.mkdir()
        env = dict(os.environ, FLEET_RUNNER_REPO=str(old))
        result = subprocess.run(['bash', '-c', setup + '\nprintf "%s" "$VALIDATOR"', str(deploy)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(Path(validation.__file__).resolve()))
        (old / 'bench').mkdir()
        for name in ('fleet_validation.py', 'fleet_source.py'):
            (old / 'bench' / name).write_text('# fixture helper\n')
        result = subprocess.run(['bash', '-c', setup + '\nprintf "%s" "$VALIDATOR"', str(deploy)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(old / 'bench/fleet_validation.py'))

    def configured_cli(self, *args):
        entry = self.root / 'validation-cli.py'
        entry.write_text('import sys\nsys.path.insert(0, ' + repr(str(Path(validation.__file__).parent)) + ')\n'
                         'import fleet_validation as v\n'
                         'v.validate = lambda *a, **k: {"receipt": sys.executable}\n'
                         'v.verify_recovery = lambda *a, **k: {"receipt": sys.executable}\n'
                         'raise SystemExit(v.main())\n')
        return subprocess.run([sys.executable, str(entry), *args], capture_output=True,
                              text=True, env=dict(os.environ))

    def test_cli_selects_configured_interpreter_and_recovery_descriptor_store(self):
        self.store.mkdir(mode=0o700)
        venv = self.root.resolve() / 'fixture-venv'
        subprocess.check_call([sys.executable, '-m', 'venv', '--without-pip', str(venv)])
        selected = venv / 'bin/python'
        (self.store / 'python').write_text(str(selected) + '\n')
        (self.store / 'python').chmod(0o600)
        args = ['validate', '--repo', str(self.repo), '--store', str(self.store), '--format', 'receipt']
        result = self.configured_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(selected))
        self.assertNotIn('FLEET_VALIDATION_BOOTSTRAP', os.environ)
        value = dict(version=validation.VERSION, repo=str(self.repo), source=self.git('rev-parse', 'HEAD'),
                     validation_receipt='/fixture/evidence', store=str(self.store.resolve()))
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        receipts = self.store / 'recovery-receipts'
        receipts.mkdir()
        receipt = receipts / (digest + '.json')
        receipt.write_text(json.dumps(value))
        other = self.root / 'wrong-store'
        other.mkdir()
        (other / 'python').write_text('/missing/interpreter\n')
        (other / 'python').chmod(0o600)
        result = self.configured_cli('verify-recovery', '--receipt', str(receipt.resolve()),
                                     '--store', str(other), '--format', 'receipt')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(selected))

    def test_cli_rejects_invalid_missing_and_symlink_interpreter_configuration(self):
        self.store.mkdir(mode=0o700)
        configuration = self.store / 'python'
        args = ['validate', '--repo', str(self.repo), '--store', str(self.store)]
        for value, expected in [('python3\n', 'absolute interpreter path'),
                                ('/missing/python\n', 'missing or not executable')]:
            configuration.write_text(value)
            configuration.chmod(0o600)
            result = self.configured_cli(*args)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(expected, result.stderr)
        configuration.unlink()
        target = self.root / 'config-target'
        target.write_text(sys.executable + '\n')
        target.chmod(0o600)
        configuration.symlink_to(target)
        result = self.configured_cli(*args)
        self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == '__main__':
    unittest.main()
