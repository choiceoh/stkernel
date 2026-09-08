#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pass deployment CPU gates before queueing, and verify their exact evidence.

Only the fixed deployment gates are cacheable. A GPU holder consumes existing
receipts; it never starts a missing CPU gate. Recovery uses an approved main
commit pinned before the reservation, even when main advances while waiting.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

import cpu_evidence

VERSION = 1
TIMEOUT = 600
LOGIC_CHECKPOINT_CONFIG = Path('/home/choiceoh/models/glm53-redhat-nvfp4/config.json')
TOKENIZER_FILES = ('tokenizer_config.json', 'config.json', 'tokenizer.json',
                   'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                   'merges.txt', 'vocab.txt', 'tokenizer.model', 'spiece.model',
                   'chat_template.jinja')
# These release checks consume only the local fast-tokenizer files below.
# Review their dependency set before accepting a changed check for reuse.
CHAT_SOURCE_AUDIT = {
    'launchers/check-glm53-chat.sh': '240af355f4969b299153abd99b48049e2aa38d2fd7cc2431c2addc21c50373c4',
    'probes/glm53_chat_contract.py': '5ecce35790d27085778963645d70c9720f992342925e3c102dc323d93c74966a',
    'tests/test_glm53_chat.py': '6bac0f1dde3ee7c62adc5a8e2566325cae38405de10a13ea32318f47a38e2510',
    'tests/test_glm53_tool_acceptance.py': '33f775aa8a893303b4df08a8414f4dab35a3ae3916f3a2217d1dcebab77b2d9c',
}


def run(argv, repo, *, env=None, timeout=30):
    result = subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise ValueError('validation failed: ' + shlex.join(argv[:5]) + '\n' + (result.stderr or result.stdout)[-3000:])
    return result.stdout.strip()


def environment():
    """CPU gates cannot inherit candidate knobs, shell startup files or Python hooks."""
    user = pwd.getpwuid(os.getuid())
    return dict(PATH=os.environ.get('PATH', os.defpath), HOME=user.pw_dir,
                USER=user.pw_name, LOGNAME=user.pw_name, LANG='C', LC_ALL='C',
                TMPDIR=tempfile.gettempdir(), PYTHONNOUSERSITE='1',
                PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')


def default_store():
    fleet = os.environ.get('FLEET_DIR')
    return Path(os.environ.get('FLEET_VALIDATION_STORE') or
                (str(Path(fleet) / 'validation') if fleet else str(Path.home() / '.cache/stkernel/fleet-validation')))


def private_directory(path):
    path = Path(path).absolute()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError('validation store must be an owned real directory: ' + str(path))
    path.chmod(0o700)
    return path.resolve()


@contextmanager
def lock(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def held():
    directory, session = os.environ.get('FLEET_DIR'), os.environ.get('FLEET_SESSION')
    if not directory or not session:
        return False
    try:
        return (Path(directory) / 'holder').read_text().split('|')[0] == session
    except FileNotFoundError:
        return False


def source(repo):
    if run(['git', 'status', '--porcelain', '--untracked-files=normal'], repo):
        raise ValueError('CPU validation requires a clean source checkout: ' + str(repo))
    return run(['git', 'rev-parse', '--verify', 'HEAD'], repo)


def profile_value(repo, profile, name):
    text = (repo / 'profiles' / (profile + '.env')).read_text()
    matches = re.findall(r'^' + re.escape(name) + r'=(.*)$', text, re.M)
    if len(matches) != 1:
        raise ValueError('profile must declare one literal ' + name)
    values = shlex.split(matches[0], comments=True)
    if len(values) != 1 or any(c in values[0] for c in '$`\n\r'):
        raise ValueError('profile ' + name + ' must be a literal value')
    return values[0]


def fixed_logic_inputs():
    # test_glm53_index_cache_layer_rule reads this optional deployed checkpoint
    # regardless of MODEL_HOST_PATH. Appearance, removal and edits all matter.
    path = LOGIC_CHECKPOINT_CONFIG
    exists = path.exists()
    if exists and not path.is_file():
        raise ValueError('optional logic checkpoint config is not a file: ' + str(path))
    return {str(path): dict(exists=exists, sha256=cpu_evidence.sha(path) if exists else None)}


def chat_inputs(repo, profile, env, *, image=None, model=None, use_profile_defaults=False):
    if profile != 'glm53':
        return None
    if any(not (repo / p).is_file() or cpu_evidence.sha(repo / p) != expected
           for p, expected in CHAT_SOURCE_AUDIT.items()):
        raise ValueError('chat CPU dependency audit changed; review tokenizer inputs before reusing this release gate')
    selected_image = image or (None if use_profile_defaults else os.environ.get('IMAGE')) or profile_value(repo, profile, 'PROFILE_IMAGE')
    selected_model = model or (None if use_profile_defaults else os.environ.get('MODEL_HOST_PATH')) or profile_value(repo, profile, 'PROFILE_MODEL_PATH')
    model_path = Path(selected_model).resolve()
    if not (model_path / 'tokenizer_config.json').is_file() or not (model_path / 'tokenizer.json').is_file():
        raise ValueError('chat CPU reuse requires local tokenizer_config.json and tokenizer.json: ' + str(model_path))
    names = set(TOKENIZER_FILES)
    # AutoTokenizer may name extra local vocabulary files. Bind those names and
    # reject custom/external loaders whose dependencies have not been audited.
    for filename in ('tokenizer_config.json', 'config.json'):
        path = model_path / filename
        if not path.exists():
            continue
        config = json.loads(path.read_text())
        if not isinstance(config, dict) or config.get('auto_map'):
            raise ValueError('chat CPU dependency audit does not permit a custom tokenizer loader')
        for key, value in config.items():
            if value is None or key in ('name_or_path', '_name_or_path'):
                continue
            if key.endswith(('_file', '_files', '_path', '_paths', '_dir')):
                if key not in ('tokenizer_file', 'fast_tokenizer_files', 'vocab_file', 'merges_file',
                               'added_tokens_file', 'special_tokens_map_file', 'chat_template_file'):
                    raise ValueError('unknown tokenizer file dependency: ' + key)
                values = value if isinstance(value, list) else [value]
                for name in values:
                    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
                        raise ValueError('tokenizer dependency must be a local filename: ' + key)
                    if not (model_path / name).is_file():
                        raise ValueError('tokenizer dependency missing: ' + name)
                    names.add(name)
    templates = model_path / 'chat_templates'
    if templates.exists():
        for path in templates.iterdir():
            if not path.is_file() or path.suffix != '.jinja':
                raise ValueError('unknown tokenizer template dependency: ' + str(path))
            names.add(path.relative_to(model_path).as_posix())
    files = {}
    for name in sorted(names):
        path = model_path / name
        if path.exists() and (not path.is_file() or not path.resolve().is_relative_to(model_path)):
            raise ValueError('tokenizer dependency escapes the local model directory: ' + name)
        files[name] = cpu_evidence.sha(path) if path.is_file() else None
    image_id = run(['docker', 'image', 'inspect', '--format', '{{.Id}}', selected_image], repo, env=env)
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
        raise ValueError('chat CPU gate requires an immutable local Docker image ID')
    import shutil
    docker = shutil.which('docker', path=env['PATH'])
    return dict(image=image_id, model=str(model_path), files=files,
                docker=[docker, cpu_evidence.sha(docker)])


def gate_spec(repo, profile, env, **options):
    if not profile or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in profile):
        raise ValueError('invalid deployment profile')
    # Missing a previously mandatory test is never a successful, empty gate.
    for name in ('tests/test_logic.py', 'bench/cpu_checks.py', 'launchers/audit-runtime-guards.py',
                 'launchers/compose-overlays.sh', 'launchers/deploy-overlays.sh', 'profiles/' + profile + '.env'):
        if not (repo / name).is_file():
            raise ValueError('deployment CPU prerequisite missing: ' + name)
    command = [sys.executable, 'bench/cpu_checks.py', '--suite', 'logic']
    if profile == 'glm53':
        if not (repo / 'tests/test_glm53_overlay_sync.py').is_file():
            raise ValueError('deployment CPU prerequisite missing: tests/test_glm53_overlay_sync.py')
        command += ['--test', 'tests/test_glm53_overlay_sync.py']
    return dict(command=command, env={}, context=dict(gate='overlay-deploy', version=VERSION,
                profile=profile, validator=cpu_evidence.sha(Path(__file__).resolve()),
                fixed_logic_inputs=fixed_logic_inputs(),
                chat_release=chat_inputs(repo, profile, env, **options)),
                inputs=[], timeout_s=TIMEOUT)


def identity(repo, profile, env, **options):
    source(repo)
    spec = gate_spec(repo, profile, env, **options)
    value = cpu_evidence.identity(repo, spec, env)
    if not value:
        raise ValueError('CPU environment cannot be identified for reuse (check interpreter/dependencies, including editable installs)')
    # rsync is exercised by the GLM publication gate and must invalidate its receipt.
    import shutil
    extras = {}
    for name in ('rsync', 'nice'):
        binary = shutil.which(name, path=env['PATH'])
        extras[name] = [binary, cpu_evidence.sha(binary)] if binary else None
    extra = [value['key'], extras]
    value['key'] = hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest()
    return value, spec


def read_receipt(path, key, spec):
    try:
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            return None
        if value.get('version') != VERSION or value.get('key') != key or value.get('passed') is not True:
            return None
        if value.get('context') != spec['context'] or not value.get('coverage_complete') or value.get('tests_run', 0) < 1:
            return None
        # A receipt names immutable neighboring evidence; moving or editing it
        # cannot turn an incomplete/failed report into a passing gate.
        for name, digest in value['artifacts'].items():
            if Path(name).name != name or cpu_evidence.sha(path.parent / name) != digest:
                return None
        report = json.loads((path.parent / (key + '.report.json')).read_text())
        if not report['passed'] or not report['coverage_complete'] or report['tests_run'] != value['tests_run']:
            return None
        return value
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def execute(argv, repo, env, output):
    import shutil
    if shutil.which('nice', path=env['PATH']):
        argv = ['nice', '-n', '19', *argv]
    process = subprocess.Popen(argv, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        rc = process.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise ValueError('deployment CPU validation timed out') from exc
    if rc:
        raise ValueError('deployment CPU validation failed with code ' + str(rc))


def validate(repo, store, profile='glm53', *, require_receipt=None, verify_only=False,
             image=None, model=None, use_profile_defaults=False):
    repo = Path(repo).resolve()
    store = private_directory(store)
    receipts = private_directory(store / 'receipts')
    env = environment()
    options = dict(image=image, model=model, use_profile_defaults=use_profile_defaults)
    ident, spec = identity(repo, profile, env, **options)
    key = ident['key']
    path = receipts / (key + '.json')
    if require_receipt and Path(require_receipt).absolute() != path:
        raise ValueError('deployment CPU receipt does not match the exact source/environment')
    value = read_receipt(path, key, spec)
    if value:
        return dict(value, receipt=str(path), reused=True)
    if verify_only or require_receipt or held():
        raise ValueError('passing deployment CPU evidence is missing or changed; prepare again before GPU reservation')
    with lock(receipts / (key + '.lock')):
        # The waiter must re-read after the producer atomically publishes.
        value = read_receipt(path, key, spec)
        if value:
            return dict(value, receipt=str(path), reused=True)
        if held():
            raise ValueError('CPU validation cannot start while this session holds GPUs')
        log = receipts / (key + '.log')
        report = receipts / (key + '.report.json')
        report.unlink(missing_ok=True)
        started = time.time()
        try:
            with log.open('w') as output:
                execute(['bash', '-n', 'launchers/deploy-overlays.sh'], repo, env, output)
                # Logic checks inspect generated profile snapshots. Fresh
                # recovery worktrees have none, and old ignored builds may be
                # stale; derive every snapshot from the pinned module sources.
                for profile_path in sorted((repo / 'profiles').glob('*.env')):
                    execute(['bash', 'launchers/compose-overlays.sh', profile_path.stem], repo, env, output)
                execute([sys.executable, 'launchers/audit-runtime-guards.py', '--self-test'], repo, env, output)
                execute([*spec['command'], '--out', str(report)], repo, env, output)
                chat = spec['context']['chat_release']
                if chat:
                    execute(['bash', 'launchers/check-glm53-chat.sh', chat['model'], chat['image']], repo, env, output)
        except ValueError as exc:
            raise ValueError(str(exc) + '; log: ' + str(log)) from exc
        # Both source and environment are checked after execution. Concurrent
        # edits or dependency changes discard results rather than bless a race.
        after, after_spec = identity(repo, profile, environment(), **options)
        if after != ident or after_spec != spec:
            raise ValueError('source or CPU environment changed during validation; evidence discarded')
        result = json.loads(report.read_text())
        if not result.get('passed') or not result.get('coverage_complete') or result.get('tests_run', 0) < 1:
            raise ValueError('deployment CPU coverage is incomplete; log: ' + str(log))
        artifacts = {p.name: cpu_evidence.sha(p) for p in receipts.glob(key + '*')
                     if p.is_file() and p.suffix not in ('.lock', '.json')}
        artifacts[report.name] = cpu_evidence.sha(report)
        value = dict(version=VERSION, key=key, context=spec['context'], scope=ident['scope'],
                     source=source(repo), profile=profile, passed=True,
                     coverage_complete=True, tests_run=result['tests_run'],
                     started_at=started, completed_at=time.time(), artifacts=artifacts)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
        temporary.chmod(0o600)
        temporary.replace(path)
        return dict(value, receipt=str(path), reused=False)


def prepare_recovery(repo, store):
    if held():
        raise ValueError('prepare approved recovery before GPU reservation')
    repo, store = Path(repo).resolve(), private_directory(store)
    # Pin approval under a repository-specific lock, then validate outside that
    # lock so independent versions can prepare concurrently.
    repo_key = hashlib.sha256(str(repo).encode()).hexdigest()
    with lock(store / ('recovery-' + repo_key + '.lock')):
        run(['git', 'fetch', '--quiet', 'origin', 'main'], repo)
        commit = run(['git', 'rev-parse', '--verify', 'origin/main^{commit}'], repo)
        checkouts = private_directory(store / 'recovery')
        checkout = checkouts / (repo_key[:16] + '-' + commit)
        if not checkout.exists():
            run(['git', 'worktree', 'add', '--quiet', '--detach', str(checkout), commit], repo)
        if source(checkout) != commit:
            raise ValueError('approved recovery checkout changed; refuse to reset it: ' + str(checkout))
    value = validate(checkout, store, 'glm53', use_profile_defaults=True)
    if source(checkout) != commit:
        raise ValueError('approved recovery checkout changed during CPU validation')
    descriptor = dict(version=VERSION, repo=str(checkout), source=commit,
                      validation_receipt=value['receipt'], store=str(store))
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    directory = private_directory(store / 'recovery-receipts')
    path = directory / (digest + '.json')
    with lock(directory / (digest + '.lock')):
        if path.exists() and json.loads(path.read_text()) != descriptor:
            raise ValueError('pinned recovery receipt changed: ' + str(path))
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(descriptor, sort_keys=True, indent=2) + '\n')
            temporary.chmod(0o600)
            temporary.replace(path)
    return dict(descriptor, receipt=str(path), reused=value['reused'])


def recovery_info(receipt):
    path = Path(receipt).absolute()
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError('recovery receipt must be an owned regular file')
    value = json.loads(path.read_text())
    fields = {'version', 'repo', 'source', 'validation_receipt', 'store'}
    if not isinstance(value, dict) or set(value) != fields or any(
            not isinstance(value[k], str) or not value[k] for k in fields - {'version'}):
        raise ValueError('invalid pinned recovery receipt')
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    if value.get('version') != VERSION or path != Path(value['store']) / 'recovery-receipts' / (digest + '.json'):
        raise ValueError('pinned recovery receipt changed')
    return value


def verify_recovery(receipt, repo=None):
    value = recovery_info(receipt)
    if repo is not None and Path(repo).resolve() != Path(value['repo']).resolve():
        raise ValueError('deployment source is not the pinned recovery checkout')
    repo, commit = Path(value['repo']), value['source']
    if source(repo) != commit:
        raise ValueError('pinned approved recovery source changed')
    # No fetch, checkout switch or CPU test execution is allowed in recovery.
    # A later approved main remains compatible; a rewritten approval rejects.
    run(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'], repo)
    validate(repo, value['store'], 'glm53', require_receipt=value['validation_receipt'], verify_only=True,
             use_profile_defaults=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'prepare-recovery', 'verify-recovery'])
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--store', type=Path, default=default_store())
    parser.add_argument('--profile', default='glm53')
    parser.add_argument('--image', help='explicit candidate serving image (otherwise IMAGE/profile)')
    parser.add_argument('--model', help='explicit candidate tokenizer directory (otherwise MODEL_HOST_PATH/profile)')
    parser.add_argument('--receipt')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--format', choices=['json', 'shell', 'receipt'], default='json')
    args = parser.parse_args()
    try:
        if args.action == 'prepare-recovery':
            if not args.repo:
                parser.error('prepare-recovery requires --repo')
            value = prepare_recovery(args.repo, args.store)
        elif args.action == 'verify-recovery':
            if not args.receipt:
                parser.error('verify-recovery requires --receipt')
            value = verify_recovery(args.receipt, args.repo)
        else:
            if not args.repo:
                parser.error('validate requires --repo')
            value = validate(args.repo, args.store, args.profile, require_receipt=args.receipt,
                             verify_only=args.verify_only, image=args.image, model=args.model)
        if args.format == 'shell':
            if args.action == 'validate':
                parser.error('shell format is only valid for recovery')
            for key in ('repo', 'source'):
                print('export FLEET_RECOVERY_' + key.upper() + '=' + shlex.quote(value[key]))
            print('export FLEET_RECOVERY_RECEIPT=' + shlex.quote(value.get('receipt', args.receipt)))
            print('export FLEET_VALIDATION_STORE=' + shlex.quote(value['store']))
            print('export FLEET_VALIDATION_REQUIRED=1')
        elif args.format == 'receipt':
            print(value.get('receipt', args.receipt))
        else:
            print(json.dumps(value, sort_keys=True))
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print('ABORT: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
