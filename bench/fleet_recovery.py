"""Reuse approved release recovery; refresh it independently of experiments."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


class ApprovalChanged(ValueError):
    """A previously approved recovery is no longer on the approved history."""


def owned_json(path):
    path = Path(path).absolute()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o022 or info.st_size > 2 * 1024 * 1024):
            raise ValueError('recovery evidence must be an owned regular file: ' + str(path))
        return json.load(stream)


def descriptor(api, receipt):
    # Keep the version-one immutable descriptor and its content-addressed path.
    raw = owned_json(receipt)
    value = api.recovery_info(receipt)
    if raw != value:
        raise ValueError('pinned recovery receipt changed during verification')
    store, repo = Path(value['store']), Path(value['repo'])
    if (not store.is_absolute() or store.resolve() != store
            or repo.parent != store / 'recovery' or repo.resolve() != repo
            or not re.fullmatch(r'[0-9a-f]{16}-' + re.escape(value['source']), repo.name)
            or not re.fullmatch(r'[0-9a-f]{40}', value['source'])):
        raise ValueError('pinned recovery checkout path changed')
    return value


def release_receipt(value):
    path = Path(value['validation_receipt'])
    receipt = owned_json(path)
    context = receipt.get('context', {}) if isinstance(receipt, dict) else {}
    key = receipt.get('key', '') if isinstance(receipt, dict) else ''
    if (not isinstance(context, dict) or context.get('gate') != 'overlay-deploy'
            or context.get('level', 'release') != 'release'
            or context.get('profile') != 'glm53'
            or receipt.get('profile') != 'glm53' or receipt.get('passed') is not True
            or receipt.get('coverage_complete') is not True
            or type(receipt.get('tests_run')) is not int or receipt['tests_run'] < 1
            or not re.fullmatch(r'[0-9a-f]{64}', key)
            or path != Path(value['store']) / 'receipts' / (key + '.json')):
        raise ValueError('approved recovery requires complete release CPU evidence')
    return receipt


def approved(api, value):
    repo, commit = Path(value['repo']), value['source']
    if api.source(repo) != commit:
        raise ValueError('pinned approved recovery source changed')
    try:
        api.run(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'], repo)
    except ValueError as exc:
        raise ApprovalChanged('pinned approved recovery is no longer an ancestor of origin/main') from exc


def verify(api, receipt, repo=None):
    value = descriptor(api, receipt)
    if repo is not None and Path(repo).resolve() != Path(value['repo']):
        raise ValueError('deployment source is not the pinned recovery checkout')
    approved(api, value)
    evidence = release_receipt(value)
    checkout = Path(value['repo'])
    expected = evidence['context'].get('validator')
    current = hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()
    if expected == current:
        api.validate(checkout, value['store'], 'glm53', level='release',
                     require_receipt=value['validation_receipt'], verify_only=True,
                     use_profile_defaults=True)
    else:
        # Release receipts include the executing validator's bytes. A newer
        # admission controller must not invalidate a still-approved older gate.
        # Only the helper committed in this exact approved checkout can consume
        # its old receipt, and --verify-only prevents any missing gate execution.
        helper = checkout / 'bench/fleet_validation.py'
        if (not helper.is_file() or helper.is_symlink()
                or hashlib.sha256(helper.read_bytes()).hexdigest() != expected):
            raise ValueError('release receipt validator is unavailable; refresh approved recovery before queueing')
        output = api.run([sys.executable, str(helper), 'validate', '--repo', str(checkout),
                          '--store', value['store'], '--profile', 'glm53', '--receipt',
                          value['validation_receipt'], '--verify-only'], checkout,
                         env=api.environment(), timeout=60)
        result = json.loads(output)
        if (not isinstance(result, dict) or result.get('receipt') != value['validation_receipt']
                or result.get('key') != evidence['key'] or result.get('passed') is not True
                or result.get('context') != evidence['context'] or result.get('reused') is not True):
            raise ValueError('pinned release validator did not verify the requested evidence')
    if descriptor(api, receipt) != value or release_receipt(value) != evidence:
        raise ValueError('pinned recovery evidence changed during verification')
    approved(api, value)
    return value


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def pointer_value(api, path, store, repo_key):
    try:
        pointer = owned_json(path)
    except FileNotFoundError:
        return None
    if (not isinstance(pointer, dict) or set(pointer) != {'version', 'receipt'}
            or pointer['version'] != api.VERSION or not isinstance(pointer['receipt'], str)):
        raise ValueError('invalid approved recovery pointer: ' + str(path))
    value = descriptor(api, pointer['receipt'])
    if value['store'] != str(store) or not Path(value['repo']).name.startswith(repo_key[:16] + '-'):
        raise ValueError('approved recovery pointer belongs to a different production repository')
    return pointer['receipt'], value


def source_release(api, checkout, store):
    """Mint portable evidence with the approved source's own release validator."""
    helper = checkout / 'bench/fleet_validation.py'
    current = hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()
    if helper.is_symlink():
        raise ValueError('approved recovery validator must be a regular source file')
    if not helper.is_file() or hashlib.sha256(helper.read_bytes()).hexdigest() == current:
        # The missing-helper case retains old fixture/pre-helper API behavior.
        return api.validate(checkout, store, 'glm53', level='release', use_profile_defaults=True)
    expected = hashlib.sha256(helper.read_bytes()).hexdigest()
    if api.held():
        raise ValueError('prepare approved recovery before GPU reservation')
    # Pre-split validators have no --level option; their default is the full
    # release gate. The sanitized environment also removes deployment overrides.
    output = api.run([sys.executable, str(helper), 'validate', '--repo', str(checkout),
                      '--store', str(store), '--profile', 'glm53'], checkout,
                     env=api.environment(), timeout=1800)
    result = json.loads(output)
    if not isinstance(result, dict) or not isinstance(result.get('receipt'), str):
        raise ValueError('pinned release validator did not return CPU evidence')
    evidence = release_receipt(dict(validation_receipt=result['receipt'], store=str(store)))
    if (evidence['context'].get('validator') != expected or result.get('context') != evidence['context']
            or result.get('key') != evidence['key'] or result.get('passed') is not True
            or type(result.get('reused')) is not bool):
        raise ValueError('pinned release validator did not produce its own release evidence')
    return result


def prepare(api, repo, store, refresh=False):
    if api.held():
        raise ValueError('prepare approved recovery before GPU reservation')
    repo, store = Path(repo).resolve(), api.private_directory(store)
    repo_key = hashlib.sha256(str(repo).encode()).hexdigest()
    lock_path = store / ('recovery-' + repo_key + '.lock')
    pointers = api.private_directory(store / 'recovery-current')
    pointer = pointers / (repo_key + '.json')
    directory = api.private_directory(store / 'recovery-receipts')
    with api.lock(lock_path):
        # Refresh approval once, without making the newest approved commit the
        # mandatory recovery target for every experiment.
        api.run(['git', 'fetch', '--quiet', 'origin', 'main'], repo)
        commit = api.run(['git', 'rev-parse', '--verify', 'origin/main^{commit}'], repo)
        previous = pointer_value(api, pointer, store, repo_key)
        if previous:
            approved(api, previous[1])  # A rewritten main is never a cache miss.
        if not refresh:
            if previous:
                try:
                    value = verify(api, previous[0])
                    return dict(value, receipt=previous[0], reused=True, selection='pinned')
                except ApprovalChanged:
                    raise
                except (ValueError, OSError, subprocess.SubprocessError):
                    pass
            # Existing version-one deployments have descriptors but no pointer.
            # Newest completed passing release is the migration preference.
            candidates = []
            for path in directory.glob('*.json'):
                if previous and str(path) == previous[0]:
                    continue
                try:
                    value = descriptor(api, path)
                    if (value['store'] != str(store)
                            or not Path(value['repo']).name.startswith(repo_key[:16] + '-')):
                        continue
                    evidence = release_receipt(value)
                    completed = evidence.get('completed_at', 0)
                    if not isinstance(completed, (int, float)):
                        continue
                    candidates.append((completed, str(path)))
                except (ValueError, OSError, KeyError, TypeError):
                    continue
            for _, selected in sorted(candidates, reverse=True):
                try:
                    value = verify(api, selected)
                except ApprovalChanged:
                    raise
                except (ValueError, OSError, subprocess.SubprocessError):
                    continue
                atomic_json(pointer, dict(version=api.VERSION, receipt=selected))
                return dict(value, receipt=selected, reused=True, selection='existing')
        checkouts = api.private_directory(store / 'recovery')
        checkout = checkouts / (repo_key[:16] + '-' + commit)
        if not checkout.exists():
            api.run(['git', 'worktree', 'add', '--quiet', '--detach', str(checkout), commit], repo)
        if api.source(checkout) != commit:
            raise ValueError('approved recovery checkout changed; refuse to reset it: ' + str(checkout))
    # A full release gate may run only for a refresh or when no usable approved
    # recovery exists. Its own content lock coalesces concurrent preparations.
    result = source_release(api, checkout, store)
    value = dict(version=api.VERSION, repo=str(checkout), source=commit,
                 validation_receipt=result['receipt'], store=str(store))
    approved(api, value)
    release_receipt(value)
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path = directory / (digest + '.json')
    with api.lock(lock_path):
        if path.exists():
            if owned_json(path) != value:
                raise ValueError('pinned recovery receipt changed: ' + str(path))
        else:
            atomic_json(path, value)
        current_pointer = pointer_value(api, pointer, store, repo_key)
        # A slower refresh must not replace a newer completed refresh's source.
        update = True
        if current_pointer and current_pointer[1]['source'] != commit:
            current_source = current_pointer[1]['source']
            approved(api, current_pointer[1])
            try:
                api.run(['git', 'merge-base', '--is-ancestor', commit, current_source], repo)
                update = False
            except ValueError:
                pass
        if update:
            atomic_json(pointer, dict(version=api.VERSION, receipt=str(path)))
    return dict(value, receipt=str(path), reused=result['reused'], selection='latest-main')
