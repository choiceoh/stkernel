#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and bind raw run inputs before queueing; cheap checks repeat before GO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

import fleet_prepared

MAX_FILE = 8 * 1024 * 1024


class PreparedPaused(ValueError):
    """A failed queued revision was retained for an explicit edit or resume."""


def run(argv, cwd, timeout=15, *, env=None):
    try:
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('prepare timed out: ' + ' '.join(argv[:3])) from exc
    if result.returncode:
        raise ValueError('prepare failed: ' + ' '.join(argv[:4]) + '\n' + (result.stderr or result.stdout)[-4000:])
    return result.stdout.strip()


def digest(path):
    if not path.is_file():
        raise ValueError('required file is missing: ' + str(path))
    if path.stat().st_size > MAX_FILE:
        raise ValueError('source exceeds 8 MiB; declare large inputs in required_paths: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_digest(path):
    """Explicit large inputs are streamed instead of loaded into memory."""
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def resolve(value, cwd):
    return str((Path(cwd) / value).resolve())


def command_environment(command, environment=None):
    """Unwrap literal env prefixes without evaluating shell text."""
    command = list(command)
    environment = dict(os.environ if environment is None else environment)
    while command and Path(command[0]).name == 'env':
        index, assignments, options = 1, False, True
        while index < len(command):
            arg = command[index]
            if options and not assignments and arg == '--':
                options = False
                index += 1
                continue
            if options and not assignments and arg in ('-i', '--ignore-environment', '-'):
                environment.clear()
                index += 1
                continue
            if options and not assignments and (arg in ('-u', '--unset') or arg.startswith('--unset=')):
                if arg.startswith('--unset='):
                    name = arg.split('=', 1)[1]
                else:
                    index += 1
                    if index >= len(command):
                        raise ValueError('env --unset requires a variable name')
                    name = command[index]
                if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
                    raise ValueError('env --unset requires a literal variable name')
                environment.pop(name, None)
                index += 1
                continue
            if arg.startswith('-'):
                raise ValueError('unsupported env prefix; use literal NAME=value, -u or -i, or explicit deployment_targets')
            name, equal, value = arg.partition('=')
            if equal and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
                environment[name] = value
                assignments = True
                index += 1
                continue
            break
        if index >= len(command):
            raise ValueError('env prefix requires a payload command')
        command = command[index:]
    return command, environment


def sources(command, cwd):
    """Only bind literal file arguments; never execute or expand shell text."""
    command, _ = command_environment(command)
    result = []
    interpreter = Path(command[0]).name
    if re.fullmatch(r'(?:ba|z|da)?sh|python(?:[0-9.]+)?', interpreter):
        for arg in command[1:]:
            if arg in ('-c', '-m', '-') or (arg.startswith('-') and 'c' in arg[1:]):
                break
            if arg.startswith('-'):
                continue
            path = Path(cwd) / arg
            if not path.is_file():
                raise ValueError('entrypoint script is missing: ' + str(path))
            result.append(path.resolve())
            break
    for arg in command:
        path = Path(cwd) / arg
        try:
            if path.is_file() and path.suffix in ('.py', '.sh', '.json', '.toml', '.yaml', '.yml'):
                result.append(path.resolve())
        except (OSError, ValueError):
            continue
    executable = Path(cwd) / command[0]
    if '/' in command[0] and executable.is_file():
        with executable.open('rb') as stream:
            if stream.read(2) == b'#!':
                result.append(executable.resolve())
    return list(dict.fromkeys(result))


def execution_cwd(path, cwd, text, environment=None):
    """Resolve the literal entry-directory forms used by reviewed wrappers."""
    environment = os.environ if environment is None else environment
    repo_value = environment.get('REPO')
    for line in text.splitlines():
        match = re.fullmatch(r'(?:export )?REPO=(.+)', line)
        if match:
            if match[1] == '$(cd "$(dirname "$0")/.." && pwd)':
                repo_value = str(path.parent.parent)
            else:
                words = shlex.split(match[1], comments=True)
                if len(words) == 1:
                    repo_value = literal_shell_value(words[0], dict(environment, PWD=str(cwd)))
        if not line.startswith('cd '):
            continue
        value = line[3:].strip()
        if value == '"$(dirname "$0")/.."':
            return path.parent.parent
        if value == '"${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"':
            return (Path(cwd) / (environment.get('REPO') or path.parent.parent)).resolve()
        if value == '"$REPO"' and repo_value:
            return (Path(cwd) / repo_value).resolve()
        try:
            words = shlex.split(value, comments=True)
        except ValueError as exc:
            raise ValueError('cannot resolve wrapper working directory; declare deployment_targets: ' + str(path)) from exc
        if len(words) == 1 and not any(c in words[0] for c in '$`;&|<>'):
            return (Path(cwd) / words[0]).resolve()
        raise ValueError('dynamic wrapper working directory requires explicit deployment_targets: ' + str(path))
    return Path(cwd)


def literal_shell_value(value, environment):
    if re.fullmatch(r'\$[A-Za-z_][A-Za-z0-9_]*', value):
        name = value[1:]
        if name not in environment:
            raise ValueError('unresolved deployment variable ' + name)
        return environment[name]
    match = re.fullmatch(r'\$\{([A-Za-z_][A-Za-z0-9_]*):-([^$`{}]*)\}', value)
    if match:
        return environment.get(match[1]) or match[2]
    if any(c in value for c in '$`;&|<>'):
        raise ValueError('dynamic deployment setting requires explicit deployment_targets')
    return value


def deployment_target(value, cwd):
    if (not isinstance(value, dict) or set(value) - {'repo', 'profile', 'image', 'model'}
            or 'repo' not in value or any(not isinstance(v, str) or not v or '\0' in v for v in value.values())):
        raise ValueError('deployment_targets entries require repo and optional literal profile, image, model')
    result = dict(value, repo=resolve(value['repo'], cwd), profile=value.get('profile', 'glm53'))
    if not re.fullmatch(r'[A-Za-z0-9_-]+', result['profile']):
        raise ValueError('deployment target profile must be a literal name')
    if result.get('model'):
        result['model'] = resolve(result['model'], result['repo'])
    return result


def deployment_targets(command, cwd, spec):
    if 'deployment_targets' in spec:
        if not isinstance(spec['deployment_targets'], list) or not spec['deployment_targets']:
            raise ValueError('deployment_targets must be a nonempty array')
        return [deployment_target(v, cwd) for v in spec['deployment_targets']]
    command, requested_environment = command_environment(command)
    result = []
    for path in sources(command, cwd):
        if path.suffix != '.sh':
            continue
        text = path.read_text(errors='replace')
        direct = path.name == 'deploy-overlays.sh' and path.parent.name == 'launchers'
        if not direct and not any('deploy-overlays.sh' in line for line in text.splitlines() if not line.lstrip().startswith('#')):
            continue
        execution = path.parent.parent if direct else execution_cwd(path, cwd, text, requested_environment)
        environment = dict(requested_environment, PWD=str(execution))
        exported = set(requested_environment)
        invocations = []
        if direct:
            index = next((i for i,v in enumerate(command) if (Path(cwd) / v).resolve() == path), len(command))
            args = command[index + 1:]
            invocations.append((str(path), environment.get('PROFILE') or (args[0] if args else 'dsv4'), dict(environment), set(exported)))
        else:
            for line in text.splitlines():
                if line.lstrip().startswith('#'):
                    continue
                try:
                    words = shlex.split(line, comments=True)
                except ValueError:
                    continue
                if not words:
                    continue
                assignment = re.match(r'\s+(?:export\s+)?(?:REPO|IMAGE|MODEL_HOST_PATH|PROFILE)=', line)
                if assignment:
                    # A conditional/function assignment cannot be equated to
                    # a literal top-level assignment in a custom wrapper.
                    raise ValueError('conditional deployment setting requires explicit deployment_targets: ' + str(path))
                if line[:1].isspace():
                    continue
                if words[0] in ('source', '.', 'eval'):
                    raise ValueError('sourced deployment settings require explicit deployment_targets: ' + str(path))
                if words[0] == 'read' and any(v in {'REPO','IMAGE','MODEL_HOST_PATH','PROFILE'} for v in words[1:]):
                    raise ValueError('runtime deployment settings require explicit deployment_targets: ' + str(path))
                if words[0] == 'bash' and len(words)>1 and words[1].endswith('/deploy-overlays.sh'):
                    profile = environment.get('PROFILE') if 'PROFILE' in exported else None
                    if not profile:
                        profile = literal_shell_value(words[2], environment) if len(words)>2 and not words[2].startswith(('>','<')) else 'dsv4'
                    invocations.append((words[1],profile,dict(environment),set(exported)))
                    continue
                is_export = words[:1] == ['export']
                values = words[1:] if is_export else words
                for word in values:
                    name, equal, raw = word.partition('=')
                    if name not in {'REPO','IMAGE','MODEL_HOST_PATH','PROFILE'}:
                        continue
                    if not is_export and values[0] != word:
                        raise ValueError('conditional deployment setting requires explicit deployment_targets: ' + str(path))
                    if is_export:
                        exported.add(name)
                    if equal:
                        environment[name] = literal_shell_value(raw, environment)
            if not invocations:
                # Some audited probe wrappers deploy only in a legacy cleanup
                # function that the supervised restore contract suppresses.
                import fleet_source
                try:
                    script_repo = run(['git', 'rev-parse', '--show-toplevel'], path.parent, 3)
                    cleanup_only = ('FLEET_RESTORE_MANAGED' in text and fleet_source.audited_wrapper(path, script_repo))
                except (ValueError, OSError, subprocess.SubprocessError):
                    cleanup_only = False
                if cleanup_only:
                    continue
                raise ValueError('cannot locate a literal deployment invocation; declare deployment_targets: ' + str(path))
        for filename,profile,environment,exported in invocations:
            if filename.startswith('$REPO/'):
                if not environment.get('REPO'):
                    raise ValueError('deployment REPO is unresolved; declare deployment_targets: ' + str(path))
                filename = environment['REPO'] + filename[len('$REPO'):]
            elif '$' in filename or '`' in filename:
                raise ValueError('dynamic deployment path requires explicit deployment_targets: ' + str(path))
            deploy = (Path(execution) / filename).resolve()
            repo = deploy.parent.parent
            # Only an actual deployment target requires a checkout here. A
            # generic CPU command remains preparable outside any Git tree.
            repo = run(['git', 'rev-parse', '--show-toplevel'], repo, 3)
            target = dict(repo=repo, profile=profile)
            for name,key in (('IMAGE','image'),('MODEL_HOST_PATH','model')):
                if name in exported and environment.get(name):
                    target[key] = environment[name]
            target = deployment_target(target,cwd)
            if target not in result:
                result.append(target)
    if not result:
        target = dict(repo=requested_environment.get('REPO') or cwd, profile=requested_environment.get('PROFILE') or 'glm53')
        for name,key in (('IMAGE','image'),('MODEL_HOST_PATH','model')):
            if requested_environment.get(name):
                target[key] = requested_environment[name]
        result.append(deployment_target(target,cwd))
    return result


def discover(command, cwd):
    command, environment = command_environment(command)
    checks = []
    for path in sources(command, cwd):
        if path.suffix != '.sh':
            continue
        text = path.read_text(errors='replace')
        # Recognize literal, read-only repository guards used by campaign scripts.
        # Dynamic shell expressions remain the campaign's responsibility.
        try:
            execution = execution_cwd(path, cwd, text, environment)
            repo = run(['git', '-C', str(execution), 'rev-parse', '--show-toplevel'], cwd, 3)
        except ValueError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith('#'):
                continue
            match = re.search(r'python3 bench/fleet_source\.py require-base ([A-Za-z0-9_./-]+)(?:\s|$)', line)
            if match:
                check = dict(kind='source-base', repo=repo, ref=match[1], source=str(path), line=line_number)
                if match[1].startswith('origin/') and re.search(r'\bgit fetch origin ' + re.escape(match[1][7:]) + r'(?:\s|$)', text):
                    check['fetch'] = match[1][7:]
                checks.append(check)
            match = re.search(r'git merge-base --is-ancestor ([A-Za-z0-9_./-]+) HEAD(?:\s|$)', line)
            if match:
                ref = match[1]
                check = dict(kind='ancestor', repo=repo, ref=ref, source=str(path), line=line_number)
                if ref.startswith('origin/') and re.search(r'\bgit fetch origin ' + re.escape(ref[7:]) + r'(?:\s|$)', text):
                    check['fetch'] = ref[7:]
                checks.append(check)
            if 'git status --porcelain' in line and re.search(r'-z\s+"?\$\(git status --porcelain(?: --untracked-files=(?:normal|all))?\)', line):
                checks.append(dict(kind='clean', repo=repo, source=str(path), line=line_number))
    return checks


def spec_read(path):
    spec = json.loads(Path(path).read_text()) if path else {}
    allowed = {'required_paths', 'absent_paths', 'git', 'images', 'cpu_command', 'timeout_seconds', 'deployment_targets'}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError('unknown preparation fields; expected ' + ', '.join(sorted(allowed)))
    for key in ('required_paths', 'absent_paths', 'images', 'cpu_command'):
        if key in spec and (not isinstance(spec[key], list) or not all(isinstance(x, str) and x and '\0' not in x for x in spec[key])):
            raise ValueError(key + ' must be an array of nonempty strings')
    if 'git' in spec and (not isinstance(spec['git'], dict) or set(spec['git']) - {'repo', 'ancestor', 'clean'}):
        raise ValueError('git accepts repo, ancestor and clean')
    timeout = spec.get('timeout_seconds', 120)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 300:
        raise ValueError('timeout_seconds must be 1..300')
    return spec


def validate(value, *, refresh=False, external=True, directory=None):
    cwd = value['cwd']
    if directory is not None and value.get('environment_digest'):
        env = fleet_prepared.request_environment(cwd, value['session'])
        if fleet_prepared.private_digest(fleet_prepared.key(directory), env) != value['environment_digest']:
            raise ValueError('prepared environment changed (values withheld); prepare again under the intended environment')
    if not Path(cwd).is_dir():
        raise ValueError('working directory disappeared: ' + cwd)
    exe = value['command'][0]
    actual = shutil.which(resolve(exe, cwd) if '/' in exe else exe)
    if not actual:
        raise ValueError('executable missing: ' + exe)
    if value.get('executable') and executable_identity(actual) != value['executable']:
        raise ValueError('queued executable changed; prepare the intended runtime again: ' + actual)
    for filename, expected in value['files'].items():
        if digest(Path(filename)) != expected:
            raise ValueError('queued input changed; edit the reservation to accept the new input: ' + filename)
    for filename in value['required_paths']:
        if not Path(filename).exists():
            raise ValueError('required path is missing: ' + filename)
        if filename in value.get('required_files', {}) and input_digest(filename) != value['required_files'][filename]:
            raise ValueError('queued required input changed; prepare again: ' + filename)
    for filename in value['absent_paths']:
        if Path(filename).exists():
            raise ValueError('fresh output path already exists: ' + filename)
    if value.get('source_identity'):
        import fleet_source
        source = value['source_identity']
        current = fleet_source.identity(source['repo'], protected_paths=value.get('protected_paths', []))
        changed = fleet_source.compare(source['identity'], current)
        if not changed['equal']:
            raise ValueError('queued source changed; edit or prepare the intended inputs: ' + ', '.join(changed['changed_paths'][:5]))
    for repo, expected in value.get('deployment_sources', {}).items():
        import fleet_source
        scope = value.get('deployment_source_scopes', {}).get(repo, 'audited-wrapper')
        current = fleet_prepared.cpu_source_identity(repo, scope, value.get('protected_paths', []))
        changed = fleet_source.compare(expected, current)
        if not changed['equal']:
            raise ValueError('queued deployment source changed; prepare again: ' + repo + ': ' + ', '.join(changed['changed_paths'][:5]))
    if value.get('head'):
        repo, expected = value['head']
        if run(['git', 'rev-parse', 'HEAD'], repo, 3) != expected:
            raise ValueError('queued checkout revision changed; edit or submit the intended revision: ' + repo)
    cpu = value.get('cpu_result', {})
    if cpu.get('successful') and cpu.get('identity'):
        repo = value.get('source_identity', {}).get('repo') or (value.get('head') or [None])[0]
        expected = cpu['identity']['source']
        current = fleet_prepared.cpu_source_identity(repo, cpu['identity']['scope'], value.get('protected_paths', []))
        if expected['sha256'] != current['sha256']:
            raise ValueError('prepared CPU source changed; prepare again: ' + fleet_prepared.difference(expected['files'], current['files']))
        if external and cpu.get('reusable'):
            current = fleet_prepared.cpu_identity(repo, spec_read(value.get('spec_path')), value['command'], cwd,
                fleet_prepared.cpu_environment(cwd, value['session']), value['required_paths'], value.get('protected_paths', []))
            if not cpu_matches(cpu['identity'], current):
                raise ValueError('prepared CPU source or runtime changed; prepare again')
    fetched = set()
    for check in value['checks']:
        repo = check['repo']
        source = f"{check.get('source', repo)}:{check.get('line', 0)}"
        if refresh and check.get('fetch') and (repo, check['fetch']) not in fetched:
            run(['git', 'fetch', '--quiet', 'origin', check['fetch']], repo)
            fetched.add((repo, check['fetch']))
        if check['kind'] == 'clean' and run(['git', 'status', '--porcelain'], repo, 3):
            raise ValueError(source + ': campaign requires a clean checkout')
        if check['kind'] == 'ancestor':
            try:
                run(['git', 'merge-base', '--is-ancestor', check['ref'], 'HEAD'], repo, 3)
            except ValueError as exc:
                raise ValueError(source + ': candidate must include ' + check['ref'] + '; update the candidate and its CPU evidence before queueing') from exc
        if check['kind'] == 'source-base':
            import fleet_source
            fleet_source.require_base(repo, check['ref'], protected_paths=value.get('protected_paths', []))
    if external:
        for image in value['images']:
            actual = run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], cwd, 5)
            expected = value.get('image_ids', {}).get(image)
            if expected and actual != expected:
                raise ValueError('queued image identity changed; prepare the intended runtime again: ' + image)
    return value


def executable_identity(filename):
    path = Path(filename).resolve()
    stat = path.stat()
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def cpu_matches(expected, current):
    return (expected['key'] == current['key'] and expected['scope'] == current['scope']
            and expected['source']['sha256'] == current['source']['sha256'])


def reuse(directory, session, command, cwd, prepared, *, spec_path=None):
    value = fleet_prepared.read(directory, prepared)
    if value.get('session') != session:
        raise ValueError('prepared receipt belongs to a different session')
    if value.get('command') != command:
        raise ValueError('prepared command argv changed; prepare the intended command again')
    if value.get('cwd') != cwd:
        raise ValueError('prepared working directory changed; prepare again')
    old_spec = value.get('spec_path')
    if spec_path is not None and str(Path(spec_path).resolve()) != old_spec:
        raise ValueError('prepared specification path changed; prepare again')
    secret = fleet_prepared.key(directory)
    spec = spec_read(spec_path or old_spec)
    if fleet_prepared.private_digest(secret, spec) != value.get('spec_digest'):
        raise ValueError('prepared specification changed; prepare again')
    env = fleet_prepared.cpu_environment(cwd, session)
    if fleet_prepared.private_digest(secret, fleet_prepared.request_environment(cwd, session)) != value.get('environment_digest'):
        raise ValueError('prepared environment changed (values withheld); prepare again under the intended environment')
    validate(value, refresh=True)
    cpu = value.get('cpu_result', {})
    if cpu.get('required'):
        if not cpu.get('successful'):
            raise ValueError('prepared receipt has no successful CPU result; prepare again')
        if not cpu.get('reusable'):
            raise ValueError('prepared CPU result cannot be reused: ' + cpu.get('reason', 'unaudited dependency scope'))
    return Path(prepared).resolve()


def prepare(directory, session, command, cwd, *, spec_path=None, fleet=None, execute_cpu=True, prepared=None):
    if not isinstance(command, (list, tuple)) or not command or not command[0] or not all(isinstance(x, str) and '\0' not in x for x in command):
        raise ValueError('prepare requires a command argv')
    command = list(command)
    cwd = str(Path(cwd).resolve())
    if prepared:
        return reuse(directory, session, command, cwd, prepared, spec_path=spec_path)
    spec = spec_read(spec_path)
    files = sources(command, cwd) + sources(spec['cpu_command'], cwd) if spec.get('cpu_command') else sources(command, cwd)
    if spec_path:
        files.append(Path(spec_path).resolve())
    value = dict(version=2, session=session, command=command, cwd=cwd,
                 files={str(p):digest(p) for p in files}, checks=discover(command, cwd),
                 required_paths=[resolve(p,cwd) for p in spec.get('required_paths', [])],
                 absent_paths=[resolve(p,cwd) for p in spec.get('absent_paths', [])],
                 images=spec.get('images', []), spec_path=str(Path(spec_path).resolve()) if spec_path else None,
                 prepared_at=time.time())
    value['deployment_targets'] = deployment_targets(command, cwd, spec)
    value['protected_paths'] = list(dict.fromkeys([*value['files'], *value['required_paths']]))
    value['required_files'] = {p: input_digest(p) for p in value['required_paths'] if Path(p).is_file()}
    source_repo = None
    try:
        repo = run(['git', 'rev-parse', '--show-toplevel'], cwd, 3)
    except ValueError:
        pass
    else:
        source_repo = repo
        import fleet_source
        if any(c['kind'] == 'source-base' and fleet_source.audited_wrapper(c['source'], repo) for c in value['checks']):
            value['source_identity'] = dict(repo=repo, identity=fleet_source.identity(repo, protected_paths=value['protected_paths']))
        else:
            value['head'] = [repo, run(['git', 'rev-parse', 'HEAD'], repo, 3)]
    value['deployment_sources'] = {}
    discovered_deployment = 'deployment_targets' in spec or any(
        p.suffix == '.sh' and (p.name == 'deploy-overlays.sh' or any(
            'deploy-overlays.sh' in line for line in p.read_text(errors='replace').splitlines()
            if not line.lstrip().startswith('#'))) for p in sources(command, cwd))
    value['deployment_source_scopes'] = {}
    trusted_wrapper = False
    if discovered_deployment:
        import fleet_source
        for path in sources(command, cwd)[:1]:
            if path.suffix == '.sh':
                try:
                    script_repo = run(['git', 'rev-parse', '--show-toplevel'], path.parent, 3)
                    trusted_wrapper |= fleet_source.audited_wrapper(path, script_repo)
                except (ValueError, OSError, subprocess.SubprocessError):
                    pass
    for target in value['deployment_targets']:
        if discovered_deployment:
            scope = 'audited-wrapper' if trusted_wrapper else 'full-tree'
            value['deployment_source_scopes'][target['repo']] = scope
            value['deployment_sources'][target['repo']] = fleet_prepared.cpu_source_identity(target['repo'], scope, value['protected_paths'])
    for check in value['checks']:
        if check['repo'] != source_repo and check['repo'] not in value['deployment_sources']:
            scope = 'audited-wrapper' if trusted_wrapper else 'full-tree'
            value['deployment_source_scopes'][check['repo']] = scope
            value['deployment_sources'][check['repo']] = fleet_prepared.cpu_source_identity(check['repo'], scope, value['protected_paths'])
    git = spec.get('git', {})
    repo = resolve(git.get('repo', cwd), cwd)
    if git.get('ancestor'):
        if not isinstance(git['ancestor'], str) or not re.fullmatch(r'[A-Za-z0-9_./-]+', git['ancestor']):
            raise ValueError('git ancestor must be a literal ref')
        check = dict(kind='ancestor', repo=repo, ref=git['ancestor'])
        if git['ancestor'].startswith('origin/'):
            check['fetch'] = git['ancestor'][7:]
        value['checks'].append(check)
    if git.get('clean'):
        value['checks'].append(dict(kind='clean', repo=repo))
    exe = command[0]
    actual = shutil.which(resolve(exe, cwd) if '/' in exe else exe)
    if not actual:
        raise ValueError('executable missing: ' + exe)
    value['executable'] = executable_identity(actual)
    value['image_ids'] = {image: run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], cwd, 5)
                          for image in value['images']}
    secret = fleet_prepared.key(directory, create=True)
    env = fleet_prepared.cpu_environment(cwd, session)
    value['environment_digest'] = fleet_prepared.private_digest(secret, fleet_prepared.request_environment(cwd, session))
    value['spec_digest'] = fleet_prepared.private_digest(secret, spec)
    value['cpu_result'] = dict(required=bool(spec.get('cpu_command')), successful=False)
    if spec.get('cpu_command'):
        value['cpu_result'].update(reusable=False)
        try:
            value['cpu_result']['identity'] = fleet_prepared.cpu_identity(source_repo, spec, command, cwd, env,
                value['required_paths'], value['protected_paths'])
            value['cpu_result']['reusable'] = True
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            value['cpu_result']['reason'] = str(exc)
    validate(value, refresh=True)
    for path in files:
        if path.suffix == '.sh':
            run(['bash', '-n', str(path)], cwd, 5)
        elif path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec')
    if execute_cpu and spec.get('cpu_command'):
        if not fleet:
            raise ValueError('CPU preparation requires the fleet classifier')
        command_class = run(['bash', str(fleet), 'classify', *spec['cpu_command']], cwd, 15)
        if command_class == 'gpu':
            raise ValueError('CPU preparation command shows GPU use; run classify --explain to locate it')
        try:
            process = subprocess.Popen(spec['cpu_command'], cwd=cwd, env=env,
                                       stdout=sys.stderr, stderr=sys.stderr, start_new_session=True)
            process.wait(timeout=spec.get('timeout_seconds',120))
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise ValueError('CPU preparation timed out') from exc
        if process.returncode:
            raise ValueError('CPU preparation failed with code ' + str(process.returncode))
        validate(value)
        value['cpu_result']['successful'] = True
        value['cpu_result']['completed_at'] = time.time()
        if value['cpu_result'].get('reusable'):
            try:
                current = fleet_prepared.cpu_identity(source_repo, spec, command, cwd, env,
                    value['required_paths'], value['protected_paths'])
                if not cpu_matches(value['cpu_result']['identity'], current):
                    raise ValueError('CPU preparation changed source or runtime inputs; prepare again')
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                value['cpu_result'].update(reusable=False, reason=str(exc))
    fleet_prepared.sign(value, secret)
    root = (Path(directory) / 'preparations').resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / (uuid.uuid4().hex + '.json')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(value, out, ensure_ascii=False, sort_keys=True)
        out.write('\n')
    return path


def validate_targets(directory, manifest, *, verify_only=False, _validated_value=None, controller=None):
    """Check actual deployment sources under their reservation's pinned contract."""
    value = _validated_value
    if value is None:
        value = fleet_prepared.read(directory, manifest)
        validate(value, refresh=True, directory=directory)
    targets = value.get('deployment_targets')
    if not isinstance(targets, list) or not targets:
        raise ValueError('prepared manifest has no deployment targets; prepare again')
    _, validation_env = command_environment(value['command'])
    # Target selections are explicit CLI arguments. Missing selections mean
    # profile defaults, including when an env -u/-i prefix removed a value.
    for name in ('IMAGE', 'MODEL_HOST_PATH', 'PROFILE'):
        validation_env.pop(name, None)
    for name in ('FLEET_DIR', 'FLEET_SESSION', 'FLEET_VALIDATION_STORE', 'FLEET_VALIDATION_REQUIRED'):
        if name in os.environ:
            validation_env[name] = os.environ[name]
    validator = Path(__file__).with_name('fleet_validation.py')
    level = 'admission'
    if controller is not None:
        fleet = controller.get('fleet')
        if not isinstance(fleet, str) or not Path(fleet).is_absolute():
            raise ValueError('reservation has no pinned validation controller')
        validator = Path(fleet).with_name('fleet_validation.py')
        accepted_env = controller.get('validation_env', {})
        level = accepted_env.get('FLEET_VALIDATION_LEVEL', 'release')
        if level not in ('admission', 'release'):
            raise ValueError('reservation has an unknown validation level')
        for name in ('FLEET_VALIDATION_STORE', 'FLEET_VALIDATION_REQUIRED'):
            if name in accepted_env:
                validation_env[name] = accepted_env[name]
        validation_env.update(FLEET_DIR=str(directory), FLEET_SESSION=controller['session'])
    # Never let an editor's managed-admission context change a legacy helper's
    # default release contract. Old helpers also do not accept --level.
    validation_env.pop('FLEET_VALIDATION_LEVEL', None)
    python = shutil.which('python3', path=validation_env.get('PATH', os.defpath))
    if not python:
        raise ValueError('deployment target environment has no python3 executable')
    results = []
    for target in targets:
        target = deployment_target(target, value['cwd'])
        argv = [python, str(validator), 'validate', '--repo', target['repo'], '--profile', target['profile']]
        if level == 'admission':
            argv += ['--level', 'admission']
        for key in ('image', 'model'):
            if target.get(key):
                argv += ['--' + key, target[key]]
        if verify_only:
            argv.append('--verify-only')
        result = json.loads(run(argv, target['repo'], 1800, env=validation_env))
        results.append(dict(target=target, validation=result))
    return results


def check_pending(directory, session, *, refresh=False, external=True, withdraw_failed=False):
    import fleet_pending
    record = fleet_pending.read_record(Path(directory), session)
    if not record or record.get('state') != 'queued' or not record.get('prepare_manifest'):
        return  # Compatibility: already queued older controllers retain their contract.
    try:
        # `fleet` is written by every fleet_pending.register(), so a record
        # without it predates the field -- the same older-controller contract
        # the early return above honours. Reading it unconditionally raised
        # KeyError, which is not in the except tuple below, so it escaped as an
        # error instead of a validation failure and took the legacy paths with
        # it (tests/test_fleet_prepare.py: 8 errors, deterministic).
        controller_path = record.get('fleet')
        if controller_path:
            controller = Path(controller_path).resolve().parent.parent
            if (controller / 'bench/fleet_onepass.py').is_file():
                from fleet_onepass import validate as validate_onepass
                validate_onepass(record['command'], record['cwd'], controller,
                                 environment=fleet_pending.supervisor_environment(record, Path(directory)),
                                 kind=record['kind'])
        value = json.loads(Path(record['prepare_manifest']).read_text())
        if not isinstance(value, dict):
            raise ValueError('preparation manifest must be an object')
        if record.get('prepare_receipt_required') or value.get('version', 1) >= 2 or value.get('receipt'):
            value = fleet_prepared.read(directory, record['prepare_manifest'])
        if value['command'] != record['command'] or value['cwd'] != record['cwd']:
            raise ValueError('preparation does not match the accepted command revision')
        validate(value, refresh=refresh, external=external, directory=directory)
        if (external and record.get('kind') == 'boot'
                and record.get('validation_env', {}).get('FLEET_VALIDATION_REQUIRED') == '1'):
            validate_targets(directory, record['prepare_manifest'], verify_only=True,
                             _validated_value=value, controller=record)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        if withdraw_failed and external:
            # pause_failed acquires .lock and compares the checked revision.
            # A stale result must never pause or delete a newer accepted edit.
            import fleet_pause
            if fleet_pause.pause_failed(Path(directory), session, record, str(exc)):
                raise PreparedPaused(str(exc)) from exc
            # False also means an unsupported legacy controller or a failed
            # ownership check. Only a demonstrably superseded result is safe
            # to ignore; an unchanged failed preparation must still refuse GO.
            with fleet_pending.lock(Path(directory)):
                current = fleet_pending.read_record(Path(directory), session)
                keys = ('ticket', 'pid', 'start', 'revision', 'prepare_manifest', 'state')
                if current and any(current.get(k) != record.get(k) for k in keys):
                    return
                if current and current.get('pause_protocol') != 1:
                    import fleet_handoff
                    rows = fleet_handoff.rows(Path(directory))
                    row = next((r for r in rows if r[1] == session and r[0] == record['ticket']
                                and r[6] == str(record['pid'])), None)
                    if row:
                        temp = Path(directory) / 'queue.prepare.tmp'
                        temp.write_text(''.join('|'.join(r) + '\n' for r in rows if r != row))
                        temp.replace(Path(directory) / 'queue')
                        for name in ('priority-front', 'priority-yield'):
                            marker = Path(directory) / name
                            if marker.exists() and marker.read_text().strip() == session:
                                marker.unlink()
        raise



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['create', 'check', 'validate-targets'])
    ap.add_argument('session')
    ap.add_argument('--spec')
    ap.add_argument('--fleet')
    ap.add_argument('--prepared')
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--local', action='store_true')
    ap.add_argument('--withdraw-failed', action='store_true')
    args, command = ap.parse_known_args(argv)
    if command[:1] == ['--']:
        command.pop(0)
    try:
        directory = Path(os.environ['FLEET_DIR'])
        if args.action == 'create':
            print(prepare(directory,args.session,command,os.getcwd(),spec_path=args.spec,fleet=args.fleet,prepared=args.prepared))
        elif args.action == 'validate-targets':
            if not args.prepared:
                raise ValueError('validate-targets requires --prepared MANIFEST')
            value = fleet_prepared.read(directory, args.prepared)
            if value.get('session') != args.session:
                raise ValueError('prepared receipt belongs to a different session')
            print(json.dumps(validate_targets(directory, args.prepared, verify_only=args.verify_only), sort_keys=True))
        else:
            check_pending(directory,args.session,refresh=args.refresh,external=not args.local,withdraw_failed=args.withdraw_failed)
    except PreparedPaused as exc:
        print('PREPARE PAUSED (reservation retained): ' + str(exc), file=sys.stderr)
        return 4
    except (ValueError, OSError, SyntaxError, subprocess.SubprocessError) as exc:
        print('PREPARE REFUSED (no GPU hold): ' + str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
