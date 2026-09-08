#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small CPU admission contract for a GPU experiment, not release evidence.

Only shell syntax, the selected overlay composition and Python compilation are
checked. Overlay modules are never imported. The isolated stdlib interpreter
does not load torch, user-site packages, tokenizer files or Docker images.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time

VERSION = 1
TIMEOUT = 30
CHECKS = ('shell-syntax', 'selected-profile-compose', 'overlay-manifest-syntax')
TOOLS = ('bash', 'git', 'dirname', 'rm', 'mkdir', 'install', 'wc', 'nice')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv, repo, env, timeout=TIMEOUT):
    result = subprocess.run(argv, cwd=repo, env=env, capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        raise ValueError(shlex.join(argv[:4]) + ': ' + (result.stderr or result.stdout)[-3000:].strip())
    return result.stdout.strip()


def local_file(repo, path):
    path = Path(path)
    if (not path.is_file() or not path.resolve().is_relative_to(repo)
            or path.resolve() != path.absolute()):
        raise ValueError('admission input must be a repository file: ' + str(path))
    return path


def gate_spec(repo, profile, env, *, validator_sha=None, **_options):
    repo = Path(repo).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', profile or ''):
        raise ValueError('invalid deployment profile')
    for name in ('launchers/compose-overlays.sh', 'launchers/deploy-overlays.sh',
                 'launchers/lib/common-tp4.sh', 'profiles/' + profile + '.env'):
        local_file(repo, repo / name)
    if validator_sha is None:
        validator_sha = sha(local_file(repo, repo / 'bench/fleet_validation.py'))
    checker = Path(__file__).resolve()
    return dict(command=[sys.executable, '-I', '-S', str(checker), '--repo', str(repo),
                         '--profile', profile], env={}, inputs=[], timeout_s=TIMEOUT,
                context=dict(gate='overlay-admission', version=VERSION, profile=profile,
                             validator=validator_sha, checker=sha(checker), selected=list(CHECKS)))


def identity(repo, profile, env, **options):
    """Bind this contract's inputs without enumerating installed distributions."""
    repo = Path(repo).resolve()
    if run(['git', 'status', '--porcelain', '--untracked-files=normal'], repo, env):
        raise ValueError('CPU admission requires a clean source checkout: ' + str(repo))
    tree = run(['git', 'rev-parse', '--verify', 'HEAD^{tree}'], repo, env)
    spec = gate_spec(repo, profile, env, **options)
    binaries = {}
    for name in (sys.executable, *TOOLS):
        path = shutil.which(name, path=env.get('PATH'))
        if not path and name != 'nice':
            raise ValueError('CPU admission tool missing: ' + name)
        binaries[name] = [path, sha(path)] if path else None
    # The compiler lives in the interpreter or libpython, not site-packages.
    library = Path(sysconfig.get_config_var('LIBDIR') or '') / (sysconfig.get_config_var('LDLIBRARY') or '')
    runtime = dict(version=sys.version, executable=sys.executable,
                   library=[str(library), sha(library)] if library.is_file() else None)
    # A queued controller is copied to an immutable runner before deployment.
    # Its bytes are already in context; its incidental location is not an input.
    invocation = list(spec['command'])
    invocation[3] = '<admission-checker>'
    identified_spec = dict(spec, command=invocation)
    data = dict(scope='admission-full-tree', tree=tree, spec=identified_spec, tools=binaries,
                runtime=runtime, environment=dict(env))
    components = {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                  for name, value in data.items() if name != 'tree'}
    return dict(key=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                scope=data['scope'], tree=tree, components=components), spec


def manifest_rows(path, *, composed):
    rows = []
    sources, targets = set(), set()
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split('\t')
        if len(fields) != 3:
            raise ValueError('malformed overlay manifest row: ' + str(path))
        source, target, contract = fields
        if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9._-]*', source):
            raise ValueError('unsafe overlay source: ' + source)
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', target) or '..' in target.split('/'):
            raise ValueError('unsafe overlay target: ' + target)
        if composed and not target.startswith('/'):
            raise ValueError('composed overlay target must be absolute: ' + target)
        if contract != 'absent' and not re.fullmatch(r'[0-9a-f]{64}', contract):
            raise ValueError('invalid base preimage contract: ' + source)
        normalized = str(PurePosixPath(target))
        if source in sources or normalized in targets:
            raise ValueError('duplicate overlay source or target: ' + source)
        sources.add(source)
        targets.add(normalized)
        rows.append((source, target, contract))
    if not rows:
        raise ValueError('overlay manifest is empty: ' + str(path))
    return rows


def selected_inputs(repo, profile):
    """Check sources before compose writes files; profiles declare literal MODULES."""
    profile_path = local_file(repo, repo / 'profiles' / (profile + '.env'))
    values = re.findall(r'^MODULES=(.*)$', profile_path.read_text(), re.M)
    if len(values) != 1:
        raise ValueError('profile must declare one literal MODULES list')
    words = shlex.split(values[0], comments=True)
    if len(words) != 1 or any(c in words[0] for c in '$`\n\r'):
        raise ValueError('profile must declare one literal MODULES list')
    modules = words[0].split()
    if not modules or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in modules):
        raise ValueError('invalid profile MODULES list')
    for name in modules:
        module = repo / 'overlay/modules' / name
        manifest = local_file(repo, module / 'manifest.tsv')
        for source, _target, _contract in manifest_rows(manifest, composed=False):
            local_file(repo, module / source)
        if (module / 'requires').exists():
            local_file(repo, module / 'requires')


def check(repo, profile, env):
    repo = Path(repo).resolve()
    # Keep the CLI usable with an older source checkout; its validator is bound
    # by the parent, not executed by this stdlib checker.
    gate_spec(repo, profile, env, validator_sha='parent-bound')
    report = dict(version=VERSION, evidence='cpu-only', scope='overlay-admission',
                  profile=profile, selected=list(CHECKS), selection='fixed-admission',
                  checks=[], failed=[], missing=[], passed=False,
                  coverage_complete=False, tests_run=0)
    started = time.monotonic()
    shell_paths = [repo / 'profiles' / (profile + '.env'),
                   repo / 'launchers/compose-overlays.sh', repo / 'launchers/deploy-overlays.sh']
    shell_paths += sorted((repo / 'launchers/lib').glob('*.sh'))
    shell_paths = list(dict.fromkeys(shell_paths))

    def command(argv):
        remaining = TIMEOUT - (time.monotonic() - started)
        if remaining <= 0:
            raise ValueError('CPU admission timed out')
        return run(argv, repo, env, timeout=remaining)

    for name in CHECKS:
        report['tests_run'] += 1
        try:
            if name == 'shell-syntax':
                for path in shell_paths:
                    command(['bash', '-n', str(local_file(repo, path))])
                details = dict(files=len(shell_paths))
            elif name == 'selected-profile-compose':
                selected_inputs(repo, profile)
                command(['bash', 'launchers/compose-overlays.sh', profile])
                details = dict(profiles=[profile])
            else:
                directory = repo / 'build' / profile
                rows = manifest_rows(local_file(repo, directory / 'manifest.tsv'), composed=True)
                python_count, shell_count = 0, 0
                for source, _target, _contract in rows:
                    path = local_file(repo, directory / source)
                    if path.suffix == '.py':
                        compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
                        python_count += 1
                    elif path.suffix == '.sh':
                        command(['bash', '-n', str(path)])
                        shell_count += 1
                details = dict(overlays=len(rows), python_files=python_count, shell_files=shell_count)
            report['checks'].append(dict(name=name, passed=True, **details))
        except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as exc:
            report['failed'].append(dict(name=name, error=str(exc)))
            report['checks'].append(dict(name=name, passed=False))
            break
    report['coverage_complete'] = len(report['checks']) == len(CHECKS) and not report['failed']
    report['passed'] = report['coverage_complete']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--profile', default='glm53')
    parser.add_argument('--out', '--json-out', dest='out', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = check(args.repo, args.profile, dict(os.environ))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report = dict(version=VERSION, scope='overlay-admission', profile=args.profile,
                      selected=list(CHECKS), selection='fixed-admission', passed=False,
                      coverage_complete=False, tests_run=0, checks=[], missing=[],
                      failed=[dict(name='prerequisites', error=str(exc))])
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
