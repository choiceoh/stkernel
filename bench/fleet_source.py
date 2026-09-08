#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source equivalence for audited fleet wrappers, never a GPU build identity.

Only reviewed wrappers and audited CPU suites may use this projection. Arbitrary
commands can read any tracked file and must retain their complete source binding.
Explicit inputs, executable files and symlinks are always source dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

VERSION = 1
OUTPUT_SUFFIXES = {'.md', '.json', '.jsonl', '.csv', '.tsv', '.log', '.txt', '.gz'}
INPUT_ENV = ('INPUT_REUSE_GPU_EVIDENCE', 'MOE_RESUME_NUMERICS')
# Reviewed wrappers produce evidence outputs and never consume the omitted prose.
# Any wrapper edit closes queued-source relaxation until its audit is renewed.
WRAPPER_AUDIT = {'probes/run_gemm_input_cta.sh': '6d8b45b97d3cb6de13964a3366fc6977e50a5ae828578b3fbfa2480d8f9ea934', 'probes/run_input_cta3.sh': 'ea24f3e5364a4456d13ab5cef0182bd924802f9a2da57f661ffda731b288500d', 'probes/run_input_cta3_production.sh': '79d745c01a2d02052cb70b12861c7c2d29ad748bed0d5bfa58a5f734b191c50b', 'probes/run_input_cta_serving.sh': '8c4b63049a8bcb4ccc52aa9e9ef5d5911f2cf32fa6d7682de55c30b899db0a80', 'probes/run_input_reuse_channels.sh': 'fe009083c29c0109907cad7d465b255b74721804396e4eb2c0bade540d844331', 'probes/run_input_reuse_serving.sh': '612ab382e8c23e69d265f05c5a3d4d3f52c31cb4f5a47846ffee7bb879dad4fc', 'probes/run_moe_reform_onepass.sh': '1dd3a3186993c3a5836e0e8507420c1bd05b22d40a7e438aabb03a7c8494dd91', 'probes/run_moe_reform_speed_only.sh': 'bd41ceaeec0e8ffb1c7c7d41ce5a96c009340a0e36280c695e865796d366d59f'}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)


def wrapper_inputs(environment=None):
    """Explicit prior-evidence inputs accepted by the reviewed wrappers."""
    environment = os.environ if environment is None else environment
    return [environment[name] for name in INPUT_ENV if environment.get(name)]


def audited_wrapper(path, repo):
    root = Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        name = candidate.resolve().relative_to(root).as_posix()
        return name in WRAPPER_AUDIT and not candidate.is_symlink() and hashlib.sha256(candidate.read_bytes()).hexdigest() == WRAPPER_AUDIT[name]
    except (OSError, ValueError):
        return False


def protected(repo, paths):
    root = Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    result = set()
    for value in paths:
        path = Path(value)
        path = path if path.is_absolute() else root / path
        # Keep the lexical symlink path as well as its target.
        for candidate in (Path(os.path.abspath(path)), path.resolve()):
            try:
                result.add(candidate.relative_to(root).as_posix())
            except ValueError:
                pass  # External inputs are separately bound by the caller.
    return result


def ignored(path, mode, protected_paths=()):
    """True only for regular nonexecutable documentation or retained outputs."""
    if mode != '100644' or any(p == '.' or path == p or path.startswith(p.rstrip('/') + '/')
                               for p in protected_paths):
        return False
    name = PurePosixPath(path)
    if name.suffix == '.md' and (path.startswith('docs/') or
            len(name.parts) == 1 and name.name.startswith('README') or
            path == 'bench/EXPERIMENTS.md'):
        return True
    return path.startswith('measurements/') and name.suffix in OUTPUT_SUFFIXES


def tree(repo, ref='HEAD'):
    result = {}
    for raw in git(repo, 'ls-tree', '-rz', ref).split(b'\0'):
        if raw:
            metadata, name = raw.decode().split('\t', 1)
            mode, kind, oid = metadata.split()
            result[name] = dict(mode=mode, oid=oid, kind=kind)
    return result


def identity(repo, *, protected_paths=()):
    """Bind effective source; harmless documentation commits retain sha256.

    HEAD metadata is intentionally absent. Modified and untracked source files
    are included so callers do not silently equate a dirty checkout to its HEAD.
    """
    repo = Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip())
    entries = tree(repo)
    pins = protected(repo, [*protected_paths, *wrapper_inputs()])
    changed = git(repo, 'ls-files', '-z', '--modified', '--others', '--exclude-standard')
    if pins:
        changed += git(repo, 'ls-files', '-z', '--others', '--ignored', '--exclude-standard', '--', *sorted(pins))
    for raw in sorted(set(changed.split(b'\0'))):
        if not raw:
            continue
        name = raw.decode()
        path = repo / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            entries.pop(name, None)
            continue
        if stat.S_ISLNK(info.st_mode):
            content, mode = os.readlink(path).encode(), '120000'
        elif stat.S_ISREG(info.st_mode):
            mode = '100755' if info.st_mode & 0o111 else '100644'
            if ignored(name, mode, pins):
                entries[name] = dict(mode=mode, oid='ignored', kind='blob')
                continue
            content = path.read_bytes()
        else:
            raise ValueError('unsupported source path: ' + str(path))
        # Match Git blob IDs, allowing a commit of unchanged bytes to retain ID.
        oid = subprocess.check_output(['git', '-C', str(repo), 'hash-object', '--stdin'], input=content).decode().strip()
        entries[name] = dict(mode=mode, oid=oid, kind='blob')
    omitted = sorted(p for p, item in entries.items() if ignored(p, item['mode'], pins))
    omitted_set = set(omitted)
    files = {p: item for p, item in entries.items() if p not in omitted_set}
    data = dict(version=VERSION, files=files)
    return dict(data, sha256=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), ignored=omitted)


def compare(expected, current):
    before, after = expected.get('files', {}), current.get('files', {})
    return dict(equal=expected.get('version') == current.get('version') == VERSION and
                bool(expected.get('sha256')) and
                expected.get('sha256') == current.get('sha256'),
                changed_paths=sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p)))


class SourceMismatch(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__('candidate must include relevant changes from ' + details['ref'] + ': ' +
                         ', '.join(details['relevant_changes'][:12]))


def require_base(repo, ref, *, protected_paths=()):
    """Require every runtime change on ref since the common ancestor.

    Candidate-only changes remain permitted. Missing upstream code is refused
    even if the candidate independently touched the same path. This does not
    relax frozen experiment revisions or serving/baseline build identities.
    """
    if not ref or ref.startswith('-') or any(c.isspace() for c in ref):
        raise ValueError('required base must be a literal Git ref')
    head = git(repo, 'rev-parse', '--verify', 'HEAD^{commit}').decode().strip()
    required = git(repo, 'rev-parse', '--verify', ref + '^{commit}').decode().strip()
    bases = git(repo, 'merge-base', '--all', head, required).decode().splitlines()
    if len(bases) != 1:
        raise ValueError('required base has no unique common ancestor')
    base = bases[0]
    before, after = tree(repo, base), tree(repo, required)
    pins = protected(repo, [*protected_paths, *wrapper_inputs()])
    relevant, omitted = [], []
    for path in sorted(before.keys() | after.keys()):
        if before.get(path) == after.get(path):
            continue
        records = [values[path] for values in (before, after) if path in values]
        (omitted if all(ignored(path, item['mode'], pins) for item in records) else relevant).append(path)
    details = dict(ok=not relevant, ref=ref, base=base, head=head, required=required,
                   ignored_changes=omitted, relevant_changes=relevant)
    if relevant:
        raise SourceMismatch(details)
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['identity', 'require-base'])
    parser.add_argument('ref', nargs='?')
    parser.add_argument('--repo', default='.')
    parser.add_argument('--input', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        if args.action == 'identity':
            result = identity(args.repo, protected_paths=args.input)
        else:
            if not args.ref:
                parser.error('require-base needs a ref')
            result = require_base(args.repo, args.ref, protected_paths=args.input)
        print(json.dumps(result, sort_keys=True))
    except SourceMismatch as exc:
        print(json.dumps(exc.details, sort_keys=True), file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print('source check refused: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
