#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Conservative content identity for the reviewed, named CPU suite runner.

Generic commands never use this cache. Unknown or changed test programs fall
back to the complete tracked tree, including documentation and file modes.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

# Reviewed startup tests only execute extracted launcher blocks against local
# fakes. The digest pins that dependency audit; changing a test automatically
# expands the scope to the full tree until this registry is reviewed again.
STARTUP_AUDIT = {'tests/test_glm53_startup.py': '62cc4553b2661fc39d178100d29ce0540a38cecf924c66b09e4c8762a23a4503', 'tests/test_glm53_attestation.py': '24d0a73afea17bd5b91366d03d0bacc2008a4d371261589ee0d2e569c6ce3b9c', 'tests/test_glm53_reclaim.py': '5b38bfe90dc96e6a9144b9f4a35a6c69480f25511142a8b00232c52b36c9f57b', 'tests/test_memfree_preflight.py': 'af09f6e117c48865021038db64484fc89e685bdb76004fd4f914d86c9dc93869'}
# The completion-race test only uses the existing temporary Store/worker-lock
# fixture and bench module dependencies; the fleet dependency closure is kept.
FLEET_AUDIT = {'tests/test_fleet_coalescing.py': 'f93a92e2f60c265c0b9861824ecec4a2b4ea2aa325e4339088698cbf6d9ca98a', 'tests/test_fleet_experiments.py': '6013261b76a9a11d29923e3f741021b6fef33e03ef6224190c1549f59aa964f4', 'tests/test_fleet_feedback.py': 'b633b503d60ae005126c7dce3eeec30c147a292564f698e3a560af5a509fc92b'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memoized(memo, key, compute):
    if memo is None:
        return compute()
    if key not in memo:
        memo[key] = compute()
    return memo[key]


def identity(repo, spec, environment, memo=None):
    cmd = spec["command"]
    if len(cmd) < 4 or cmd[1] != "bench/cpu_checks.py":
        return None
    args = cmd[2:]
    if len(args) % 2 or any(args[i] not in {"--suite", "--test", "--contract"} for i in range(0, len(args), 2)):
        return None
    suites = sorted({args[i+1] for i in range(0, len(args), 2) if args[i] == '--suite'})
    if any(s not in {"startup", "fleet", "logic", "core", "sensitivity"} for s in suites):
        return None
    from cpu_contracts import CONTRACTS, dependencies
    contracts = sorted({args[i+1] for i in range(0,len(args),2) if args[i] == '--contract'})
    if any(c not in CONTRACTS for c in contracts):
        return None
    # Git object IDs cover transitive source dependencies without re-reading
    # all source bytes. No caller supplied include/exclude paths are accepted.
    tree = memoized(memo, ('tree',str(repo)), lambda: subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-rz", "HEAD"]))
    entries = [v.decode() for v in tree.split(b"\0") if v]
    scope = "full-tree"
    audit_names = list(CONTRACTS) if suites == ['sensitivity'] else contracts
    closure = dependencies(repo,audit_names) if audit_names else None
    if closure and '--test' not in args and (not suites or suites == ['sensitivity']):
        scope = 'audited-contracts'
        # Module path inventory also pins the loader's unique-name resolution.
        entries = [v if v.split('\t',1)[1] in closure or v.split('\t',1)[1].startswith('bench/')
                   else 'path\t'+v.split('\t',1)[1] for v in entries if v.split('\t',1)[1] in closure
                   or v.split('\t',1)[1].startswith(('bench/','overlay/'))]
    if not contracts and '--test' not in args and suites == ["startup"] and STARTUP_AUDIT and all(
            (repo / p).is_file() and sha(repo / p) == h for p, h in STARTUP_AUDIT.items()):
        scope = "audited-startup"
        entries = [v for v in entries if v.split("\t", 1)[1].startswith(
            ("tests/", "launchers/", "bench/", "profiles/")) or v.split("\t", 1)[1] == ".gitignore"]
    if (not contracts and '--test' not in args and suites == ['fleet'] and FLEET_AUDIT
            and {str(p.relative_to(repo)):sha(p) for p in (repo/'tests').glob('test_fleet*.py')} == FLEET_AUDIT):
        fleet_closure = dependencies(repo,list(CONTRACTS))
        if fleet_closure:
            scope = 'audited-fleet'
            prefixes = ('tests/','bench/','profiles/','launchers/')
            entries = [v if v.split('\t',1)[1] in fleet_closure or v.split('\t',1)[1].startswith(prefixes)
                       else 'path\t'+v.split('\t',1)[1] for v in entries
                       if v.split('\t',1)[1] in fleet_closure or v.split('\t',1)[1].startswith((*prefixes,'overlay/'))]
    env = dict(environment, **spec["env"])
    # Query the exact interpreter selected by this request, not this process's
    # Python. Installed wheel metadata, paths, interpreter and shell tools are
    # included; editable installs disable reuse (their sources are external).
    python = shutil.which(cmd[0], path=env.get("PATH"))
    if not python:
        return None
    code = '''import hashlib, importlib.metadata as m, json, platform, sys
rows=[]
for d in m.distributions():
 direct=d.read_text('direct_url.json') or '{}'
 if json.loads(direct).get('dir_info',{}).get('editable'): raise SystemExit(3)
 files=[]
 for f in d.files or []:
  if str(f).endswith('.pyc'): continue
  try:
   s=d.locate_file(f).stat()
   files.append((str(f),s.st_size,s.st_mtime_ns))
  except OSError: files.append((str(f),None,None))
 rows.append((d.metadata['Name'],d.version,str(d.locate_file('')),hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest(),hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest()))
print(json.dumps([sys.version,sys.executable,platform.platform(),sorted(rows)]))'''
    try:
        result = memoized(memo, ('runtime',python,json.dumps(env,sort_keys=True)),
                          lambda: subprocess.run([python, "-c", code], env=env, text=True, capture_output=True, timeout=20))
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    runtime = json.loads(result.stdout)
    tools = {}
    for name in (cmd[0], "bash", "/bin/bash", "git", "python3", "sh", "awk", "sed", "head", "tail", "timeout"):
        path = shutil.which(name, path=env.get("PATH"))
        tools[name] = [path, memoized(memo,('binary',path),lambda: sha(path))] if path else None
    data = dict(scope=scope, tree=entries, suites=suites, command=cmd, runtime=runtime,
                tools=tools, environment={k: v for k, v in env.items() if k != "SSH_AUTH_SOCK"},
                context=spec["context"], inputs={p: sha(p) for p in spec["inputs"]},
                timeout_s=spec["timeout_s"], resources=spec.get('resources'), outputs=spec.get('outputs', []))
    return dict(key=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                scope=scope, files=len(entries))
