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

# The reviewed-closure registries (STARTUP_AUDIT, FLEET_AUDIT, LOGIC_SOURCE_AUDIT)
# retired with the vLLM overlay stack they audited (2026-09-18): their startup,
# startup-memory and overlay-logic subjects were deleted, and the fleet tests they
# pinned changed in the same decommission. Until a registry is reviewed again,
# every command resolves at full-tree scope, which remains exact.


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
    if len(args) % 2 or any(args[i] not in {"--suite", "--test"} for i in range(0, len(args), 2)):
        return None
    suites = sorted({args[i+1] for i in range(0, len(args), 2) if args[i] == '--suite'})
    if any(s not in {"fleet"} for s in suites):
        return None
    # Git object IDs cover transitive source dependencies without re-reading
    # all source bytes. No caller supplied include/exclude paths are accepted.
    tree = memoized(memo, ('tree',str(repo)), lambda: subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-rz", "HEAD"]))
    entries = [v.decode() for v in tree.split(b"\0") if v]
    scope = "full-tree"
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
    # Keep component fingerprints for explanations without storing environment
    # values or installed package metadata in user-facing cache-miss reports.
    components = {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                  for k, v in data.items() if k != 'tree'}
    dependencies = {v.split('\t', 1)[1]: v.split('\t', 1)[0] for v in entries}
    return dict(key=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                scope=scope, files=len(entries), dependencies=dependencies, components=components)


def difference(expected, current):
    """Explain exact cache identity changes; historical opaque keys stay opaque."""
    if not expected or not current:
        return dict(equal=False, changed_paths=[], changed_components=[],
                    reason='CPU command or environment is not eligible for evidence reuse')
    before, after = expected.get('dependencies'), current.get('dependencies')
    if before is None or after is None:
        return dict(equal=expected['key'] == current['key'], changed_paths=[], changed_components=[],
                    reason='Older CPU evidence has no component fingerprints')
    left, right = expected.get('components', {}), current.get('components', {})
    return dict(equal=expected['key'] == current['key'],
                changed_paths=sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p)),
                changed_components=sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k)))
