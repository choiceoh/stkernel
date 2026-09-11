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
# Memory-reclaim and pinned-budget cases execute the emitted/extracted shell
# only through local fake commands; their inputs remain in this source closure.
STARTUP_AUDIT = {'tests/test_glm53_startup.py': '62cc4553b2661fc39d178100d29ce0540a38cecf924c66b09e4c8762a23a4503', 'tests/test_glm53_attestation.py': '24d0a73afea17bd5b91366d03d0bacc2008a4d371261589ee0d2e569c6ce3b9c', 'tests/test_glm53_reclaim.py': '5b38bfe90dc96e6a9144b9f4a35a6c69480f25511142a8b00232c52b36c9f57b', 'tests/test_memfree_preflight.py': 'c999b616eae42ce564529942af6801b112bb379d1bac7de85028cb5601fda4eb', 'tests/test_glm53_memory_preflight.py': '44f00db9836d2669de01dc1062d913d597a9d9b80c701e18b968613c10f92710'}
# Fleet cases use per-test temporary repositories, stores, queues and timing
# profiles. Queued controller paths and bare-wait ownership fixtures retain
# the same temporary-file and bench dependency closure (re-audited 28e5e57b).
# Local/remote GPU observation and foreign-residency cases use only existing
# bench code and stdlib fakes; they retain the temporary fixture closure.
# The deploy-ref CLI regression reads launchers/deploy-overlays.sh and invokes
# bench/fleet_source.py in a temporary Git fixture; both prefixes are retained.
FLEET_AUDIT = {'tests/test_fleet_admission.py': 'd452bec3a6ae34be36547174f956d27a4bfe72c341627dec400c470c20c7f3e0', 'tests/test_fleet_approval.py': '769685a2ef184c4c29689242662f7f08c838a6fa1e815c34760c294d1a20b508', 'tests/test_fleet_cache_explain.py': '911f1bf7bc4678384dbb38514254fda81ebfbf56e43c0e8e229de123fb689d45', 'tests/test_fleet_classify.py': '371547d98b361a13da2874a47b23cca830718ab5116bbee291bb576660cc17ba', 'tests/test_fleet_cleanup.py': '3ec7cba8da87eaac9a78df418c1201aafeda73150859d4de9bdf3958d7bfab93', 'tests/test_fleet_coalescing.py': '7477e284ec430b90feb0ee722af4958f4fdb58656aa9a01eb03844e6218655b9', 'tests/test_fleet_experiments.py': 'a7fc94693a4b705d3c204bd13f1704522ab8a92f56f1aee64f488ef12c736cd2', 'tests/test_fleet_feedback.py': 'f3495b30e2d83edbbe01737f824271dcf182253829e4bc605bf84dddd957560e', 'tests/test_fleet_handoff.py': 'd2f0d4c62b6a62ca8452239b4ca7d8e4fbe86ad6d50c337642fe47257f4c8e6d', 'tests/test_fleet_history.py': '609b5eeeffe427f3ac7dc64f33c1addfe5d9e613cc15730ce2cdbf120854dcd8', 'tests/test_fleet_idle.py': 'a5ec3f8d3ac44c60805022e38009669a5221a1383a2b5445373630665d118b4b', 'tests/test_fleet_inspect.py': '68c415d524c4397fdfe3d93e16fecdb0f1a79dfe67f3c76be08511fc80185f6e', 'tests/test_fleet_launch.py': 'd47e87c939e2ad6b523f5bc2cb7301418fa5f7fab047c5c03349e1c095c0ac63', 'tests/test_fleet_onepass.py': '962e299db8796226cb98fec011532104de39bdd3813fc0eed6e3bcc398832fa5', 'tests/test_fleet_onepass_integration.py': '72cddc8ecdd62c1dddd9629772511ec90dd1ba36babf7c512610099753af05ea', 'tests/test_fleet_pause.py': 'b5b414500418b330f5dfd72ddce71a37ee4a14dbfacad961c3898e8ac932c6a0', 'tests/test_fleet_pending.py': '0c2cf8a389461e6a901bbdf0fbbcfe5b7590f8402442b396184f5b44e4431792', 'tests/test_fleet_prepare.py': '4978a59e84c4119cdcd3f69ee97cc54bc0ebc56d8c15dc1ca6b6fa9c3bcbc402', 'tests/test_fleet_recovery.py': 'b994b03e073637ea4c4e13838163701731853683bf180702c6b6676fd4809356', 'tests/test_fleet_retry.py': 'a2f1bef4bf260f2888e4574922ed4027a0b9c337cf98156f4851a82b990e4e59', 'tests/test_fleet_runtime.py': 'ce973514d6307256872c1c23ecf0bc9208d1750853bbe2b202fed19222c3c810', 'tests/test_fleet_source.py': '4700e7f4dc853bf48441a5efb13e278afd390e2dda6fed08ea00f841c6630c51', 'tests/test_fleet_validation.py': '5f0f8c89aee50950121995e79bf7dd9c7ab3d2a15a172bae975620146e45eba9'}


# Complete logic/deployment gates read profile/module READMEs and the campaign
# runbook, all retained by fleet_source. Merged SF6, boot-memory, EP warmup
# and structured proof assertions retain the same source closure. PR #508
# adds one profiles/glm53.env assertion and its explicit test invocation.
# PR #509 only parses the already-read boot-stamp source to compare its phase
# table with TARGETS; it introduces no file, import or execution dependency.
# The startup-only memory-gate registration leaves the logic command and
# dependencies unchanged. Changed test or runner disables pruning.
LOGIC_SOURCE_AUDIT = {'tests/test_logic.py': '40e681a1e7a3c5e99e1c14b6df4105395c75cb154e5ae52b77cc0ff881d5f244', 'tests/test_glm53_overlay_sync.py': 'bafa1e1e9ebcb964d36f08030f5bcd9309403b02dc2684f604c40a62b8c2563a', 'bench/cpu_checks.py': 'f61b906b52d1e288ac9c70fb286f3fffeedbd2337a324c82c13e713a94fd4efd', 'bench/cpu_unittest.py': 'd61adde7318ade385523c2a3aca9b404e5b4bbc5e363cffb6bf58656d12bfade'}


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
    all_entries = entries
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
            prefixes = ('tests/','bench/','profiles/','launchers/','probes/')
            entries = [v if v.split('\t',1)[1] in fleet_closure or v.split('\t',1)[1].startswith(prefixes)
                       else 'path\t'+v.split('\t',1)[1] for v in entries
                       if v.split('\t',1)[1] in fleet_closure or v.split('\t',1)[1].startswith((*prefixes,'overlay/'))]
    selected_tests = sorted({args[i+1] for i in range(0, len(args), 2) if args[i] == '--test'})
    if (not contracts and suites == ['logic'] and selected_tests in ([], ['tests/test_glm53_overlay_sync.py'])
            and all((repo / p).is_file() and sha(repo / p) == h for p, h in LOGIC_SOURCE_AUDIT.items())):
        scope = 'audited-logic'
    # Only an already reviewed closure may omit retained prose/output files.
    # Full-tree fallback remains exact when a test or dependency audit changes.
    if scope != 'full-tree':
        from fleet_source import ignored, protected
        pins = protected(repo, spec['inputs']) | set(closure or ())
        # A caller-declared input is never lost by an earlier suite projection;
        # retain its Git mode as well as the separately hashed input bytes.
        included = {v.split('\t', 1)[1] for v in entries}
        entries = entries + [v for v in all_entries if v.split('\t', 1)[1] not in included and
                            any(v.split('\t', 1)[1] == p or v.split('\t', 1)[1].startswith(p.rstrip('/') + '/')
                                for p in pins)]
        entries = [v for v in entries if not ignored(v.split('\t', 1)[1],
                    v.split(' ', 1)[0], pins)]
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
