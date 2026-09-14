"""Check report manifests against their frozen commits, including rejected runs."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
repo = root.parents[1]
revision = '153b6f8bdbb9ebf18af78c79bb9173f96a6769c2'
reports = {name: revision for name in ('cpu.json', 'cpu_benchmark.json', 'compile.json', 'gpu.json')}
reports.update(profile_before='3275c783', gpu_alignment_failure='77472b7d')
result = dict(implementation_revision=revision,
    merged_main_revision=subprocess.check_output(['git', 'rev-parse', '2ac7de6f'], cwd=repo, text=True).strip(),
    cpu_compiler_image='sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc',
    gpu_image='sha256:8190d08e822e1f9d18dda5a127a5d9a9e8c53ef4b5136d1154f9ea4b7727f1ed',
    gpu_ticket='17893566183296004',
    gpu_source_tree='/home/choiceoh/st-worktrees/codex-gb10-mixed-prepare3', reports={})
for name, source in reports.items():
    name = name if name.endswith('.json') else name + '.json'
    path = root / name
    report = json.loads(path.read_text())
    manifest = report['source_sha256']
    for filename, expected in manifest.items():
        content = subprocess.check_output(['git', 'show', source + ':' + filename], cwd=repo)
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f'{name}: {filename} differs from frozen source {source}')
    result['reports'][name] = dict(
        source_revision=subprocess.check_output(['git', 'rev-parse', source], cwd=repo, text=True).strip(),
        status=report['status'], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        manifest_entries=len(manifest), source_matches=True)
result['cpu_runner_sha256'] = hashlib.sha256((root / 'cpu_runner.py').read_bytes()).hexdigest()
result['note'] = ('Final CPU/compiler/GPU evidence shares the implementation commit. '
    'The earlier profile is attribution only; the alignment failure is retained and excluded from performance tables. '
    'The later main decode-pool integration is checked separately in postmerge_continuity.json; '
    'whole-head file identity is not claimed after that merge.')
(root / 'source_identity.json').write_text(json.dumps(result, indent=2) + '\n')
print('All report source manifests match their frozen commits.')
