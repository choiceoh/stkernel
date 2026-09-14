"""Verify the FFN evidence boundary after main's unrelated decode-pool merge."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
repo = root.parents[1]
before = '153b6f8bdbb9ebf18af78c79bb9173f96a6769c2'
after = 'cc6ccf8bbbb82bdf5b8e432a887e8e08d0888f81'


def source(rev, path):
    return subprocess.check_output(['git', 'show', rev + ':' + path], cwd=repo)


def dump(node):
    return ast.dump(node, include_attributes=False)


gpu = json.loads((root / 'gpu.json').read_text())
changed = [p for p, sha in gpu['source_sha256'].items()
           if hashlib.sha256(source(after, p)).hexdigest() != sha]
assert set(changed) == {'engine/profiles/glm53/net.py', 'engine/profiles/glm53/lanes.py'}
compile_report = json.loads((root / 'compile.json').read_text())
assert all(hashlib.sha256(source(after, p)).hexdigest() == sha
           for p, sha in compile_report['source_sha256'].items())

# Only the DSA preparation guard and pool-id reader change in the network.
net_trees = [ast.parse(source(rev, 'engine/profiles/glm53/net.py')) for rev in (before, after)]
for tree in net_trees:
    net = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Glm53Net')
    net.body = [n for n in net.body if not isinstance(n, ast.FunctionDef)
                or n.name not in ('prepare_decode_dsa_inputs', '_select_rows')]
assert dump(net_trees[0]) == dump(net_trees[1])

lanes = [ast.parse(source(rev, 'engine/profiles/glm53/lanes.py')) for rev in (before, after)]
old, new = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'served')
            for tree in lanes]
helper = copy.deepcopy(next(n for n in lanes[1].body
                           if isinstance(n, ast.FunctionDef) and n.name == '_apply_reference_lanes'))
swapped = next(n for n in helper.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'swapped')
assert len(swapped.body) == 2
assert ast.unparse(swapped.body[-1].test) == "'kpool_compress' in swapped and table.decode_rows is not None"
swapped.body.pop()  # New reference-only direct-pool replacement, separately CPU tested.
assert ast.unparse(new.body[-1]) == 'return _apply_reference_lanes(table, ref, reference_for)'
new.body[-1:] = helper.body
counts = dict(imports=0, bindings=0)
for node in ast.walk(new):
    if isinstance(node, ast.ImportFrom) and node.module == 'engine.kernels.kpool':
        assert node.names[-1].name == 'compress_decode_pools'
        node.names.pop()
        counts['imports'] += 1
    if isinstance(node, ast.Tuple) and len(node.elts) >= 2 and all(isinstance(n, ast.Name) for n in node.elts):
        if [n.id for n in node.elts[-2:]] == ['update_pool_cache', 'compress_decode_pools']:
            node.elts.pop()
            counts['bindings'] += 1
assert counts == dict(imports=1, bindings=1)
# Includes every MoE, packet, compact-state and mixed-preparation statement.
assert dump(old) == dump(new)
for tree in lanes:
    tree.body = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 or n.name not in ('served', '_apply_reference_lanes', 'DecodeRows', 'reference_decode_rows')]
assert dump(lanes[0]) == dump(lanes[1])

cpu = json.loads((root / 'cpu_postmerge.json').read_text())
assert cpu['status'] == 'PASS'
assert all(hashlib.sha256(source(after, p)).hexdigest() == sha for p, sha in cpu['source_sha256'].items())
result = dict(status='PASS', gpu_source_revision=before, integration_revision=after,
    integrated_main='6522564a', changed_gpu_manifest_files=changed,
    byte_identical_gpu_manifest_files=len(gpu['source_sha256']) - len(changed),
    byte_identical_compiler_manifest_files=len(compile_report['source_sha256']),
    network_changes_outside_ffn=['prepare_decode_dsa_inputs', '_select_rows'],
    served_ffn_ast_unchanged_after_reviewed_decode_only_normalization=True,
    cpu_postmerge=dict(tests=cpu['tests'], passed=cpu['passed'], skipped=cpu['skipped']),
    note='No new GPU timing is claimed. Main adds a direct pool reader and extracts reference-lane selection; '
         'the FFN kernels, preparation, router, shared readers and served FFN binding statements are retained.')
(root / 'postmerge_continuity.json').write_text(json.dumps(result, indent=2) + '\n')
print('Postmerge FFN source continuity and focused CPU integration pass.')
