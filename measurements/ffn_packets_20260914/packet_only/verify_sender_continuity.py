"""Check which parts of the measured component changed in the main merge.

This is source continuity, not a second GPU run or four-rank serving proof.
The measured probe uses FP8Linear and emulates communication with torch.cat;
it does not enter the fleet boot or the BF16/W4 C1 bound-input kernel.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

folder = Path(__file__).resolve().parent
measured = '894a16f45d97f33ad434e75e28cff8c43a3d66f3'
integrated = 'adb6eb52e78d9cb0fc1dc075c49221e2d21bab65'
report = json.loads((folder/'gpu-sender-v2.json').read_text())


def source(revision, path):
    return subprocess.check_output(['git', 'show', revision+':'+path])


changed = {}
unchanged = []
for path, expected in report['source_sha256'].items():
    before, after = source(measured, path), source(integrated, path)
    assert hashlib.sha256(before).hexdigest() == expected, path
    if before == after:
        unchanged.append(path)
    else:
        changed[path] = dict(before_sha256=expected, after_sha256=hashlib.sha256(after).hexdigest())
assert set(changed) == {'engine/kernels/dense/__init__.py', 'engine/profiles/glm53/boot.py'}

path = 'engine/kernels/dense/__init__.py'
trees = [ast.parse(source(rev, path)) for rev in (measured, integrated)]
for tree in trees:
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bound_input_cell')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, 'exec'), namespace)
    for rows in (8193, 8194, 8195, 9216, 32768):
        assert not namespace['bound_input_cell'](rows, 1024, 4096)
        assert not namespace['bound_input_cell'](rows, 4096, 512)
    tree.body.remove(function)
assert ast.dump(trees[0], include_attributes=False) == ast.dump(trees[1], include_attributes=False)
changed[path]['change'] = 'Only bound_input_cell at eight rows; all FP8Linear and packet reader code unchanged'
changed['engine/profiles/glm53/boot.py']['change'] = (
    'Main one-shot rail/inline boot configuration; fleet boot is outside this component probe')

result = dict(status='PASS', measured_revision=measured, integrated_revision=integrated,
    scope='source continuity of the measured local component; not a new GPU or serving result',
    unchanged_source_files=unchanged, changed_source_files=changed,
    serving_qualification_pending=True)
(folder/'sender_runtime_continuity.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS: measured sender/router/expert/shared implementations survive the main merge')
