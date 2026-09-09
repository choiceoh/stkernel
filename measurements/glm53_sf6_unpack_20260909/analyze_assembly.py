"""Count retained straight-line isolated SASS, never infer runtime speed."""
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'codegen-v3' / 'sf-unpack-codegen'
receipt = json.loads((OUT / 'result.json').read_text())
assert receipt['status'] == 'PASS' and receipt['cuda_initialized'] is False
assert sorted((c['words'], c['arm']) for c in receipt['cases']) == [
    (n, arm) for n in (1, 4, 8) for arm in ('scalar', 'u8x4')]
cases = []
for case in receipt['cases']:
    for artifact in case['artifacts'].values():
        data = (OUT / artifact['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact['sha256']
    sass = (OUT / case['artifacts']['sass']['path']).read_text()
    instructions = re.findall(r'^\s*/\*[0-9a-f]+\*/\s+([A-Z][A-Z0-9_.]*)\b', sass, re.M)
    assert instructions.count('EXIT') == 1
    end = instructions.index('EXIT') + 1
    assert not any(op in ('BRA', 'CALL', 'RET') for op in instructions[:end])
    assert set(instructions[end:]) <= {'BRA', 'NOP'}, 'unexpected code after exit'
    cases.append(dict(words=case['words'], arm=case['arm'], instructions=end,
                      opcodes=dict(collections.Counter(instructions[:end]))))
comparisons = []
for words in (1, 4, 8):
    arms = {c['arm']: c['instructions'] for c in cases if c['words'] == words}
    comparisons.append(dict(words=words, **arms,
                            reduction_pct=100 * (1 - arms['u8x4'] / arms['scalar'])))
result = dict(evidence='isolated-unpack-compile-only',
              method='SASS instructions through EXIT inclusive; unreachable BRA/NOP padding excluded',
              scope='Includes identical source wrapper loads/stores and compiler-selected instructions; not full MoE kernel or runtime timing',
              cases=cases, comparisons=comparisons)
(ROOT / 'assembly-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(comparisons, indent=2))
