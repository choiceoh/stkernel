"""Verify that the final CPU-only reference fallback does not alter packet math."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

folder = Path(__file__).resolve().parent
measured = 'b29b4083'
final = '22c993bc'
report = json.loads((folder/'gpu-v8.json').read_text())
records = {}
for path, before_hash in report['source_sha256'].items():
    before = subprocess.check_output(['git', 'show', measured+':'+path])
    after = subprocess.check_output(['git', 'show', final+':'+path])
    assert hashlib.sha256(before).hexdigest() == before_hash, path
    after_hash = hashlib.sha256(after).hexdigest()
    record = dict(measured_sha256=before_hash, final_sha256=after_hash)
    if after_hash == before_hash:
        record['status'] = 'byte-identical'
    else:
        old, new = ast.parse(before), ast.parse(after)
        if path == 'engine/profiles/glm53/lanes.py':
            function = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_reference_lanes')
            outer = next(n for n in function.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'swapped')
            guard = next(n for n in outer.body if isinstance(n, ast.If) and ast.unparse(n.test) == "'moe' in swapped")
            assert ast.unparse(guard.body[0]) == 'table = replace(table, moe_packets=None, moe_packets_supported=None)'
            assert len(guard.body) == 1 and not guard.orelse
            outer.body.remove(guard)
            record['status'] = 'only explicit reference-MoE fallback added; empty reference_for remains unchanged'
        elif path == 'tests/test_engine_ffn_packets.py':
            cls = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == 'PacketContractTests')
            case = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'test_reference_expert_selection_disables_its_packet_reader')
            cls.body.remove(case)
            record['status'] = 'only CPU reference-fallback test added'
        else:
            raise SystemExit(f'unreviewed source change: {path}')
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False), path
    records[path] = record
result = dict(measured_revision=subprocess.check_output(['git','rev-parse',measured],text=True).strip(),
              final_code_revision=subprocess.check_output(['git','rev-parse',final],text=True).strip(),
              status='PASS', unchanged_files=sum(v['status']=='byte-identical' for v in records.values()),
              scope='same packet kernels and ordinary/reference_for-empty control path; reference fallback has CPU proof, no new GPU timing claim',
              sources=records)
(folder/'runtime_continuity_final.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: packet math and measured ordinary binding unchanged; explicit reference fallback is the only behavior added')
