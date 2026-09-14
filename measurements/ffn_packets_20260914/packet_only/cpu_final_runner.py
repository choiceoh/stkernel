import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from probes.engine_ffn_packets_compile import fingerprint

modules = (
    'test_engine_ffn_packets', 'test_engine_prefill_fp8_consumer',
    'test_engine_router_widths', 'test_engine_knobs', 'test_engine_execution_plans',
    'test_engine_graph_contracts', 'test_engine_native_execution',
    'test_glm53_modelopt_serving', 'test_glm53_modelopt_kernel_dispatch',
    'test_engine_kda_deferred_batch', 'test_engine_kda_ring',
    'test_engine_early_observe', 'test_engine_decode_bundle',
    'test_engine_draft_qk', 'test_engine_kernel_common',
    'test_engine_fleet_lease', 'test_engine_turn_retention', 'test_engine_draft_agreement',
    'test_fleet_onepass', 'test_fleet_onepass_integration',
)
records = []
out = Path('/out')
code = '''import json, sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromName('tests.'+sys.argv[1])
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps(dict(tests=result.testsRun, skips=len(result.skipped), failures=len(result.failures), errors=len(result.errors))))
raise SystemExit(not result.wasSuccessful())
'''
for module in modules:
    start = time.monotonic()
    result = subprocess.run([sys.executable, '-c', code, module], capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    (out/(module+'.log')).write_text(log)
    record = dict(module=module, returncode=result.returncode, seconds=time.monotonic()-start,
                  log_sha256=hashlib.sha256(log.encode()).hexdigest())
    try: record.update(json.loads(result.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError): record['summary_missing'] = True
    records.append(record)
    print(json.dumps(record), flush=True)
report = dict(status='PASS' if all(r['returncode']==0 for r in records) else 'FAIL',
              revision=os.environ['TEST_REVISION'], image=os.environ['ST_IMAGE'],
              gpu_used=False, scope='isolated Linux ARM64 CPU modules; CUDA hidden',
              source_sha256={**fingerprint(), **{'tests/'+m+'.py': hashlib.sha256(Path('/repo/tests/'+m+'.py').read_bytes()).hexdigest() for m in modules}}, modules=records)
(out/'cpu.json').write_text(json.dumps(report, indent=2)+'\n')
raise SystemExit(report['status'] != 'PASS')
