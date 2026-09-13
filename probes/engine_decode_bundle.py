"""Use one admitted GPU hold for all candidates, isolating failed processes."""
import json
from pathlib import Path
import subprocess
import sys
import time


def check(ranks):
    rows = []
    root = Path(__file__).resolve().parents[1]
    for lane, seconds in [('mhc_single', 240), ('moe_waves', 300),
                          ('input_pack', 180), ('short_gemm', 180), ('shared_direct', 180)]:
        command = [sys.executable, '-u', str(root/'probes/engine_kernel_check.py'), '--lanes', lane]
        if ranks:
            command += ['--ranks', ranks]
        start = time.monotonic()
        print(json.dumps(dict(bundle_lane=lane, status='starting', timeout_s=seconds)), flush=True)
        try:
            result = subprocess.run(command, cwd=root, timeout=seconds)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        row = dict(bundle_lane=lane, exit_code=code, elapsed_s=time.monotonic()-start)
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps(dict(decode_bundle=rows, passed=all(row['exit_code']==0 for row in rows),
                          scope='component numerics and timing; no serving default enabled')), flush=True)
    if any(row['exit_code'] for row in rows):
        raise RuntimeError('one or more decode candidates failed; retain other component results')
