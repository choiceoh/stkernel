"""Use one admitted GPU hold for all candidates, isolating failed processes."""
import json
from pathlib import Path
import subprocess
import sys
import time


def check(ranks, *, bundle="capacity_bundle"):
    rows = []
    root = Path(__file__).resolve().parents[1]
    bundles = {
        'capacity_bundle': [('moe_batch', 240), ('moe_stage_fc1_shared', 180), ('moe_stage_fc2', 180),
                            ('router_batch', 120), ('mhc_batch', 180), ('moe_raw_scale', 180)],
        'scatter_bundle': [('moe_route_scatter', 300), ('moe_direct_scatter', 300),
                           ('moe_route_direct', 300), ('paired_projection', 180), ('shared_serial', 180)],
    }
    lanes = bundles[bundle]
    for lane, seconds in lanes:
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
