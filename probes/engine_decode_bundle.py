"""Use one admitted GPU hold for all candidates, isolating failed processes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


# D11: these private candidates expire after this qualification campaign.
EXPIRES = datetime(2026, 9, 16, tzinfo=timezone.utc)


def require_current_probe():
    if datetime.now(timezone.utc) >= EXPIRES:
        raise RuntimeError('decode scatter campaign expired; promote or remove the measured candidates')


def check(ranks, *, bundle="scatter_bundle"):
    require_current_probe()
    rows = []
    root = Path(__file__).resolve().parents[1]
    bundles = {
        'scatter_bundle': [('moe_route_scatter', 240), ('moe_direct_scatter', 240),
                           ('moe_route_direct', 240)],
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
