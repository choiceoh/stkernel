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
        'k7_commit_bundle': [('kda_commit', 300), ('decode_k7', 300)],
        'k7_output_bundle': [('kda_commit', 300), ('decode_k7', 300), ('moe_output', 300)],
        'batch_fusions': [('paired_projection', 180), ('indexer_boundary', 180), ('wide_input', 480)],
        'batch_integration': [('paired_projection', 180), ('indexer_boundary', 180), ('wide_input', 480),
                              ('direct_producer', 240)],
        'batch_boundaries': [('indexer_boundary', 180), ('wide_input', 480)],
        'scatter_bundle': [('moe_route_scatter', 240), ('moe_direct_scatter', 240),
                           ('moe_route_direct', 240)],
    }
    lanes = bundles[bundle]
    for lane, seconds in lanes:
        if lane == 'kda_commit':
            command = [sys.executable, '-u', str(root/'probes/engine_kda_deferred_check.py'),
                       '--commit-only', '--samples', '12', '--output', '/cache/k7-commit-bundle.json']
        else:
            command = [sys.executable, '-u', str(root/'probes/engine_kernel_check.py'), '--lanes', lane]
        if ranks and lane != 'kda_commit':
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
