"""Read completed onepass diagnostic traces on srv2; never start a profiler.

Usage: python3 collect_launches.py /path/to/consumer.jsonl > consumer-launch-proof.json
Only launch counts are retained: profiler timings are not consumer measurements.
"""
import hashlib
import json
from pathlib import Path
import sys


def collect(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        for diagnostic in run.get('diagnostics', []):
            if not diagnostic.get('complete'):
                raise ValueError('incomplete diagnostic')
            source = Path(run['artifacts']) / diagnostic['phase'] / 'latency-summary.json'
            raw = source.read_bytes()
            operations = json.loads(raw)['operations']
            for rank in range(4):
                launches = {}
                for op in operations:
                    name = op.get('kernel') or ''
                    if (op['rank'] == rank and op['phase'] == 'decode'
                            and op['kind'] == 'gpu_activity'
                            and any(n in name.lower() for n in ('mk_mhc', 'mk_input_pack', 'mk_mla', 'moestatic', 'st_router_fused', 'st_rt_kernel'))):
                        launches[name] = launches.get(name, 0) + op['samples']
                if not launches:
                    raise ValueError(f'no selected launches for {source}, rank {rank}')
                records.append(dict(arm=run['name'], run_id=run['run_id'], rank=rank,
                    diagnostic=diagnostic['phase'], source=str(source),
                    source_sha256=hashlib.sha256(raw).hexdigest(), launches=launches))
    return dict(scope='Separate diagnostic replay; launch counts only, no consumer timing claims', records=records)


if __name__ == '__main__':
    print(json.dumps(collect(sys.argv[1]), indent=2))
