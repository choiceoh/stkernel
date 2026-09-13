"""Recompute the retained L7 summary from immutable scored request records.

This is CPU postprocessing of a historical, quality-failing run, not a new
engine measurement. Neither the retained archive nor the queued L8 is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench'))
from onepass import prefill_summary
from st_judge import prefill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive = ROOT / 'measurements/st_prefill_phase2_20260913/gpu-L7-qualification/operator-stop.tar.gz'
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive_hash == '7b55e97aeba0ddcda785533673c00114146a1e23c05cfe7246e931c716add669'
    with tarfile.open(archive) as source:
        record = json.load(source.extractfile('run/record-before-user-cancellation.json'))
        raw = [json.loads(line) for line in source.extractfile('run/requests.jsonl') if line.strip()]
    by_hash = {r['request_sha256']: r for r in raw}
    assert len(by_hash) == len(raw)
    report = dict(scope='historical CPU summary correction only; no new GPU result',
                  archive_sha256=archive_hash, candidate=record['arm_sha'],
                  run_id=record['run_id'], contexts=[], quality_unchanged=record['quality'])
    for ctx, count in ((2000, 3), (32000, 1), (128000, 1)):
        requests = [r for r in record['requests'] if r['ctx'] == ctx and not r.get('fixed_decode')]
        assert len(requests) == count
        for request in requests:
            original = by_hash[request['request_sha256']]
            assert original['phase'] == 'measure-c1' and original['cached_tokens'] == 0
            assert all(original[key] == request[key]
                       for key in ('prompt_tokens', 'ttft_s', 'output_sha256', 'cached_tokens'))
        before = next(r for r in record['prefill'] if r['ctx'] == ctx)
        after = prefill_summary([(r['prompt_tokens'], r['ttft_s']) for r in requests], 0)
        assert prefill(record, ctx, 'cold_tok_s') == after['cold_tok_s']
        assert prefill(record, ctx, 'warm_tok_s') == after['warm_tok_s']
        report['contexts'].append(dict(ctx=ctx, before=before, after=after,
            request_sha256=[r['request_sha256'] for r in requests]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps([dict(ctx=r['ctx'], old_first_tok_s=r['before']['cold_tok_s'],
                          paired_first_tok_s=r['after']['cold_tok_s'],
                          paired_median_tok_s=r['after']['warm_tok_s'])
                      for r in report['contexts']], indent=2))


if __name__ == '__main__':
    main()
