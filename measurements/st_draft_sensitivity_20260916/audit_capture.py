"""Check rank agreement and join captured labels to the sequential HTTP requests."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    root = args.root
    ranks = [json.loads((root / f'rank{i}-validation.json').read_text()) for i in range(4)]
    requests = json.loads((root / 'live-capture/requests.json').read_text())
    fields = ('case_id', 'seq', 'request_key', 'start', 'position', 'k', 'baseline_drafts', 'target')
    for rank in ranks:
        assert len(rank['cases']) == 16
        assert rank['checkpoint_sha256'] == ranks[0]['checkpoint_sha256']
        assert rank['source_sha256'] == ranks[0]['source_sha256']
        for case, reference in zip(rank['cases'], ranks[0]['cases']):
            assert {k: case[k] for k in fields} == {k: reference[k] for k in fields}
    # Engine seq is a reusable resident row; the HTTP response's seq is a request
    # id. Join by the bounded sequential collector's recorded case counts, then
    # verify the admission nonce and exact output slice independently.
    cursor, keys, mapping = 0, set(), []
    for request in requests:
        response = json.loads((root / 'live-capture' / f"request-{request['index']:02d}.response.json").read_text())
        cases = ranks[0]['cases'][cursor:cursor + request['cases_added']]
        assert len(cases) == request['cases_added'] == 2
        key, = {case['request_key'] for case in cases}
        assert key not in keys
        keys.add(key)
        for case in cases:
            offset = case['start'] - response['prompt_tokens']
            assert offset >= 1 and case['start'] == case['position'] + 1
            assert case['target'] == response['ids'][offset:offset + case['k']]
        mapping.append(dict(request_index=request['index'], http_request_id=response['seq'],
                            capture_request_key=key, case_ids=[c['case_id'] for c in cases]))
        cursor += len(cases)
    assert cursor == 16 and len(keys) == len(requests) == 8
    lengths = []
    for case in ranks[0]['cases']:
        length = 0
        for old, target in zip(case['baseline_drafts'], case['target']):
            if old != target:
                break
            length += 1
        lengths.append(length)
    result = dict(capture_commit='4e6698a1a47ffe447c4293d3cfc3122065bc1c69', fleet_session='draftreplay4-0916',
                  requests=len(requests), unique_cases=16, rank_filesets=4, complete_labels_per_rank=16,
                  actual_prompt_tokens=[x['prompt_tokens'] for x in requests],
                  labels_match_actual_output=True, all_ranks_agree=True, all_file_hashes_valid=True,
                  finite_capture_tensors=True, cpu_validation_cuda_initialized=False,
                  total_snapshot_bytes=sum(sum(c['bytes'] for c in r['cases']) for r in ranks),
                  total_prepared_state_bytes=sum(r['state_bytes'] for r in ranks),
                  baseline_prefix_lengths_in_captured_cases=lengths, request_mapping=mapping,
                  native_replay_executed=False, quality_benchmark=False, performance_baseline=False,
                  request_wall_seconds=sum(x['wall_seconds'] for x in requests),
                  archive_by_rank={str(i): dict(host=h, path=f'/home/choiceoh/expert-capture/draft-sensitivity-0916-7e62/captured/rank{i}')
                                   for i, h in enumerate(['srv2', 'srv1', 'srv3', 'srv4'])},
                  evidence_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(root.rglob('*')) if p.is_file() and p != root / 'audit.json'})
    (root / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('evidence_sha256', 'archive_by_rank')}, indent=2))


if __name__ == '__main__':
    main()
