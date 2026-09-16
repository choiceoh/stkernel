"""CPU validation of the real rank snapshot files; no native replay verdict."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    root = args.directory
    manifest = json.loads((root / 'manifest.json').read_text())
    assert digest(root / 'state.pt') == manifest['state_sha256']
    state = torch.load(root / 'state.pt', map_location='cpu', weights_only=True, mmap=True)
    assert state['rank'] == manifest['rank'] and state['world'] == manifest['world'] == 4
    assert len(state['dense']) == 30
    assert state['checkpoint_sha256'] == manifest['checkpoint_sha256']
    result = dict(rank=state['rank'], world=state['world'], torch=state['torch'],
                  state_sha256=manifest['state_sha256'], state_bytes=(root / 'state.pt').stat().st_size,
                  checkpoint_sha256=manifest['checkpoint_sha256'], readers=sorted(state['dense']),
                  source_sha256=state['source_sha256'], cases=[])
    files = sorted(root.glob('case-*.pt'))
    assert len(files) == len(list(root.glob('case-*.json'))) == 16
    for path in files:
        label = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == label['snapshot_sha256']
        case = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert case['state_sha256'] == manifest['state_sha256']
        assert case['case_id'] == label['case_id'] == path.stem
        assert case['temperature'] == 0 and label['label_source'] == 'committed_greedy_continuation'
        assert label['complete'] and len(label['target']) == len(case['baseline_drafts']) == case['k']
        assert torch.isfinite(case['ring']).all() and torch.isfinite(case['embedding']).all()
        result['cases'].append(dict(case_id=case['case_id'], seq=case['seq'], request_key=case['request_key'],
                                   start=case['start'], position=case['position'], k=case['k'],
                                   baseline_drafts=case['baseline_drafts'], target=label['target'],
                                   snapshot_sha256=label['snapshot_sha256'], bytes=path.stat().st_size,
                                   ring_shape=list(case['ring'].shape), embedding_shape=list(case['embedding'].shape)))
    result['cuda_initialized'] = torch.cuda.is_initialized()
    assert not result['cuda_initialized']
    result['native_replay_executed'] = False
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
