#!/usr/bin/env python3
"""Reconcile archived CPU15 bytes and receipts without importing GPU modules."""
import hashlib
import itertools
import json
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parent
    result = json.loads((root/'result.json').read_text())
    submission = json.loads((root/'head/submission.json').read_text())
    exit_receipt = json.loads((root/'head/exit.json').read_text())
    capture = json.loads((root/'capture.json').read_text())
    assert submission['revision'] == '71c407bbab59004a9a865acc17f3d2b5c48013c8'
    assert result['verdict'] == 'FAIL' and result['phase'] == 'micro-cute-compile' and 'error' in result
    assert 'contracts' not in result and result['micro_passes'] == []
    assert '(Int32(?), Int32(?)) to integer conversion is not supported' in result['error']
    assert 'cuda_initialized' not in result and 'binding_runtime_rechecked' not in result
    assert exit_receipt['returncode'] == exit_receipt['payload_returncode'] == 1
    assert exit_receipt['copy_returncode'] is None
    for key in ('mounted_sources', 'contract_sources'):
        assert result[key] == submission['worker_sources'][key]
        for phase in ('before', 'after'):
            assert result[key] == capture[phase]['worker']['source_receipts'][key]
    archive = root/'evidence.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == capture['evidence_tar_sha256']
    files = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            assert not member.issym() and not member.islnk()
            if member.isdir():
                continue
            assert member.isfile() and member.name.startswith('evidence/')
            relative = member.name[len('evidence/'):]
            assert '..' not in Path(relative).parts and relative not in files
            files[relative] = tar.extractfile(member).read()
    identities = {name:dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                  for name, raw in files.items()}
    assert len(files) == 1 and files['result.json'] == (root/'result.json').read_bytes()
    for phase in ('before', 'after'):
        snap = capture[phase]
        assert identities == snap['worker']['files']
        assert snap['head_evidence'] is None
        for node in (snap, snap['worker']):
            assert node['revision'] == submission['revision'] and not node['status']
            assert node['shallow'] == 'false' and node['alternates'] is False
        assert snap['worker']['image_id'] == submission['image']
        capsule = snap['worker']['capsule']
        assert capsule['manifest_sha256'] == submission['manifest_sha256']
        assert capsule['strict_validation'] == 'PASS' and capsule['files'] == 116
        for name, expected in snap['files'].items():
            raw = (root/'head'/name).read_bytes()
            assert dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == expected
    assert set(files) == {'result.json'}
    report = dict(verdict='PASS', scope='CPU15 archive integrity only; no new tests or GPU execution',
                  original_verdict=result['verdict'], contracts=None,
                  original_files=len(files), ptx=0, cubin=0, resource_logs=0,
                  source_revision=submission['revision'],
                  evidence_tar_sha256=capture['evidence_tar_sha256'],
                  result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
