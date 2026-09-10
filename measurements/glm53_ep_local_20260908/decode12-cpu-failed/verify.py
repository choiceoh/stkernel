#!/usr/bin/env python3
"""Reconcile archived CPU12 bytes and receipts without importing GPU modules."""
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
    assert submission['revision'] == 'cb3f5c754a7c4a65a89affd0c436f4ea570014b0'
    assert result['verdict'] == 'FAIL' and result['phase'] == 'cpu-contracts' and 'error' in result
    assert result['contracts'] == dict(tests_run=82, failures=1, errors=0, skips=0)
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
    assert len(files) == 61 and files['result.json'] == (root/'result.json').read_bytes()
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
    assert len(result['micro_keys']) == 3
    assert [key[10] for key in result['micro_keys']] == [[32, 128], [64, 128], [64, 128]]
    assert [key[17] for key in result['micro_keys']] == [72, None, None]
    assert [key[7:9] for key in result['micro_keys']] == [[8,64],[1,8],[8,64]]
    assert [(key[-1] == 'glm53_ep_micro_scatter_fp32_v1') for key in result['micro_keys']] == [True,True,False]
    passes = result['micro_passes']
    assert [row['arm'] for row in passes] == ['m32-topk8-fp32','m64-topk1-fp32','m64-topk8-bf16']
    assert [row['cache_key'] for row in passes] == result['micro_keys']
    for row in passes:
        assert len(row['artifacts']) == 2
        for artifact in row['artifacts']:
            name = artifact['path']
            assert name.startswith('micro/'+row['arm']+'/')
            assert Path(name).name == artifact['original_name']
            assert hashlib.sha256(files[name]).hexdigest() == artifact['sha256']
    prefill = result['prefill_pass']
    assert prefill['cache_key'][-1] == 'glm53_ep_prefill_local_fp32_v2'
    expected = {'result.json'}
    for name, rows, count in (
            ('micro PTX', result['micro_artifacts'], 3),
            ('micro cubin', result['micro_resources'], 3),
            ('prefill PTX', prefill['artifacts'], 1),
            ('prefill cubin', prefill['resources'], 1)):
        assert len(rows) == count, name
        for row in rows:
            path = row['path']
            assert path not in expected and hashlib.sha256(files[path]).hexdigest() == row['sha256']
            expected.add(path)
            if 'resources' in row:
                log = str(Path(path).with_suffix('.resources.log'))
                assert files[log].decode() == row['resources']
                expected.add(log)
    labels = {'-'.join((kind, ids, weight, mapping))
              for kind in ('mapped', 'empty', 'offset')
              for ids, weight, mapping in itertools.product(
                  ('i32', 'i64'), ('fp32', 'fp16', 'bf16'),
                  ('i32', 'i64') if kind == 'mapped' else ('i32',))}
    variants = result['prepare_variants']
    assert len(variants) == 24 and {row['label'] for row in variants} == labels
    for row in variants:
        for suffix in ('ptx', 'cubin'):
            path = f"prepare/{row['label']}/kernel.{suffix}"
            assert path not in expected and hashlib.sha256(files[path]).hexdigest() == row[suffix+'_sha256']
            expected.add(path)
    assert set(files) == expected
    report = dict(verdict='PASS', scope='CPU12 archive integrity only; no new tests or GPU execution',
                  original_verdict=result['verdict'], contracts=result['contracts'],
                  original_files=len(files), ptx=28, cubin=28, resource_logs=4,
                  source_revision=submission['revision'],
                  evidence_tar_sha256=capture['evidence_tar_sha256'],
                  result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
