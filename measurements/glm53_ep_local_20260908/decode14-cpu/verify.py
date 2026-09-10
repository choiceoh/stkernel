#!/usr/bin/env python3
"""Reconcile archived CPU14 bytes and receipts without importing GPU modules."""
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
    assert submission['revision'] == 'c3696a76c6674ad38e01967fa587b8d31a6d121e'
    assert result['verdict'] == 'PASS' and result['phase'] == 'complete' and 'error' not in result
    assert result['contracts'] == dict(tests_run=85, failures=0, errors=0, skips=0)
    assert result['cuda_initialized'] is False and result['binding_runtime_rechecked'] is True
    assert all(exit_receipt[key] == 0 for key in ('returncode', 'payload_returncode', 'copy_returncode'))
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
        assert identities == snap['worker']['files'] == snap['head_evidence']
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
    scatter_sites = {}
    publication_sites = {}
    for row in result['micro_artifacts']:
        arm = row['path'].split('/')[1]
        text = files[row['path']].decode()
        counts = dict(fp32_v2=text.count('red.relaxed.gpu.global.add.v2.f32'),
                      bf16x2=text.count('red.relaxed.gpu.global.add.noftz.bf16x2'))
        assert counts == (dict(fp32_v2=4, bf16x2=0) if arm.endswith('fp32')
                          else dict(fp32_v2=0, bf16x2=4))
        lines = text.splitlines()
        sites = []
        for i, line in enumerate(lines):
            red = ('red.relaxed.gpu.global.add.v2.f32' if arm.endswith('fp32')
                   else 'red.relaxed.gpu.global.add.noftz.bf16x2')
            if red not in line:
                continue
            loads = [j for j in range(i) if 'ld.shared.b16' in lines[j]][-2:]
            assert len(loads) == 2
            store = max(j for j in range(loads[0]) if 'st.shared.b32' in lines[j])
            pre = [j for j in range(store+1, loads[0]) if 'bar.sync' in lines[j]]
            post = next(j for j in range(i+1, len(lines)) if 'bar.sync' in lines[j])
            assert len(pre) == (1 if arm.endswith('fp32') else 0)
            assert all(', 128;' in lines[j] for j in pre+[post])
            assert 'fence.proxy.async.shared::cta;' in lines[store+1]
            sites.append(dict(store_end=store+1, pre_barriers=[j+1 for j in pre],
                              loads=[j+1 for j in loads], red=i+1, post_barrier=post+1,
                              instructions={str(j+1):lines[j].strip()
                                            for j in [store, store+1, *pre, *loads, i, post]}))
        assert len(sites) == 4
        publication_sites[arm] = dict(ptx_sha256=row['sha256'], sites=sites,
                                     scope='all four statically unrolled scatter copies; no runtime traffic count')
        scatter_sites[arm] = dict(path=row['path'], ptx_sha256=row['sha256'],
                                 static_instruction_sites=counts)
    report = dict(verdict='PASS', scope='CPU14 archive integrity only; no new tests or GPU execution',
                  original_verdict=result['verdict'], contracts=result['contracts'],
                  original_files=len(files), ptx=28, cubin=28, resource_logs=4,
                  source_revision=submission['revision'],
                  scatter_sites=scatter_sites, publication_sites=publication_sites,
                  evidence_tar_sha256=capture['evidence_tar_sha256'],
                  result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
