#!/usr/bin/env python3
"""Reconcile archived CPU20 bytes and receipts without importing GPU modules."""
import hashlib
import itertools
import json
import re
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parent
    result = json.loads((root/'result.json').read_text())
    submission = json.loads((root/'head/submission.json').read_text())
    exit_receipt = json.loads((root/'head/exit.json').read_text())
    capture = json.loads((root/'capture.json').read_text())
    assert submission['revision'] == '82d81d65b6a74e09abd9be0a8f05508e4855f6f1'
    assert result['verdict'] == 'PASS' and result['phase'] == 'complete' and 'error' not in result
    assert result['contracts'] == dict(tests_run=120, failures=0, errors=0, skips=0)
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
    assert [key[10] for key in result['micro_keys']] == [[16, 128], [64, 128], [64, 128]]
    assert [key[17] for key in result['micro_keys']] == [72, None, None]
    assert [key[7:9] for key in result['micro_keys']] == [[8,64],[1,8],[8,64]]
    assert [('glm53_ep_micro_scatter_fp32_v1' in key[22:]) for key in result['micro_keys']] == [True,True,False]
    assert [('glm53_ep_micro_direct_scatter_v1' in key[22:]) for key in result['micro_keys']] == [True,False,False]
    assert [('glm53_ep_micro_shared_fc1_a_v1' in key[22:]) for key in result['micro_keys']] == [True,False,False]
    assert result['micro_keys'][0][22:] == ['glm53_ep_micro_scatter_fp32_v1','glm53_ep_micro_direct_scatter_v1','glm53_ep_micro_shared_fc1_a_v1','glm53_ep_micro_m16_v1']
    passes = result['micro_passes']
    assert [row['arm'] for row in passes] == ['m16-topk8-shared-a-direct-fp32','m64-topk1-fp32','m64-topk8-bf16']
    assert [(row['scatter_fp32'],row['ep_direct_scatter']) for row in passes] == [(True,True),(True,False),(False,False)]
    assert [row['shared_fc1_a'] for row in passes] == [True,False,False]
    assert [row['ep_m16'] for row in passes] == [True,False,False]
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
    shared_fc1 = {}
    for row in result['micro_artifacts']:
        arm = row['path'].split('/')[1]
        text = files[row['path']].decode()
        counts = dict(fp32_v2=text.count('red.relaxed.gpu.global.add.v2.f32'),
                      bf16x2=text.count('red.relaxed.gpu.global.add.noftz.bf16x2'))
        direct = arm == 'm16-topk8-shared-a-direct-fp32'
        assert counts == (dict(fp32_v2=64, bf16x2=0) if direct else
                          dict(fp32_v2=4, bf16x2=0) if arm.endswith('fp32')
                          else dict(fp32_v2=0, bf16x2=4))
        lines = text.splitlines()
        sites = []
        if direct:
            assert row['sha256'] == '48cfbfd07e1a20f899fd989184accd175d091296187966c20fd23a54cd67727e'
            assert '.reqntid 96, 1, 1' in text
            assert 'setmaxnreg' not in text
            for start, end in ((2934,3287),(3288,3621),(3622,3953),(3954,4300)):
                block = lines[start-1:end]
                assert not any('st.shared.' in line or 'ld.shared.b16' in line or 'fence.proxy' in line for line in block)
                reds = [start+j for j,line in enumerate(block) if 'red.relaxed.gpu.global.add.v2.f32' in line]
                barriers = [start+j for j,line in enumerate(block) if 'bar.sync' in line]
                assert len(reds) == 16 and barriers == [end] and ', 64;' in lines[end-1]
                assert sum('ld.shared.s32' in line for line in block) == 2
                assert sum('ld.shared.f32' in line for line in block) == 2
                id_loads = [start+j for j,line in enumerate(block) if 'ld.shared.s32' in line]
                guards = [max(j+1 for j in range(start-1,pos-1)
                              if re.search(r'@%p\d+ bra',lines[j])) for pos in id_loads]
                assert len(set(guards)) == 2
                sites.append(dict(line_range=[start,end], fp32_red_sites=reds,
                                  shared_bf16_loads=0, shared_stores=0, proxy_fences=0,
                                  metadata_id_loads=2, metadata_weight_loads=2,
                                  row_guards=guards, id_load_lines=id_loads,
                                  pre_barriers=[], post_barrier=end,
                                  instructions={str(j):lines[j-1].strip() for j in [reds[0],end]}))
            assert not any('ld.local' in line or 'st.local' in line for line in lines)
            stages = []
            def address_offset(expression, before):
                match = re.fullmatch(r'%r268(?:\+(\d+))?', expression)
                if match:
                    return int(match[1] or 0), None
                definitions = [(j+1, re.fullmatch(
                    r'\s*add.s32\s+'+re.escape(expression)+r', %r268, (\d+);', lines[j]))
                    for j in range(before-1)]
                definitions = [(j,int(m[1])) for j,m in definitions if m]
                assert len(definitions) == 1, (expression,definitions)
                return definitions[0][1], definitions[0][0]
            for start,end,phase in ((4462,4510,0),(4529,4578,1),(4596,4644,0),(4662,4710,1)):
                tma = [(i+1,lines[i]) for i in range(start-1,end) if 'cp.async.bulk.tensor.' in lines[i]]
                assert len(tma) == 6
                expected_offsets = ([2048,51200,18432,53248,34816,55296] if phase == 0
                                    else [10240,52224,26624,54272,43008,56320])
                resolved = []
                definition_lines = set()
                for pos,line in tma:
                    addresses = re.findall(r'\[([^\]]+)\]',line)
                    offset, definition = address_offset(addresses[0],pos)
                    barrier_offset, barrier_definition = address_offset(addresses[-1],pos)
                    assert barrier_offset == 16+phase*8
                    resolved.append(offset)
                    definition_lines.update(x for x in (definition,barrier_definition) if x is not None)
                assert resolved == expected_offsets
                assert '27648;' in lines[start-2]
                assert 'mbarrier.arrive.expect_tx.shared.b64' in lines[start-1]
                expect_barrier = re.findall(r'\[([^\]]+)\]',lines[start-1])[0]
                assert address_offset(expect_barrier,start)[0] == 16+phase*8
                stages.append(dict(line_range=[start,end], phase=phase, expected_bytes=27648,
                                   destination_offsets=resolved, barrier_offset=16+phase*8,
                                   instructions={str(i):lines[i-1].strip() for i in
                                                 [start-1,start,*sorted(definition_lines),*[p for p,_ in tma]]}))
            for pos in (1660,4715,4304,4860):
                assert 'bar.sync' in lines[pos-1] and ', 96;' in lines[pos-1]
            assert sum('cp.async.bulk.tensor.' in x for x in lines) == 32
            assert 'cp.async.bulk.tensor.' in lines[4742]
            shared_fc1 = dict(ptx_sha256=row['sha256'], stages=stages,
                              consumer_end_fc1_barrier=1660, producer_end_fc1_barrier=4715,
                              first_fc2_tma=4743, final_task_barriers=[4304,4860],
                              static_tma_sites=32, required_threads=96, setmaxnreg_sites=0,
                              scope='four statically unrolled combined FC1 stages plus four two-transfer FC2 stages; no DRAM traffic or speed measurement')
            publication_sites[arm] = dict(ptx_sha256=row['sha256'], sites=sites,
                scope='four FC2 output unroll blocks, 16 register pairs per block; static sites are not runtime traffic counts')
            scatter_sites[arm] = dict(path=row['path'], ptx_sha256=row['sha256'], static_instruction_sites=counts)
            continue
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
    previous_root = root.parent/'decode19-cpu'
    previous = json.loads((previous_root/'result.json').read_text())
    with tarfile.open(previous_root/'evidence.tar.gz') as old_tar:
        assert old_tar.extractfile('evidence/result.json').read() == (previous_root/'result.json').read_bytes()
        old_direct = previous['micro_artifacts'][0]
        old_direct_raw = old_tar.extractfile('evidence/'+old_direct['path']).read()
        assert hashlib.sha256(old_direct_raw).hexdigest() == old_direct['sha256']
        assert old_direct_raw.count(b'cp.async.bulk.tensor.') == 32
        assert old_direct_raw.count(b'setmaxnreg.') == 2
        for field in ('micro_artifacts','micro_resources'):
            for index in (1,2):
                old,current = previous[field][index],result[field][index]
                old_raw = old_tar.extractfile('evidence/'+old['path']).read()
                assert old['sha256'] == current['sha256'] == hashlib.sha256(old_raw).hexdigest()
                assert old_raw == files[current['path']]
    resource_rows = [row['resources'] for row in result['micro_resources']]
    assert all('STACK:0' in row and 'LOCAL:0' in row for row in resource_rows)
    assert [re.search(r'REG:(\d+)',row)[1] for row in resource_rows] == ['157','222','220']
    report = dict(verdict='PASS', scope='CPU20 archive integrity only; no new tests or GPU execution',
                  original_verdict=result['verdict'], contracts=result['contracts'],
                  original_files=len(files), ptx=28, cubin=28, resource_logs=4,
                  source_revision=submission['revision'],
                  scatter_sites=scatter_sites, publication_sites=publication_sites,
                  shared_fc1=shared_fc1, micro_resources=result['micro_resources'],
                  controls_identical_to_cpu19=['m64-topk1-fp32','m64-topk8-bf16'],
                  previous_result_sha256=hashlib.sha256((previous_root/'result.json').read_bytes()).hexdigest(),
                  previous_candidate_ptx_sha256=old_direct['sha256'], previous_static_tma_sites=32,
                  evidence_tar_sha256=capture['evidence_tar_sha256'],
                  result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
