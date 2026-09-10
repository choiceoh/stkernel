#!/usr/bin/env python3
"""Normal CPU fleet payload: local/global-route EP tiled compiler witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import glm53_ep_capsule_runtime as capsule_runtime
from glm53_ep_tiled_compile import (source_receipt, STATIC_ROWS, DYNAMIC_ROWS,
    CPU_TEST_COUNTS, EXPECTED_CPU_TESTS, static_specialization, validate_scatter_helper_receipt,
    GLOBAL_STATIC_CASES, global_static_specialization, OPT_STATIC_CASES, opt_static_specialization,
    opt_shared_capacity)


def validate_artifacts(output,result):
    output=output.resolve(strict=True)
    expected=set()
    groups=(('static',STATIC_ROWS),('global_static',GLOBAL_STATIC_CASES),
            ('opt_static',OPT_STATIC_CASES),('dynamic',DYNAMIC_ROWS))
    for kind,rows in groups:
        passes=result[kind+'_passes']
        arms = ([kind.replace('_','-')+'/'+case[0] for case in rows]
                if kind in ('global_static','opt_static')
                else [kind+'/M'+str(m) for m in rows])
        assert [p['arm'] for p in passes]==arms
        for rows_count,passed in zip(rows,passes):
            if kind == 'static':
                selected = passed['specialization']
                assert selected == static_specialization(rows_count,passed['cache_key'],
                    selected['a_ring'],selected['word_unpack'],
                    selected['scatter_bf16'],selected['output_dtype'])
            if kind == 'global_static':
                selected = passed['specialization']
                assert selected == global_static_specialization(rows_count,passed['cache_key'],
                    selected['a_ring'],selected['word_unpack'],selected['scatter_bf16'],
                    selected['output_dtype'],selected['route'])
            if kind == 'opt_static':
                selected = passed['specialization']
                assert selected == opt_static_specialization(rows_count,passed['cache_key'],
                    selected['a_ring'],selected['word_unpack'],selected['scatter_bf16'],
                    selected['output_dtype'],selected.get('route'),
                    selected['decode_opt'],selected['storage_bytes'])
                assert passed['shared_capacity'] == opt_shared_capacity(passed)
            for name,suffix in (('artifacts','.ptx'),('resources','.cubin')):
                assert len(passed[name])==1
                for row in passed[name]:
                    rel=Path(row['path'])
                    assert not rel.is_absolute() and '..' not in rel.parts and str(rel)==row['path']
                    assert str(rel.parent)==passed['arm'] and rel.suffix==suffix
                    path=output/rel
                    assert path.resolve(strict=True)==path and path.is_file()
                    assert str(rel) not in expected
                    raw=path.read_bytes()
                    assert raw and hashlib.sha256(raw).hexdigest()==row['sha256']
                    expected.add(str(rel))
                    if suffix=='.cubin':
                        assert path.with_suffix('.resources.log').read_text()==row['resources']
    actual={str(p.relative_to(output)) for suffix in ('*.ptx','*.cubin') for p in output.rglob(suffix)}
    assert actual==expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capsule-root',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    a=p.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',a.image):p.error('immutable image ID required')
    root=Path(__file__).resolve().parents[1]
    capsule=capsule_runtime.validate_capsule_input(a.capsule_root,a.manifest_sha256)
    output=a.output.resolve()
    if any(output.is_relative_to(x) or x.is_relative_to(output) for x in (root,capsule)):
        p.error('output must be separate from source/capsule')
    if output.exists():p.error('fresh evidence directory required')
    available=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available<12*1024*1024:p.error('12 GiB available required; no serving memory reclaim')
    sources=source_receipt(root)
    output.mkdir(parents=True)
    command=['docker','run','--rm','--runtime=runc','--network=none','--memory=4g',
        '--memory-swap=4g','--cpus=2','--pids-limit=128','-e','NVIDIA_VISIBLE_DEVICES=void',
        '-e','MAX_JOBS=1','--entrypoint=python3','-v',f'{root}:/repo:ro','-v',f'{output}:/evidence']
    for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,target,*_=line.split('\t')
        if '/flashinfer/' in target or name in ('flashinfer_b12x_moe.py','gpu_worker.py'):
            command+=['-v',f'{root/"build/glm53"/name}:{target}:ro']
    command+=capsule_runtime.docker_capsule_args(capsule,a.manifest_sha256)
    command += [a.image,'-B','/repo/probes/glm53_ep_tiled_compile.py','--output','/evidence',
        '--capsule-root',capsule_runtime.CAPSULE_MOUNT,'--manifest-sha256',a.manifest_sha256]
    completed=subprocess.run(command)
    capsule_runtime.validate_capsule_input(capsule,a.manifest_sha256)
    if completed.returncode:return completed.returncode
    result=json.loads((output/'result.json').read_text())
    assert result['verdict']=='PASS' and result['phase']=='complete' and result['cuda_initialized'] is False
    assert result['binding_runtime_rechecked'] is True and result['compile_only'] is True
    assert not any(k in result for k in ('error','cleanup_error','recheck_error'))
    assert result['contracts']['tests_run'] == EXPECTED_CPU_TESTS
    assert all(result['contracts'][key] == 0 for key in ('failures','errors','skips'))
    assert result['selected_test_counts'] == CPU_TEST_COUNTS
    assert result['contracts_process_isolated'] is True
    contracts=json.loads((output/'contracts.json').read_text())
    assert contracts['verdict']=='PASS' and contracts['phase']=='complete'
    assert contracts['cuda_initialized'] is False and contracts['binding_runtime_rechecked'] is True
    assert not any(k in contracts for k in ('error','cleanup_error','recheck_error'))
    for key in ('contracts','selected_test_counts','binding_runtime','mounted_sources','contract_sources'):
        assert contracts[key]==result[key]
    capsule_runtime.validate_runtime_receipt(result['binding_runtime'])
    validate_scatter_helper_receipt(root,result['scatter_helper'])
    assert source_receipt(root)==sources
    for key,value in sources.items():assert result[key]==value
    validate_artifacts(output,result)
    return 0


if __name__=='__main__':raise SystemExit(main())
