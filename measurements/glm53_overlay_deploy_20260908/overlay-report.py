"""Read-only summary and verification of the identical-source deployment bracket."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import subprocess
import sys
root, repo = map(Path, sys.argv[1:3])
subprocess.run([sys.executable, str(Path(__file__).with_name('startup-cache-report.py')), str(root)], check=True, stdout=subprocess.DEVNULL)
data = json.loads((root/'report.json').read_text())
ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
for arm, row in data['arms'].items():
    for suffix, key in (('first-requests.json','first_requests'), ('graph-env.json','graph_environment'), ('cache-env.json','runtime_environment')):
        p = root/f'{arm}-{suffix}'
        if p.exists(): row[key] = json.loads(p.read_text())
    boot = root/f'{arm}-boot.out'
    row['compile_cache'] = [json.loads(x) for x in re.findall(r'\[compile-cache\] (\{[^\n]+\})', boot.read_text())] if boot.exists() else []
    for node, detail in row.get('nodes',{}).items():
        text = ansi.sub('', (root/f'{arm}-{node}.log').read_text(errors='replace'))
        detail['graph_profile_skipped'] = '[glm53-graph-profile] skipped unused estimate' in text
        detail['kv_memory_gib'] = [float(x) for x in re.findall(r'Available KV cache memory: ([\d.-]+) GiB', text)]
        detail['kv_blocks'] = [int(x) for x in re.findall(r'num_gpu_blocks[=: ]+(\d+)',text)]
        detail['pack_io'] = [{k: float(v) if k.endswith('_s') else int(v) for k,v in re.findall(r'(\w+)=([\d.]+)',fields)} for fields in re.findall(r'\[mk-pack-io\] \S+ ([^\n]*)', text)]
        if node == 'srv2':
            health = text.find('GET /health HTTP/1.1" 200')
            posts = list(re.finditer(r'(\d+\.\d+\.\d+\.\d+):\d+ - "POST /',text))
            row['post_completions'] = dict(loopback=sum(m[1].startswith('127.') for m in posts), non_loopback=sum(not m[1].startswith('127.') for m in posts), before_health=sum(m.start()<health for m in posts))
summary = {}
for kind in ('BASE','FAST'):
    rows = [r for a,r in data['arms'].items() if re.search(kind+r'[12]$',a)]
    values = {'health_wall_s':[r['health_wall_s'] for r in rows if 'health_wall_s' in r]}
    for phase in ('load-model','encoder-profile','profile-run','cudagraph-memory-profile','profile/determine-memory','cudagraph-capture','compile+warmup'):
        values[phase] = [r['nodes']['srv2']['phase_s'].get(phase,[0])[0] for r in rows if 'srv2' in r.get('nodes',{})]
    for label in ('text','image','video'):
        values['first_'+label+'_ttft_s']=[x['ttft_s'] for r in rows for x in r.get('first_requests',{}).get('requests',[]) if x['kind']==label]
    summary[kind] = {k:dict(samples=v, mean=statistics.mean(v), min=min(v), max=max(v)) for k,v in values.items() if v and all(x is not None for x in v)}

deploy_times = dict(line.split() for line in (root/'deployment-seconds.tsv').read_text().splitlines())
for arm,row in data['arms'].items():
    row['deployment_s'] = int(deploy_times[arm])
    paths = {key:root/f'{arm}-{key}.json' for key in ('before-deploy','after-deploy','after-boot')}
    if not all(p.exists() for p in paths.values()): continue
    snapshots = {k:json.loads(p.read_text()) for k,p in paths.items()}
    details = {}
    for node in ('srv1','srv2','srv3','srv4'):
        a,b,c = (snapshots[k][node] for k in ('before-deploy','after-deploy','after-boot'))
        common = sorted(a['files'].keys() & b['files'].keys())
        same = [k for k in common if a['files'][k]['sha256']==b['files'][k]['sha256']]
        changed_times = [k for k in same if a['files'][k]['mtime_ns']!=b['files'][k]['mtime_ns'] or a['files'][k]['inode']!=b['files'][k]['inode']]
        ninja_changes = [k for k in c['ninja'] if k not in b['ninja'] or b['ninja'][k]['sha256']!=c['ninja'][k]['sha256']]
        details[node] = dict(file_count=len(b['files']),same_content_count=len(same),rewritten_identical_files=changed_times,ninja_logs_changed=ninja_changes)
        if '--verify' in sys.argv and not arm.endswith('PRIME'):
            assert len(same)==len(a['files'])==len(b['files']), (arm,node,'different source contents')
            if 'FAST' in arm:
                assert not changed_times and not ninja_changes, (arm,node,changed_times,ninja_changes)
            else:
                assert any(k.endswith('.cu') for k in changed_times) and ninja_changes, (arm,node,'control did not recompile')
    row['deployment_receipts'] = details

data['head_comparison'] = summary
lines = ['# GLM unchanged overlay deployment', '', f"Runtime source: `{data['source_commit']}`.", '',
    'PRIME is excluded. Timed order: BASE1, FAST1, FAST2, BASE2. Same source and image; only DEPLOY_PRESERVE_IDENTICAL changes during each identical-source publication replay. PRIME uses the official deployer with current-main admission; timed arms require identical manifest/source hashes on all four nodes before reproducing install/scp versus the production rsync helper. No revision is deployed by the timed replay. PREFILL_WARMUP=0 disables the separate background prefill benchmark in every arm. The graph-profile optimization remains disabled in every arm. Both arms retain all model/MM profiles, actual graph capture and kernel warmup; both use early CPU MM warmup. Health wall starts after deployment; source publication duration is reported separately (timed arms exclude repeated CPU admission checks from the official deployer). First text/image/video requests precede the canonical 2K/32K Korean onepass.', '',
    '| Arm | Health s | Model s | Encoder s | Dry capture s | Memory profile s | Real capture s | Compile/warmup s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for arm,r in data['arms'].items():
    p=r.get('nodes',{}).get('srv2',{}).get('phase_s',{})
    vals=[r.get('health_wall_s'),*[p.get(k,[0])[0] for k in ('load-model','encoder-profile','cudagraph-memory-profile','profile/determine-memory','cudagraph-capture','compile+warmup')]]
    lines.append('| '+arm+' | '+' | '.join(map(str,vals))+' |')
lines += ['', '| Metric | BASE mean | FAST mean | Difference |', '|---|---:|---:|---:|']
for key,b in summary['BASE'].items():
    f=summary['FAST'].get(key)
    if f: lines.append(f"| {key} | {b['mean']:.3f} | {f['mean']:.3f} | {f['mean']-b['mean']:+.3f} |")
lines += ['', 'Phase timers are nested and include host work and existing synchronizations. First-request samples do not establish general throughput/quality. POST completions and response evidence are retained to detect interference.', '']
for arm,r in data['arms'].items():
    lines.append(f"- {arm}: quality={r.get('onepass',{}).get('quality')}; corruption={r.get('onepass',{}).get('korean')}; posts={r.get('post_completions')}")
lines += ['', '| Arm | Publish s | Node | Identical source files rewritten | Ninja build logs changed |', '|---|---:|---|---:|---:|']
for arm,row in data['arms'].items():
    for node,d in row.get('deployment_receipts',{}).items():
        lines.append(f"| {arm} | {row['deployment_s']} | {node} | {len(d['rewritten_identical_files'])} | {len(d['ninja_logs_changed'])} |")
if '--verify' in sys.argv:
    assert data['exit_code']==0
    assert list(data['arms'])==['DEPLOYCACHEPRIME','DEPLOYCACHEBASE1','DEPLOYCACHEFAST1','DEPLOYCACHEFAST2','DEPLOYCACHEBASE2']
    # Later main merges must not change the canonical identity of this trial.
    def measured_bytes(name):
        return subprocess.check_output(['git','-C',str(repo),'show',data['source_commit']+':build/glm53/'+name])
    expected={name:hashlib.sha256(measured_bytes(name)).hexdigest() for name in ('gpu_worker.py','deneb_boot_stamps.py','glm53_megakernel.py','glm53_rank_cache.py','glm53_startup_cache.py')}
    manifest=measured_bytes('manifest.tsv').decode()
    names=[line.split('\t')[0] for line in manifest.splitlines() if line and not line.startswith('#')]
    canonical={name:hashlib.sha256(measured_bytes(name)).hexdigest() for name in names}
    canonical['manifest.tsv']=hashlib.sha256(('# source_commit='+data['source_commit']+'\n'+manifest).encode()).hexdigest()
    images,content,environments=set(),set(),set()
    deployed_sources=set()
    for arm,r in data['arms'].items():
        fast=False
        environments.add(tuple(sorted(r['runtime_environment'])))
        deployed = json.loads((root/f'{arm}-after-deploy.json').read_text())
        assert len(r.get('deployment_receipts',{})) == 4, (arm,'missing deployment receipts')
        for node, snapshot in deployed.items():
            actual_sources={name:v['sha256'] for name,v in snapshot['files'].items()}
            assert actual_sources==canonical, (arm,node,'deployed source differs from canonical build')
            deployed_sources.add(tuple(sorted(actual_sources.items())))
        assert dict(x.split('=',1) for x in r['graph_environment'])=={'VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE':str(int(fast)),'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS':'0'}
        assert r['first_requests']['ok'] and len(r['first_requests']['requests'])==3
        assert r['onepass']['quality']=={'ok':6,'total':6} and r['onepass']['korean']['dirty']==0
        assert r['post_completions']==dict(loopback=7,non_loopback=0,before_health=0)
        assert len(r['nodes'])==4
        assert len(r['compile_cache'])==1
        content.add(r['compile_cache'][0]['content_sha256'])
        if not arm.endswith('PRIME'): assert r['compile_cache'][0]['action']=='reuse'
        for node,d in r['nodes'].items():
            assert d['graph_profile_skipped']==fast
            assert ('cudagraph-memory-profile' in d['phase_s'])!=fast
            for label in ('encoder-profile','profile-run','cudagraph-capture','compile+warmup'): assert label in d['phase_s']
            assert not d['cache_warnings'] and not any(d['copy_disarmed'])
            assert sum(x['errors'] for x in d['fp8'])==0
            if not arm.endswith('PRIME'):
                assert len(d['rank'])==1 and d['rank'][0]['kind']=='hit'
                assert sum(x['hit'] for x in d['fp8'])==244 and sum(x['miss'] for x in d['fp8'])==0
            io=d['pack_io'][-1]
            assert io['sha_hits']==io['fast_hits']==258
            assert io['legacy_hits']==io['md5_fallback']==io['aliases']==io['alias_errors']==0
            receipt=(root/f'{arm}-{node}.{"sha256" if node=="srv2" else "state"}').read_text()
            actual={Path(p).name:h for h,p in re.findall(r'([0-9a-f]{64})\s+(\S+\.py)',receipt)}
            assert actual==expected,(arm,node,actual,expected)
            state=(root/f'{arm}-{node}.state').read_text()
            assert state.startswith('running 0 false sha256:')
            images.add(state.splitlines()[0].split()[-1])
    assert len(content)==len(images)==len(environments)==len(deployed_sources)==1
    data['verification']=dict(ok=True, source_hashes=expected, images=list(images), compile_content=list(content), boots=5, nodes_per_boot=4, runtime_environment_identical=True, all_deployed_source_hashes_identical=True, canonical_source_hashes=canonical)
(root/'report.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(root/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:16]))
