"""Verify all arm receipts and curate the completed renderer experiment."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys

root = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
dest = repo / 'measurements/glm53_early_mm_20260908'
report = json.loads((root / 'report.json').read_text())
assert report['exit_code'] == 0
expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    repo / 'build/glm53/glm53_megakernel.py', repo / 'build/glm53/glm53_startup_cache.py',
    repo / 'build/glm53/glm53_rank_cache.py', repo / 'build/glm53/async_llm.py',
    repo / 'build/glm53/glm53_renderer_warmup.py')}
assert len(report['arms']) == 5
content, images, receipts = set(), set(), []
for arm, row in report['arms'].items():
    assert row['onepass']['quality'] == {'ok': 6, 'total': 6}
    assert row['onepass']['korean']['dirty'] == 0
    assert len(row['nodes']) == 4
    assert row['serving_requests']['loopback'] == 4
    assert row['serving_requests']['completed_before_first_health'] == 0
    assert len(row['compile_cache']) == 1
    content.add(row['compile_cache'][0]['content_sha256'])
    if not arm.endswith('PRIME'):
        assert row['compile_cache'][0]['action'] == 'reuse'
    early = 'FAST' in arm or arm.endswith('PRIME')
    env = row['environment']
    assert env['VLLM_GLM53_EARLY_MM_WARMUP'] == str(int(early))
    assert env['VLLM_GLM53_RANK_CACHE'] == '/cache/glm53-ranks'
    assert env['VLLM_GLM53_FP8_CACHE'] == '/cache/glm53-fp8'
    assert all(env[k] == '1' for k in ('VLLM_GLM53_MK_PACK_SHA256', 'VLLM_GLM53_MK_PACK_FAST_IO'))
    for node, data in row['nodes'].items():
        assert not data['cache_warnings'] and not any(data['copy_disarmed'])
        assert sum(x['errors'] for x in data['fp8']) == 0
        if not arm.endswith('PRIME'):
            assert len(data['rank']) == 1 and data['rank'][0]['kind'] == 'hit'
            assert sum(x['hit'] for x in data['fp8']) == 244
            assert sum(x['miss'] for x in data['fp8']) == 0
        io = data['pack_io'][-1]
        assert io['fast_hits'] == io['sha_hits'] == 255
        assert io['legacy_hits'] == io['md5_fallback'] == io['aliases'] == io['alias_errors'] == 0
        identity = root / f'{arm}-{node}.{"sha256" if node == "srv2" else "state"}'
        found = {Path(path).name: sha for sha, path in re.findall(r'([0-9a-f]{64})\s+(\S+\.py)', identity.read_text())}
        wanted = expected if node == 'srv2' else {k: v for k, v in expected.items() if k not in ('async_llm.py', 'glm53_renderer_warmup.py')}
        assert found == wanted, (arm, node, found, wanted)
        state = (root / f'{arm}-{node}.state').read_text()
        assert state.startswith('running 0 false sha256:'), (arm, node, state)
        images.add(state.splitlines()[0].split()[-1])
        text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', (root / f'{arm}-{node}.log').read_text())
        if node == 'srv2':
            mm = data['renderer']
            assert len(mm['stock_warmup_s']) == 2
            assert mm['completed_before_model'] == early
            assert len(mm['reuse_join_s']) == 2 * int(early)
            assert 'warmup failed' not in text.lower()
        receipts.append(f'\n=== {arm} {node} ===\n')
        receipts.extend(line+'\n' for line in text.splitlines() if any(key in line for key in (
            '[rank-cache]', '[fp8-cache]', '[mk-pack-io]', 'host-seconds=', 'MK W4 packs:',
            'load-model took', 'profile/determine-memory took', 'Available KV cache memory',
            '[early-mm-warmup]', 'warmup completed in')))
assert len(content) == 1
assert len(images) == 1 and next(iter(images)) == (root / 'screen-image-id.txt').read_text().strip()
screen = json.loads((root / 'screen.json').read_text())
assert screen['ok'] and screen['checks'] == 6
assert screen['source_sha256'] == expected['glm53_renderer_warmup.py']
for key in ('warmup', 'requests'):
    assert screen['arms']['0'][key] == screen['arms']['1'][key]
dest.mkdir(exist_ok=True)
for name in ('report.json', 'report.md', 'health-wall-seconds.tsv', 'onepass.jsonl', 'screen.json',
             'screen-image-id.txt', 'source-commit.txt', 'runtime-source-commit.txt', 'deployed-manifest.tsv',
             'exit-code', 'final-control.txt', 'fleet-driver.sh', 'renderer-report.py',
             'startup-cache-report.py', 'collect_receipts.py', 'validation.json', 'logic-full.log', 'screen.log'):
    shutil.copy2(root / name, dest / name)
(dest / 'boot-receipts.txt').write_text(''.join(receipts))
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
          if p.is_file() and p.suffix in ('.log', '.out', '.jsonl', '.state', '.sha256', '.json', '.tsv')}
(dest / 'raw-file-sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
print('Verified five boots: source/image identity, all-rank warm caches, quality and exact CPU multimodal preprocessing.')
