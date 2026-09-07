from pathlib import Path
import hashlib
import json
import re
import shutil
import sys

root = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[2]
dest = repo / 'measurements/glm53_pack_key_20260907'
r = json.loads((root / 'report.json').read_text())
assert r['exit_code'] == 0
expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    repo / 'build/glm53/glm53_megakernel.py', repo / 'build/glm53/glm53_startup_cache.py',
    repo / 'build/glm53/glm53_rank_cache.py')}
content = set()
receipts = []
for arm, row in r['arms'].items():
    assert row['onepass']['quality'] == {'ok': 6, 'total': 6}, row['onepass']['quality']
    assert row['onepass']['korean']['dirty'] == 0
    assert len(row['nodes']) == 4
    content.add(row['compile_cache'][0]['content_sha256'])
    if not arm.endswith('PRIME'):
        assert row['compile_cache'][0]['action'] == 'reuse'
    for node, data in row['nodes'].items():
        assert not data['cache_warnings'] and not any(data['copy_disarmed'])
        assert sum(x['errors'] for x in data['fp8']) == 0
        if not arm.endswith('PRIME'):
            assert all(x['kind'] == 'hit' for x in data['rank'])
            assert sum(x['hit'] for x in data['fp8']) == 244
            assert sum(x['miss'] for x in data['fp8']) == 0
        io = data['pack_io'][-1]
        assert io['fast_hits'] == 254 and io['legacy_hits'] == io['alias_errors'] == 0
        if 'FAST' in arm:
            assert io['sha_hits'] == 254 and io['md5_fallback'] == io['aliases'] == 0
        identity = root / f'{arm}-{node}.{"sha256" if node == "srv2" else "state"}'
        found = {Path(path).name: sha for sha, path in re.findall(r'([0-9a-f]{64})\s+(\S+\.py)', identity.read_text())}
        assert found == expected, (arm, node, found)
        state = (root / f'{arm}-{node}.state').read_text()
        assert state.startswith('running 0 false'), (arm, node, state)
        text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', (root / f'{arm}-{node}.log').read_text())
        receipts.append(f'\n=== {arm} {node} ===\n')
        receipts.extend(line+'\n' for line in text.splitlines() if any(key in line for key in (
            '[rank-cache]', '[fp8-cache]', '[mk-pack-io]', 'host-seconds=', 'MK W4 packs:',
            'load-model took', 'profile/determine-memory took', 'Available KV cache memory')))
assert len(content) == 1
for name in ('screen.json', 'pack-key-gpu.json'):
    result = json.loads((root / name).read_text())
    assert result['ok'] and result['checks'] == 36
    assert all(expected[k] == v for k, v in result['source_sha256'].items())
images = (root / 'image-identities.txt').read_text().splitlines()
assert len(images) == 4 and len({line.split()[1] for line in images}) == 1
assert images[0].split()[1] == (root / 'screen-image-id.txt').read_text().strip()
dest.mkdir(exist_ok=True)
for name in ('report.json', 'report.md', 'health-wall-seconds.tsv', 'onepass.jsonl',
             'screen.json', 'pack-key-gpu.json', 'hash-cpu.json', 'image-identities.txt',
             'source-commit.txt', 'runtime-source-commit.txt', 'deployed-manifest.tsv',
             'exit-code', 'final-control.txt', 'fleet-driver.sh', 'hash-cpu.py',
             'pack-key-report.py', 'startup-cache-report.py', 'collect_receipts.py'):
    shutil.copy2(root / name, dest / name)
(dest / 'boot-receipts.txt').write_text(''.join(receipts))
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
            if p.is_file() and p.suffix in ('.log', '.out', '.jsonl', '.state', '.sha256', '.json', '.tsv')}
(dest / 'raw-file-sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Verified five boots, all-node source/image/cache/quality receipts and 72 GPU exact checks.')
