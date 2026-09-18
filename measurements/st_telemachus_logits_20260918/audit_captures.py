"""Validate private full-token captures and prepare #1167-readable paired views."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import torch

torch.set_num_threads(1)
ROOT = Path('/tmp/telemachus-quality-0917')
OUT = ROOT / 'lossless-logits-replay'
SOURCE = '8f87c211515cb5896036af53c3b18ed8d908aacd'
DUMP = Path('/home/choiceoh/glm53-logs/st-bracket-dumps') / ('st-lossless-logits0918-hold-' + SOURCE[:12]) / 'incident-logits'
spec = importlib.util.spec_from_file_location('margin1167', ROOT / 'draw-audit-repair/tools/incident_logit_margin.py')
margin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(margin)
sys.path.insert(0, '/code')
from engine.base import draws


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


record = json.loads((ROOT / 'original-engine-record.json').read_text())
prompt = record['tokens'][:record['prompt_len']]
assert len(prompt) == 50005
assert digest(prompt) == 'e8aefde846b0c67641f9285d2646ca3977ab790534761306422ba81dcc4bb1dc'
results = sorted(OUT.glob('case*-mode*-original-seed7.json'))
admissions = sorted(int(path.name.split('-')[0][5:]) for path in DUMP.glob('admit*-gen0.pt'))
assert len(admissions) >= len(results)
manifest = dict(source=SOURCE, dump=str(DUMP), prompt_sha256=digest(prompt),
                margin_tool_pr=1170, margin_tool_base_commit='e2c6f14275150b546613d68515f70bc96b60ca95',
                margin_tool_sha256=hashlib.sha256((ROOT / 'draw-audit-repair/tools/incident_logit_margin.py').read_bytes()).hexdigest(),
                alignment='Each case is the same single incident request. View filenames use logical admission 1. '
                          'Symlink targets and payloads preserve the actual runtime admission. Prefix hashes guard comparisons.',
                cases=[])
for path, admission in zip(results, admissions):
    result = json.loads(path.read_text())
    receipt = result['receipt']
    ids = result['response']['ids']
    assert receipt['source'] == SOURCE and receipt['cached_tokens'] == 0
    captures = sorted(DUMP.glob(f'admit{admission}-gen*.pt'), key=lambda p: int(p.stem.split('-gen')[1]))
    assert len(captures) == len(ids), (admission, len(captures), len(ids))
    directory = OUT / 'margin-input' / f"case{receipt['case']:02d}-mode{receipt['mode']}"
    directory.mkdir(parents=True, exist_ok=True)
    for generation, capture in enumerate(captures):
        data = torch.load(capture, map_location='cpu', weights_only=True)
        assert (data['generation'], data['admission'], data['mode']) == (generation, admission, receipt['mode'])
        assert data['seed'] == receipt['seed'] == 7
        assert data['prompt_len'] == len(prompt) and data['prefix_len'] == len(prompt) + generation
        assert data['prompt_sha256'] == digest(prompt)
        assert data['prefix_sha256'] == digest(prompt + ids[:generation])
        assert data['input_tail'] == (prompt + ids[:generation])[-8:]
        assert data['picks'] == data['committed'] == [ids[generation]]
        key = draws.row_key(7, 0, generation)
        assert data['row_key'] == key
        assert data['uniform'] == draws.uniform(key, draws.RICH, 0) == float(data['uniforms'][0])
        assert data['raw'].shape == data['processed'].shape == data['probabilities'].shape == (1, 154880)
        assert torch.isfinite(data['raw']).all()
        assert torch.isfinite(data['probabilities']).all()
        assert abs(float(data['probabilities'].sum()) - 1.) < 1e-5
        parsed = margin.read_capture(capture)
        assert parsed['prefix'] == data['prefix_sha256'] and parsed['uniform'] == data['uniform']
        assert parsed['width'] == 154880
        assert parsed['row_key'] == data['row_key']
        assert margin.draw_address(parsed) == (draws.RICH, 0), (admission, generation, margin.draw_address(parsed))
        link = directory / f'admit1-gen{generation}.pt'
        if not link.exists():
            link.symlink_to(capture)
        assert link.resolve() == capture.resolve()
    item = dict(case=receipt['case'], mode=receipt['mode'], admission=admission, captures=len(captures),
                bytes=sum(p.stat().st_size for p in captures), margin_input=str(directory),
                result=str(path), output_ids_sha256=receipt['output_ids_sha256'], verified=True)
    manifest['cases'].append(item)
    print(json.dumps(item), flush=True)
(OUT / 'capture-manifest.json').write_text(json.dumps(manifest, indent=2))


def sample_stats(data):
    probabilities = data['probabilities'][0].double()
    probabilities /= probabilities.sum()
    picked = data['committed'][0]
    cdf = probabilities.cumsum(0)
    lower = float(cdf[picked - 1]) if picked else 0.
    upper = float(cdf[picked])
    uniform = data['uniform']
    top = data['processed'][0].float().topk(2)
    return dict(picked=picked, probability=float(probabilities[picked]), top1=int(top.indices[0]),
                top1_margin=float(top.values[0] - top.values[1]), uniform=uniform,
                cdf_lower=lower, cdf_upper=upper, cdf_margin=min(uniform - lower, upper - uniform),
                cdf_replay=int(torch.searchsorted(cdf, torch.tensor(uniform, dtype=torch.float64), right=True)))


comparisons = []
if manifest['cases']:
    baseline = manifest['cases'][0]
    left_profile = margin.profile(Path(baseline['margin_input']))
    for candidate in manifest['cases'][1:]:
        right_profile = margin.profile(Path(candidate['margin_input']))
        flips, skipped = margin.flips(left_profile, right_profile)
        shared = sorted(set(left_profile) & set(right_profile))
        same_prefix = [key for key in shared if left_profile[key]['prefix'] == right_profile[key]['prefix']]
        assert len(same_prefix) + len(skipped) == len(shared)
        same_uniform = all(left_profile[key]['uniform'] == right_profile[key]['uniform'] for key in shared)
        first = None
        raw_identical = 0
        max_tv = 0.
        for _, generation in same_prefix:
            left = torch.load(DUMP / f"admit{baseline['admission']}-gen{generation}.pt", map_location='cpu', weights_only=True)
            right = torch.load(DUMP / f"admit{candidate['admission']}-gen{generation}.pt", map_location='cpu', weights_only=True)
            raw_identical += int(torch.equal(left['raw'], right['raw']))
            p, q = left['probabilities'].double(), right['probabilities'].double()
            p, q = p / p.sum(), q / q.sum()
            tv = float((p-q).abs().sum() / 2)
            max_tv = max(max_tv, tv)
            if first is None and left['committed'] != right['committed']:
                first = dict(generation=generation, prefix_sha256=left['prefix_sha256'],
                             same_uniform=left['uniform'] == right['uniform'],
                             left=sample_stats(left), right=sample_stats(right), distribution_tv=tv,
                             raw_logit_max_abs=float((left['raw'].float()-right['raw'].float()).abs().max()))
        item = dict(left_case=baseline['case'], right_case=candidate['case'], right_mode=candidate['mode'],
                    shared_positions=len(shared), compared_same_prefix=len(same_prefix),
                    skipped_different_prefix=len(skipped), all_shared_uniforms_equal=same_uniform,
                    equal_raw_rows=raw_identical, max_same_prefix_tv=max_tv,
                    top1_flips=len(flips), first_sampled_divergence=first,
                    output_ids_identical=baseline['output_ids_sha256'] == candidate['output_ids_sha256'])
        comparisons.append(item)
        print(json.dumps(item), flush=True)
(OUT / 'capture-comparison.json').write_text(json.dumps(comparisons, indent=2))
print('VERIFIED: complete captures, #1170-compatible fields and RICH@0 address, exact prefixes and draws, and matched-prefix comparisons', flush=True)
