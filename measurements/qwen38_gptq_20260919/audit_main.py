"""Audit clean-main serving against the original 330K collection and exact packs."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--boot', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/cache'))
    p.add_argument('--packs', type=Path)
    p.add_argument('--rtn', action='store_true')
    args = p.parse_args()
    import torch
    from safetensors import safe_open
    from engine.profiles.qwen38 import calibration, facts, mtp_side, vision
    from engine.kernels.dense.store import PackStore
    from probes.qwen38_gptq_audit import counters, validate_blob
    torch.set_num_threads(2)
    ref = json.loads(args.reference.read_bytes())
    rank = ref['rank']
    checkpoint = Path('/home/choiceoh/models/st-qwen38-tep4')
    F = facts.load(checkpoint)
    files = sorted(checkpoint.glob('rank*of4.safetensors'))
    if (checkpoint / vision.FILE).is_file():
        files.append(checkpoint / vision.FILE)
    files += [mtp_side.path(mtp_side.DIRS['bf16'], rank, 'bf16'), checkpoint / facts.ple_file(rank, facts.TP)]
    with safe_open(str(checkpoint / f'rank{rank}of4.safetensors'), framework='pt', device='cpu') as weights:
        identity = calibration.identity(weights.metadata(), files, F.config, hc_fp8=False)
    assert identity == ref['weights_id'], 'latest serving checkpoint differs from calibration'
    store = PackStore(args.root, rank, weights_id=identity, require_identity=True)
    result = dict(rank=rank, weights_id=identity, algorithm=store.algorithm)
    if args.boot:
        seen = counters(json.loads(args.boot.read_bytes())['root'])
        assert seen.get('calibration_GiB') == 0 and seen.get('calibration_deferred') == 0
        if args.rtn:
            assert seen.get('packs_rtn') == 192 and seen.get('packs_gptq', 0) == seen.get('packs_fp8_gptq', 0) == 0
            result.update(serving_rtn_verified=True, collection_disabled=True)
        else:
            assert seen.get('packs_gptq') == seen.get('packs_cache') == 192
            assert seen.get('packs_fp8_gptq') == seen.get('packs_fp8_cache') == 193
            assert seen.get('packs_built', 0) == seen.get('packs_fp8_built', 0) == 0
            result.update(serving_gptq_verified=True, evaluated_pack_bytes_consumed=True, collection_disabled=True)
        result['boot_sha256'] = digest(args.boot)
    if not args.rtn:
        wanted = {r['name']: r for r in ref['records']}
        for name, row in wanted.items():
            blob = torch.load(args.root / 'mkcalib' / f'rank{rank}' / (name + '.pt'), map_location='cpu', mmap=True, weights_only=True)
            actual = validate_blob(blob, name, row['width'], identity, 330000)
            assert all(actual[k] == row[k] for k in ('ntok', 'hessian_sha256', 'amax_sha256'))
        packs = []
        for path in sorted((args.root / 'st-dense-packs').glob('*.pt')):
            blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            ident = blob['identity']
            if ident['name'] not in wanted or ident['calibration'] != wanted[ident['name']]['hessian_sha256']:
                continue
            assert ident['algorithm'] == store.algorithm
            packs.append(dict(filename=path.name, bytes=path.stat().st_size, sha256=digest(path)))
        assert len(packs) == 385
        if args.packs:
            previous = json.loads(args.packs.read_bytes())
            assert previous['weights_id'] == identity and previous['packs'] == packs
        result['packs'] = packs
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'packs'}), flush=True)


if __name__ == '__main__':
    main()
