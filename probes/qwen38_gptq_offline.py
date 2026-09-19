"""Build 330K GPTQ packs on RTX 5050 and score separate, frozen held-out H.

This proves weight packing error only. A later fleet boot must load these exact
files before the consumer quality, acceptance, and speed comparison can count.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from probes.qwen38_gptq_subset import file_sha


def check_fit(report):
    records = report['records']
    if (not report['statistics_valid'] or report['sites'] != 193 or len(records) != 193
            or len({r['name'] for r in records}) != 193
            or min(r['ntok'] for r in records) < 330000):
        raise ValueError('all 193 sites must have at least 330000 real fit rows')


def verify_lane():
    if os.environ.get('ST_QWEN_5050_LOCK') != '/lane.lock':
        raise RuntimeError('run through the host runner holding gpu-probe.lock')
    with open('/lane.lock', 'r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
    raise RuntimeError('the host runner no longer holds gpu-probe.lock')


def manifest_files(manifest, directory):
    """Verify every exact cache byte that the 5050 result attests."""
    entries = manifest['packs']
    if (manifest['stage'] != 'offline_packed_and_scored'
            or manifest['serving_gptq_verified'] is not False or len(entries) != 385
            or len({p['filename'] for p in entries}) != 385):
        raise ValueError('need the complete offline pack manifest, not a serving claim')
    for row in entries:
        filename = row['filename']
        if Path(filename).name != filename or not filename.endswith('.pt'):
            raise ValueError('invalid cache filename')
        path = Path(directory) / filename
        if path.stat().st_size != row['bytes'] or file_sha(path) != row['sha256']:
            raise ValueError('pack bytes differ from the evaluated 5050 artifact: ' + filename)


def load_checked(path, row, weights_id, minimum):
    import torch
    from probes.qwen38_gptq_audit import validate_blob
    blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    actual = validate_blob(blob, row['name'], row['width'], weights_id, minimum)
    for field in ('ntok', 'hessian_sha256', 'amax_sha256'):
        if actual[field] != row[field]:
            raise ValueError('statistics changed after their fleet audit: ' + str(path))
    return blob


def pack_index(directory):
    import torch
    result = {}
    for path in sorted(Path(directory).glob('*.pt')):
        blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        identity = blob['identity']
        key = identity['name'], identity.get('kind', 'w4'), identity['calibration']
        if key in result:
            raise ValueError('ambiguous cache identity')
        result[key] = path
    return result


def run(args):
    import torch
    from safetensors import safe_open
    from engine.kernels.dense.store import PackStore
    from engine.kernels.dense.packing import fp8_rtn, mk_w4_dequant, FP8_BLOCK
    from engine.profiles.qwen38.net import HEAD_NAME
    from probes.qwen38_gptq_score import output_error
    verify_lane()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if 'RTX 5050' not in torch.cuda.get_device_name():
        raise ValueError('this experiment is explicitly for the RTX 5050 lane')
    fit = json.loads((args.inputs / 'fit330-audit.json').read_bytes())
    held = json.loads((args.inputs / 'heldout-audit.json').read_bytes())
    subset = json.loads((args.inputs / 'dense.json').read_bytes())
    check_fit(fit)
    if (fit['rank'] != held['rank'] or fit['rank'] != subset['rank']
            or fit['weights_id'] != held['weights_id'] or fit['weights_id'] != subset['weights_id']):
        raise ValueError('fit, held-out, and checkpoint identities differ')
    held_rows = {r['name']: r for r in held['records']}
    if set(held_rows) != {r['name'] for r in fit['records']}:
        raise ValueError('held-out sites differ from fit')
    if file_sha(args.inputs / 'dense.safetensors') != subset['sha256']:
        raise ValueError('exported weight bytes changed during transfer')
    # Validate all H before any expensive work; in particular an old 131K blob
    # may never silently replace the newly collected 330K one.
    for row in fit['records']:
        for label, record, minimum in (('fit330', row, 330000), ('heldout', held_rows[row['name']], 50512)):
            path = args.inputs / label / 'mkcalib' / f"rank{fit['rank']}" / (row['name'] + '.pt')
            load_checked(path, record, fit['weights_id'], minimum)
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.out / 'fit330'
    root.mkdir()
    (root / 'mkcalib').symlink_to(args.inputs / 'fit330/mkcalib', target_is_directory=True)
    store = PackStore(root, fit['rank'], fit['weights_id'], require_identity=True)
    rtn = pack_index(args.inputs / 'fit330/st-dense-packs')
    cases, start = [], time.time()
    progress = args.out / 'progress.json'
    with safe_open(str(args.inputs / 'dense.safetensors'), framework='pt', device='cpu') as weights:
        for index, row in enumerate(fit['records']):
            verify_lane()
            name = row['name']
            raw = weights.get_tensor(row['key'])
            if raw.ndim != 2 or raw.shape[1] > row['width']:
                raise ValueError('audited width would truncate the checkpoint projection')
            weight = torch.nn.functional.pad(raw, (0, row['width'] - raw.shape[1])).to('cuda')
            digest = store.weight_digest(weight)
            weight_sha = digest.of(weight)
            held_path = args.inputs / 'heldout/mkcalib' / f"rank{fit['rank']}" / (name + '.pt')
            h = torch.load(held_path, map_location='cpu', mmap=True, weights_only=True)['H'].to('cuda')
            for kind in (('fp8',) if name == HEAD_NAME else ('w4', 'fp8')):
                if kind == 'w4':
                    baseline = torch.load(rtn[name, kind, 'rtn'], map_location='cuda', weights_only=True)
                    identity = baseline['identity']
                    if (identity['weight'] != weight_sha or identity['algorithm'] != store.algorithm
                            or identity['smooth'] != 'none' or not identity['per_row']):
                        raise ValueError('RTN source or packing algorithm differs from the 330K candidate')
                    q = mk_w4_dequant(baseline['data'], baseline['scale'], weight.shape[0], rgs=baseline['rowscale'])
                    del baseline
                else:
                    q8, scale = fp8_rtn(weight)
                    q = q8.float() * scale.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
                    del q8, scale
                baseline_error = output_error(weight, q[:weight.shape[0], :weight.shape[1]], h)
                del q
                if kind == 'w4':
                    packed = store.pack(weight, name, digest=digest)
                    if not packed.calibrated:
                        raise ValueError('330K GPTQ unexpectedly fell back to RTN')
                    q = mk_w4_dequant(packed.data, packed.scale, weight.shape[0], rgs=packed.rowscale)
                    del packed
                else:
                    packed = store.pack_fp8(weight, name, digest=digest)
                    if packed is None:
                        raise ValueError('330K FP8 GPTQ unexpectedly fell back to RTN')
                    q8, scale = packed
                    q = q8.float() * scale.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
                    del q8, scale, packed
                candidate_error = output_error(weight, q[:weight.shape[0], :weight.shape[1]], h)
                del q
                cases.append(dict(name=name, key=row['key'], lane=kind, weight_sha256=weight_sha,
                                  fit_rows=row['ntok'], fit_hessian_sha256=row['hessian_sha256'],
                                  heldout_rows=held_rows[name]['ntok'],
                                  heldout_hessian_sha256=held_rows[name]['hessian_sha256'],
                                  rtn=baseline_error, gptq=candidate_error))
            del weight, raw, h, digest
            store.release_pages()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            state = dict(sites_done=index + 1, sites_total=193, rank=fit['rank'], elapsed_seconds=time.time() - start)
            progress.with_suffix('.tmp').write_text(json.dumps(state) + '\n')
            progress.with_suffix('.tmp').replace(progress)
            print(json.dumps(dict(state, site=row['key'])), flush=True)
    files = []
    index = pack_index(root / 'st-dense-packs')
    for row in fit['records']:
        for kind in (('fp8',) if row['name'] == HEAD_NAME else ('w4', 'fp8')):
            path = index[row['name'], kind, row['hessian_sha256']]
            identity = torch.load(path, map_location='cpu', mmap=True, weights_only=True)['identity']
            files.append(dict(filename=path.name, bytes=path.stat().st_size, sha256=file_sha(path), identity=identity))
    if len(files) != 385 or store.stats['gptq'] != 192 or store.stats['fp8_gptq'] != 193:
        raise ValueError('incomplete 330K GPTQ build')
    summary = {}
    for kind in ('w4', 'fp8'):
        lane = [r for r in cases if r['lane'] == kind]
        summary[kind] = dict(sites=len(lane),
            improved=sum(r['gptq']['relative_rmse'] < r['rtn']['relative_rmse'] for r in lane),
            worsened=sum(r['gptq']['relative_rmse'] > r['rtn']['relative_rmse'] for r in lane),
            median_rtn_rmse=statistics.median(r['rtn']['relative_rmse'] for r in lane),
            median_gptq_rmse=statistics.median(r['gptq']['relative_rmse'] for r in lane),
            median_paired_ratio=statistics.median(r['gptq']['relative_rmse'] / r['rtn']['relative_rmse'] for r in lane))
    result = dict(stage='offline_packed_and_scored', serving_gptq_verified=False,
                  source_sha=args.source_sha, engine_tree=args.engine_tree,
                  rank=fit['rank'], weights_id=fit['weights_id'], minimum_fit_rows=fit['minimum_rows'],
                  minimum_heldout_rows=held['minimum_rows'], algorithm=store.algorithm,
                  device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  input_weight_sha256=subset['sha256'], fit_audit_sha256=file_sha(args.inputs / 'fit330-audit.json'),
                  heldout_audit_sha256=file_sha(args.inputs / 'heldout-audit.json'),
                  summary=summary, cases=cases, packs=files, pack_counters=dict(store.stats),
                  scope=__doc__, elapsed_seconds=time.time() - start)
    manifest_files(result, root / 'st-dense-packs')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--engine-tree', required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
