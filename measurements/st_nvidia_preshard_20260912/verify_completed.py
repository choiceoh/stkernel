"""Audit the completed offline target; do not alter the published conversion.

The converter already reads every rank tensor back and records file hashes.
This independently checks source coverage, rank headers, vision source bytes,
metadata and the checksum list, then writes a separate verification receipt.
"""
import argparse
import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from safetensors import safe_open

from engine.base.checkpoint import Checkpoint
from engine.base.loader import RankLoader
from engine.profiles.glm53 import vision
from engine.profiles.glm53.preshard_modelopt import file_hash, plan, tensor_hash


def verify(ckpt, directory):
    torch.set_num_threads(2)
    saved = json.loads((directory / 'preshard-manifest.json').read_text())
    if saved['layers'] != list(range(45)) or saved['world'] != 4:
        raise ValueError('expected complete GLM-5.3 TP4 target')
    _, groups, current = plan(ckpt, saved['layers'])
    for key in ('weight_layout', 'world', 'layers', 'tensors_per_rank',
                'payload_bytes_per_rank', 'vision_payload_bytes',
                'source_config_sha256', 'source_index_sha256'):
        if saved[key] != current[key]:
            raise ValueError(('conversion plan changed', key))
    specs = {s.name: s for _, group in groups for s in group}
    if len(saved['rank_files']) != 4:
        raise ValueError('missing rank verification record')
    ranks = []
    for rank, entry in enumerate(saved['rank_files']):
        path = directory / f'rank{rank}of4.safetensors'
        if entry['name'] != path.name or entry['bytes'] != path.stat().st_size:
            raise ValueError('rank file size/name mismatch')
        if not entry['all_tensor_bytes_exact'] or entry['tensors_verified'] != len(specs):
            raise ValueError('converter did not verify every rank tensor')
        # Use the independent standard safetensors reader for layout checks.
        with safe_open(str(path), framework='pt', device='cpu') as reader:
            metadata = reader.metadata()
            if metadata['rank'] != str(rank) or metadata['weight_layout'] != saved['weight_layout']:
                raise ValueError('rank identity mismatch')
            keys = {k for k in reader.keys() if not k.startswith('__st_padding__.')}
            if keys != set(specs):
                raise ValueError('rank tensor set mismatch')
            for name, spec in specs.items():
                if tuple(reader.get_slice(name).get_shape()) != tuple(spec.shape):
                    raise ValueError(('rank tensor shape mismatch', rank, name))
        ranks.append(dict(entry, standard_safetensors_header_verified=True))

    original = Checkpoint(str(ckpt))
    target = RankLoader(directory / vision.FILE)
    vision_specs = vision.specs(vision.load(ckpt))
    for spec in vision_specs:
        source = original.load([spec.name], max_run=16 << 20)[spec.name]
        output = target.load([spec.name], device='cpu', max_run=16 << 20)[spec.name]
        if output.dtype != source.dtype or output.shape != source.shape or tensor_hash(output) != tensor_hash(source):
            raise ValueError(('vision source byte mismatch', spec.name))
        del source, output
    if file_hash(directory / vision.FILE) != saved['vision']['sha256']:
        raise ValueError('vision file hash mismatch')
    for item in saved['metadata_files']:
        if file_hash(directory / item['name']) != item['sha256'] or file_hash(ckpt / item['name']) != item['sha256']:
            raise ValueError(('metadata source mismatch', item['name']))
    expected = {e['name']: e['sha256'] for e in saved['rank_files'] + [saved['vision']]}
    listed = {}
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name in listed:
            raise ValueError('duplicate checksum entry')
        listed[name] = digest
    if listed != expected:
        raise ValueError('checksum manifest mismatch')
    return dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_revision=saved['source_revision'], source_coverage=current['source_coverage'],
                rank_files=ranks, rank_byte_readback='all tensors checked by the completed converter',
                rank_hashes_recomputed_by_this_audit=False,
                vision_tensors_source_bytes_verified=len(vision_specs),
                vision=saved['vision'], metadata_files_verified=len(saved['metadata_files']),
                manifest_sha256=file_hash(directory / 'preshard-manifest.json'),
                checksum_list_sha256=file_hash(directory / 'SHA256SUMS'),
                verifier_sha256=file_hash(__file__), live_serving_compatible=False,
                output_bytes=sum(p.stat().st_size for p in directory.iterdir() if p.is_file()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.ckpt, args.directory)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
