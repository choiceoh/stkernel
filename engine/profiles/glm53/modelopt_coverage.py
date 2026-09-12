"""Account for every original tensor; the fleet's DFlash2 replaces native MTP.

This reads safetensors headers only. MTP remains in the source checkpoint and
is intentionally absent from the TP rank files and the replicated vision file.
"""
import hashlib
import json
from pathlib import Path
import struct


def source_coverage(ckpt, text_sources, vision_sources):
    root = Path(ckpt).resolve()
    config = json.loads((root / 'config.json').read_text())['text_config']
    first = config['num_hidden_layers']
    count = config.get('num_nextn_predict_layers', 0)
    prefixes = tuple(f'model.language_model.layers.{i}.' for i in range(first, first + count))
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    mapping = index['weight_map']
    text, vision = set(text_sources), set(vision_sources)
    if text & vision:
        raise ValueError('text and vision source groups overlap')
    if missing := (text | vision) - set(mapping):
        raise ValueError(('missing source tensors', sorted(missing)[:8]))
    mtp = {key for key in mapping if key.startswith(prefixes)}
    if mtp & (text | vision):
        raise ValueError('native MTP must not be included in the DFlash2 target layout')
    if unknown := set(mapping) - text - vision - mtp:
        raise ValueError(('unaccounted checkpoint tensors', sorted(unknown)[:8]))
    for prefix in prefixes:
        if not any(key.startswith(prefix) for key in mtp):
            raise ValueError(('declared MTP layer is missing', prefix))

    sizes = {}
    for filename in sorted(set(mapping.values())):
        path = root / filename
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('source shard must be inside checkpoint')
        with path.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError('truncated safetensors header')
            length, = struct.unpack('<Q', raw)
            if length > 64 << 20:
                raise ValueError('safetensors header exceeds bound')
            raw = stream.read(length)
            if len(raw) != length:
                raise ValueError('truncated safetensors header')
            header = json.loads(raw)
        limit = path.stat().st_size - 8 - length
        for key, item in header.items():
            if key == '__metadata__':
                continue
            if key in sizes or mapping.get(key) != filename:
                raise ValueError(('header/index source mismatch', key))
            start, end = item['data_offsets']
            if not 0 <= start <= end <= limit:
                raise ValueError(('source tensor outside shard', key))
            sizes[key] = end - start
    if set(sizes) != set(mapping):
        raise ValueError('source index references absent header tensors')
    total = sum(sizes.values())
    if index.get('metadata', {}).get('total_size', total) != total:
        raise ValueError('source tensor byte total differs from index')

    def group(keys):
        return dict(tensors=len(keys), payload_bytes=sum(sizes[k] for k in keys),
                    source_keys_sha256=hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest())

    return dict(source_tensors=len(sizes), source_payload_bytes=total,
                text=group(text), vision=group(vision),
                intentionally_excluded_mtp=dict(group(mtp), layers=list(range(first, first + count)),
                    reason='ST drafts with DFlash2; native MTP is not used.',
                    retained_in_source_checkpoint=True, written_to_preshards=False),
                unaccounted_tensors=0, all_source_tensors_accounted_for=True)
