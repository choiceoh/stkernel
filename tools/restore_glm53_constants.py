"""Write a separate GLM rank with verified original FP32 control constants.

Only the selected router biases / KDA constants change. All other payload
bytes, including routed expert weights and scales, are copied unchanged.
The source rank is read-only. This tool is an incident candidate, not an
instruction to promote a checkpoint without a matched quality replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_header(stream):
    size = struct.unpack('<Q', stream.read(8))[0]
    if not 2 <= size <= 64 << 20:
        raise ValueError('invalid rank header size')
    raw = stream.read(size)
    return json.loads(raw), 8 + size


def copy_with_constants(source, destination, replacements, metadata):
    """Stream a rank into a new file; reject overwrites and malformed patches."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not replacements:
        raise ValueError('no control constants selected')
    with source.open('rb') as reader:
        before = os.fstat(reader.fileno())
        header, base = read_header(reader)
        patches = []
        for name, data in replacements.items():
            if not name.endswith(('.moe.bias', '.kda.A_log', '.kda.dt_bias')):
                raise ValueError('not a GLM control constant: ' + name)
            entry = header[name]
            lo, hi = entry['data_offsets']
            if entry['dtype'] != 'F32' or hi - lo != len(data) or not 0 <= lo < hi <= before.st_size - base:
                raise ValueError('invalid replacement shape or dtype: ' + name)
            patches.append((lo, hi, data))
        patches.sort()
        if any(a[1] > b[0] for a, b in zip(patches, patches[1:])):
            raise ValueError('overlapping control constants')
        if (header.get('__metadata__') or {}).get('fp32_constants_sha256'):
            raise ValueError('start from the original unmodified rank')
        header['__metadata__'] = dict(header.get('__metadata__') or {}, **metadata)
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
        copied = changed = 0
        payload_hash = hashlib.sha256()
        try:
            with os.fdopen(fd, 'wb') as writer:
                writer.write(struct.pack('<Q', len(raw))); writer.write(raw)
                cursor = 0
                for lo, hi, replacement in patches + [(before.st_size - base, before.st_size - base, b'')]:
                    while cursor < lo:
                        chunk = reader.read(min(16 << 20, lo - cursor))
                        if not chunk:
                            raise EOFError('source rank truncated during copy')
                        writer.write(chunk); payload_hash.update(chunk)
                        cursor += len(chunk); copied += len(chunk)
                    previous = reader.read(hi - lo)
                    if len(previous) != hi - lo:
                        raise EOFError('source constant truncated during copy')
                    writer.write(replacement); payload_hash.update(replacement)
                    changed += len(replacement); cursor = hi
                writer.flush(); os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('source rank changed during copy')
            # Atomic publication without replacing a concurrently created file.
            os.link(temporary, destination)
        finally:
            os.unlink(temporary)
    return dict(copied_bytes=copied, restored_bytes=changed, tensors=len(replacements),
                payload_sha256=payload_hash.hexdigest(), metadata=metadata)


def original_constants(rank_file, checkpoint, rank, kinds):
    import contextlib
    import torch
    from safetensors import safe_open
    from engine.profiles.glm53.facts import architecture
    from engine.profiles.glm53.specs import all_specs
    torch.set_num_threads(1)
    root = Path(checkpoint)
    facts = architecture(json.loads((root / 'config.json').read_text()))
    specs = {spec.name: spec for spec in all_specs(facts)}
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    names = [name for name in specs if
             (kinds in ('router', 'all') and name.endswith('.moe.bias')) or
             (kinds in ('kda', 'all') and name.endswith(('.kda.A_log', '.kda.dt_bias')))]
    expected_count = {'router': 42, 'kda': 68, 'all': 110}[kinds]
    if len(names) != expected_count:
        raise ValueError('unexpected model control-constant geometry')
    restored, digest = {}, hashlib.sha256()
    with contextlib.ExitStack() as stack:
        current = stack.enter_context(safe_open(rank_file, framework='pt', device='cpu'))
        handles = {}
        def load(spec):
            result = {}
            for key in spec.sources:
                shard = index[key]
                if shard not in handles:
                    handles[shard] = stack.enter_context(safe_open(root / shard, framework='pt', device='cpu'))
                result[key] = handles[shard].get_tensor(key)
            return result
        checked = set()
        for name in sorted(names):
            spec = specs[name]; source = load(spec)
            if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in source.values()):
                raise ValueError('original constants must be finite FP32: ' + name)
            expected, actual = spec.build(source, rank, 4), current.get_tensor(name)
            if actual.dtype != torch.float32 or actual.shape != expected.shape:
                raise ValueError('serving control geometry differs: ' + name)
            if not torch.equal(actual, expected.bfloat16().float()):
                raise ValueError('serving values are not the rounded original: ' + name)
            # The full router matrix or both low-rank KDA projections must also
            # match; do not accept a same-shape checkpoint as the same model.
            prefix = name.split('.')[0]
            controls = ([prefix + '.moe.gate'] if '.moe.' in name else
                        [prefix + '.kda.f_b', prefix + '.kda.g_b'])
            for control in controls:
                if control not in checked:
                    s = specs[control]
                    if not torch.equal(current.get_tensor(control), s.build(load(s), rank, 4)):
                        raise ValueError('checkpoint provenance mismatch: ' + control)
                    checked.add(control)
            for key in sorted(source):
                # Hash unsharded original constants: identical identity on all ranks.
                digest.update(key.encode() + b'\0')
                digest.update(source[key].contiguous().view(torch.uint8).numpy().tobytes())
            restored[name] = expected.contiguous().view(torch.uint8).numpy().tobytes()
    return restored, dict(fp32_constants_sha256=digest.hexdigest(), fp32_constants_kinds=kinds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank-file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--rank', required=True, type=int, choices=range(4))
    parser.add_argument('--kinds', required=True, choices=('router', 'kda', 'all'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    replacements, metadata = original_constants(args.rank_file, args.checkpoint, args.rank, args.kinds)
    print(json.dumps(copy_with_constants(args.rank_file, args.output, replacements, metadata), indent=2))


if __name__ == '__main__':
    main()
