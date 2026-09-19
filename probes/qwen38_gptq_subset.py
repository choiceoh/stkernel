"""Stream only the audited dense tensors out of a fleet checkpoint, without CUDA.

No prompts, activations, or expert tensors are exported. The original checkpoint
is read in bounded chunks; the small safetensors file retains the exact bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def subset(source, output, keys):
    source, output = Path(source), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('an exported checkpoint must not be overwritten')
    keys = sorted(keys)
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate tensor key')
    before = source.stat()
    with source.open('rb') as src:
        size = struct.unpack('<Q', src.read(8))[0]
        if not 2 <= size <= 100_000_000:
            raise ValueError('invalid safetensors header length')
        raw = src.read(size)
        header = json.loads(raw)
        new, records, total = {}, [], 0
        for key in keys:
            entry = header[key]
            shape, offsets = entry['shape'], entry['data_offsets']
            if (entry['dtype'] != 'BF16' or len(shape) != 2 or min(shape) <= 0
                    or len(offsets) != 2 or offsets[0] < 0
                    or offsets[1] - offsets[0] != shape[0] * shape[1] * 2
                    or 8 + size + offsets[1] > before.st_size):
                raise ValueError('expected a complete dense BF16 matrix: ' + key)
            length = offsets[1] - offsets[0]
            new[key] = dict(dtype=entry['dtype'], shape=shape, data_offsets=[total, total + length])
            total += length
        encoded = json.dumps(new, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix('.partial')
        try:
            with temp.open('xb') as dst:
                dst.write(struct.pack('<Q', len(encoded)))
                dst.write(encoded)
                for key in keys:
                    first, last = header[key]['data_offsets']
                    src.seek(8 + size + first)
                    remaining, h = last - first, hashlib.sha256()
                    while remaining:
                        block = src.read(min(1 << 20, remaining))
                        if not block:
                            raise ValueError('truncated checkpoint tensor')
                        dst.write(block)
                        h.update(block)
                        remaining -= len(block)
                    records.append(dict(key=key, shape=new[key]['shape'], dtype='BF16', sha256=h.hexdigest()))
                dst.flush()
                os.fsync(dst.fileno())
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError('checkpoint changed while exporting')
            temp.rename(output)
        finally:
            temp.unlink(missing_ok=True)
    return dict(source_file=source.name, source_size=before.st_size,
                source_mtime_ns=before.st_mtime_ns, source_header_sha256=hashlib.sha256(raw).hexdigest(),
                filename=output.name, sha256=file_sha(output), tensors=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    audit = json.loads(args.audit.read_bytes())
    if audit['sites'] != 193 or len(audit['records']) != 193:
        raise ValueError('expected all 193 Qwen dense sites')
    result = subset(args.source, args.out, [r['key'] for r in audit['records']])
    result.update(rank=audit['rank'], weights_id=audit['weights_id'],
                  purpose='exact dense weight bytes only; no calibration rows or GPU work')
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}), flush=True)


if __name__ == '__main__':
    main()
