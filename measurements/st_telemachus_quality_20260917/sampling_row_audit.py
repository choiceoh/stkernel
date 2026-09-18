"""Is the engine's sampling row a copy of the model's row? Sweep every captured position.

`raw` is bf16 and `processed` is fp32, so the test is exact only if it accounts for the upcast: an
fp32 value that came from bf16 has its low 16 bits zero and its top 16 bits equal to the bf16 pattern.
Comparing the *strided byte slices* tests that for all 154,880 lanes at C speed -- which is what makes
a 2,313-file sweep feasible on a host with neither torch nor numpy.

This is written down because the naive version lies. Slicing the bf16 storage with fp32's four bytes
per element reads 77,440 lanes instead of 154,880 and then reports a half-masked vocabulary that does
not exist; the same mistake with the *wrong* tensor produces a transform that was never applied. Run
the reader against `torch.load` on one file before believing a sweep (the receipt records that check).

    python3 sampling_row_audit.py --out /tmp/sampling-row-audit.json
"""
from __future__ import annotations

import argparse
import array
import collections
import glob
import json
import os
import pickle
import struct
import zipfile

DTYPES = {'FloatStorage': ('f', 4), 'DoubleStorage': ('d', 8), 'HalfStorage': ('e', 2),
          'LongStorage': ('q', 8), 'IntStorage': ('i', 4), 'ShortStorage': ('h', 2),
          'CharStorage': ('b', 1), 'ByteStorage': ('B', 1), 'BoolStorage': ('?', 1),
          'BFloat16Storage': ('bf16', 2)}
NEG_INF = b'\x00\x00\x80\xff'


class StorageRef:
    def __init__(self, dtype, key, numel, location):
        self.dtype, self.key, self.numel, self.location = dtype, key, numel, location


class TensorRef:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


_STORAGE_TYPES = {name: type(name, (object,), {}) for name in DTYPES}


def _rebuild_tensor(storage, offset, size, stride, *rest):
    return TensorRef(storage, offset, size, stride)


class Reader:
    """A .pt archive: pickle the structure with stubs, then read storages straight out of the zip."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        self.names = self.zip.namelist()
        self.root = self._load()

    def _load(self):
        class U(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith('torch'):
                    if name in DTYPES:
                        return _STORAGE_TYPES[name]
                    if name.startswith('_rebuild_tensor'):
                        return _rebuild_tensor
                    return type('Stub', (object,), {})
                return super(U, self).find_class(module, name)

            def persistent_load(self, pid):
                kind, storage_type, key, location, numel = pid
                if kind != 'storage':
                    raise ValueError(f'unexpected persistent id {kind}')
                return StorageRef(getattr(storage_type, '__name__', str(storage_type)), key, numel, location)

        entry = 'archive/data.pkl' if 'archive/data.pkl' in self.names else next(n for n in self.names if n.endswith('data.pkl'))
        with self.zip.open(entry) as handle:
            return U(handle).load()

    def storage_bytes(self, tensor):
        """The tensor's own elements, as bytes, from the storage the archive holds."""
        entry = f'archive/data/{tensor.storage.key}'
        if entry not in self.names:
            raise KeyError(entry)
        with self.zip.open(entry) as handle:
            data = handle.read()
        size = DTYPES[tensor.storage.dtype][1]
        start = tensor.offset * size
        return data[start:start + self.lanes(tensor) * size]

    def lanes(self, tensor):
        total = 1
        for d in tensor.size:
            total *= d
        return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', default='/home/choiceoh/glm53-logs/st-bracket-dumps/'
                                      'st-lossless-logits0918-hold-8f87c211515c/incident-logits')
    ap.add_argument('--vocab', type=int, default=154856, help='tokenizer.json entries plus added tokens')
    ap.add_argument('--out', default='/tmp/sampling-row-audit.json')
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dump, 'admit*-gen*.pt')),
                   key=lambda p: (int(os.path.basename(p).split('-')[0][5:]), int(os.path.basename(p).split('-gen')[1][:-3])))
    if not paths:
        raise SystemExit(f'no captures under {a.dump}')
    per_mode = collections.defaultdict(collections.Counter)
    cuts = collections.Counter()
    transformed, anomalies, padding_top = [], [], []
    for done, path in enumerate(paths, 1):
        reader = Reader(path)
        d = reader.root
        assert d['raw'].storage.dtype == 'BFloat16Storage' and d['processed'].storage.dtype == 'FloatStorage', \
            'the capture schema moved: this sweep comparisons assume bf16 raw against fp32 processed'
        raw = reader.storage_bytes(d['raw'])
        proc = reader.storage_bytes(d['processed'])
        width = reader.lanes(d['processed'])
        mode, gen, admit = d['mode'], d['generation'], d['admission']
        pick = d['picks'][0]
        if width != width or reader.lanes(d['raw']) != width:                       # both tensors are one row
            anomalies.append(dict(admit=admit, gen=gen, mode=mode, width=width, raw_lanes=reader.lanes(d['raw'])))
        same = (proc[2:a.vocab * 4:4] == raw[0:a.vocab * 2:2]) and (proc[3:a.vocab * 4:4] == raw[1:a.vocab * 2:2])
        tail = proc[a.vocab * 4:width * 4]
        cuts[(width - a.vocab, tail == NEG_INF * (width - a.vocab))] += 1
        if not same:
            processed = array.array('f'); processed.frombytes(proc[:width * 4])
            halves = array.array('H'); halves.frombytes(raw[0:width * 2])
            finite = [i for i, x in enumerate(processed[:a.vocab]) if x != float('-inf')]
            transformed.append(dict(admit=admit, gen=gen, mode=mode, pick=pick, finite_lanes=len(finite),
                                    raw_bf16_top1=max(range(a.vocab), key=lambda i: struct.unpack('<f', struct.pack('<I', halves[i] << 16))[0]),
                                    proc_top1=max(finite, key=lambda i: processed[i]) if finite else None))
        if pick >= a.vocab:
            anomalies.append(dict(admit=admit, gen=gen, mode=mode, pick=pick, why='pick past the tokenizer vocabulary'))
        pad = array.array('H'); pad.frombytes(raw[a.vocab * 2:width * 2])
        if len(pad):
            padding_top.append(struct.unpack('<f', struct.pack('<I', max(pad) << 16))[0])
        per_mode[mode]['positions'] += 1
        per_mode[mode]['processed_equals_raw'] += int(same)
        if done % 500 == 0:
            print(f'  {done}/{len(paths)}', flush=True)

    summary = dict(files=len(paths), vocab=a.vocab, dump=a.dump,
                   processed_equals_raw_on_decodable=sum(c['processed_equals_raw'] for c in per_mode.values()),
                   per_mode={str(m): dict(c) for m, c in sorted(per_mode.items())},
                   cut_lanes={f'{n} lanes past the vocabulary, all -inf={m}': c for (n, m), c in cuts.items()},
                   transforms_applied=transformed[:10], transforms_applied_count=len(transformed),
                   anomalies=anomalies[:10], anomaly_count=len(anomalies),
                   padding_lane_bf16_top_max=round(max(padding_top), 4) if padding_top else None)
    with open(a.out, 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))
    print('per_mode', summary['per_mode'])
    print('WROTE', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
