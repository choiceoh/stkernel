"""Compare local GLM checkpoints using headers and deterministic CPU samples.

No CUDA, weight conversion, or writes to either checkpoint. Weight distances
compare two quantized models, except dense Red Hat BF16 versus NVIDIA NVFP4;
they are not a full-model accuracy score.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import struct


class Source:
    def __init__(self, root):
        self.root = Path(root)
        index = json.loads((self.root / 'model.safetensors.index.json').read_text())
        self.map = index['weight_map']
        self.headers, self.bases, self.fds = {}, {}, {}
        for name in sorted(set(self.map.values())):
            fd = os.open(self.root / name, os.O_RDONLY)
            size = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
            self.headers.update(json.loads(os.pread(fd, size, 8)))
            self.bases[name], self.fds[name] = size + 8, fd

    def raw(self, key, offset, size):
        name = self.map[key]
        entry = self.headers[key]
        assert 0 <= offset and offset + size <= entry['data_offsets'][1] - entry['data_offsets'][0]
        data = os.pread(self.fds[name], size,
                        self.bases[name] + entry['data_offsets'][0] + offset)
        assert len(data) == size
        return data

    def scalar(self, key):
        return struct.unpack('<f', self.raw(key, 0, 4))[0]

    def stats(self):
        result = defaultdict(Counter)
        for key in self.map:
            entry = self.headers[key]
            category = ('mtp' if '.layers.45.' in key else
                        'vision' if 'model.visual.' in key else
                        'routed_experts' if '.mlp.experts.' in key else
                        'dense_mlp' if any(f'.layers.{i}.mlp.' in key for i in (0, 1, 2)) else
                        'other')
            result[category]['tensors'] += 1
            result[category]['bytes'] += entry['data_offsets'][1] - entry['data_offsets'][0]
            result[category][entry['dtype']] += 1
        return dict(result)


def positions(length, count):
    return sorted({i * (length - 1) // (min(length, count) - 1)
                   for i in range(min(length, count))}) if length > 1 else [0]


def bf16(data):
    return [struct.unpack('<f', struct.pack('<I', value << 16))[0]
            for value in struct.unpack('<' + 'H' * (len(data) // 2), data)]


def fp8(value):
    sign = -1 if value & 128 else 1
    exponent, mantissa = (value >> 3) & 15, value & 7
    assert (value & 127) != 127, 'nonfinite E4M3 weight scale'
    return sign * (mantissa * 2.0 ** -9 if exponent == 0 else
                   (1 + mantissa / 8) * 2.0 ** (exponent - 7))


def block(source, prefix, index, vendor, dense):
    if vendor == 'redhat' and dense:
        return bf16(source.raw(prefix + 'weight', index * 32, 32))
    packed = 'weight_packed' if vendor == 'redhat' else 'weight'
    global_scale = (1 / source.scalar(prefix + 'weight_global_scale') if vendor == 'redhat'
                    else source.scalar(prefix + 'weight_scale_2'))
    scale = fp8(source.raw(prefix + 'weight_scale', index, 1)[0]) * global_scale
    lut = (0, .5, 1, 1.5, 2, 3, 4, 6)
    values = []
    for byte in source.raw(prefix + packed, index * 8, 8):
        for nibble in (byte & 15, byte >> 4):
            values.append(lut[nibble & 7] * (-1 if nibble & 8 else 1) * scale)
    return values


def distance(left, right):
    aa = math.fsum(x * x for x in left)
    bb = math.fsum(x * x for x in right)
    diff = math.fsum((x - y) ** 2 for x, y in zip(left, right))
    return dict(samples=len(left), relative_l2=math.sqrt(diff / max(aa, 1e-30)),
                cosine=math.fsum(x * y for x, y in zip(left, right)) / max(math.sqrt(aa * bb), 1e-30),
                max_abs=max(abs(x - y) for x, y in zip(left, right)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--redhat', required=True)
    ap.add_argument('--nvidia', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    redhat, nvidia = Source(args.redhat), Source(args.nvidia)
    configs = [json.loads((s.root / 'config.json').read_text()) for s in (redhat, nvidia)]
    a, b = configs
    differences = {}
    for section in ('text_config', 'vision_config'):
        differences[section] = {k: [a[section].get(k), b[section].get(k)]
                                for k in a[section].keys() | b[section].keys()
                                if a[section].get(k) != b[section].get(k)}
    tokenizer = [json.loads((s.root / 'tokenizer.json').read_text()) for s in (redhat, nvidia)]
    differences['tokenizer'] = [k for k in tokenizer[0].keys() | tokenizer[1].keys()
                                if tokenizer[0].get(k) != tokenizer[1].get(k)]
    differences['tokenizer_truncation'] = [d['truncation'] for d in tokenizer]
    for filename in ('generation_config.json', 'tokenizer_config.json', 'processor_config.json'):
        a, b = [json.loads((s.root / filename).read_text()) for s in (redhat, nvidia)]
        differences[filename] = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys()
                                 if a.get(k) != b.get(k)}
    controls, changed_dtypes = [], []
    for key in sorted(redhat.map.keys() & nvidia.map.keys()):
        if '.layers.45.' in key or '.mlp.experts.' in key:
            continue
        a, b = redhat.headers[key], nvidia.headers[key]
        if a['shape'] != b['shape'] or a['dtype'] not in ('BF16', 'F32') or b['dtype'] not in ('BF16', 'F32'):
            continue
        count = math.prod(a['shape'])
        values = [[], []]
        for pos in positions(count, 64):
            for i, (source, entry) in enumerate(((redhat, a), (nvidia, b))):
                if entry['dtype'] == 'BF16':
                    values[i].extend(bf16(source.raw(key, pos * 2, 2)))
                else:
                    values[i].append(struct.unpack('<f', source.raw(key, pos * 4, 4))[0])
        row = dict(tensor=key, dtypes=[a['dtype'], b['dtype']], **distance(*values))
        controls.append(row)
        if a['dtype'] != b['dtype']:
            changed_dtypes.append(row)
    projections = []
    for layer in (0, 1, 2, 3, 17, 31, 44):
        dense = layer < 3
        for expert in ([None] if dense else [0, 31, 127, 287]):
            prefix = f'model.language_model.layers.{layer}.mlp.'
            if expert is not None:
                prefix += f'experts.{expert}.'
            for projection in ('up', 'gate', 'down'):
                p = prefix + projection + '_proj.'
                shape = nvidia.headers[p + 'weight']['shape']
                count = math.prod(shape) * 2 // 16
                left, right = [], []
                for index in positions(count, 256):
                    left.extend(block(redhat, p, index, 'redhat', dense))
                    right.extend(block(nvidia, p, index, 'nvidia', dense))
                row = dict(layer=layer, expert=expert, projection=projection,
                           **distance(left, right), nvidia_input_scale=nvidia.scalar(p + 'input_scale'),
                           nvidia_weight_scale=nvidia.scalar(p + 'weight_scale_2'))
                if not dense:
                    row.update(redhat_input_scale=1 / redhat.scalar(p + 'input_global_scale'),
                               redhat_weight_scale=1 / redhat.scalar(p + 'weight_global_scale'))
                projections.append(row)
    result = dict(paths=dict(redhat=args.redhat, nvidia=args.nvidia),
                  method='Headers cover every indexed tensor; values sample 64 elements per floating control tensor and 256 evenly spaced 16-element blocks per selected projection. No model or kernel execution.',
                  interpretation='Routed distances compare two NVFP4 quantizations; dense compares Red Hat BF16 to NVIDIA NVFP4. Neither establishes downstream quality.',
                  stats=dict(redhat=redhat.stats(), nvidia=nvidia.stats()),
                  metadata_differences=differences, control_tensors=len(controls),
                  control_samples=sum(r['samples'] for r in controls),
                  control_exact_tensors=sum(r['max_abs'] == 0 for r in controls),
                  control_changed=[r for r in controls if r['max_abs'] != 0],
                  changed_dtypes=changed_dtypes, projections=projections)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('control_tensors', 'control_samples', 'control_exact_tensors')}, indent=2))
    for dense in (True, False):
        rows = [r for r in projections if (r['layer'] < 3) == dense]
        print('dense' if dense else 'routed', 'projections', len(rows),
              'L2 range', min(r['relative_l2'] for r in rows), max(r['relative_l2'] for r in rows),
              'cosine range', min(r['cosine'] for r in rows), max(r['cosine'] for r in rows))


if __name__ == '__main__':
    main()
