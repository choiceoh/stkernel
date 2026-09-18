"""Splice one layer's routed experts as BF16 into a ModelOpt NVFP4 checkpoint.

Purpose: the head-NLL A/B that measures what NVFP4 quantisation costs in nats
(layer-by-layer: vLLM boot of the spliced checkpoint vs the pure NVFP4 one,
prompt_logprobs on the same corpus, paired positions). One layer's BF16
experts add ~3.6 GB and run through vLLM's exclude_modules path (the same
mechanism that keeps layer 0/1 self_attn in BF16).

Input: a ModelOpt NVFP4 checkpoint dir (191 GB, 33 shards), the BF16 expert
dumps of one layer (tools/nvfp4_reanchor_ranks.py's sibling fetch, one
`{name}.bin` per tensor), the layer number. Output: a new checkpoint dir with
- layer-N expert `weight` tensors replaced by their BF16 originals,
- their `weight_scale` tensors dropped,
- hf_quant_config.json exclude_modules extended with the layer-N experts,
everything else byte-identical, streamed (memory bounded per tensor).
"""
import argparse
import json
import os
import struct
from pathlib import Path

import torch


def read_header(path):
    with open(path, 'rb') as f:
        hlen, = struct.unpack('<Q', f.read(8))
        return json.loads(f.read(hlen)), 8 + hlen


def run(args):
    src = Path(args.source)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    bins = Path(args.bf16_dir)
    layer_tag = f"layers.{args.layer}.mlp.experts."

    index_file = src / 'model.safetensors.index.json'
    index = json.load(open(index_file)) if index_file.is_file() else None
    if index is not None:
        shards = sorted(set(index['weight_map'].values()))
    else:
        shards = sorted(p.name for p in src.glob('*.safetensors'))

    spliced, dropped, copied_bytes = 0, 0, 0
    for shard in shards:
        header, data_start = read_header(src / shard)
        out_entries = {}
        with open(src / shard, 'rb') as src_f, open(dst / shard, 'wb') as out_f:
            out_f.write(b'\x00' * 8)                              # header length, patched after
            for name in sorted(header, key=lambda n: header[n]['data_offsets'][0]):
                if name == '__metadata__':
                    continue
                meta = header[name]
                lo, hi = meta['data_offsets']
                size = hi - lo
                if layer_tag in name and name.endswith('_scale'):
                    dropped += 1
                    continue
                if layer_tag in name and name.endswith('.weight'):
                    raw = (bins / f'{name}.bin').read_bytes()
                    bf = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
                    out_f.write(raw)
                    out_entries[name] = dict(dtype='BF16', shape=list(bf.shape),
                                             data_offsets=[out_f.tell() - len(raw), out_f.tell()])
                    spliced += 1
                    copied_bytes += len(raw)
                    continue
                src_f.seek(data_start + lo)
                remaining = size
                start = out_f.tell()
                out_entries[name] = dict(dtype=meta['dtype'], shape=meta['shape'],
                                         data_offsets=[start, 0])
                while remaining > 0:
                    chunk = src_f.read(min(1 << 24, remaining))
                    out_f.write(chunk)
                    remaining -= len(chunk)
                out_entries[name]['data_offsets'] = [start, out_f.tell()]
                copied_bytes += size
            final = dict(out_entries)
            final['__metadata__'] = header.get('__metadata__', {'format': 'pt'})
            blob = json.dumps(final, separators=(',', ':')).encode()
            out_f.seek(0)
            out_f.write(struct.pack('<Q', len(blob)))
            out_f.write(blob)
    print(json.dumps(dict(event='spliced', layer=args.layer, weight_tensors=spliced,
                          scale_tensors_dropped=dropped, copied_gb=round(copied_bytes / 2**30, 1))), flush=True)

    for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'processor_config.json', 'chat_template.jinja'):
        if (src / name).is_file():
            shutil_copy(src / name, dst / name)
    quant = json.load(open(src / 'hf_quant_config.json'))
    pattern = f"model.language_model.{layer_tag}*"
    if pattern not in quant['quantization']['exclude_modules']:
        quant['quantization']['exclude_modules'].append(pattern)
    json.dump(quant, open(dst / 'hf_quant_config.json', 'w'), indent=4)
    if index is not None:
        json.dump(index, open(dst / 'model.safetensors.index.json', 'w'))
    print(json.dumps(dict(event='config-patched', exclude=pattern)), flush=True)


def shutil_copy(a, b):
    import shutil
    shutil.copyfile(a, b)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--bf16-dir', type=Path, required=True)
    ap.add_argument('--layer', type=int, required=True)
    run(ap.parse_args())
