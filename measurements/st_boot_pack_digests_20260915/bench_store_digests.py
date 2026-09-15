"""CPU A/B of the pack store's per-boot identity work on one rank's real calibration blobs.

The boot's cache-hit path per dense weight, as `DenseLinear` drives it: the W4 lane (`pack`) and the FP8 lane
(`pack_fp8`) look their packs up by sha256 of the weight bytes and of the validated, smoothed calibration Hessian.

  old   engine/kernels/dense/store.py at main bb72154d (/old/store_bb72154d.py, beside that commit's packing.py):
        every lane copies and hashes the
        weight, and loads, checks, smooths and hashes the blob itself
  new   this branch: one WeightDigest per weight -- the weight hashes once on the store's worker while the
        calibration is loaded, checked, smoothed and hashed once for all of the weight's lanes

Weights are synthetic BF16 at the rank file's real shapes (rank 3 of GLM-5.3 and the DFlash2 drafter); the
calibration blobs are the real ones, read cold (POSIX_FADV_DONTNEED before every run). Pack caches are stub
blobs of the served layouts filed under the store's own identities, so both variants take the hit path and
must find the same files. No GPU: the boot's device-to-host copy of each weight is a host copy here.

usage: bench_store_digests.py setup ROOT | run ROOT old|new
"""
import hashlib
import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path
from unittest import mock

import torch

RANK = 3
RANK_FILE = Path('/models/rank3of4.safetensors')
DRAFTER_FILE = Path('/drafter/model.safetensors')
CALIBRATION = Path('/calibration')               # mkcalib, mounted read-only
HERE = Path(__file__).resolve().parent

# engine/profiles/glm53/net.py Glm53Net.dense_weight_names, and the norms whose outputs smoothing divides
TARGET = {"kda.in_proj": "self_attn.in_proj_qkvbfg_a", "kda.o_proj": "self_attn.o_proj",
          "mla.qkv_a": "self_attn.fused_qkv_a_proj", "mla.q_b": "self_attn.q_b_proj",
          "mla.o_proj": "self_attn.o_proj", "idx.wq_b": "self_attn.indexer.wq_b",
          "mlp.gate_up": "mlp.gate_up_proj", "mlp.down": "mlp.down_proj",
          "moe.sh_gate_up": "mlp.shared_experts.gate_up_proj", "moe.sh_down": "mlp.shared_experts.down_proj"}
SMOOTHED = {"kda.in_proj", "mla.qkv_a", "mla.q_b", "idx.wq_b", "mlp.gate_up"}
HEAD = "Glm5NextForCausalLM/lm_head"
DRAFT = "DFlash2Qwen3ForCausalLM/model."


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop('__metadata__', None)
    return h


def items():
    """(label, name, shape, smoothed, lanes) in boot order: target projections, head, drafter fc, drafter blocks."""
    h = header(RANK_FILE)
    out = []
    for key in sorted(h, key=lambda k: (int(k.split('.')[0][1:]) if k.startswith('L') else 1 << 30, k)):
        layer, _, suffix = key.partition('.')
        if suffix in TARGET:
            name = f"Glm5NextForCausalLM/model.layers.{layer[1:]}.{TARGET[suffix]}"
            out.append((key, name, tuple(h[key]['shape']), suffix in SMOOTHED, ('w4', 'fp8')))
    out.append(('head', HEAD, tuple(h['head']['shape']), False, ('fp8',)))
    d = header(DRAFTER_FILE)
    out.append(('fc.weight', DRAFT + 'fc', tuple(d['fc.weight']['shape']), False, ('fp8', 'decode-fp8')))
    layers = sorted({int(k.split('.')[1]) for k in d if k.startswith('layers.')})
    for L in layers:
        n = f'layers.{L}.'
        q, k = d[n + 'self_attn.q_proj.weight']['shape'], d[n + 'self_attn.k_proj.weight']['shape']
        gate, down = d[n + 'mlp.gate_proj.weight']['shape'], d[n + 'mlp.down_proj.weight']['shape']
        o = d[n + 'self_attn.o_proj.weight']['shape']
        conv = d[n + 'attention_conv.kernel_projection.weight']['shape']
        out += [(n + 'self_attn.qkv', DRAFT + n + 'self_attn.qkv_proj', ((q[0] + 2 * k[0]) // 4, q[1]), True, ('w4',)),
                (n + 'mlp.gate_up', DRAFT + n + 'mlp.gate_up_proj', (2 * gate[0] // 4, gate[1]), True, ('w4',)),
                (n + 'self_attn.o_proj', DRAFT + n + 'self_attn.o_proj', (o[0], o[1] // 4), False, ('w4',)),
                (n + 'mlp.down_proj', DRAFT + n + 'mlp.down_proj', (down[0], down[1] // 4), False, ('w4',)),
                (n + 'attention_conv', DRAFT + n + 'attention_conv.kernel_projection', tuple(conv), True, ('w4',)),
                (n + 'mlp_conv', DRAFT + n + 'mlp_conv.kernel_projection', tuple(conv), True, ('w4',))]
    return out


def tensors(label, shape, smoothed):
    g = torch.Generator().manual_seed(int(hashlib.sha256(label.encode()).hexdigest()[:8], 16))
    weight = (torch.randn(shape, generator=g) * 0.02).bfloat16()
    smooth = torch.exp2(torch.round(torch.log2(torch.rand(shape[1], generator=g) + 0.5))) if smoothed else None
    return weight, smooth


def store_module(variant):
    path = Path('/old/store_bb72154d.py') if variant == 'old' else Path('/repo/engine/kernels/dense/store.py')
    spec = importlib.util.spec_from_file_location(f'store_{variant}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forget(root):
    paths = list(CALIBRATION.rglob('*.pt')) + list((root / 'st-dense-packs').glob('*.pt'))
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def lanes(store, variant, weight, name, smooth, which):
    digest = store.weight_digest(weight) if variant == 'new' and which != ('fp8',) else None
    kw = {} if digest is None else {'digest': digest}
    for lane in which:
        if lane == 'w4':
            store.pack(weight, name, smooth=smooth, **kw)
        elif lane == 'fp8':
            store.pack_fp8(weight, name, smooth=smooth, **kw)
        else:
            store.pack_fp8(weight, name + '.committed-decode-v1', smooth=smooth, **kw)


def setup(root):
    """File stub packs under the identities the store computes: zeros of the served layouts."""
    from engine.kernels.dense import W4Pack
    root.mkdir(parents=True, exist_ok=True)
    link = root / 'mkcalib'
    if not link.exists():
        link.symlink_to(CALIBRATION)
    module = store_module('new')

    def w4(weight, hessian=None, per_row=True, factor=None, **_):
        n, k = weight.shape
        p = (n + 127) // 128 * 128
        return W4Pack(torch.zeros(p // 128, k // 128, 128, 64, dtype=torch.uint8),
                      torch.zeros(p // 128, k // 128, 128, 8, dtype=torch.int8), torch.ones(p), n, k, hessian is not None)

    def fp8(weight, hessian, factor=None, **_):
        n, k = weight.shape
        p = (n + 127) // 128 * 128
        return torch.zeros(p, k).to(torch.float8_e4m3fn), torch.ones(p // 128, k // 128)
    store = module.PackStore(root, RANK)                            # its algorithm identity reads the real pack_w4
    with mock.patch('engine.kernels.dense.pack_w4', side_effect=w4), \
            mock.patch('engine.kernels.dense.packing.fp8_gptq', side_effect=fp8), \
            mock.patch.object(module.PackStore, '_factor', return_value=None):
        for label, name, shape, smoothed, which in items():
            weight, smooth = tensors(label, shape, smoothed)
            lanes(store, 'new', weight, name, smooth, which)
        store.release_pages()
    print(json.dumps(dict(setup=dict(store.stats), items=len(items()))))


def run(root, variant):
    module = store_module(variant)
    store = module.PackStore(root, RANK)
    forget(root)
    reads = []
    real = module.PackStore._hessian

    def counted(self, name, k, smooth=None):
        reads.append(name)
        return real(self, name, k, smooth)
    seconds = 0.0
    with mock.patch.object(module.PackStore, '_hessian', counted):
        for label, name, shape, smoothed, which in items():
            weight, smooth = tensors(label, shape, smoothed)          # made outside the clock, one at a time
            start = time.perf_counter()
            lanes(store, variant, weight, name, smooth, which)
            seconds += time.perf_counter() - start
            del weight, smooth
    store.release_pages()
    bad = {k: v for k, v in store.stats.items() if 'built' in k}
    print(json.dumps(dict(variant=variant, seconds=round(seconds, 3), hessian_reads=len(reads),
                          stats=dict(store.stats), built=bad, weights=len(items()),
                          threads=torch.get_num_threads())))
    if bad:
        raise SystemExit('a variant built packs: it did not find the filed identities')


if __name__ == '__main__':
    command, root = sys.argv[1], Path(sys.argv[2])
    if command == 'setup':
        setup(root)
    else:
        run(root, sys.argv[3])
