"""Stand up one DSv4.1 block on this rank's real weights, with our kernels.

This is the bridge, not the model: the reference's `Block` is the module tree
(CHARTER D13 says we write our own eventually; D14 says the reference is the
oracle meanwhile), while the loader, the kernels and the budget are ours. It
exists so that every layer below it -- loader.py, kernels.py, shapes.py,
budget.py -- is exercised against real bytes rather than against each other.

Two load-time conversions are NOT optional, and both are things convert.py does
that a naive `load_state_dict` would silently skip:

  wo_a is dequantized to bf16.  convert.py multiplies the fp8 weight by its
    block scale and stores bf16, because `Attention.__init__` declares wo_a
    with `dtype=torch.bfloat16`. Loading the fp8 bytes into that parameter
    without the scale gives a tensor that is the right shape, the right dtype,
    and off by a per-block factor -- so it runs, and it is wrong. This is the
    +0.34 GiB/rank the budget prices under `--wo-a bf16`.

  compressor weights above ratio 1 are promoted to fp32.  model.py declares
    `wkv`/`wgate` fp32 when compress_ratio > 1 while the checkpoint stores
    bf16, so the promotion is a load step and not a storage dtype
    (dsv41_shapes.py says the same).

Anything whose stored dtype and parameter dtype disagree for a reason NOT in
that list raises. A silent cast is how a checkpoint loads and computes noise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REF = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash/inference")
RANKS = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash-tp4")

_FP8_BLOCK = 32          # config.quantization_config.weight_block_size
N_BACKBONE = 40          # config.text_config.num_hidden_layers


def reference(world: int = 4, rank: int = 0):
    """Import the reference module tree with our kernels underneath it.

    Loaded by explicit path under a distinct name: the reference file is also
    called `model.py`, and a plain `import model` picks up whichever directory
    is first on the path -- which, from inside this file, is this file.
    """
    import importlib.util

    import torch.distributed as dist

    from engine.profiles.dsv41 import kernels as our_kernels

    our_kernels.install()
    if not dist.is_initialized():
        dist.all_reduce = lambda *a, **k: None

    if "dsv41_reference" in sys.modules:
        ref = sys.modules["dsv41_reference"]
    else:
        sys.path.insert(0, str(REF))          # the reference imports `kernel`
        spec = importlib.util.spec_from_file_location("dsv41_reference", REF / "model.py")
        ref = importlib.util.module_from_spec(spec)
        sys.modules["dsv41_reference"] = ref
        spec.loader.exec_module(ref)
    ref.world_size, ref.rank = world, rank
    return ref


def model_args(ref, batch: int, seq: int):
    cfg = json.loads((REF / "config.json").read_text())
    cfg["max_batch_size"], cfg["max_seq_len"] = batch, seq
    return ref.ModelArgs(**cfg)


def build_block(layer: int, batch: int, seq: int, world: int = 4, rank: int = 0,
                device: str = "cuda"):
    import torch

    ref = reference(world, rank)
    args = model_args(ref, batch, seq)
    torch.cuda.set_device(0)
    with ref.set_dtype(torch.bfloat16), torch.device(device):
        block = ref.Block(layer, args)
    return ref, args, block


def _dequant(weight, scale, block: int = _FP8_BLOCK):
    """fp8 [O, I] with e8m0 scale [O/block, I/block] -> bf16 [O, I]."""
    import torch

    out, inn = weight.shape
    w = weight.float().unflatten(0, (out // block, block)).unflatten(-1, (inn // block, block))
    return (w * scale.float()[:, None, :, None]).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)


def _tp_narrow(name: str, src, param, world: int, rank: int):
    """Take this rank's slice when the shard on disk still holds all of them.

    The presharded files were built with `--dense replicate`, so tensors that
    convert.py assigns an axis -- attn_sink, wq_b, wo_a, wo_b, embed, head,
    weights_proj -- arrive whole. tp_plan.py reads that axis out of the pinned
    converter, so the slice happens here instead of requiring a 293 GiB rebuild
    before anything can run.
    """
    from engine.profiles.dsv41 import placement as tp_plan

    mapping = tp_plan.reference_mapping(REF / "convert.py")
    kind, dim = tp_plan.placement(name, mapping)
    if kind != "tp":
        return None, None
    size = src.shape[dim] // world
    if size != param.shape[dim]:
        return None, None
    return src.narrow(dim, rank * size, size).contiguous(), dim


def load_block(block, layer: int, rank_file: "str | Path" = None, device: str = "cuda",
               recorder=None, world: int = 4, rank: int = 0) -> dict:
    """Fill `block` from this rank's shard. Returns a report, never silence."""
    import torch

    from engine.base.loader import RankLoader

    loader = RankLoader(rank_file or RANKS / "rank0of4.safetensors")
    # convert.py strips `model.` and keeps MTP layers under their own prefix,
    # so a backbone index and an MTP index do not live in the same namespace.
    prefix = f"layers.{layer}." if layer < N_BACKBONE else f"mtp.{layer - N_BACKBONE}."
    keys = [k for k in loader.keys() if k.startswith(prefix)]
    tensors = loader.load(keys, device=device, recorder=recorder)
    have = {k[len(prefix):]: v for k, v in tensors.items()}

    loaded, promoted, dequantized, missing, extra, sharded = [], [], [], [], [], []
    axes = {}
    with torch.no_grad():
        # Only what state_dict() carries has to come from the checkpoint. The
        # caches and freqs_cis are registered `persistent=False` -- runtime
        # state, never stored -- and calling them "missing" would drown the one
        # case D3 exists for: a weight that really is not there.
        expected = set(block.state_dict())
        runtime = [n for n, _ in block.named_buffers() if n not in expected]
        for name, param in list(block.named_parameters()) + list(block.named_buffers()):
            src = have.pop(name, None)
            if src is None:
                if name not in expected:
                    continue                      # runtime state, by design
                missing.append(name)
                continue
            if src.shape != param.shape:
                narrowed, axis = _tp_narrow(name, src, param, world, rank)
                if narrowed is None:
                    raise ValueError(
                        f"{name}: shard has {tuple(src.shape)}, parameter wants "
                        f"{tuple(param.shape)}, and convert.py gives it no axis "
                        "that reconciles them.")
                src = narrowed
                sharded.append(name)
                axes[name] = axis
            if src.dtype == param.dtype:
                param.copy_(src)
                loaded.append(name)
            elif src.dtype == torch.float8_e4m3fn and param.dtype in (
                    torch.bfloat16, torch.float32):
                scale = have.pop(name.replace(".weight", ".scale"), None)
                if scale is None:
                    missing.append(name + " (.scale for dequant)")
                    continue
                # A scale carries the same axis as its weight; narrowing one
                # and not the other is a shape error at best and a per-block
                # factor at worst.
                axis = axes.get(name)
                if axis is not None:
                    step = scale.shape[axis] // world
                    scale = scale.narrow(axis, rank * step, step).contiguous()
                param.copy_(_dequant(src, scale).to(param.dtype))
                dequantized.append(name)
            elif (src.dtype == torch.int8
                  and param.dtype == torch.float4_e2m1fn_x2):
                # convert.py's last loop: `.view(torch.float4_e2m1fn_x2)`.
                # A VIEW, not a cast -- the bytes already hold two e2m1 values
                # each and casting would reinterpret them as small integers.
                param.copy_(src.view(torch.float4_e2m1fn_x2))
                loaded.append(name)
            elif src.dtype == torch.bfloat16 and param.dtype == torch.float32:
                param.copy_(src.float())
                promoted.append(name)
            else:
                raise TypeError(
                    f"{name}: checkpoint has {src.dtype}, parameter wants "
                    f"{param.dtype}, and that pair is not a conversion this "
                    "loader knows. Refusing to cast silently.")
    extra = [k for k in have if not k.endswith(".scale")]
    return {"runtime": runtime, "loaded": len(loaded), "dequantized": dequantized,
            "promoted": promoted, "missing": missing, "unused": extra,
            "sharded": sharded, "tensors": len(keys)}


def load_full(net, rank_file: "str | Path", world: int = 4, rank: int = 0,
              device: str = "cuda", recorder=None) -> dict:
    """Every tensor of a whole rank, with the same conversions load_block does.

    The engram tables are NOT here: the preshard excluded them
    (`"engram": "excluded"` in its metadata) and engram_ssd owns them. Anything
    else absent is a real miss and comes back in `missing`.
    """
    import torch

    from engine.base.loader import RankLoader

    loader = RankLoader(rank_file)
    keys = loader.keys()
    tensors = loader.load(keys, device=device, recorder=recorder)
    have = dict(tensors)

    expected = set(net.state_dict())
    loaded, promoted, dequantized, missing, sharded = [], [], [], [], []
    axes = {}
    with torch.no_grad():
        for name, param in list(net.named_parameters()) + list(net.named_buffers()):
            src = have.pop(name, None)
            if src is None:
                if name in expected and ".engram." not in name:
                    missing.append(name)
                continue
            if src.shape != param.shape:
                narrowed, axis = _tp_narrow(name, src, param, world, rank)
                if narrowed is None:
                    missing.append(f"{name} shape {tuple(src.shape)} vs "
                                   f"{tuple(param.shape)}")
                    continue
                src, axes[name] = narrowed, axis
            if src.dtype == param.dtype:
                param.copy_(src)
                loaded.append(name)
            elif src.dtype == torch.float8_e4m3fn and param.dtype in (
                    torch.bfloat16, torch.float32):
                scale = have.pop(name.replace(".weight", ".scale"), None)
                if scale is None:
                    missing.append(name + " (.scale)")
                    continue
                axis = axes.get(name)
                if axis is not None:
                    step = scale.shape[axis] // world
                    scale = scale.narrow(axis, rank * step, step).contiguous()
                param.copy_(_dequant(src, scale).to(param.dtype))
                dequantized.append(name)
            elif src.dtype == torch.int8 and param.dtype == torch.float4_e2m1fn_x2:
                param.copy_(src.view(torch.float4_e2m1fn_x2))
                loaded.append(name)
            elif src.dtype == torch.bfloat16 and param.dtype == torch.float32:
                param.copy_(src.float())
                promoted.append(name)
            else:
                raise TypeError(f"{name}: {src.dtype} -> {param.dtype} is not a "
                                "conversion this loader knows.")
    return {"loaded": len(loaded), "dequantized": dequantized, "promoted": promoted,
            "missing": missing, "sharded": sharded,
            "unused": [k for k in have if not k.endswith(".scale")]}


def _main(argv=None) -> int:
    import argparse
    import time

    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layer", default="8",
                        help="one index, or a comma list run in order. A chain "
                             "is the only way to exercise CED: an index-source "
                             "layer that is not a kv-source reads the shared "
                             "index_k a lower layer wrote.")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--rank-file", default=None)
    args = parser.parse_args(argv)

    chain = [int(x) for x in str(args.layer).split(",") if x != ""]
    ref, margs, _ = build_block(chain[0], args.batch, args.seq)
    blocks = []
    for layer in chain:
        with ref.set_dtype(torch.bfloat16), torch.device("cuda"):
            block = ref.Block(layer, margs)
        report = load_block(block, layer)
        blocks.append((layer, block, report))
        print(f"  layer {layer}: {report['tensors']:,} tensors, "
              f"loaded {report['loaded']:,}, dequantized {len(report['dequantized'])}, "
              f"promoted {len(report['promoted'])}, tp-sharded {len(report['sharded'])}, "
              f"runtime state {len(report['runtime'])}, MISSING {len(report['missing'])}"
              + (f" {report['missing'][:3]}" if report["missing"] else ""))

    hc = margs.hc_mult
    with torch.device("cuda"):
        x = torch.randn(args.batch, args.tokens, hc, margs.dim, dtype=torch.bfloat16)
        mix = torch.rand(args.batch, args.tokens, hc, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.device("cuda"), ref.set_dtype(torch.bfloat16):
        for layer, block, _ in blocks:
            x, mix = block(x, 0, mix, None)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    finite = torch.isfinite(x).all().item()
    missing = sum(len(r["missing"]) for _, _, r in blocks)
    print(f"  chain {chain}: {tuple(x.shape)} in {elapsed:.2f} s "
          f"({args.tokens * len(chain) / elapsed:,.0f} layer-tok/s, torch kernels)")
    print(f"  finite: {finite}   |x| mean {x.float().abs().mean().item():.4f} "
          f"max {x.float().abs().max().item():.3f}")
    return 0 if finite and not missing else 1


if __name__ == "__main__":
    raise SystemExit(_main())
