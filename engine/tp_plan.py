"""Where every DSv4.1 tensor goes at TP=4, read out of the reference converter.

``tools/dsv41_preshard.py`` calls the tp/replicate split for the ~17 GiB of
dense and attention weights "a HYPOTHESIS until vLLM has DeepSeek-V4.1 model
code to compare against", and defaults to replicating because slicing on the
wrong axis produces a checkpoint that loads and computes garbage. That caution
is right and the conclusion is now stale twice over: we are writing the model
ourselves (CHARTER D13), and the axis was never a hypothesis -- the shipped
reference converter states it.

    inference/convert.py::mapping = {name -> (new_name, dim)}

Seven keys carry a dim; everything else is replicated. Three placements are
special-cased in the same file: routed experts go whole to one rank, engram
tables split into contiguous row blocks, and an MTP layer's embed/head are
tied to the backbone's and dropped.

The mapping is EXTRACTED, not copied. Copying it here would drift the first
time DeepSeek changes it and nothing would say so. This module pins the file
by SHA-256 and reads the dict out of its AST, so a vendor change FAILS rather
than drifts -- the same contract dsv41_sparse_contract.py holds the TileLang
kernel to.
"""
from __future__ import annotations

import ast
import hashlib
import json
import struct
from pathlib import Path

GIB = 1 << 30

# inference/convert.py as shipped with DeepSeek-V4.1-Flash.
REFERENCE_SHA256 = "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"

# dtype -> bytes per element, for the dtypes this checkpoint actually uses.
_ITEMSIZE = {"I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4, "F16": 2}


def reference_mapping(reference: "str | Path") -> "dict[str, int]":
    """{component: dim} lifted from convert.py's ``mapping`` literal."""
    path = Path(reference)
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if REFERENCE_SHA256 != "REPLACED_AT_WRITE_TIME" and digest != REFERENCE_SHA256:
        raise ValueError(
            f"{path.name} is not the pinned reference (sha256 {digest}). "
            "Re-read the mapping before trusting any placement derived from it."
        )
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "mapping" for t in node.targets):
            continue
        literal = ast.literal_eval(node.value)
        out = {}
        for key, value in literal.items():
            new_name, dim = value
            if new_name != key:
                raise ValueError(f"convert.py renames {key!r} to {new_name!r}; "
                                 "placement by component name no longer holds")
            out[key] = dim
        return out
    raise ValueError("convert.py has no top-level `mapping` assignment")


# --- the three special cases convert.py handles outside `mapping` ------------

def component(name: str) -> str:
    """convert.py's own key rule, applied to an already-stripped name."""
    if any(x in name for x in ("hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_")):
        return name.split(".")[-1]
    return name.split(".")[-2]


def placement(name: str, mapping: "dict[str, int]") -> "tuple[str, int | None]":
    """(kind, dim) for one tensor. kind is expert / engram / tp / replicate / tied."""
    if name.startswith("model."):
        name = name[len("model."):]
    if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
        return "tied", None            # an MTP layer ties these to the backbone
    canonical = name.replace("self_attn", "attn")
    if not canonical.startswith("vision."):
        canonical = canonical.replace("mlp", "ffn")
    canonical = canonical.replace("weight_scale_inv", "scale")
    if "experts" in canonical and "shared_experts" not in canonical:
        return "expert", None          # whole experts, one to a rank
    if ".engram.embed." in canonical:
        return "engram", 0             # contiguous row blocks
    dim = mapping.get(component(canonical))
    return ("tp", dim) if dim is not None else ("replicate", None)


# --- applying it to the real checkpoint --------------------------------------

def _headers(repo: Path):
    index = json.loads((repo / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        path = repo / shard
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        header.pop("__metadata__", None)
        yield header


def rank_plan(repo: "str | Path", world_size: int = 4,
              reference: "str | Path | None" = None) -> dict:
    """Bytes each rank holds, per placement kind, from the shipped mapping."""
    repo = Path(repo)
    mapping = reference_mapping(reference or repo / "inference" / "convert.py")
    kinds, counts = {}, {}
    # Two lines the engine gets to choose, so they are tracked separately.
    #  vision   the reference replicates the ViT+aligner; this fleet is text only.
    #  wo_a     convert.py DEQUANTIZES wo_a to bf16 after sharding it -- see the
    #           `wo_a.weight` block near the end of main(). Keeping it fp8 halves
    #           it but needs a kernel that reads [32,32] block scales.
    vision_gib = 0.0
    wo_a_fp8_gib = 0.0
    for header in _headers(repo):
        for name, entry in header.items():
            shape = entry["shape"]
            item = _ITEMSIZE[entry["dtype"]]
            total = item
            for extent in shape:
                total *= extent
            kind, dim = placement(name, mapping)
            if kind == "tied":
                continue
            if kind == "expert":
                mine = total / world_size
            elif kind == "engram":
                rows = -(-shape[0] // world_size)      # ceil, padded
                mine = total / shape[0] * rows if shape[0] else 0
            elif kind == "tp":
                if shape[dim] % world_size:
                    raise ValueError(f"{name}: dim {dim} = {shape[dim]} not divisible "
                                     f"by {world_size}; convert.py asserts this")
                mine = total / world_size
            else:
                mine = total
            label = f"tp:{dim}" if kind == "tp" else kind
            kinds[label] = kinds.get(label, 0.0) + mine
            counts[label] = counts.get(label, 0) + 1
            stripped = name.removeprefix("model.")
            if stripped.startswith(("vision.", "aligner.")):
                vision_gib += mine
            if ".wo_a." in stripped:
                wo_a_fp8_gib += mine
    return {"world_size": world_size, "mapping": mapping,
            "gib": {k: v / GIB for k, v in kinds.items()}, "counts": counts,
            "vision_gib": vision_gib / GIB, "wo_a_fp8_gib": wo_a_fp8_gib / GIB}


def resident_gib(plan: dict, engram_on_ssd: bool = True, vision: bool = True,
                 wo_a: str = "fp8") -> float:
    """What a rank must hold in the box, under the engine's own choices."""
    total = sum(plan["gib"].values())
    if engram_on_ssd:
        total -= plan["gib"].get("engram", 0.0)
    if not vision:
        total -= plan["vision_gib"]
    if wo_a == "bf16":
        total += plan["wo_a_fp8_gib"]      # fp8 weight+scale -> bf16 weight
    elif wo_a != "fp8":
        raise ValueError("wo_a must be 'fp8' or 'bf16'")
    return total


def report(plan: dict, engram_on_ssd: bool = True, vision: bool = True) -> str:
    gib = dict(plan["gib"])
    out = [f"  DSv4.1-Flash, TP={plan['world_size']}, axes from inference/convert.py",
           f"  mapping: " + ", ".join(f"{k}->dim{v}" for k, v in sorted(plan["mapping"].items())),
           ""]
    width = max(len(k) for k in gib)
    for label in sorted(gib, key=lambda k: -gib[k]):
        out.append(f"  {label:<{width}}  {gib[label]:>8.2f} GiB  "
                   f"{plan['counts'][label]:>7,} tensors")
    total = sum(gib.values())
    out.append(f"  {'-' * width}  {'-' * 8}")
    out.append(f"  {'per rank':<{width}}  {total:>8.2f} GiB")
    out.append(f"  {'resident':<{width}}  "
               f"{resident_gib(plan, engram_on_ssd, True, 'fp8'):>8.2f} GiB"
               + ("   (engram on SSD, wo_a fp8, vision in)" if engram_on_ssd else ""))
    out.append("")
    out.append("  the engine's two choices, priced:")
    out.append(f"    drop vision + aligner   -{plan['vision_gib']:>6.2f} GiB   this fleet serves text only")
    out.append(f"    wo_a bf16 (reference)   +{plan['wo_a_fp8_gib']:>6.2f} GiB   "
               "convert.py dequantizes it; fp8 needs a [32,32] block-scale kernel")
    out.append(f"    text-only, wo_a fp8      "
               f"{resident_gib(plan, engram_on_ssd, False, 'fp8'):>6.2f} GiB resident")
    out.append(f"    text-only, wo_a bf16     "
               f"{resident_gib(plan, engram_on_ssd, False, 'bf16'):>6.2f} GiB resident")
    return "\n".join(out)


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="/home/choiceoh/models/DeepSeek-V4.1-Flash")
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args(argv)
    plan = rank_plan(args.repo, args.world_size)
    print(report(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
