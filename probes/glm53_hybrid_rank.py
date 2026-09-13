"""Build one rank of GLM-5.3's ModelOpt BF16-dense hybrid (st-glm53-modelopt-up-gate-bf16-dense-v1) from ranks already
on the node, streamed through engine/base/preshard.RankWriter (aligned offsets, exactly the layout's specs):

  routed experts   NVIDIA's ModelOpt rank file; w13/w13_sf/w2/w2_sf replaced by a probes/expert_rank_patch.py patch
                   when --patch names one; the activation global scales a13_scale/a2_scale set to 1.0 with
                   --unit-activation (dynamic group scales on the raw input, as Red Hat's folding serves them)
  dense MLPs 0-2   BF16 gate_up/down from the Red Hat (b12x) rank file of the same rank
  everything else  NVIDIA's rank file, byte for byte

--meta is a checkpoint metadata directory whose config.json is NVIDIA's with the dense MLPs added to
quantization_config.exclude_modules (that is what selects the layout). The output directory gets the rank file, a link to
NVIDIA's vision.safetensors and a receipt; every tensor is read back and checked against its source.

    python3 probes/glm53_hybrid_rank.py --meta DIR --nvidia-ranks DIR --redhat-ranks DIR [--patch DIR]
        [--unit-activation] --rank R --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from engine.base.preshard import RankWriter  # noqa: E402
from engine.profiles.glm53 import modelopt_weights  # noqa: E402
from engine.profiles.glm53.weights import MODELOPT_BF16_DENSE_LAYOUT, rank_loader  # noqa: E402

PATCHED = ("w13", "w13_sf", "w2", "w2_sf")


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def patch_index(directory: "Path | None") -> dict:
    """tensor name -> (patch file, recorded sha256) for every tensor a patch directory carries."""
    index = {}
    if directory is None:
        return index
    for path in sorted(directory.glob("L*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            for name in f.keys():
                if name in index:
                    raise ValueError(f"{name} is patched twice ({index[name][0]} and {path})")
                index[name] = (path, meta.get(name))
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--nvidia-ranks", type=Path, required=True)
    ap.add_argument("--redhat-ranks", type=Path, required=True)
    ap.add_argument("--patch", type=Path, default=None, help="this rank's patch directory (expert_rank_patch.py build)")
    ap.add_argument("--unit-activation", action="store_true")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    F = modelopt_weights.load_facts(a.meta)
    if F.weight_layout != MODELOPT_BF16_DENSE_LAYOUT:
        raise SystemExit(f"{a.meta}/config.json does not exclude the dense MLPs: layout {F.weight_layout}")
    specs = modelopt_weights.all_specs(F)
    name = f"rank{a.rank}of{a.world}.safetensors"
    nv_path, rh_path = a.nvidia_ranks / name, a.redhat_ranks / name
    target = a.out / name
    if target.exists():
        raise SystemExit(f"{target} exists: an output rank file is immutable")
    patches = patch_index(a.patch)
    a.out.mkdir(parents=True, exist_ok=True)
    partial = a.out / (name + ".partial")
    sources = {"nvidia": 0, "redhat": 0, "patch": 0, "unit": 0}
    expected = {}
    # Only the dense MLPs come from Red Hat's rank: read those and close it before NVIDIA's is mapped -- under strict
    # overcommit (srv2) two whole 44 GiB copy-on-write maps do not fit the commit limit.
    dense_names = [s.name for s in specs if s.name.partition(".")[0].startswith("L")
                   and not F.is_moe(int(s.name.partition(".")[0][1:])) and s.name.endswith((".mlp.gate_up", ".mlp.down"))]
    with safe_open(str(rh_path), framework="pt") as rh:
        if (rh.metadata() or {}).get("weight_layout") != "st-glm53-b12x-up-gate-v1":
            raise SystemExit(f"{rh_path}: not a Red Hat b12x rank file")
        missing_dense = [n for n in dense_names if n not in set(rh.keys())]
        if missing_dense:
            raise SystemExit(f"{rh_path} lacks {missing_dense}")
        redhat_dense = {n: rh.get_tensor(n).clone() for n in dense_names}
    with safe_open(str(nv_path), framework="pt") as nv:
        nv_meta = dict(nv.metadata() or {})
        if nv_meta.get("weight_layout") != modelopt_weights.WEIGHT_LAYOUT:
            raise SystemExit(f"{nv_path}: not a ModelOpt rank file ({nv_meta.get('weight_layout')})")
        nv_keys = set(nv.keys())
        metadata = dict(nv_meta, weight_layout=MODELOPT_BF16_DENSE_LAYOUT,
                        hybrid=json.dumps(dict(nvidia=str(nv_path), redhat=str(rh_path),
                                               patch=str(a.patch) if a.patch else None,
                                               unit_activation=a.unit_activation)))
        writer = RankWriter(partial, specs, metadata)
        try:
            for spec in specs:
                layer, _, suffix = spec.name.partition(".")
                dense_layer = layer.startswith("L") and not F.is_moe(int(layer[1:]))
                if spec.name in patches:
                    path, recorded = patches[spec.name]
                    with safe_open(str(path), framework="pt") as f:
                        t = f.get_tensor(spec.name)
                    if recorded is not None and digest(t) != recorded:
                        raise SystemExit(f"{path}: {spec.name} does not match its recorded sha256")
                    sources["patch"] += 1
                elif a.unit_activation and suffix in ("moe.a13_scale", "moe.a2_scale"):
                    t = torch.ones(spec.shape, dtype=spec.dtype)
                    sources["unit"] += 1
                elif dense_layer and suffix in ("mlp.gate_up", "mlp.down"):
                    t = redhat_dense.pop(spec.name)
                    sources["redhat"] += 1
                else:
                    if spec.name not in nv_keys:
                        raise SystemExit(f"{nv_path} lacks {spec.name}")
                    t = nv.get_tensor(spec.name)
                    sources["nvidia"] += 1
                expected[spec.name] = digest(t)
                writer.put(spec.name, t)
                del t
        finally:
            writer.close()
    loader = rank_loader(partial, expected_layout=MODELOPT_BF16_DENSE_LAYOUT)
    missing = sorted(set(expected) - set(loader.keys()))
    if missing:
        raise SystemExit(f"written file lacks {missing[:4]}")
    with safe_open(str(partial), framework="pt") as f:
        for tensor_name, sha in expected.items():
            if digest(f.get_tensor(tensor_name)) != sha:
                raise SystemExit(f"{tensor_name}: read-back sha256 differs")
    os.replace(partial, target)
    vision = a.nvidia_ranks / "vision.safetensors"
    if vision.exists() and not (a.out / vision.name).exists():
        try:
            os.link(vision, a.out / vision.name)
        except OSError:
            import shutil
            shutil.copyfile(vision, a.out / vision.name)
    receipt = dict(rank=a.rank, layout=MODELOPT_BF16_DENSE_LAYOUT, tensors=len(expected), sources=sources,
                   unit_activation=a.unit_activation, patch=str(a.patch) if a.patch else None,
                   seconds=round(time.time() - t0, 1), sha256=expected)
    (a.out / f"hybrid-receipt-rank{a.rank}.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "sha256"}), flush=True)


if __name__ == "__main__":
    main()
