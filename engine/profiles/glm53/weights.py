"""GLM's presharded weight contract, checked before allocating its arena."""
from engine.base.loader import RankLoader

WEIGHT_LAYOUT = "st-glm53-b12x-up-gate-v1"
MODELOPT_WEIGHT_LAYOUT = "st-glm53-modelopt-up-gate-v1"
# ModelOpt routed experts with the three dense MLPs in BF16: the engine packs those from calibration exactly as it
# does Red Hat's, instead of serving them as one-expert NVFP4. Selected by a config excluding the dense MLPs.
MODELOPT_BF16_DENSE_LAYOUT = "st-glm53-modelopt-up-gate-bf16-dense-v1"
MODELOPT_LAYOUTS = (MODELOPT_WEIGHT_LAYOUT, MODELOPT_BF16_DENSE_LAYOUT)


def rank_loader(path, *, expected_layout=None):
    loader = RankLoader(path)
    marker = (loader.metadata or {}).get("weight_layout")
    packed = [name for name in loader.keys() if name.endswith((".moe.w13", ".mlp.w13"))]
    if packed and marker not in (WEIGHT_LAYOUT, *MODELOPT_LAYOUTS):
        raise ValueError(
            f"{path}: missing or incompatible b12x up|gate layout; "
            "regenerate rank files with engine/profiles/glm53/preshard.py"
        )
    if expected_layout is not None and marker != expected_layout:
        raise ValueError(f"{path}: checkpoint/rank weight layout mismatch: {expected_layout} != {marker}")
    if marker in MODELOPT_LAYOUTS:
        for name in packed:
            prefix = name.removesuffix('w13')
            experts = loader.header[name]['shape'][0]
            for suffix in ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale'):
                entry = loader.header.get(prefix + suffix)
                if entry is None or entry['dtype'] != 'F32' or entry['shape'] != [experts]:
                    raise ValueError(f"{path}: missing or malformed ModelOpt scale {prefix + suffix}")
    return loader
