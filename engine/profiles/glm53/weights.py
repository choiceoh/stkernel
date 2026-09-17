"""GLM's presharded weight contract, checked before allocating its arena."""
import re

from engine.base.loader import RankLoader

WEIGHT_LAYOUT = "st-glm53-b12x-up-gate-v1"
MODELOPT_WEIGHT_LAYOUT = "st-glm53-modelopt-up-gate-v1"
# ModelOpt routed experts with the three dense MLPs in BF16: the engine packs those from calibration exactly as it
# does Red Hat's, instead of serving them as one-expert NVFP4. Selected by a config excluding the dense MLPs.
MODELOPT_BF16_DENSE_LAYOUT = "st-glm53-modelopt-up-gate-bf16-dense-v1"
MODELOPT_LAYOUTS = (MODELOPT_WEIGHT_LAYOUT, MODELOPT_BF16_DENSE_LAYOUT)


def restored_constants_id(metadata):
    """Identity for state caches made with restored original control values."""
    metadata = metadata or {}
    digest, kinds = (metadata.get(name) for name in ('fp32_constants_sha256', 'fp32_constants_kinds'))
    if digest is None and kinds is None:
        return None
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None or kinds not in ('router', 'kda', 'all'):
        raise ValueError('incomplete or invalid original FP32 constant identity')
    return f'{kinds}-{digest}'


def rank_loader(path, *, expected_layout=None):
    loader = RankLoader(path)
    restored_constants_id(loader.metadata)
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
