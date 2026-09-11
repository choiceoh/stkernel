"""GLM's presharded weight contract, checked before allocating its arena."""
from engine.base.loader import RankLoader

WEIGHT_LAYOUT = "st-glm53-b12x-up-gate-v1"


def rank_loader(path):
    loader = RankLoader(path)
    has_experts = any(name.endswith(".moe.w13") for name in loader.keys())
    if has_experts and (loader.metadata or {}).get("weight_layout") != WEIGHT_LAYOUT:
        raise ValueError(
            f"{path}: missing or incompatible b12x up|gate layout; "
            "regenerate rank files with engine/profiles/glm53/preshard.py"
        )
    return loader
