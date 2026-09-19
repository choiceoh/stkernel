"""GLM-5.3's ModelOpt NVFP4 experts read through the module's contract (engine/modules/modelopt_scales).

This file carried a copy of it -- the same class, line for line -- until 2026-09-19. The name stays: the preshard
manifest records `serving_adapter='engine.profiles.glm53.modelopt_scales'` (preshard_modelopt.py) and probes import
from here, so both keep resolving, to the one implementation.
"""
from engine.modules.modelopt_scales import ModelOptScales

__all__ = ["ModelOptScales"]
