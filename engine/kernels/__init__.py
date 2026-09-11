"""ST GPU kernels. Importing this package does not initialize a device."""

# Probe hook for the TileLang lowering passes of every mHC kernel (TMA lowering, warp
# specialisation; the image's stock dict disables both). Unmeasured on GLM (mHC is
# 11.4% of a prefill step, 9/1 trace). The mhc package captures it when it is imported,
# so a bracket calls configure_mhc_passes() BEFORE `from engine.kernels import mhc`;
# never an environment read, never a serving switch.
MHC_PASSES: "tuple[bool, bool] | None" = None


def configure_mhc_passes(tma: bool, ws: bool) -> None:
    import sys
    global MHC_PASSES
    if __name__ + ".mhc.tilelang_kernels" in sys.modules:
        raise RuntimeError("configure_mhc_passes: the mHC kernels are already compiled with the previous pass set")
    MHC_PASSES = (bool(tma), bool(ws))
