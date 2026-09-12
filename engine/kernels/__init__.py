"""ST GPU kernels. Importing this package does not initialize a device."""

# Offline hook for the shared mHC pass configuration, captured at import.
# The post kernel overrides this with its fixed TMA-on/warp-specialization-off
# policy. Call before importing mhc; this is never an environment read or a
# serving switch.
MHC_PASSES: "tuple[bool, bool] | None" = None


def configure_mhc_passes(tma: bool, ws: bool) -> None:
    import sys
    global MHC_PASSES
    if __name__ + ".mhc.tilelang_kernels" in sys.modules:
        raise RuntimeError("configure_mhc_passes: the mHC kernels are already compiled with the previous pass set")
    MHC_PASSES = (bool(tma), bool(ws))
