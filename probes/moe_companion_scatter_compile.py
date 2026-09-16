"""Compile the m=16 companion lane -- the one a mixed-provenance checkpoint needs.

A checkpoint whose experts do not all carry 6-bit-packable scales builds, beside every sf6 lane, a
companion with reform_sf_pack off. At m=16 `batch` turns direct register scatter on for it too, and
moe_static_kernel_v4.__init__ refused the pair, so the hybrid arm could not boot
(measurements/st_hybrid_boot_block_20260916). #1056 bought the boot by turning direct scatter off
for that companion -- it runs the staged path instead. This asks the other question: can the
companion keep direct scatter?

    MK_PROBE_NO_GPU=1 bash probes/run_mk_probe.sh <this file>
"""
import os
import sys
import time

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
sys.path.insert(0, os.environ.get("MK_PKG_PATH", "/usr/local/lib/python3.12/dist-packages"))
sys.path.insert(0, "/repo")

import torch  # noqa: E402

# the device queries the dispatcher makes before compiling, answered for a GB10 so no CUDA context
# is created: this runs on a node that may be serving production.
torch.cuda.is_available = lambda: True  # type: ignore[assignment]
torch.cuda.get_device_capability = lambda *a, **k: (12, 1)  # type: ignore[assignment]


class _Props:                      # get_num_sm reads this before mac_override is consulted
    multi_processor_count = 48


torch.cuda.get_device_properties = lambda *a, **k: _Props()  # type: ignore[assignment]

E, HID, INTER, TOPK = 288, 4096, 512, 8


def main() -> int:
    from engine.kernels.b12x import moe_dispatch as md
    base = md._parse_glm53_static_v2("t,r,sf6,batch", probe=True)
    ok = True
    for label, over in (("served sf6 lane", {}), ("companion (reform_sf_pack off)", {"reform_sf_pack": False})):
        cfg = md._static_v2_decode_config(dict(base, **over), 16)
        print(f"[{label}] m=16 reform={cfg.get('decode_reform')} sf6={cfg.get('reform_sf_pack')} "
              f"direct={cfg.get('c2_direct_scatter')} reuse={cfg.get('c2_scatter_reuse')} "
              f"prefetch={cfg.get('c2_fc2_prefetch')} vec4={cfg.get('scatter_vec4')}", flush=True)
        md._STATIC_V2_KERNEL_CACHE.clear()
        t0 = time.time()
        try:
            _, mac = md._get_static_kernel_v2(
                E, E, 16, HID, INTER, TOPK, 512, config=cfg, mac_override=48,
                activation="swigluoai_uninterleave", swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.0)
        except Exception as exc:  # noqa: BLE001 -- report both lanes
            import traceback
            traceback.print_exc()
            print(f"[{label}] COMPILE FAIL after {time.time() - t0:.1f} s: {type(exc).__name__}: {str(exc)[:600]}")
            ok = False
            continue
        print(f"[{label}] compiled in {time.time() - t0:.1f} s mac={mac}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
