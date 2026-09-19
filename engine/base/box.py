"""The box every profile is written for, and its assertion (base).

The charter's D5 fixes the hardware and leaves the model open (2026-09-19): one DGX Spark a rank -- a GB10, SM121,
48 SMs, one device, and one memory pool the host and the device share. Each profile asserted that box with its own
copy of the same dict and the same function, one written from the other; the fact belongs to the engine, so it is
written once here and the profiles' `facts.BOX` / `facts.check_box` name it.

`check_box` is D3 for the box: a node that is not this one stops the boot before anything is allocated, with the
number that differs. It initialises CUDA, so a boot calls it on its main thread before the ranks meet.
"""
from __future__ import annotations

BOX = {"name": "GB10 (DGX Spark)", "capability": (12, 1), "sms": 48, "devices": 1, "unified": True}


def check_box() -> str:
    """The node the engine is written for, asserted (D3): one GB10, unified memory."""
    import torch
    if torch.cuda.device_count() != BOX["devices"]:
        raise SystemExit(f"box: {torch.cuda.device_count()} devices, this profile is written for {BOX['devices']} ({BOX['name']})")
    cap = torch.cuda.get_device_capability(0)
    if cap != BOX["capability"]:
        raise SystemExit(f"box: capability {cap}, this profile's kernels are SM{BOX['capability'][0]}{BOX['capability'][1]} ({BOX['name']})")
    free, total = torch.cuda.mem_get_info()
    mem_total = 0
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            mem_total = int(line.split()[1]) * 1024
    if abs(total - mem_total) > mem_total // 64:
        raise SystemExit(f"box: device total {total / 2**30:.1f} GiB != host {mem_total / 2**30:.1f} GiB: not unified memory")
    return f"{BOX['name']}: SM{cap[0]}{cap[1]}, unified {total / 2**30:.0f} GiB ({free / 2**30:.0f} free)"


__all__ = ["BOX", "check_box"]
