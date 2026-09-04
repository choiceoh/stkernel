#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile the one-shot AR extension the way the shim does and confirm the
prefetch-hint bindings exist. A ptxas rejection of a new instruction would
otherwise surface as "[osar] local setup failed" on every rank of a real
boot -- and that boot falls back to NCCL, silently losing the AR module.

    bash probes/run_mk_probe.sh probes/osar_build_check.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.environ.get("MK_PKG_PATH",
                                  "/usr/local/lib/python3.12/dist-packages"))

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

SRC = "/repo/overlay/modules/tp_oneshot_ar/dsv4_oneshot_ar.cu"


def main() -> int:
    build = tempfile.mkdtemp(prefix="osar_build_")
    ext = load(name="dsv4_oneshot_ar_check", sources=[SRC],
               extra_cuda_cflags=["-O2", "-arch=sm_121a"],
               extra_ldflags=["-libverbs"], build_directory=build,
               verbose=False)
    names = [n for n in dir(ext) if not n.startswith("_")]
    print("bound:", " ".join(sorted(names)))
    ok = all(n in names for n in ("oneshot_ar", "oneshot_ar_hint",
                                  "phase_counters", "healthy"))
    print("osar build:", "PASS" if ok else "FAIL (missing binding)")
    # no RDMA here: init() needs the fabric. The compile is the gate.
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
