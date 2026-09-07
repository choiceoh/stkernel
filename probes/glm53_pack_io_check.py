#!/usr/bin/env python3
"""Fleet-holder GPU check of exact W4 transport, using existing pack files.

Run inside glm53 after deployment, before the timed boot bracket. Does not
write model/cache artifacts or change the serving process's environment.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from vllm.model_executor.layers import glm53_megakernel as mk
from vllm.model_executor.layers import glm53_startup_cache as common


def digest(tensor):
    return hashlib.md5(tensor.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def main():
    torch.set_num_threads(1)
    stream = torch.cuda.Stream()
    report = {"hash": [], "restore": [], "checks": 0}
    generator = torch.Generator().manual_seed(733)
    for shape in ((129, 256), (6416, 4096), (4096, 8192)):
        cpu = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
        expected = digest(cpu)
        weight = cpu.cuda()
        timings = {0: [], 1: []}
        for fast in (0, 1, 1, 0, 0, 1):
            os.environ["VLLM_GLM53_MK_PACK_FAST_IO"] = str(fast)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.cuda.stream(stream):
                actual = mk._weight_md5(weight)
            timings[fast].append(time.perf_counter() - started)
            assert actual == expected, (shape, fast, "hash mismatch")
            report["checks"] += 1
        report["hash"].append({"shape": shape, "median_s": {k: statistics.median(v) for k, v in timings.items()}, "samples_s": timings})
        del weight, cpu

    root = Path(os.environ.get("VLLM_GLM53_MK_PACK_CACHE", "/cache/mkpacks")) / "rank0"
    files = sorted(root.glob("*.pt"), key=lambda p: p.stat().st_size, reverse=True)[:3]
    assert files, f"no existing W4 packs in {root}"
    for path in files:
        reference = torch.load(path, map_location="cpu", weights_only=True)
        keys = [key for key in ("wq4", "ws4", "rgs", "lr_a", "lr_b") if reference[key] is not None]
        timings = {0: [], 1: []}
        for fast in (0, 1, 1, 0, 0, 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.cuda.stream(stream):
                blob = mk._load_pack_blob(path, fast)
                copies = [mk._pack_tensor_to_device(blob[key], "cuda", fast) for key in keys]
            stream.synchronize()
            timings[fast].append(time.perf_counter() - started)
            del blob
            if fast and mk._pack_staging().buffer is not None:
                # Host memory may be reused immediately after the helper returns.
                mk._pack_staging().buffer.zero_()
            for key, value in zip(keys, copies):
                assert value.shape == reference[key].shape and value.stride() == reference[key].stride()
                assert torch.equal(value.cpu(), reference[key]), (path.name, fast, key)
                report["checks"] += 1
            del copies
        report["restore"].append({"file": path.name, "bytes": path.stat().st_size,
                                  "median_s": {k: statistics.median(v) for k, v in timings.items()}, "samples_s": timings})

    # Cross the 64 MiB staging boundary, retain data after mapped storage dies,
    # and exercise the synchronous fallback on the same non-default stream.
    cpu = torch.randint(0, 256, (common.TRANSFER_BYTES + 513,), dtype=torch.uint8, generator=generator)
    with torch.cuda.stream(stream):
        target = mk._pack_tensor_to_device(cpu, "cuda", True)
    assert torch.equal(target.cpu(), cpu)
    report["checks"] += 1
    fallback = common.HostStaging()
    fallback.disabled = True
    mk._PACK_STAGING = fallback
    with torch.cuda.stream(stream):
        target = mk._pack_tensor_to_device(cpu, "cuda", True)
    assert torch.equal(target.cpu(), cpu)
    report["checks"] += 1
    report["ok"] = True
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
