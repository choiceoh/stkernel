"""Bounded CUDA prefill tile consumer gate; admitted through bench/fleet.sh."""
import hashlib
import json
from pathlib import Path
import time
import unittest


def main():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("requires an admitted GB10")
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "tests.test_engine_prefill_tiles_cuda", "tests.test_engine_prefill_fp8_consumer"])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("prefill tile gate failed or skipped")
    root = Path(__file__).resolve().parents[1]
    files = ("engine/kernels/prefill_collectives/tiles.py", "engine/kernels/prefill_collectives/__init__.py",
             "engine/kernels/prefill_collectives/consumer.py", "engine/kernels/dense/__init__.py",
             "engine/kernels/dense/fp8.py", "engine/profiles/glm53/net.py",
             "engine/profiles/glm53/execution.py", "tests/test_engine_prefill_tiles_cuda.py",
             "tests/test_engine_prefill_fp8_consumer.py")
    report = dict(scope="CUDA slot reuse and FP8 projection with synthetic peers; not NIC or serving speed proof",
                  tests=result.testsRun, skipped=0, seconds=time.monotonic()-start,
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
    Path("/cache/prefill-tiles.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
