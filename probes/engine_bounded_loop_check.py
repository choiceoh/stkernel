"""Compile-only or admitted GB10 conditional graph and decode-commit gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    if args.compile_only:
        from engine.kernels.bounded_graph import build
        from engine.kernels.decode_queue import build as queue_build
        result = dict(module=build().__name__, queue=queue_build().__name__)
        assert not torch.cuda.is_initialized()
        scope = "SM121a conditional graph compilation only; no CUDA context"
    else:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("requires an admitted GB10")
        run = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromNames(("tests.test_engine_bounded_loop_cuda",
                                                          "tests.test_engine_burst_decode_cuda",
                                                          "tests.test_engine_decode_queue_cuda")))
        if not run.wasSuccessful() or run.skipped:
            raise RuntimeError("bounded loop gate failed or skipped")
        result = dict(tests=run.testsRun, skipped=0)
        scope = "single GB10 conditional graph and serving adapter with toy target; not real TP4 or model-quality proof"
    root = Path(__file__).resolve().parents[1]
    files = ("engine/kernels/bounded_graph/loop.cu", "engine/kernels/bounded_graph/__init__.py",
             "engine/profiles/glm53/bounded_loop.py", "engine/kernels/decode_commit.py",
             "engine/profiles/glm53/burst_decode.py", "engine/profiles/glm53/pipeline.py",
             "engine/profiles/glm53/adapter.py", "engine/base/runner.py",
             "engine/base/graphs.py", "engine/profiles/glm53/decode_graphs.py",
             "engine/profiles/glm53/drafter.py",
             "engine/kernels/decode_queue/queue.cu", "engine/kernels/decode_queue/__init__.py",
             "engine/base/serve.py", "tests/test_engine_decode_queue_cuda.py",
             "tests/test_engine_bounded_loop_cuda.py", "tests/test_engine_burst_decode_cuda.py")
    report = dict(scope=scope, result=result, seconds=time.monotonic()-start,
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
    Path("/cache/bounded-loop.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
