"""Admitted FP32 deferred-state gate, then paired 34-layer and candidate timings."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("/cache/kda-deferred.json"))
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--commit-only", action="store_true",
                    help="check batched commit and replay, then compare flat/tiled commit on identical factors")
    args = ap.parse_args()
    if args.samples < 4:
        ap.error("at least four timing samples are required")
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("requires an admitted GB10")
    torch.cuda.set_per_process_memory_fraction((6 << 30) / torch.cuda.get_device_properties(0).total_memory)
    root = Path(__file__).resolve().parents[1]
    files = ("engine/kernels/kda/deferred.py", "engine/kernels/kda/ring.py",
             "engine/profiles/glm53/facts.py",
             "engine/kernels/kda/fused_recurrent.py", "engine/base/graph_labels.py",
             "engine/profiles/glm53/net.py", "engine/profiles/glm53/decode_graphs.py",
             "engine/profiles/glm53/pipeline.py", "engine/profiles/glm53/adapter.py",
             "tests/test_engine_kda_deferred.py", "tests/test_engine_kda_deferred_batch.py",
             "tests/test_engine_burst_decode_cuda.py", "tests/test_engine_graph_labels.py", "probes/engine_kda_deferred_check.py",
             "probes/engine_kda_batch_bench.py", "probes/engine_candidate_packet_bench.py",
             "engine/kernels/common/vocab_candidates.py", "engine/modules/vocab.py")
    report = dict(scope="single GB10 component and graph gate; no NIC or model throughput claim",
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
                  torch=torch.__version__, cuda=torch.version.cuda, status="RUNNING")
    started = time.monotonic()
    try:
        names = ("tests.test_engine_kda_deferred_batch",) if args.commit_only else (
            "tests.test_engine_kda_deferred", "tests.test_engine_kda_deferred_batch")
        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report["correctness"] = dict(tests=result.testsRun, skips=len(result.skipped),
                                      errors=result.errors, failures=result.failures)
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError("deferred-state gate failed or skipped")
        if args.commit_only:
            from probes.engine_kda_batch_bench import measure_commit
            with torch.inference_mode():
                report["commit"] = measure_commit(args.samples)
            report["status"] = "PASS"
            return
        from probes.engine_kda_batch_bench import measure
        from probes.engine_candidate_packet_bench import measure as packets
        with torch.inference_mode():
            report["kda"] = measure(args.samples)
            report["candidate_packets"] = packets()
        # Keep valid component timings even when an independent attribution
        # or toy serving-graph check fails afterwards.
        suite = unittest.defaultTestLoader.loadTestsFromNames(("tests.test_engine_burst_decode_cuda",
                                                               "tests.test_engine_graph_labels"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report["graph_correctness"] = dict(tests=result.testsRun, skips=len(result.skipped),
                                          errors=result.errors, failures=result.failures)
        from engine.base import graph_labels
        report["graph_label_errors"] = list(graph_labels.ERRORS)
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError("serving graph gate failed or skipped")
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["seconds"] = time.monotonic()-started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str)+"\n")
        print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    main()
