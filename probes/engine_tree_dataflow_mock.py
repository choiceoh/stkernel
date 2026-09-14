"""Record CPU correctness gates, per-test latency, source hashes and storage arithmetic.

No GPU, serving tok/s, acceptance forecast or calibrated speedup is produced.
Run alongside engine_tree_dataflow_compile.py for offline SM121 codegen proof.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import unittest

import torch

from engine.modules.nvfp4_dataflow import NVFP4Plan
from engine.modules.w4a8_dataflow import W4A8Plan, W4A8PipelinePlan


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("tests.test_engine_speculative_tree", "tests.test_engine_tree_decode",
           "tests.test_engine_tile_dataflow", "tests.test_engine_nvfp4_dataflow", "tests.test_engine_w4a8_dataflow",
           "tests.test_engine_drafter", "tests.test_engine_draft_agreement",
           "tests.test_engine_moe_output", "tests.test_engine_execution_plans",
           "tests.test_engine_ffn_packets", "tests.test_engine_tree_dataflow_gpu")
SOURCES = ("engine/modules/speculative_tree.py", "engine/modules/tree_kda.py", "engine/kernels/kda/tree.py",
           "engine/modules/tile_dataflow.py", "engine/modules/nvfp4_dataflow.py", "engine/modules/w4a8_dataflow.py",
           "engine/kernels/tile_dataflow.py", "engine/kernels/w4a8_pipeline.py",
           "engine/profiles/glm53/tree_decode.py", "engine/profiles/glm53/net.py",
           "engine/profiles/glm53/drafter.py", "engine/profiles/glm53/lanes.py")


class TimedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        self.records.append({"test": test.id(), "cpu_seconds": time.perf_counter()-self.started})
        super().stopTest(test)


def main(output):
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("hide CUDA devices before running this CPU mock")
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in MODULES)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=TimedResult).run(suite)
    plan = NVFP4Plan(16, 4096, 3072)
    w4a8 = W4A8Plan(16, 4096, 3072)
    staged = W4A8PipelinePlan(16, 4096, 3072)
    nodes, heads, dim = 16, 16, 128
    report = dict(scope="CPU correctness and storage arithmetic; no serving performance verdict",
        gpu_used=False, cuda_initialized=torch.cuda.is_initialized(), torch=torch.__version__,
        tests=dict(run=result.testsRun, skipped=len(result.skipped), failures=len(result.failures),
                   errors=len(result.errors), passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors)),
        per_test=result.records, source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
        storage_arithmetic=dict(tree_nodes=nodes, kda_heads=heads, kda_dim=dim,
            full_per_node_state_bytes=nodes*heads*dim*dim*4,
            fp32_factor_bytes=nodes*heads*3*dim*4, owned_initial_state_bytes=heads*dim*dim*4,
            nvfp4_mlp_rows=plan.rows, nvfp4_hidden=plan.hidden, nvfp4_intermediate=plan.intermediate,
            nvfp4_mlp_scratch_bytes=plan.scratch_bytes, dataflow_tasks=len(plan.tasks),
            w4a8_mlp_scratch_bytes=w4a8.scratch_bytes, w4a8_dataflow_tasks=len(w4a8.tasks),
            w4a8_pipeline_scratch_bytes=staged.scratch_bytes,
            additional_resident_weight_bytes=0),
        unmeasured=["real-weight GPU numerics", "worker liveness on GB10", "route predictor recall",
                    "C1/C4 serving tok/s", "TTFT", "acceptance", "reasoning quality", "peak process memory"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"mock.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"tests.txt").write_text(log.getvalue())
    print(log.getvalue().split("\n----------------------------------------------------------------------")[-1])
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
