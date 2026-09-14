"""Matched CPU experiment comparison and exact structural counts; never reserves GPUs.

The baseline is the merged #947 experiment, not production serving. Both
variants use this checkout's common target operators and identical tensors.
CPU wall time is useful for catching Python/control regressions, not for
predicting GB10 tok/s or accepting an uncaptured serving implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import torch

from engine.modules.speculative_tree import Tree, dflash_candidates
from engine.modules.w4a8_dataflow import W4A8Plan, W4A8PipelinePlan
from engine.profiles.glm53.tree_decode import Verification
from engine.profiles.glm53.net import Step
from tests.test_engine_tree_decode import TreeDecodeTests

BASELINE = "b8e42ab9b57b179a66a7af4ef28f5ab89d7e3e5a"
ROOT = Path(__file__).resolve().parents[1]


def load_baseline(ref):
    modules, hashes = {}, {}
    for name, path in (("kda", "engine/modules/tree_kda.py"),
                       ("decode", "engine/profiles/glm53/tree_decode.py"),
                       ("proposal", "engine/modules/speculative_tree.py")):
        source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT).decode()
        module = types.ModuleType("_tree_bench_baseline_"+name)
        sys.modules[module.__name__] = module
        exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
        modules[name], hashes[path] = module, hashlib.sha256(source.encode()).hexdigest()
    modules["decode"].tree_kda = modules["kda"]
    return modules, hashes


def bracket(before, after, iterations, prepare=None):
    for fn in (before, after):
        for _ in range(3):
            if prepare:
                prepare()
            fn()
    samples = {"before": [], "after": []}
    for _ in range(2):
        for name, fn in (("before", before), ("after", after), ("after", after), ("before", before)):
            for _ in range(iterations):
                if prepare:
                    prepare()
                start = time.perf_counter_ns()
                fn()
                samples[name].append((time.perf_counter_ns()-start)/1.e6)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return dict(median_ms=medians, reduction_percent=100*(1-medians["after"]/medians["before"]), samples_ms=samples)


def main(output, baseline, iterations):
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("CPU comparison requires CUDA devices to be hidden")
    torch.set_num_threads(1)
    old, baseline_hashes = load_baseline(baseline)
    cases, linear_cases = [], []
    for parents in ((-1, 0, 1, 2, 3, 4, 5, 6), (-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)):
        tree = Tree(tuple(range(1, len(parents)+1)), parents)
        for context in (0, 129, 1024):
            net, caches, slot = TreeDecodeTests().prepare(context)
            def run(cls):
                with cls(net, caches, tree, seq=0, slot=slot, context=context) as v:
                    return v.verify()
            expected, actual = run(old["decode"].Verification), run(Verification)
            torch.testing.assert_close(actual, expected, atol=.008, rtol=.008)
            same_tokens = torch.equal(net.head_tokens(actual), net.head_tokens(expected))
            if not same_tokens:
                raise AssertionError("tiny target token mismatch")
            timing = bracket(lambda: run(old["decode"].Verification), lambda: run(Verification), iterations)
            def digest(t):
                return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            cases.append(dict(kind="tiny-target-verify-including-prepare", nodes=len(parents), context=context,
                state_updates=tree.state_updates, before_output_sha256=digest(expected), after_output_sha256=digest(actual),
                head_tokens_equal=same_tokens, max_abs_error=float((expected.float()-actual.float()).abs().max()), **timing))
            if tree.parents == (-1,)+tuple(range(len(parents)-1)):
                state, paged = caches.state.clone(), caches.paged.clone()
                def restore():
                    caches.state.copy_(state); caches.paged.copy_(paged)
                def linear():
                    step = Step.decode([(torch.tensor(tree.tokens), context, 0, slot)])
                    caches.prepare(step)
                    return net.forward(step, caches)
                restore()
                default = linear()
                torch.testing.assert_close(default, actual, atol=.008, rtol=.008)
                same_tokens = torch.equal(net.head_tokens(default), net.head_tokens(actual))
                if not same_tokens:
                    raise AssertionError("ordinary linear target token mismatch")
                timing = bracket(linear, lambda: run(Verification), iterations, restore)
                linear_cases.append(dict(context=context, nodes=len(parents), head_tokens_equal=same_tokens,
                    scope="CPU target only: ordinary linear cache writes vs tree private verification; restore excluded",
                    max_abs_error=float((default.float()-actual.float()).abs().max()), **timing))
    # Same 7x64 selector, weights and support; isolate high/low confidence.
    proposal = []
    for gap in (0., 5.):
        gen = torch.Generator().manual_seed(382)
        support, steps = 64, 7
        ids = torch.arange(2, 2+support*steps).view(steps, support)
        unary = torch.zeros(steps, support); unary[:, 0] = gap
        codes = torch.randn(2+support*steps, 8, generator=gen)*.02
        args = (1, unary, ids, torch.ones(steps, 8), codes, codes, (1.,)*steps)
        def rows(fn):
            return fn(*args, width=2, max_nodes=31)
        timing = bracket(lambda: rows(old["proposal"].dflash_candidates), lambda: rows(dflash_candidates), iterations)
        records = {}
        for name, fn in (("before", old["proposal"].dflash_candidates), ("after", dflash_candidates)):
            cs = rows(fn)
            selected = old["proposal"].select(cs, 8, bytes_per_expert=1, fixed_node_bytes=1, cost_weight=0)
            records[name] = dict(max_depth=max(selected.tree.depths), proposal_mass=selected.proposal_mass,
                                 tokens=selected.tree.tokens, parents=selected.tree.parents)
        proposal.append(dict(gap=gap, support=support, selected_nodes=8, expansion=31, candidates=records,
                             **timing))
    queued, staged = W4A8Plan(16, 4096, 3072), W4A8PipelinePlan(16, 4096, 3072)
    source_paths = ("engine/modules/speculative_tree.py", "engine/modules/tree_kda.py",
                    "engine/modules/w4a8_dataflow.py", "engine/kernels/kda/tree.py",
                    "engine/kernels/w4a8_pipeline.py", "engine/profiles/glm53/tree_decode.py",
                    "engine/profiles/glm53/net.py", "engine/profiles/glm53/lanes.py")
    report = dict(scope="CPU #947 experiment comparison; not production serving or GPU performance", baseline=baseline,
        torch=torch.__version__, gpu_used=False, cuda_initialized=torch.cuda.is_initialized(),
        iterations_per_block=iterations, bracket="B/A/A/B repeated twice after warmup", target=cases, proposal=proposal,
        ordinary_linear=linear_cases,
        baseline_source_sha256=baseline_hashes,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        structural=dict(w4a8_rows=16, w4a8_hidden=4096, w4a8_intermediate=3072,
            w4a8_scratch_before=queued.scratch_bytes, w4a8_scratch_after=staged.scratch_bytes,
            w4a8_partial_write_read_bytes_removed=2*queued.producers*queued.rows*queued.hidden*4,
            w4a8_host_completion_reads_before=1, w4a8_host_completion_reads_after=0,
            dsa_indexer_calls_before="N", dsa_indexer_calls_after="1 if any pools, else 0",
            dsa_mla_calls_before="N", dsa_mla_calls_after=1,
            selector_transfer_before=7*64*64*4+7*64*8, selector_transfer_after=7*64*2*3*8+8),
        unmeasured=["real-weight GPU numerics", "native graph replay", "production baseline tok/s and TTFT",
                    "real acceptance and reasoning quality", "C4", "32K/128K peak memory"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"target_reduction_percent": [c["reduction_percent"] for c in cases],
                      "ordinary_linear_reduction_percent": [c["reduction_percent"] for c in linear_cases],
                      "proposal_reduction_percent": [c["reduction_percent"] for c in proposal],
                      "structural": report["structural"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    main(args.output, args.baseline, args.iterations)
