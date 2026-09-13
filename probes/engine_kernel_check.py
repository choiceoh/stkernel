"""Judge ST kernel imports and numerical contracts with vLLM imports forbidden.

Use --imports-only in a CPU container, or --lanes to run a bounded GPU subset.
The default checks engine references and graph replay. MoE migration is judged
against the seed's original FlashInfer kernel, with the torch oracle's error
reported separately because FP4 quantization has discontinuous rounding.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.abc
import json
from pathlib import Path
import pkgutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ForbidVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise AssertionError(f"ST kernel attempted a vLLM import: {fullname}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument("--lanes", default="conv,kda,mhc,indexer,kpool,mla,moe")
    parser.add_argument("--ranks", help="exact consumer rank directory for real router validation")
    parser.add_argument("--moe-experts", type=int, choices=(8, 288), default=8,
                        help="8 for bounded smoke; 288 for GLM's full TP4 expert geometry")
    parser.add_argument("--moe-static", default="stock", help="served b12x static-lane spec (STK_moe_static): stock | t,r,sf6[,q0]")
    parser.add_argument("--mla-prefill", default="stock", help="served MLA prefill mode (STK_mla_prefill): stock | tile32 | pair | pair4")
    args = parser.parse_args()
    if args.lanes in ('scatter_bundle', 'batch_fusions', 'batch_boundaries'):
        from probes.engine_decode_bundle import check as decode_bundle
        decode_bundle(args.ranks, bundle=args.lanes)
        return
    sys.meta_path.insert(0, ForbidVllm())
    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)

    import torch
    import engine.kernels
    from engine.profiles.glm53 import lanes

    imported = [m.name for m in pkgutil.walk_packages(engine.kernels.__path__, "engine.kernels.")]
    for name in imported:
        importlib.import_module(name)
    native, ref = lanes.served(moe_static=args.moe_static, mla_prefill=args.mla_prefill), lanes.reference()
    rows = []

    def report(name, **values):
        row = dict(lane=name, **values)
        rows.append(row)
        print(json.dumps(row), flush=True)

    report("imports", modules=len(imported), table=native.name, vllm_loaded=False)
    if args.imports_only:
        return
    assert torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    torch.manual_seed(29)
    selected = set(args.lanes.split(","))
    assert selected <= {"conv", "kda", "kda-storage", "mhc", "indexer", "kpool", "mla", "moe", "moe_route_scatter", "moe_direct_scatter", "moe_route_direct", "paired_projection", "indexer_boundary", "wide_input", "calibration", "pointwise", "residency", "latency", "shared_mlp", "kda_ring", "decode7", "decode_rows"}, selected

    if selected & {'moe_route_scatter', 'moe_direct_scatter', 'moe_route_direct', 'paired_projection', 'indexer_boundary', 'wide_input'}:
        from probes.engine_decode_bundle import require_current_probe
        require_current_probe()
    if 'indexer_boundary' in selected:
        from probes.engine_decode_batch import indexer_check
        indexer_check(report, args.ranks)
    if 'wide_input' in selected:
        from probes.engine_decode_batch import wide_check
        wide_check(report, args.ranks)
    if selected & {'moe_route_scatter', 'moe_direct_scatter', 'moe_route_direct'}:
        from probes.engine_decode_scatter_check import moe_check
        for lane in ('moe_route_scatter', 'moe_direct_scatter', 'moe_route_direct'):
            if lane in selected:
                moe_check(report, args.ranks, lane)

    if 'paired_projection' in selected:
        from probes.engine_decode_projection import paired_check
        if 'paired_projection' in selected:
            paired_check(report, args.ranks)

    if "decode_rows" in selected:
        import unittest
        # every kernel a captured decode step folds over its rows (45차, the C=4 question) against its one-row
        # launches, byte for byte: the KDA rings, the conv ring, the pool/tail writers and the slot finalizer
        suite = unittest.defaultTestLoader.loadTestsFromNames(["tests.test_engine_kda_ring", "tests.test_engine_conv_ring",
                                                               "tests.test_engine_state", "tests.test_engine_pool_slots",
                                                               "tests.test_engine_indexer_rows"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "decode row-fold checks did not pass"
        report("decode_rows", passed=True, tests=result.testsRun)

    if "decode7" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_decode_seven")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "seven-row dense/router numerical gates did not pass"
        report("decode7", passed=True, tests=result.testsRun)
        from probes.engine_decode_fusions import seven_row_dense, tensorcore_router
        seven_row_dense(report)
        tensorcore_router(report, ranks=args.ranks)

    if "kda_ring" in selected:
        import unittest
        # the conv ring rides with the recurrent ring: a decode step folds both over its rows (net._kda)
        suite = unittest.defaultTestLoader.loadTestsFromNames(["tests.test_engine_kda_ring", "tests.test_engine_conv_ring"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "KDA ring numerical/replay checks did not pass"
        report("kda_ring", passed=True, tests=result.testsRun)

    if "shared_mlp" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_shared_mlp")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "shared MLP numerical/replay checks did not pass"
        report("shared_mlp", passed=True, tests=result.testsRun)
        from probes.engine_decode_fusions import shared_mlp
        shared_mlp(report, native)


    if "kda-storage" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames(
            ["tests.test_engine_kda_ring", "tests.test_engine_boundary_stage"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "KDA FP32/FP16 storage checks did not pass"
        report("kda-storage", passed=True, tests=result.testsRun, arithmetic="fp32", storage=["fp32", "fp16"])

    if "residency" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_decode_residency")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "decode residency checks did not pass"
        report("residency", passed=True, tests=result.testsRun)
        from probes.engine_decode_fusions import residency
        residency(report)

    if "calibration" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_calibration_gram")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "calibration GPU checks did not pass"
        report("calibration", passed=True, tests=result.testsRun)

    if "pointwise" in selected:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_glm_pointwise")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and not result.skipped, "pointwise GPU checks did not pass"
        report("pointwise", passed=True, tests=result.testsRun)

    # Numerical/replay contracts always precede timings. These explicitly
    # scoped component results do not replace the four-node onepass gate.
    if selected & {"calibration", "pointwise"}:
        from probes.engine_decode_fusions import calibration, pointwise
        if "pointwise" in selected:
            pointwise(report)
        if "calibration" in selected:
            calibration(report)

    if 'latency' in selected:
        from probes.engine_latency_check import check as latency_check
        report('latency', **latency_check())

    def rand(*shape, scale=1.):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale

    def check(name, got, expected, tolerance=.02):
        error = ((got.float() - expected.float()).abs().max() /
                 expected.float().abs().max().clamp_min(1e-6)).item()
        assert torch.isfinite(got).all().item() and error <= tolerance, (name, error, tolerance)
        return error

    if "conv" in selected:
        for t in (1, 6, 96):
            x, w = rand(t, 512), rand(512, 4, scale=.3).float()
            for seeded in (False, True):
                initial = rand(512, 3) if seeded else None
                saved = initial.clone() if seeded else None
                expected, state = ref.conv_prefill(x, w, initial)
                out, actual = native.conv_prefill(x, w, initial)
                assert not seeded or torch.equal(initial, saved)
                report("conv", tokens=t, seeded=seeded,
                       output=check("conv", out, expected), state=check("conv state", actual, state, 0.))

    if "kda" in selected:
        from probes.engine_kda_check import main as check_kda
        check_kda()
        report("kda", passed=True, cases=6, every_verify_state=True)

    if "mhc" in selected:
        fn, scale, base, norm = rand(24, 16384, scale=.01).float(), rand(3).float(), rand(24).float(), rand(4096)
        for t in (1, 6, 8, 65):
            residual = rand(t, 4, 4096)
            inputs = (residual, fn, scale, base, 1e-6, 1e-6, 2., 20, norm, 1e-6)
            expected = ref.mhc_pre(*inputs)
            actual = native.mhc_pre(*inputs)
            pre = [check("mhc pre", a, b) for a, b in zip(actual, expected)]
            x = rand(t, 4096)
            post = check("mhc post", native.mhc_post(x, residual, *actual[:2]),
                         ref.mhc_post(x, residual, *actual[:2]))
            report("mhc", tokens=t, pre=pre, post=post)

    if "indexer" in selected:
        q = rand(8, 16, 128).to(torch.float8_e4m3fn)
        k = rand(256, 128).to(torch.float8_e4m3fn)
        scale, weight = rand(256).float().abs(), rand(8, 16).float().abs()
        ends = torch.tensor([0, 1, 2, 8, 16, 64, 128, 256], device="cuda", dtype=torch.int32)
        actual = native.indexer_logits(q, k, scale, weight, ends)
        expected = ref.indexer_logits(q, k, scale, weight, ends)
        mask = torch.arange(256, device="cuda")[None, :] < ends[:, None]
        report("indexer", valid_logits=check("indexer logits", actual[mask], expected[mask]))

    if "kpool" in selected:
        for pools in (1, 3, 17):
            k, score, ape = rand(pools, 4, 128), rand(pools, 4, 128), rand(4, 128).float()
            actual = native.kpool_compress(k, score, ape)
            expected = ref.kpool_compress(k, score, ape)
            assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8)), "kpool fp8 bytes"
            assert torch.equal(actual[1], expected[1]), "kpool scales"
            report("kpool", pools=pools, bytes_equal=True, scales_equal=True)

    if "mla" in selected:
        from engine.kernels import mla
        mla.maybe_arm()  # Includes six ragged/decode/prefill numerical fixtures.
        q = rand(1, 16, 512, scale=.3)
        cache = rand(64, 512, scale=.5).to(torch.float8_e4m3fn)
        slots = torch.arange(64, device="cuda", dtype=torch.int32)[None, :]
        valid = torch.tensor([64], device="cuda", dtype=torch.int32)
        expected = ref.mla_sparse(q, cache, slots, valid, 512**-.5, .7)
        native.mla_sparse(q, cache, slots, valid, 512**-.5, .7)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = native.mla_sparse(q, cache, slots, valid, 512**-.5, .7)
        for _ in range(3):
            graph.replay()
        report("mla", boot_selftest=True, graph_replays=3, output=check("mla graph", out, expected))

    if "moe" in selected:
        torch.manual_seed(29)  # Independent of which other lanes were selected.
        # Only the migration judge calls the original library implementation;
        # the served table always calls engine.kernels.b12x.
        from flashinfer.fused_moe import b12x_fused_moe as original_b12x
        from engine.modules.nvfp4_sf import mma_sf_view, swizzle_sf
        e, h, intermediate = args.moe_experts, 4096, 512
        topk = 8 if e == 288 else 2
        w13 = torch.randint(0, 256, (e, 2*intermediate, h//2), device="cuda", dtype=torch.uint8)
        w2 = torch.randint(0, 256, (e, h, intermediate//2), device="cuda", dtype=torch.uint8)
        s13 = torch.stack([swizzle_sf(torch.full((2*intermediate, h//16), .015625, device="cuda").to(torch.float8_e4m3fn)) for _ in range(e)])
        s2 = torch.stack([swizzle_sf(torch.full((h, intermediate//16), .015625, device="cuda").to(torch.float8_e4m3fn)) for _ in range(e)])
        sf13, sf2 = mma_sf_view(s13, 2*intermediate, h), mma_sf_view(s2, h, intermediate)
        ones = torch.ones(e, device="cuda")
        for t in (1, 8, 129):
            x = rand(t, h, scale=.3)
            sel = (torch.arange(t*topk, device="cuda").reshape(t, topk) % e).to(torch.int32)
            weights = torch.full((t, topk), 1/topk, device="cuda")
            inputs = (x, sel, weights, w13, s13, w2, s2, 10.)
            out = native.moe(*inputs)
            expected = original_b12x(x=x, w1_weight=w13, w1_weight_sf=sf13, w2_weight=w2, w2_weight_sf=sf2,
                                    token_selected_experts=sel, token_final_scales=weights,
                                    num_experts=e, top_k=topk, w1_alpha=ones, w2_alpha=ones, fc2_input_scale=ones,
                                    activation="swigluoai_uninterleave", swiglu_alpha=1., swiglu_beta=0.,
                                    swiglu_limit=10., activation_precision="fp4", quant_mode="nvfp4")
            oracle = ref.moe(*inputs)
            oracle_error = ((out.float()-oracle.float()).abs().max()/oracle.float().abs().max().clamp_min(1e-6)).item()
            report("moe", tokens=t, experts=e, topk=topk, original_kernel=check("b12x migration", out, expected),
                   torch_oracle_relative=oracle_error)
            zero = native.moe(x, sel, weights * 0, w13, s13, w2, s2, 10.)
            assert torch.count_nonzero(zero).item() == 0, "zero-weight routes must not contribute"
            if t == 1:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = native.moe(*inputs)
                for _ in range(3):
                    graph.replay()
                report("moe_graph", replays=3, output=check("b12x graph", captured, expected))

    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
    print(json.dumps(dict(passed=True, vllm_loaded=False, checks=rows)), flush=True)


if __name__ == "__main__":
    main()
