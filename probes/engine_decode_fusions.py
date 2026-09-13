"""Bounded kernel timings after the numerical gates; never an engine speed claim."""
import torch


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def _time(graph, iterations=256, flush=None):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        graph.replay()
    if flush:
        flush()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def seven_row_dense(report):
    from engine.kernels.dense import DenseLinear, W4Pack, w4_gemm, extension
    ext = extension()
    before, mode, state = ext.gemm_input_mode(), ext.gemm_input_cta_mode(), ext.probe_state()
    try:
        ext.set_gemm2(0)
        ext.set_input_cta(4)
        for n in (6416, 4096, 6144):
            layer = DenseLinear((torch.randn(n, 4096, device='cuda') * .02).bfloat16(), prefill=False)
            p = layer.packs[0]
            # More than L2: an isolated warm weight is not a model's stream.
            packs = [W4Pack(p.data.clone(), p.scale.clone(), p.rowscale.clone(), p.rows, p.cols)
                     for _ in range(8)]
            x = torch.randn(7, 4096, device='cuda', dtype=torch.bfloat16)
            graphs, outputs = [], []
            try:
                for enabled in (0, 1):
                    ext.set_gemm_input(enabled)
                    graph, out = _capture(lambda: [w4_gemm(x, weight) for weight in packs])
                    graphs.append(graph); outputs.append(out)
                for _ in range(3):
                    x.normal_()
                    for graph in graphs:
                        graph.replay()
                    for got, want in zip(outputs[1], outputs[0]):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
                measurements = [dict(arm=label, ms=_time(graphs[i], iterations=64)/len(packs)) for label, i in
                                (('B', 0), ('A', 1), ('A', 1), ('B', 0))]
                report('seven_row_dense_timing', rows=7, n=n, k=4096, distinct_packs=len(packs),
                       exact=True, plan=ext.gemm_input_cta_plan(7, n, 4096, False, False),
                       measurements=measurements, scope='same packs, captured projection; not consumer speed')
            finally:
                for graph in graphs:
                    graph.reset()
    finally:
        ext.set_gemm_input(before)
        ext.set_input_cta(mode)
        ext.restore_probe_state(state)


def tensorcore_router(report, ranks=None):
    from pathlib import Path
    from safetensors import safe_open
    from engine.kernels.glm_pointwise import router_logits, route_weights
    from engine.profiles.glm53 import facts
    root = Path(ranks or facts.RANKS)
    if not root.is_absolute():
        root = facts.RANKS.parent / root
    rank_file = root / 'rank0of4.safetensors'
    weights = []
    with safe_open(str(rank_file), framework='pt', device='cpu') as source:
        for key in sorted(k for k in source.keys() if k.endswith('.moe.gate')):
            gate = source.get_tensor(key).cuda()
            bias = source.get_tensor(key.removesuffix('gate')+'bias').cuda()
            weights.append((gate, gate.float(), bias))
    assert len(weights) == 42, 'real GLM router gate must cover every MoE layer'
    x = torch.randn(7, 4096, device='cuda', dtype=torch.bfloat16)
    max_logit_error = 0.
    checked_rows = 0
    for magnitude in (.01, .1, 1., 10.):
        for correlated in (False, True):
            x.normal_().mul_(magnitude)
            if correlated:
                x[1:].mul_(.02).add_(x[:1])
            for gate, fp32, bias in weights:
                base = x.float() @ fp32.T
                cand = router_logits(x, gate)
                torch.testing.assert_close(cand, base, rtol=5e-5, atol=3e-4)
                max_logit_error = max(max_logit_error, (cand-base).abs().max().item())
                ids, values = route_weights(cand, bias, 8, 2.5)
                ref_ids, ref_values = route_weights(base, bias, 8, 2.5)
                torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
                torch.testing.assert_close(values, ref_values, rtol=5e-5, atol=3e-6)
                checked_rows += x.shape[0]
    graphs = []
    try:
        for tensorcore in (False, True):
            def run():
                return [route_weights(router_logits(x, gate) if tensorcore else x.float() @ fp32.T,
                                      bias, 8, 2.5) for gate, fp32, bias in weights]
            graph, _ = _capture(run)
            graphs.append(graph)
        measurements = [dict(arm=label, ms=_time(graphs[i], iterations=64)) for label, i in
                        (('B', 0), ('A', 1), ('A', 1), ('B', 0))]
        report('tensorcore_router_timing', rows=7, layers=len(weights), checked_rows=checked_rows,
               rank_file=str(rank_file),
               selected_ids_exact=True, max_logit_error=max_logit_error, measurements=measurements,
               scope='real router weights, synthetic hidden states, captured; not consumer speed')
    finally:
        for graph in graphs:
            graph.reset()


def shared_mlp(report, native):
    from tests.test_engine_shared_mlp import SharedMLPTests
    gu, down, fused = SharedMLPTests.layers()
    for rows in (7, 28):
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
        base, _ = _capture(lambda: SharedMLPTests.reference(x, gu, down))
        cand, _ = _capture(lambda: fused(x))
        try:
            measurements = [dict(arm=label, ms=_time(graph)) for label, graph in
                            (("B", base), ("A", cand), ("A", cand), ("B", base))]
            report("shared_mlp_timing", rows=rows, intermediate=512, measurements=measurements,
                   scope="same-pack captured component; not consumer speed")
        finally:
            base.reset(); cand.reset()
    # The independent routed branch is the real serving CuTe lane, so this
    # also catches races between its workspace and native W4 graph scratch.
    from engine.kernels.dense.shared_mlp import SharedOverlap
    from engine.modules.nvfp4_sf import swizzle_sf
    experts, hidden, width = 288, 4096, 512
    w13 = torch.randint(0, 256, (experts, 2 * width, hidden // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, width // 2), device="cuda", dtype=torch.uint8)
    s13 = torch.stack([swizzle_sf(torch.full((2 * width, hidden // 16), .015625, device="cuda").to(torch.float8_e4m3fn))
                       for _ in range(experts)])
    s2 = torch.stack([swizzle_sf(torch.full((hidden, width // 16), .015625, device="cuda").to(torch.float8_e4m3fn))
                      for _ in range(experts)])
    overlap = SharedOverlap("cuda")
    for rows in (7, 28):
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * .3
        selected = (torch.arange(rows * 8, device="cuda").reshape(rows, 8) % experts).int()
        weights = torch.full((rows, 8), 1 / 8, device="cuda")
        def routed():
            return native.moe(x, selected, weights, w13, s13, w2, s2, 10.)
        graphs = []
        outputs = []
        try:
            for fn in (lambda: routed() + SharedMLPTests.reference(x, gu, down),
                       lambda: routed() + fused(x),
                       lambda: overlap(fused, x, routed)):
                graph, output = _capture(fn)
                graphs.append(graph); outputs.append(output)
            for regime in ("distinct_routes", "reused_routes"):
                if regime == "reused_routes":
                    selected.copy_(torch.arange(8, device="cuda", dtype=torch.int32).expand(rows, 8))
                for _ in range(3):
                    x.normal_()
                    for graph in graphs:
                        graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)
                    SharedMLPTests().close(outputs[1], outputs[0])
                measurements = [dict(arm=label, ms=_time(graphs[i], iterations=64)) for label, i in
                                (("B", 0), ("F", 1), ("O", 2), ("O", 2), ("F", 1), ("B", 0))]
                report("shared_moe_overlap_timing", rows=rows, experts=experts, topk=8,
                       regime=regime, replay_exact=True, measurements=measurements,
                       scope="captured routed plus shared components; no model or communication")
        finally:
            for graph in graphs:
                graph.reset()


def pointwise(report):
    from engine.kernels.glm_pointwise import swiglu_clamped, route_weights, layernorm
    from engine.kernels.norm_rope import norm
    from engine.profiles.glm53.lanes import swiglu_clamped as activation_ref
    from engine.profiles.glm53.net import rmsnorm
    x = torch.randn(7, 2560, device="cuda", dtype=torch.bfloat16)
    g, u = x.chunk(2, -1)
    logits = torch.randn(7, 288, device="cuda")
    bias = torch.randn(288, device="cuda")
    w = torch.randn(128, device="cuda")
    k = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    nw = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
    def route_ref():
        s = logits.sigmoid()
        sel = (s + bias).topk(8, -1).indices
        weights = s.gather(-1, sel)
        return sel.int(), weights / weights.sum(-1, keepdim=True) * 2.5
    for name, baseline, candidate in (
        ("shared_activation", lambda: activation_ref(g, u, 10.), lambda: swiglu_clamped(g, u, 10.)),
        ("router_post_projection", route_ref, lambda: route_weights(logits, bias, 8, 2.5)),
        ("rmsnorm", lambda: rmsnorm(g, nw, 1e-6), lambda: norm(g, nw, 1e-6)),
        ("indexer_norm", lambda: torch.nn.functional.layer_norm(k.float(), (128,), w, w, 1e-6).bfloat16(),
         lambda: layernorm(k, w, w, 1e-6)),
        ("expert_join", lambda: (g.float() + u.float()).bfloat16(), lambda: g + u),
    ):
        base, base_out = _capture(baseline)
        cand, cand_out = _capture(candidate)
        measurements = []
        for label, graph in (("B", base), ("A", cand), ("A", cand), ("B", base)):
            measurements.append(dict(arm=label, ms=_time(graph)))
        report("pointwise_timing", operation=name, rows=7, measurements=measurements,
               scope="single-GPU captured kernel; not consumer speed")


def residency(report):
    from engine.kernels.draft_observe import write_context
    from engine.kernels.draft_attention import write_draft_kv_rows
    from engine.kernels.norm_rope import norm_rope, warm
    n, t, layers, heads, dim = 1, 7, 5, 2, 128
    context = torch.randn(n, t, layers, 2, heads, dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(layers, dim, device='cuda', dtype=torch.bfloat16)
    field = torch.zeros(5, layers, 2, 256, heads, dim, device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(t, device='cuda', dtype=torch.int64).reshape(n, t) + 254
    slots = torch.tensor([3], device='cuda', dtype=torch.int64)
    valid = torch.tensor([t], device='cuda', dtype=torch.int64)
    warm(context.device, dim, 10000.)
    def original_write():
        for layer in range(layers):
            key = norm_rope(context[:, :, layer, 0].reshape(n * t, heads, dim), weights[layer], 1e-6,
                            positions.reshape(-1), 10000.).reshape(n, t, heads, dim)
            write_draft_kv_rows(field, slots, layer, positions, key, context[:, :, layer, 1], valid=valid)
    x = torch.randn(7, 4096, device='cuda', dtype=torch.bfloat16)
    gate = torch.randn(288, 4096, device='cuda', dtype=torch.bfloat16)
    gate_fp32 = gate.float()
    for name, base_fn, cand_fn in (
        ('all_layer_context_write', original_write,
         lambda: write_context(field, slots, positions, context, weights, valid, 1e-6, 10000.)),
        ('router_projection_resident', lambda: x.float() @ gate.float().T, lambda: x.float() @ gate_fp32.T),
    ):
        base, base_out = _capture(base_fn)
        cand, cand_out = _capture(cand_fn)
        if base_out is not None:
            torch.testing.assert_close(base_out, cand_out, rtol=0, atol=0)
        measurements = [dict(arm=label, ms=_time(graph)) for label, graph in (('B', base), ('A', cand), ('A', cand), ('B', base))]
        report('residency_timing', operation=name, rows=t, measurements=measurements,
               scope='single-GPU captured component; not consumer speed')


def calibration(report):
    from engine.kernels.dense.calibration import Calibration
    class Layer:
        observer = None
        input_dtype = torch.bfloat16
    # Wide drafter fc, the Gram which dominates the uncalibrated serving boot.
    width = 20480
    for rows in (1, 7):
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(rows, device="cuda", dtype=torch.bool)
        c = Calibration("cuda", max_decode_rows=28)
        layer = Layer()
        c.attach("fc", layer, [("fc", 0, width, True)], small_rows=True)
        h = torch.zeros_like(c.H["fc"])
        peaks = torch.zeros_like(c.amax["fc"])
        count = torch.zeros_like(c.rows["fc"])
        def baseline():
            xf = x.float() * mask.float().view(-1, 1) * c.armed
            h.addmm_(xf.T, xf)
            count.add_(mask.float().sum() * c.armed)
            torch.maximum(peaks, xf.abs().amax(0), out=peaks)
        base, _ = _capture(baseline)
        cand, _ = _capture(lambda: layer.observer(x, mask))
        c.flush()  # compile force-flush while disarmed, outside every timed arm
        c.arm()
        measurements = []
        for label, graph in (("B", base), ("A", cand), ("A", cand), ("B", base)):
            measurements.append(dict(arm=label, ms=_time(graph, flush=c.flush if label == "A" else None)))
        # Both arms saw exactly 512 timed updates; captures were disarmed.
        torch.testing.assert_close(c.H["fc"], h, rtol=8e-5, atol=.02)
        torch.testing.assert_close(c.rows["fc"], count, rtol=0, atol=0)
        torch.testing.assert_close(c.amax["fc"], peaks, rtol=0, atol=0)
        report("calibration_timing", rows=rows, width=width, measurements=measurements,
               includes_partial_flush=True, staging_dtype=str(c.staging['fc'][0].dtype),
               scope="single-GPU captured observer; not consumer speed")
        del base, cand, h, c, layer
