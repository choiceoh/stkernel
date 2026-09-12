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


def kda_ring(report):
    """The prior BV=8 against BV=16 on distinct per-layer state, no product knob."""
    from unittest.mock import patch
    from engine.kernels.kda import ring as impl
    kernel = impl.fused_recurrent_gated_delta_rule_fwd_kernel

    class ValueTile:
        def __init__(self, width):
            self.width = width

        def __getitem__(self, grid):
            def launch(**kwargs):
                kwargs['BV'] = self.width
                return kernel[(1, (kwargs['V'] + self.width - 1) // self.width, kwargs['HV'])](**kwargs)
            return launch

    for seqs in (1, 4):
        layers, t, h, d = 34, 7, 16, 128
        states = torch.randn(layers, seqs + 1, t, h, d, d, device='cuda') * .1
        initial = states.clone()
        inputs = []
        for _ in range(layers):
            merged = torch.randn(seqs, t, 3 * h * d, device='cuda', dtype=torch.bfloat16)
            q, k, v = (x.reshape(seqs, t, h, d) for x in merged.split(h * d, dim=-1))
            g = torch.randn_like(q)
            beta = torch.randn(seqs, t, h, device='cuda', dtype=torch.bfloat16)
            a = torch.randn(h, device='cuda') * .2
            bias = torch.randn(h * d, device='cuda') * .1
            inputs.append((q, k, v, g, beta, a, bias))
        slot = torch.arange(1, seqs + 1, device='cuda', dtype=torch.int64)
        context = torch.full((seqs,), 2048, device='cuda', dtype=torch.int64)

        def run():
            return [impl.recurrent_kda_ring(*(x[s:s+1] for x in args[:5]), *args[5:],
                                            states[L], slot[s:s+1], context[s:s+1], -5.)
                    for L, args in enumerate(inputs) for s in range(seqs)]

        graphs, outputs = [], []
        try:
            for bv in (8, 16):
                with patch.object(impl, 'fused_recurrent_gated_delta_rule_fwd_kernel', ValueTile(bv)):
                    graph, output = _capture(run)
                graphs.append(graph); outputs.append(output)
            for step in range(3):
                context.fill_(2048 + step * 3)
                for args in inputs:
                    for x in args[:5]:
                        x.normal_()
                states.copy_(initial)
                graphs[0].replay()
                reference_states = states.clone()
                reference_outputs = [x.clone() for x in outputs[0]]
                states.copy_(initial)
                graphs[1].replay()
                assert torch.equal(states, reference_states), 'KDA BV=16 changed a rollback snapshot'
                for actual, expected in zip(outputs[1], reference_outputs):
                    assert torch.equal(actual, expected), 'KDA BV=16 changed an output'
                del reference_states, reference_outputs
            measurements = [dict(arm=label, ms=_time(graphs[i], iterations=64)) for label, i in
                            (('B8', 0), ('A16', 1), ('A16', 1), ('B8', 0))]
            report('kda_ring_timing', seqs=seqs, tokens=t, layers=layers, every_state_exact=True,
                   measurements=measurements, scope='captured 34-layer recurrence; not consumer speed')
        finally:
            for graph in graphs:
                graph.reset()
        del states, initial, inputs, outputs, graphs


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
