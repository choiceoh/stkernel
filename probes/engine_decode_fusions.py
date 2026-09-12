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
        c.arm()
        measurements = []
        for label, graph in (("B", base), ("A", cand), ("A", cand), ("B", base)):
            measurements.append(dict(arm=label, ms=_time(graph, flush=c.flush if label == "A" else None)))
        # Both arms saw exactly 512 timed updates; captures were disarmed.
        torch.testing.assert_close(c.H["fc"], h, rtol=8e-5, atol=.02)
        torch.testing.assert_close(c.rows["fc"], count, rtol=0, atol=0)
        torch.testing.assert_close(c.amax["fc"], peaks, rtol=0, atol=0)
        report("calibration_timing", rows=rows, width=width, measurements=measurements,
               includes_partial_flush=True, scope="single-GPU captured observer; not consumer speed")
        del base, cand, h, c, layer
