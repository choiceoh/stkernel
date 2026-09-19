"""Qwen3.8 concurrent decode: split a wide EP launch into existing micro launches.

Above eight tokens the served lane uses the static kernel and maps foreign routes
to expert zero at weight zero. The micro kernel skips these routes. Compare the
served launch with two balanced micro chunks, including their output
concatenation, through both a prototype and the opt-in serving lane. No serving
default changes here. Inputs, weights, activation scale search and route order
are shared.

The router is excluded from both arms: local ids are staged before replay, exactly
as one fused router would produce them (foreign id zero for static, E for micro).
Every arm must pass the independent activation-quantised oracle, finite/zero
checks and replay on new inputs before timing. Alternate cold/warm samples over
different routes and a 64 MiB cache scrub. These are single-GPU component times.

    bash bench/fleet.sh run --gpu qwen38-moe-chunks 12 'concurrent decode MoE chunks' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_moe_chunks
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics


def chunks(rows: int, limit: int) -> tuple[tuple[int, int], ...]:
    """Balanced contiguous ranges, never a one-token micro remainder."""
    if rows < 2 or not 2 <= limit <= 8:
        raise ValueError("micro chunks need rows >= 2 and a limit in 2..8")
    count = (rows + limit - 1) // limit
    width, extra = divmod(rows, count)
    if width < 2:
        raise ValueError("this limit cannot cover the rows without a single-token chunk")
    offset = 0
    result = []
    for index in range(count):
        end = offset + width + (index < extra)
        result.append((offset, end))
        offset = end
    return tuple(result)


def run(output=None):
    import torch
    from engine.base import kernel_shape as ks
    from engine.kernels.b12x import moe_dispatch as md
    from engine.profiles.qwen38 import lanes
    from probes import engine_qwen38_moe as p
    from probes.engine_decode_fusions import _capture
    from probes.engine_qwen38_moe_precision import Quant
    from probes.probe_report import write_report

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    events, metrics = [], {}

    def report(event, **values):
        row = dict(event=event, **values)
        events.append(row)
        print(json.dumps(row), flush=True)
        if output:
            Path(output).write_text("".join(json.dumps(e) + "\n" for e in events))
        return row

    shape = ks.bind(p.kernel_shape())
    c = p.cell_of(shape)
    probe = p._Probe(report, md, lanes, c, Quant(lanes.MOE_ACTIVATION_SCALE_SEARCH))
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    checks = 0
    try:
        with torch.inference_mode(), p.pinned(md, _EP_ZERO_WEIGHT_MICRO_CELL=md._EP_ZERO_WEIGHT_MICRO_CELL,
                                             **{name: None for name in p.HOOKS}):
            probe.setup(shape)
            integrated_lane = lanes.served(moe_decode_chunks=True)
            # Largest first, as capture at boot; then a C=1 guard and the C=2 shape.
            for rows in (16, 15, 14, 13, 12, 11, 10, 9, 8, 4):
                patterns = [p.pattern(rows, c, probe.generator, "cuda", min_local=2) for _ in range(8)]
                x0, ids0, w0 = patterns[0]
                scores = torch.randn(rows, c.experts, generator=probe.generator, device="cuda")
                route_args = dict(experts=c.experts, first_expert=c.first_expert,
                                  w13=probe.layers[0].w13, hidden=c.hidden)
                base_ids, base_weights = probe.lane.route_local(scores, c.topk, **route_args)
                split_ids, split_weights = integrated_lane.route_local(scores, c.topk, **route_args)
                base_sentinel = c.local if rows <= 8 else 0
                comparable = torch.where(split_ids == c.local, base_sentinel, split_ids)
                if not torch.equal(base_weights, split_weights) or not torch.equal(base_ids, comparable):
                    raise AssertionError(f"split router changed local experts or weights at {rows} tokens")
                foreign = (x0, p.foreign_routes(ids0, c), w0)
                zeros = (x0, ids0, torch.zeros_like(w0))
                # Two independent oracle inputs catch chunk output aliasing and row reordering.
                oracles = [p.oracle(*source, probe.layers[0], c, probe.quant) for source in patterns[:2]]
                arms = []
                variants = [("served", None, None), ("chunks8", 8, None), ("integrated", None, None)]
                for name, limit, tile in variants:
                    if limit is not None and rows <= limit and tile is None:
                        continue
                    ranges = ((0, rows),) if limit is None else chunks(rows, limit)
                    expected_ranges = lanes.moe_decode_ranges(rows, True) if name == "integrated" else ranges
                    sentinel = c.local if max(end - begin for begin, end in expected_ranges) <= 8 else None
                    lane = integrated_lane if name == "integrated" else probe.lane

                    def local(source):
                        x, ids, weights = source
                        li, lw = lanes.local_routes(ids, weights, c.first_expert, c.local, sentinel)
                        return x, li, lw

                    staged = [local(source) for source in patterns]
                    inputs = [t.clone() for t in staged[0]]
                    layer = probe.layers[0]

                    def call():
                        x, ids, weights = inputs
                        parts = [lane.moe(x[begin:end], ids[begin:end], weights[begin:end],
                                                layer.w13, layer.w13_sf, layer.w2, layer.w2_sf,
                                                scales=layer.scales, first_expert=c.first_expert, local=True)
                                 for begin, end in ranges]
                        return parts[0] if len(parts) == 1 else torch.cat(parts)

                    with p.pinned(md, _MICRO_TILE_M_OVERRIDE=tile), p.Launches(md) as launches:
                        graph, out = _capture(call)
                        probe.owners.append(lane.graph_resources())
                        launch_rows = launches.log[-len(expected_ranges):]
                        eager = call().clone()
                        graph.replay()
                        replay = out.clone()
                        p.load(inputs, staged[1])
                        eager_new = call().clone()
                        graph.replay()
                        replay_new = out.clone()
                        nonzero = []
                        for source in (foreign, zeros):
                            p.load(inputs, local(source))
                            graph.replay()
                            nonzero.append(int(torch.count_nonzero(out)))
                        p.load(inputs, staged[0])
                    relative = max(p.relative(replay, oracles[0]), p.relative(replay_new, oracles[1]))
                    control = (replay, replay_new) if name == "served" else arms[0]["control"]
                    baseline_relative = max(p.relative(replay, control[0]), p.relative(replay_new, control[1]))
                    baseline_equal = bool(torch.equal(replay, control[0]) and torch.equal(replay_new, control[1]))
                    valid = (bool(torch.isfinite(replay).all()) and bool(torch.isfinite(replay_new).all())
                             and relative <= p.ORACLE_RELATIVE and not any(nonzero)
                             and p.stable(replay, eager) and p.stable(replay_new, eager_new))
                    if rows <= 8:
                        valid = valid and baseline_equal  # unchanged C=1/C=2 arithmetic, byte for byte
                    expected = "micro" if sentinel is not None else "static"
                    valid = valid and len(launch_rows) == len(expected_ranges) and all(
                        row["kernel"] == expected and (sentinel is None or row.get("skip") == c.local)
                        and (tile is None or row.get("tile") == [tile, 128]) for row in launch_rows)
                    report("chunk_check", tokens=rows, arm=name, ranges=expected_ranges, launched=launch_rows,
                           oracle_relative=relative, zero_nonzero=nonzero,
                           baseline_relative=baseline_relative, baseline_equal=baseline_equal, router_same=True,
                           replay_stable=p.stable(replay, eager), new_replay_stable=p.stable(replay_new, eager_new),
                           passed=valid)
                    if not valid:
                        graph.reset()
                        raise AssertionError(f"{name} failed its correctness gate at {rows} tokens")
                    checks += 1
                    arms.append(dict(name=name, graph=graph, inputs=inputs, out=out, staged=staged,
                                     expected=replay, control=(replay, replay_new), cold=[], warm=[]))

                # Replaying another arm may share dispatcher scratch. Verify isolation before timing.
                for arm in reversed(arms):
                    p.load(arm["inputs"], arm["staged"][0])
                    arm["graph"].replay()
                    if not p.stable(arm["out"], arm["expected"]):
                        raise AssertionError(f"scratch alias after other arms: {rows} {arm['name']}")
                for repeat in range(4):
                    order = arms if repeat % 2 == 0 else list(reversed(arms))
                    for index in range(len(patterns)):
                        for arm in order:
                            p.load(arm["inputs"], arm["staged"][index])
                            probe.trash.fill_(repeat + index)
                            arm["cold"].append(probe.timed(arm["graph"].replay))
                            arm["warm"].append(probe.timed(arm["graph"].replay))
                baseline = statistics.median(arms[0]["cold"])
                for arm in arms:
                    median = statistics.median(arm["cold"])
                    report("chunk_timing", tokens=rows, arm=arm["name"], cold_us=median,
                           warm_us=statistics.median(arm["warm"]), speedup=baseline / median,
                           cold_samples=arm["cold"], warm_samples=arm["warm"])
                    metrics[f"m{rows}_{arm['name']}_cold_us"] = median
                    arm["graph"].reset()
                torch.cuda.synchronize()
            write_report(metrics, {"oracle_and_replay": True, "scratch_isolation": True}, checks,
                         torch.cuda.get_device_name())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return events
