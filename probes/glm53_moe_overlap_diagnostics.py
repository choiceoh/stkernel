"""Component attribution only; this module cannot grant a serving gate pass.

Use the same fixed seed, weights, routes and unmodified numerical thresholds
as glm53_moe_overlap_check. Collect failures rather than retrying until pass.
"""
import hashlib
import json
import math
from pathlib import Path


def routed_tile_budget(counts, tile_m):
    if type(tile_m) is not int or tile_m <= 0:
        raise ValueError('positive tile size required')
    if len(counts) != 288 or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('288 nonnegative integer expert counts required')
    tiles = sum((n + tile_m - 1) // tile_m for n in counts)
    return dict(tile_m=tile_m, routed_rows=sum(counts), active_experts=sum(n > 0 for n in counts),
                physical_tiles=tiles, padded_rows=tiles * tile_m,
                padding_rows=tiles * tile_m - sum(counts), counts=counts)


def timing_orders():
    # Every arm appears twice in each position; include both pair orders.
    return ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1))


def diagnostic_record(*, transport, rows, skew, phase, ranks):
    if transport not in ('bf16', 'fp8-v3') or len(ranks) != 4:
        raise ValueError('transport and all four ranks required')
    return dict(kind='MOE_OVERLAP_DIAGNOSTIC', serving_gate=False,
                transport=transport, rows=rows, skew=skew, phase=phase,
                activation_seed=9211 + rows, weight_seeds=list(range(73209, 73213)), ranks=ranks)


def error_summary(errors, noise, peaks, noise_peaks, *, limits=None):
    if not errors or len({len(v) for v in (errors, noise, peaks, noise_peaks)}) != 1:
        raise ValueError('nonempty, aligned per-row metrics required')
    if limits is not None and (len(limits) != 2 or any(len(v) != len(errors) for v in limits)):
        raise ValueError('aligned device-computed limits required')
    indices, details = [], []
    finite = True
    clean = lambda value: value if math.isfinite(value) else None
    for i, (error, repeat, peak, repeat_peak) in enumerate(zip(errors, noise, peaks, noise_peaks)):
        valid = all(math.isfinite(v) and v >= 0 for v in (error, repeat, peak, repeat_peak))
        finite &= valid
        l2_limit, peak_limit = (max(3*repeat, .02), max(3*repeat_peak, .04)) if limits is None else (limits[0][i], limits[1][i])
        if not valid or error > l2_limit or peak > peak_limit:
            indices.append(i)
            if len(details) < 32:
                details.append(dict(row=i, relative_l2=clean(error), relative_peak=clean(peak),
                    repeat_l2=clean(repeat), repeat_peak=clean(repeat_peak),
                    l2_limit=clean(l2_limit), peak_limit=clean(peak_limit)))
    maximum = lambda values: max((v for v in values if math.isfinite(v)), default=None)
    return dict(finite=finite, bad_rows=len(indices), bad_row_indices=indices, first_bad_rows=details,
                max_row_relative_l2=maximum(errors), max_row_relative_abs=maximum(peaks),
                repeat_l2=maximum(noise), repeat_peak=maximum(noise_peaks))


def row_errors(torch, a, b, repeat):
    a, b, r = (v.float() for v in (a, b, repeat))
    finite = all(bool(torch.isfinite(v).all()) for v in (a, b, r))
    norm = b.norm(dim=1).clamp_min(1e-6)
    peak = b.abs().amax(dim=1).clamp_min(1e-6)
    error = (a-b).norm(dim=1)/norm
    noise = (r-b).norm(dim=1)/norm
    worst = (a-b).abs().amax(dim=1)/peak
    npeak = (r-b).abs().amax(dim=1)/peak
    # Identical to the gate: no enlarged tolerance or repeated-sample envelope.
    l2_limit = torch.maximum(3*noise, torch.full_like(noise, .02))
    peak_limit = torch.maximum(3*npeak, torch.full_like(npeak, .04))
    report = error_summary(error.tolist(), noise.tolist(), worst.tolist(), npeak.tolist(),
                           limits=(l2_limit.tolist(), peak_limit.tolist()))
    report.update(finite=finite and report['finite'], equal=torch.equal(a, b))
    return report


def run_diagnostics(*, args, torch, dist, h, md, mlp, context, group, rank, provenance, require):
    from vllm.forward_context import override_forward_context

    def gather(value):
        ranks = [None]*4
        dist.all_gather_object(ranks, value, group=group.cpu_group)
        return ranks

    def emit(rows, skew, phase, value):
        record = diagnostic_record(transport=args.transport, rows=rows, skew=skew,
                                   phase=phase, ranks=gather(value))
        if rank == 0:
            print(json.dumps(record, allow_nan=False), flush=True)

    def partial(x):
        with override_forward_context(context), h.partial_tp_output(num_tokens=x.shape[0]):
            return mlp(x)

    def budget(x):
        counts = torch.bincount(mlp.routes(x)[0].flatten().long(), minlength=288).tolist()
        tile_m = md._select_dynamic_tile_m(x.shape[0]*8, 288, 'swigluoai_uninterleave')
        return routed_tile_budget(counts, tile_m)

    def slowest_ms(start, end):
        elapsed = torch.tensor(start.elapsed_time(end), device='cuda')
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group.device_group)
        return float(elapsed)

    for rows in args.rows:
        for skew in (False, True):
            mlp.skew = skew
            generator = torch.Generator(device='cuda').manual_seed(9211+rows)
            full = torch.randn(rows, 4096, generator=generator, device='cuda', dtype=torch.bfloat16)*.5
            shard = h.prefill_shard(full)
            expected_shard = shard.clone()
            valid = min(shard.shape[0], max(0, rows-rank*shard.shape[0]))

            def stock():
                x = h.prefill_all_gather(shard, num_tokens=rows)
                return h.prefill_reduce_scatter(partial(x))

            def overlap():
                with override_forward_context(context):
                    out = h.prefill_moe_overlap(mlp, shard, num_tokens=rows)
                return stock() if out is None else out

            def serial(spans=None):
                reduced = []
                for index, (start, stop, actual) in enumerate(h._mlp_overlap_slices(rows)):
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if spans is not None else None
                    if events: events[0].record()
                    x = h.prefill_all_gather(shard[start:stop], num_tokens=actual, _transport_tokens=rows)
                    if events: events[1].record()
                    value = partial(x)
                    if events: events[2].record()
                    reduced.append(h.prefill_reduce_scatter(value, _transport_tokens=rows))
                    if events:
                        events[3].record()
                        spans.append(events)
                return torch.cat(reduced, dim=0)

            # Four independent calls, keeping the original first-three order.
            b0, b1, a, b2 = stock(), stock(), overlap(), stock()
            torch.cuda.synchronize()
            emit(rows, skew, 'whole_path', dict(overlap_admitted=rows >= 6144,
                candidate=row_errors(torch, a[:valid], b0[:valid], b1[:valid]),
                stock_control=row_errors(torch, b2[:valid], b0[:valid], b1[:valid])))
            del b0, b1, a, b2

            # Gather once so the partial-MoE replay sees exactly the same input.
            x = h.prefill_all_gather(shard, num_tokens=rows)
            x_repeat = h.prefill_all_gather(shard, num_tokens=rows)
            fixed = [partial(x) for _ in range(4)]
            torch.cuda.synchronize()
            emit(rows, skew, 'local_moe_partials', dict(gather_equal=torch.equal(x, x_repeat),
                repeats=[row_errors(torch, q, fixed[0], fixed[1]) for q in fixed[2:]]))

            # Identical sender buffers isolate codec/collective repeatability.
            replay = [h.prefill_reduce_scatter(fixed[0]) for _ in range(3)]
            varied = [h.prefill_reduce_scatter(q) for q in fixed]
            torch.cuda.synchronize()
            emit(rows, skew, 'transport_replay', dict(
                fixed=row_errors(torch, replay[2][:valid], replay[0][:valid], replay[1][:valid]),
                varied=[row_errors(torch, q[:valid], varied[0][:valid], varied[1][:valid]) for q in varied[2:]]))
            del fixed, replay, varied, x_repeat

            if rows >= 6144:
                require(torch.equal(shard, expected_shard), 'diagnostic input changed during eager replay')
                # The failed full gate's timing ladder follows this exact
                # activation/route change. Preserve it for the cost diagnosis.
                shard.mul_(-.75)
                expected_shard = shard.clone()
                del x
                x = h.prefill_all_gather(shard, num_tokens=rows)
                stripes = [h.prefill_all_gather(shard[start:stop], num_tokens=actual, _transport_tokens=rows)
                           for start, stop, actual in h._mlp_overlap_slices(rows)]
                full_budget = budget(x)
                split_budget = [budget(s) for s in stripes]
                counts_match = all(a+b == c for a, b, c in zip(
                    split_budget[0]['counts'], split_budget[1]['counts'], full_budget['counts']))
                require(counts_match, 'diagnostic stripe routing counts differ from full input')
                emit(rows, skew, 'changed_routed_tiles', dict(input_multiplier=-.75, stock=full_budget, stripes=split_budget,
                     stripe_physical_tiles=sum(v['physical_tiles'] for v in split_budget),
                     stripe_padded_rows=sum(v['padded_rows'] for v in split_budget)))
                del stripes
                values = [serial(), serial(), overlap()]
                torch.cuda.synchronize()
                emit(rows, skew, 'changed_serial_vs_overlap', dict(input_multiplier=-.75,
                    comparison=row_errors(torch, values[2][:valid], values[0][:valid], values[1][:valid])))
                del values
                timing = [[], [], []]
                for order in timing_orders():
                    for arm in order:
                        dist.barrier(group=group.device_group)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(); value = (stock, serial, overlap)[arm](); end.record(); end.synchronize()
                        timing[arm].append(slowest_ms(start, end))
                # Event-instrumented serial spans are separate from the balanced
                # full-call timings above; these spans must not be added to overlap.
                spans = []
                value = serial(spans); torch.cuda.synchronize()
                component_ms = [[slowest_ms(events[i], events[i+1]) for i in range(3)] for events in spans]
                emit(rows, skew, 'changed_timing', dict(input_multiplier=-.75, slowest_rank_ms=dict(zip(('stock', 'serial', 'overlap'), timing)),
                    serial_stripe_components_ms=component_ms, component_order=['all_gather', 'moe', 'reduce_scatter']))
            require(torch.equal(shard, expected_shard), 'diagnostic input changed')
            del x, shard, expected_shard, full
    if rank == 0:
        source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        print(json.dumps(dict(verdict='MOE_OVERLAP_DIAGNOSTIC_COMPLETE', serving_gate=False,
              transport=args.transport, provenance=provenance, diagnostic_source_sha256=source_sha)), flush=True)
