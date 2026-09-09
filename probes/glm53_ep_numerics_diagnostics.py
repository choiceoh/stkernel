"""Bounded failure-only records; no new candidate execution or acceptance rule."""
from itertools import islice
import math
import struct

MAX_BAD_ROWS = 8
MAX_COLUMNS = 8
OUTPUTS = ("B1", "B2", "B3", "C1")


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bf16(word):
    return struct.unpack("<f", struct.pack("<I", (int(word) & 0xffff) << 16))[0]


def build_failure_record(rows, *, total_bad_rows):
    """Format bounded CPU records, selecting worst FP32 differences from BF16.

    Metrics and limits are the values from the actual failed comparison;
    they are never inferred from the global maxima or recomputed in Python.
    Full captured rows are discarded after selecting at most eight columns.
    """
    captured = []
    for row in islice(rows, MAX_BAD_ROWS):
        raw = row["raw_bf16"]
        fields = OUTPUTS + (("X",) if "X" in raw else ())
        widths = {len(raw[name]) for name in fields}
        if len(widths) != 1 or not next(iter(widths)):
            raise ValueError("diagnostic output rows must have matching nonempty widths")
        metrics = row["metrics"]
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("diagnostic metrics must be finite")
        violations = dict(l2=metrics["relative_l2"] > metrics["l2_limit"],
                          peak=metrics["relative_peak"] > metrics["peak_limit"])
        if not any(violations.values()):
            raise ValueError("diagnostic row did not fail its original limits")
        if len(row["routes"]) != 8:
            raise ValueError("diagnostics require the original top8 routes")
        # Match the original comparison's FP32 subtraction. Stable column
        # ordering only breaks equal-magnitude ties for this diagnostic view.
        delta = [abs(_f32(_bf16(c) - _bf16(b)))
                 for b, c in zip(raw["B1"], raw["C1"])]
        columns = sorted(range(len(delta)), key=lambda col: (-delta[col], col))[:MAX_COLUMNS]
        captured.append(dict(
            row_id=int(row["row_id"]), metrics=dict(metrics), violations=violations,
            routes=row["routes"],
            worst_columns=[dict(column=col, absolute_delta_f32=delta[col],
                                raw_bf16_u16={name: int(raw[name][col]) & 0xffff
                                              for name in fields}) for col in columns]))
    if (not captured or type(total_bad_rows) is not int
            or total_bad_rows < len(captured)):
        raise ValueError("diagnostic bad-row count is inconsistent")
    return dict(schema=1, diagnostic_only=True, first_failure_preserved=True,
                total_bad_rows=total_bad_rows, captured_bad_rows=len(captured),
                truncated=total_bad_rows > len(captured),
                selection="first bad rows in ascending row order; worst absolute-delta columns",
                input_column_scope="X is the same input-column index, not a causal attribution",
                raw_dtype="bfloat16", rows=captured)


def capture_failure(candidate, baseline, repeat, third, *, bad, error, peak,
                    noise, peak_noise, l2_limits, peak_limits, total_bad_rows,
                    route_ids, route_weights, expert_map, scales, inputs):
    """Copy only after FAIL: <=8 full output rows, route metadata and scales.

    At H4096 the four outputs plus input snapshot total at most 320 KiB. Only the
    worst eight columns are retained in JSON. No model/kernel call is made.
    """
    import torch
    if any(t.dtype != torch.bfloat16 for t in (baseline, repeat, third, candidate, inputs)):
        raise ValueError("raw failure capture requires BF16 outputs")
    if route_ids.shape[1] != 8 or route_weights.shape[1] != 8:
        raise ValueError("raw failure capture requires top8 routing")
    indices = bad.nonzero(as_tuple=False).flatten()[:MAX_BAD_ROWS].detach().cpu().tolist()
    mapping = expert_map.detach().cpu().tolist()
    scale_values = {name: values.detach().cpu().tolist() for name, values in scales.items()}

    def records():
        for row_id in indices:
            b1 = baseline[row_id].float()
            metrics = dict(relative_l2=float(error[row_id]), relative_peak=float(peak[row_id]),
                           stock_relative_l2=float(noise[row_id]),
                           stock_relative_peak=float(peak_noise[row_id]),
                           l2_limit=float(l2_limits[row_id]), peak_limit=float(peak_limits[row_id]),
                           b1_l2_norm=float(b1.norm()), b1_absmax=float(b1.abs().amax()))
            ids = route_ids[row_id].detach().cpu().tolist()
            weights = route_weights[row_id].detach().cpu().tolist()
            routes = []
            for slot, (expert, weight) in enumerate(zip(ids, weights)):
                mapped = mapping[expert] if 0 <= expert < len(mapping) else -1
                values = {name: items[mapped] if 0 <= mapped < len(items) else None
                          for name, items in scale_values.items()}
                routes.append(dict(slot=slot, global_expert_id=int(expert),
                                   local_expert_id=int(mapped), weight=weight, scales=values))
            raw = {name: tensor[row_id].detach().cpu().view(torch.int16).tolist()
                   for name, tensor in zip(OUTPUTS + ("X",),
                                           (baseline, repeat, third, candidate, inputs))}
            yield dict(row_id=row_id, metrics=metrics, routes=routes, raw_bf16=raw)

    return build_failure_record(records(), total_bad_rows=total_bad_rows)
