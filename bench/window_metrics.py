"""Traffic checks for C=1 measurements; missing counters are not zero."""
import re


def metric_sum(text, name):
    values = re.findall(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", text, re.M)
    return sum(map(float, values)) if values else None


def traffic_state(text):
    return {key: metric_sum(text, metric) for key, metric in (
        ("finished", "vllm:request_success_total"),
        ("running", "vllm:num_requests_running"),
        ("waiting", "vllm:num_requests_waiting"))}


def exclusive_errors(before, after, samples, expected_requests):
    errors = []
    if any(v is None for state in (before, after, *samples) for v in state.values()):
        return ["missing traffic counters: exclusivity cannot be established"]
    if before["running"] + before["waiting"] != 0:
        errors.append("server was not idle before the workload")
    if after["running"] + after["waiting"] != 0:
        errors.append("requests remain after the workload")
    delta = after["finished"] - before["finished"]
    if delta != expected_requests:
        errors.append(f"completed requests {delta:g} != own requests {expected_requests}")
    if any(s["running"] + s["waiting"] > 1 for s in samples):
        errors.append("concurrent or queued external request observed")
    if any(b["finished"] < a["finished"] for a, b in zip([before, *samples], [*samples, after])):
        errors.append("request counter reset during workload")
    return errors


def decode_windows(samples, phases, fixed_phases=(), margin=1.0):
    """Keep intervals wholly inside one response, excluding both edge tails."""
    by_ctx, fixed = {}, []
    for (ta, sa), (tb, sb) in zip(samples, samples[1:]):
        if tb <= ta or sb <= sa:
            continue
        for phase in phases:
            ctx, start, end = phase
            if ta >= start + margin and tb <= end - margin:
                by_ctx.setdefault(ctx, []).append((sb - sa) / (tb - ta))
                if phase in fixed_phases:
                    fixed.append({"request": fixed_phases.index(phase),
                                  "steps": sb - sa, "seconds": tb - ta,
                                  "start": ta, "end": tb})
                break
    return by_ctx, fixed
