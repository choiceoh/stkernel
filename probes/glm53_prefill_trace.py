"""CPU-only validation and descriptive attribution of one worker's fresh trace.

No trace statistic is a TTFT result. Category sums overlap on different streams;
their interval union and the unattributed category remain explicit.
"""
import math
import re

PREFIX = 'GLM53_PREFILL_OBSERVER '
BUCKETS = [(label, re.compile(pattern, re.I)) for label, pattern in (
    ('transport', r'nccl|all_?reduce|reduce_scatter|all_gather|allgather'),
    ('moe', r'moe|expert|grouped_gemm|group_gemm|router|routing'),
    ('attention', r'\bmla\b|_mla_|fmha|flashinfer.*attention|attn|indexer|mqa|paged|chunk_kda|kda_'),
    ('mhc', r'mhc|sinkhorn'),
    ('dense_gemm', r'gemm|matmul|cublas|nvjet|wgmma|einsum|f8f8'),
    ('norm_rope_quant', r'norm|rope|rotary|quant'),
)]


def union_us(intervals):
    total = 0.0
    end = None
    for a, b in sorted(intervals):
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (a, b)) or b < a:
            raise ValueError('invalid trace interval')
        total += b - a if end is None or a >= end else max(0, b - end)
        end = b if end is None else max(end, b)
    return total


def fresh_trace(before, after):
    """One new regular trace on a rank; reject replacements and ambiguity."""
    if any(name not in after or after[name] != value for name, value in before.items()):
        raise ValueError('pre-existing trace changed or disappeared')
    new = set(after) - set(before)
    if len(new) != 1:
        raise ValueError('one new trace per rank required')
    name = new.pop()
    state = after[name]
    if (not name.endswith(('.pt.trace.json', '.pt.trace.json.gz'))
            or state.get('symlink') is not False or state.get('regular') is not True
            or type(state.get('size')) is not int or state['size'] <= 0
            or any(type(state.get(k)) is not int for k in ('device', 'inode', 'mtime_ns'))):
        raise ValueError('invalid new trace identity')
    return name


def analyze(events, observation):
    import json
    if (observation.get('mode') != 'profile' or observation.get('complete') is not True
            or observation.get('hook_restored') is not True or observation.get('errors')
            or not observation.get('records') or not observation.get('moe_forward_groups')):
        raise ValueError('complete restored profile observation required')
    annotations, intervals, categories, kernels = [], [], {}, {}
    devices = set()
    for event in events:
        if event.get('ph') != 'X':
            continue
        name = event.get('name', '')
        if name.startswith(PREFIX):
            annotations.append(json.loads(name[len(PREFIX):]))
        cat = event.get('cat', '')
        if 'kernel' not in cat and cat not in ('gpu_memcpy', 'gpu_memset'):
            continue
        ts, dur = event.get('ts'), event.get('dur')
        if (type(ts) not in (int, float) or type(dur) not in (int, float)
                or not math.isfinite(ts) or not math.isfinite(dur) or dur < 0):
            raise ValueError('invalid GPU event duration')
        device = event.get('args', {}).get('device')
        if device is None:
            raise ValueError('GPU event device identity missing')
        devices.add(device)
        interval = (ts, ts + dur)
        intervals.append(interval)
        label = 'memops' if cat in ('gpu_memcpy', 'gpu_memset') else next(
            (label for label, pattern in BUCKETS if pattern.search(name)), 'unknown')
        categories.setdefault(label, []).append(interval)
        row = kernels.setdefault(name, dict(name=name, category=label, calls=0, sum_us=0.0))
        row['calls'] += 1
        row['sum_us'] += dur
    if sorted(annotations, key=lambda r: r['call']) != observation['records']:
        raise ValueError('trace annotations differ from exact request/rank/call coverage')
    if not intervals or len(devices) != 1:
        raise ValueError('one worker GPU device and nonempty trace required')
    window = max(b for _, b in intervals) - min(a for a, _ in intervals)
    busy = union_us(intervals)
    if window <= 0:
        raise ValueError('empty GPU capture window')
    return dict(schema=1, rank=observation['rank'], request_id=observation['request_id'],
        source_sha256=observation['source_sha256'], moe_forward_groups=observation['moe_forward_groups'],
        capture_gpu_span_us=window, gpu_interval_union_us=busy,
        gap_inside_gpu_span_us=window-busy, kernel_and_memop_sum_us=sum(b-a for a,b in intervals),
        categories={name:dict(calls=len(items), sum_us=sum(b-a for a,b in items), union_us=union_us(items))
                    for name, items in sorted(categories.items())},
        kernels=sorted(kernels.values(), key=lambda r:r['sum_us'], reverse=True),
        limitations=['Category names are heuristic; unknown kernels are retained.',
                     'GPU capture span is not request wall time or TTFT.',
                     'Category sums and unions overlap; they are not additive TTFT shares.',
                     'Executed MoE rows are observed wrapper input sizes, not an inferred scheduler token count.'],
        performance_acceptance=False)
