"""Offline latency attribution. Durations are sums; elapsed time is an interval union.

No device imports. The original trace remains authoritative, including unmapped
events. A kernel name alone cannot identify a layer or a graph invocation.
"""
from collections import defaultdict
import math
from statistics import median


def union_us(intervals):
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if b < a:
            continue
        total += max(0, b - max(a, end if end is not None else a))
        end = max(b, end if end is not None else b)
    return total


def _arg(args, *names):
    normalized = {str(k).lower().replace('_', ' '): v for k, v in args.items()}
    return next((normalized[n] for n in names if n in normalized), None)


def attribute(trace, graph_labels=None):
    """Keep each GPU activity, joining eager external IDs or captured graph node IDs.

    Chrome trace timestamps are local to this rank. Cross-rank clocks must not
    be subtracted. CPU launch duration is never substituted for device time.
    """
    events = trace.get('traceEvents', [])
    scopes = [e for e in events if e.get('ph') == 'X' and str(e.get('name', '')).startswith('st.op/')]
    launches = {}
    for e in events:
        if e.get('ph') != 'X' or not str(e.get('cat', '')).startswith(('cpu_op', 'cuda_runtime', 'cuda_driver')):
            continue
        external = _arg(e.get('args', {}), 'external id')
        if external is None:
            continue
        parents = [s for s in scopes if s.get('tid') == e.get('tid') and
                   s['ts'] <= e['ts'] and e['ts'] + e.get('dur', 0) <= s['ts'] + s.get('dur', 0)]
        if parents:
            launches[str(external)] = min(parents, key=lambda s: s['dur'])['name'][6:]
    rows = []
    graph_labels = graph_labels if graph_labels is not None else trace.get('st_graph_labels', {})
    for e in events:
        cat = str(e.get('cat', '')).lower()
        if e.get('ph') != 'X' or not any(k in cat for k in ('kernel', 'gpu_memcpy', 'gpu_memset')):
            continue
        start, duration = e.get('ts'), e.get('dur')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, duration)) or duration < 0:
            continue
        args = e.get('args', {})
        node = _arg(args, 'graph node id', 'cuda graph node id')
        graph = _arg(args, 'graph id', 'cuda graph id')
        external = _arg(args, 'external id')
        op = graph_labels.get(str(node)) if node is not None else None
        origin = 'graph_node' if op else 'external_id'
        if not op:
            op = launches.get(str(external))
        rows.append(dict(kind='gpu_activity', operation=op or 'unmapped', attribution=origin if op else 'unmapped',
                         kernel=e.get('name'), category=cat, start_us=start, duration_us=duration,
                         device=args.get('device'), stream=args.get('stream'), graph_id=graph, graph_node_id=node))
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        duration = row.get('duration_us')
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
            key = tuple(row.get(k) for k in ('rank', 'phase', 'kind', 'operation', 'kernel'))
            groups[key].append(row)
    result = []
    for key, members in groups.items():
        values = [r['duration_us'] for r in members]
        result.append(dict(zip(('rank', 'phase', 'kind', 'operation', 'kernel'), key),
                           samples=len(values), sum_us=sum(values), mean_us=sum(values) / len(values),
                           median_us=median(values), min_us=min(values), max_us=max(values)))
    return sorted(result, key=lambda r: r['sum_us'], reverse=True)
