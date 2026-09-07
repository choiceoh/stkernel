#!/usr/bin/env python3
"""Attribute only explicitly annotated pure-prefill GPU ranges.

Per-category work sums, interval unions and non-overlapping occupancy are
separate fields. Profiler durations describe an instrumented execution, not
an uninstrumented latency prediction. No decode-step denominator is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import sys


def events(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        head = ''
        while '"traceEvents"' not in head:
            chunk = stream.read(65536)
            if not chunk:
                raise ValueError('traceEvents missing')
            head = head[-32:] + chunk
        tail = head[head.index('"traceEvents"'):]
        buf, depth, quoted, escaped, started = [], 0, False, False, False
        while True:
            for char in tail:
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == '\\':
                        escaped = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char == '{':
                    depth += 1
                    if depth == 1:
                        buf, started = [], True
                elif char == '}':
                    depth -= 1
                    if depth == 0 and started:
                        buf.append(char)
                        yield json.loads(''.join(buf))
                        buf, started = [], False
                        continue
                if started:
                    buf.append(char)
            tail = stream.read(1 << 20)
            if not tail:
                return


def category(name, parent=''):
    n, p = name.lower(), parent.lower()
    if 'nccl' in n or n.startswith(('k_oneshot', 'k_reduce', 'k_wait', 'k_signal', 'k_copy_in')):
        return 'communication'
    if n.startswith(('_pack_gather', '_unpack_gather', '_pack_rs', '_unpack_sum')):
        return 'transport_codec'
    if 'mhc' in n or 'hc_prenorm' in n or 'mhc' in p:
        return 'mhc'
    if 'moecute' in n or 'moe_static' in n:
        return 'moe_expert'
    if re.search(r'kda|chunk_|solve_tril|wy_|causal_conv|gather_initial_states|layer_norm_gated|qk_l2norm|recompute_w_u|merge_16x16_to_64x64', n):
        return 'kda'
    if re.search(r'mla|mqa|kpool|indexer|topkperrow|fwht', n) or re.search(r'mla|indexer|kpool', p):
        return 'mla_and_indexer'
    if re.search(r'moe|expert|single_group_topk|deneb_gate|moe_forward', n+' '+p):
        return 'moe_shared_and_glue'
    if re.search(r'gemm|cutlass|cublas|matmul|nvjet|aten::mm', n+' '+p):
        return 'dense_gemm_and_quant'
    if 'memcpy' in n or 'memset' in n:
        return 'memory_copy'
    if re.search(r'norm|quant|rope|silu|act_and_mul', n):
        return 'norm_quant_other'
    if re.search(r'sampl|reject|dflash|spec|argmax', n):
        return 'sampling_and_drafter'
    if re.search(r'elementwise|vectorized|unrolled|copy|scatter|gather|index_|triton_poi', n):
        return 'elementwise_and_copy'
    return 'other'


def union(intervals):
    out = []
    for start, end in sorted(intervals):
        if not out or start > out[-1][1]:
            out.append([start, end])
        else:
            out[-1][1] = max(end, out[-1][1])
    return out


def occupancy(kernels, start, end):
    boundaries = defaultdict(Counter)
    work, count = Counter(), Counter()
    for k in kernels:
        a, b = max(start, k['ts']), min(end, k['ts'] + k['dur'])
        if a >= b:
            continue
        cat = k['category']
        boundaries[a][cat] += 1
        boundaries[b][cat] -= 1
        work[cat] += b-a
        count[cat] += 1
    active, occupied, exclusive = Counter(), Counter(), Counter()
    busy = overlap = comm_compute_overlap = 0.0
    previous = start
    for t, changes in sorted(boundaries.items()):
        duration = t - previous
        cats = {c for c,n in active.items() if n > 0}
        if cats:
            busy += duration
            for c in cats:
                occupied[c] += duration
            if len(cats) == 1:
                exclusive[next(iter(cats))] += duration
            else:
                overlap += duration
            if 'communication' in cats and cats - {'communication'}:
                comm_compute_overlap += duration
        active.update(changes)
        previous = t
    span = end - start
    cats = {c: {'count':count[c], 'work_ms':work[c]/1000,
                'occupied_ms':occupied[c]/1000,
                'exclusive_ms':exclusive[c]/1000,
                'occupied_pct_span':100*occupied[c]/span}
            for c in sorted(count, key=lambda c: -occupied[c])}
    return {'span_ms':span/1000, 'busy_ms':busy/1000,
            'idle_ms':(span-busy)/1000, 'cross_category_overlap_ms':overlap/1000,
            'communication_compute_overlap_ms':comm_compute_overlap/1000,
            'summed_work_ms':sum(work.values())/1000, 'categories':cats}


def analyze(path):
    kernels, annotations, parents = [], [], {}
    for e in events(path):
        cat, args = e.get('cat'), e.get('args', {})
        if cat == 'cpu_op' and 'External id' in args:
            parents[args['External id']] = sys.intern(e['name'])
        elif cat in ('kernel', 'gpu_memcpy', 'gpu_memset') and e.get('ph') == 'X':
            kernels.append({'name':sys.intern(e['name']), 'ts':e['ts'],
                            'dur':e['dur'], 'external_id':args.get('External id'),
                            'stream':e.get('tid'), 'graph_id':args.get('graph id',0)})
        elif cat == 'gpu_user_annotation':
            match = re.fullmatch(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)',e['name'])
            if match and int(match[1]) > 0 and int(match[2]) > 0 and int(match[3]) == 0:
                annotations.append({'ts':e['ts'],'dur':e['dur'],'rows':int(match[2]),
                                    'external_id':args.get('External id'),'stream':e.get('tid')})
    if not annotations:
        raise ValueError('no explicit pure-prefill GPU ranges; refusing decode heuristic')
    chunks = {}
    for a in annotations:
        chunks.setdefault(a['external_id'],[]).append(a)
    ranges = []
    for ext, anns in chunks.items():
        if len({a['rows'] for a in anns}) != 1:
            raise ValueError('inconsistent row counts in a prefill range')
        ranges.append({'start':min(a['ts'] for a in anns),
                       'end':max(a['ts']+a['dur'] for a in anns),
                       'rows':anns[0]['rows'],'external_id':ext})
    ranges.sort(key=lambda a:a['start'])
    selected = []
    for k in kernels:
        if any(k['ts'] < r['end'] and k['ts']+k['dur'] > r['start'] for r in ranges):
            k['parent'] = parents.get(k['external_id'],'')
            k['category'] = category(k['name'],k['parent'])
            selected.append(k)
    start, end = ranges[0]['start'], max(r['end'] for r in ranges)
    report = occupancy(selected,start,end)
    report.update(path=str(path), prefill_tokens=sum(r['rows'] for r in ranges),
                  prefill_chunks=len(ranges), selected_events=len(selected),
                  excluded_events=len(kernels)-len(selected),
                  graph_events=sum(bool(k['graph_id']) for k in selected),
                  chunks=[dict(rows=r['rows'], **occupancy(selected,r['start'],r['end']))
                          for r in ranges])
    # The #439 standalone microbenchmark only covers RS unpack + MHC post.
    # Report that work on FP8-eligible chunks separately from all MHC work.
    target_intervals=[]
    for r in ranges:
        if r['rows'] < 4096:
            continue
        for k in selected:
            if k['name'] not in ('mhc_post_tilelang_kernel','_unpack_sum_payload'):
                continue
            a,b=max(r['start'],k['ts']),min(r['end'],k['ts']+k['dur'])
            if b>a:
                target_intervals.append((a,b))
    target_us=sum(b-a for a,b in union(target_intervals))
    report['rs_unpack_and_post_fp8_chunks']={
        'occupied_ms':target_us/1000,'pct_span':100*target_us/(end-start),
        'note':'Scope of the standalone microbenchmark; not an observed removable latency or speedup'}
    grouped = defaultdict(lambda: {'count':0,'work_us':0.0,'parents':Counter()})
    for k in selected:
        v=grouped[(k['category'],k['name'])]
        v['count']+=1; v['work_us']+=k['dur']; v['parents'][k['parent']]+=1
    report['kernels']=[dict(category=c,name=n,**v) for (c,n),v in
                       sorted(grouped.items(),key=lambda kv:-kv[1]['work_us'])]
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=analyze(args.trace)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('kernels','chunks')},indent=2))


if __name__ == '__main__':
    main()
