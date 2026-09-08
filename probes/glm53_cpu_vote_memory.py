"""Read-only private worker/API memory receipts for the rank-cache vote pair."""
import hashlib
import os
from pathlib import Path
import re

import glm53_prefill_observer as observer

KNOB = 'VLLM_GLM53_RANK_CACHE_CPU_VOTE'


def mapping_summary(text):
    rows = []
    current = None
    for line in text.splitlines():
        if re.match(r'^[0-9a-f]+-[0-9a-f]+\s', line):
            current = {} if line.endswith('/dev/zero (deleted)') else None
            if current is not None:
                rows.append(current)
        elif current is not None:
            match = re.fullmatch(r'(Size|Rss|Pss):\s+(\d+) kB', line)
            if match:
                current[match[1]] = int(match[2])
    if any(set(r) != {'Size', 'Rss', 'Pss'} for r in rows):
        raise ValueError('incomplete zero mapping counters')
    histogram = {}
    for row in rows:
        size = str(row['Size'])
        histogram[size] = histogram.get(size, 0) + 1
    return dict(count=len(rows), size_histogram_kib=histogram,
                totals_kib={key: sum(row[key] for row in rows) for key in ('Size', 'Rss', 'Pss')})


def snapshot(torch):
    before = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    result = observer.memory_snapshot(torch)
    status = Path('/proc/self/status').read_text()
    result['vmpin_kib'] = int(re.search(r'^VmPin:\s+(\d+) kB$', status, re.M)[1])
    result['zero_mappings'] = mapping_summary(Path('/proc/self/smaps').read_text())
    after = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    if before != after:
        raise ValueError('process identity changed during memory snapshot')
    result.update(pid=os.getpid(), start_ticks=before)
    return result


class WorkerExtension(observer.WorkerExtension):
    def glm53_cpu_vote_memory(self):
        import torch
        import torch.distributed as dist
        from vllm.model_executor.layers import glm53_rank_cache
        if observer._SESSION is not None:
            raise RuntimeError('memory receipt requires idle observation hooks')
        parallel = self.vllm_config.parallel_config
        if (dist.get_world_size() != 4 or parallel.tensor_parallel_size != 4
                or parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
                or parallel.enable_expert_parallel):
            raise RuntimeError('memory receipt requires plain TP4')
        return dict(rank=dist.get_rank(), active=False, policy=os.environ.get(KNOB, '0'),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    rank_cache_sha256=hashlib.sha256(Path(glm53_rank_cache.__file__).read_bytes()).hexdigest(),
                    memory=snapshot(torch))


def validate(report, source_sha, rank_cache_sha, policy):
    ranks = report.get('ranks', [])
    if (len(ranks) != 4 or any(type(r.get('rank')) is not int for r in ranks)
            or {r['rank'] for r in ranks} != {0, 1, 2, 3}):
        raise ValueError('exact four rank memory receipts required')
    for row in [*ranks, report.get('api', {})]:
        if row.get('active') is not False or row.get('source_sha256') != source_sha:
            raise ValueError('memory receipt source or idle state differs')
        memory = row['memory']
        for value in (memory['pid'], memory['process_kib']['Pss'], memory['host_kib']['MemAvailable']):
            if type(value) is not int or value <= 0:
                raise ValueError('invalid process memory counters')
        if type(memory['vmpin_kib']) is not int or memory['vmpin_kib'] < 0:
            raise ValueError('invalid pinned memory counter')
        maps = memory['zero_mappings']
        if (type(maps['count']) is not int or maps['count'] < 0
                or any(not re.fullmatch(r'\d+', k) or type(v) is not int or v < 0
                       for k, v in maps['size_histogram_kib'].items())
                or sum(maps['size_histogram_kib'].values()) != maps['count']
                or sum(int(k)*v for k, v in maps['size_histogram_kib'].items()) != maps['totals_kib']['Size']
                or any(type(v) is not int or v < 0 for v in maps['totals_kib'].values())):
            raise ValueError('invalid zero mapping totals')
    if any(r.get('policy') != policy or r.get('rank_cache_sha256') != rank_cache_sha
           or r['memory']['cuda_initialized'] is not True for r in ranks):
        raise ValueError('worker policy, loaded source or CUDA state differs')
    return sorted(ranks, key=lambda r: r['rank'])


async def middleware(request, call_next):
    if request.url.path != '/glm53/cpu-vote-memory':
        return await observer.middleware(request, call_next)
    from fastapi.responses import JSONResponse
    if request.method != 'POST' or request.client is None or request.client.host not in ('127.0.0.1', '::1'):
        return JSONResponse({'error': 'local POST required'}, status_code=403)
    try:
        if await request.json() != {}:
            raise ValueError('memory receipt accepts no options')
        ranks = await request.app.state.engine_client.collective_rpc('glm53_cpu_vote_memory', timeout=60)
        import torch
        from starlette.concurrency import run_in_threadpool
        api = dict(active=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   memory=await run_in_threadpool(snapshot, torch))
        return JSONResponse(dict(ranks=ranks, api=api))
    except Exception as exc:
        return JSONResponse({'error': repr(exc)}, status_code=400)
