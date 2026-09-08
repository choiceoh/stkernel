"""Temporary worker instrumentation for an isolated, owned prefill baseline.

Loaded only with --worker-extension-cls glm53_prefill_observer.WorkerExtension
and --middleware glm53_prefill_observer.middleware on the diagnostic server.
No model hook is installed while idle. Profile mode records CPU annotations and
call metadata; routes mode copies selected expert IDs in a separate request.
Neither mode supplies performance acceptance. The collector must enforce fleet
ownership, private endpoint, all-rank identity and profiler-off timing separately.
"""
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
from pathlib import Path
import re


MAX_CALLS = 4096
EXPERTS = 288
TOP_K = 8
_SESSION = None


def histogram(ids, rows):
    if len(ids) != rows * TOP_K:
        raise ValueError('route slot count mismatch')
    counts = [0] * EXPERTS
    for expert in ids:
        if type(expert) is not int or not 0 <= expert < EXPERTS:
            raise ValueError('invalid or padded expert ID in exact TP4 census')
        counts[expert] += 1
    padded = {str(m): sum(((c + m - 1) // m) * m for c in counts) for m in (64, 128)}
    return dict(expert_counts=counts, routed_slots=sum(counts), padded_rows=padded,
                padded_work_reduction_pct=100 * (1 - padded['64'] / padded['128']))


def forward_groups(records, layers):
    """Require complete ordered target MoE cycles; never infer from max batch."""
    if not records or not layers or len(records) % len(layers):
        raise ValueError('missing or partial target MoE forward coverage')
    groups = []
    for start in range(0, len(records), len(layers)):
        block = records[start:start + len(layers)]
        if [r['layer'] for r in block] != layers or len({r['rows'] for r in block}) != 1:
            raise ValueError('target MoE layer order or executed rows differ')
        groups.append(dict(index=len(groups), executed_moe_rows=block[0]['rows'],
                           first_call=start, calls=len(block)))
    return groups


def validate_ranks(reports, *, request_id, mode, source_sha256):
    """Reject partial RPC success or incompatible worker/request evidence."""
    if (len(reports) != 4 or any(type(r.get('rank')) is not int for r in reports)
            or {r['rank'] for r in reports} != {0, 1, 2, 3}):
        raise ValueError('exactly four distinct worker reports required')
    ordered = sorted(reports, key=lambda r:r['rank'])
    for report in ordered:
        if (report.get('request_id') != request_id or report.get('mode') != mode
                or report.get('source_sha256') != source_sha256 or report.get('complete') is not True
                or report.get('hook_restored') is not True or report.get('errors')
                or report.get('performance_acceptance') is not False
                or report.get('numerical_acceptance') is not False):
            raise ValueError('worker identity, cleanup or observation completeness mismatch')
        records = report['records']
        if len(records) > MAX_CALLS or [r['call'] for r in records] != list(range(len(records))):
            raise ValueError('invalid or reordered worker call coverage')
        if any(type(r.get('rows')) is not int or not 1 <= r['rows'] <= 131072 for r in records):
            raise ValueError('invalid executed MoE row count')
        if any((r['rank'],r['request_id'],r['mode']) != (report['rank'],request_id,mode) for r in records):
            raise ValueError('mixed request/rank call records')
        if forward_groups(records, report['layers']) != report['moe_forward_groups']:
            raise ValueError('reported forward groups do not match calls')
        if mode == 'routes':
            for record in records:
                counts = record.get('expert_counts', [])
                if len(counts) != EXPERTS or any(type(c) is not int or c < 0 for c in counts):
                    raise ValueError('invalid expert histogram')
                total = record['rows'] * TOP_K
                if sum(counts) != total or record.get('routed_slots') != total:
                    raise ValueError('histogram lost routed slots')
                padded = {str(m):sum(((c+m-1)//m)*m for c in counts) for m in (64,128)}
                zero = record.get('zero_weight_slots')
                if record.get('padded_rows') != padded or type(zero) is not int or not 0 <= zero <= total:
                    raise ValueError('padding or zero-weight accounting mismatch')
                reduction = record.get('padded_work_reduction_pct')
                if (type(reduction) not in (int,float) or not math.isfinite(reduction)
                        or abs(reduction - 100*(1-padded['64']/padded['128'])) > 1e-9):
                    raise ValueError('padding reduction differs from recorded counts')
        elif mode != 'profile' or any('expert_counts' in r for r in records):
            raise ValueError('profiling and histogram instrumentation must be separate')
    if any(r['layers'] != ordered[0]['layers'] or r['moe_forward_groups'] != ordered[0]['moe_forward_groups'] for r in ordered[1:]):
        raise ValueError('target layer or executed MoE row coverage differs across ranks')
    return ordered


class Observation:
    def __init__(self, *, torch, wrapper_class, weights, rank, request_id, mode,
                 limit=MAX_CALLS):
        if mode not in ('profile', 'routes') or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', request_id):
            raise ValueError('invalid observation mode or request ID')
        if type(limit) is not int or not 1 <= limit <= MAX_CALLS:
            raise ValueError('invalid bounded call limit')
        if not weights or len(set(weights.values())) != len(weights):
            raise ValueError('missing or aliased target expert weights')
        self.torch, self.cls, self.weights = torch, wrapper_class, dict(weights)
        self.rank, self.request_id, self.mode, self.limit = rank, request_id, mode, limit
        self.layers = sorted(weights.values(), key=lambda n: int(re.search(r'(?:^|\.)layers\.(\d+)\.', n).group(1)))
        self.records, self.errors = [], []
        self.original = wrapper_class.run
        self.signature = inspect.signature(self.original)
        if not {'x', 'w1_weight', 'token_selected_experts', 'token_final_scales'} <= set(self.signature.parameters):
            raise ValueError('B12x run signature changed')
        self.hook = None

    def install(self):
        if self.hook is not None or self.cls.run is not self.original:
            raise RuntimeError('observation hook already installed or changed')
        def observed(wrapper, *args, **kwargs):
            # The original receives the exact objects and returns the exact
            # output. Any collection error is retained and invalidates end().
            annotation = nullcontext()
            try:
                bound = self.signature.bind(wrapper, *args, **kwargs).arguments
                if self.torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('capture occurred during eager observation')
                if len(self.records) >= self.limit:
                    raise RuntimeError('observation call budget exceeded')
                ids, x = bound['token_selected_experts'], bound['x']
                layer = self.weights.get(bound['w1_weight'].data_ptr())
                if layer is None:
                    raise RuntimeError('unmapped expert weight; target coverage unknown')
                if (len(ids.shape) != 2 or tuple(ids.shape)[1] != TOP_K
                        or len(x.shape) != 2 or x.shape[0] != ids.shape[0] or x.shape[1] != 4096
                        or not 1 <= ids.shape[0] <= 131072
                        or wrapper.num_experts != EXPERTS or wrapper.num_local_experts != EXPERTS):
                    raise RuntimeError('unsupported TP4 model or routing geometry')
                record = dict(call=len(self.records), layer=layer, rows=int(ids.shape[0]),
                              rank=self.rank, request_id=self.request_id, mode=self.mode)
                if self.mode == 'routes':
                    # No tensor, DLPack capsule or weight reference is retained.
                    flat = ids.detach().to(device='cpu', dtype=self.torch.int64).reshape(-1).tolist()
                    record.update(histogram(flat, record['rows']))
                    scales = bound['token_final_scales']
                    if tuple(scales.shape) != tuple(ids.shape):
                        raise ValueError('routing weight shape mismatch')
                    record['zero_weight_slots'] = int((scales == 0).sum().item())
                self.records.append(record)
                if self.mode == 'profile':
                    annotation = self.torch.profiler.record_function('GLM53_PREFILL_OBSERVER ' + json.dumps(record, sort_keys=True))
            except Exception as exc:
                if len(self.errors) < 16:
                    self.errors.append(type(exc).__name__ + ': ' + str(exc))
            with annotation:
                return self.original(wrapper, *args, **kwargs)
        self.hook = observed
        self.cls.run = observed

    def finish(self):
        if self.hook is None or self.cls.run is not self.hook:
            raise RuntimeError('observation hook identity changed; refusing to replace foreign code')
        self.cls.run = self.original
        self.hook = None
        try:
            groups = forward_groups(self.records, self.layers)
        except ValueError as exc:
            groups = []
            self.errors.append(str(exc))
        return dict(schema=1, rank=self.rank, request_id=self.request_id, mode=self.mode,
                    complete=not self.errors, errors=self.errors, layers=self.layers,
                    records=self.records, moe_forward_groups=groups, hook_restored=True,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    performance_acceptance=False, numerical_acceptance=False)


class WorkerExtension:
    def glm53_prefill_observe(self, op='status', **kwargs):
        global _SESSION
        import torch
        import torch.distributed as dist
        rank = dist.get_rank()
        if op == 'status':
            return dict(rank=rank, active=_SESSION is not None,
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if op == 'end':
            if _SESSION is None:
                raise RuntimeError('no active observation')
            session = _SESSION
            result = session.finish()
            _SESSION = None
            return result
        if op != 'begin' or _SESSION is not None:
            raise ValueError('unknown operation or observation already active')
        parallel = self.vllm_config.parallel_config
        if (dist.get_world_size() != 4 or parallel.tensor_parallel_size != 4
                or parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
                or parallel.enable_expert_parallel):
            raise RuntimeError('observer requires exact TP4, PP1, DP1 without EP')
        from flashinfer.fused_moe import B12xMoEWrapper
        model = self.model_runner.model
        targets = [(name, module) for name, module in model.named_modules()
                   if type(module).__name__ == 'Glm5NextModel']
        if len(targets) != 1:
            raise RuntimeError('one identifiable target GLM model required')
        prefix, target = targets[0]
        weights = {}
        for name, value in target.named_parameters(prefix=prefix):
            if name.endswith('.w13_weight') and re.search(r'(?:^|\.)layers\.\d+\.', name):
                pointer = value.data_ptr()
                if pointer in weights:
                    raise RuntimeError('aliased target expert weights')
                weights[pointer] = name
        session = Observation(torch=torch, wrapper_class=B12xMoEWrapper, weights=weights,
                              rank=rank, **kwargs)
        session.install()
        _SESSION = session
        return dict(rank=rank, active=True, request_id=session.request_id, mode=session.mode,
                    layers=session.layers,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


async def middleware(request, call_next):
    if request.url.path != '/glm53/prefill-observe':
        return await call_next(request)
    from fastapi.responses import JSONResponse
    # This extension is a probe mount, not a production overlay or public API.
    if request.method != 'POST' or request.client is None or request.client.host not in ('127.0.0.1', '::1'):
        return JSONResponse({'error': 'local POST required'}, status_code=403)
    try:
        body = await request.json()
        if set(body) - {'op', 'request_id', 'mode', 'limit'}:
            raise ValueError('unknown observation arguments')
        op = body.pop('op')
        if op not in ('status', 'begin', 'end'):
            raise ValueError('unknown observation operation')
        if op != 'begin' and body:
            raise ValueError('arguments only supported for begin')
        ranks = await request.app.state.engine_client.collective_rpc(
            'glm53_prefill_observe', timeout=60, args=(op,), kwargs=body)
        return JSONResponse(dict(op=op, ranks=ranks))
    except Exception as exc:
        return JSONResponse({'error': repr(exc)}, status_code=500)
