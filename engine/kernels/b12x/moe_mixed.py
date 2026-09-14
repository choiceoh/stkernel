"""Private M1 hot-route component; deliberately absent from serving dispatch.

This owner computes decode plus selected prefill routes. It does not complete
a prefill token: cold routes and the shared expert remain mandatory work.
The explicit route/part output is a separate reduction candidate, not a claim
of byte-identical served atomic output or of end-to-end speed.
"""
from functools import lru_cache

import torch

from engine.modules.mixed_experts import plan_experts_packed


@lru_cache(maxsize=1)
def _producer():
    from .moe_mixed_frontend import compile_producer
    return compile_producer()


class PreparedMixedExperts:
    CAPACITY = 384                 # decode <= 256 routes, hot <= 128

    def __init__(self, decode, prefill, decode_ids, prefill_ids,
                 decode_routes, prefill_routes, *, weights, input_scale, down_scale,
                 identity, hot_route_quota=128):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('mixed expert preparation is eager component work')
        device = decode.device
        if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (12, 1):
            raise ValueError('mixed experts require a GB10 CUDA source')
        for x, ids, routes, maximum in ((decode, decode_ids, decode_routes, 32),
                                       (prefill, prefill_ids, prefill_routes, 32768)):
            if (x.ndim != 2 or x.shape[1] != 4096 or x.dtype != torch.bfloat16
                    or not 1 <= x.shape[0] <= maximum
                    or ids.shape != (x.shape[0], 8) or ids.dtype != torch.int32
                    or routes.shape != ids.shape or routes.dtype != torch.float32
                    or any(t.device != device or not t.is_contiguous() for t in (x, ids, routes))):
                raise ValueError('mixed sources require contiguous BF16 H4096 and FP32 top-8 weights')
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(routes).all()):
                raise ValueError('mixed source values must be finite')
        if (not weights.tiled or weights.reform_scales is None or not weights.reform_scales.enabled
                or tuple(weights.w1_storage.shape) != (288, 1024, 2048)
                or tuple(weights.w2_storage.shape) != (288, 4096, 256)):
            raise ValueError('mixed experts require the prepared immutable TP4 SF6 weight owner')
        self.weights = weights
        for scale in (input_scale, down_scale, weights.w1_alpha, weights.w2_alpha):
            if (scale.shape != (288,) or scale.dtype != torch.float32 or scale.device != device
                    or not scale.is_contiguous() or not bool(torch.isfinite(scale).all())
                    or not bool((scale > 0).all())):
                raise ValueError('mixed experts require positive finite per-expert scales')
        weight_tensors = (weights.w1_storage, weights.w2_storage, weights.sfb1_packed, weights.sfb2_packed)
        if any(t is None or t.device != device or not t.is_contiguous() for t in weight_tensors):
            raise ValueError('mixed weight planes must share the source device')
        self.plan = plan_experts_packed(decode_ids.cpu().numpy(), prefill_ids.cpu().numpy(),
                                       identity=identity, hot_route_quota=hot_route_quota)
        from . import moe_dispatch as md
        self.decode, self.prefill = decode, prefill
        self.ids, self.route_weights = decode_ids, decode_routes
        self.prefill_routes = prefill_routes
        self.input_scale, self.down_scale = input_scale, down_scale
        self.sources = torch.tensor(self.plan.sources, dtype=torch.int32, device=device)
        self.workspace = md.allocate_sm120_static_workspace(state_E=288, weight_E=288,
            max_rows=32, k=4096, n=512, num_topk=8, device=device)
        ws = self.workspace
        counts = [self.plan.decode_counts[e] + self.plan.hot_counts[e] for e in self.plan.experts]
        ws.row_counts.copy_(torch.tensor(counts + [0]*(288-len(counts)), dtype=torch.int32, device=device))
        ws.weight_expert_ids[:len(counts)].copy_(torch.tensor(self.plan.experts, dtype=torch.int32, device=device))
        ws.active_expert_count.fill_(len(counts))
        # Padded rows are never scattered. Initialize them once so TMA/MMA do
        # not read uninitialized data; later runs overwrite every live route.
        ws.packed_input.zero_()
        ws.packed_input_scale.zero_()
        config = dict(md._parse_glm53_static_v2('t,r,sf6'), probe_prepared_routes=self.CAPACITY)
        self.compiled, mac = md._get_static_kernel_v2(288, 288, len(self.plan.decode), 4096, 512, 8, 32,
            config=config, activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
        self.producer = _producer()
        self.partials = torch.empty((self.CAPACITY * 4, 4096), dtype=torch.float32, device=device)
        self.stamps = torch.zeros((mac, md._STATIC_V2_STAMP_SLOTS), dtype=torch.int64, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._producer_args = (decode, prefill, decode_routes, prefill_routes,
            self.sources, ws.weight_expert_ids, input_scale, ws.packed_a_flat, ws.scale_flat,
            ws.token_map, ws.token_weights)
        sf1, sf2 = md._scale_runtime_addresses(weights, direct_sf6=True)
        # Freeze the actual tensors/pointers once. Rebinding a mutable weight
        # view object later cannot replace a resource behind this invocation.
        self._compute_args = (decode, decode_ids.reshape(-1), decode_routes.reshape(-1),
            ws.packed_a_view, ws.packed_input_scale.data_ptr(), ws.packed_a_flat, ws.scale_flat,
            ws.barrier_count, ws.barrier_epoch, weights.w13_fp4, sf1, weights.down_fp4, sf2,
            ws.row_counts, ws.active_expert_count, ws.weight_expert_ids, ws.global_to_local_expert,
            input_scale, weights.w1_alpha, weights.w2_alpha, down_scale, self.partials,
            ws.token_map, ws.token_weights, self.stamps, self.counter, weights.sfb1_packed, weights.sfb2_packed)
        self._owned = (decode, prefill, decode_ids, prefill_ids, decode_routes, prefill_routes,
                       input_scale, down_scale, weights.w1_alpha, weights.w2_alpha, *weight_tensors,
                       weights.w13_fp4, weights.down_fp4, self.sources,
                       ws.row_counts, ws.weight_expert_ids, ws.active_expert_count)
        self._versions = tuple(t._version for t in self._owned)
        self.stream = torch.cuda.current_stream(device)
        self.last_reader = torch.cuda.Event()
        self.last_reader.record(self.stream)

    def validate(self, identity):
        if identity != self.plan.identity:
            raise ValueError('stale layer, epoch, slot or source generation')
        if torch.cuda.is_current_stream_capturing() or torch.cuda.current_stream(self.decode.device) != self.stream:
            raise RuntimeError('prepared mixed routes require their original eager stream')
        if tuple(t._version for t in self._owned) != self._versions:
            raise RuntimeError('prepared source or weights changed; prepare a new invocation')

    def run(self, identity):
        self.validate(identity)
        self.producer(*self._producer_args)
        self.compiled(*self._compute_args)
        self.last_reader.record(self.stream)
        # Borrowed only until the next run. Every route owns four BF16-rounded
        # weighted FC2 partials in FP32 storage, including zero-weight routes.
        return self.partials[:len(self.plan.sources)*4].view(-1, 4, 4096)
