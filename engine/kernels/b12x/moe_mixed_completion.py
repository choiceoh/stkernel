"""Eager component: decode first, bounded cold work, then complete prefill.

All eight routes and either reference or bound shared readers are required.
MixedLayerScheduler owns TP agreement, output reductions and cancellation;
this component only submits work on its original eager stream.
Output events are recorded after writes, and views are borrowed until begin.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _producer():
    from .moe_cold_frontend import compile_cold_producer
    return compile_cold_producer()


@lru_cache(maxsize=1)
def _padding():
    from .moe_cold_frontend import compile_cold_padding
    return compile_cold_padding()


def shared_ffn(x, weights):
    """Same BF16 rounding/clamp boundary in both component arms."""
    from engine.kernels.glm_pointwise import swiglu_clamped
    up, down = weights
    gate, value = torch.nn.functional.linear(x, up).chunk(2, -1)
    return torch.nn.functional.linear(swiglu_clamped(gate, value, 10.), down)


class PreparedMixedCompletion:
    def __init__(self, decode, prefill, decode_ids, prefill_ids,
                 decode_routes, prefill_routes, *, weights, input_scale, down_scale,
                 identity, shared_up=None, shared_down=None, shared_execution=None,
                 hot_route_quota=128, cold_task_quota=48, overlap_shared=False):
        if type(overlap_shared) is not bool:
            raise TypeError('shared overlap experiment selector must be bool')
        if overlap_shared and (shared_execution is None or shared_execution.overlap is None):
            raise ValueError('cold overlap requires a bound shared side stream')
        self.overlap_shared = overlap_shared
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('mixed completion preparation requires eager execution')
        if not 8192 < prefill.shape[0] <= 32768:
            raise ValueError('cold completion requires long prefill in 8193..32768')
        if shared_execution is not None:
            if shared_up is not None or shared_down is not None:
                raise ValueError('choose either the reference shared weights or the bound shared execution')
            shared_execution.validate()
        for value, shape in (() if shared_execution is not None else
                             ((shared_up, (1024, 4096)), (shared_down, (4096, 512)))):
            if (value is None or value.shape != shape or value.dtype != torch.bfloat16 or value.device != decode.device
                    or not value.is_contiguous() or not bool(torch.isfinite(value).all())):
                raise ValueError('shared expert requires finite contiguous BF16 TP4 weights')
        from .moe_mixed import PreparedMixedExperts
        from . import moe_dispatch as md
        self.hot = PreparedMixedExperts(decode, prefill, decode_ids, prefill_ids,
            decode_routes, prefill_routes, weights=weights, input_scale=input_scale,
            down_scale=down_scale, identity=identity, hot_route_quota=hot_route_quota,
            cold_task_quota=cold_task_quota)
        self.plan = self.hot.plan
        self.cold = self.hot.cold
        self.stream = self.hot.stream
        self._shared_execution = shared_execution
        self._shared_weights = (shared_up, shared_down) if shared_execution is None else ()
        device = decode.device
        metadata = self.hot.metadata
        self.sources = metadata['cold_rows'].view(-1, 8)
        self.workspace = md.allocate_sm120_dynamic_workspace(state_E=288, weight_E=288,
            routed_rows=len(self.cold.sources), k=4096, n=512, num_topk=8,
            device=device, activation='swigluoai_uninterleave', tile_m=128)
        ws = self.workspace
        if (self.cold.physical_rows > ws.max_rows
                or len(self.cold.task_expert) > ws.task_capacity):
            raise RuntimeError('prepared cold layout exceeds allocated task storage')
        for target, name in ((ws.row_counts, 'cold_counts'), (ws.expert_tile_base, 'cold_bases'),
                             (ws.task_expert, 'cold_tasks'), (ws.task_valid_rows, 'cold_valid')):
            values = metadata[name]
            target[:len(values)].copy_(values)
        # All live rows are overwritten at begin. Clear only reachable tile
        # padding, preserving deterministic TMA/MMA reads without clearing the
        # entire ~600 MiB input plane at 32K.
        _padding()(ws.task_expert[:len(self.cold.task_expert)],
                   ws.task_valid_rows[:len(self.cold.task_expert)], ws.packed_a_flat, ws.scale_flat)
        self.cold_output = torch.empty_like(prefill)
        self._accumulator = self.cold_output
        self._window_args = (ws.task_head, ws.task_tail)
        self._decode_source = decode
        self.producer = _producer()
        self.compiled, _ = md._get_dynamic_kernel(288, len(self.plan.prefill), 4096, 512, 8, ws.max_rows,
            activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
            tile_m=128, tiled=True, reform_sf_pack=True, _prepared_prefill=True)
        sf1, sf2 = md._scale_runtime_addresses(weights, direct_sf6=True)
        self._producer_args = (prefill, prefill_routes, self.sources, prefill_ids, input_scale,
            ws.packed_a_flat, ws.scale_flat, ws.token_map, ws.token_weights)
        self._compute_args = (prefill.data_ptr(), prefill_ids.data_ptr(), prefill_routes.data_ptr(),
            ws.packed_a_view.data_ptr(), ws.packed_input_scale.data_ptr(), ws.packed_a_flat.data_ptr(),
            ws.scale_flat.data_ptr(), ws.barrier_count, ws.barrier_epoch, ws.pair_head,
            ws.task_head, ws.task_tail, ws.task_expert.data_ptr(), ws.task_valid_rows.data_ptr(),
            weights.w13_fp4, sf1, weights.down_fp4, sf2, ws.row_counts,
            ws.expert_write_rows, ws.expert_tile_base, input_scale, weights.w1_alpha,
            weights.w2_alpha, down_scale, self.cold_output.data_ptr(), ws.token_map.data_ptr(),
            ws.token_weights.data_ptr(), weights.sfb1_packed, weights.sfb2_packed,
            len(self.plan.prefill), ws.max_rows, ws.physical_tiles_capacity * 128, ws.task_capacity)
        # Sparse completion reduction visits only the <=128 moved routes.
        self._hot_rows = metadata['hot_rows'].long()
        self._hot_dest = metadata['hot_dest'].long()
        self._hot_sum = torch.empty((len(self._hot_rows), 4096), dtype=torch.float32, device=device)
        self._owned = (*self._shared_weights, self.sources, ws.row_counts, ws.expert_tile_base,
                       ws.task_expert, ws.task_valid_rows, self._hot_rows, self._hot_dest)
        self._versions = tuple(t._version for t in self._owned)
        self.decode_ready = torch.cuda.Event()
        self.prefill_ready = torch.cuda.Event()
        self.state, self.next_window = 'new', 0
        self._decode_output = self._prefill_output = None

    def validate(self, identity):
        self.hot.validate(identity)
        if self._shared_execution is not None:
            self._shared_execution.validate()
        if tuple(t._version for t in self._owned) != self._versions:
            raise RuntimeError('prepared cold or shared ownership changed')

    def begin(self, identity):
        self.validate(identity)
        if self.state not in ('new', 'complete'):
            raise RuntimeError('previous prefill completion is still owed')
        self.state, self.next_window = 'running', 0
        self._prefill_output = None
        self._prefill_shared = None
        def routed():
            partials = self.hot.run(identity)
            return partials[:self.plan.decode_routes].reshape(len(self.plan.decode), 32, 4096).sum(1).bfloat16()
        self._decode_output = (routed() + shared_ffn(self._decode_source, self._shared_weights)
            if self._shared_execution is None else self._shared_execution.decode(self._decode_source, routed))
        self.decode_ready.record(self.stream)
        self.state = 'decode'
        return self._decode_output

    def advance(self, identity):
        """Enqueue at most cold_task_quota MMA tiles; preserve previous sums.

        The first call also packs all cold routes. This is a work-count bound,
        not a bound on launch latency or a preemptible serving quantum.
        """
        return self._dispatch_cold(identity, min(self.next_window + 1, len(self.cold.windows)))

    def drain(self, identity):
        """Explicitly finish remaining cold work with one launch.

        Use when no intervening decode dispatch is required. This may consume
        the entire invocation; it does not promise the advance() work quota.
        """
        if self.overlap_shared:
            self.validate(identity)
            if self.state not in ('decode', 'cold'):
                raise RuntimeError('cold drain requires an unfinished invocation')
            try:
                self._prefill_shared = self._shared_execution.prefill_during(self._producer_args[0],
                    lambda: self._dispatch_cold(identity, len(self.cold.windows)))
            except BaseException:
                # SharedOverlap has joined all submitted work, but no output
                # can be retried or published after a failed side branch.
                self.state = 'dispatching'
                raise
            return True
        return self._dispatch_cold(identity, len(self.cold.windows))

    def _dispatch_cold(self, identity, stop_window):
        self.validate(identity)
        if self.state not in ('decode', 'cold'):
            raise RuntimeError('cold work requires a decode result and unfinished routes')
        first = self.state == 'decode'
        self.state = 'dispatching'
        if first:
            self._accumulator.zero_()
            if self.cold.sources:
                self.producer(*self._producer_args)
        if self.next_window < len(self.cold.windows):
            start = self.cold.windows[self.next_window][0]
            stop = self.cold.windows[stop_window - 1][1]
            # The private inherited body claims [head, tail); it never resets
            # these counters or output. Stores and launches share one stream.
            self._window_args[0].fill_(start)
            self._window_args[1].fill_(stop)
            self.compiled(*self._compute_args)
            self.next_window = stop_window
        self.state = 'routed' if self.next_window == len(self.cold.windows) else 'cold'
        return self.state == 'routed'

    def finish(self, identity):
        self.validate(identity)
        if self.state not in ('decode', 'cold', 'routed'):
            raise RuntimeError('prefill completion requires exactly one begun invocation')
        while self.state != 'routed':
            self.advance(identity)
        self.state = 'finishing'
        if self.plan.hot_routes:
            self._hot_sum.zero_()
            partials = self.hot.partials[:len(self.plan.sources)*4].view(-1, 4, 4096)
            self._hot_sum.index_add_(0, self._hot_dest, partials[self.plan.decode_routes:].sum(1))
            self._accumulator[self._hot_rows] = (self._accumulator[self._hot_rows].float() + self._hot_sum).bfloat16()
        shared = self._prefill_shared
        if shared is None:
            shared = (shared_ffn(self._producer_args[0], self._shared_weights) if self._shared_execution is None
                      else self._shared_execution.prefill(self._producer_args[0]))
        self._prefill_output = self._accumulator + shared
        self.prefill_ready.record(self.stream)
        self.state = 'complete'
        return self._prefill_output

    def prefill_result(self, identity):
        self.validate(identity)
        if self.state != 'complete':
            raise RuntimeError('all eight routed experts and the shared expert must finish first')
        return self._prefill_output, self.prefill_ready

    def reader_fence(self):
        """Fence all submitted readers, including partial work after failure.

        Do not validate source versions here: a changed source still needs
        its outstanding readers drained. SharedOverlap joins on all exits.
        """
        if torch.cuda.is_current_stream_capturing() or torch.cuda.current_stream(self._decode_source.device) != self.stream:
            raise RuntimeError('mixed reader fences require the original eager stream')
        event = torch.cuda.Event()
        event.record(self.stream)
        return event
