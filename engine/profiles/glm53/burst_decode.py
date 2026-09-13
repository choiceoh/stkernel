"""Serve deterministic bounded decode bursts through the existing async chain.

One burst may be in flight. The entire write horizon is reserved first; all
iteration outcomes are read back together. A prefix crossing stops the burst
before another iteration can overwrite its staged recurrent checkpoint.
"""
from __future__ import annotations

import torch
from contextlib import contextmanager

from engine.base.graphs import frozen_gc
from engine.base.graph_labels import capture as label_capture
from engine.profiles.glm53.bounded_loop import agree_stop, stop_at_boundary
from engine.profiles.glm53.net import Segment, Step
from engine.profiles.glm53.pipeline import AsyncDecode


class DeviceStages:
    NAMES = ("forward", "sample", "commit", "boundaries", "observe", "propose")

    def __init__(self, times, count):
        self.times, self.count = times, count

    @contextmanager
    def mark(self, name):
        column = 2 * self.NAMES.index(name)
        if self.times.is_cuda:
            from engine.kernels.bounded_graph import build
            build().stamp(self.times, self.count, column)
        yield
        if self.times.is_cuda:
            build().stamp(self.times, self.count, column + 1)


class BurstPending:
    def __init__(self, pipeline, seqs, shape, event, staged):
        self.pipeline, self.seqs, self.shape = pipeline, tuple(seqs), shape
        self.event, self.staged = event, staged
        self.iteration_seconds = None

    def resolve(self):
        return self.pipeline.resolve_burst(self)


class BurstDecode(AsyncDecode):
    END_IDS = 8

    def __init__(self, engine, iterations):
        super().__init__(engine)
        if type(iterations) is not int or iterations not in (2, 4):
            raise ValueError("served bursts require 2 or 4 iterations")
        if iterations * self.t >= engine.F.block:
            raise ValueError("a burst must fit within one prefix block")
        self.iterations = iterations
        self.states, self.logs, self.controls, self.loops = {}, {}, {}, {}
        self.pool = None
        pin = engine.caches.device.type == "cuda"
        n = engine.caches.pool.max_seqs
        self.readback = dict(tokens=torch.empty(4, n, self.t, dtype=torch.int64, pin_memory=pin),
                             count=torch.empty(4, n, dtype=torch.int64, pin_memory=pin),
                             done=torch.empty(4, n, dtype=torch.bool, pin_memory=pin),
                             accepted=torch.empty(4, n, dtype=torch.int64, pin_memory=pin),
                             before=torch.empty(4, n, dtype=torch.int64, pin_memory=pin),
                             iterations=torch.empty(1, dtype=torch.int64, pin_memory=pin),
                             timings=torch.empty(4, 2, dtype=torch.int64, pin_memory=pin),
                             stages=torch.empty(4, 2*len(DeviceStages.NAMES), dtype=torch.int64, pin_memory=pin))
        try:
            self._capture()
        except BaseException:
            self.close()
            raise

    def reserve_steps(self, seq):
        e = self.e
        ends = e.ends.get(seq, e.eos)
        return self.iterations if e.limits[seq][1] <= 0 and len(ends) <= self.END_IDS else 1

    def ready_for(self, seqs, slots=None):
        # A second burst could overwrite its readback/stage. Resolve first;
        # cancellation, prefill and row admission then use the ordinary runner.
        return not any(isinstance(p, BurstPending) for p in self.pending) and super().ready_for(seqs, slots)

    def _state(self, n):
        e, dev, t = self.e, self.e.caches.device, self.t
        b = {k: torch.zeros(n, dtype=torch.int64, device=dev) for k in
             ("seqs", "real_slot", "slot", "ctx", "generated", "limit", "anchor")}
        b.update(ends=torch.full((n, self.END_IDS), -1, dtype=torch.int64, device=dev),
                 temps=torch.zeros(n, device=dev), top_k=torch.zeros(n, dtype=torch.int32, device=dev),
                 top_p=torch.ones(n, device=dev), alive=torch.ones(n, dtype=torch.bool, device=dev),
                 ids=torch.zeros(n*t, dtype=torch.int64, device=dev),
                 drafts=torch.zeros(n, t-1, dtype=torch.int64, device=dev),
                 stochastic=False, qcand=None, qprob=None)
        b["seqs"].copy_(torch.arange(n, device=dev))
        b["real_slot"].copy_(b["seqs"] + 1)
        b["slot"].copy_(b["real_slot"])
        b["limit"].fill_(128)
        logs = {k: torch.empty(4, n, dtype=torch.int64, device=dev)
                for k in ("count", "accepted", "before")}
        logs["tokens"] = torch.empty(4, n, t, dtype=torch.int64, device=dev)
        logs["done"] = torch.empty(4, n, dtype=torch.bool, device=dev)
        controls = dict(count=torch.zeros(1, dtype=torch.int64, device=dev),
                        stop=torch.zeros(1, dtype=torch.int64, device=dev),
                        interrupt=torch.zeros(1, dtype=torch.int64, device=dev),
                        reserved=torch.full((n,), e.max_context, dtype=torch.int64, device=dev),
                        stages=torch.full((4, 2*len(DeviceStages.NAMES)), -1, dtype=torch.int64, device=dev))
        return b, logs, controls

    def _body(self, shape):
        n = shape[0]
        b, log, control = self.states[n], self.logs[n], self.controls[n]
        result = self.iterate(shape, b, stage_clock=DeviceStages(control["stages"], control["count"]))
        for key, value in result.items():
            log[key].index_copy_(0, control["count"], value.unsqueeze(0))
        vote = stop_at_boundary(result["before"], b["ctx"], b["alive"], control["reserved"], shape[2],
                                control["interrupt"], step_tokens=self.t, block=self.e.F.block)
        control["stop"].copy_(agree_stop(self.e.net.comm, vote))

    def _capture(self):
        from engine.kernels.bounded_graph import BoundedGraph, build
        e = self.e
        if e.caches.device.type != "cuda" or e.net.comm.world_size != 4 or e.net.comm.transport is None:
            raise ValueError("served bounded decode requires CUDA GB10 TP4 one-shot transport")
        build()  # pay native compilation before recording any graph
        self.pool = torch.cuda.graph_pool_handle()
        side, recording = torch.cuda.Stream(), torch.cuda.Stream()
        try:
            with frozen_gc():
                for shape in e.decode_graphs.graphs.graphs:
                    n = shape[0]
                    if n not in self.states:
                        self.states[n], self.logs[n], self.controls[n] = self._state(n)
                    b, controls = self.states[n], self.controls[n]
                    b["ctx"].zero_(); b["generated"].zero_(); b["alive"].fill_(True)
                    controls["count"].zero_()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        self._body(shape)
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    captured = torch.cuda.CUDAGraph(keep_graph=True)
                    try:
                        with torch.cuda.stream(recording):
                            captured.capture_begin(self.pool, capture_error_mode="global")
                            try:
                                with label_capture(recording, f"bounded/{shape}/{self.iterations}"):
                                    self._body(shape)
                            finally:
                                captured.capture_end()
                        loop = BoundedGraph(captured, controls["count"], controls["stop"], self.iterations,
                                            owners=(b, self.logs[n], controls, e.decode_graphs,
                                                    e.sampling_graphs, e.drafter.decode_graphs))
                    except BaseException:
                        captured.reset()
                        raise
                    self.loops[shape] = loop
                    if e.memory is not None:
                        e.memory.checkpoint(f"bounded-decode/{shape}/{self.iterations}")
                torch.cuda.synchronize()
        finally:
            e.caches.reset()  # real warmup writes happened only before admission

    def launch(self, seqs, slots):
        seqs, slots = list(seqs), list(slots)
        if not self.ready_for(seqs, slots):
            raise RuntimeError("resolve the current burst before changing or launching rows")
        # Keep stochastic draws and rich sampling on their existing path. No
        # captured RNG offset is repeated and arbitrary stop-id sets still work.
        if self.pending or any(self.reserve_steps(s) == 1 for s in seqs):
            return super().launch(seqs, slots)
        e, n, t = self.e, len(seqs), self.t
        if self.stale:
            self._build(seqs, slots)
        elif (tuple(seqs) != self.batch or self.dirty.intersection(seqs)
              or any(self.slots.get(s) != slot for s, slot in zip(seqs, slots))):
            self._merge(seqs, slots)
        reserved = [min(e.caches.pool.tokens[s], e.max_context) for s in seqs]
        if any(end < e.ctx[s] + t for s, end in zip(seqs, reserved)):
            raise ValueError("first bounded iteration exceeds its KV reservation")
        shape = e.decode_graphs.shape_for(n, max(min(end, e.ctx[s] + t*self.iterations)
                                                 for s, end in zip(seqs, reserved)))
        zeros = self._zeros.get(n)
        if zeros is None:
            zeros = self._zeros[n] = torch.zeros(n*t, dtype=torch.int64, device=e.caches.device)
        host_step = Step(zeros, tuple(Segment(s, slot, e.ctx[s], i*t, t)
                                     for i, (s, slot) in enumerate(zip(seqs, slots))))
        e.caches.prepare(host_step)
        b, controls = self.states[n], self.controls[n]
        for key, target in b.items():
            source = self.buf[key]
            if not isinstance(target, torch.Tensor) or source is target:
                continue
            if key == "ends" and source.shape != target.shape:
                target.fill_(-1)
                target[:, :source.shape[1]].copy_(source)
            else:
                target.copy_(source)
        # Separate the mapping from the captured mapping: shrink/merge may
        # replace the pipeline's tensor entries without changing graph owners.
        self.buf = dict(b)
        controls["reserved"].copy_(self._upload(reserved, torch.int64))
        controls["interrupt"].zero_()  # host cancellation drains this finite burst
        controls["stages"].fill_(-1)
        loop = self.loops[shape]
        loop.replay()
        for key, value in self.logs[n].items():
            self.readback[key][:, :n].copy_(value, non_blocking=True)
        self.readback["iterations"].copy_(controls["count"], non_blocking=True)
        self.readback["timings"].copy_(loop.timings, non_blocking=True)
        self.readback["stages"].copy_(controls["stages"], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        for s in seqs:
            e.inflight[s] = e.inflight.get(s, 0) + self.iterations
        pending = BurstPending(self, seqs, shape, event, self._staged)
        self._staged = []
        self.pending.append(pending)
        return pending

    def resolve_burst(self, pending):
        if not self.pending or self.pending[0] is not pending:
            raise RuntimeError("decode bursts resolve in launch order")
        if pending.event is not None:
            pending.event.synchronize()
        e, host, n = self.e, self.readback, len(pending.seqs)
        iterations = int(host["iterations"][0])
        if not 1 <= iterations <= self.iterations:
            raise RuntimeError("invalid bounded decode iteration count")
        times = host["timings"][:iterations].tolist()
        pending.iteration_seconds = [(end-start)*1e-9 for start, end in times]
        if any(t < 0 for t in pending.iteration_seconds):
            raise RuntimeError("invalid bounded decode device timestamps")
        counts, dones, accepts, tokens, before = (host[k][:iterations, :n].tolist()
            for k in ("count", "done", "accepted", "tokens", "before"))
        stages = host["stages"][:iterations].tolist()
        finished = [False] * n
        pending.iteration_records = []
        for j in range(iterations):
            active = [i for i, seq in enumerate(pending.seqs) if seq in e.tokens]
            if active and not any(counts[j][i] > 0 or dones[j][i] for i in active):
                raise RuntimeError("bounded decode made no progress")
            for i, seq in enumerate(pending.seqs):
                finished[i] |= bool(dones[j][i])
                if seq not in e.tokens:
                    continue
                c = counts[j][i]
                if not 0 <= c <= self.t or e.ctx[seq] != before[j][i]:
                    raise RuntimeError("bounded decode readback lost row/context order")
                if c > 0:
                    e.tokens[seq] += tokens[j][i][:c]
                    e.ctx[seq] += c
                    e.accepted_total += accepts[j][i]
                    e.drafted_total += e.drafter.k
                    boundary = (e.ctx[seq] // e.F.block) * e.F.block
                    if boundary > before[j][i]:
                        e.staged[seq] = boundary
            stage_us = {name: (stages[j][2*k+1]-stages[j][2*k])*1e-3
                        for k, name in enumerate(DeviceStages.NAMES) if stages[j][2*k] >= 0}
            for name, us in stage_us.items():
                if us < 0:
                    raise RuntimeError("invalid bounded stage timestamps")
                self.clock.totals[name] = self.clock.totals.get(name, 0.) + us * 1e-6
            pending.iteration_records.append(dict(positions=before[j], committed=counts[j], accepted=accepts[j],
                                                 stages_us=stage_us))
        self.clock.samples += iterations
        for seq in pending.seqs:
            e.inflight[seq] = max(0, e.inflight.get(seq, 0) - self.iterations)
        e.steps += iterations
        self.pending.pop(0)
        pending.staged = []
        return finished

    def close(self):
        while self.pending:
            self.pending[0].resolve()
        for loop in self.loops.values():
            body = loop.body
            loop.close()
            body.reset()
        self.loops.clear()
        self.states.clear(); self.logs.clear(); self.controls.clear()
        self.buf = None
        self.pool = None
