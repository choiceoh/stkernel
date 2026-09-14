"""Bind mixed FFN work to the profile's actual shared readers and C1 overlap."""
from engine.kernels.dense import DenseLinear


def _resources(reader):
    values = [t for p in reader.packs for t in (p.data, p.scale, p.rowscale)]
    for fp8 in (reader.fp8, getattr(reader, 'decode_fp8', None)):
        if fp8 is not None:
            values.extend(fp8.weight)
    if reader.workspace is not None:
        values.append(reader.workspace)
    return tuple(values)


def _stamp(reader):
    return (reader.rows, reader.cols, reader.decode_precision, reader.observer,
            tuple(reader.decode_input_rows),
            tuple((p.rows, p.cols) for p in reader.packs),
            # The native packers produce inference tensors, whose immutable
            # storage has no version counter. Identity is still pinned; normal
            # arena tensors additionally detect tracked in-place mutations.
            tuple((id(t), t.data_ptr(), None if t.is_inference() else t._version)
                  for t in _resources(reader)),
            tuple((id(p), p.observer) if p is not None else None
                  for p in (reader.fp8, getattr(reader, 'decode_fp8', None))))


class BoundMixedShared:
    def __init__(self, net, layer):
        self.net, self.layer = net, layer
        self.names = (f'L{layer}.moe.sh_gate_up', f'L{layer}.moe.sh_down')
        self.up, self.down = (net.dense.get(n) for n in self.names)
        if (not isinstance(self.up, DenseLinear) or not isinstance(self.down, DenseLinear)
                or (self.up.rows, self.up.cols, self.down.rows, self.down.cols) != (1024, 4096, 4096, 512)
                or net.F.swiglu_limit != 10. or self.up.fp8 is None or self.down.fp8 is None):
            raise ValueError('mixed shared work requires the bound TP4 dense readers')
        self.overlap, self.fused = net.shared_overlap, net.shared_mlp.get(layer)
        self.side_stream = None if self.overlap is None else self.overlap.stream
        if self.overlap is not None and (self.fused is None or self.fused.gate_up is not self.up
                or self.fused.down is not self.down or self.fused.limit != 10.):
            raise ValueError('mixed C1 overlap must retain its bound shared MLP')
        self.activation, self.decode_width = net._activation, net.F.spec_k + 1
        # Keep actual tensors alive even if a caller subsequently replaces a
        # reader/pack. Validation refuses that replacement before execution.
        self.resources = (*_resources(self.up), *_resources(self.down))
        self.stamps = (_stamp(self.up), _stamp(self.down))
        self.validate()

    def validate(self):
        net = self.net
        if (net.dense.get(self.names[0]) is not self.up or net.dense.get(self.names[1]) is not self.down
                or net.shared_overlap is not self.overlap or net.shared_mlp.get(self.layer) is not self.fused
                or net._activation is not self.activation or net.F.spec_k+1 != self.decode_width
                or net.F.swiglu_limit != 10.
                or (_stamp(self.up), _stamp(self.down)) != self.stamps):
            raise RuntimeError('mixed shared reader or pack changed after binding')
        if any(r.observer is not None or r.fp8.observer is not None for r in (self.up, self.down)):
            raise RuntimeError('mixed shared work does not support calibration observers')
        if self.overlap is not None and (self.fused.gate_up is not self.up
                or self.fused.down is not self.down or self.fused.limit != 10.
                or self.overlap.stream is not self.side_stream):
            raise RuntimeError('mixed shared overlap changed after binding')

    def _sequential(self, x):
        gate, value = self.up(x).chunk(2, -1)
        return self.down(self.activation(gate, value, 10.))

    def decode(self, x, routed):
        self.validate()
        if self.overlap is not None and len(x) <= self.decode_width:
            # routed is only the prepared expert producer/body/reducer. No
            # prefill shared GEMM can touch the side stream's native scratch.
            return self.overlap(self.fused, x, routed)
        return routed() + self._sequential(x)

    def prefill(self, x):
        self.validate()
        return self._sequential(x)

    def prefill_during(self, x, routed):
        """Join shared prefill around one explicit, noninterleaved cold drain.

        The existing helper joins on every failure path. No next decode/dense
        call may start inside routed: it only submits this owner's cold work.
        """
        self.validate()
        if self.overlap is None:
            raise ValueError('shared prefill overlap requires the bound side stream')
        def dispatch(consume):
            routed()
            return consume(None)
        return self.overlap(self._sequential, x, dispatch, finish=lambda _, shared: shared)
