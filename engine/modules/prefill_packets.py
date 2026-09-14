"""The fixed TP4 FFN packet ABI; geometry is inspectable without CUDA.

One invocation owns four rank-ordered FP8-v3 packets. Consumers reconstruct
the transport's BF16 values in registers; padded rows never become routes.
All three readers use the producer's stream and finish before this owner
can be reused. There is no cross-invocation storage cache or raw-address key.
"""
from dataclasses import dataclass


def ffn_packet_rows(rows):
    return type(rows) is int and 8192 < rows <= 32768


@dataclass(frozen=True)
class PacketGeometry:
    rows: int
    local_rows: int
    hidden: int = 4096
    world: int = 4
    block: int = 2048
    routed: bool = False

    def __post_init__(self):
        if (type(self.rows) is not int or type(self.local_rows) is not int
                or self.local_rows < 32 or not 0 < self.rows <= self.local_rows * 4
                or self.local_rows != (self.rows + 3) // 4
                or (self.hidden, self.world, self.block) != (4096, 4, 2048)
                or type(self.routed) is not bool):
            raise ValueError('FFN packets require exact rank-ordered TP4/H4096 transport geometry')

    @property
    def padded_rows(self):
        return self.world * self.local_rows

    @property
    def local_elements(self):
        return self.local_rows * self.hidden

    @property
    def activation_bytes(self):
        return ((self.local_elements + 4*(self.local_elements//self.block) + 127)//128)*128

    @property
    def route_weights_offset(self):
        return self.activation_bytes + self.local_rows*8*2

    @property
    def stride(self):
        # v2 appends lossless top-8 uint16 IDs and FP32 weights to each rank's
        # v1 activation. Existing value/scale offsets remain unchanged.
        size = self.activation_bytes + (self.local_rows*48 if self.routed else 0)
        return ((size+127)//128)*128

    @property
    def nbytes(self):
        return self.world * self.stride

    def workspace(self):
        """Declared intermediates, not measured traffic or an allocator peak."""
        result = dict(received_bytes=self.nbytes,
                    source_payload_bytes=self.stride,
                    replaced_bf16_bytes=self.rows*self.hidden*2,
                    shared_q_scale_bytes=self.rows*(self.hidden + 4*(self.hidden//128)),
                    expert_shared_stage_bytes=4*self.hidden*2)
        if self.routed:
            result.update(sender_roundtrip_bytes=self.local_elements*2,
                          route_metadata_bytes=self.world*(self.stride-self.activation_bytes))
        return result


@dataclass(frozen=True)
class PacketBatch:
    received: object
    geometry: PacketGeometry

    def __post_init__(self):
        import torch
        x = self.received
        if (not isinstance(self.geometry, PacketGeometry) or not isinstance(x, torch.Tensor)
                or not x.is_cuda or x.ndim != 1 or x.dtype != torch.uint8
                or not x.is_contiguous() or x.numel() != self.geometry.nbytes):
            raise ValueError('FFN packet storage must be contiguous CUDA bytes of the declared length')
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('FFN packet consumption is eager-only')


def agreed_layers(comm, layers, supported):
    """One control-plane vote per eligible prefill, before any FFN exchange.

    Availability includes calibration observers. A missing reader on one
    rank selects the ordinary FFN on all ranks. Data collectives stay at one
    full all-gather per FFN. Decode and short-prefill callers never enter here.
    """
    from engine.base.tripwire import Tripwire
    layers = tuple(layers)
    if len(layers) > 64 or len(set(layers)) != len(layers):
        raise ValueError('packet FFN plan requires distinct layers fitting the control vote')
    wire = Tripwire.of(comm)
    # A slot per model layer keeps the vote independent of a rank's supported
    # subset. The caller's model/layer list is already part of the boot contract.
    votes = wire.vote('prefill:ffn-packets-v2-routed', [int(L in supported) for L in layers])
    return frozenset(L for L, yes in zip(layers, votes) if yes == wire.world)
