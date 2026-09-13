"""K=7 DSA query readers sharing one input pack; existing weight owners stay intact."""
import torch

from . import bound_input_cell, extension


class QueryPair:
    def __init__(self, first, second, *, rows):
        self.rows = tuple(rows)
        if (not self.rows or self.rows != (8, 16, 24, 32)[:len(self.rows)]
                or any(type(r) is not int for r in self.rows)):
            raise ValueError('query pair requires contiguous K=7 capture widths')
        for layer in (first, second):
            if (layer.cols != 1536 or layer.rows not in (4096, 8192)
                    or layer.decode_precision != 'w4' or len(layer.packs) != 1
                    or layer.workspace is not None):
                raise ValueError('query pair requires ordinary single-pack DSA W4 owners')
        if first is second or first.packs[0].data.device != second.packs[0].data.device:
            raise ValueError('query pair requires distinct owners on one device')
        self.layers = first, second
        self.executed = set()

    def __call__(self, x):
        if (x.ndim != 2 or x.shape[0] not in self.rows or x.shape[1] != 1536
                or x.dtype != torch.bfloat16 or x.stride(1) != 1
                or x.stride(0) < 1536 or x.stride(0) % 4 or x.data_ptr() % 8
                or x.device != self.layers[0].packs[0].data.device):
            raise ValueError('query pair input is outside its bound rows/strides/device')
        packs = [layer.packs[0] for layer in self.layers]
        outputs = [torch.empty((len(x), layer.rows), device=x.device, dtype=x.dtype) for layer in self.layers]
        for layer in self.layers:
            if layer.observer is not None:
                layer.observer(x, None)
        extension().run_query_pair(x, [p.data for p in packs], [p.scale for p in packs],
                                   [p.rowscale for p in packs], outputs)
        for layer in self.layers:
            layer.executed |= 1
            if len(x) in layer.decode_input_rows and bound_input_cell(len(x), layer.rows, layer.cols):
                layer.bound_input_executed.add(len(x))
        self.executed.add(len(x))
        return tuple(outputs)
