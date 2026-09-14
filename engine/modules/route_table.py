"""Immutable little-endian int32 route rows, without one Python object per route.

The bytes own their storage: neither the input array nor a returned NumPy view
can mutate an admitted descriptor. Tensor uploads must copy this host storage.
"""
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteTable(Sequence):
    data: bytes
    columns: int

    def __post_init__(self):
        if (type(self.data) is not bytes or type(self.columns) is not int
                or self.columns < 1 or len(self.data) % (4 * self.columns)):
            raise ValueError('route table requires whole immutable int32 rows')

    @classmethod
    def pack(cls, rows):
        import numpy as np
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.dtype.kind != 'i' or rows.dtype.itemsize != 4:
            raise ValueError('route table requires a two-dimensional int32 array')
        return cls(rows.astype('<i4', copy=False).tobytes(), rows.shape[1])

    def array(self):
        import numpy as np
        return np.frombuffer(self.data, dtype='<i4').reshape(-1, self.columns)

    def __len__(self):
        return len(self.data) // (4 * self.columns)

    def __getitem__(self, key):
        rows = self.array()[key]
        if isinstance(key, slice):
            return tuple(map(tuple, rows.tolist()))
        return tuple(rows.tolist())
