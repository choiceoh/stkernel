"""Independent, boot-time DFlash acceptance experiments."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftPolicy:
    fc_precision: str = 'w4'
    fc_calibration: str = 'shared'
    diagnostics: bool = False

    def __post_init__(self):
        if self.fc_precision not in ('w4', 'fp8'):
            raise ValueError('draft FC precision must be w4 or fp8')
        if self.fc_calibration not in ('shared', 'collect', 'decode'):
            raise ValueError('draft FC calibration must be shared, collect or decode')
        if type(self.diagnostics) is not bool:
            raise ValueError('draft diagnostics must be a boolean')

    @property
    def separate_decode_fp8(self):
        return self.fc_precision == 'fp8' and self.fc_calibration == 'decode'

    @property
    def active(self):
        return self.fc_precision != 'w4' or self.fc_calibration != 'shared' or self.diagnostics

    def label(self):
        return f'fc={self.fc_precision},calibration={self.fc_calibration},diagnostics={int(self.diagnostics)}'


def decode_name(name):
    return name + '.committed-decode-v1'


def require_decode_calibration(store, name, cols):
    """A missing or foreign collection must never silently become an RTN arm."""
    import torch
    from engine.kernels.dense.calibration import ROWS_FLOOR
    key = decode_name(name)
    path = store.calibration_path(key)
    if not path.is_file():
        raise ValueError(f'decode FC calibration missing: collect committed decode rows first: {path}')
    blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    store.read_files.add(path)  # return validation's clean file pages before arena admission
    if not isinstance(blob, dict):
        raise ValueError(f'decode FC calibration is not a statistics blob: {path}')
    ntok = blob.get('ntok')
    if (blob.get('input_scope') != 'committed_decode_v1' or blob.get('name') != key
            or type(ntok) is not int or ntok < ROWS_FLOOR):
        raise ValueError(f'decode FC calibration is incomplete or has the wrong input scope: {path}')
    hessian, peaks = blob.get('H'), blob.get('amax')
    if (not isinstance(hessian, torch.Tensor) or hessian.shape != (cols, cols)
            or hessian.dtype != torch.float32 or hessian.layout != torch.strided
            or not isinstance(peaks, torch.Tensor) or peaks.shape != (cols,)
            or peaks.dtype != torch.float32 or peaks.layout != torch.strided):
        raise ValueError(f'decode FC calibration statistics have incompatible shapes or dtypes: {path}')
    # This runs before the arena admission. Keep validation scratch bounded
    # even for FC's 20480-column Hessian; a full isfinite mask is 400 MiB.
    finite = all(bool(torch.isfinite(part).all()) for part in hessian.split(128))
    diagonal = hessian.diagonal()
    if (not finite or not bool(torch.isfinite(peaks).all()) or bool((peaks < 0).any())
            or bool((diagonal < 0).any()) or not bool((diagonal > 0).any())
            or not bool((peaks > 0).any())):
        raise ValueError(f'decode FC calibration statistics are nonfinite, negative or empty: {path}')
    return key
