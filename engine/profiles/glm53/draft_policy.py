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
    if (blob.get('input_scope') != 'committed_decode_v1' or int(blob.get('ntok', 0)) < ROWS_FLOOR
            or store.missing_calibration(key, cols)):
        raise ValueError(f'decode FC calibration is incomplete or has the wrong input scope: {path}')
    return key
