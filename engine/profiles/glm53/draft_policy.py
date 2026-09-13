"""DFlash serving defaults and independently reproducible baseline arms."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DraftPolicy:
    fc_precision: str = 'w4'
    fc_calibration: str = 'shared'
    diagnostics: bool = False

    def __post_init__(self):
        if self.fc_precision not in ('w4', 'fp8'):
            raise ValueError('draft FC precision must be w4 or fp8')
        if self.fc_calibration not in ('shared', 'collect', 'decode', 'auto'):
            raise ValueError('draft FC calibration must be shared, collect, decode or auto')
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


# Keep DraftPolicy() as the explicit no-experiment baseline used by stock/local
# probes. Both serving declarations take their defaults from this one recipe.
SERVING_POLICY = DraftPolicy('fp8', 'auto', True)


def resolve_calibration(policy, store, name, cols, comm):
    """Resolve auto once, before arena sizing: all ranks consume, or all collect.

    Missing statistics bootstrap through the existing bounded collector. An
    invalid file is an error on every rank, including ranks whose own file is
    valid. Explicit decode arms still require completed statistics everywhere.
    The preparation rendezvous absorbs file-validation skew before the short
    host-control collective; no CUDA vote or serving-step work is added.
    """
    if policy.fc_calibration not in ('auto', 'decode'):
        return policy
    ready, error = False, None
    try:
        if store is None:
            raise ValueError('decode FC calibration requires a pack store')
        if policy.fc_calibration == 'decode' or store.calibration_path(decode_name(name)).is_file():
            require_decode_calibration(store, name, cols)
            ready = True
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    comm.wait_prepared('draft-calibration')
    reports = comm.gather_objects({'ready': ready, 'error': error})
    errors = [f'rank {rank}: {report["error"]}' for rank, report in enumerate(reports) if report['error']]
    if errors:
        raise ValueError('TP decode FC calibration failed: ' + '; '.join(errors))
    if policy.fc_calibration == 'decode' and not all(report['ready'] for report in reports):
        raise ValueError('explicit decode FC calibration requires completed statistics on every rank')
    mode = 'decode' if all(report['ready'] for report in reports) else 'collect'
    return replace(policy, fc_calibration=mode)


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
