"""GLM's serving-input GPTQ collection, at Qwen's target projection boundaries.

No norm fold: Qwen uses unit-offset norms. MTP stays at its declared precision; its short, speculative rows do not
enter target calibration. Only real prefill rows collect, and only after boot's warmup and graph capture.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import torch

from engine.base.config import Config, Knob
from engine.kernels.dense.calibration import BUDGET_BYTES, Calibration, ROWS_FLOOR, ROWS_TARGET
from engine.profiles.qwen38.net import HEAD_NAME


# D11: temporary size-comparison axis. The campaign must remove this knob when
# choosing a measured default; changing it never appends to an existing blob.
ROWS_EXPERIMENT = Knob("qwen38_calibration_rows", ROWS_TARGET, date(2026, 9, 26),
                       "qwen38_gptq_20260919: 131K/240K/330K on separate validation inputs",
                       "remove override; preserve existing packs and the shared 131072 default", int)


def rows_target(value=None, *, today=None):
    config = Config([], [ROWS_EXPERIMENT], env={}, today=today)
    value = config[ROWS_EXPERIMENT.name] if value is None else value
    if type(value) is not int or not 0 < value <= 1 << 24:
        raise ValueError("calibration rows must be a positive integer no greater than 2**24")
    return value


def identity(metadata, files, config, *, hc_fp8: bool) -> str:
    """Immutable export revision/config plus file fingerprints and the input arithmetic version.

    File size/mtime changes conservatively invalidate the sums, including a replaced PLE table. When present the
    preshard manifest's content hashes are included too; this is not a fresh hash of every multi-GB weight file.
    """
    sources = []
    manifests = {}
    for file in sorted({Path(p) for p in files}):
        stat = file.stat()
        sources.append((file.name, stat.st_size, stat.st_mtime_ns))
        manifest = file.parent / "preshard-manifest.json"
        if manifest.is_file():
            manifests[str(manifest)] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    payload = dict(version="qwen38-fp32-router-w8a16-moe-fp32-as2-v2", metadata=metadata, sources=sources,
                   manifests=sorted(manifests.values()), config=config, hc_fp8=hc_fp8)
    return "qwen38:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def plan(net, specs, store, *, budget=BUDGET_BYTES):
    """(accepted projection plan, admitted bytes, deferred names). Match the lane's padded input domain exactly."""
    from engine.kernels.dense import padded_columns
    shapes = {s.name: s.shape for s in specs}
    names = {k: v for k, v in net.dense_names(shapes).items() if not k.startswith("mtp.")}
    # The head gets all target prefill rows, before forward selects the last row of each prompt.
    names["head"] = HEAD_NAME
    entries, used, deferred = [], 0, []
    for key, name in names.items():
        cols = shapes[key][1] if key == "head" else padded_columns(shapes[key][1])
        missing = store.missing_calibration(name, cols)
        if not missing:
            continue
        need = Calibration.nbytes(missing)
        if used + need > budget:
            deferred.append(name)
            continue
        entries.append((key, name, missing))
        used += need
    return entries, used, deferred


def attach(net, entries, arena, *, max_decode_rows, budget=BUDGET_BYTES, row_target=ROWS_TARGET):
    """Attach disarmed observers before warmup. Large captured batches are excluded by the full graph row ceiling."""
    if not entries:
        return None
    c = Calibration(net.p["head"].device, budget, arena=arena, max_decode_rows=max_decode_rows,
                    row_target=row_target)
    for key, name, missing in entries:
        # Head calls see only the final prompt row (or speculative verification rows). Observe the closing mixer
        # instead: all its real prefill rows, even when forward(last_hidden_only=True) later selects just one.
        layer = SimpleNamespace(input_dtype=torch.bfloat16) if key == "head" else net.dense[key]
        if not c.attach(name, layer, missing, small_rows=False):
            raise RuntimeError(f"calibration admission disagrees with attachment: {name}")
        if key == "head":
            net.head_observer = layer.observer
    return c


class Lifecycle:
    """The shared server's calibration control and after-step hooks, with one provenance stamp on every save path."""
    calibration = None
    calibration_root = None
    calibration_weights_id = None

    def housekeeping(self, steps):
        c = self.calibration
        if c is not None and c.filed is None and steps % 256 == 0 and c.complete():
            self.file_calibration()

    def file_calibration(self, root=None):
        c = self.calibration
        if c is None:
            return None
        if c.filed is not None and root is None:
            return c.filed
        if c.progress() < ROWS_FLOOR:
            return None
        written = c.save(root or self.calibration_root, self.composition.net.comm.rank,
                         weights_id=self.calibration_weights_id)
        print(f"  Qwen calibration: filed {len(written)} blobs ({c.status()}); next boot packs GPTQ", flush=True)
        return written
