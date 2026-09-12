"""Self-calibration of the W4 packs: the Gram sums the pack store lacks, summed from what this boot serves.

The store builds a dense weight's W4 pack GPTQ when `<root>/mkcalib/rank<r>/<name>.pt` holds the Hessian of that
weight's input and round-to-nearest when it does not (kernels/dense/store). Those blobs used to come from the vLLM
stack's MK_CALIB dumps. This engine writes its own, by default and without a knob (45차 §23 GPU 판정 6차): a boot
that finds calibration missing sums X^T X of those weights' inputs inside its own serving -- the packs it serves
meanwhile are round-to-nearest, counted as such -- files the blobs once enough rows were seen (or at shutdown), and
the next boot packs GPTQ from them. Every rank sums its own inputs: TP-sharded projections see different columns.

The sums live on the device (carved from the arena, within a fixed budget: what does not fit waits for a later boot,
drafter first) and are added to inside the captured graphs: every contribution is multiplied by a device scalar
that is 0 until `arm()` (warm-ups and captures feed the same paths with junk) and by the row mask the caller hands
over (the pipeline's ghost rows, a masked observation's positions past the committed count -- the 33차 lesson:
padding rows poison a Hessian). A layer whose small-row calls may carry junk without a mask (the target's decode
steps run ghost rows through the null slot) sums only its large-row calls (prefill, every row real) -- the same
tokens' hidden states, which is what the 33차 dumps summed too.
"""
import os
from pathlib import Path

import torch

ROWS_TARGET = 32768        # rows per blob before the sums are filed on their own (33차: 33K tokens)
ROWS_FLOOR = 4096          # fewer than this at shutdown is not filed: a starved Hessian would pack worse than none
BUDGET_BYTES = 2 << 30     # per rank, from the arena; the rest waits for a later boot
GRAM_ROWS = 256           # small calls share one Gram update; bounded staging is part of the same budget


class Calibration:
    def __init__(self, device, budget_bytes: int = BUDGET_BYTES, arena=None, max_decode_rows: int = 32):
        if type(max_decode_rows) is not int or max_decode_rows <= 0:
            raise ValueError('max_decode_rows must be a positive integer')
        self.max_decode_rows = max_decode_rows
        self.device = torch.device(device)
        self.budget, self.used = budget_bytes, 0
        self.arena = arena
        self.H, self.rows, self.tiles = {}, {}, {}          # blob key -> sums, rows, (start, width); layer name -> tiles
        self.staging = {}                                  # key -> (declared-input row buffer, device cursor); CUDA small-row Hessians only
        self.amax, self.unsmooth = {}, {}                   # blob key -> channel peaks [width]; layer name -> the s its input was divided by
        self.deferred = []                                  # (layer name, key) that did not fit this boot's budget
        self.armed = torch.zeros((), dtype=torch.float32, device=self.device)
        self.filed = None                                   # the paths written, once

    @staticmethod
    def _needs(missing):
        """(key, start, width, its Hessian is missing too) of each tile, from the store's Need or a plain triple.
        A tile whose Hessian the store already has is summed for its channel peaks alone -- [K] floats rather than
        [K, K] (kernels/dense/store.Need): on this fleet that is 204 of the 228 packs and the difference between a
        9.96 GiB plan that does not fit a boot's budget and a 1.82 GiB one that does."""
        for tile in missing:
            yield tile[0], tile[1], tile[2], (bool(tile[3]) if len(tile) > 3 else True)

    def attach(self, name: str, layer, missing, small_rows: bool, unsmooth=None) -> bool:
        """Sum `layer`'s input over the tiles in `missing` (store.Need); `small_rows`: its calls of <= max_decode_rows
        rows are real (with the caller's mask) -- False for a layer whose decode rows may be ghosts. `unsmooth` [K]:
        the factor this boot divided the input by (smoothing folded into its norm) -- the sums are filed in the
        unsmoothed domain so every boot derives its own factors from scratch. Returns whether the tiles fit the
        budget (all or nothing per layer)."""
        decode_rows = self.max_decode_rows if small_rows and self.device.type == "cuda" else 0
        input_dtype = getattr(layer, 'input_dtype', torch.float32)
        need = self.nbytes(missing, max_decode_rows=decode_rows, input_dtype=input_dtype)
        if self.used + need > self.budget:
            self.deferred.extend((name, key) for key, _start, _width, _h in self._needs(missing))
            return False
        for key, start, width, hessian in self._needs(missing):
            if hessian and self.arena is not None:
                self.H[key] = self.arena.carve(width * width * 4, f"calibration/{key}").view(torch.float32).view(width, width).zero_()
            elif hessian:
                self.H[key] = torch.zeros(width, width, dtype=torch.float32, device=self.device)
            if hessian and decode_rows:
                shape = (GRAM_ROWS + decode_rows - 1, width)
                itemsize = torch.empty((), dtype=input_dtype).element_size()
                if self.arena is not None:
                    buffer = self.arena.carve(shape[0] * width * itemsize, f"calibration/staging/{key}").view(input_dtype).view(shape)
                else:
                    buffer = torch.empty(shape, dtype=input_dtype, device=self.device)
                self.staging[key] = (buffer, torch.zeros((), dtype=torch.int32, device=self.device))
            self.rows[key] = torch.zeros((), dtype=torch.float32, device=self.device)
            self.amax[key] = torch.zeros(width, dtype=torch.float32, device=self.device)
        self.used += need
        self.tiles[name] = [(key, start, width, hessian) for key, start, width, hessian in self._needs(missing)]
        if unsmooth is not None:
            self.unsmooth[name] = unsmooth.detach().float().to(self.device)
        layer.observer = lambda flat, rows_ok, name=name, small=small_rows: self.observe(name, flat, rows_ok, small)
        return True

    @staticmethod
    def nbytes(missing, max_decode_rows: int = 0, input_dtype=torch.float32) -> int:
        if input_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('calibration staging supports declared FP32 or BF16 inputs')
        itemsize = torch.empty((), dtype=input_dtype).element_size()
        return sum((width * width * 4 if hessian else 0) + width * 4 + 4096
                   + ((GRAM_ROWS + max_decode_rows - 1) * width * itemsize if hessian and max_decode_rows else 0)
                   for _key, _start, width, hessian in Calibration._needs(missing))

    def arm(self) -> None:
        if self.filed is not None:
            raise ValueError('filed calibration is complete; use a new observer for another collection')
        self.armed.fill_(1.0)

    def observe(self, name: str, flat: torch.Tensor, rows_ok, small_rows: bool) -> None:
        if self.filed is not None:
            return  # eager prefill needs no more statistics after the blobs are durable
        if flat.shape[0] <= self.max_decode_rows and not small_rows:
            return
        if self.device.type == "cuda" and flat.shape[0] <= self.max_decode_rows:
            from .calibration_gram import observe
            for key, start, width, _hessian in self.tiles[name]:
                buffer, cursor = self.staging.get(key, (None, None))
                if buffer is not None and buffer.dtype == torch.bfloat16:
                    if flat.dtype != torch.bfloat16 or (rows_ok is not None and rows_ok.dtype != torch.bool):
                        raise ValueError('BF16 calibration requires BF16 inputs and a boolean committed-row mask')
                observe(flat[:, start:start + width], rows_ok, buffer, self.H.get(key), cursor,
                        self.armed, self.rows[key], self.amax[key])
            return
        xf = flat.float()
        if rows_ok is not None:
            xf = xf * rows_ok.to(xf.dtype).view(-1, 1)
        xf = xf * self.armed
        # `torch.tensor(python_scalar, device="cuda")` performs a host-to-device
        # copy that CUDA graph capture rejects.  `new_full` emits a device-side
        # fill and remains valid in both eager execution and captured graphs.
        count = (rows_ok.to(torch.float32).sum() if rows_ok is not None
                 else xf.new_full((), flat.shape[0], dtype=torch.float32)) * self.armed
        for key, start, width, hessian in self.tiles[name]:
            part = xf[:, start:start + width]
            if hessian:
                self.flush(key)
                self.H[key].addmm_(part.t(), part)
            self.rows[key] += count
            torch.maximum(self.amax[key], part.abs().amax(0), out=self.amax[key])

    def flush(self, key=None) -> None:
        """Include the pending small rows before a large update, inspection or filing.

        All operations stay on the caller's stream; no device counter is read by Python.
        Row counts and peaks already include staging, so progress needs no flush.
        """
        for name in ([key] if key is not None else self.staging):
            if name in self.staging:
                from .calibration_gram import flush
                buffer, cursor = self.staging[name]
                flush(buffer, self.H[name], cursor, self.armed)

    def progress(self) -> int:
        """The fewest rows any blob has (a device read: ask rarely)."""
        return int(min(float(r) for r in self.rows.values())) if self.rows else 0

    def complete(self, target: int = ROWS_TARGET) -> bool:
        return bool(self.rows) and self.progress() >= target

    def save(self, root: "str | Path", rank: int) -> "list[Path]":
        """One blob per tile under `<root>/mkcalib/rank<rank>/`, in the store's form. Overwrites what an older stack
        left, through a temporary file. A tile summed for its peaks alone keeps the Hessian the store already had,
        and its token count with it: only the peaks are this boot's."""
        self.flush()
        written = []
        back = {}                                                          # blob key -> the s to undo (H -> s H s, amax -> amax * s)
        for name, tiles in self.tiles.items():
            s = self.unsmooth.get(name)
            if s is not None:
                for key, start, width, _hessian in tiles:
                    back[key] = s[start:start + width]
        for key in self.rows:
            path = Path(root) / "mkcalib" / f"rank{rank}" / (key + ".pt")
            amax = self.amax[key].detach().float()
            s = back.get(key)
            if s is not None:
                amax = amax * s
            H = self.H.get(key)
            if H is None:                                                  # summed for its peaks alone: the store's Hessian stays
                if not path.is_file():
                    continue                                               # its blob went away under us: a later boot sums the whole thing
                blob = torch.load(path, map_location="cpu", weights_only=True)
                H, ntok = blob["H"].float(), int(blob["ntok"])
            else:
                H, ntok = H.detach().float(), int(self.rows[key])
                if s is not None:
                    H = (H * s[:, None]) * s[None, :]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")            # a boot that dies mid-write leaves the old blob, not a truncated one
            try:
                torch.save({"H": H.cpu().contiguous(), "amax": amax.cpu().contiguous(), "ntok": ntok, "name": key}, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            written.append(path)
        # These blobs are filed once per boot. Captured small-row observers
        # retain the device scalar's address, so disarming also makes old
        # decode graphs stop accumulating without recapture or a host read.
        # Do this only after every write succeeded: a failed save must remain
        # collectable and retryable.
        self.armed.zero_()
        self.filed = written
        return written

    def status(self) -> str:
        if not self.rows:
            return "nothing to sum"
        peaks = len(self.rows) - len(self.H)
        return (f"{'filed' if self.filed else 'collecting'} {self.progress()}/{ROWS_TARGET} rows over {len(self.rows)} blobs"
                + (f" ({peaks} for their channel peaks alone)" if peaks else "")
                + (f", {len(self.deferred)} tiles deferred to a later boot" if self.deferred else ""))
