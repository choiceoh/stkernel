"""Calibration capture and precision measurement for a measurement boot of GLM-5.3-Flash.

A served boot with this attached (boot.EXPERT_CAPTURE: a measurement arm, never production) watches EAGER PREFILL
chunks once every graph capture and warm-up is behind it. It changes no number the engine returns; the one change to
the path is that a prefill chunk keeps every final hidden row instead of the last one, which the head measurement reads.

  * MoE: each MoE layer's input rows and router choice (x bf16 [T, 4096], sel int16 [T, 8], w f32 [T, 8]) go to the
    capture rank's disk, one safetensors file per (layer, chunk), with the chunk's token ids beside them. They are
    the routed experts' own calibration: the engine's dense sums cover the shared experts and dense projections only,
    and a re-quantization of 288 experts needs each expert's inputs (and, for down_proj, rows only the expert's own
    gate/up produce). The split into fit and held-out is made offline, by document.
  * KV: at sampled query rows of every sparse-MLA layer, the attention output over the SAME selected slots from the
    latent as served (an e4m3 cast at scale 1), at static power-of-two scales (what the kernels' `ckv_scale` scalar
    could carry for free), with dynamic per-row and per-128-tile scales, and from the unquantized BF16 row. Relative
    errors against BF16 per layer, and the latent's magnitude histogram, so a static scale can be chosen from them.
  * Head: at sampled prefill positions, the served FP8 head's logits against the rank file's BF16 head, gathered over
    ranks: top-1 agreement, KL(bf16 || fp8), and the NLL of the true next token under each -- summed in stats.jsonl,
    and position by position in head.jsonl (document, position, target, the two NLLs, KL, agreement), so two boots fed
    the same documents are compared position for position and split by document offline.

Only the capture rank writes: <directory>/moe/L<layer>/d<document>-c<ctx>.safetensors, <directory>/ids/, chunks.jsonl,
head.jsonl and stats.jsonl. Every rank runs the head's two gathers, which is why its sampling depends only on what every rank shares.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import queue
import shutil
import struct
import threading
import time

import torch

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
TILE = 128
POW2_SHIFTS = (1, 2, 3, 4, 5, 6, 7, 8)   # static latent scales 2^-k; k = 0 is the served cast itself
LOG2_LOW, LOG2_HIGH = -16, 12             # latent magnitude histogram: one bin per octave of |x| in [2^-16, 2^12)
CAPTURE_RANK = 3                          # srv4: the box the re-quantization runs on, so the rows never cross the fabric
TINY = torch.finfo(torch.float32).tiny
ALL_SECTIONS = ("moe", "kv", "head", "dense")
SECTIONS = ("moe", "kv", "head")         # the first capture: routed-expert rows, KV latent and head precision
DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int16: "I16",
          torch.int32: "I32", torch.int64: "I64", torch.uint8: "U8"}


# -- latent variants (pure, CPU-testable) ------------------------------------------------------------------------
def served_latent(rows: torch.Tensor) -> torch.Tensor:
    """The latent exactly as the engine writes it (`kv_n.to(E4M3)`), read back at scale 1."""
    return rows.to(E4M3).float()


def static_latent(rows: torch.Tensor, scale: float) -> torch.Tensor:
    """e4m3 at one static scale -- the kernels' ckv_scale -- saturating at the grid's end, read back."""
    return (rows.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).float() * scale


def dynamic_latent(rows: torch.Tensor, width: int) -> torch.Tensor:
    """e4m3 with one absmax/448 scale per `width` dims of each row (the row: 512; a tile: 128), read back."""
    n, d = rows.shape
    x = rows.float().reshape(n, d // width, width)
    scale = (x.abs().amax(-1, keepdim=True) / E4M3_MAX).clamp_min(TINY)
    return ((x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).float() * scale).reshape(n, d)


def latent_variants(rows: torch.Tensor) -> "dict[str, torch.Tensor]":
    out = {"bf16": rows.float(), "served": served_latent(rows)}
    for k in POW2_SHIFTS:
        out[f"pow2_{k}"] = static_latent(rows, 2.0 ** -k)
    out["row"] = dynamic_latent(rows, rows.shape[1])
    out["tile"] = dynamic_latent(rows, TILE)
    return out


def positions_of(slots: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """For each slot id in `slots`, its position in `table` (table[p] = the slot of position p); -1 when absent."""
    order = torch.argsort(table)
    ordered = table[order]
    idx = torch.searchsorted(ordered, slots).clamp_max(table.numel() - 1)
    return torch.where(ordered[idx] == slots, order[idx], torch.full_like(order[idx], -1))


def magnitude_histogram(rows: torch.Tensor) -> torch.Tensor:
    """Counts of |x| per octave [2^k, 2^(k+1)) for k in [LOG2_LOW, LOG2_HIGH), with the underflow first and zeros
    folded into it, the overflow last: float64 [2 + octaves]."""
    x = rows.float().abs().reshape(-1)
    octaves = LOG2_HIGH - LOG2_LOW
    k = torch.floor(torch.log2(x.clamp_min(TINY))).clamp(LOG2_LOW - 1, LOG2_HIGH) - (LOG2_LOW - 1)
    return torch.bincount(k.long(), minlength=octaves + 2).double().cpu()


class Ratio:
    """Sum of squared errors over sum of squared references: one relative error for a whole population."""
    __slots__ = ("err", "ref", "n")

    def __init__(self):
        self.err, self.ref, self.n = 0.0, 0.0, 0

    def add(self, estimate: torch.Tensor, reference: torch.Tensor) -> None:
        self.err += float((estimate.double() - reference.double()).pow(2).sum())
        self.ref += float(reference.double().pow(2).sum())
        self.n += 1

    def value(self):
        return math.sqrt(self.err / self.ref) if self.ref > 0 else None


# -- head (pure) ---------------------------------------------------------------------------------------------------
def bf16_logits(hs: torch.Tensor, head: torch.Tensor, block: int = 8192) -> torch.Tensor:
    """hs [k, H], head [V, H] bf16 -> float32 logits [k, V], the head's rows widened a block at a time (a float32
    copy of a whole 38,720-row shard is 634 MiB; a block is 125)."""
    x = hs.float()
    out = torch.empty((x.shape[0], head.shape[0]), dtype=torch.float32, device=x.device)
    for a in range(0, head.shape[0], block):
        b = min(head.shape[0], a + block)
        out[:, a:b] = x @ head[a:b].float().T
    return out


def head_position_metrics(fp8: torch.Tensor, bf16: torch.Tensor, targets: torch.Tensor, decodable: int) -> dict:
    """fp8, bf16 [n, V] logits over the whole vocabulary; targets [n] the next token. Per position [n]: the NLL of the
    target under each head, KL(bf16 || fp8), top-1 agreement; and the logit error's two sums."""
    lf, lb = fp8.float()[:, :decodable], bf16.float()[:, :decodable]
    pf, pb = torch.log_softmax(lf, -1), torch.log_softmax(lb, -1)
    t = targets.long().clamp(0, decodable - 1)[:, None]
    return dict(nll_fp8=-pf.gather(1, t)[:, 0], nll_bf16=-pb.gather(1, t)[:, 0], kl=(pb.exp() * (pb - pf)).sum(-1),
                agree=lf.argmax(-1) == lb.argmax(-1),
                logit_err=float((lf.double() - lb.double()).pow(2).sum()), logit_ref=float(lb.double().pow(2).sum()))


def head_sums(per: dict) -> dict:
    """head_position_metrics as sums, not means."""
    return dict(n=int(per["agree"].numel()), top1_agree=float(per["agree"].double().sum()), kl=float(per["kl"].double().sum()),
                nll_fp8=float(per["nll_fp8"].double().sum()), nll_bf16=float(per["nll_bf16"].double().sum()),
                logit_err=per["logit_err"], logit_ref=per["logit_ref"])


def head_metrics(fp8: torch.Tensor, bf16: torch.Tensor, targets: torch.Tensor, decodable: int) -> dict:
    """fp8, bf16 [n, V] logits over the whole vocabulary; targets [n] the next token. Sums, not means."""
    return head_sums(head_position_metrics(fp8, bf16, targets, decodable))


def head_record(doc, seq: int, ctx: int, positions: torch.Tensor, targets: torch.Tensor, per: dict) -> dict:
    """One head.jsonl line: a chunk's scored positions (absolute, in its sequence), in increasing order."""
    order = torch.argsort(positions.cpu())
    col = lambda t, nd=6: [round(float(v), nd) for v in t.detach().cpu()[order].tolist()]   # noqa: E731
    return dict(doc=doc, seq=int(seq), ctx=int(ctx), pos=[int(v) for v in positions.cpu()[order].tolist()],
                target=[int(v) for v in targets.cpu()[order].tolist()], nll_fp8=col(per["nll_fp8"]),
                nll_bf16=col(per["nll_bf16"]), kl=col(per["kl"], 8), agree=[int(v) for v in per["agree"].cpu()[order].tolist()])


def head_from_rank_file(ranks_dir, rank: int, device, world: int = 4) -> torch.Tensor:
    """This rank's BF16 head shard, read from its rank file: the boot consumed its own copy into the FP8 packs."""
    from safetensors import safe_open
    path = Path(ranks_dir) / f"rank{rank}of{world}.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        head = f.get_tensor("head")
    if head.dtype != torch.bfloat16 or head.ndim != 2:
        raise ValueError(f"{path}: head is {head.dtype} {tuple(head.shape)}, not a bf16 matrix")
    return head.to(device)


# -- files -----------------------------------------------------------------------------------------------------------
def write_safetensors(path: Path, tensors: dict, metadata: dict) -> int:
    """A safetensors file written without an intermediate copy, synced, and dropped from the page cache: on the GB10's
    unified memory the page cache is the engine's memory too. World-writable, so the user who owns the box (not the
    container's root) can delete it. Returns the bytes written."""
    header, order, offset = {}, [], 0
    for name, t in tensors.items():
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        n = t.numel() * t.element_size()
        header[name] = {"dtype": DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + n]}
        order.append(t)
        offset += n
    header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for t in order:
            if t.numel():
                f.write(t.reshape(-1).view(torch.uint8).numpy().data)
        f.flush()
        os.fsync(f.fileno())
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    os.chmod(tmp, 0o666)
    os.replace(tmp, path)
    return 8 + len(blob) + offset


def open_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass
    return path


class Writer:
    """One background thread, a byte-bounded queue: a layer's rows leave the step while the next layer runs, and the
    step waits only when `limit_bytes` are already in flight. A writer that stops draining drops rows (and says so)
    rather than holding the fleet's step: base/stall kills a step at 300 s, and the other ranks wait inside a collective."""

    def __init__(self, limit_bytes: int, wait_s: float = 30.0):
        self.q = queue.Queue()
        self.limit, self.wait_s = limit_bytes, wait_s
        self.pending = 0
        self.cv = threading.Condition()
        self.error = None
        self.written = self.files = self.dropped = 0
        self.seconds = 0.0
        self.thread = threading.Thread(target=self._run, name="expert-capture-writer", daemon=True)
        self.thread.start()

    def put(self, path: Path, tensors: dict, metadata: dict, nbytes: int) -> bool:
        deadline = time.monotonic() + self.wait_s
        with self.cv:
            while self.pending and self.pending + nbytes > self.limit:
                if time.monotonic() > deadline or not self.thread.is_alive():
                    self.dropped += 1
                    self.error = self.error or "writer stalled: rows dropped"
                    return False
                self.cv.wait(timeout=1.0)
            self.pending += nbytes
        self.q.put((path, tensors, metadata, nbytes))
        return True

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, tensors, metadata, nbytes = item
            t0 = time.perf_counter()
            try:
                self.written += write_safetensors(path, tensors, metadata)
                self.files += 1
            except Exception as exc:                       # noqa: BLE001 -- a failed write is a counted loss, not a crash
                self.error = f"{type(exc).__name__}: {exc}"
                self.dropped += 1
            self.seconds += time.perf_counter() - t0
            del tensors
            with self.cv:
                self.pending -= nbytes
                self.cv.notify_all()

    def close(self):
        self.q.put(None)
        self.thread.join()


# -- the hooks -------------------------------------------------------------------------------------------------------
class Capture:
    def __init__(self, engine, directory, *, head_bf16=None, capture_rank: int = CAPTURE_RANK, kv_rows: int = 8,
                 head_rows: int = 64, limit_gib: float = 160.0, disk_floor_gib: float = 250.0, queue_gib: float = 0.75,
                 kv_positions: int = 32768, kv_sequences: int = 2, report_every: int = 16, seed: int = 0,
                 sections=SECTIONS):
        unknown = set(sections) - set(ALL_SECTIONS)
        if unknown:
            raise ValueError(f"capture sections are {ALL_SECTIONS}, not {sorted(unknown)}")
        self.engine, self.net, self.caches = engine, engine.net, engine.caches
        self.comm, self.F = self.net.comm, self.net.F
        self.rank = self.comm.rank
        self.capture_rank = capture_rank
        self.mine = self.rank == capture_rank
        self.dir = Path(directory)
        self.head_bf16 = head_bf16
        self.kv_rows, self.head_rows = kv_rows, head_rows
        self.limit = int(limit_gib * 2**30)
        self.disk_floor = int(disk_floor_gib * 2**30)
        self.kv_positions, self.kv_sequences = kv_positions, kv_sequences
        self.report_every = report_every
        self.decodable = int(getattr(engine, "decodable", None) or self.F.vocab)
        self.moe_layers = [L for L in self.net.layers if self.F.is_moe(L)]
        self.dense_layers = [L for L in self.net.layers if not self.F.is_moe(L)]
        self.kv_weights = {L: self.net.p[f"L{L}.mla.kv_a_norm"] for L in self.net.layers if self.F.is_dsa(L)}
        self.gen = torch.Generator(device="cpu").manual_seed(seed + 7919 * self.rank)
        self.step = self.layer = self.doc = None
        self.docs = {}                         # seq -> (document, the context its next chunk continues at)
        self.next_doc = 0
        self.chunk_on = False
        self.on = {k: k in sections for k in ALL_SECTIONS}
        self.head_enabled = "head" in sections and head_bf16 is not None   # fixed at attach: every rank must agree on the gathers
        self.errors = {}
        self.stopped = None
        self.chunks = self.captured_chunks = self.captured_tokens = self.moe_bytes = 0
        self.kv_hist = OrderedDict()           # document -> {layer: [positions, 512] bf16 from position 0}
        self.kv = {}                           # layer -> {metric: Ratio}
        self.kv_skipped = {}                   # layer -> query rows whose selection reached a position not seen here
        self.latent_hist = {}                  # layer -> magnitude histogram
        self.latent_absmax = {}
        self.head = {}
        self.started = time.time()
        self.writer = None
        if self.mine:
            open_directory(self.dir)
            open_directory(self.dir / "ids")
            if self.on["moe"]:
                for L in self.moe_layers:
                    open_directory(open_directory(self.dir / "moe") / f"L{L:02d}")
            if self.on["dense"]:
                for L in self.dense_layers:
                    open_directory(open_directory(self.dir / "dense") / f"L{L:02d}")
            self.writer = Writer(int(queue_gib * 2**30))
        self._install()

    # installation: instance attributes shadow what the engine calls through `self.`
    def _install(self):
        engine, net = self.engine, self.net
        self._saved = (engine.__dict__.get("_prefill_forward"), net.__dict__.get("route"), net.__dict__.get("_dsa"),
                       net.__dict__["_norm"], net.lanes, net.__dict__.get("_dense"))
        forward, route, dsa, norm, dense = engine._forward, net.route, net._dsa, net._norm, net._dense
        mla = net.lanes.mla_sparse

        def prefill_forward(step):
            self.doc = self._document(step)
            self._chunk_start(step)
            self.step = step
            try:
                h, aux = forward(step, last_hidden_only=False)
            finally:
                self.step = self.layer = None
            self._head(step, h)
            self._chunk_done(step)
            return h[-1:], aux

        def route_hook(L, x):
            sel, w = route(L, x)
            if self.chunk_on and self.step is not None and self.on["moe"]:
                self._guard("moe", self._moe, L, x, sel, w)
            return sel, w

        def dsa_hook(L, x, step, caches, reduce=None, **kwargs):
            if self.step is None:
                return dsa(L, x, step, caches, reduce, **kwargs)
            self.layer = L
            try:
                return dsa(L, x, step, caches, reduce, **kwargs)
            finally:
                self.layer = None

        def norm_hook(x, weight, eps):
            out = norm(x, weight, eps)
            L = self.layer
            if L is not None and self.mine and self.on["kv"] and weight is self.kv_weights.get(L):
                self._guard("kv", self._kv_rows, L, out)
            return out

        def mla_hook(*args, **kwargs):
            out = mla(*args, **kwargs)
            L = self.layer
            if L is not None and self.mine and self.on["kv"] and len(args) == 6 and not kwargs:
                q_abs, _latent, slots, valid, scale, _ckv = args
                self._guard("kv", self._kv_errors, L, q_abs, slots, valid, scale, out)
            return out

        def dense_hook(L, x, *args, **kwargs):
            if self.chunk_on and self.step is not None and self.on["dense"]:
                self._guard("dense", self._dense_rows, L, x)
            return dense(L, x, *args, **kwargs)

        engine._prefill_forward = prefill_forward
        net.route, net._dsa, net._norm, net._dense = route_hook, dsa_hook, norm_hook, dense_hook
        net.lanes = replace(net.lanes, mla_sparse=mla_hook)

    def detach(self):
        engine, net = self.engine, self.net
        prefill, route, dsa, norm, lanes, dense = self._saved
        for owner, name, value in ((engine, "_prefill_forward", prefill), (net, "route", route), (net, "_dsa", dsa),
                                   (net, "_dense", dense)):
            if value is None:
                owner.__dict__.pop(name, None)
            else:
                setattr(owner, name, value)
        net._norm, net.lanes = norm, lanes

    def _guard(self, name, fn, *args):
        """A measurement that raises is switched off and reported; the request it rode on is served as usual."""
        try:
            fn(*args)
        except Exception as exc:                           # noqa: BLE001
            self.on[name] = False
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"  expert capture: rank {self.rank} {name} measurement off: {self.errors[name]}", flush=True)

    # chunks -----------------------------------------------------------------------------------------------------------
    def _document(self, step):
        """The runner's sequence ids come back for the next request (at C=1 two of them alternate), so nothing is
        keyed by one: a document begins at a chunk with context 0, or at one that does not continue where its
        sequence's last chunk ended (a prefix hit, a restore)."""
        if len(step.segments) != 1:
            return None
        s = step.segments[0]
        doc, continues_at = self.docs.get(s.seq, (None, None))
        if doc is None or s.ctx == 0 or s.ctx != continues_at:
            doc = self.next_doc
            self.next_doc += 1
        self.docs[s.seq] = (doc, s.ctx + s.length)
        return doc

    def _chunk_start(self, step):
        self.chunk_on = False
        if not self.mine or not (self.on["moe"] or self.on["dense"]) or self.stopped or self.doc is None:
            return
        n = step.ids.numel()
        need = (n * len(self.moe_layers) * (self.F.hidden * 2 + self.F.topk_experts * 6) * self.on["moe"]
                + n * len(self.dense_layers) * self.F.hidden * 2 * self.on["dense"] + n * 4)
        if self.moe_bytes + need > self.limit:
            self.stopped = f"byte cap {self.limit / 2**30:.0f} GiB"
        elif shutil.disk_usage(self.dir).free - self.writer.pending - need < self.disk_floor:
            self.stopped = f"disk floor {self.disk_floor / 2**30:.0f} GiB"
        if self.stopped:
            print(f"  expert capture: stopped after {self.captured_tokens} tokens ({self.stopped})", flush=True)
            return
        self.chunk_on = True

    def _chunk_done(self, step):
        self.chunks += 1
        if self.chunk_on:
            s = step.segments[0]
            ids = step.ids.detach().to("cpu", torch.int32)
            self.writer.put(self.dir / "ids" / f"d{self.doc:06d}-c{s.ctx:07d}.safetensors", dict(ids=ids),
                            dict(doc=self.doc, seq=s.seq, ctx=s.ctx, tokens=ids.numel()), ids.numel() * 4)
            with open(self.dir / "chunks.jsonl", "a") as f:
                f.write(json.dumps(dict(doc=self.doc, seq=s.seq, ctx=s.ctx, tokens=int(ids.numel()), t=round(time.time(), 3))) + "\n")
            self.captured_chunks += 1
            self.captured_tokens += int(ids.numel())
        self.chunk_on = False
        if self.mine and self.report_every and self.chunks % self.report_every == 0:
            self.report()

    # MoE -------------------------------------------------------------------------------------------------------------
    def _moe(self, L, x, sel, w):
        s = self.step.segments[0]
        tensors = dict(x=x.detach().to("cpu", torch.bfloat16), sel=sel.detach().to("cpu", torch.int16),
                       w=w.detach().to("cpu", torch.float32))
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        path = self.dir / "moe" / f"L{L:02d}" / f"d{self.doc:06d}-c{s.ctx:07d}.safetensors"
        if self.writer.put(path, tensors, dict(layer=L, doc=self.doc, seq=s.seq, ctx=s.ctx, tokens=x.shape[0]), nbytes):
            self.moe_bytes += nbytes

    def _dense_rows(self, L, x):
        """A dense MLP layer's input rows (layers 0-2): what its GPTQ fits gate/up to, and down through them."""
        s = self.step.segments[0]
        tensors = dict(x=x.detach().to("cpu", torch.bfloat16))
        nbytes = tensors["x"].numel() * 2
        path = self.dir / "dense" / f"L{L:02d}" / f"d{self.doc:06d}-c{s.ctx:07d}.safetensors"
        if self.writer.put(path, tensors, dict(layer=L, doc=self.doc, seq=s.seq, ctx=s.ctx, tokens=x.shape[0]), nbytes):
            self.moe_bytes += nbytes

    # KV ----------------------------------------------------------------------------------------------------------------
    def _kv_rows(self, L, rows):
        s = self.step.segments[0]
        hist = self.latent_hist.get(L)
        counts = magnitude_histogram(rows)
        self.latent_hist[L] = counts if hist is None else hist + counts
        self.latent_absmax[L] = max(self.latent_absmax.get(L, 0.0), float(rows.float().abs().max()))
        doc = self.kv_hist.get(self.doc)
        if doc is None:
            if s.ctx != 0:
                return                                   # a document whose beginning this capture never saw
            doc = self.kv_hist[self.doc] = {}
            while len(self.kv_hist) > self.kv_sequences:
                self.kv_hist.popitem(last=False)
        self.kv_hist.move_to_end(self.doc)
        kept = doc.get(L)
        have = 0 if kept is None else kept.shape[0]
        if have != s.ctx or s.ctx + rows.shape[0] > self.kv_positions:
            doc.pop(L, None)                             # a gap or past the kept horizon
            return
        doc[L] = rows.detach().clone() if kept is None else torch.cat([kept, rows.detach()])

    def _kv_errors(self, L, q_abs, slots, valid, scale, out):
        from engine.modules.sparse_attention import mla_sparse_mqa
        s = self.step.segments[0]
        kept = self.kv_hist.get(self.doc, {}).get(L)
        if kept is None or kept.shape[0] != s.ctx + q_abs.shape[0]:
            return
        dev = q_abs.device
        table = self.caches.token_slots(L, s.seq, torch.arange(kept.shape[0], device=dev)).long()
        picks = torch.randperm(q_abs.shape[0], generator=self.gen)[: self.kv_rows].tolist()
        counts = valid.detach().to("cpu")
        stats = self.kv.setdefault(L, {})
        for r in picks:
            v = int(counts[r])
            if v < 1:
                continue
            pos = positions_of(slots[r, :v].long(), table)
            if bool((pos < 0).any()):
                self.kv_skipped[L] = self.kv_skipped.get(L, 0) + 1
                continue
            variants = latent_variants(kept[pos])
            q = q_abs[r:r + 1].float()
            idx = torch.arange(v, device=dev, dtype=torch.int32)[None, :]
            n = torch.tensor([v], device=dev, dtype=torch.int32)
            outs = {k: mla_sparse_mqa(q, rows, idx, n, scale, 1.0).float() for k, rows in variants.items()}
            for k, o in outs.items():
                if k != "bf16":
                    stats.setdefault("out_" + k, Ratio()).add(o, outs["bf16"])
                    stats.setdefault("lat_" + k, Ratio()).add(variants[k], variants["bf16"])
            stats.setdefault("kernel_vs_served_reference", Ratio()).add(out[r:r + 1].float(), outs["served"])
            stats.setdefault("selected_rows", Ratio()).n += v

    # head --------------------------------------------------------------------------------------------------------------
    def _head(self, step, h):
        """Every rank, in the same order: which positions depends only on the step's sequence and context."""
        if not self.head_enabled or self.head_rows <= 0 or len(step.segments) != 1 or h.shape[0] < 2:
            return
        s = step.segments[0]
        k = min(self.head_rows, h.shape[0] - 1)
        g = torch.Generator(device="cpu").manual_seed((s.seq * 1_000_003 + s.ctx) % (2**62))
        idx = torch.randperm(h.shape[0] - 1, generator=g)[:k].to(h.device)
        hs = h.index_select(0, idx).contiguous()
        fp8 = self.comm.all_gather(self.net.head_local(hs).float().contiguous(), dim=-1)
        bf16 = self.comm.all_gather(bf16_logits(hs, self.head_bf16), dim=-1)
        if self.mine and self.on["head"]:
            self._guard("head", self._head_stats, fp8, bf16, step.ids.index_select(0, idx + 1), s, idx + s.ctx)

    def _head_stats(self, fp8, bf16, targets, segment=None, positions=None):
        per = head_position_metrics(fp8, bf16, targets, self.decodable)
        for key, value in head_sums(per).items():
            self.head[key] = self.head.get(key, 0) + value
        if segment is not None:
            with open(self.dir / "head.jsonl", "a") as f:
                f.write(json.dumps(head_record(self.doc, segment.seq, segment.ctx, positions, targets, per)) + "\n")

    # report --------------------------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        kv = {}
        for L, stats in sorted(self.kv.items()):
            row = {k: (r.value() if k != "selected_rows" else r.n) for k, r in stats.items()}
            row["query_rows"] = stats["kernel_vs_served_reference"].n if "kernel_vs_served_reference" in stats else 0
            row["skipped_rows"] = self.kv_skipped.get(L, 0)
            kv[str(L)] = row
        latent = {str(L): dict(absmax=self.latent_absmax.get(L), octaves_from=LOG2_LOW,
                               counts=[int(c) for c in hist.tolist()]) for L, hist in sorted(self.latent_hist.items())}
        head = {}
        if self.head.get("n"):
            n = self.head["n"]
            head = dict(positions=n, top1_agree=self.head["top1_agree"] / n, kl=self.head["kl"] / n,
                        nll_fp8=self.head["nll_fp8"] / n, nll_bf16=self.head["nll_bf16"] / n,
                        logit_rel_err=math.sqrt(self.head["logit_err"] / self.head["logit_ref"]) if self.head["logit_ref"] else None)
        w = self.writer
        return dict(t=round(time.time(), 3), seconds=round(time.time() - self.started, 1), rank=self.rank,
                    chunks=self.chunks, captured_chunks=self.captured_chunks, captured_tokens=self.captured_tokens,
                    moe_gib=round(self.moe_bytes / 2**30, 3), stopped=self.stopped, on=dict(self.on), errors=dict(self.errors),
                    writer=None if w is None else dict(files=w.files, gib=round(w.written / 2**30, 3), seconds=round(w.seconds, 1),
                                                       dropped=w.dropped, pending_mib=round(w.pending / 2**20, 1), error=w.error),
                    kv=kv, latent=latent, head=head)

    def report(self) -> dict:
        row = self.summary()
        if self.mine:
            with open(self.dir / "stats.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            head = row["head"]
            print(f"  expert capture: {row['captured_tokens']} tokens in {row['captured_chunks']} chunks, {row['moe_gib']} GiB"
                  + (f"; head top1 {head['top1_agree']:.4f} kl {head['kl']:.2e}" if head else "")
                  + (f"; errors {row['errors']}" if row["errors"] else ""), flush=True)
        return row

    def close(self) -> dict:
        self.detach()
        if self.writer is not None:
            self.writer.close()
        return self.report()


def attach(engine, directory, ranks_dir, sections=SECTIONS, **kwargs) -> Capture:
    """Every rank: the BF16 head from its own rank file when the head is measured (every rank takes part in its
    gathers)."""
    comm = engine.net.comm
    head = head_from_rank_file(ranks_dir, comm.rank, engine.caches.device, comm.world_size) if "head" in sections else None
    return Capture(engine, directory, head_bf16=head, sections=sections, **kwargs)


def arm_calibration_phases(engine) -> None:
    """The calibration arm's fit/held-out cut (boot.CALIBRATION_CAPTURE). The door's POST /v1/engine/calibration
    files every rank's sums at the same step; filed under a root ending in "-fit", the sums are then zeroed and
    re-armed in place (no second budget), so the documents fed afterwards sum a held-out set filed under another
    root. Only eager prefill contributes: the arm plans target tiles without small rows."""
    original = engine.file_calibration
    # housekeeping files the sums on its own once `complete()` -- whose default target was bound at import (32K rows):
    # in the arm only the door's two POSTs file
    engine.calibration.complete = lambda target=None: False

    def file_calibration(root=None):
        written = original(root)
        c = engine.calibration
        if written and c is not None and root is not None and str(root).rstrip("/").endswith("-fit"):
            for t in list(c.H.values()) + list(c.rows.values()) + list(c.amax.values()):
                t.zero_()
            c.filed = None
            c.armed.fill_(1.0)
            print(f"  calibration: rank {engine.net.comm.rank} filed {len(written)} fit blobs under {root}; "
                  "sums zeroed for the held-out documents", flush=True)
        return written

    engine.file_calibration = file_calibration
