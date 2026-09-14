"""Model-owned MK post/pre fusion and one-shot AR consumer PDL.

Immutable coefficient packs and scratch stay alive with the model, across
every captured graph. Calls must be ordered on its execution stream.

The geometry comes from the bound kernel shape (engine/base/kernel_shape):
`hidden` residual streams `hc` wide. The kernel itself is compiled for two
hidden widths (kernels.cu: HIDDEN 4096 and HIDDEN_V41 5120) at hc 4, so a
shape outside that is refused here, by name, before any weight is packed.

`MHC` computes GLM-5.3's mhc form (run_mhc). `MHCV41` computes DeepSeek-V4.1's
split-sinkhorn form through the megakernel's V4.1 seam (run_mhc_v41) at the same
compiled widths: glue (cells.GLUE), admitted by cells.mhc_v41_refusal.
"""
import math

import torch
from engine.kernels.dense import extension
# The compiled cell, stated once in engine/kernels/cells.py: MHC_MAX_TOK_DEF rows, HCHUNK, the HIDDEN/HIDDEN_V41
# instances mk_run_mhc accepts, HC.
from engine.kernels.cells import (MHC_HC as COMPILED_HC, MHC_HCHUNK as HCHUNK, MHC_HIDDEN as COMPILED_HIDDEN, MHC_VARIANT,
                                  MHC_MAX_TOK, mhc_v41_refusal)


def geometry(shape=None) -> "tuple[int, int, int, int]":
    """(hidden, hc, nout, nchunk) for the bound kernel shape; a shape the kernel is not compiled for is refused."""
    if shape is None:
        from engine.base.kernel_shape import bound
        shape = bound()
    if shape.hc_variant != MHC_VARIANT:
        raise ValueError(f"MK MHC computes the {MHC_VARIANT} form; the bound kernel shape mixes by {shape.hc_variant}")
    if shape.hidden not in COMPILED_HIDDEN or shape.hc != COMPILED_HC:
        raise ValueError(f"MK MHC is compiled for hidden {COMPILED_HIDDEN} at hc {COMPILED_HC}; "
                         f"the bound kernel shape asks for hidden {shape.hidden} hc {shape.hc}")
    return shape.hidden, shape.hc, shape.hc * (2 + shape.hc), shape.hidden // HCHUNK


def workspace_sizes(hidden: int, hc: int, nout: int, nchunk: int) -> "list[tuple[int, torch.dtype]]":
    """The kernel's scratch, in its argument order: yp [nchunk, tok, nout], rp [nchunk, tok], sq, pmix [tok, hc],
    ol_stash [tok, hidden] and the barrier word."""
    return [(nchunk * MHC_MAX_TOK * nout, torch.float32), (nchunk * MHC_MAX_TOK, torch.float32),
            (nchunk * MHC_MAX_TOK, torch.float32), (MHC_MAX_TOK * hc, torch.float32),
            (MHC_MAX_TOK * hidden, torch.bfloat16), (8, torch.int32)]


class MHC:
    # Rows whose packet consumer reads the lossless BF16 pack: C=1 and C=2 verify steps at K=7. `packet_rows=8` is the
    # same-build control (the C=1-only gate) for the probe that qualifies the 16-row form; serving never passes it.
    PACKET_ROWS = 16

    def __init__(self, weights, *, prefill=False, packet_rows=PACKET_ROWS):
        if type(prefill) is not bool:
            raise ValueError("private MHC prefill selection must be a boolean")
        if type(packet_rows) is not int or packet_rows not in (8, self.PACKET_ROWS):
            raise ValueError(f"private MHC packet rows are 8 (the control) or {self.PACKET_ROWS}")
        self.prefill_enabled = prefill
        self.packet_rows = packet_rows
        self.ext = extension()
        self.hidden, self.hc, self.nout, nchunk = geometry()
        device = next(iter(weights.values())).device
        self.weights = {}
        self.executed = set()
        for key, fn in weights.items():
            if fn.shape != (self.nout, self.hc * self.hidden) or fn.dtype != torch.float32 or not fn.is_contiguous():
                raise ValueError(f"MK MHC requires FP32 [{self.nout},{self.hc}*{self.hidden}] weights")
            bf16 = fn.bfloat16()
            # BF16 storage is lossless only for BF16-origin checkpoint values.
            packed = (bf16.reshape(self.nout, self.hc, self.hidden).transpose(1, 2).contiguous()
                      if torch.equal(fn, bf16.float()) else None)
            self.weights[key] = fn, packed
        self.workspace = [torch.zeros(size, device=device, dtype=dtype)
                          for size, dtype in workspace_sizes(self.hidden, self.hc, self.nout, nchunk)]

    def prefill(self,key,x,res,post,comb,scale,base,norm,eps,hc_eps,post_mult,sinkhorn):
        """Reuse the already-proven-lossless pack; unsupported weights stay on the original lane."""
        if not self.prefill_enabled:
            return None
        packed = self.weights[key][1]
        if packed is None or not 64 < x.shape[0] <= 32768:
            return None
        from engine.kernels.prefill_mhc import post_pre
        return post_pre(x,res,post,comb,packed,scale,base,norm,eps,hc_eps,post_mult,sinkhorn)

    def __call__(self,key,x,res,post,comb,scale,base,norm,eps,hc_eps,post_mult,sinkhorn,*,packets=None):
        n = x.shape[0]
        if not 1 <= n <= 64 or res.shape != (n, self.hc, self.hidden):
            raise ValueError("MK MHC decode geometry mismatch")
        # The BF16 pack is the consumer kernels' [output, hidden, stream] layout; the ordinary persistent grid reads
        # FP32 only. An all-reduce consumer takes up to 8 rows; a packet consumer takes the pack up to packet_rows and,
        # above 8 rows, expands each block's coefficients once instead of at every multiply.
        small = n <= 8
        fp32, packed = self.weights[key]
        rows = 8 if packets is None else self.packet_rows
        weight = packed if n <= rows and packed is not None else fp32
        rc = torch.empty_like(res)
        pm = torch.empty((n, self.hc, 1), device=x.device, dtype=torch.float32)
        cm = torch.empty((n, self.hc, self.hc), device=x.device, dtype=torch.float32)
        li = torch.empty_like(x)
        tensors = [x,res,post,comb,weight,scale,base,norm,rc,pm,cm,li,*self.workspace]
        args = ([t.data_ptr() for t in tensors], [eps,hc_eps,hc_eps,post_mult,eps], [n, sinkhorn, self.hidden])
        if packets is None:
            self.ext.run_mhc(*args,weight is packed,small)
        else:
            if (packets.device != x.device or packets.dtype != torch.int64 or
                    packets.shape != (4,) or not packets.is_contiguous()):
                raise ValueError("MHC needs a same-device contiguous int64[4] rank descriptor")
            self.ext.run_mhc_packets(*args,packets,weight is packed,weight is packed and not small)
        self.executed.add(key)
        return rc,pm,cm,li


def geometry_v41(shape=None) -> "tuple[int, int, int, int]":
    """(hidden, hc, nout, nchunk) for the megakernel's V4.1 seam; a shape it does not serve is refused by the rule the
    wizard's table uses (cells.mhc_v41_refusal)."""
    if shape is None:
        from engine.base.kernel_shape import bound
        shape = bound()
    why = mhc_v41_refusal(shape)
    if why is not None:
        raise ValueError(f"MK MHC V4.1: {why}")
    return shape.hidden, shape.hc, shape.hc * (2 + shape.hc), shape.hidden // HCHUNK


def pieces(tokens: int) -> "list[tuple[int, int]]":
    """[start, stop) token ranges of at most MHC_MAX_TOK covering a prefill, in order."""
    if type(tokens) is not int or tokens <= 0:
        raise ValueError(f"a prefill has a positive token count, not {tokens!r}")
    return [(start, min(tokens, start + MHC_MAX_TOK)) for start in range(0, tokens, MHC_MAX_TOK)]


class MHCV41:
    """The megakernel's V4.1 seam (run_mhc_v41) mixing DeepSeek-V4.1's split-sinkhorn hyper-connection: glue that puts
    that form on the MK mHC segment's compiled widths (cells.GLUE).

    One call mixes the previous sublayer's output into the residual with that sublayer's post and comb, projects this
    sublayer's split-sinkhorn mixes from the result, collapses the layer input with the PREVIOUS sublayer's pre, and
    hands this sublayer's pre to the next call. Its torch form is probes/mk_mhc_geometry_bench.py
    v41_component_reference, the released HF V4.1 seam; the GPU probe has not run (measurements/dsv41_mhc_20260910).

    The seam mixes each token alone, so `prefill` runs any token count in MHC_MAX_TOK-token pieces, exactly: contiguous
    row slices of the inputs and of the preallocated outputs, one launch per piece. Coefficient packs and scratch live
    with the model across captured graphs; calls are ordered on the execution stream (the seam shares device ticket
    counters with every MHC launch).
    """

    def __init__(self, weights, *, ext=None):
        self.hidden, self.hc, self.nout, nchunk = geometry_v41()
        self.ext = extension() if ext is None else ext      # `ext`: a stand-in for the native module, CPU tests only
        device = next(iter(weights.values())).device
        self.weights = {}
        self.executed = set()
        for key, fn in weights.items():
            if fn.shape != (self.nout, self.hc * self.hidden) or fn.dtype != torch.float32 or not fn.is_contiguous():
                raise ValueError(f"MK MHC V4.1 requires contiguous FP32 [{self.nout},{self.hc}*{self.hidden}] weights")
            self.weights[key] = fn
        self.workspace = [torch.zeros(size, device=device, dtype=dtype)
                          for size, dtype in workspace_sizes(self.hidden, self.hc, self.nout, nchunk)]

    def __call__(self, key, x, res, post, comb, scale, base, norm, pre, rms_eps, pre_eps, sinkhorn_eps, post_mult,
                 norm_eps, sinkhorn):
        """A decode or verify step of 1..MHC_MAX_TOK tokens: x [T,hidden] bf16 (the previous sublayer's output), res
        [T,hc,hidden] bf16, post [T,hc] f32 and comb [T,hc,hc] or [T,hc*hc] f32 (the previous sublayer's), scale [3],
        base [nout] f32, norm [hidden] bf16, pre [T,hc] f32 (the previous sublayer's) -> (residual [T,hc,hidden] bf16,
        post [T,hc] f32, comb [T,hc*hc] f32, layer input [T,hidden] bf16, pre [T,hc] f32 for the next sublayer)."""
        n = x.shape[0]
        if not 1 <= n <= MHC_MAX_TOK:
            raise ValueError(f"MK MHC V4.1 takes 1..{MHC_MAX_TOK} tokens per launch; `prefill` runs more in pieces")
        outs = self._outputs(n, x.device)
        self._launch(key, (x, res, post, comb, pre), (scale, base, norm), outs,
                     (rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps), sinkhorn)
        return outs

    def prefill(self, key, x, res, post, comb, scale, base, norm, pre, rms_eps, pre_eps, sinkhorn_eps, post_mult,
                norm_eps, sinkhorn):
        """`__call__`'s contract for any token count, in MHC_MAX_TOK-token pieces."""
        n = x.shape[0]
        outs = self._outputs(n, x.device)
        for lo, hi in pieces(n):
            self._launch(key, tuple(t[lo:hi] for t in (x, res, post, comb, pre)), (scale, base, norm),
                         tuple(o[lo:hi] for o in outs), (rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps), sinkhorn)
        return outs

    def _outputs(self, n, device):
        return (torch.empty((n, self.hc, self.hidden), device=device, dtype=torch.bfloat16),
                torch.empty((n, self.hc), device=device, dtype=torch.float32),
                torch.empty((n, self.hc * self.hc), device=device, dtype=torch.float32),
                torch.empty((n, self.hidden), device=device, dtype=torch.bfloat16),
                torch.empty((n, self.hc), device=device, dtype=torch.float32))

    def _check(self, rows, coefficients, fn, scalars, sinkhorn):
        """Raw pointers cannot check bounds, dtype or placement: everything the kernel reads, before it reads it."""
        x, res, post, comb, pre = rows
        scale, base, norm = coefficients
        n, hc, hidden = x.shape[0], self.hc, self.hidden
        expected = ((x, (n, hidden), torch.bfloat16), (res, (n, hc, hidden), torch.bfloat16),
                    (post, (n, hc), torch.float32), (fn, (self.nout, hc * hidden), torch.float32),
                    (scale, (3,), torch.float32), (base, (self.nout,), torch.float32), (norm, (hidden,), torch.bfloat16),
                    (pre, (n, hc), torch.float32),
                    (comb, (n, hc, hc) if comb.ndim == 3 else (n, hc * hc), torch.float32))
        for tensor, shape, dtype in expected:
            if (tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != x.device
                    or not tensor.is_cuda or not tensor.is_contiguous()):
                raise ValueError("MK MHC V4.1 requires matching contiguous CUDA tensors "
                                 f"(expected {tuple(shape)} {dtype}, got {tuple(tensor.shape)} {tensor.dtype} on {tensor.device})")
        if x.device.index != torch.cuda.current_device():
            raise ValueError("select the MHC input device before launching on its current stream")
        rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps = map(float, scalars)
        if (not all(math.isfinite(v) for v in (rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps))
                or min(rms_eps, sinkhorn_eps, norm_eps) <= 0 or min(pre_eps, post_mult) < 0
                or type(sinkhorn) is not int or not 1 <= sinkhorn <= 64):
            raise ValueError("MK MHC V4.1: finite epsilons (RMS, sinkhorn and norm positive, pre and post nonnegative) "
                             "and 1..64 sinkhorn repeats")

    def _launch(self, key, rows, coefficients, outs, scalars, sinkhorn):
        fn = self.weights[key]
        self._check(rows, coefficients, fn, scalars, sinkhorn)
        x, res, post, comb, pre = rows
        scale, base, norm = coefficients
        residual, post_out, comb_out, layer_input, pre_out = outs
        # kernels.cu mk_run_mhc_v41: the legacy twelve, the six scratch words, then the V4.1 pair
        tensors = [x, res, post, comb, fn, scale, base, norm, residual, post_out, comb_out, layer_input,
                   *self.workspace, pre, pre_out]
        self.ext.run_mhc_v41([t.data_ptr() for t in tensors], [float(v) for v in scalars], [x.shape[0], sinkhorn],
                             self.hidden)
        self.executed.add(key)
