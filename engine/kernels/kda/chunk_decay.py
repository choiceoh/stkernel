"""Chunked prefill of the gated delta rule on a decay computed outside the kernel -- GDN's (glue, cells.GLUE).

`chunk_kda_with_fused_gate` (kda.py) computes KDA's per-channel gate inside the chunk pipeline. GDN's recurrence is the
same with one log-decay per head (engine/modules/linear_attention), so the pipeline serves it once the decay is in the
form the pipeline reads: summed within each FLA_CHUNK_SIZE chunk, scaled by RCP_LN2 for the exp2 kernels -- the
invariant `_chunk_kda_fwd_with_cumulative_g` names. A per-head decay is summed per head (the scalar cumsum kernel) and
handed over as it is, [B,T,HV]: the pipeline's kernels read it as its widened form, one value a head with stride 0
along the key channels (kda.py G_HEAD), which is the sum of the widened decay. From there the arithmetic is
chunk_kda_with_fused_gate's, `out` and `states_at` included.

A model with fewer key heads than value heads hands its normalised queries and keys over as they are, [B,T,H,K]: value
head i reads key head i // (HV // H) (kda.py QG), the grouping the recurrent kernel reads (fused_recurrent.py: i_h =
i_hv // (HV // H)).

Until carry K3 this entry built both widened forms first -- the decay expanded to [B,T,HV,K] fp32 and q, k repeated to
HV heads: at Qwen3.8's per-rank cell (4 key and 12 value heads of 128) and a 32,256-token chunk that is 189 MiB of
decay and 2 x 94.5 MiB of q and k written, and read back by four kernels, a layer -- about 13 GiB of writes over the
36 GDN layers of one prefill chunk. `widen=True` still builds them: the same values at other strides, so the two forms
are byte for byte the same output, and the tests hold them to that.
"""
import torch

from .cumsum import chunk_local_cumsum
from .index import prepare_chunk_indices
from .kda import RCP_LN2, _chunk_kda_fwd_with_cumulative_g, _glm53_qk_l2norm_strided, _validate_chunk_kda_output
from .l2norm import l2norm_fwd
from .utils import FLA_CHUNK_SIZE


def chunk_kda_with_decay(q, k, v, decay, beta, scale=None, initial_state=None, output_final_state=False,
                         use_qk_l2norm_in_kernel=False, cu_seqlens=None, out=None, states_at=None, widen=False):
    """`chunk_kda_with_fused_gate` with the log-decay itself in place of (raw_g, A_log, g_bias, safe_gate, lower_bound).

    q, k [B,T,H,K]; v [B,T,HV,V], HV a multiple of H; `decay` the natural-log decay (<= 0) per head [B,T,HV] or per
    channel [B,T,HV,K]; beta [B,T,HV] after its sigmoid, as the fused entry takes it; `initial_state` [N,HV,V,K] fp32
    in the kernel's [V,K] layout (the lane transposes the engine's [K,V], as it does for the fused entry) or None.
    Returns (o [B,T,HV,V], the final state [N,HV,V,K] fp32 or None) and, with `states_at` (ascending FLA_CHUNK_SIZE-chunk
    indices, one sequence), the fp32 states at those chunk starts [len(states_at),HV,V,K] -- the kernel's layout, as
    the fused entry. Without `out` the output is written over a contiguous `v`'s storage, as the fused entry (and FLA's
    chunk_kda) does. `widen`: build the widened decay and the repeated q, k the kernels read natively (the module
    docstring) -- for the comparison only."""
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("chunk decay KDA takes q, k [B,T,H,K] and v [B,T,HV,V]")
    b, t, h, kd = k.shape
    hv = v.shape[2]
    if hv % h:
        raise ValueError(f"{hv} value heads are not a multiple of {h} key heads")
    if tuple(decay.shape) not in ((b, t, hv), (b, t, hv, kd)):
        raise ValueError("the decay is per head [B,T,HV] or per channel [B,T,HV,K]")
    if tuple(beta.shape) != (b, t, hv):
        raise ValueError("beta is one value per value head [B,T,HV]")
    # Validate before contiguous/l2norm copies can hide an input alias.
    _validate_chunk_kda_output(out, v, (q, k, v, decay, beta, initial_state, cu_seqlens))
    if scale is None:
        scale = kd ** -0.5
    if use_qk_l2norm_in_kernel:
        normalized = _glm53_qk_l2norm_strided(q, k)
        if normalized is None:
            q = l2norm_fwd(q.contiguous())
            k = l2norm_fwd(k.contiguous())
        else:
            q, k = normalized
    if widen and hv != h:
        q = q.repeat_interleave(hv // h, dim=2)
        k = k.repeat_interleave(hv // h, dim=2)
    chunk_indices = prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE) if cu_seqlens is not None else None
    g = chunk_local_cumsum(decay, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    if widen and g.ndim == 3:
        g = g.unsqueeze(-1).expand(b, t, hv, kd).contiguous()
    g.mul_(RCP_LN2)                              # exp2(x / ln 2) == exp(x), as chunk_kda_fwd scales it
    states_out = None
    if states_at:
        n_seq = (len(cu_seqlens) - 1) if cu_seqlens is not None else b
        if n_seq != 1:
            raise ValueError("states_at is served for one sequence per call")
        states_out = torch.empty(1, len(states_at), hv, v.shape[-1], kd, dtype=torch.float32, device=q.device)
    o, final_state = _chunk_kda_fwd_with_cumulative_g(
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), g=g, beta=beta.contiguous(), scale=scale,
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE, out=out, states_at=list(states_at) if states_at else None, states_out=states_out)
    if states_out is not None:
        return o, final_state, states_out[0]
    return o, final_state


__all__ = ["chunk_kda_with_decay"]
