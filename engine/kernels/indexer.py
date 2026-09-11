"""Integer-only finalization of sparse indexer positions into latent slots."""
import torch
import triton
import triton.language as tl


@triton.jit
def _indexer_slots(tokens, table, out, counts, width: tl.constexpr,
                   token_s0, token_s1, table_s0, out_s0, out_s1, count_s0,
                   block_size: tl.constexpr, block_stride, layer_offset,
                   MAPPED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    positions = tl.load(tokens + row * token_s0 + cols * token_s1,
                        cols < width, other=-1)
    # Positions are already sorted. Mask padding before any block-table read.
    valid = (cols < width) & (positions >= 0)
    if MAPPED:
        page = positions.to(tl.int64) // block_size
        physical = tl.load(table + page * table_s0, valid, other=0)
        slots = physical * block_stride + layer_offset + positions % block_size
    else:
        slots = positions
    tl.store(out + row * out_s0 + cols * out_s1, tl.where(valid, slots, -1), cols < width)
    tl.store(counts + row * count_s0, tl.sum(valid.to(tl.int32), 0))


def indexer_slots(tokens, block_table, block_size, block_stride, layer_offset, out, counts):
    """Sort int32 token positions descending, map and write slots/counts in place.

    Negative positions are padding. Nonnegative positions must belong to the
    reserved block row. A None table means the identity map used by chain
    checks. Input and output views may be strided; they must not overlap.
    """
    assert tokens.ndim == 2 and tokens.dtype == torch.int32
    assert out.shape == tokens.shape and out.dtype == torch.int32
    assert counts.shape == (tokens.shape[0],) and counts.dtype == torch.int32
    assert block_size > 0
    if block_table is not None:
        assert block_table.ndim == 1 and block_table.dtype == torch.int32
    rows, width = tokens.shape
    if rows == 0:
        return
    # Keep PyTorch's integer sort: a single-program bitonic sort over the
    # padded 4096-column vector costs more than the launches it eliminates.
    # Only the subsequent count/map/mask/write chain is fused.
    tokens = tokens.sort(dim=1, descending=True).values
    size = triton.next_power_of_2(max(1, width))
    _indexer_slots[(rows,)](
        tokens, block_table if block_table is not None else tokens, out, counts, width,
        *tokens.stride(), block_table.stride(0) if block_table is not None else 0,
        *out.stride(), counts.stride(0), block_size, block_stride, layer_offset,
        MAPPED=block_table is not None, BLOCK=size, num_warps=8 if size > 1024 else 4)
