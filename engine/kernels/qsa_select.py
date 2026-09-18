"""A decode step's QSA block selection in one launch: each row's `k` best blocks among its visible ones.

engine/kernels/qsa.select_blocks took torch.topk where the prefill radix select (engine/kernels/prefill_topk: eager,
k 512, 65+ rows) does not admit the step -- every captured step. Around it sat the mask and the unpacking: arange, a
compare, a not, masked_fill, the top-k's own launches, a compare, a cast, where, the copy into `out` -- a dozen and
more launches a QSA layer, thirteen layers a step (carry Q7; on a GB10 a launch inside a captured graph has a fixed
cost the campaign's ledger puts at 15-25 us, MEASUREMENTS_ARCHIVE.md).

One program a row does it all:

    the k-th value   found by value, not by sorting: a float's bits are mapped to an unsigned key in the floats' order
                     (negatives inverted below the positives), columns past the row's visible blocks take the lowest, and 32
                     counting passes -- one a bit, from the top -- leave the largest threshold that still has `k` keys
                     at or above it.
    ties             keys above the threshold are in; of the keys equal to it, the lowest blocks first until `k` are
                     chosen. That is prefill_topk's rule and GLM's st_dsa_select's ("ties to the lower block"), so a
                     decode step and a prefill step now choose the same set from the same scores. torch.topk ordered
                     equal scores as its candidates fell, and relu leaves many equal zeros just past the budget's reach.
    the ids          written ascending -- the order the block attention sorts any selection into before it reads it
                     (carry Q6) -- by a prefix sum over the chosen columns, -1 after a row's min(visible, k) picks. The
                     two stores write disjoint slots.

A program is one block of threads, so the 32 passes over a row stop paying where the row outgrows them: measured on an
RTX 5050 (sm_120, triton 3.6, a captured graph's replay, 2 rows, k 512; 2026-09-19, not a GB10 number) the launch
against the torch form was 8 against 30 us at 1,024 columns, 13 against 36 at 4,096, 20 against 65 at 16,384, 38
against 53 at 32,768 -- and 231 against 59 at 65,536. WIDEST keeps the last bucket (contexts past 131K tokens) on the
torch form; where the two cross on a GB10, whose launches cost more, is a lane measurement (carry Q9).

What differs from the torch form it replaces: the ORDER of a row's picks (ascending block, not descending score; only
the set is read downstream), which of several equal scores straddling the k-th place are taken, and a visible block
scored -inf is an ordinary candidate rather than -1 (the scorer's logits are relu sums, never -inf).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _select_rows(Logits, Visible, Out, sL, sO, COLUMNS, K: tl.constexpr, KB: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    visible = tl.minimum(tl.load(Visible + r), COLUMNS)
    live = c < visible
    x = tl.load(Logits + r * sL + c, mask=c < COLUMNS, other=0.0)
    bits = x.to(tl.int32, bitcast=True)
    # the floats' order as SIGNED int32 keys: a non-negative float's bits already are; a negative's magnitude bits are
    # inverted and its sign bit kept, so a larger magnitude is a smaller key. INT32_MIN is below -inf's key (-2^31 +
    # 0x007FFFFF): a column past the visible blocks is never chosen while a block is live
    key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key = tl.where(live, key, -2147483648)
    want = tl.minimum(visible, K)
    # the largest threshold with `want` keys at or above it: one counting pass a bit, from the top, over the keys as
    # unsigned numbers (a signed key plus 2^31) -- the scalar search runs in int64, the rows compare in int32
    threshold = tl.zeros([], dtype=tl.int64)
    one = tl.full([], 1, dtype=tl.int64)
    for b in range(32):                                            # a loop, not 32 unrolled reductions: a bucket compiles once
        candidate = threshold + (one << (31 - b))
        enough = tl.sum((key >= (candidate - 2147483648).to(tl.int32)).to(tl.int32), 0) >= want
        threshold = tl.where(enough, candidate, threshold)
    kth = (threshold - 2147483648).to(tl.int32)
    above = key > kth
    tied = (key == kth) & live
    room = want - tl.sum(above.to(tl.int32), 0)                     # how many of the tied blocks fit, lowest first
    chosen = (above | (tied & (tl.cumsum(tied.to(tl.int32), 0) <= room))) & live
    slot = tl.cumsum(chosen.to(tl.int32), 0) - 1
    tl.store(Out + r * sO + slot, c.to(tl.int32), mask=chosen)
    p = tl.arange(0, KB)
    tl.store(Out + r * sO + p, tl.full([KB], -1, dtype=tl.int32), mask=(p >= want) & (p < K))


WIDEST = 32768                  # columns one program's threads still beat the torch form at (the module docstring)


def admits(logits: torch.Tensor, block_topk: int) -> bool:
    """Whether `select` serves these logits: CUDA FP32 rows with packed columns, no wider than one program pays for."""
    return (logits.is_cuda and logits.dtype == torch.float32 and logits.ndim == 2 and logits.stride(1) == 1
            and 0 < logits.shape[1] <= WIDEST and 0 < block_topk <= (1 << 16))


def select(logits: torch.Tensor, visible_blocks: torch.Tensor, block_topk: int, out: torch.Tensor) -> torch.Tensor:
    """out int32 [rows, block_topk]: each row's min(visible, block_topk) best blocks among its first `visible_blocks`
    columns, ascending, ties to the lower block, -1 after them. Columns past a row's visible blocks are not read for
    their values (the scorer never writes them)."""
    rows, columns = logits.shape
    if not admits(logits, block_topk):
        raise ValueError("the decode selection takes CUDA FP32 logits [rows, columns] with packed columns")
    if (visible_blocks.shape != (rows,) or visible_blocks.dtype != torch.int32 or visible_blocks.device != logits.device
            or not visible_blocks.is_contiguous()):
        raise ValueError("visible_blocks is a packed int32 [rows] on the logits' device")
    if (out.shape != (rows, block_topk) or out.dtype != torch.int32 or out.device != logits.device
            or out.stride(1) != 1):
        raise ValueError("block selection writes int32 [rows, block_topk] with packed columns")
    if rows:
        block = triton.next_power_of_2(columns)
        _select_rows[(rows,)](logits, visible_blocks, out, logits.stride(0), out.stride(0), columns, K=block_topk,
                              KB=triton.next_power_of_2(block_topk), BLOCK=block,
                              num_warps=8 if block >= 4096 else 4)
    return out


__all__ = ["WIDEST", "admits", "select"]
