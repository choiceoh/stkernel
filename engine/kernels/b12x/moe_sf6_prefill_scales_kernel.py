"""SF6-v1 bytes to the original scale plane, one CTA per 2048-byte stage."""
import triton
import triton.language as tl


@triton.jit
def expand(Packed, Out, K_TILES: tl.constexpr, FC2: tl.constexpr):
    stage = tl.program_id(0).to(tl.int64)
    byte = tl.arange(0, 2048)
    base = tl.load(Packed + stage * 1552 + 1536).to(tl.int32)
    low = (tl.load(Packed + stage * 1552 + byte // 2).to(tl.int32)
           >> ((byte % 2) * 4)) & 15
    high = (tl.load(Packed + stage * 1552 + 1024 + byte // 4).to(tl.int32)
            >> ((byte % 4) * 2)) & 3
    code = (base + low + (high << 4)).to(tl.uint8)
    if FC2:
        # Stored stages interleave two N128 tiles inside each K64 group;
        # the ordinary scale descriptor places each N128 tile contiguously.
        row_pair, kt = stage // K_TILES, stage % K_TILES
        k64, rest = byte // 1024, byte % 1024
        row_half, inner = rest // 512, rest % 512
        destination = ((row_pair * 2 + row_half) * K_TILES + kt) * 1024 + k64 * 512 + inner
    else:
        destination = stage * 2048 + byte
    tl.store(Out + destination, code)
