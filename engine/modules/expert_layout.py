"""Expert weight storage layouts (module layer: no kernel import).

The served b12x static lane may re-lay packed expert weights out tile-major IN
PLACE at bind time (`engine.kernels.b12x.moe_dispatch.tile_expert_weights_inplace`,
spec cell `t`): the tensor keeps its logical shape [E, rows, K/2] bytes, but its
bytes become [E, K/K_IN, rows, K_IN/2] so every TMA box is one contiguous run.
The marker attribute below is what that routine sets; the reference lane and
the judge read the same arena tensors, so they must see row-major bytes again.
"""
import torch

TILE_MAJOR_ATTR = "_b12x_tile_major"   # moe_dispatch._TILE_MAJOR_ATTR: False / "plain"
W13_K_IN_BYTES = 256                    # moe_static_kernel_v5.TILED_W13_K_IN // 2 (512 fp4 per k tile)
W2_K_IN_BYTES = 64                      # moe_static_kernel_v5.TILED_W2_K_IN // 2 (128 fp4 per k tile)


def is_tile_major(w: torch.Tensor) -> bool:
    return bool(getattr(w, TILE_MAJOR_ATTR, False))


def row_major_expert(w: torch.Tensor, e: int, k_in_bytes: int) -> torch.Tensor:
    """Expert `e` of a packed [E, rows, K/2] byte tensor as row-major [rows, K/2].

    Row-major storage: a view. Tile-major storage (marker set): one transient
    copy of that expert (3 MB for GLM's w13, 1 MB for w2) -- the reference lane
    is held to fidelity, not speed."""
    if not is_tile_major(w):
        return w[e]
    rows, kb = w.shape[1], w.shape[2]
    if kb % k_in_bytes:
        raise ValueError(f"tile-major expert weights: K/2 = {kb} B is not a multiple of {k_in_bytes}")
    tiled = w[e].view(kb // k_in_bytes, rows, k_in_bytes)      # bytes as [K tiles, rows, K_IN]
    return tiled.permute(1, 0, 2).reshape(rows, kb)


def _selfcheck() -> None:
    torch.manual_seed(0)
    E, rows, kb = 3, 8, 512
    w = torch.randint(0, 256, (E, rows, kb), dtype=torch.uint8)
    # the in-place relayout's byte order (moe_dispatch._tile_expert_weights), applied to a copy
    tiled = w.reshape(E, rows, kb // W13_K_IN_BYTES, W13_K_IN_BYTES).permute(0, 2, 1, 3).contiguous().view(E, rows, kb)
    setattr(tiled, TILE_MAJOR_ATTR, "plain")
    for e in range(E):
        assert torch.equal(row_major_expert(tiled, e, W13_K_IN_BYTES), w[e])
        assert row_major_expert(w, e, W13_K_IN_BYTES).data_ptr() == w[e].data_ptr()
    print("  expert_layout: tile-major bytes read back row-major per expert OK")


if __name__ == "__main__":
    _selfcheck()
