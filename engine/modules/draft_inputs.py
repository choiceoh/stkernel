"""The draft block's anchor/mask IDs and absolute positions, with a CPU reference."""
import torch


def build(anchors, positions, k: int, mask_id: int):
    if (type(k) is not int or k < 0 or type(mask_id) is not int
            or not -(1 << 63) <= mask_id < (1 << 63)
            or anchors.ndim != 1 or anchors.dtype != torch.int64 or anchors.numel() == 0):
        raise ValueError('draft inputs require int64 anchors, a nonnegative K and an int64 mask ID')
    if torch.is_tensor(positions):
        if (positions.ndim > 1 or positions.numel() != anchors.numel() or positions.dtype != torch.int64
                or positions.device != anchors.device):
            raise ValueError('draft positions must be int64 on the anchor device, one per row')
        positions = positions.reshape(-1)
    elif type(positions) is not int or anchors.numel() != 1 or not -(1 << 63) <= positions < (1 << 63):
        raise ValueError('a scalar draft position belongs to exactly one anchor')
    if anchors.is_cuda:
        from engine.kernels.decode_inputs import draft_inputs
        return draft_inputs(anchors, positions, k, mask_id)
    n, t = anchors.numel(), k + 1
    ids = torch.cat((anchors[:, None], torch.full((n, k), mask_id, dtype=torch.int64, device=anchors.device)), 1)
    base = positions[:, None] if torch.is_tensor(positions) else positions
    pos = base + torch.arange(t, dtype=torch.int64, device=anchors.device)
    return ids.flatten(), pos.flatten()
