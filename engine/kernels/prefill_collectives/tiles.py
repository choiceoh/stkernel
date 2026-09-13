"""Two-slot rank-major prefill input projection; eager only, the owner's world and row width."""
from collections import deque


def tile_pipeline(rows, tile_rows, launch, consume):
    """Reuse a slot only after its prior consumer has submitted a release."""
    if type(rows) is not int or rows <= 0 or type(tile_rows) is not int or tile_rows <= 0:
        raise ValueError("positive row and tile sizes required")
    pending = deque()
    for number, start in enumerate(range(0, rows, tile_rows)):
        if len(pending) == 2:
            consume(*pending.popleft())
        end = min(rows, start + tile_rows)
        pending.append((start, end, launch(start, end, number % 2)))
    while pending:
        consume(*pending.popleft())


class TiledProjection:
    TILE_ROWS = 256

    def __init__(self, owner):
        import torch
        self.owner = owner
        self.stream = torch.cuda.Stream()

    def __call__(self, x, project, *, packet_project=None):
        import torch
        import torch.distributed as dist
        from engine.kernels.prefill_collectives import BLOCK, FP8_MIN_ROWS
        from .kernels import _pack_rs_payload, _unpack_gather
        owner = self.owner
        world, hidden = owner.world, owner.hidden
        owner.check(x)
        pieces = min(4, x.shape[0] // self.TILE_ROWS)
        if pieces <= 1:
            return project(owner.all_gather(x))
        # At most four messages even after the main profile's 32K chunk
        # increase. Balanced, 32-aligned tiles also avoid a tiny tail falling
        # into the <=32-row W4 decode GEMM instead of the FP8 prefill GEMM.
        tile_rows = ((x.shape[0] + pieces * 32 - 1) // (pieces * 32)) * 32
        # This decision belongs to the original full operation, not each tile.
        fp8 = x.shape[0] * world >= FP8_MIN_ROWS
        fused = fp8 and packet_project is not None and (world, hidden) == (4, 4096)
        parent = torch.cuda.current_stream(x.device)
        self.stream.wait_stream(parent)
        x.record_stream(self.stream)
        limit = tile_rows * hidden
        packet_limit = ((limit + 4 * (limit // BLOCK) + 127) // 128) * 128     # 4: the FP32 scale of every block
        slots = []
        with torch.cuda.stream(self.stream):
            for _ in range(2):
                slots.append(dict(
                    payload=torch.empty(packet_limit, device=x.device, dtype=torch.uint8) if fp8 else None,
                    received=torch.empty(packet_limit * world, device=x.device, dtype=torch.uint8) if fp8 else None,
                    value=None if fused else torch.empty((tile_rows * world, hidden), device=x.device, dtype=x.dtype),
                    ready=torch.cuda.Event(), released=torch.cuda.Event(), used=False))
        output = None

        def launch(start, end, slot_id):
            slot = slots[slot_id]
            rows = end - start
            with torch.cuda.stream(self.stream):
                if slot["used"]:
                    self.stream.wait_event(slot["released"])
                value = None if fused else slot["value"][:rows * world]
                source = x[start:end]
                if fp8:
                    local = rows * hidden
                    stride = ((local + 4 * (local // BLOCK) + 127) // 128) * 128
                    payload, received = slot["payload"][:stride], slot["received"][:stride * world]
                    _pack_rs_payload[(local // BLOCK,)](
                        source, payload.view(torch.float8_e4m3fn), payload.view(torch.float32),
                        local, local, stride, BLOCK=BLOCK)
                    work = dist.all_gather_into_tensor(received, payload, group=owner.comm.group, async_op=True)
                    work.wait()  # orders the current CUDA stream, not a host polling loop
                    if fused:
                        value = received
                    else:
                        _unpack_gather[(local * world // BLOCK,)](
                            received.view(torch.float8_e4m3fn), received.view(torch.float32), value,
                            local, stride, BLOCK=BLOCK)
                else:
                    work = dist.all_gather_into_tensor(value, source, group=owner.comm.group, async_op=True)
                    work.wait()
                slot["ready"].record()
            return slot, value, work

        def consume(start, end, pending):
            nonlocal output
            slot, value, work = pending
            parent.wait_event(slot["ready"])
            value.record_stream(parent)
            projected = packet_project(value, end-start) if fused else project(value)
            if projected.ndim != 2 or projected.shape[0] != (end-start)*world:
                raise ValueError("tile projection must preserve rows")
            if output is None:
                output = torch.empty((world, x.shape[0], projected.shape[1]),
                                     device=projected.device, dtype=projected.dtype)
            # Gather orders ranks within each tile; restore the original
            # rank-major token order before any recurrent kernel sees it.
            output[:, start:end].copy_(projected.view(world, end-start, -1))
            slot["released"].record(parent)
            slot["used"] = True

        try:
            tile_pipeline(x.shape[0], tile_rows, launch, consume)
        finally:
            # Includes exceptions: queued transfers cannot outlive their input
            # owner on the caller stream. No fallback after an issued collective.
            parent.wait_stream(self.stream)
        owner.executed.add("fp8_tiled_projection" if fp8 else "bf16_tiled_projection")
        if fused:
            owner.executed.add("fp8_packet_projection")
        return output.flatten(0, 1)
