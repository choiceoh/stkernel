"""Rank-agreed stop policy for the serving burst's pre-reserved decode steps."""
import torch


def stop_at_boundary(before, current, alive, reserved_end, bucket_end, interrupted,
                     *, step_tokens, block):
    """Return a local int64[1] vote; TP4 MAX must precede conditional branching.

    First iteration eligibility is checked by the caller before submission.
    Interrupts supplied here must be ordered device inputs; a concurrent raw
    CPU write to ordinary CUDA memory is not a supported cancellation API.
    """
    if type(step_tokens) is not int or step_tokens <= 0 or type(block) is not int or block <= step_tokens:
        raise ValueError("a positive decode width must fit within one prefix block")
    if before.ndim != 1 or not 1 <= before.numel() <= 4:
        raise ValueError("bounded decode supports one to four rows")
    for t in (current, alive, reserved_end):
        if t.shape != before.shape or t.device != before.device:
            raise ValueError("bounded decode controls must preserve row identity")
    if any(t.dtype != torch.int64 for t in (before, current, reserved_end)) or alive.dtype != torch.bool:
        raise ValueError("contexts/reservations must be int64 and alive must be bool")
    if (interrupted.device != before.device or interrupted.dtype != torch.int64
            or interrupted.shape != (1,) or type(bucket_end) is not int or bucket_end <= 0):
        raise ValueError("bounded decode needs an ordered int64[1] interrupt and fixed context bucket")
    stop = ((~alive).any() | ((before // block) != (current // block)).any()
            | (current + step_tokens > reserved_end).any()
            | (current + step_tokens > bucket_end).any() | (interrupted != 0).any())
    return stop.to(torch.int64).reshape(1)


def agree_stop(comm, vote):
    """No rank may leave a GPU loop while a peer starts the next collective."""
    if vote.shape != (1,) or vote.dtype != torch.int64:
        raise ValueError("bounded stop agreement requires one int64 vote")
    if comm.world_size != 4 or comm.transport is None:
        raise ValueError("bounded TP4 loops require the owned one-shot transport")
    if not comm.transport.eligible_max(vote):
        raise ValueError("bounded stop vote requires the native int64 packet")
    return comm.all_reduce_max(vote)
