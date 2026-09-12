"""Move prepared immutable tensors into a retired arena weight region.

The caller explicitly retires the BF16/raw-scale reader before serving.
This is preparation, never graph-time compaction. All destinations are
validated before the first write, and each view retains the arena storage.
"""
import torch


def consume(storage, tensors):
    if not storage.is_contiguous() or storage.device.type != 'cuda':
        raise ValueError('packed storage requires a contiguous CUDA arena region')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('packed storage must be finalized before capture')
    raw=storage.view(torch.uint8).reshape(-1)
    plans=[]
    end=0
    for tensor in tensors:
        if not tensor.is_contiguous() or tensor.device!=storage.device:
            raise ValueError('prepared packs must be contiguous on the arena device')
        start=(end+255)//256*256
        end=start+tensor.numel()*tensor.element_size()
        if end>raw.numel():
            raise ValueError('prepared packs exceed the retired arena region')
        # Independent source allocations are required; no partially
        # overlapping move can overwrite a later pack before it is copied.
        lo,hi=tensor.data_ptr(),tensor.data_ptr()+tensor.numel()*tensor.element_size()
        if lo<raw.data_ptr()+raw.numel() and hi>raw.data_ptr():
            raise ValueError('packed-storage sources overlap their destination')
        plans.append((start,end,tensor))
    out=[]
    for start,end,tensor in plans:
        view=raw[start:end].view(tensor.dtype).view(tensor.shape)
        view.copy_(tensor)
        out.append(view)
    return tuple(out)
