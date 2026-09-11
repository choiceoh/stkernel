"""Record the actual loaded GB10 kernel's launch resources (no timing)."""
import json
import torch
import triton
from engine.kernels.indexer import _pool_slots
ids = torch.arange(512, device='cuda', dtype=torch.int32).reshape(1, 512)
lengths = torch.tensor([2051], device='cuda', dtype=torch.int32)
table = torch.arange(64, device='cuda', dtype=torch.int32)
out = torch.empty((1, 2051), device='cuda', dtype=torch.int32)
counts = torch.empty(1, device='cuda', dtype=torch.int32)
kernel = _pool_slots[(1,)](ids, lengths, table, out, counts, 512,
    *ids.stride(), lengths.stride(0), table.stride(0), *out.stride(), counts.stride(0),
    64, 2112, 512, POOL=4, MAPPED=True, BLOCK=512, num_warps=4)
torch.cuda.synchronize()
print(json.dumps(dict(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
    triton=triton.__version__, registers_per_thread=kernel.n_regs, spills=kernel.n_spills,
    shared_bytes=kernel.metadata.shared, warps=kernel.metadata.num_warps,
    global_scratch_bytes=kernel.metadata.global_scratch_size), indent=2))
