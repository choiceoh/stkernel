"""Read GB10 driver attributes without creating a CUDA context or allocating memory."""

import datetime
import json
import platform

from cuda.bindings import driver as cuda


def checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(str(status))
    return values[0] if len(values) == 1 else values


checked(cuda.cuInit(0))
device = checked(cuda.cuDeviceGet(0))
names = (
    "COMPUTE_CAPABILITY_MAJOR COMPUTE_CAPABILITY_MINOR MULTIPROCESSOR_COUNT "
    "WARP_SIZE MAX_THREADS_PER_BLOCK MAX_THREADS_PER_MULTIPROCESSOR "
    "MAX_BLOCKS_PER_MULTIPROCESSOR MAX_REGISTERS_PER_MULTIPROCESSOR "
    "MAX_REGISTERS_PER_BLOCK MAX_SHARED_MEMORY_PER_MULTIPROCESSOR "
    "MAX_SHARED_MEMORY_PER_BLOCK MAX_SHARED_MEMORY_PER_BLOCK_OPTIN "
    "RESERVED_SHARED_MEMORY_PER_BLOCK L2_CACHE_SIZE MAX_PERSISTING_L2_CACHE_SIZE "
    "GLOBAL_MEMORY_BUS_WIDTH MEMORY_CLOCK_RATE CLOCK_RATE ASYNC_ENGINE_COUNT "
    "CLUSTER_LAUNCH COOPERATIVE_LAUNCH TENSOR_MAP_ACCESS_SUPPORTED "
    "UNIFIED_ADDRESSING INTEGRATED MANAGED_MEMORY CONCURRENT_MANAGED_ACCESS "
    "PAGEABLE_MEMORY_ACCESS PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES "
    "HOST_NATIVE_ATOMIC_SUPPORTED ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED "
    "DIRECT_MANAGED_MEM_ACCESS_FROM_HOST CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM "
    "GPU_DIRECT_RDMA_SUPPORTED MEMORY_POOLS_SUPPORTED CONCURRENT_KERNELS"
).split()
attributes = {}
for name in names:
    enum = getattr(cuda.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_" + name, None)
    if enum is None:
        attributes[name] = {"unavailable": "attribute absent from installed bindings"}
        continue
    result = cuda.cuDeviceGetAttribute(enum, device)
    attributes[name] = int(result[1]) if int(result[0]) == 0 else {"error": str(result[0])}

print(json.dumps({
    "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "machine": platform.machine(),
    "gpu": checked(cuda.cuDeviceGetName(100, device)).split(b"\0")[0].decode(),
    "driver_api_version": checked(cuda.cuDriverGetVersion()),
    "device_total_bytes": checked(cuda.cuDeviceTotalMem(device)),
    "method": "CUDA driver attribute queries only; no context, allocations or kernel launches",
    "attributes": attributes,
}, indent=2))
