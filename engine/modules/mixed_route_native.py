"""One fixed-shape C ABI call, with immutable per-invocation result ownership.

Only the tiny host planner is compiled. The source-addressed cache reuses a
binary across worktrees and row counts; no Torch, CUDA or Python extension ABI
is linked. Compilation is part of first preparation, never hidden as warm time.
"""
import ctypes
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile


@lru_cache(maxsize=1)
def planner():
    from engine.kernels.common.native_cache import prepare_sources
    command = shlex.split(os.environ.get('CXX', 'c++'))
    compiler = shutil.which(command[0]) if command else None
    if compiler is None:
        raise RuntimeError('mixed host planning needs the existing C++ toolchain')
    command[0] = compiler
    version = subprocess.check_output([*command, '--version'], text=True, timeout=20)
    flags = ['-std=c++17', '-O3', '-shared', '-fPIC']
    root = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'st'/'mixed-planner'
    _, directory, sources = prepare_sources(root, [Path(__file__).with_name('mixed_route_plan.cpp')],
        dict(compiler=command, version=version, platform=platform.system(), machine=platform.machine(), flags=flags))
    binary = directory/'plan.so'
    with (directory/'.build.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                output = Path(temporary)/'plan.so'
                subprocess.run([*command, *flags, *sources, '-o', str(output)], check=True,
                               capture_output=True, text=True, timeout=90)
                output.replace(binary)
    library = ctypes.CDLL(str(binary))
    fn = library.st_mixed_plan
    ptr = ctypes.POINTER(ctypes.c_int32)
    fn.argtypes = [ptr, ctypes.c_int, ptr, ctypes.c_int, ctypes.c_int] + [ptr]*11
    fn.restype = ctypes.c_int
    fn._library = library
    return fn


def prepare_native(decode, prefill, *, identity, hot_route_quota=128, cold_task_quota=48):
    import numpy as np
    from .mixed_experts import ExpertInvocation, MixedExpertPlan
    from .mixed_completion import ColdExpertPlan
    from .route_table import RouteTable
    if not isinstance(identity, ExpertInvocation):
        raise ValueError('an explicit source/layer/generation identity is required')
    if type(hot_route_quota) is not int or not 0 <= hot_route_quota <= 128:
        raise ValueError('hot route quota must be in 0..128')
    if type(cold_task_quota) is not int or not 1 <= cold_task_quota <= 128:
        raise ValueError('cold task quota must be in 1..128')
    error = 'packed routes require int32 rows with eight distinct in-range experts'
    for rows, maximum in ((decode, 32), (prefill, 32768)):
        if (not isinstance(rows, np.ndarray) or rows.dtype.kind != 'i' or rows.dtype.itemsize != 4
                or rows.ndim != 2 or rows.shape[1] != 8 or not 1 <= len(rows) <= maximum):
            raise ValueError(error)
    decode, prefill = (np.ascontiguousarray(x, dtype=np.int32) for x in (decode, prefill))
    routes = prefill.size
    sizes = np.empty(4, dtype=np.int32)
    sources = np.empty((384, 5), dtype=np.int32)
    experts, dc, hot, cc = (np.empty(288, dtype=np.int32) for _ in range(4))
    bases = np.empty(289, dtype=np.int32)
    tasks, valid = (np.empty(routes//128+288, dtype=np.int32) for _ in range(2))
    cold_routes = np.empty((routes, 2), dtype=np.int32)
    cold_sources = np.empty((routes, 4), dtype=np.int32)
    def ptr(array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    result = planner()(ptr(decode), len(decode), ptr(prefill), len(prefill), hot_route_quota,
        *(ptr(x) for x in (sizes, sources, experts, dc, hot, cc, bases, tasks, valid, cold_routes, cold_sources)))
    if result:
        raise ValueError(error)
    ne, ns, nc, nt = sizes.tolist()
    plan = MixedExpertPlan(identity, tuple(map(tuple, decode.tolist())), RouteTable.pack(prefill),
        16 if len(decode) <= 8 else 32, tuple(map(tuple, sources[:ns].tolist())),
        tuple(experts[:ne].tolist()), tuple(dc.tolist()), tuple(hot.tolist()),
        RouteTable.pack(cold_routes[:nc]), hot_route_quota)
    windows = tuple((start, min(start+cold_task_quota, nt)) for start in range(0, nt, cold_task_quota))
    cold = ColdExpertPlan(RouteTable.pack(cold_sources[:nc]), tuple(cc.tolist()), tuple(bases.tolist()),
                         tuple(tasks[:nt].tolist()), tuple(valid[:nt].tolist()), windows, cold_task_quota)
    return plan, cold
