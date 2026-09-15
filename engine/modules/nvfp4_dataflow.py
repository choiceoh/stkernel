"""Native NVFP4 binding for the rank-local persistent MLP dataflow.

Borrow the model's existing packed bytes and scale owner. Weights remain
E2M1; activations use the existing group-16 E4M3/E2M1 W4A4 convention.
There is no BF16 weight materialization or second packed weight allocation.
Full dequantization below is restricted to the independent CPU oracle.
"""
from dataclasses import dataclass
import random

import torch

from engine.modules.expert_layout import is_tile_major, row_major_expert, w13_chunk_bytes, W2_K_IN_BYTES
from engine.modules.tile_dataflow import MLPPlan
from engine.modules.nvfp4_sf import unswizzle_sf


@dataclass(frozen=True)
class NVFP4Plan(MLPPlan):
    tile: int = 64

    def __post_init__(self):
        super().__post_init__()
        if self.tile != 64 or self.hidden % 16 or self.intermediate % 16:
            raise ValueError("NVFP4 dataflow requires 64-wide tiles and complete group-16 blocks")

    @property
    def scratch_bytes(self):
        return (self.rows*(self.intermediate//2 + self.intermediate//16)
                + self.producers*self.rows*self.hidden*4 + (4+self.producers+self.outputs)*4
                + self.rows*self.hidden*2)


@dataclass(frozen=True)
class NVFP4Weights:
    w13: torch.Tensor                  # existing [1,2I,H/2], UP then GATE
    w2: torch.Tensor                   # existing [1,H,I/2]
    sf13: torch.Tensor                 # borrowed raw-interleaved bytes or SF6 owner
    sf2: torch.Tensor
    scales: object                    # existing ModelOptScales, positive FP32 a and a*w
    tile13: int = 0
    tile2: int = 0
    sf6: bool = False

    def __post_init__(self):
        if (self.w13.ndim != 3 or self.w13.shape[0] != 1 or self.w13.shape[1] % 2
                or self.w2.shape != (1, self.w13.shape[2]*2, self.w13.shape[1]//4)
                or any(t.dtype != torch.uint8 or t.device != self.w13.device or not t.is_contiguous()
                       for t in (self.w13, self.w2, self.sf13, self.sf2))):
            raise ValueError("NVFP4 dataflow needs a single dense expert's packed bytes and scale planes")
        h, i = self.hidden, self.intermediate
        if h % 16 or i % 16 or self.tile13 not in (0, 64, 128, 256) or self.tile2 not in (0, 64):
            raise ValueError("invalid group-16 or tile-major NVFP4 geometry")
        if (self.tile13 and h//2 % self.tile13) or (self.tile2 and i//2 % self.tile2):
            raise ValueError("packed weight tile does not divide its input dimension")
        if self.sf6:
            if 2*i % 128 or h % 256 or i % 128:
                raise ValueError("SF6 planes need complete FC1/FC2 stage geometry")
            sizes = ((2*i//128)*(h//256)*1552, (h//256)*(i//128)*1552)
        else:
            sizes = tuple(((r+127)//128)*128 * (((k//16+3)//4)*4) for r, k in ((2*i, h), (h, i)))
        if (self.sf13.numel(), self.sf2.numel()) != sizes:
            raise ValueError("NVFP4 scale planes do not match their declared layout")
        for name in ("weight13", "input13", "weight2", "input2", "alpha13", "alpha2"):
            value = getattr(self.scales, name)
            if (value.shape != (1,) or value.dtype != torch.float32 or value.device != self.w13.device
                    or not value.is_contiguous() or not torch.isfinite(value).all() or not (value > 0).all()):
                raise ValueError("NVFP4 dataflow requires valid ModelOpt per-expert FP32 scales")

    @property
    def hidden(self):
        return self.w13.shape[2]*2

    @property
    def intermediate(self):
        return self.w13.shape[1]//2

    @classmethod
    def from_net(cls, net, layer):
        if not net.dense_nvfp4 or net.F.is_moe(layer):
            raise ValueError("bind the ModelOpt fixed-one-expert NVFP4 dense path")
        p, name = net.p, f"L{layer}.mlp."
        first, second = p[name+"w13_sf"], p[name+"w2_sf"]
        retired = [bool(getattr(t, "_st_sf6_consumed", False)) for t in (first, second)]
        if any(retired):
            owner = getattr(net, "_expert_views", {}).get(layer)
            reform = getattr(owner, "reform_scales", None)
            if not all(retired) or reform is None or not reform.enabled:
                raise ValueError("retired raw scales require their actual prepared SF6 owner")
            first, second = reform.fc1, reform.fc2
        w13, w2 = p[name+"w13"], p[name+"w2"]
        return cls(w13, w2, first.view(torch.uint8), second.view(torch.uint8), net._quant_scales[layer],
                   w13_chunk_bytes(w13), W2_K_IN_BYTES if is_tile_major(w2) else 0,
                   all(retired))

    def raw_scales_cpu(self, *, second=False):
        """Independent inverse of both scale layouts, for CPU numerical tests."""
        source = self.sf2 if second else self.sf13
        rows, k = (self.hidden, self.intermediate) if second else (2*self.intermediate, self.hidden)
        if source.device.type != "cpu":
            raise ValueError("scale dequantization is restricted to the CPU oracle")
        if not self.sf6:
            return unswizzle_sf(source, rows, k//16).view(torch.float8_e4m3fn)
        # Decode byte planes, then invert the stage transpose. Keep this CPU
        # implementation independent of the GPU scalar address calculations.
        stages = source.reshape(-1, 1552).to(torch.int16)
        byte = torch.arange(2048)
        low = (stages[:, byte//2] >> (byte % 2 * 4)) & 15
        high = (stages[:, 1024+byte//4] >> (byte % 4 * 2)) & 3
        decoded = stages[:, 1536:1537] + low + (high << 4)
        if (decoded > 255).any() or stages[:, 1537:].any():
            raise ValueError("invalid SF6 byte code or reserved header")
        if second:
            decoded = decoded.view(rows//256, k//128, 2, 2, 512).permute(0, 3, 1, 2, 4)
        raw = decoded.reshape(-1).to(torch.uint8)
        return unswizzle_sf(raw, rows, k//16).view(torch.float8_e4m3fn)


def reference(plan, x, weights, limit, *, seed=0):
    """CPU tile DAG compared against lanes.reference().moe's complete W4A4 MLP."""
    from engine.modules.moe import quant_nvfp4_act, dequant_nvfp4_act, dequant_nvfp4
    from engine.profiles.glm53.lanes import swiglu_clamped
    if (x.device.type != "cpu" or weights.w13.device.type != "cpu" or x.dtype != torch.bfloat16
            or x.shape != (plan.rows, weights.hidden) or plan.intermediate != weights.intermediate):
        raise ValueError("NVFP4 scheduling oracle requires matching CPU operands")
    s = weights.scales
    w13 = dequant_nvfp4(row_major_expert(weights.w13, 0, w13_chunk_bytes(weights.w13)), weights.raw_scales_cpu(), s.weight13[0])
    w2 = dequant_nvfp4(row_major_expert(weights.w2, 0, W2_K_IN_BYTES), weights.raw_scales_cpu(second=True), s.weight2[0])
    packed, sf = quant_nvfp4_act(x, s.input13[0])
    input_q = dequant_nvfp4_act(packed, sf, s.input13[0])
    activation = torch.empty((plan.rows, plan.intermediate), dtype=torch.float32)
    partial = torch.empty((plan.producers, plan.rows, plan.hidden), dtype=torch.float32)
    output = torch.empty_like(x)
    tasks, done, trace = plan.tasks, set(), []
    pending, rng = set(range(len(tasks))), random.Random(seed)
    while pending:
        ready = [t for t in sorted(pending) if all(d in done for d in tasks[t][2])]
        if not ready:
            raise RuntimeError("NVFP4 dataflow graph cannot make progress")
        task = rng.choice(ready)
        kind, index, _ = tasks[task]
        if kind == "gate_up":
            lo, hi = index*plan.tile, min((index+1)*plan.tile, plan.intermediate)
            up = input_q @ w13[lo:hi].T
            gate = input_q @ w13[plan.intermediate+lo:plan.intermediate+hi].T
            value = swiglu_clamped(gate, up, limit)        # no BF16 FC1 accumulator cast
            packed, sf = quant_nvfp4_act(value, s.input2[0])
            activation[:, lo:hi] = dequant_nvfp4_act(packed, sf, s.input2[0])
        elif kind == "down":
            producer, col = divmod(index, plan.outputs)
            lo, hi = producer*plan.tile, min((producer+1)*plan.tile, plan.intermediate)
            start, end = col*plan.tile, min((col+1)*plan.tile, plan.hidden)
            partial[producer, :, start:end] = activation[:, lo:hi] @ w2[start:end, lo:hi].T
        else:
            start, end = index*plan.tile, min((index+1)*plan.tile, plan.hidden)
            value = torch.zeros((plan.rows, end-start), dtype=torch.float32)
            for producer in range(plan.producers):
                value += partial[producer, :, start:end]
            output[:, start:end] = value.to(x.dtype)
        pending.remove(task); done.add(task); trace.append(task)
    return output, tuple(trace)


class PersistentNVFP4:
    """Fixed NVFP4 binding used by Verification/decode_once; no dtype fallback."""
    def __init__(self, *, backend="cuda", workers=48, max_scratch_bytes=64 << 20):
        if backend not in ("cuda", "reference"):
            raise ValueError("choose the explicit CUDA NVFP4 executor or CPU oracle")
        self.backend, self.workers, self.max_scratch_bytes = backend, workers, max_scratch_bytes
        self.plans, self.weights, self.executed = {}, {}, set()

    def validate(self, net, rows):
        plans, weights = {}, {}
        for L in net.layers:
            if net.F.is_moe(L):
                continue
            w = NVFP4Weights.from_net(net, L)
            if w.w13.is_cuda != (self.backend == "cuda"):
                raise ValueError("NVFP4 dataflow backend does not match the packed weights")
            plans[L] = NVFP4Plan(rows, w.hidden, w.intermediate, workers=self.workers,
                                  max_scratch_bytes=self.max_scratch_bytes)
            weights[L] = w
        if not plans:
            raise ValueError("NVFP4 dataflow needs at least one fixed-one-expert dense layer")
        self.plans, self.weights = plans, weights

    def __call__(self, net, L, x):
        plan, weights = self.plans[L], self.weights[L]
        if self.backend == "reference":
            out, _ = reference(plan, x.contiguous(), weights, net.F.swiglu_limit)
        else:
            from engine.kernels.tile_dataflow import execute_nvfp4
            out, error = execute_nvfp4(plan, x.contiguous(), weights, net.F.swiglu_limit)
            errors = net.comm.gather_objects(error)
            if any(errors):
                raise RuntimeError(f"persistent NVFP4 worker failure on ranks: {errors}")
        self.executed.add(L)
        return net.comm.all_reduce(out)
