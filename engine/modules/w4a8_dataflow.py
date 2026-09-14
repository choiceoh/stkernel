"""Persistent dense MLP bound to the existing GPTQ W4Pack / FP8 activation lane.

W4A8 is distinct from ModelOpt's W4A4 expert lane: signed compact weight
scales, per-row undo scales, GATE|UP order, and group-128 E4M3 activations.
No weight repacking or resident BF16 expansion is performed by this binding.
"""
from dataclasses import dataclass
import random

import torch

from engine.kernels.dense import DenseLinear, W4Pack
from engine.modules.tile_dataflow import MLPPlan


@dataclass(frozen=True)
class W4A8Plan(MLPPlan):
    tile: int = 128

    def __post_init__(self):
        if self.tile != 128 or self.hidden % 128 or self.intermediate % 128:
            raise ValueError("W4A8 dataflow preserves group-128 activation quantization")
        super().__post_init__()

    @property
    def scratch_bytes(self):
        return (self.rows*self.intermediate + self.rows*self.producers*4
                + self.producers*self.rows*self.hidden*4
                + (4+self.producers+self.outputs)*4 + self.rows*self.hidden*2)


@dataclass(frozen=True)
class W4A8Weights:
    gate_up: W4Pack
    down: W4Pack

    def __post_init__(self):
        g, d = self.gate_up, self.down
        if (not isinstance(g, W4Pack) or not isinstance(d, W4Pack)
                or g.rows != 2*d.cols or g.cols != d.rows):
            raise ValueError("W4A8 dataflow needs matching gate_up and down W4Pack owners")
        for p in (g, d):
            if (p.rows <= 0 or p.cols <= 0 or p.cols % 128
                    or p.data.shape != ((p.rows+127)//128, p.cols//128, 128, 64)
                    or p.scale.shape != (*p.data.shape[:3], 8)
                    or p.rowscale.shape != (((p.rows+127)//128)*128,)
                    or (p.data.dtype, p.scale.dtype, p.rowscale.dtype) != (torch.uint8, torch.int8, torch.float32)
                    or any(t.device != g.data.device or not t.is_contiguous() for t in (p.data, p.scale, p.rowscale))):
                raise ValueError("W4A8 dataflow needs native contiguous W4Pack planes and signed scale bytes")

    @property
    def hidden(self):
        return self.gate_up.cols

    @property
    def intermediate(self):
        return self.down.cols

    def validate_input(self, plan, x):
        if (plan.hidden != self.hidden or plan.intermediate != self.intermediate
                or x.shape != (plan.rows, self.hidden) or x.dtype != torch.bfloat16
                or x.device != self.gate_up.data.device or not x.is_contiguous()):
            raise ValueError("W4A8 dataflow requires matching contiguous BF16 inputs")

    @classmethod
    def from_net(cls, net, layer):
        if net.dense_nvfp4 or net.F.is_moe(layer):
            raise ValueError("W4A8 dataflow binds ordinary dense layers, not ModelOpt W4A4 experts")
        pairs = tuple(net.dense.get(f"L{layer}.mlp.{name}") for name in ("gate_up", "down"))
        for op in pairs:
            if (not isinstance(op, DenseLinear) or op.decode_precision != "w4" or len(op.packs) != 1
                    or op.observer is not None):
                raise ValueError("W4A8 dataflow requires a prepared single-pack decode lane without calibration observers")
            if op.rows != op.packs[0].rows or op.cols != op.packs[0].cols:
                raise ValueError("W4A8 dataflow cannot reinterpret padded or multi-pack projections")
        # Input smoothing already lives in the norm and packed columns. The
        # current DenseLinear call applies no extra scaling to the input.
        return cls(pairs[0].packs[0], pairs[1].packs[0])


def reference(plan, x, weights, limit, *, seed=0):
    """CPU scheduling oracle using the deployed lane's independent quant twin."""
    from engine.kernels.dense.packing import _mk_quant_x_ref, mk_w4_dequant
    from engine.profiles.glm53.lanes import swiglu_clamped
    weights.validate_input(plan, x)
    if x.device.type != "cpu":
        raise ValueError("the W4A8 numerical/scheduling oracle is CPU-only")
    def dequant(p):
        return mk_w4_dequant(p.data, p.scale, p.rows, rgs=p.rowscale)
    g, down = dequant(weights.gate_up), dequant(weights.down)
    xq = _mk_quant_x_ref(x)
    act = torch.empty((plan.rows, plan.intermediate), dtype=torch.float32)
    partial = torch.empty((plan.producers, plan.rows, plan.hidden), dtype=torch.float32)
    out = torch.empty_like(x)
    tasks, pending, done, trace = plan.tasks, set(range(len(plan.tasks))), set(), []
    rng = random.Random(seed)
    while pending:
        ready = [i for i in sorted(pending) if all(d in done for d in tasks[i][2])]
        if not ready:
            raise RuntimeError("W4A8 dataflow cannot make progress")
        task = rng.choice(ready)
        kind, idx, _ = tasks[task]
        if kind == "gate_up":
            lo, hi = idx*128, (idx+1)*128
            gate = (xq @ g[lo:hi].T).bfloat16()
            up = (xq @ g[plan.intermediate+lo:plan.intermediate+hi].T).bfloat16()
            act[:, lo:hi] = _mk_quant_x_ref(swiglu_clamped(gate, up, limit))
        elif kind == "down":
            p, c = divmod(idx, plan.outputs)
            partial[p, :, c*128:(c+1)*128] = act[:, p*128:(p+1)*128] @ down[c*128:(c+1)*128, p*128:(p+1)*128].T
        else:
            acc = torch.zeros((plan.rows, 128), dtype=torch.float32)
            for p in range(plan.producers):
                acc += partial[p, :, idx*128:(idx+1)*128]
            out[:, idx*128:(idx+1)*128] = acc.bfloat16()
        pending.remove(task); done.add(task); trace.append(task)
    return out, tuple(trace)


class PersistentW4A8:
    """Explicit packed dense binding for Verification/decode_once."""
    def __init__(self, *, backend="cuda", workers=48, max_scratch_bytes=64 << 20):
        if backend not in ("cuda", "reference"):
            raise ValueError("choose the explicit CUDA W4A8 executor or CPU oracle")
        self.backend, self.workers, self.max_scratch_bytes = backend, workers, max_scratch_bytes
        self.plans, self.weights, self.executed = {}, {}, set()

    def validate(self, net, rows):
        plans, weights = {}, {}
        for layer in net.layers:
            if net.F.is_moe(layer):
                continue
            w = W4A8Weights.from_net(net, layer)
            if w.gate_up.data.is_cuda != (self.backend == "cuda"):
                raise ValueError("W4A8 dataflow backend does not match the existing packs")
            plans[layer] = W4A8Plan(rows, w.hidden, w.intermediate, workers=self.workers,
                                    max_scratch_bytes=self.max_scratch_bytes)
            weights[layer] = w
        if not plans:
            raise ValueError("W4A8 dataflow needs at least one prepared dense layer")
        self.plans, self.weights = plans, weights

    def __call__(self, net, layer, x):
        plan, weights = self.plans[layer], self.weights[layer]
        # Refuse a changed calibration/pack owner instead of silently skipping
        # an observer or continuing to read retired arena bytes.
        current = W4A8Weights.from_net(net, layer)
        if current.gate_up is not weights.gate_up or current.down is not weights.down:
            raise RuntimeError("W4A8 pack owner changed after binding")
        if self.backend == "reference":
            out, _ = reference(plan, x.contiguous(), weights, net.F.swiglu_limit)
        else:
            from engine.kernels.tile_dataflow import execute_w4a8
            out, error = execute_w4a8(plan, x.contiguous(), weights, net.F.swiglu_limit)
            errors = net.comm.gather_objects(error)
            if any(errors):
                raise RuntimeError(f"persistent W4A8 worker failure on ranks: {errors}")
        self.executed.add(layer)
        for name in ("gate_up", "down"):
            net.dense[f"L{layer}.mlp.{name}"].executed |= 1
        return net.comm.all_reduce(out)
