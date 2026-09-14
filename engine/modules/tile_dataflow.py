"""A bounded rank-local MLP dataflow, with an independently scheduled CPU oracle.

Gate/up tiles publish activations. Down tiles consume one activation tile and
publish FP32 partials. Output tiles reduce only their own completed partials.
The CUDA worker queue executes this graph in one kernel; it is not a sequence
of CUDA graph replays. TP reduction remains at the existing MLP boundary.
"""
from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class MLPPlan:
    rows: int
    hidden: int
    intermediate: int
    tile: int = 32
    workers: int = 48                  # GB10's fixed SM budget, no reserved scheduler SM
    max_scratch_bytes: int = 64 << 20

    def __post_init__(self):
        if (any(type(v) is not int or v <= 0 for v in (self.rows, self.hidden, self.intermediate,
                self.tile, self.workers, self.max_scratch_bytes)) or self.rows > 32 or self.tile not in (32, 64, 128)
                or self.workers > 48 or self.hidden > 8192 or self.intermediate > 8192):
            raise ValueError("MLP dataflow requires 1..32 rows, 32/64/128-wide tiles, and 1..48 GB10 workers")
        if self.scratch_bytes > self.max_scratch_bytes:
            raise ValueError(f"MLP dataflow scratch requires {self.scratch_bytes} bytes")

    @property
    def producers(self):
        return (self.intermediate + self.tile - 1) // self.tile

    @property
    def outputs(self):
        return (self.hidden + self.tile - 1) // self.tile

    @property
    def scratch_bytes(self):
        return (self.rows*self.intermediate*2 + self.producers*self.rows*self.hidden*4
                + (4+self.producers+self.outputs)*4 + self.rows*self.hidden*2)

    @property
    def tasks(self):
        p, o = self.producers, self.outputs
        return (tuple(("gate_up", i, ()) for i in range(p))
                + tuple(("down", i*o+j, (i,)) for i in range(p) for j in range(o))
                + tuple(("reduce", j, tuple(p+i*o+j for i in range(p))) for j in range(o)))

    def validate_tensors(self, x, gate_up, down):
        if (x.shape != (self.rows, self.hidden) or gate_up.shape != (2*self.intermediate, self.hidden)
                or down.shape != (self.hidden, self.intermediate)
                or any(t.dtype != torch.bfloat16 or t.device != x.device or not t.is_contiguous()
                       for t in (x, gate_up, down))):
            raise ValueError("MLP dataflow requires matching contiguous BF16 operands")


def reference(plan, x, gate_up, down, limit, *, seed=0):
    """Execute ready tasks in randomized order, not the device worker's order."""
    from engine.profiles.glm53.lanes import swiglu_clamped
    plan.validate_tensors(x, gate_up, down)
    if x.device.type != "cpu":
        raise ValueError("the dataflow scheduling oracle is CPU-only")
    tasks, pending, done, trace = plan.tasks, set(range(len(plan.tasks))), set(), []
    rng = random.Random(seed)
    activation = torch.empty((plan.rows, plan.intermediate), dtype=x.dtype)
    partial = torch.empty((plan.producers, plan.rows, plan.hidden), dtype=torch.float32)
    output = torch.empty_like(x)
    while pending:
        ready = [i for i in sorted(pending) if all(d in done for d in tasks[i][2])]
        if not ready:
            raise RuntimeError("dataflow graph cannot make progress")
        task = rng.choice(ready)
        kind, index, _ = tasks[task]
        if kind == "gate_up":
            lo, hi = index*plan.tile, min((index+1)*plan.tile, plan.intermediate)
            # Ordinary projection output rounds before the nonlinear activation.
            g = (x.float() @ gate_up[lo:hi].float().T).to(x.dtype)
            u = (x.float() @ gate_up[plan.intermediate+lo:plan.intermediate+hi].float().T).to(x.dtype)
            activation[:, lo:hi] = swiglu_clamped(g, u, limit)
        elif kind == "down":
            producer, column = divmod(index, plan.outputs)
            lo, hi = producer*plan.tile, min((producer+1)*plan.tile, plan.intermediate)
            start, end = column*plan.tile, min((column+1)*plan.tile, plan.hidden)
            partial[producer, :, start:end] = activation[:, lo:hi].float() @ down[start:end, lo:hi].float().T
        else:
            start, end = index*plan.tile, min((index+1)*plan.tile, plan.hidden)
            acc = torch.zeros((plan.rows, end-start), dtype=torch.float32)
            for producer in range(plan.producers):
                acc += partial[producer, :, start:end]
            output[:, start:end] = acc.to(output.dtype)
        pending.remove(task); done.add(task); trace.append(task)
    return output, tuple(trace)


class PersistentDense:
    """Explicit binding for unprepared BF16 dense MLPs in the tree experiment.

    Prepared FP8/NVFP4 operators are refused, even if an old BF16 copy remains.
    This first device dataflow slice is a numerical/execution prototype, not a
    replacement for the fleet's packed dense or routed expert operators.
    """
    def __init__(self, *, backend="cuda", workers=48, max_scratch_bytes=64 << 20):
        if backend not in ("cuda", "reference"):
            raise ValueError("choose the explicit CUDA executor or CPU scheduling oracle")
        self.backend, self.workers, self.max_scratch_bytes = backend, workers, max_scratch_bytes
        self.plans, self.executed = {}, set()

    def validate(self, net, rows):
        plans = {}
        for L in net.layers:
            if net.F.is_moe(L):
                continue
            name = f"L{L}.mlp."
            if (net.dense_nvfp4 or any(name+k in net.dense for k in ("gate_up", "down"))
                    or any(not isinstance(net.p.get(name+k), torch.Tensor) for k in ("gate_up", "down"))):
                raise ValueError("persistent BF16 MLP cannot consume prepared/retired packed weights")
            w, down = net.p[name+"gate_up"], net.p[name+"down"]
            plan = MLPPlan(rows, net.F.hidden, w.shape[0]//2, workers=self.workers,
                           max_scratch_bytes=self.max_scratch_bytes)
            plan.validate_tensors(torch.empty((rows, net.F.hidden), dtype=torch.bfloat16, device=w.device), w, down)
            if (w.is_cuda) != (self.backend == "cuda"):
                raise ValueError("MLP binding backend does not match the weight device")
            plans[L] = plan
        if not plans:
            raise ValueError("persistent dense binding needs at least one eligible dense layer")
        self.plans = plans

    def __call__(self, net, L, x):
        plan, name = self.plans[L], f"L{L}.mlp."
        w, down = net.p[name+"gate_up"], net.p[name+"down"]
        if self.backend == "reference":
            out, _ = reference(plan, x.contiguous(), w, down, net.F.swiglu_limit)
        else:
            from engine.kernels.tile_dataflow import execute
            out, error = execute(plan, x.contiguous(), w, down, net.F.swiglu_limit)
            # Every rank reaches the same control vote before the existing output
            # collective. Never send partial/invalid output after a worker timeout.
            errors = net.comm.gather_objects(error)
            if any(errors):
                raise RuntimeError(f"persistent MLP worker failure on ranks: {errors}")
        self.executed.add(L)
        return net.comm.all_reduce(out)
