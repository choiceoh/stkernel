"""Compare actual frontend address expressions to the installed CuTe layout.

Run with the pinned serving image, CUDA hidden, and PYTHONPATH set to the
audited source snapshot. Checks all live rows/blocks, including C2 rows 8..15.
No quantizer arithmetic or GPU execution is simulated by this check.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import cutlass.cute as cute
import cutlass
import cutlass.utils.blockscaled_layout as blockscaled_utils
import engine


assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
source = Path(engine.__file__).resolve().parent / "kernels/b12x/moe_static_kernel_v4.py"
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoEStaticKernelV4")
kernel = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "kernel")
branch = next(n for n in kernel.body if isinstance(n, ast.If)
              and ast.unparse(n.test) == "cutlass.const_expr(self.input_reuse == 4)")


def addressing(nodes):
    names = {"output_offset", "scale_offset", "m_tile_idx", "k_tile_idx",
             "outer_m_idx", "inner_m_idx", "inner_k_idx"}
    selected = sorted((n for root in nodes for n in ast.walk(root)
                       if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and ast.unparse(n.targets[0]) in names), key=lambda n: n.lineno)
    return compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec")


fanout, cached = addressing(branch.body), addressing(branch.orelse)
owner = next(n for n in branch.body if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == "reuse_idx")
owner_code = compile(ast.Module(body=[owner], type_ignores=[]), str(source), "exec")
common = dict(Int32=int, self=SimpleNamespace(sf_vec_size=16), output_bytes_per_row=2048,
              num_k_tiles=64)
cells = []
def address_audit():
    for m in (8, 16):
        for max_rows in (m, m*8):
            layout = blockscaled_utils.tile_atom_to_shape_SF((max_rows, 4096, 288), 16)
            stride = int(cute.crd2idx((0, 0, 1), layout))
            checked, offsets = 0, set()
            for expert in (0, 1, 287):
                for row in range(m):
                    for block in range(256):
                        env = dict(common, max_rows=max_rows, expert_scale_stride=stride,
                                   local_expert_id=expert, row=row, sf_idx=block)
                        other = dict(env)
                        exec(fanout, env)
                        exec(cached, other)
                        expected = int(cute.crd2idx((row, block*16, expert), layout))
                        assert env["scale_offset"] == other["scale_offset"] == expected
                        assert env["output_offset"] == other["output_offset"]
                        assert env["output_offset"] % 8 == 0
                        assert expected not in offsets
                        offsets.add(expected)
                        checked += 1
            # Even an all-distinct expert assignment stays below the reserved tail.
            live_end = m*8*max_rows*2048
            capacity = 288*max_rows*2048
            for mode in (3, 4):
                tail_bytes = m*8*8 + (m*2304 if mode == 3 else 0)
                assert live_end <= capacity-tail_bytes
            cells.append(dict(m=m, max_rows=max_rows, checked_addresses=checked,
                              expert_scale_stride=stride, duplicate_scales=0,
                              scale_matches_actual_cute_layout=True, packed_modes_equal=True,
                              scratch_tail_disjoint=True))
        for grid in (32, 36, 40, 44, 48):
            counts = Counter()
            for cta in range(grid):
                for lane in range(160):
                    env = dict(common, tidx=lane, bidz=cta, gdim_z=grid)
                    exec(owner_code, env)
                    counts.update(range(env["reuse_idx"], m*256, grid*160))
            assert counts == Counter({i: 1 for i in range(m*256)})
@cute.jit
def trace(dummy: cute.Tensor):
    address_audit()


cute.compile(trace, cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,)))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(event="address_audit", gpu_used=False,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      scope="source expressions and real CuTe layout; not a native memory-order test",
                      grids=[32, 36, 40, 44, 48], owner_coverage="exactly once", cells=cells)))
