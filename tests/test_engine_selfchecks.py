"""Every module's `_selfcheck` is run here, not only under `python -m` (45차 §89).

A `_selfcheck` states what a module believes about itself, and several of them assert the profile's numbers.
Reached only through `if __name__ == "__main__"`, nothing runs them -- so when the profile moves, they rot
quietly and the thing they were written to catch goes unwatched. `glm53.net` and `glm53.drafter` were both
failing at the draft width the profile had already moved to.

Modules whose self-check needs the checkpoint on disk, a package the ST image carries, or a GPU are named here
with the reason. Those lists are the point: they say exactly which beliefs go unexamined on a given box,
instead of leaving the whole set unrun.
"""
import importlib
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# module -> why its self-check cannot run outside the ST image
NEEDS_MORE = {
    "engine.base.cache_spec": "reads the checkpoint's config from disk",
    "engine.modules.moe": "reads the checkpoint's config from disk",
    "engine.modules.nvfp4_linear": "reads the checkpoint's config from disk",
    "engine.profiles.glm53.specs": "reads the checkpoint's config from disk",
    "engine.profiles.glm53.shims": "binds the served model's shim registry",
}

# module -> what its self-check does on the device
NEEDS_CUDA = {
    "engine.base.arena": "reserves a real arena",
    "engine.base.graphs": "captures a graph",
    "engine.base.kv_tier": "moves blocks to and from the tier",
    "engine.base.tiered_kv": "moves blocks to and from the tier",
    "engine.modules.linear": "compares a row-parallel split against the whole GEMM",
    "engine.modules.sparse_indexer": "compares the fused selection against its reference",
    "engine.profiles.glm53.facts": "asserts the box is the one the profile is written for",
}


def modules():
    """Every engine module that defines a module-level `_selfcheck`."""
    for path in sorted((ROOT / "engine").rglob("*.py")):
        if "\ndef _selfcheck(" in path.read_text():
            yield ".".join(path.relative_to(ROOT).with_suffix("").parts)


class SelfCheckTests(unittest.TestCase):
    def test_every_self_check_this_box_can_run_passes(self):
        ran = []
        for name in modules():
            if name in NEEDS_MORE or (name in NEEDS_CUDA and not torch.cuda.is_available()):
                continue
            with self.subTest(module=name):
                importlib.import_module(name)._selfcheck()
                ran.append(name)
        self.assertGreater(len(ran), 20, ran)

    def test_the_excused_lists_name_only_modules_that_have_a_self_check(self):
        """An excuse that outlives its module hides a self-check that is no longer run."""
        found = set(modules())
        self.assertEqual(sorted((set(NEEDS_MORE) | set(NEEDS_CUDA)) - found), [])
        self.assertEqual(sorted(set(NEEDS_MORE) & set(NEEDS_CUDA)), [])

    @unittest.skipUnless(torch.cuda.is_available(), "the excused-for-CUDA list is only checkable with one")
    def test_the_ones_excused_for_a_gpu_pass_when_there_is_one(self):
        for name in sorted(NEEDS_CUDA):
            with self.subTest(module=name):
                importlib.import_module(name)._selfcheck()


if __name__ == "__main__":
    unittest.main()
