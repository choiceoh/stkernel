"""A captured C1 input pack must preserve bytes, guards and strided inputs."""
import importlib.util
import unittest

if importlib.util.find_spec('torch') is not None:
    import torch
else:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires reserved GB10')
class InputPackGridGpuTests(unittest.TestCase):
    def test_changed_strided_inputs_and_poisoned_graph_replay_are_bit_exact(self):
        from engine.kernels.dense import extension
        from probes.engine_input_pack_grid import exact_packs
        exact_packs(lambda *a, **kw: None, extension())

    def test_direct_and_matrix_consumers_keep_their_bytes_and_tx_guards(self):
        from probes.engine_input_pack_grid import projection_checks
        projection_checks(lambda *a, **kw: None, None, 1, timing=False)
