"""engine/kernels/b12x/moe_dispatch.select_sm120_moe_backend: an expert-local eager prefill -- one route a row over the
rank's whole experts, more rows than a decode step -- takes the dynamic prefill kernel, whose artifact is free of the
row count; decode-sized launches and every other geometry keep their choice. Qwen3.8's first fleet boot (2026-09-18)
compiled 75 static kernels on one rank for a 67-token prompt because the static kernel is keyed by rows."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


def select(**kw):
    from engine.kernels.b12x import moe_dispatch as md
    args = dict(activation_precision="fp4", quant_mode="nvfp4", activation="silu", swiglu_limit=None)
    args.update(kw)
    return md.select_sm120_moe_backend(**args)


@unittest.skipUnless(torch is not None and importlib.util.find_spec("flashinfer") is not None,
                     "requires torch and flashinfer (the dispatcher imports it)")
class ExpertLocalPrefillBackendTests(unittest.TestCase):
    QWEN = dict(num_topk=1, num_experts=128, num_local_experts=128, hidden_size=2560, intermediate_size=640)

    def test_an_expert_local_prefill_of_more_than_a_decode_step_is_dynamic(self):
        from engine.kernels.b12x import moe_dispatch as md
        for rows in (md._MICRO_MAX_TOKENS + 1, 40, 168, 320, 639, 640, 641, 5000):
            with self.subTest(rows=rows):
                self.assertEqual(select(num_tokens=rows, **self.QWEN), "dynamic")

    def test_decode_sized_expert_local_launches_keep_the_static_path(self):
        from engine.kernels.b12x import moe_dispatch as md
        for rows in range(1, md._MICRO_MAX_TOKENS + 1):
            with self.subTest(rows=rows):
                self.assertEqual(select(num_tokens=rows, **self.QWEN), "static")

    def test_other_geometries_keep_their_choice(self):
        glm_tp = dict(num_topk=8, num_experts=288, num_local_experts=288, hidden_size=4096, intermediate_size=2048,
                      activation="swigluoai_uninterleave", swiglu_limit=10.)
        self.assertEqual(select(num_tokens=16, **glm_tp), "static")             # 128 pairs, below the cutover
        self.assertEqual(select(num_tokens=2304, **glm_tp), "dynamic")          # a prefill chunk
        dense = dict(num_topk=1, num_experts=1, num_local_experts=1, hidden_size=4096, intermediate_size=3072,
                     activation="swigluoai_uninterleave", swiglu_limit=10.)
        self.assertEqual(select(num_tokens=512, **dense), "static")             # the dense MLP contract
        captured = dict(num_topk=10, num_experts=128, num_local_experts=128, hidden_size=2560, intermediate_size=640)
        self.assertEqual(select(num_tokens=8, **captured), "static")            # Qwen3.8's captured decode step
        self.assertEqual(select(num_tokens=64, **captured), "static")           # 640 pairs: at the cutover
        self.assertEqual(select(num_tokens=65, **captured), "dynamic")
        partial = dict(num_topk=1, num_experts=512, num_local_experts=128, hidden_size=2560, intermediate_size=640)
        self.assertEqual(select(num_tokens=168, **partial), "static")           # not every expert local: the old rule


if __name__ == "__main__":
    unittest.main()
