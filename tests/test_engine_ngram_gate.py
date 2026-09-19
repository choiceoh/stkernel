"""engine/kernels/ngram_gate: the PLE injection's gate and conv norm in one launch -- its torch form transcribed cast for
cast from engine/modules/ngram_embedding.NGramInjection (byte for byte), and the launch held to that form within the
band its own three reductions allow (on a GPU, or under TRITON_INTERPRET=1 with tests/test_engine_qwen38_kernels'
accommodations for the interpreter's casts).

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_ngram_gate
"""
import unittest

from tests.test_engine_qwen38_kernels import DEVICE, RUNS, RUNS_REASON, served_kernels, torch

HC, HID, HEADS, WIDTH, EPS = 4, 256, 8, 32, 1e-6


def case(rows, seed=0):
    gen = torch.Generator().manual_seed(seed)
    width = HC * HID
    h = torch.randn(rows, width, generator=gen).bfloat16()
    embeddings = torch.randn(rows, HEADS * WIDTH, generator=gen).bfloat16()
    kv = (torch.randn(width + HID, HEADS * WIDTH, generator=gen) * 0.1).bfloat16()
    norms = {name: (torch.randn(width, generator=gen) * 0.1).bfloat16() for name in ("q_norm", "k_norm", "conv_norm")}
    return h, embeddings, kv, norms


def feature(kv, norms):
    from engine.modules.ngram_embedding import VARIANTS, NGramInjection
    weights = dict(norms, kv=kv)
    return NGramInjection(hidden=HID, hc=HC, ngram_size=3, conv=4, eps=EPS, **VARIANTS["ple"], hash=None,
                          weights=lambda layer, name: weights[name], table=None)


class TorchFormTests(unittest.TestCase):
    def test_the_transcription_is_the_module_s_arithmetic(self):
        from engine.kernels import ngram_gate
        h, embeddings, kv, norms = case(7)
        f = feature(kv, norms)
        want = ngram_gate.reference(f, h, embeddings, lambda name: f.weights(0, name))
        key, value = torch.nn.functional.linear(embeddings, kv).split([HC * HID, HID], dim=-1)
        got = ngram_gate.torch_form(h, key, value, norms["q_norm"], norms["k_norm"], norms["conv_norm"], EPS, HC)
        self.assertTrue(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))


@unittest.skipUnless(RUNS, RUNS_REASON)
class GateTests(unittest.TestCase):
    def test_the_launch_is_the_torch_form_within_a_few_bf16_steps(self):
        from engine.kernels import ngram_gate
        for rows in (1, 5, 33):
            h, embeddings, kv, norms = case(rows, seed=rows)
            key, value = torch.nn.functional.linear(embeddings, kv).split([HC * HID, HID], dim=-1)
            want = ngram_gate.torch_form(h, key, value, norms["q_norm"], norms["k_norm"], norms["conv_norm"], EPS, HC)
            d = lambda t: t.to(DEVICE)                                 # noqa: E731
            with served_kernels():
                got = ngram_gate.gate(d(h), d(key), d(value), d(norms["q_norm"]), d(norms["k_norm"]),
                                      d(norms["conv_norm"]), EPS, HC)
            for name, a, b in (("gated", got[0], want[0]), ("normed", got[1], want[1])):
                a = a.cpu().float()
                b = b.float()
                with self.subTest(rows=rows, out=name):
                    self.assertEqual(a.shape, (rows, HC * HID))
                    self.assertLess(float((a - b).abs().max() / b.abs().max()), 2.0 ** -6)
                    self.assertLess(float((a != b).float().mean()), 0.02)      # the reductions' last bit, rarely carried

    def test_it_refuses_rows_it_does_not_take(self):
        from engine.kernels import ngram_gate
        h, embeddings, kv, norms = case(3)
        key, value = torch.nn.functional.linear(embeddings, kv).split([HC * HID, HID], dim=-1)
        with self.assertRaises(ValueError):
            ngram_gate.gate(h.float(), key.float(), value.float(), norms["q_norm"], norms["k_norm"], norms["conv_norm"],
                            EPS, HC)
        with self.assertRaises(ValueError):
            ngram_gate.gate(h, key[:, :-1], value, norms["q_norm"], norms["k_norm"], norms["conv_norm"], EPS, HC)


if __name__ == "__main__":
    unittest.main()
