"""engine/modules/ngram_embedding as a family: one hash, one gate-and-write, the axes of its docstring.

The PLE variant is held to transformers qwen4_exp (Qwen4ExpTextNGramEmbedding's hash ids, Qwen4ExpTextPLELayer's
addend; the whole model by tests/test_engine_composition.py). The engram variant is held to the vendor DeepSeek-V4.1
inference code -- engram.py (EngramLayout, compute_hash_multipliers, build_compressed_token_map, NgramHashState) and
model.py's ParallelEngramEmbedding and Engram, extracted verbatim as probes/dsv41_engram_gate_diff.py does -- kept
beside the transformers oracle in the git-excluded .oracle-site/dsv41/ (MIT; model.py is the sha
profiles/dsv41/caches.py pins). Hash ids, token maps and the lookup are compared with torch.equal, the fp32 write too;
the served bf16 layer within its rounding."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
    import torch.nn as nn

VENDOR = Path(__file__).resolve().parents[1] / ".oracle-site" / "dsv41"
MODEL_SHA = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"


def _qwen_present() -> bool:
    try:
        return importlib.util.find_spec("transformers.models.qwen4_exp") is not None
    except Exception:
        return False


def _vendor_present() -> bool:
    return (torch is not None and (VENDOR / "engram.py").exists() and (VENDOR / "model.py").exists()
            and all(importlib.util.find_spec(m) is not None for m in ("numpy", "sympy", "tokenizers")))


def vendor_engram():
    """inference/engram.py as a module."""
    name = "dsv41_vendor_engram"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, VENDOR / "engram.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def vendor_model_classes(block: int):
    """ParallelEngramEmbedding and Engram from inference/model.py, verbatim, with the globals they read at world 1."""
    import hashlib
    source = (VENDOR / "model.py").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != MODEL_SHA:
        raise ValueError("model.py is not the pinned reference")
    start = source.index("class ParallelEngramEmbedding(nn.Module):")
    end = source.index("@lru_cache(2)", start)
    ns = {"torch": torch, "nn": nn, "F": torch.nn.functional, "dist": None, "world_size": 1, "rank": 0,
          "fp8_block_size": block, "scale_dtype": torch.float8_e8m0fnu, "ModelArgs": object,
          "EngramLayout": vendor_engram().EngramLayout,
          "Linear": lambda i, o, bias=False, dtype=None: nn.Linear(i, o, bias=False)}
    exec(compile(source[start:end], str(VENDOR / "model.py"), "exec"), ns)
    return ns["ParallelEngramEmbedding"], ns["Engram"]


class FakeBackend:
    def __init__(self, texts):
        self.texts = texts

    def decode(self, ids, skip_special_tokens=False):
        return self.texts[ids[0]]

    def id_to_token(self, token_id):
        return f"<raw {token_id}>"


class FakeTokenizer:
    """The two members build_compressed_token_map touches, over texts chosen to collapse: case, accents, full-width
    and ligature forms (NFKC), whitespace runs, a lone space, partial UTF-8 bytes."""
    def __init__(self, seed=0):
        import random
        rng = random.Random(seed)
        texts = ["<pad>", "<bos>", "<eos>", "The", "the", " the", "THE", " The ", "café", "cafe", "CAFÉ", "Ｔｈｅ", "ﬁne",
                 "fine", " ", "  ", "\n", "\t \n", " ", "�", "��", "a�", "Hello", "hello",
                 " hello", "naïve", "naive", "Ångström", "angstrom", "", "x"]
        letters = "abcdeABCDEéü "
        while len(texts) < 300:
            texts.append("".join(rng.choice(letters) for _ in range(rng.randint(1, 4))))
        self.backend_tokenizer = FakeBackend(texts)
        self.n = len(texts)

    def __len__(self):
        return self.n


def draw(mod, seed=0, unit_offset=True):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in mod.named_parameters():
            if p.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                continue
            if "norm" in name:
                p.normal_(0.0 if unit_offset else 1.0, 0.3)
            elif name.endswith(("q_weight", "k_weight")):
                p.normal_(1.0, 0.3)
            elif p.ndim == 3:
                p.normal_(0, 0.5)
            elif p.ndim == 2:
                p.normal_(0, p.shape[-1] ** -0.5)
            else:
                p.normal_(0, 0.5)
    return mod


def run(feat, h, ids, pieces, state=None):
    from engine.base.composition import State, Step
    state = State() if state is None else state
    outs, at = [], 0
    for n in pieces:
        step = Step.of([(0, at, ids[at:at + n])])
        state.check(step)
        outs.append(feat(0, h[at:at + n], step, state))
        state.commit(step)
        at += n
    return torch.cat(outs), state


class Held(unittest.TestCase):
    def close(self, got, want, tol):
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        diff = float((got.float() - want.float()).abs().max())
        self.assertLessEqual(diff, tol, f"max |got - want| = {diff}")


# ---------------------------------------------------------------------------------------------------------------------
# Qwen3.8's PLE
# ---------------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None and _qwen_present(), "requires transformers with qwen4_exp on the path")
class QwenPLETests(Held):
    H, HC, T, EOS = 32, 4, 48, 0

    def config(self):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        return Qwen4ExpTextConfig(vocab_size=512, hidden_size=self.H, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=16, hc_count=self.HC, ple_layer_ids=[1, 2],
                                  ple_embed_dim=32, ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
                                  make_ngram_vocab_size_divisible_by=8, ple_conv_kernel_size=4, eos_token_id=self.EOS,
                                  rms_norm_eps=1e-6, hidden_act="silu", seed=1234,
                                  layer_types=["linear_attention", "linear_attention"])

    def ids(self, seed):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(1, 512, (self.T,), generator=g)
        ids[[5, 6, 19, 30]] = self.EOS                                        # documents end, one right after another
        return ids

    def hash(self, cfg, table_index):
        from engine.modules.ngram_embedding import NGramHash
        return NGramHash.splitmix(ngram_size=cfg.ngram_size, heads=cfg.heads_per_ngram, unigram_vocab=cfg.vocab_size,
                                  base=cfg.ngram_vocab_size_base, table_index=table_index, seed=cfg.seed, eos=self.EOS)

    def test_the_hash_ids_are_the_models(self):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
        from engine.modules.ngram_embedding import DEAD
        cfg = self.config()
        for table_index in (0, 1):
            emb = Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx=table_index, ple_layer_index=table_index)
            with torch.no_grad():
                emb.ngram_embedding.weight.zero_()
                emb.ngram_embedding.weight[:, 0] = torch.arange(emb.ngram_embedding.weight.shape[0], dtype=torch.float32)
            ids = self.ids(table_index)
            with torch.no_grad():
                rows = emb(ids[None], None)[0].view(self.T, emb.ngram_heads, -1)[..., 0].long()
            made = self.hash(cfg, table_index)
            self.assertEqual(made.total, emb.total_vocab_size)
            got = made.rows(torch.cat([torch.full((2,), DEAD), ids]), self.T)
            self.assertTrue(torch.equal(got, rows))

    def test_the_layer_is_the_model(self):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
        from engine.modules.ngram_embedding import VARIANTS, NGramInjection, named
        cfg = self.config()
        mod = draw(Qwen4ExpTextPLELayer(cfg, layer_idx=1, ple_layer_index=1).eval(), seed=3)
        sd = {k: v.detach().clone() for k, v in mod.state_dict().items()}
        source = lambda hf: sd[f"{hf}.weight"] if f"{hf}.weight" in sd else sd[hf]
        table = sd["ple_embedding.ngram_embedding.weight"]
        feat = NGramInjection(hidden=self.H, hc=self.HC, ngram_size=3, eps=1e-6, conv=4, dtype="float32",
                              **VARIANTS["ple"], hash=lambda layer: self.hash(cfg, 1),
                              weights=lambda layer, name: named("qwen4_exp", source)(name),
                              table=lambda layer, rows: table[rows])
        ids = self.ids(7)
        h = torch.randn(self.T, self.HC * self.H, generator=torch.Generator().manual_seed(8))
        with torch.no_grad():
            ref = mod(h[None], ids[None], None)[0]
            whole, _ = run(feat, h, ids, [self.T])
            chunked, _ = run(feat, h, ids, [13, 35])
            decoded, _ = run(feat, h, ids, [self.T - 6] + [1] * 6)
        self.close(whole, ref, 1e-5)
        self.close(chunked, whole, 1e-5)
        self.close(decoded, whole, 1e-5)


# ---------------------------------------------------------------------------------------------------------------------
# DeepSeek-V4.1's engram
# ---------------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(_vendor_present(), "requires the vendor DeepSeek-V4.1 inference code in .oracle-site/dsv41")
class EngramTests(Held):
    DIM, HC, HEAD_DIM, BLOCK, N, HEADS, BASE, LAYERS = 32, 4, 16, 4, 4, 2, 50, (1, 3)

    def setUp(self):
        self.vendor = vendor_engram()
        self.tokenizer = FakeTokenizer()
        token_map, vocab = self.vendor.build_compressed_token_map(self.tokenizer)
        primes = self.vendor.EngramLayout.from_args(self.args(vocab, (0, 0))).primes
        sums = tuple(sum(p for per in layer for p in per) for layer in primes)
        self.cfg = self.args(vocab, sums)
        self.layout = self.vendor.EngramLayout.from_args(self.cfg)
        self.state = self.vendor.NgramHashState(self.cfg, self.layout, self.tokenizer)
        self.token_map, self.vocab = torch.tensor(token_map), vocab

    def args(self, vocab, sums):
        return types.SimpleNamespace(engram_layer_ids=self.LAYERS, engram_max_ngram_size=self.N, engram_n_heads=self.HEADS,
                                     engram_vocab_size=self.BASE, engram_head_dim=self.HEAD_DIM, engram_num_embeddings=sums,
                                     engram_compressed_vocab_size=vocab, engram_pad_id=2, max_batch_size=2, max_seq_len=128)

    def hash(self, table_index):
        from engine.modules.ngram_embedding import NGramHash
        return NGramHash.rng(ngram_size=self.N, heads=self.HEADS, base=self.BASE, table_index=table_index,
                             layer_id=self.LAYERS[table_index], vocab=self.vocab, token_map=self.token_map,
                             pad=int(self.token_map[2]))

    def test_the_real_layout_and_multipliers_are_the_vendors(self):
        """DeepSeek-V4.1's own numbers: 16,000,000 base, 4-grams, 8 heads, layers 1 and 14, 99,092 keys -- the vendor's
        sympy walk and numpy draws against ours, and the per-layer sums against the checkpoint's table heights."""
        from engine.modules.ngram_embedding import NGramHash
        real = types.SimpleNamespace(engram_layer_ids=(1, 14), engram_max_ngram_size=4, engram_n_heads=8,
                                     engram_vocab_size=16_000_000, engram_head_dim=256,
                                     engram_num_embeddings=(384_006_168, 384_016_682))
        layout = self.vendor.EngramLayout.from_args(real)
        multipliers = self.vendor.compute_hash_multipliers((1, 14), 4, 99_092)
        for t, layer_id in enumerate((1, 14)):
            ours = NGramHash.rng(ngram_size=4, heads=8, base=16_000_000, table_index=t, layer_id=layer_id, vocab=99_092,
                                 token_map=None, pad=0)
            self.assertEqual(ours.sizes.tolist(), [p for per in layout.primes[t] for p in per])
            self.assertTrue(torch.equal(ours.multipliers, multipliers[t]))
            self.assertEqual(ours.total, real.engram_num_embeddings[t])

    def test_the_token_map_is_the_vendors(self):
        from engine.modules.ngram_embedding import normalized_token_map
        ours, vocab = normalized_token_map(self.tokenizer)
        self.assertEqual((ours, vocab), (self.token_map.tolist(), self.vocab))
        self.assertLess(vocab, len(self.tokenizer))                        # tokens did collapse
        self.assertEqual(len({ours[3], ours[4], ours[6], ours[11]}), 1)     # The, the, THE, Ｔｈｅ

    def test_the_hash_ids_are_the_vendors(self):
        """Two sequences in one vendor batch; ours one sequence at a time, whole, in pieces carrying the context, and
        with an image span that n-grams must not reach across."""
        from engine.modules.ngram_embedding import DEAD
        g = torch.Generator().manual_seed(11)
        L = 40
        ids = torch.randint(0, len(self.tokenizer), (2, L), generator=g)
        mask = torch.ones(2, L, dtype=torch.bool)
        mask[1, 12:17] = False                                             # sequence 1 carries an image
        vendor_hashes = torch.cat([self.state(ids[:, :15], 0, mask[:, :15]), self.state(ids[:, 15:28], 15, mask[:, 15:28]),
                                   self.state(ids[:, 28:], 28, mask[:, 28:])], dim=1)
        whole = self.state(ids, 0, mask)
        self.assertTrue(torch.equal(vendor_hashes, whole))                  # the vendor's own chunks agree
        moved = (whole[1] != self.state(ids, 0)[1]).any(-1).any(-1)       # the span, and the three positions after it
        self.assertEqual(torch.nonzero(moved).flatten().tolist(), list(range(12, 20)))
        for t in range(2):
            made = self.hash(t)
            self.assertEqual(made.sizes.tolist(), [p for per in self.layout.primes[t] for p in per])
            for seq in range(2):
                dead = torch.cat([torch.zeros(3, dtype=torch.bool), ~mask[seq]])
                history = torch.cat([torch.full((3,), DEAD), ids[seq]])
                self.assertTrue(torch.equal(made.rows(history, L, dead), whole[seq, :, t]))
                carried, pieces = torch.full((3,), DEAD), []
                for lo, hi in ((0, 15), (15, 28), (28, L)):
                    hist = torch.cat([carried, ids[seq, lo:hi]])
                    d = torch.cat([carried == DEAD, ~mask[seq, lo:hi]])
                    pieces.append(made.rows(hist, hi - lo, d))
                    carried = torch.where(d, torch.full_like(hist, DEAD), hist)[-3:]
                self.assertTrue(torch.equal(torch.cat(pieces), whole[seq, :, t]))

    def table(self, rows, seed=12):
        g = torch.Generator().manual_seed(seed)
        weight = (torch.randn(rows, self.HEAD_DIM, generator=g) * 3).to(torch.float8_e4m3fn)
        scale = torch.randint(120, 130, (rows, self.HEAD_DIM // self.BLOCK), generator=g, dtype=torch.uint8)
        return weight, scale.view(torch.float8_e8m0fnu)

    def engram(self, t, dtype):
        _, Engram = vendor_model_classes(self.BLOCK)
        args = types.SimpleNamespace(dim=self.DIM, hc_mult=self.HC, norm_eps=1e-6)
        mod = Engram(args, self.LAYERS[t], self.layout)
        weight, scale = self.table(self.layout.num_embeddings[t], seed=20 + t)
        with torch.no_grad():
            mod.embed.weight = nn.Parameter(weight, requires_grad=False)
            mod.embed.scale = nn.Parameter(scale, requires_grad=False)
        draw(mod, seed=30 + t, unit_offset=False)
        return mod.to(dtype) if dtype != torch.float32 else mod

    def feature(self, mod, t, dtype, table):
        from engine.modules.ngram_embedding import VARIANTS, NGramInjection, named
        params = {"wkv": mod.wkv.weight.detach(), "q_weight": mod.q_weight.detach(), "k_weight": mod.k_weight.detach()}
        return NGramInjection(hidden=self.DIM, hc=self.HC, ngram_size=self.N, eps=1e-6, dtype=dtype, **VARIANTS["engram"],
                              hash=lambda layer: self.hash(t), weights=lambda layer, name: named("dsv41", params.__getitem__)(name),
                              table=table)

    def test_the_lookup_is_the_vendors(self):
        from engine.modules.ngram_embedding import block_fp8_rows
        mod = self.engram(0, torch.float32)
        rows = torch.randint(0, self.layout.num_embeddings[0], (40, 6), generator=torch.Generator().manual_seed(13))
        with torch.no_grad():
            self.assertTrue(torch.equal(block_fp8_rows(mod.embed.weight, mod.embed.scale, rows, self.BLOCK), mod.embed(rows)))

    def test_the_write_is_the_vendors_at_fp32(self):
        """The vendor Engram.forward after its lookup, verbatim at fp32 (the lookup's bf16 cast is the test above):
        its embed returns the fp32 dequantised rows, ours reads the same rows."""
        from engine.modules.ngram_embedding import DEAD
        L = 40
        ids = torch.randint(0, len(self.tokenizer), (1, L), generator=torch.Generator().manual_seed(14))
        hashes = self.state(ids, 0)
        for t in range(2):
            mod = self.engram(t, torch.float32)
            weight, scale = mod.embed.weight.detach(), mod.embed.scale.detach()

            def fp32_rows(rows, weight=weight, scale=scale):
                v = torch.nn.functional.embedding(rows, weight).float().unflatten(-1, (-1, self.BLOCK))
                return (v * torch.nn.functional.embedding(rows, scale).float().unsqueeze(-1)).flatten(-2)

            class Rows(nn.Module):
                def forward(self, rows):
                    return fp32_rows(rows)
            mod.embed = Rows()
            x = torch.randn(1, L, self.HC, self.DIM, generator=torch.Generator().manual_seed(15 + t))
            feat = self.feature(mod, t, "float32", lambda layer, rows: fp32_rows(rows).view(*rows.shape, self.HEAD_DIM))
            with torch.no_grad():
                ref = mod(x, hashes[:, :, t, :])[0].flatten(-2)
                h = x[0].flatten(-2)
                whole, _ = run(feat, h, ids[0], [L])
                pieces, _ = run(feat, h, ids[0], [17, 23])
            self.assertTrue(torch.equal(h + whole, ref))
            self.close(pieces, whole, 1e-6)
            self.assertGreater(float(whole.abs().mean()), 0.1 * float(h.abs().mean()))     # the memory writes something

    def test_the_served_bf16_layer_is_within_its_rounding(self):
        """The vendor verbatim in bf16 (its lookup casts to bf16; wkv and the stream bf16) against ours at bf16: the vendor
        adds in fp32 and rounds once, ours rounds the addend and adds -- two bf16 rounding steps apart."""
        from engine.modules.ngram_embedding import block_fp8_rows
        L = 40
        ids = torch.randint(0, len(self.tokenizer), (1, L), generator=torch.Generator().manual_seed(16))
        hashes = self.state(ids, 0)
        mod = self.engram(1, torch.float32)
        mod.wkv.to(torch.bfloat16)
        weight, scale = mod.embed.weight.detach(), mod.embed.scale.detach()
        feat = self.feature(mod, 1, "bfloat16",
                            lambda layer, rows: block_fp8_rows(weight, scale, rows, self.BLOCK).view(*rows.shape, self.HEAD_DIM))
        x = torch.randn(1, L, self.HC, self.DIM, generator=torch.Generator().manual_seed(17)).to(torch.bfloat16)
        with torch.no_grad():
            ref = mod(x, hashes[:, :, 1, :])[0].flatten(-2)
            h = x[0].flatten(-2)
            addend, _ = run(feat, h, ids[0], [L])
        got = h + addend
        bound = (ref.float().abs() + addend.float().abs()) * 2.0 ** -6 + 1e-6            # two ulps of 7-bit mantissas
        self.assertTrue(bool(((got.float() - ref.float()).abs() <= bound).all()))
        self.assertFalse(torch.equal(addend, torch.zeros_like(addend)))


# ---------------------------------------------------------------------------------------------------------------------
# the axes
# ---------------------------------------------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, "requires torch")
class AxesTests(unittest.TestCase):
    def test_primes_are_consecutive_and_exact(self):
        from engine.modules.ngram_embedding import _is_prime, consecutive_primes, head_tables
        sieve = [True] * 3000
        sieve[0] = sieve[1] = False
        for i in range(2, 55):
            if sieve[i]:
                sieve[i * i::i] = [False] * len(sieve[i * i::i])
        want = [i for i in range(1000, 3000) if sieve[i]][:40]
        self.assertEqual(list(consecutive_primes(1000, 40)), want)
        for carmichael in (561, 1105, 1729, 2465, 2821, 6601, 8911, 3215031751, 3825123056546413051):
            self.assertFalse(_is_prime(carmichael))
        self.assertTrue(_is_prime(2 ** 61 - 1) and _is_prime(16000057) and not _is_prime(16000059))
        sizes, offsets, total = head_tables(3, 4, 1000, 1)
        self.assertEqual(sizes.tolist(), want[8:16])
        self.assertEqual(offsets.tolist(), [sum(want[8:8 + i]) for i in range(8)])
        self.assertEqual(total, sum(want[8:16]))

    def test_the_window_rule(self):
        from engine.modules.ngram_embedding import DEAD, ngram_windows
        E = 9
        history = torch.tensor([DEAD, DEAD, 5, E, 7, 8])                  # carried context, then a, EOS, b, c
        tokens, blocked = ngram_windows(history, 4, 3, eos=E)
        self.assertEqual(tokens[:, 0].tolist(), [5, E, 7, 8])
        self.assertEqual(blocked.tolist(), [[False, True, True],           # a: the sequence start
                                            [False, False, True],          # EOS: its own segment reaches back to a
                                            [False, True, True],           # b: the EOS one place back closes it
                                            [False, False, True]])         # c: b, then the EOS
        _, blocked = ngram_windows(history, 4, 3, eos=None)                # no document boundary
        self.assertEqual(blocked[:, 1].tolist(), [True, False, False, False])
        dead = torch.tensor([DEAD, DEAD, 5, DEAD, 7, 8])                   # a dead token blocks at any lookback
        _, blocked = ngram_windows(dead, 4, 3)
        self.assertEqual(blocked.tolist(), [[False, True, True], [True, True, True], [False, True, True], [False, False, True]])

    def test_the_axes_are_checked(self):
        from engine.modules.ngram_embedding import SCHEMES, VARIANTS, NGramHash, NGramInjection, named
        ok = dict(hidden=8, hc=2, ngram_size=3, eps=1e-6, hash=None, weights=None, table=None)
        NGramInjection(**ok)
        for bad in (dict(norm_order="late"), dict(signed_sqrt="abs"), dict(ngram_size=1), dict(conv=1), dict(hc=0)):
            with self.assertRaises(ValueError):
                NGramInjection(**{**ok, **bad})
        self.assertEqual(set(VARIANTS), {"ple", "engram"})
        self.assertEqual(set(SCHEMES), {"qwen4_exp", "dsv41"})
        with self.assertRaises(KeyError):
            named("dsv41", lambda hf: None)("key")
        with self.assertRaises(ValueError):
            NGramHash.rng(ngram_size=3, heads=2, base=100, table_index=0, layer_id=1, vocab=7,
                          token_map=torch.tensor([0, 1, 2]), pad=0)
        feat = NGramInjection(**{**ok, "conv": 4, "dtype": "bfloat16"})
        self.assertEqual([(s.key, s.bytes_per_seq) for s in feat.cache_specs([0])],
                         [("ngram_context", 2 * 8), ("ngram_conv", 2 * 8 * 3 * 3 * 2)])
        self.assertEqual([s.key for s in NGramInjection(**ok).cache_specs([0])], ["ngram_context"])
        wrong = NGramInjection(**{**ok, "hash": lambda layer: NGramHash.splitmix(ngram_size=4, heads=1, unigram_vocab=9,
                                                                                  base=10, table_index=0, seed=1, eos=0)})
        with self.assertRaises(ValueError):
            wrong.hashes(0)


if __name__ == "__main__":
    unittest.main()
