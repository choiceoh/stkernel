"""engine/profiles/qwen38/mtp_tune: the MTP head fine-tuned on the target's own streams, the served chain teacher-forced.

Held on the CPU over the small Qwen3.8-shaped composition with a random MTP head (tests/test_engine_mtp.py's):

- the tuning's chain IS the reference head's: depth 1 over a window's positions, depth d at the next position from
  the head's own streams and the text's token, each against engine/modules/mtp's head (profiles/qwen38/composition
  .build_mtp) run through base/composition's State the way the drafter runs it -- logits within fp32 rounding;
- the loss: KL to the target's own distribution per depth, weighted, gradients on the dense weights only (the experts,
  the embedding, the head and the target's mixer are frozen), Spec-AUF's cut, the chain metrics;
- the data: fleet --tap-mtp-inputs shards (MTPInputTap) -> contiguous runs a sequence, a later record of a position
  winning, two boots' sequence ids kept apart, a held-out split by sequence, windows drawn from the runs;
- the head's served names an export writes, and the boot's flag.
"""
import importlib.util
import json
import random
import tempfile
import time
import unittest
from pathlib import Path

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

ROOT = Path(__file__).resolve().parents[1]


def tiny(seed=0, budget=64):
    """The composition and its MTP head over random weights, the QSA budget wide enough that a test's windows are
    covered (the tuning's attention selects nothing)."""
    from engine.profiles.qwen38 import composition as qc
    from engine.profiles.qwen38.weights import random_weights
    from tests.test_engine_composed import TINY
    cfg = dict(TINY, ple_conv_kernel_size=4, mtp_num_hidden_layers=1, indexer_budget=budget)
    weights = random_weights(cfg, seed)
    return qc.build(cfg, weights.__getitem__), qc.build_mtp(cfg, weights.__getitem__)[0], cfg, weights


@unittest.skipUnless(torch is not None, "requires torch")
class ChainTests(unittest.TestCase):
    def test_every_depth_is_the_reference_heads(self):
        from engine.base.composition import State, Step
        from engine.profiles.qwen38.mtp_tune import Head
        from tests.test_engine_composed import prompt
        comp, head, cfg, weights = tiny()
        tokens = prompt(5, 24)
        L, T, depth = 20, 12, 3
        with torch.no_grad():
            _, streams = comp.forward(Step.of([(0, 0, torch.tensor(tokens[:L]))]), State(), logits="all", hidden=True)
            model = Head(cfg, weights.__getitem__, prefix="model.", dtype=torch.float32)
            nxt = torch.tensor(tokens[1:])                                  # the token after each position
            hidden, _ = model.chain(streams[:T], nxt, 0, depth)
            mine = hidden @ weights["lm_head.weight"].T                     # [depth, T, vocab]
            for i in (0, 4, T - 1):
                state = State()
                logits, own = head.forward(Step.of([(0, 0, torch.tensor(tokens[1:i + 2]))]), state, logits="all",
                                           hidden=True, given=streams[:i + 1])
                want = [logits[i]]
                given = own[i:i + 1]
                for d in range(2, depth + 1):                               # the chain, teacher-forced
                    step_logits, given = head.forward(Step.of([(0, i + d - 1, torch.tensor([tokens[i + d]]))]), state,
                                                      hidden=True, given=given)
                    want.append(step_logits[0])
                for d in range(depth):
                    with self.subTest(start=i, depth=d + 1):
                        self.assertTrue(torch.allclose(mine[d, i], want[d], rtol=1e-4, atol=1e-5),
                                        float((mine[d, i] - want[d]).abs().max()))

    def test_a_window_past_the_covered_reach_is_refused(self):
        from engine.profiles.qwen38.mtp_tune import Head
        _, _, cfg, weights = tiny(budget=8)                                 # covers 11 positions
        model = Head(cfg, weights.__getitem__, dtype=torch.float32)
        width = cfg["hc_count"] * cfg["hidden_size"]
        with self.assertRaisesRegex(ValueError, "passes the 11"):
            model.chain(torch.zeros(10, width), torch.zeros(20, dtype=torch.int64), 0, 3)


@unittest.skipUnless(torch is not None, "requires torch")
class LossTests(unittest.TestCase):
    def window(self, T=10, depth=3):
        from engine.base.composition import State, Step
        from tests.test_engine_composed import prompt
        comp, _, cfg, weights = tiny()
        tokens = prompt(7, T + depth + 1)
        with torch.no_grad():
            _, streams = comp.forward(Step.of([(0, 0, torch.tensor(tokens[:T + depth]))]), State(), logits="all",
                                      hidden=True)
        return cfg, weights, streams, torch.tensor(tokens[1:T + depth + 1])

    def test_the_gradient_reaches_the_dense_weights_only(self):
        from engine.profiles.qwen38.mtp_tune import EXPERTS, TRAINED, Head, _key, window_loss
        cfg, weights, streams, tokens = self.window()
        model = Head(cfg, weights.__getitem__, dtype=torch.float32)
        loss, metrics = window_loss(model, streams, tokens, 0, 3, chunk=4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and float(loss.detach()) > 0)
        for name in TRAINED:
            grad = model.weights[_key(name)].grad
            self.assertIsNotNone(grad, name)
        self.assertGreater(sum(float(model.weights[_key(n)].grad.abs().sum()) for n in TRAINED), 0)
        for name in EXPERTS:
            self.assertFalse(model.frozen[name].requires_grad)
        self.assertEqual(set(metrics) >= {"loss_1", "agree_3", "chain_3", "text_chain_1", "tokens_a_step"}, True)
        self.assertLessEqual(metrics["chain_3"], metrics["chain_2"])
        self.assertLessEqual(metrics["chain_2"], metrics["chain_1"])
        self.assertAlmostEqual(metrics["tokens_a_step"],
                               1 + metrics["chain_1"] + metrics["chain_2"] + metrics["chain_3"], places=6)

    def test_the_loss_is_zero_where_the_head_is_the_target(self):
        """KL(target || head) of a head whose hidden is the target's own: 0 -- the labels are the target's
        distribution at the position the draft predicts (depth d, row i: its streams at i + d)."""
        from engine.profiles.qwen38.mtp_tune import kl_chunk
        g = torch.Generator().manual_seed(1)
        h, head = torch.randn(6, 8, generator=g), torch.randn(30, 8, generator=g)
        self.assertTrue(torch.allclose(kl_chunk(h, h, head), torch.zeros(6), atol=1e-6))
        self.assertTrue(bool((kl_chunk(h, torch.randn(6, 8, generator=g), head) > 0).all()))

    def test_the_cut_counts_a_depth_only_where_the_depths_before_it_were_kept(self):
        from engine.profiles.qwen38.mtp_tune import Head, window_loss
        cfg, weights, streams, tokens = self.window()
        model = Head(cfg, weights.__getitem__, dtype=torch.float32)
        with torch.no_grad():
            plain, m = window_loss(model, streams, tokens, 0, 3)
            cut, m_cut = window_loss(model, streams, tokens, 0, 3, auf=True)
        self.assertEqual(m["loss_1"], m_cut["loss_1"])                     # depth 1 has no depth before it
        if m["chain_1"] < 1.0:
            self.assertNotEqual(m["loss_2"], m_cut["loss_2"])


@unittest.skipUnless(torch is not None, "requires torch")
class DataTests(unittest.TestCase):
    def test_tapped_shards_become_runs_a_sequence(self):
        from engine.profiles.qwen38.fleet import MTPInputTap
        from engine.profiles.qwen38.mtp_tune import Runs, build_runs, shards
        width = 8
        with tempfile.TemporaryDirectory() as d:
            taps = Path(d) / "taps"
            tap = MTPInputTap(taps, rows=16, every_s=0.2)
            rows = lambda n, v: torch.full((n, width), float(v), dtype=torch.bfloat16)
            tap(1, 0, list(range(100, 140)), rows(40, 1), False)             # a prompt of 40 positions
            tap(1, 40, [140, 141], rows(2, 2), True)                         # a verify step kept two
            tap(2, 0, list(range(200, 205)), rows(5, 3), False)              # a short sequence
            tap(1, 41, [999], rows(1, 4), True)                              # position 41 again: the later wins
            tap(1, 42, [142], rows(1, 5), True)
            import numpy as np
            written = lambda: sum(np.load(f)["meta"].shape[0] for f in shards([taps]))
            deadline = time.time() + 30          # the timer is 0.2 s; the budget is for a loaded box (check.py
            while time.time() < deadline and written() < 49:   # runs four of these at once, and a daemon thread waits)
                time.sleep(0.1)
            files = shards([taps])
            self.assertEqual(written(), 49)
            self.assertGreaterEqual(len(files), 2)                             # a full shard, then the timer's
            index = build_runs(files, Path(d) / "data", holdout=0.0, min_length=8)
            self.assertEqual([(r["seq"], r["start"], r["length"], r["decoded"]) for r in index["runs"]], [(1, 0, 43, 3)])
            run = np.load(Path(d) / "data" / index["runs"][0]["file"])
            self.assertEqual(run["tokens"].tolist()[-3:], [140, 999, 142])
            got = torch.from_numpy(run["streams"]).view(torch.bfloat16)
            self.assertEqual(got[:, 0].tolist()[38:], [1.0, 1.0, 2.0, 4.0, 5.0])
            windows = list(Runs(Path(d) / "data", "train", window=16, depth=3).windows(random.Random(0), 5))
            for streams, tokens, start in windows:
                self.assertEqual((streams.shape, tokens.shape), ((19, width), (19,)))
                self.assertTrue(0 <= start <= 43 - 19)

    def test_two_boots_keep_their_sequences_apart_and_a_sequence_is_held_out_whole(self):
        import numpy as np
        from engine.profiles.qwen38.mtp_tune import build_runs
        with tempfile.TemporaryDirectory() as d:
            for boot in ("20260919-100000", "20260919-110000"):
                meta = np.array([[7, p, 1000 + p, 0] for p in range(40)], dtype=np.int64)
                np.savez(Path(d) / f"mtp-inputs-{boot}-00000.npz", streams=np.zeros((40, 4), np.int16), meta=meta)
            index = build_runs(sorted(Path(d).glob("*.npz")), Path(d) / "data", holdout=0.5, min_length=8, seed=3)
            self.assertEqual(len(index["runs"]), 2)
            self.assertEqual({r["boot"] for r in index["runs"]}, {"20260919-100000", "20260919-110000"})
            self.assertEqual(index["train"] + index["eval"], 80)

    def test_a_reader_never_sees_a_half_written_shard(self):
        """The trainer reads the tap's directory WHILE the fleet writes it (`shards()` globs it, `data` loads what it
        returns). A `np.savez` straight to the final name is a zip a reader can catch half-written -- an EOFError in
        the middle of a data window. Here the shard's bytes are handed over in two halves with the writer stopped in
        between: what the glob returns while it is stopped must be nothing, and what it returns after must load."""
        import io
        import threading
        from unittest.mock import patch
        import numpy as np
        from engine.profiles.qwen38.fleet import MTPInputTap
        from engine.profiles.qwen38.mtp_tune import shards

        inside, release, real = threading.Event(), threading.Event(), np.savez

        def halfway(file, **arrays):
            """The real bytes, written half now and half after the test looks -- a write caught in the middle."""
            buffer = io.BytesIO()
            real(buffer, **arrays)
            data = buffer.getvalue()
            handle = file if hasattr(file, "write") else open(file, "wb")
            handle.write(data[:len(data) // 2]); handle.flush()
            inside.set()
            release.wait(10)
            handle.write(data[len(data) // 2:]); handle.flush()
            if handle is not file:
                handle.close()

        with tempfile.TemporaryDirectory() as d, patch("numpy.savez", halfway):
            taps = Path(d) / "taps"
            tap = MTPInputTap(taps, rows=2, every_s=0.2)
            tap(7, 0, [11, 12], torch.zeros(2, 8, dtype=torch.bfloat16), True)
            self.assertTrue(inside.wait(10), "the writer never reached the shard")
            self.assertEqual([f.name for f in shards([taps])], [])        # half of a zip is not a shard
            self.assertTrue([f for f in taps.iterdir() if f.name.endswith(".part")])
            release.set()
            deadline = time.time() + 30
            while time.time() < deadline and not shards([taps]):
                time.sleep(0.05)
            files = shards([taps])
            self.assertEqual(len(files), 1)
            self.assertEqual(np.load(files[0])["meta"].shape[0], 2)       # complete the moment it is visible
            self.assertEqual([f for f in taps.iterdir() if f.name.endswith(".part")], [])


@unittest.skipUnless(torch is not None, "requires torch")
class ServedTests(unittest.TestCase):
    def test_the_export_replaces_the_heads_served_dense_tensors(self):
        from engine.profiles.qwen38.mtp_tune import LAYOUT, TRAINED, tuned_file
        source = (ROOT / "engine/profiles/qwen38/mtp_tune.py").read_text()
        self.assertIn("layout.mtp_specs(F, routed=False)", source)
        self.assertEqual(tuned_file(2), "mtp-tuned-r2of4.safetensors")
        self.assertEqual(LAYOUT, "qwen38-mtp-tuned-v1")
        self.assertEqual(len(TRAINED), len(set(TRAINED)))
        self.assertFalse(any("experts" in n or "indexer" in n for n in TRAINED))

    def test_a_boot_serves_a_tuned_head_from_its_side_files(self):
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn("tuned = set(mtp_tune.served_names(F)) & {s.name for s in specs}", fleet)
        self.assertIn("rank_loader(tuned_path, expected_layout=mtp_tune.LAYOUT)", fleet)
        self.assertIn("if s.name not in side and s.name not in tuned", fleet)
        self.assertIn("mtp_tuned_dir=a.mtp_tuned", fleet)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text()
        self.assertIn('EXPERTS_ARG="$EXPERTS_ARG --mtp-tuned $TUNED_DIR"', launcher)
        self.assertIn("test -f $TUNED_DIR/mtp-tuned-r${r}of4.safetensors", launcher)

    def test_the_tap_is_rank_zeros_and_off_by_default(self):
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn("if a.tap_mtp_inputs and comm.rank == 0 and model.drafter is not None:", fleet)
        adapter = (ROOT / "engine/profiles/qwen38/adapter.py").read_text()
        self.assertIn("self.inputs_tap = None", adapter)
        self.assertIn("hidden[segment.start:segment.start + fed], decoded=True)", adapter)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text()
        self.assertIn('1) ADAPT_ARG="$ADAPT_ARG --tap-mtp-inputs" ;;', launcher)


if __name__ == "__main__":
    unittest.main()
