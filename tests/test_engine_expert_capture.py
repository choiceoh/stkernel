"""The expert-capture measurement boot (engine/profiles/glm53/capture.py): what the CPU can pin -- the latent variants
and their arithmetic, the file format, and the hooks' plumbing over a stand-in composition that calls them the way
net.py does (route inside the MoE block, the kv_a_norm and the sparse MLA inside a DSA block, the full-row prefill)."""
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine.profiles.glm53 import capture as cap  # noqa: E402
from engine.modules.sparse_attention import mla_sparse_mqa  # noqa: E402

E4M3 = torch.float8_e4m3fn


class LatentTests(unittest.TestCase):
    def test_served_is_the_engine_s_cast(self):
        rows = torch.randn(9, 512, dtype=torch.bfloat16) * 0.3
        self.assertTrue(torch.equal(cap.served_latent(rows), rows.to(E4M3).float()))

    def test_static_scale_saturates_and_refines(self):
        rows = torch.tensor([[0.01, -0.02, 1000.0, -1000.0] * 128], dtype=torch.float32)
        out = cap.static_latent(rows, 0.25)
        self.assertEqual(float(out.max()), 448.0 * 0.25)                    # saturated, not NaN
        self.assertEqual(float(out.min()), -448.0 * 0.25)
        # e4m3 is a float: a scale changes nothing for normal values (eight steps an octave either way) and only
        # rescues what sits below 2^-6, where the grid is a fixed 2^-9
        small = torch.full((1, 512), 0.003)
        err1 = float((cap.served_latent(small) - small).abs().max())
        err8 = float((cap.static_latent(small, 2.0 ** -8) - small).abs().max())
        self.assertAlmostEqual(err1, 2 * 2.0 ** -9 - 0.003, places=6)        # 0.003 -> two subnormal steps
        self.assertLess(err8, err1 / 8)
        normal = torch.full((1, 512), 0.3)
        self.assertAlmostEqual(float((cap.served_latent(normal) - normal).abs().max()),
                               float((cap.static_latent(normal, 2.0 ** -4) - normal).abs().max()), places=6)

    def test_dynamic_scales_bound_the_error_per_group(self):
        torch.manual_seed(0)
        rows = torch.randn(33, 512) * torch.logspace(-3, 1, 33)[:, None]
        for width in (512, 128):
            out = cap.dynamic_latent(rows, width)
            groups = rows.view(33, 512 // width, width)
            step = groups.abs().amax(-1) / 448.0                            # the grid's coarsest spacing is 64 steps of the scale
            err = (out.view_as(groups) - groups).abs().amax(-1)
            self.assertTrue(bool((err <= step * 64 / 2 * 1.001 + 1e-12).all()), width)
        variants = cap.latent_variants(rows.to(torch.bfloat16))
        self.assertEqual(set(variants), {"bf16", "served", "row", "tile"} | {f"pow2_{k}" for k in cap.POW2_SHIFTS})

    def test_positions_of_inverts_a_slot_table(self):
        torch.manual_seed(1)
        table = torch.randperm(5000)[:700] + 11
        pos = torch.tensor([0, 699, 350, 5])
        self.assertTrue(torch.equal(cap.positions_of(table[pos], table), pos))
        self.assertEqual(int(cap.positions_of(torch.tensor([3]), table)[0]), -1)   # below every slot: absent

    def test_magnitude_histogram_octaves(self):
        x = torch.tensor([0.0, 2.0 ** -20, 1.0, 1.5, 2.0 ** -16, 2.0 ** 13])
        h = cap.magnitude_histogram(x)
        self.assertEqual(h.numel(), cap.LOG2_HIGH - cap.LOG2_LOW + 2)
        self.assertEqual(int(h[0]), 2)                                       # zero and 2^-20: underflow
        self.assertEqual(int(h[1]), 1)                                       # 2^-16: the first octave
        self.assertEqual(int(h[1 - cap.LOG2_LOW]), 2)                        # 1.0 and 1.5: octave [1, 2)
        self.assertEqual(int(h[-1]), 1)                                      # 2^13: overflow
        self.assertEqual(int(h.sum()), 6)


class HeadTests(unittest.TestCase):
    def test_block_logits_are_the_product(self):
        torch.manual_seed(2)
        hs, head = torch.randn(5, 64, dtype=torch.bfloat16), torch.randn(301, 64, dtype=torch.bfloat16)
        self.assertTrue(torch.allclose(cap.bf16_logits(hs, head, block=100), hs.float() @ head.float().T, atol=1e-5))

    def test_metrics(self):
        torch.manual_seed(3)
        logits = torch.randn(7, 50)
        targets = torch.randint(0, 50, (7,))
        same = cap.head_metrics(logits, logits, targets, 50)
        self.assertEqual((same["n"], same["top1_agree"]), (7, 7.0))
        self.assertAlmostEqual(same["kl"], 0.0, places=5)
        self.assertAlmostEqual(same["nll_fp8"], same["nll_bf16"], places=5)
        other = cap.head_metrics(logits + torch.randn(7, 50), logits, targets, 40)   # undecodable tail ignored
        self.assertGreater(other["kl"], 0.0)

    def test_position_metrics_are_the_sums_and_the_record_is_ordered(self):
        torch.manual_seed(4)
        fp8, bf16 = torch.randn(5, 30), torch.randn(5, 30)
        targets = torch.randint(0, 30, (5,))
        per = cap.head_position_metrics(fp8, bf16, targets, 30)
        sums = cap.head_metrics(fp8, bf16, targets, 30)
        self.assertAlmostEqual(float(per["nll_fp8"].sum()), sums["nll_fp8"], places=4)
        self.assertAlmostEqual(float(per["kl"].sum()), sums["kl"], places=4)
        expected = -torch.log_softmax(fp8, -1)[torch.arange(5), targets]
        self.assertTrue(torch.allclose(per["nll_fp8"], expected, atol=1e-5))
        positions = torch.tensor([40, 12, 33, 7, 21])
        rec = cap.head_record(3, 1, 7, positions, targets, per)
        self.assertEqual(rec["pos"], [7, 12, 21, 33, 40])
        order = torch.argsort(positions)
        self.assertEqual(rec["target"], targets[order].tolist())
        self.assertAlmostEqual(rec["nll_bf16"][0], float(per["nll_bf16"][order[0]]), places=5)
        self.assertEqual((rec["doc"], rec["seq"], rec["ctx"]), (3, 1, 7))


class FileTests(unittest.TestCase):
    def test_safetensors_roundtrip(self):
        from safetensors import safe_open
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.safetensors"
            x = torch.randn(17, 8, dtype=torch.bfloat16)
            sel = torch.randint(0, 288, (17, 8), dtype=torch.int16)
            w = torch.rand(17, 8)
            n = cap.write_safetensors(path, dict(x=x, sel=sel, w=w, empty=torch.zeros(0, 8)), dict(layer=3, seq=9))
            self.assertEqual(n, path.stat().st_size)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o666)
            with safe_open(str(path), framework="pt") as f:
                self.assertEqual(f.metadata(), {"layer": "3", "seq": "9"})
                self.assertTrue(torch.equal(f.get_tensor("x"), x))
                self.assertTrue(torch.equal(f.get_tensor("sel"), sel))
                self.assertTrue(torch.equal(f.get_tensor("w"), w))
                self.assertEqual(tuple(f.get_tensor("empty").shape), (0, 8))

    def test_writer_bounds_what_is_in_flight(self):
        with tempfile.TemporaryDirectory() as d:
            writer = cap.Writer(limit_bytes=100)
            for i in range(5):
                self.assertTrue(writer.put(Path(d) / f"{i}.safetensors", dict(v=torch.zeros(40)), {}, 160))
                self.assertLessEqual(writer.pending, 160)                   # one oversized item at a time, never two
            writer.close()
            self.assertEqual((writer.files, writer.dropped, writer.pending), (5, 0, 0))


# -- a stand-in composition ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FakeLanes:
    mla_sparse: object


@dataclass
class FakeFacts:
    hidden: int = 64
    topk_experts: int = 4
    vocab: int = 4 * 50
    layers: int = 3
    rms_eps: float = 1e-6
    mla_scale: float = 512 ** -0.5

    def is_moe(self, L):
        return L >= 1

    def is_dsa(self, L):
        return L in (0, 2)


@dataclass(frozen=True)
class Segment:
    seq: int
    slot: int
    ctx: int
    start: int
    length: int


@dataclass(frozen=True)
class Step:
    ids: torch.Tensor
    segments: tuple


class FakeComm:
    rank, world_size = 3, 4
    gathers = 0

    def all_gather(self, t, dim=-1):
        self.gathers += 1
        return torch.cat([t] * 4, dim=dim)


class FakeCaches:
    """Latent slots that are not positions: slot = 1000 + 3 * position (per sequence offset)."""
    device = torch.device("cpu")

    def __init__(self):
        self.latent = {L: torch.zeros(1000 + 3 * 4096 + 64, 512, dtype=E4M3) for L in (0, 2)}

    def token_slots(self, layer, seq, positions):
        return (1000 + 3 * positions + seq).to(torch.int32)


class FakeNet:
    def __init__(self):
        self.F = FakeFacts()
        self.comm = FakeComm()
        self.layers = [0, 1, 2]
        torch.manual_seed(4)
        # a small norm weight puts the latent below e4m3's normal range, where the variants differ
        self.p = {f"L{L}.mla.kv_a_norm": (torch.rand(512) + 0.5) * 0.01 for L in (0, 2)}
        self.p.update({f"L{L}.moe.gate": torch.randn(288, 64) for L in (1, 2)})
        self.p["kv"] = torch.randn(512, 64) * 0.05
        self.p["q"] = torch.randn(4 * 512, 64) * 0.2
        self.head_shard = torch.randn(50, 64, dtype=torch.bfloat16)
        self._norm = lambda x, w, eps: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w
        self.lanes = FakeLanes(mla_sparse=mla_sparse_mqa)
        self.routes = []
        self._dense = lambda L, x, reduce=None: x * 0.5            # the modelopt build binds its dense path per instance

    def route(self, L, x):
        s = torch.sigmoid(x.float() @ self.p[f"L{L}.moe.gate"].T)
        sel = s.topk(self.F.topk_experts, dim=-1).indices
        w = s.gather(-1, sel)
        self.routes.append(L)
        return sel.to(torch.int32), w / w.sum(-1, keepdim=True)

    def _dsa(self, L, x, step, caches, reduce=None, *, project=None):
        s = step.segments[0]
        kv_n = self._norm((x.float() @ self.p["kv"].T).to(torch.bfloat16), self.p[f"L{L}.mla.kv_a_norm"], self.F.rms_eps)
        positions = torch.arange(s.ctx, s.ctx + s.length)
        latent = caches.latent[L]
        latent[caches.token_slots(L, s.seq, positions).long()] = kv_n.to(E4M3)
        g = torch.Generator().manual_seed(L * 7 + s.ctx)
        width = 24
        slots = torch.full((s.length, width), -1, dtype=torch.int32)
        valid = torch.zeros(s.length, dtype=torch.int32)
        for r, p in enumerate(positions.tolist()):
            chosen = torch.randperm(p + 1, generator=g)[:width].sort(descending=True).values
            slots[r, : chosen.numel()] = caches.token_slots(L, s.seq, chosen)
            valid[r] = chosen.numel()
        q_abs = (x.float() @ self.p["q"].T).view(-1, 4, 512).to(torch.bfloat16)
        out = self.lanes.mla_sparse(q_abs.contiguous(), latent, slots, valid, self.F.mla_scale, 1.0)
        return out.float().mean(1)[:, :64].to(x.dtype)

    def _moe(self, L, x):
        sel, w = self.route(L, x)
        return x * w.sum(-1, keepdim=True).to(x.dtype)

    def head_local(self, h):
        return (h.float() @ self.head_shard.float().T).to(torch.bfloat16)

    def forward(self, step, caches, last_hidden_only=False):
        x = torch.nn.functional.embedding(step.ids, torch.randn(200, 64, generator=torch.Generator().manual_seed(5))).to(torch.bfloat16)
        for L in self.layers:
            if self.F.is_dsa(L):
                x = x + self._dsa(L, x, step, caches)
            if self.F.is_moe(L):
                x = x + self._moe(L, x)
            else:
                x = x + self._dense(L, x)
        return x[-1:] if last_hidden_only else x


class FakeEngine:
    decodable = 200

    def __init__(self):
        self.net = FakeNet()
        self.caches = FakeCaches()

    def _forward(self, step, **kwargs):
        return self.net.forward(step, self.caches, **kwargs), None

    def _prefill_forward(self, step):
        h, aux = self._forward(step, last_hidden_only=True)
        return h[-1:], aux

    def prefill(self, seq, ids, ctx):
        return self._prefill_forward(Step(ids, (Segment(seq, 1, ctx, 0, ids.numel()),)))


class HookTests(unittest.TestCase):
    def test_capture_passes_numbers_through_and_records(self):
        torch.manual_seed(6)
        engine = FakeEngine()
        ids = torch.randint(0, 200, (300,))
        again = torch.randint(0, 200, (150,))

        def requests():
            # the runner hands the same sequence id to the next request (at C=1 the served door alternates two):
            # seq 7 is a two-chunk document, then another document
            return [engine.prefill(7, ids[:180], 0)[0], engine.prefill(7, ids[180:], 180)[0], engine.prefill(7, again, 0)[0]]
        reference = requests()
        engine = FakeEngine()
        net = engine.net
        with tempfile.TemporaryDirectory() as d:
            c = cap.Capture(engine, Path(d) / "capture", head_bf16=net.head_shard, kv_rows=8, head_rows=16,
                            limit_gib=1.0, disk_floor_gib=0.0, report_every=1)
            got = requests()
            for a, b in zip(reference, got):
                self.assertTrue(torch.equal(a, b))                          # the last row is the last row
            row = c.close()
            self.assertNotIn("_prefill_forward", engine.__dict__)            # detached: the class method again
            self.assertIs(net.lanes.mla_sparse, mla_sparse_mqa)
            self.assertNotIn("route", net.__dict__)
            self.assertEqual(row["errors"], {})
            self.assertEqual((row["captured_chunks"], row["captured_tokens"]), (3, 450))
            from safetensors import safe_open
            for L in (1, 2):
                files = sorted((Path(d) / "capture" / "moe" / f"L{L:02d}").iterdir())
                self.assertEqual([f.name for f in files], ["d000000-c0000000.safetensors", "d000000-c0000180.safetensors",
                                                           "d000001-c0000000.safetensors"])   # a reused seq id overwrites nothing
                with safe_open(str(files[1]), framework="pt") as f:
                    self.assertEqual(tuple(f.get_tensor("x").shape), (120, 64))
                    self.assertEqual(f.get_tensor("sel").dtype, torch.int16)
                    self.assertEqual((f.metadata()["doc"], f.metadata()["seq"], f.metadata()["ctx"]), ("0", "7", "180"))
            chunks = [json.loads(line) for line in (Path(d) / "capture" / "chunks.jsonl").read_text().splitlines()]
            self.assertEqual([(r["doc"], r["seq"], r["ctx"], r["tokens"]) for r in chunks],
                             [(0, 7, 0, 180), (0, 7, 180, 120), (1, 7, 0, 150)])
            kv = row["kv"]
            self.assertEqual(set(kv), {"0", "2"})
            for L in ("0", "2"):
                self.assertEqual(kv[L]["skipped_rows"], 0)                  # every selected slot mapped back to a kept row
                self.assertEqual(kv[L]["query_rows"], 24)                   # eight a chunk, three chunks, both documents
                self.assertLess(kv[L]["kernel_vs_served_reference"], 0.02)  # the served lane's bytes are the reference's
                self.assertLess(kv[L]["out_tile"], kv[L]["out_served"])
            head = row["head"]
            self.assertEqual(head["positions"], 48)
            self.assertGreater(head["top1_agree"], 0.9)
            self.assertLess(head["kl"], 1e-3)
            records = [json.loads(line) for line in (Path(d) / "capture" / "head.jsonl").read_text().splitlines()]
            self.assertEqual([(r["doc"], r["seq"], r["ctx"], len(r["pos"])) for r in records],
                             [(0, 7, 0, 16), (0, 7, 180, 16), (1, 7, 0, 16)])
            second = records[1]
            self.assertEqual(second["pos"], sorted(second["pos"]))
            self.assertTrue(all(180 <= p < 299 for p in second["pos"]))       # absolute positions, the last row has no target
            self.assertEqual(second["target"], [int(ids[p + 1]) for p in second["pos"]])
            self.assertAlmostEqual(sum(sum(r["nll_fp8"]) for r in records) / 48, head["nll_fp8"], places=4)
            stats = [json.loads(line) for line in (Path(d) / "capture" / "stats.jsonl").read_text().splitlines()]
            self.assertEqual(len(stats), 4)                                  # every chunk (report_every=1) and the close

    def test_only_the_capture_rank_writes_and_a_cap_stops_whole_chunks(self):
        engine = FakeEngine()
        engine.net.comm.rank = 0
        with tempfile.TemporaryDirectory() as d:
            c = cap.Capture(engine, Path(d) / "capture", head_bf16=None, report_every=0)
            engine.prefill(1, torch.randint(0, 200, (64,)), 0)
            c.close()
            self.assertFalse((Path(d) / "capture").exists())
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as d:
            per_chunk = 100 * 2 * (64 * 2 + 4 * 6) + 100 * 4
            c = cap.Capture(engine, Path(d) / "capture", head_bf16=None, report_every=0, disk_floor_gib=0.0,
                            limit_gib=1.5 * per_chunk / 2**30)
            engine.prefill(1, torch.randint(0, 200, (100,)), 0)
            engine.prefill(2, torch.randint(0, 200, (100,)), 0)
            row = c.close()
            self.assertEqual(row["captured_chunks"], 1)
            self.assertIn("byte cap", row["stopped"])
            self.assertEqual(len(list((Path(d) / "capture" / "moe" / "L01").iterdir())), 1)

    def test_dense_section_writes_dense_rows_and_nothing_else(self):
        engine = FakeEngine()
        ids = torch.randint(0, 200, (140,))
        reference = engine.prefill(5, ids, 0)[0]
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as d:
            c = cap.Capture(engine, Path(d) / "capture", head_bf16=engine.net.head_shard, sections=("dense",),
                            limit_gib=1.0, disk_floor_gib=0.0, report_every=0)
            self.assertTrue(torch.equal(engine.prefill(5, ids, 0)[0], reference))
            row = c.close()
            self.assertEqual(engine.net.comm.gathers, 0)                  # the head is off: no gathers on any rank
            self.assertEqual((row["kv"], row["head"], row["errors"]), ({}, {}, {}))
            self.assertFalse((Path(d) / "capture" / "moe").exists())
            from safetensors import safe_open
            files = sorted((Path(d) / "capture" / "dense" / "L00").iterdir())
            self.assertEqual([f.name for f in files], ["d000000-c0000000.safetensors"])
            with safe_open(str(files[0]), framework="pt") as f:
                self.assertEqual(tuple(f.get_tensor("x").shape), (140, 64))
                self.assertEqual(f.metadata()["layer"], "0")
            self.assertEqual(row["captured_tokens"], 140)
        with self.assertRaises(ValueError):
            cap.Capture(FakeEngine(), "/nonexistent", sections=("moe", "logits"))

    def test_calibration_phases_reset_only_after_the_fit_filing(self):
        class FakeCalibration:
            def __init__(self):
                self.H = {"a": torch.ones(2, 2)}
                self.rows = {"a": torch.tensor(5.0)}
                self.amax = {"a": torch.ones(2)}
                self.armed = torch.tensor(1.0)
                self.filed = None

        class Engine:
            def __init__(self):
                self.calibration = FakeCalibration()
                self.net = FakeNet()
                self.roots = []

            def file_calibration(self, root=None):
                self.roots.append(root)
                self.calibration.filed = ["blob"]
                self.calibration.armed.zero_()
                return ["blob"]

        engine = Engine()
        cap.arm_calibration_phases(engine)
        self.assertFalse(engine.calibration.complete())                # housekeeping never files on its own
        engine.file_calibration("/cache/calib-v2-fit")
        c = engine.calibration
        self.assertEqual((float(c.H["a"].sum()), float(c.rows["a"]), float(c.amax["a"].sum()), float(c.armed)), (0.0, 0.0, 0.0, 1.0))
        self.assertIsNone(c.filed)
        c.H["a"] += 3
        engine.file_calibration("/cache/calib-v2-heldout")
        self.assertEqual(float(c.H["a"].sum()), 12.0)                  # the held-out sums are filed as they are
        self.assertEqual(c.filed, ["blob"])
        self.assertEqual(engine.roots, ["/cache/calib-v2-fit", "/cache/calib-v2-heldout"])

    def test_boot_arms_only_in_the_measurement_commit(self):
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn("EXPERT_CAPTURE = ", source)
        self.assertIn("capture_mod.attach(engine", source)
        self.assertIn("expert_capture.close()", source)
        self.assertIn("CALIBRATION_CAPTURE = False", source)
        self.assertIn('PACK_ROOT = "/cache"', source)                          # production's store; an arm moves it
        self.assertIn("head_rows=CAPTURE_HEAD_ROWS", source)
        self.assertIn("capture_mod.arm_calibration_phases(engine)", source)


if __name__ == "__main__":
    unittest.main()
