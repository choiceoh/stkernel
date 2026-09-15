"""A restart may skip the far prefill memory pass only on a record that describes this boot, on every rank."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

from engine.base import prefill_record as pr
from engine.base.prefill_record import Limits, PrefillRecord

GIB, MIB = 1 << 30, 1 << 20
ROOT = Path(__file__).resolve().parents[1]

# The near and far rows of a real same-build restart (c2opt-profile-main963, rank 0, 2026-09-15):
# four boots of three builds produced these two peaks to the byte.
ARENA, BASELINE, LIMIT = 59544280104, 2097152, 72431279144
NEAR = dict(phase="prefill/32256/0/prepared", seconds=74.2441, allocated_bytes=60453716640,
            reserved_bytes=60536389632, peak_allocated_bytes=65000000000, peak_reserved_bytes=66605547520,
            peak_workspace_bytes=7059170264, immediately_free_bytes=40299 * MIB, available_bytes=34982 * MIB,
            oom_margin_bytes=34982 * MIB - 6 * GIB, passed=True)
FAR = dict(NEAR, phase="prefill/32256/1016320/prepared", seconds=30.2209, reserved_bytes=66035122176,
           peak_reserved_bytes=70170705920, peak_workspace_bytes=10624328664,
           immediately_free_bytes=34970 * MIB, available_bytes=29658 * MIB)
LIMITS = Limits(baseline_reserved_bytes=BASELINE, arena_bytes=ARENA, allocator_limit_bytes=LIMIT,
                os_reserve_bytes=7 * GIB, sigterm_bytes=6 * GIB)


def components(*changes):
    parts = dict(
        engine="tree-a",
        runtime=dict(packages={"torch": "2.12.0", "flashinfer-python": "0.6.18", "nvidia-cutlass-dsl": "4.3"},
                     torch_cuda="13.2", torch_git="abc", manifest=dict(size=10, sha256="m")),
        weights=dict(rank=dict(size=44, mtime_ns=1, sha256="r"), drafter=dict(size=2, mtime_ns=1, sha256="d"),
                     vision=dict(size=1, mtime_ns=1, sha256="v")),
        meta={"config.json": dict(size=1, sha256="c")},
        config=dict(values={"moe_static": "'t,r,sf6,batch,q0'", "mla_prefill": "'tile32'"},
                    environment={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, rank=0, world=4,
                    kv_gib=24.0, tier=False, arena_bytes=ARENA, workspace_bytes=12 * GIB, os_reserve_bytes=7 * GIB,
                    host_budget_bytes=0, max_context=1048576, prefill_chunk=32256, max_seqs=2, blocks=1398,
                    snapshots=96, decode_tokens=8, lanes="served", lane_info={"mla_prefill": "tile32"},
                    vision=True, grammar=True),
        node=dict(hostname="srv2", gpu_uuid="GPU-1", gpu_name="GB10", gpu_total_memory=128, driver="n"))
    for path, value in changes:                       # a path is a tuple: file names carry dots
        target = parts
        for name in path[:-1]:
            target = target[name]
        target[path[-1]] = value
    return parts


def ledger(near=NEAR, far=FAR, *extra):
    return [dict(NEAR, phase="loaded"), dict(near), dict(far), *extra, dict(NEAR, phase="production/ready")]


class RecordTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.gate = self.root / "st-gate"

    def record(self, parts=None, **kwargs):
        return PrefillRecord(self.gate, 0, parts or components(), **kwargs)

    def kept(self, parts=None, phases=None):
        """A record written the only way one can be: by a boot that ran the far pass and passed every row."""
        record = self.record(parts)
        record.ran_full(NEAR["phase"], FAR["phase"])
        record.write(phases or ledger(), release="test")
        return record


class KeyTests(RecordTestCase):
    def test_every_component_the_far_pass_depends_on_moves_the_key_and_the_reason_names_it(self):
        self.kept()
        for path, value in ((("engine",), "tree-b"), (("runtime", "packages", "torch"), "2.12.1"),
                            (("runtime", "packages", "nvidia-cutlass-dsl"), "4.4"), (("runtime", "manifest"), None),
                            (("weights", "rank"), dict(size=44, mtime_ns=1, sha256="r2")),
                            (("weights", "drafter"), None), (("meta", "config.json"), dict(size=2, sha256="c")),
                            (("config", "values", "moe_static"), "'stock'"),
                            (("config", "environment", "STK_mla_prefill"), "stock"),
                            (("config", "kv_gib"), 7.0), (("config", "workspace_bytes"), 10 * GIB),
                            (("config", "max_context"), 131072), (("config", "prefill_chunk"), 9216),
                            (("config", "max_seqs"), 4), (("config", "decode_tokens"), 7), (("config", "tier"), True),
                            (("config", "lanes"), "reference"), (("config", "lane_info", "mla_prefill"), "stock"),
                            (("config", "vision"), False), (("config", "grammar"), False), (("config", "rank"), 1),
                            (("node", "hostname"), "srv4"), (("node", "gpu_uuid"), "GPU-2")):
            dotted = ".".join(path)
            with self.subTest(component=dotted):
                changed = self.record(components((path, value)))
                self.assertNotEqual(changed.key, pr.key_of(components()))
                self.assertIsNone(changed.record)
                self.assertIn(dotted, changed.reason)
                self.assertFalse(changed.verdict(NEAR, LIMITS).reuse)
        same = self.record()
        self.assertIsNotNone(same.record)
        self.assertEqual(same.reason, "the record matches")

    def test_the_tree_digest_follows_content_and_paths_but_not_bytecode(self):
        tree = self.root / "engine"
        (tree / "base").mkdir(parents=True)
        (tree / "base" / "a.py").write_text("x = 1\n")
        first = pr.digest_tree(tree)
        (tree / "base" / "__pycache__").mkdir()
        (tree / "base" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0bytecode")
        self.assertEqual(pr.digest_tree(tree), first)
        (tree / "base" / "a.py").write_text("x = 2\n")
        second = pr.digest_tree(tree)
        self.assertNotEqual(second, first)
        (tree / "base" / "a.py").rename(tree / "base" / "b.py")
        self.assertNotEqual(pr.digest_tree(tree), second)

    def test_a_weight_file_is_known_by_its_header_size_and_mtime_not_by_reading_its_body(self):
        path = self.root / "rank0of4.safetensors"
        header = json.dumps({"w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x01\x02\x03\x04")
        stamp = path.stat().st_mtime_ns
        first = pr.file_identity(path)
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x09\x09\x09\x09")
        os.utime(path, ns=(stamp, stamp))
        self.assertEqual(pr.file_identity(path), first, "the body is not read")
        other = header.replace(b"F16", b"F32")
        path.write_bytes(struct.pack("<Q", len(other)) + other + b"\x01\x02\x03\x04")
        os.utime(path, ns=(stamp, stamp))
        self.assertNotEqual(pr.file_identity(path)["sha256"], first["sha256"])
        path.write_bytes(struct.pack("<Q", 1 << 40) + header)
        with self.assertRaisesRegex(ValueError, "implausible"):
            pr.file_identity(path)

    def test_metadata_is_known_by_content_because_every_launch_copies_it_afresh(self):
        path = self.root / "config.json"
        path.write_text('{"layers": 45}')
        first = pr.file_identity(path)
        os.utime(path, ns=(1, 1))                                     # launchers/start-st-glm53.sh: cp, no -p
        self.assertEqual(pr.file_identity(path), first)
        path.write_text('{"layers": 46}')
        self.assertNotEqual(pr.file_identity(path), first)


class LifecycleTests(RecordTestCase):
    def test_without_a_record_the_boot_runs_the_full_gate_and_a_passing_one_keeps_it(self):
        record = self.record()
        self.assertIsNone(record.record)
        self.assertEqual(record.reason, "no record on this node")
        self.assertFalse(record.verdict(NEAR, LIMITS).reuse)
        record.ran_full(NEAR["phase"], FAR["phase"])
        self.assertEqual(record.write(ledger(), release="test"), record.path)
        self.assertEqual(record.path.parent, self.gate)
        kept = json.loads(record.path.read_text())
        self.assertEqual((kept["schema"], kept["key"], kept["rank"], kept["release"]), (pr.SCHEMA, record.key, 0, "test"))
        self.assertEqual(kept["rows"]["far"]["peak_workspace_bytes"], FAR["peak_workspace_bytes"])
        self.assertEqual(sorted(kept["rows"]["near"]), sorted(pr.ROW_FIELDS))
        self.assertEqual([p.name for p in self.gate.iterdir()], [record.path.name], "no temporary left")
        self.assertTrue(self.record().verdict(NEAR, LIMITS).reuse)

    def test_only_a_boot_that_ran_the_far_pass_and_passed_every_row_writes(self):
        record = self.record()
        with self.assertRaisesRegex(RuntimeError, "ran the full gate"):
            record.write(ledger())                                     # no far pass ran
        record.ran_full(NEAR["phase"], FAR["phase"])
        for name, phases in (("a failed row", ledger(NEAR, FAR, dict(NEAR, phase="target/x", passed=False))),
                             ("a reused row", ledger(NEAR, dict(FAR, phase="prefill/32256/1016320/reused", reused=True))),
                             ("no far row", [dict(NEAR)]), ("an empty ledger", [])):
            with self.subTest(ledger=name), self.assertRaises(RuntimeError):
                record.write(phases)
        self.assertFalse(self.gate.exists())
        broken = PrefillRecord(self.gate, 0, None, error="boom")
        broken.ran_full(NEAR["phase"], FAR["phase"])
        with self.assertRaisesRegex(RuntimeError, "ran the full gate"):
            broken.write(ledger())

    def test_an_interrupted_write_leaves_the_previous_record_whole(self):
        path = self.kept().path
        before = path.read_bytes()
        again = self.record(force_full=True)                            # a forced full gate rewrites the same key
        again.ran_full(NEAR["phase"], FAR["phase"])
        with patch.object(pr.json, "dump", side_effect=OSError("disk full")), self.assertRaises(OSError):
            again.write(ledger())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual([p.name for p in self.gate.iterdir()], [path.name])

    def test_production_and_tickets_keep_their_own_records_and_the_least_recently_used_are_pruned(self):
        production = self.kept().path
        ticket = self.kept(components((("config", "tier"), True), (("engine",), "tree-ticket"))).path
        self.assertNotEqual(production, ticket)
        self.assertTrue(self.record().verdict(NEAR, LIMITS).reuse, "a ticket between restarts does not cost production its record")
        for index, path in enumerate((production, ticket)):
            os.utime(path, ns=(index + 1, index + 1))
        written = []
        for index in range(pr.KEEP):
            path = self.kept(components((("engine",), f"tree-{index}"))).path
            os.utime(path, ns=(100 + index, 100 + index))
            written.append(path)
            if index == pr.KEEP // 2:
                self.record().used("prefill/32256/1016320")            # production restarts in between and reuses
        remaining = set(self.gate.glob("prefill-rank0-*.json"))
        self.assertEqual(len(remaining), pr.KEEP)
        self.assertIn(production, remaining, "the record production keeps reusing is not the one pruned")
        self.assertFalse(ticket.exists())
        self.assertEqual(remaining - {production}, set(written[1:]))
        other_rank = PrefillRecord(self.gate, 1, components((("config", "rank"), 1)))
        self.assertEqual(other_rank.reason, "no record on this node", "a rank reads only its own records")

    def test_unreadable_foreign_or_incomplete_records_are_full_gates(self):
        path = self.kept().path
        good = json.loads(path.read_text())
        for name, text, reason in (("garbage", "{not json", "not JSON"),
                                   ("schema", json.dumps(dict(good, schema=0)), "schema differs"),
                                   ("key", json.dumps(dict(good, key="0" * 64)), "names another"),
                                   ("rows", json.dumps(dict(good, rows={"near": good["rows"]["near"]})), "incomplete"),
                                   ("failed far row", json.dumps(dict(good, rows=dict(good["rows"], far=dict(
                                       good["rows"]["far"], passed=False)))), "incomplete")):
            with self.subTest(record=name):
                path.write_text(text)
                record = self.record()
                self.assertIsNone(record.record)
                self.assertIn(reason, record.reason)
                self.assertFalse(record.verdict(NEAR, LIMITS).reuse)

    def test_forcing_the_full_gate_or_failing_to_compute_the_key_never_reuses(self):
        self.kept()
        forced = self.record(force_full=True)
        self.assertEqual((forced.record, forced.reason), (None, "the full gate was asked for"))
        def refuse():
            raise OSError("rank file missing")
        broken = PrefillRecord.build(self.gate, 0, refuse)
        self.assertIsNone(broken.key)
        self.assertIn("rank file missing", broken.reason)
        self.assertFalse(broken.verdict(NEAR, LIMITS).reuse)


class VerdictTests(RecordTestCase):
    def test_the_recorded_boot_reuses_with_its_far_peak_projected_onto_todays_box(self):
        self.kept()
        verdict = self.record().verdict(NEAR, LIMITS)
        self.assertTrue(verdict.reuse, verdict.reason)
        extra = BASELINE + ARENA + FAR["peak_workspace_bytes"] - NEAR["reserved_bytes"]
        self.assertEqual(verdict.peak_workspace_bytes, FAR["peak_workspace_bytes"])
        self.assertEqual(verdict.immediately_free_bytes, NEAR["immediately_free_bytes"] - extra)
        self.assertEqual(verdict.available_bytes, NEAR["available_bytes"] - extra)
        # the projection is stricter than the full gate's own post-release reading of the far row
        self.assertLess(verdict.available_bytes, FAR["available_bytes"])

    def test_a_near_pass_that_peaks_above_its_record_runs_the_far_pass(self):
        self.kept()
        record = self.record()
        within = dict(NEAR, peak_workspace_bytes=NEAR["peak_workspace_bytes"] + pr.TOLERANCE_BYTES)
        verdict = record.verdict(within, LIMITS)
        self.assertTrue(verdict.reuse)
        self.assertEqual(verdict.peak_workspace_bytes, FAR["peak_workspace_bytes"] + pr.TOLERANCE_BYTES,
                         "the far peak moves with today's near peak")
        above = dict(NEAR, peak_workspace_bytes=NEAR["peak_workspace_bytes"] + pr.TOLERANCE_BYTES + 1)
        self.assertIn("above its record", record.verdict(above, LIMITS).reason)
        lower = dict(NEAR, peak_workspace_bytes=NEAR["peak_workspace_bytes"] - GIB)
        self.assertEqual(record.verdict(lower, LIMITS).peak_workspace_bytes, FAR["peak_workspace_bytes"],
                         "never below the recorded far peak")

    def test_a_box_with_less_room_today_runs_the_far_pass(self):
        """The floor is re-read by this boot: the record's room is not today's."""
        self.kept()
        record = self.record()
        extra = BASELINE + ARENA + FAR["peak_workspace_bytes"] - NEAR["reserved_bytes"]
        edge = dict(NEAR, immediately_free_bytes=LIMITS.os_reserve_bytes + extra)
        self.assertTrue(record.verdict(edge, LIMITS).reuse)
        tight = dict(NEAR, immediately_free_bytes=LIMITS.os_reserve_bytes + extra - 1)
        self.assertIn("OS reserve", record.verdict(tight, LIMITS).reason)
        tenant = dict(NEAR, available_bytes=LIMITS.sigterm_bytes + extra - 1)
        self.assertIn("SIGTERM line", record.verdict(tenant, LIMITS).reason)
        smaller = Limits(BASELINE, ARENA, BASELINE + ARENA + FAR["peak_workspace_bytes"] - 1, 7 * GIB, 6 * GIB)
        self.assertIn("workspace ceiling", record.verdict(NEAR, smaller).reason)

    def test_a_near_row_of_another_shape_or_a_failed_one_does_not_vouch(self):
        self.kept()
        record = self.record()
        self.assertFalse(record.verdict(dict(NEAR, phase="prefill/9216/0/prepared"), LIMITS).reuse)
        self.assertFalse(record.verdict(dict(NEAR, passed=False), LIMITS).reuse)
        self.assertIn("verdict failed", record.verdict({"phase": NEAR["phase"], "passed": True}, LIMITS).reason)


class RuntimeMemoryVoteTests(unittest.TestCase):
    def memory(self, peers):
        import torch
        from engine.base.runtime_memory import RuntimeMemory
        from tests.test_engine_runtime_memory import Cuda
        memory = RuntimeMemory(400, 200, 100, cuda=Cuda(), host_free=lambda: 800, host_available=lambda: 700,
                               floor=(60, 45))
        memory.comm = NS(all_reduce_max=lambda t: t.fill_(max([int(t.item()), *peers])))
        memory.status = torch.zeros(1, dtype=torch.int32)
        self.addCleanup(memory.close)
        return memory

    def test_the_vote_is_every_ranks_largest_flag(self):
        self.assertEqual(self.memory([0, 0, 0]).agree(0), 0)
        self.assertEqual(self.memory([0, 2, 0]).agree(0), 2)
        self.assertEqual(self.memory([0, 0, 0]).agree(2), 2)

    def test_a_reused_row_carries_the_projection_into_the_measured_split(self):
        memory = self.memory([0, 0, 0])
        memory.checkpoint("prefill/256/0/prepared", release_cache=True)
        verdict = pr.Verdict(True, "the record matches", peak_workspace_bytes=150, immediately_free_bytes=300,
                             available_bytes=250)
        row = memory.reused("prefill/256/768/reused", verdict, "/cache/st-gate/prefill-rank0-x.json")
        self.assertTrue(row["reused"] and row["passed"])
        self.assertEqual(row["peak_reserved_bytes"], memory.baseline_reserved + 400 + 150)
        self.assertEqual(row["oom_margin_bytes"], 250 - 60)
        measured = memory.measured()
        self.assertEqual(measured["prefill_peak_bytes"], 150)
        self.assertEqual(measured["reused_phases"], ["prefill/256/768/reused"])
        self.assertEqual(measured["oom_margin_phase"], "prefill/256/768/reused")


class AdapterReuseTests(RecordTestCase):
    """The profile's gate on CPU: which passes run, which rows are voted, what the record learns."""

    class Memory:
        def __init__(self, peers):
            self.peers, self.rows, self.votes, self.reused_rows = peers, [], [], []
            self.stamps = []
            self.baseline_reserved, self.arena_bytes, self.allocator_limit_bytes = BASELINE, ARENA, LIMIT
            self.os_reserve_bytes, self.sigterm_bytes = 7 * GIB, 6 * GIB
        def checkpoint(self, phase, release_cache=False, stamps=None):
            self.stamps.append((phase, stamps))
            template = FAR if phase.startswith("prefill/256/768/") else NEAR
            row = dict(template, phase=phase)
            self.rows.append(row)
            return row
        def agree(self, flag):
            self.votes.append(flag)
            return max([flag, *self.peers])
        def reused(self, phase, verdict, source):
            self.reused_rows.append((phase, verdict, source))

    def engine(self, record, peers=(0, 0, 0)):
        import torch
        from engine.profiles.glm53.adapter import Glm53Engine
        caches = MagicMock(device="cpu", snapshots=0)
        caches.pool.rows_in_use, caches.pool.num_blocks = 0, 16
        caches.slots.owner = [-1, -1]
        caches.slots.take.return_value = 1
        engine = Glm53Engine.__new__(Glm53Engine)
        engine.caches, engine.F, engine.prefill_chunk, engine.max_context = caches, NS(block=64), 256, 4096
        engine.memory = self.Memory(list(peers))
        engine.net = NS(comm=NS(rank=0, all_reduce_max=lambda x: x), head=lambda x: x)
        engine._prefill_forward = MagicMock(return_value=(torch.ones(1, 8), None))
        engine.prefill_record = record
        engine.PREFILL_CONTINUATION_TOKENS = 128
        return engine

    def passes(self, engine):
        return [(c.args[0].segments[0].ctx, c.args[0].ids.numel()) for c in engine._prefill_forward.call_args_list]

    def matching(self):
        # 256 tokens at 0 and at 1024 - 256 = 768 are this test's near and far passes
        near, far = dict(NEAR, phase="prefill/256/0/prepared"), dict(FAR, phase="prefill/256/768/prepared")
        record = self.record()
        record.ran_full(near["phase"], far["phase"])
        record.write([near, far])
        return self.record()

    def test_without_a_bound_record_nothing_is_voted_and_both_ends_run(self):
        engine = self.engine(None)
        engine._warmup_prefill_memory()
        self.assertEqual(self.passes(engine), [(0, 256), (768, 256)])
        self.assertEqual(engine.memory.votes, [])

    def test_every_rank_matching_skips_the_far_pass_for_a_continuation(self):
        record = self.matching()
        engine = self.engine(record)
        with patch("builtins.print"):
            engine._warmup_prefill_memory()
        self.assertEqual(self.passes(engine), [(0, 256), (256, 128)])
        self.assertEqual(engine.memory.votes, [0])
        self.assertEqual([phase for phase, _, _ in engine.memory.reused_rows], ["prefill/256/768/reused"])
        self.assertEqual((record.reused, record.full), ("prefill/256/768", None))
        self.assertEqual([row["phase"] for row in engine.memory.rows][-2:],
                         ["prefill/128/256/before", "prefill/128/256/prepared"])
        engine.caches.pool.release.assert_called_once_with(0)

    def test_one_rank_that_cannot_reuse_sends_every_rank_through_the_far_pass(self):
        record = self.matching()
        engine = self.engine(record, peers=(0, 2, 0))
        with patch("builtins.print"):
            engine._warmup_prefill_memory()
        self.assertEqual(self.passes(engine), [(0, 256), (768, 256)])
        self.assertEqual(engine.memory.reused_rows, [])
        self.assertEqual(record.full, ("prefill/256/0/prepared", "prefill/256/768/prepared"))
        self.assertIsNone(record.reused)

    def test_a_rank_without_a_record_still_votes_and_runs_the_full_gate(self):
        record = self.record()                                          # nothing kept on this node
        engine = self.engine(record)
        with patch("builtins.print"):
            engine._warmup_prefill_memory()
        self.assertEqual(engine.memory.votes, [2])
        self.assertEqual(self.passes(engine), [(0, 256), (768, 256)])
        self.assertIsNotNone(record.full)

    def test_a_peer_that_failed_before_the_vote_stops_every_rank(self):
        engine = self.engine(self.matching(), peers=(1, 0, 0))
        with patch("builtins.print"), self.assertRaisesRegex(MemoryError, "TP peer failed"):
            engine._warmup_prefill_memory()
        self.assertEqual(self.passes(engine), [(0, 256)])
        engine.caches.pool.release.assert_called_once_with(0)


class BootWiringTests(unittest.TestCase):
    def setUp(self):
        self.boot = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()

    def test_every_rank_binds_its_record_before_capture_and_keeps_it_only_after_the_last_row(self):
        fleet = self.boot[self.boot.index("def fleet(a)"):]
        bind = fleet.index("engine.prefill_record = PrefillRecord.build(PREFILL_RECORD_ROOT, comm.rank,")
        self.assertLess(bind, fleet.index('engine.capture_decode(MAX_SEQS)'))
        keep = fleet.index("engine.prefill_record.write(engine.memory.phases")
        self.assertLess(fleet.index('engine.memory.checkpoint("production/ready"'), keep)
        self.assertLess(keep, fleet.index('engine.memory.write(Path(a.dump_dir)'))

    def test_the_full_gate_can_be_asked_for_at_the_launcher_and_the_boot(self):
        start = self.launcher.index('GATE_ARG=""')
        block = self.launcher[start:self.launcher.index("esac", start) + len("esac")]
        self.assertIn("--dump-dir $DUMP_DIR $GATE_ARG'", self.launcher)
        for env, want in (({}, ""), ({"ST_FULL_MEMORY_GATE": "0"}, ""), ({"ST_FULL_MEMORY_GATE": "1"}, "--full-memory-gate")):
            out = subprocess.run(["bash", "-c", block + '\nprintf %s "$GATE_ARG"'], env={"PATH": os.environ["PATH"], **env},
                                 capture_output=True, text=True, check=True).stdout
            self.assertEqual(out, want)
        refused = subprocess.run(["bash", "-c", block], env={"PATH": os.environ["PATH"], "ST_FULL_MEMORY_GATE": "yes"},
                                 capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        from engine.profiles.glm53 import boot
        with patch("engine.runtime.verify.verify"), patch.object(boot, "fleet", return_value=0) as fleet:
            boot.main([])
            boot.main(["--full-memory-gate"])
        self.assertEqual([c.args[0].full_memory_gate for c in fleet.call_args_list], [False, True])

    def test_the_key_names_the_rank_the_config_and_the_node_but_not_what_each_boot_samples(self):
        from engine.profiles.glm53 import boot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ranks").mkdir()
            (root / "meta").mkdir()
            header = json.dumps({"w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}).encode()
            (root / "ranks" / "rank1of4.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            (root / "meta" / "config.json").write_text("{}")
            a = NS(ranks=str(root / "ranks"), ckpt_meta=str(root / "meta"), drafter_dir=str(root / "none"),
                   kv_gib=24.0, tier_dir="")
            cfg = NS(values={"port": 8000, "moe_static": "t,r,sf6,batch,q0"})
            engine = NS(memory=NS(arena_bytes=ARENA, workspace_bytes=12 * GIB, os_reserve_bytes=7 * GIB,
                                  host_budget_bytes=0),
                        max_context=1048576, prefill_chunk=32256, drafter=NS(k=7), vision=object(),
                        lane_info={"mla_prefill": "tile32", "oneshot_latency_us": "sum_16rows=45.5"})
            caches = NS(pool=NS(max_seqs=2, num_blocks=1398), snapshots=96)
            comm, lanes = NS(rank=1, world_size=4), NS(name="served")
            node = dict(hostname="srv1", gpu_uuid="GPU-1")
            with patch.object(pr, "node_identity", return_value=node):
                parts = boot.prefill_record_components(a, cfg, engine, caches, lanes, comm)
                engine.lane_info["oneshot_latency_us"] = "sum_16rows=46.0"
                os.utime(root / "meta" / "config.json", ns=(1, 1))        # the launcher's copy at the next launch
                self.assertEqual(pr.key_of(boot.prefill_record_components(a, cfg, engine, caches, lanes, comm)),
                                 pr.key_of(parts), "what a launch or a boot refreshes does not invalidate the record")
                a.kv_gib = 7.0
                self.assertNotEqual(pr.key_of(boot.prefill_record_components(a, cfg, engine, caches, lanes, comm)),
                                    pr.key_of(parts))
        self.assertEqual(parts["node"], node)
        self.assertEqual((parts["config"]["rank"], parts["config"]["max_seqs"], parts["config"]["decode_tokens"]), (1, 2, 8))
        self.assertNotIn("port", parts["config"]["values"])
        self.assertNotIn("oneshot_latency_us", parts["config"]["lane_info"])
        self.assertIsNotNone(parts["weights"]["rank"])
        self.assertIsNone(parts["weights"]["drafter"])
        self.assertEqual(set(parts["meta"]), {"config.json"})
        self.assertIn("torch", parts["runtime"]["packages"])
        self.assertEqual(len(parts["engine"]), 64)


if __name__ == "__main__":
    unittest.main()
