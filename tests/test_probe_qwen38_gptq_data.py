import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from probes.qwen38_gptq_data import digest, grouped_rows, interleave, private_directory, snapshot
from probes.qwen38_gptq_feed import collect, load_split, verify_owner


class SplitTests(unittest.TestCase):
    def fixtures(self):
        rows = [dict(id=k, split=s, messages=[dict(role="user", content=text)])
                for k, s, text in [("a", "train", "first"), ("b", "test", "second")]]
        provenance = {"rows": [dict(id=k, session="session-" + k) for k in ("a", "b")]}
        return rows, provenance

    def test_inherited_conversation_is_not_counted_twice(self):
        rows, p = self.fixtures()
        merged = grouped_rows(rows, p, [dict(rows[0], category="conversation")], {"rows": []})
        self.assertEqual(len(merged), 2)
        self.assertEqual([r["split"] for r in merged], ["train", "test"])

    def test_same_session_cannot_cross_splits(self):
        rows, p = self.fixtures()
        p["rows"][1]["session"] = p["rows"][0]["session"]
        with self.assertRaisesRegex(ValueError, "source group spans"):
            grouped_rows(rows, p, [], {"rows": []})

    def test_same_text_from_different_sources_cannot_cross_splits(self):
        rows, p = self.fixtures()
        workload = dict(id="mail", split="test", text="first", category="mail")
        wp = {"rows": [dict(id="mail", split="test", group="thread")]}
        with self.assertRaisesRegex(ValueError, "identical input spans"):
            grouped_rows(rows, p, [workload], wp)

    def test_inherited_row_cannot_be_promoted_to_train(self):
        rows, p = self.fixtures()
        with self.assertRaisesRegex(ValueError, "changed content or split"):
            grouped_rows(rows, p, [dict(rows[1], split="train")], {"rows": []})

    def test_interleaves_categories_before_exhausting_any_source(self):
        rows = [dict(id=i, category=c) for c in ("mail", "conversation", "notification") for i in range(3)]
        ordered = interleave(rows)
        self.assertEqual(len(ordered), 9)
        self.assertEqual(len({r["category"] for r in ordered[:3]}), 3)
        self.assertEqual(ordered, interleave(rows))

    def test_changed_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "prompts.json"
            path.write_text("[]")
            path.with_suffix(".provenance.json").write_text(json.dumps(dict(output_sha256=digest(b"[]"))))
            self.assertEqual(snapshot(path)[0], [])
            path.write_text("[{}]")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                snapshot(path)

    def test_private_output_cannot_enter_git_or_replace_an_existing_split(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / ".git").touch()
            with self.assertRaisesRegex(ValueError, "outside Git"):
                private_directory(root / "data")
            (root / ".git").unlink()
            private_directory(root / "data")
            with self.assertRaises(FileExistsError):
                private_directory(root / "data")


class CollectionTests(unittest.TestCase):
    def test_deferred_window_never_passes_sessions_queue_or_pending_handover(self):
        module_path = Path(__file__).resolve().parents[1] / "measurements/qwen38_gptq_20260919/wait_for_window.py"
        spec = importlib.util.spec_from_file_location("gptq_window_waiter", module_path)
        waiter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(waiter)
        with tempfile.TemporaryDirectory() as root:
            lease, queue = Path(root) / "lease", Path(root) / "queue"
            queue.write_text("")
            self.assertTrue(waiter.available(lease, queue)[0])
            lease.write_text(json.dumps(dict(kind="session", owner="session/another")))
            self.assertFalse(waiter.available(lease, queue)[0])
            lease.write_text(json.dumps(dict(kind="production", owner="production/model")))
            self.assertTrue(waiter.available(lease, queue)[0])
            queue.write_text("waiting-job\n")
            self.assertFalse(waiter.available(lease, queue)[0])
            queue.write_text("")
            lease.write_text(json.dumps(dict(kind="production", yield_to=dict(requester="other"))))
            self.assertFalse(waiter.available(lease, queue)[0])
            lease.write_text("not json")
            self.assertFalse(waiter.available(lease, queue)[0])

    def test_nested_rank_scoring_requires_the_inherited_verified_owner(self):
        from probes.qwen38_gptq_score import verify_gpu_owner
        args = SimpleNamespace(device="cuda", parent_verified=True, owner="session/ours")
        with mock.patch.dict("os.environ", {"ST_LEASE_OWNER": "session/ours"}):
            verify_gpu_owner(args)
        with mock.patch.dict("os.environ", {"ST_LEASE_OWNER": "session/another"}):
            with self.assertRaisesRegex(RuntimeError, "verified parent's"):
                verify_gpu_owner(args)

    def test_other_fleet_owner_prevents_every_http_request(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = b'{"split":"train"}\n'
            (root / "train.jsonl").write_bytes(data)
            (root / "manifest.json").write_text(json.dumps({"splits_sha256": {"train": digest(data)}}))
            (root / "lease.json").write_text(json.dumps({"owner": "session/another"}))
            args = SimpleNamespace(dataset=root / "train.jsonl", owner="session/ours", lease=root / "lease.json")
            with mock.patch("probes.qwen38_gptq_feed.request") as request:
                with self.assertRaisesRegex(RuntimeError, "does not hold"):
                    collect(args)
                request.assert_not_called()

    def test_dataset_mutation_after_preparation_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.jsonl"
            path.write_bytes(b'{"split":"train"}\n')
            manifest = {"splits_sha256": {"train": digest(path.read_bytes())}}
            self.assertEqual(load_split(path, manifest)[0]["split"], "train")
            path.write_bytes(b'{"split":"test"}\n')
            with self.assertRaisesRegex(ValueError, "changed after"):
                load_split(path, manifest)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class BlobTests(unittest.TestCase):
    def test_hessian_score_matches_explicit_held_out_projection_error(self):
        import torch
        from probes.qwen38_gptq_score import output_error
        g = torch.Generator().manual_seed(919)
        x = torch.randn(37, 9, generator=g, dtype=torch.float64)
        w = torch.randn(13, 9, generator=g, dtype=torch.float64)
        q = w.round()
        actual = output_error(w, q, x.T @ x, chunk=4)
        expected = ((x @ (w - q).T).square().sum() / (x @ w.T).square().sum()).sqrt()
        self.assertAlmostEqual(actual["relative_rmse"], float(expected), places=13)

    def test_saved_hessian_must_match_boot_domain_and_coverage(self):
        import torch
        from probes.qwen38_gptq_audit import validate_blob
        blob = dict(H=torch.eye(4), amax=torch.ones(4), name="site", ntok=8192, weights_id="qwen:one")
        row = validate_blob(blob, "site", 4, "qwen:one", 4096)
        self.assertEqual(row["ntok"], 8192)
        for changes in (dict(weights_id="qwen:two"), dict(ntok=32), dict(H=torch.full((4, 4), float("nan"))),
                        dict(H=torch.zeros(4, 4)), dict(H=torch.eye(3)), dict(amax=torch.ones(4).bfloat16())):
            with self.assertRaises(ValueError):
                validate_blob(dict(blob, **changes), "site", 4, "qwen:one", 4096)


if __name__ == "__main__":
    unittest.main()
