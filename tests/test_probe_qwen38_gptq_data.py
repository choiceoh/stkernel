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
