"""Recheck archived PTX excerpt references against original compiler bytes."""

import gzip
import hashlib
import json
from pathlib import Path


def verify_excerpts(node, lines):
    if isinstance(node, dict):
        if isinstance(node.get("line"), int) and isinstance(node.get("text"), str):
            # The inspection normalizes compiler tabs and spacing only.
            assert lines[node["line"] - 1].split() == node["text"].split(), node
            return 1
        return sum(verify_excerpts(value, lines) for value in node.values())
    if isinstance(node, list):
        return sum(verify_excerpts(value, lines) for value in node)
    return 0


def main():
    root = Path(__file__).resolve().parent
    inspection = json.loads((root / "scale-state-inspection.json").read_text())
    report = {"source_revision": inspection["source_revision"], "receipts": {}}
    for side, archive in (("before", root.parent / "cpu12"), ("after", root)):
        item = inspection[side]
        hashes = {}
        for kind in ("ptx", "cubin"):
            descriptor = item[kind]
            raw = gzip.decompress((archive / "local" / (descriptor["file"] + ".gz")).read_bytes())
            assert len(raw) == descriptor["bytes"], (side, kind, "size")
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == descriptor["sha256"], (side, kind, "hash")
            hashes[kind] = digest
            if kind == "ptx":
                lines = raw.decode().splitlines()
        report["receipts"][side] = {
            "sha256": hashes,
            "verified_excerpt_references": verify_excerpts(item, lines),
        }
    report["status"] = "PASS"
    report["scope"] = "Original PTX/cubin bytes and excerpt references; not GPU execution."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
