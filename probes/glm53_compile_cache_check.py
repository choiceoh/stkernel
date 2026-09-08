#!/usr/bin/env python3
"""CPU-only cache lifecycle proof using copied live overlays and a temporary cache.

Run through fleet.sh run --cpu. Docker only creates/removes tiny fixture files;
it receives no GPU devices and never mounts the serving cache or overlay tree.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=Path("/home/choiceoh/overlays/glm53"))
    parser.add_argument("--image", default="glm53:v13-b12x-it")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    helper = repo / "launchers/lib/glm53-compile-cache.py"
    spec = importlib.util.spec_from_file_location("compile_cache", helper)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    image_id = subprocess.check_output(["docker", "image", "inspect", args.image,
                                        "--format", "{{.Id}}"], text=True).strip()
    report = {"source_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
              "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
              "image_id": image_id, "steps": []}
    with tempfile.TemporaryDirectory(prefix="glm53-compile-proof-") as temp:
        root = Path(temp)
        overlay, cache = root / "overlay", root / "cache"
        overlay.mkdir()
        cache.mkdir()
        manifest = overlay / "manifest.tsv"
        shutil.copyfile(args.overlay / "manifest.tsv", manifest)
        rows = [line for line in manifest.read_text().splitlines() if line and not line.startswith("#")]
        for row in rows:
            name = row.split("\t")[0]
            assert Path(name).name == name and not name.startswith(".")
            shutil.copyfile(args.overlay / name, overlay / name)
        report["overlay_files"] = len(rows)
        report["live_deployment_sha256"], report["live_content_sha256"] = cc.fingerprints(manifest, image_id)

        def prepare(expected):
            start = time.monotonic()
            result = cc.prepare(cache, manifest, image_id)
            result["elapsed_s"] = time.monotonic() - start
            assert result["action"] == expected, result
            assert (cache / ".overlay-sha").read_text() == hashlib.sha256(manifest.read_bytes()).hexdigest()
            report["steps"].append(result)

        prepare("invalidate")
        # Reproduce root-owned artifacts made by the serving image in a tiny,
        # isolated mount. Do not mount any real model or persistent cache.
        subprocess.run(["docker", "run", "--rm", "-v", f"{cache}:/cache",
                        "--entrypoint", "/bin/sh", image_id, "-c",
                        "mkdir -p /cache/vllm/torch_compile_cache && "
                        "printf compiled-fixture > /cache/vllm/torch_compile_cache/graph.bin"], check=True)
        artifact = cache / "vllm/torch_compile_cache/graph.bin"
        before = artifact.stat()
        report["fixture_uid"] = before.st_uid
        assert before.st_uid == 0
        try:
            # Same runtime, different deploy provenance: legacy logic clears.
            manifest.write_text("# source_commit=metadata-only-probe\n" + "\n".join(rows) + "\n")
            report["legacy_would_invalidate"] = (
                (cache / ".overlay-sha").read_text() != hashlib.sha256(manifest.read_bytes()).hexdigest())
            assert report["legacy_would_invalidate"]
            prepare("reuse")
            prepare("reuse")
            after = artifact.stat()
            assert artifact.read_bytes() == b"compiled-fixture"
            assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
            report["artifact_preserved"] = True
            source = overlay / rows[0].split("\t")[0]
            source.write_bytes(source.read_bytes() + b"\n# runtime-change-probe\n")
            prepare("invalidate")
            assert not artifact.exists()
            report["changed_runtime_removed_artifact"] = True
        finally:
            # Root-owned directories may survive a failed assertion. Cleanup
            # is confined to this TemporaryDirectory, even on failure.
            subprocess.run(["docker", "run", "--rm", "-v", f"{cache}:/cache",
                            "--entrypoint", "rm", image_id, "-rf", "/cache/vllm"], check=True)
    report["ok"] = True
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
