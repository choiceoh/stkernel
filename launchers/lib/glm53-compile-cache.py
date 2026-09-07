#!/usr/bin/env python3
"""Invalidate head compile artifacts by runtime content, retain fleet provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def fingerprints(manifest, image_id):
    data = manifest.read_bytes()
    rows = []
    sources, targets = set(), set()
    for line in data.decode().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("malformed overlay manifest")
        source, target, base = fields
        if (not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", source)
                or not target.startswith("/") or ".." in Path(target).parts
                or not (base == "absent" or re.fullmatch(r"[0-9a-f]{64}", base))
                or source in sources or target in targets):
            raise ValueError("invalid overlay manifest row")
        sources.add(source)
        targets.add(target)
        with (manifest.parent / source).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append([source, target, base, digest])
    if not rows or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("empty overlay manifest or missing attested image ID")
    content = json.dumps(["glm53-compile-v1", image_id, sorted(rows)], separators=(",", ":"))
    return hashlib.sha256(data).hexdigest(), hashlib.sha256(content.encode()).hexdigest()


def read_text(path):
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return ""


def decision(cache, manifest, image_id):
    deployment, content = fingerprints(manifest, image_id)
    try:
        previous = json.loads(read_text(cache / ".compile-overlay.json"))
    except (ValueError, UnicodeError):
        previous = None
    # The provenance link detects an older launcher clearing/replacing the
    # cache without updating our receipt (A -> old-launcher B -> A).
    reuse = (isinstance(previous, dict) and previous.get("version") == 1
             and previous.get("content_sha256") == content
             and previous.get("deployment_sha256") == read_text(cache / ".overlay-sha"))
    return {"version": 1, "deployment_sha256": deployment, "content_sha256": content,
            "action": "reuse" if reuse else "invalidate"}


def atomic_write(path, text):
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def prepare(cache, manifest, image_id):
    result = decision(cache, manifest, image_id)
    if result["action"] == "invalidate":
        # Only the existing head torch.compile directory is cleared. Use the
        # attested immutable image ID, and do not publish a receipt on failure.
        subprocess.run(["docker", "run", "--rm", "-v", f"{cache}:/cache",
                        "--entrypoint", "rm", image_id,
                        "-rf", "/cache/vllm/torch_compile_cache"], check=True)
    receipt = {key: value for key, value in result.items() if key != "action"}
    atomic_write(cache / ".compile-overlay.json", json.dumps(receipt, sort_keys=True) + "\n")
    # Keep the existing manifest SHA contract used by fleet and onepass. A
    # crash between writes leaves a mismatch and forces invalidation next boot.
    atomic_write(cache / ".overlay-sha", result["deployment_sha256"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--inspect", action="store_true", help="report decision without writes or Docker")
    args = parser.parse_args()
    start = time.monotonic()
    result = (decision if args.inspect else prepare)(args.cache, args.manifest, args.image_id)
    result["elapsed_s"] = round(time.monotonic() - start, 6)
    print("[compile-cache] " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
