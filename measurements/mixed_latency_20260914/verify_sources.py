"""Verify retained records against their admitted revisions, not moving HEAD."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
repo = root.parents[1]
manifest = json.loads((root/'sources.json').read_text())
verified = []
for row in manifest['gpu_runs'] + manifest['cpu_runs']:
    report = json.loads((root/row['file']).read_text())
    hashes = report['source_sha256']
    for path, digest in hashes.items():
        source = subprocess.check_output(['git', 'show', row['revision']+':'+path], cwd=repo)
        if hashlib.sha256(source).hexdigest() != digest:
            raise RuntimeError('source mismatch: '+row['file']+' '+path)
    verified.append(dict(file=row['file'], revision=row['revision'], source_files=len(hashes),
                         status=report['status']))
print(json.dumps(dict(source_verification='PASS', records=verified), indent=2))
