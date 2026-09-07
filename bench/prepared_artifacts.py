# SPDX-License-Identifier: Apache-2.0
"""Hash CPU build outputs and copy verified artifacts into a GPU snapshot."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def output_path(repo, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or path.parts[0] != 'build' or '..' in path.parts:
        raise ValueError('prepared outputs must be relative files under build/')
    destination = repo / path
    current = repo
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('prepared output path contains a symlink')
    if subprocess.run(['git', '-C', str(repo), 'check-ignore', '-q', '--', relative]).returncode:
        raise ValueError('prepared output must be ignored by git: ' + relative)
    return destination


def capture(payload):
    from experiments import digest
    repo = Path(payload['repo'])
    rows = []
    for relative in payload['spec'].get('outputs', []):
        path = output_path(repo, relative)
        if not path.is_file():
            raise ValueError('prepared output missing: ' + relative)
        rows.append(dict(relative=relative, path=str(path), sha256=digest(path)))
    return rows


def intact(rows):
    from experiments import digest
    return all(Path(r['path']).is_file() and not Path(r['path']).is_symlink()
               and digest(r['path']) == r['sha256'] for r in rows)


def materialize(store, payload):
    rows = []
    incoming = list(payload.get('prepared_artifacts', []))
    for dep in payload['spec']['depends_on']:
        source = store.get(dep)
        artifacts = (source['result'] or {}).get('artifacts', [])
        if artifacts and source['payload']['spec'].get('context') != payload['spec'].get('context'):
            raise ValueError('prepared artifact runtime context does not match consumer')
        if not intact(artifacts):
            raise ValueError('prepared artifact changed after its CPU result')
        incoming.extend(dict(a, source_job=dep) for a in artifacts)
    if not intact(incoming):
        raise ValueError('prepared artifact changed before transfer')
    for artifact in incoming:
        dest = output_path(Path(payload['repo']), artifact['relative'])
        if any(r['relative'] == artifact['relative'] and r['sha256'] != artifact['sha256'] for r in rows):
            raise ValueError('conflicting prepared outputs')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(artifact['path'], temporary)
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
        rows.append(dict(artifact, path=str(dest)))
    if not intact(rows):
        raise ValueError('prepared artifact transfer verification failed')
    return rows
