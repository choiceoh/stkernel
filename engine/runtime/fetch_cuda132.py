"""Fetch the SHA256-locked ARM64 CUDA runtime overlay; never uses a GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry, directory):
    name = entry['filename']
    if Path(name).name != name or not name.endswith('.whl'):
        raise ValueError(f'invalid wheel filename: {name}')
    destination = directory / name
    if destination.is_file() and sha256(destination) == entry['sha256']:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix='.download-', delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(entry['url'], timeout=90) as response:
                for block in iter(lambda: response.read(1 << 20), b''):
                    stream.write(block)
        if sha256(temporary) != entry['sha256']:
            raise RuntimeError(f'CUDA runtime wheel checksum mismatch: {name}')
        temporary.replace(destination)
        print(f'verified {name}', flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name('cuda132.lock.json').read_text())
    args.directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda entry: fetch(entry, args.directory), lock['wheels']))


if __name__ == '__main__':
    main()
