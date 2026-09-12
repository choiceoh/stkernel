"""Passive PR760 capture bound to one immutable consumer container."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def inspect(name):
    result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=10)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    owner = 'queue/' + args.session
    deadline = time.monotonic() + 1800
    container_id = None
    while time.monotonic() < deadline:
        data = inspect(container_id or 'st-glm53')
        if container_id:
            if not data or not data['State']['Running']:
                raise RuntimeError('owned consumer exited before metrics became available')
            try:
                with urllib.request.urlopen(args.url.rstrip('/') + '/metrics', timeout=3) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
        elif data:
            env = dict(value.split('=', 1) for value in data['Config']['Env'] if '=' in value)
            if env.get('ST_LEASE_OWNER') == owner and env.get('ST_RELEASE') == args.release:
                container_id = data['Id']
                print(json.dumps(dict(container_id=container_id, owner=owner, release=args.release)), flush=True)
        time.sleep(2)
    else:
        raise RuntimeError('owned consumer metrics were not ready within 30 minutes')
    child = subprocess.Popen([sys.executable, '-u', str(Path(args.repo) / 'bench/step_peek.py'),
                              '--url', args.url, '--seconds', '7200', '--period', '1', '--out', args.out])
    try:
        while child.poll() is None:
            data = inspect(container_id)
            if not data or not data['State']['Running']:
                break
            time.sleep(3)
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=15)
    print(json.dumps(dict(completed=True, container_id=container_id,
                          metrics_returncode=child.returncode, output=args.out)), flush=True)


if __name__ == '__main__':
    main()
