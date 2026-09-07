#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Boot entry accepts absent serving, but never missing metrics on a live server."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request


def inspect():
    result = subprocess.run(['docker', 'inspect', 'glm53'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        # Distinguish a missing container from an unavailable Docker daemon.
        subprocess.run(['docker', 'info'], check=True, stdout=subprocess.DEVNULL, timeout=10)
        return None
    return json.loads(result.stdout)[0]


def idle(container, url):
    if not container or not container['State']['Running']:
        return 'stopped'
    # A predecessor may have served on its isolated loopback experiment port.
    command = ' '.join(container['Config'].get('Cmd') or [])
    port = re.search(r'--port(?:=|\s+)([0-9]+)(?:\s|$)', command)
    if port:
        url = 'http://127.0.0.1:' + port[1]
    with urllib.request.urlopen(url + '/health', timeout=5) as response:
        if response.status != 200:
            raise ValueError('serving is not healthy')
    text = urllib.request.urlopen(url + '/metrics', timeout=5).read().decode()
    for key in ('num_requests_running', 'num_requests_waiting'):
        values = re.findall(r'^vllm:' + key + r'(?:\{[^}]*\})?\s+(\S+)', text, re.M)
        if not values or any(float(v) != 0 for v in values):
            raise ValueError('live serving lacks idle request counters: ' + key)
    return text


def production_current(repo, container):
    if not container or not container['State']['Running']:
        return False
    env = dict(v.split('=', 1) for v in container['Config']['Env'] if '=' in v)
    # The explicit public bind is visible in the container command, not VLLM knobs.
    command = ' '.join(container['Config'].get('Cmd') or [])
    if not re.search(r'--host(?:=|\s+)0\.0\.0\.0(?:\s|$)', command) or not re.search(r'--port(?:=|\s+)8000(?:\s|$)', command):
        return False
    profile = (repo / 'profiles/glm53.env').read_text()
    match = re.search(r'^PROFILE_OVERLAY_DIR=["\']?([^"\'\n]+)', profile, re.M)
    if not match:
        return False
    manifest = Path(match[1]) / 'manifest.tsv'
    data = manifest.read_bytes()
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    stamp = Path(os.environ.get('MK_OVERLAY_STAMP', '/home/choiceoh/glm53-cache/.overlay-sha')).read_text().strip()
    if f'# source_commit={revision}' not in data.decode().splitlines() or hashlib.sha256(data).hexdigest() != stamp:
        return False
    from onepass import _served_build
    build = _served_build(str(repo))
    if 'knobs' not in build or build['knobs'] or not build.get('boot_id'):
        return False
    return idle(container, 'http://10.10.10.2:8000') != 'stopped'


def main():
    container = inspect()
    if sys.argv[1] == 'production-current':
        return 0 if production_current(Path(sys.argv[2]), container) else 1
    session = os.environ['FLEET_SESSION']
    directory = Path(os.environ.get('FLEET_DIR', '/home/choiceoh/glm53-logs/fleet'))
    if (directory / 'holder').read_text().split('|')[0] != session:
        raise ValueError('entry requires the fleet hold')
    text = idle(container, os.environ.get('HEAD_URL', 'http://10.10.10.2:8000'))
    Path(sys.argv[2]).write_text(text + '\n')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
