#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Boot entry accepts absent serving, but never missing metrics on a live server."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.request


def serve_args(container):
    """Read argv or the launcher's static base64 script, never execute shell."""
    command = container['Config'].get('Cmd') or []
    if len(command) == 2 and command[0] in ('-c', '-lc'):
        script = command[1]
        wrapped = re.fullmatch(r'echo ([A-Za-z0-9+/=]+) \| base64 -d > /tmp/serve\.sh; bash /tmp/serve\.sh', script)
        if wrapped:
            script = base64.b64decode(wrapped[1], validate=True).decode()
        lines = [line for line in script.splitlines() if re.match(r'^\s*vllm\s+serve\s', line)]
        command = shlex.split(lines[0]) if len(lines) == 1 else []
    args = {}
    for index, token in enumerate(command):
        for key in ('host', 'port'):
            if token == '--' + key and index + 1 < len(command):
                args[key] = command[index + 1]
            elif token.startswith('--' + key + '='):
                args[key] = token.split('=', 1)[1]
    return args


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
    port = serve_args(container).get('port')
    if port:
        if not port.isdigit() or not 0 < int(port) < 65536:
            raise ValueError('invalid serving port')
        url = 'http://127.0.0.1:' + port
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
    # The explicit public bind is visible in the container command, not VLLM knobs.
    args = serve_args(container)
    if args.get('host') != '0.0.0.0' or args.get('port') != '8000':
        return False
    profile = (repo / 'profiles/glm53.env').read_text()
    images = re.findall(r'^PROFILE_IMAGE=(.+)$', profile, re.M)
    if len(images) != 1:
        return False
    image = shlex.split(images[0], comments=True)
    if len(image) != 1:
        return False
    approved_image = subprocess.check_output(
        ['docker', 'image', 'inspect', '--format', '{{.Id}}', image[0]], text=True, timeout=10).strip()
    if not approved_image or container.get('Image') != approved_image:
        return False
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
