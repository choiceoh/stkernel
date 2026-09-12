"""Wait for an ST boot without treating a completed late boot as a failure."""
import argparse
import json
import subprocess
import time
import urllib.request


def wait_ready(probe, running, *, timeout=1800, interval=5,
               clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while True:
        if not running():
            raise RuntimeError('candidate stopped or its container identity changed during boot')
        # Check readiness before the deadline, including after the final sleep.
        # An already healthy engine must not be rolled back because startup
        # finished at the end of a polling interval.
        if probe():
            return
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(f'engine was not ready within {timeout} seconds')
        sleep(min(interval, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--image', required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()

    def probe():
        try:
            with urllib.request.urlopen(args.base.rstrip('/') + '/v1/models', timeout=3) as response:
                return response.status == 200 and bool(json.load(response).get('data'))
        except (OSError, ValueError):
            return False

    def running():
        result = subprocess.run(['docker', 'inspect', 'st-glm53'], capture_output=True, text=True)
        if result.returncode:
            return False
        container = json.loads(result.stdout)[0]
        return container['State']['Running'] and container['Config']['Image'] == args.image

    wait_ready(probe, running, timeout=args.timeout)
    print('ST engine is ready.', flush=True)


if __name__ == '__main__':
    main()
