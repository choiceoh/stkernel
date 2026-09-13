"""Qualify request retirement, then run two canonical onepasses on an owned hold.

Run on srv2. This never boots or stops a container directly. The canonical
hold owns the boot and notices its own stop file in this script's finally.
Functional traffic precedes prefix reset and both recorded consumer passes.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
import urllib.request
import uuid

SHA = '8c8b031b94bb80175cfccd49cfb6329d18cdecae'
SESSION = 'st-decode-ranksum0913'
REPO = Path('/home/choiceoh/stkernel-worktrees/codex-st-oneshot-rank-sum')
LOGS = Path('/home/choiceoh/glm53-logs')
OUT = LOGS / 'st-decode-ranksum-8c8b031b-functional'
BASE = 'http://127.0.0.1:18123'
NODES = (None, 'choiceoh@10.10.10.1', 'choiceoh@10.10.10.3', 'choiceoh@10.10.10.4')


def inspect(node, name):
    cmd = ['docker', 'inspect', name]
    if node:
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', node, shlex.join(cmd)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return json.loads(p.stdout)[0] if p.returncode == 0 else None


def owned(node, name='st-glm53'):
    value = inspect(node, name)
    if not value or not value['State']['Running']:
        return None
    env = dict(e.split('=', 1) for e in value['Config']['Env'] if '=' in e)
    if env.get('ST_LEASE_OWNER') != 'queue/' + SESSION or env.get('ST_RELEASE') != SHA[:12]:
        return None
    return value


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as response:
        return response.read().decode()


def post(body, path='/v1/chat/completions', timeout=180):
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def counters():
    metrics = get('/metrics')
    return {key: int(float(re.search(r'^vllm:num_requests_' + key + r'\{[^}]*\}\s+(\S+)',
                                    metrics, re.M).group(1))) for key in ('running', 'waiting')}


def request(body, client):
    started = time.monotonic()
    response = post(body)
    choice = response['choices'][0]
    tokens = response['usage']['completion_tokens']
    assert 0 < tokens <= body['max_tokens'], response
    assert choice['finish_reason'] in ('stop', 'length'), response
    return dict(client=client, max_tokens=body['max_tokens'], completion_tokens=tokens,
                finish_reason=choice['finish_reason'], elapsed_s=time.monotonic()-started,
                response_sha256=hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest())


def main():
    OUT.mkdir(exist_ok=False)
    report = dict(source=SHA, session=SESSION, status='waiting',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='request lifecycle qualification; no throughput or answer-quality verdict', waves=[])
    def save():
        (OUT/'functional.json').write_text(json.dumps(report, indent=2)+'\n')
    def check_ranks():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda item: owned(*item), enumerate_ids))
        assert all(values), 'an owned rank exited or changed identity'
    save()
    bound = False
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            head = owned(None)
            if head:
                bound = True
                try:
                    if json.loads(get('/v1/models')).get('data'):
                        break
                except (OSError, ValueError):
                    pass
            time.sleep(3)
        else:
            raise TimeoutError('the owned door was not ready in 30 minutes')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            identities = list(pool.map(owned, NODES))
        assert all(identities), 'all four owned ranks are required'
        enumerate_ids = [(node, data['Id']) for node, data in zip(NODES, identities)]
        report['containers'] = [dict(rank=rank, id=data['Id'], image=data['Image'],
                                     started=data['State']['StartedAt']) for rank, data in enumerate(identities)]
        print('owned four-rank door ready; functional gate starts', flush=True)
        # Exactly the production supervisor's sampled four-token health request.
        body = dict(model='glm-5.3-flash', messages=[dict(role='user', content='ping')],
                    max_tokens=4, chat_template_kwargs=dict(thinking=False))
        report['health'] = request(body, 0)
        for _ in range(5):
            time.sleep(7)
            check_ranks()
            assert counters() == dict(running=0, waiting=0)
        print('sampled health survived 35 seconds and retirement', flush=True)
        save()
        for temperature in (0., 1.):
            maxima, samples = (128, 256, 384, 512), []
            finished = threading.Event()
            monitor_errors = []
            def monitor():
                while not finished.is_set():
                    try:
                        samples.append(dict(t=time.monotonic(), **counters()))
                    except Exception as error:
                        monitor_errors.append(repr(error))
                        return
                    finished.wait(.25)
            observer = threading.Thread(target=monitor)
            observer.start()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futures = []
                    for client, maximum in enumerate(maxima):
                        body = dict(model='glm-5.3-flash', temperature=temperature, max_tokens=maximum,
                                    cache_salt=uuid.uuid4().hex, chat_template_kwargs=dict(thinking=False),
                                    messages=[dict(role='user', content=f'등장인물 이름은 민수{client}입니다. '
                                        '조선 시대를 배경으로 아주 긴 모험 소설을 써 주세요. 최소 10000자 이상으로 계속 이어서 쓰세요.')])
                        futures.append(pool.submit(request, body, client))
                    replies = []
                    for future in concurrent.futures.as_completed(futures):
                        reply = future.result()
                        reply['running_when_completed'] = counters()['running']
                        replies.append(reply)
            finally:
                finished.set()
                observer.join(timeout=10)
            assert not monitor_errors, monitor_errors
            assert max(s['running'] for s in samples) == 4, 'the wave never admitted four rows'
            assert any(r['running_when_completed'] > 0 for r in replies), 'no staggered retirement observed'
            time.sleep(5)
            check_ranks()
            assert counters() == dict(running=0, waiting=0)
            report['waves'].append(dict(temperature=temperature, requests=replies, metrics=samples))
            save()
            print(f'C4 temperature={temperature:g}: four responses, staggered retirement, unchanged ranks PASS', flush=True)
        report['status'] = 'PASS'
        save()
        environment = dict(os.environ, GLM53_API_PORT='18123', BENCH_MODEL='glm-5.3-flash',
                           ONEPASS_JSONL=str(LOGS/'st-decode-ranksum-8c8b031b.jsonl'),
                           ST_BRACKET_SHA=SHA, ST_BRACKET_COLD='reset', FLEET_SESSION=SESSION)
        # Functional traffic is not part of either measured pass. Canonical
        # preparation, unique salts, quality grading and profiles stay intact.
        for index in (1, 2):
            check_ranks()
            post({}, '/v1/prefix/reset', timeout=60)
            environment['ONEPASS_RUN_INDEX'] = str(index)
            log = OUT/f'onepass-{index}.log'
            print(f'canonical onepass {index}/2 starts on the same boot; full log {log}', flush=True)
            with log.open('w') as stream:
                subprocess.run(['python3', '-u', 'bench/onepass.py', '--name', 'rank-sum', '--require-exclusive'],
                               cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=5400)
            print(f'canonical onepass {index}/2 complete', flush=True)
        check_ranks()
        (OUT/'consumer-complete.json').write_text(json.dumps(dict(source=SHA, complete_onepasses=2,
              same_boot=True, first_pass_follows='functional gate and prefix reset'))+'\n')
    except BaseException as error:
        report['error'] = repr(error)
        if report['status'] != 'PASS':
            report['status'] = 'FAIL'
        save()
        raise
    finally:
        if bound:
            (LOGS/'st-bracket'/SESSION/'stop').touch()
            print('requested the canonical hold to stop its own boot and release', flush=True)


if __name__ == '__main__':
    main()
