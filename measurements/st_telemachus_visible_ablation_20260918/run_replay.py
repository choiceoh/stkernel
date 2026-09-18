import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from urllib.request import Request, urlopen
import uuid

ROOT = Path('/tmp/telemachus-quality-0917/visible-assistant-ablation')
SOURCE = 'f42be51dd3c60de5d7f680efdc7a1f92663de24b'
REPO = '/home/choiceoh/st-worktrees/telemachus-visible-ablation-0918'
OWNER = 'queue/st-visible-ablation0918b'
FILES = ['engine/profiles/glm53/adapter.py', 'engine/profiles/glm53/net.py',
         'engine/profiles/glm53/boot.py', 'engine/profiles/glm53/lanes.py',
         'engine/profiles/glm53/specs.py', 'engine/base/draws.py',
         'engine/base/sampler.py', 'engine/base/serve.py', 'engine/kernels/common/sampler.py']

parser = argparse.ArgumentParser(description='Replay the private visible-answer ablation only on its owned fleet hold.')
parser.add_argument('--neutralized-greedy-only', action='store_true',
                    help='Follow up the five original cases with the matched neutralized T=0 control.')
args = parser.parse_args()


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def fetch(path, body=None, timeout=10):
    req = Request('http://127.0.0.1:8001' + path,
                  data=None if body is None else json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def idle():
    state = fetch('/')
    assert state['fleet']['owner'] == OWNER and state['model'], state
    assert not any(state.get(k) for k in ('running', 'waiting', 'queued')), state
    return state


prepared = json.loads((ROOT / 'prepared-inputs.json').read_text())
baseline, neutralized = prepared['baseline'], prepared['neutralized']
start, end = prepared['receipt']['replaced_visible_body']
assert (start, end) == (47379, 47663)
assert len(baseline) == len(neutralized) == 50005
assert baseline[:start] == neutralized[:start] and baseline[end:] == neutralized[end:]
assert neutralized[start:end] == [220] * 284
assert digest(baseline) == prepared['receipt']['original_sha256']
assert digest(neutralized) == prepared['receipt']['candidate_sha256']
print(json.dumps(dict(event='prepared', owner=OWNER, source=SOURCE, changed_tokens=284)), flush=True)
deadline = time.monotonic() + 7200
while True:
    try:
        idle()
        break
    except Exception:
        if time.monotonic() >= deadline:
            raise
        time.sleep(3)

expected = {name: hashlib.sha256(subprocess.check_output(['git', '-C', REPO, 'show', SOURCE + ':' + name])).hexdigest()
            for name in FILES}


def runtime(rank_host):
    rank, host = rank_host
    code = ('import hashlib,json,os,pathlib; files=' + repr(FILES) +
            ';print(json.dumps(dict(release=os.environ.get("ST_RELEASE"),hashes={p:hashlib.sha256((pathlib.Path("/repo")/p).read_bytes()).hexdigest() for p in files})))')
    command = ['sudo', '-n', 'docker', 'exec', 'st-glm53', 'python3', '-c', code]
    if rank:
        command = ['ssh', '-o', 'BatchMode=yes', 'choiceoh@' + host, shlex.join(command)]
    result = json.loads(subprocess.check_output(command, text=True, timeout=30))
    assert result['hashes'] == expected and SOURCE.startswith(result['release'] or 'missing'), (rank, result)
    return rank, result


with concurrent.futures.ThreadPoolExecutor(4) as pool:
    ranks = dict(pool.map(runtime, enumerate(['10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4'])))
(ROOT / 'runtime-identity.json').write_text(json.dumps(dict(source=SOURCE, owner=OWNER, ranks=ranks), indent=2))
cases = [('baseline-t1-seed7', baseline, 1, 7),
         ('neutralized-t1-seed7', neutralized, 1, 7),
         ('baseline-t0-seed7', baseline, 0, 7),
         ('baseline-t1-seed11', baseline, 1, 11),
         ('neutralized-t1-seed11', neutralized, 1, 11)]
if args.neutralized_greedy_only:
    assert all((ROOT / (name + '.json')).exists() for name, *_ in cases)
    cases = [('neutralized-t0-seed7', neutralized, 0, 7)]
for name, prompt, temperature, seed in cases:
    path = ROOT / (name + '.json')
    assert not path.exists()
    before = idle()
    print(json.dumps(dict(event='request', case=name, before_served=before['served'])), flush=True)
    started = time.monotonic()
    response = fetch('/v1/engine/completions', dict(ids=prompt, temperature=temperature,
                     top_p=1, top_k=-1, seed=seed, max_tokens=2048, retain=False,
                     cache_salt='visible-ablation-' + uuid.uuid4().hex), timeout=600)
    seconds = time.monotonic() - started
    after = idle()
    assert after['served'] - before['served'] == 1 and response['cached_tokens'] == 0
    text = fetch('/detokenize', dict(tokens=response['ids']))['prompt']
    receipt = dict(case=name, source=SOURCE, owner=OWNER, prompt_tokens=len(prompt), prompt_sha256=digest(prompt),
                   temperature=temperature, seed=seed, top_p=1, top_k=-1, max_tokens=2048,
                   completion_tokens=response['completion_tokens'], cap_hit=response['completion_tokens'] == 2048,
                   cached_tokens=0, seconds=seconds, output_ids_sha256=digest(response['ids']),
                   output_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   served_before=before['served'], served_after=after['served'])
    path.write_text(json.dumps(dict(receipt=receipt, response=response, text=text), ensure_ascii=False, indent=2))
    print(json.dumps(receipt), flush=True)
state_file = 'final-state-additional.json' if args.neutralized_greedy_only else 'final-state.json'
(ROOT / state_file).write_text(json.dumps(idle(), indent=2))
print(f'DONE: {len(cases)} matched controls completed; manual semantic review required', flush=True)
