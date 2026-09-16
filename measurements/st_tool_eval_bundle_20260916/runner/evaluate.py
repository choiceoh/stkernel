"""Run the unchanged official benchmark against an owned candidate hold, then release it."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--sha', required=True)
parser.add_argument('--session', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--protocol', choices=('original_t1', 'reference_t0', 'reference_t1'), default='original_t1')
parser.add_argument('--probes-first', action='store_true')
parser.add_argument('--diagnostics-only', action='store_true')
parser.add_argument('--probe-script', default='probe_tool_context.py')
args = parser.parse_args()
root = Path(args.out)
root.mkdir(parents=True, exist_ok=True)
url = 'http://10.10.10.2:8001'
cli_sha = ('992a6978ecbee2d72fa2ead9ccc509436769d088' if args.protocol == 'original_t1' else
           '6be685f0e6b9e0df05ed024848cf7fe1eca48752')

def now():
    return datetime.now().astimezone().isoformat()

def head(script):
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                               '10.10.10.2', 'python3 -'], input=script, text=True,
                              capture_output=True, timeout=45)
    if response.returncode:
        raise RuntimeError(response.stderr[-1600:])
    return json.loads(response.stdout)

def identity():
    return head('''
from pathlib import Path
import json,subprocess
h=Path.home()
lease=json.loads((h/'glm53-logs/st-fleet.lock').read_text())
out={'lease':lease}
p=subprocess.run(['docker','inspect','st-glm53'],capture_output=True,text=True)
if p.returncode==0:
 c=json.loads(p.stdout)[0]
 out['container']={'id':c['Id'],'running':c['State']['Running'],'started':c['State']['StartedAt'],
  'mount':next(m['Source'] for m in c['Mounts'] if m['Destination']=='/repo'),'command':c['Config']['Cmd']}
print(json.dumps(out))
''')

def snapshot(label):
    data = {'at': now()}
    try:
        data['identity'] = identity()
    except Exception as exc:
        data['identity_error'] = str(exc)
    for endpoint in ('metrics', 'v1/models'):
        try:
            with urllib.request.urlopen(url + '/' + endpoint, timeout=10) as response:
                body = response.read().decode()
                data[endpoint] = response.status
            (root / (label + '-' + endpoint.replace('/', '-') + '.txt')).write_text(body)
        except Exception as exc:
            data[endpoint] = str(exc)
    (root / (label + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return data

command = [str(Path.home()/'.local/bin/uv'), 'tool', 'run', '--from',
           'git+https://github.com/SeraphimSerapis/tool-eval-bench.git@'+cli_sha,
           'tool-eval-bench', 'run', '--base-url', url, '--model', 'glm-5.3-flash',
           '--backend', 'vllm', '--format', 'openai', '--hardmode', '--temperature', '1',
           '--top-p', '0.95', '--seed', '42', '--parallel', '1', '--trials', '1', '--timeout', '120',
           '--backend-kwargs', json.dumps({'chat_template_kwargs': {'thinking': True}, 'retain': False}),
           '--json-file', str(root/'result.json'), '--output-dir', str(root/'runs'), '--no-live',
           '--label', f'ST {args.sha[:8]} parser fix T1 C1 tool-eval 2.6.0']
manifest = {'status': 'waiting_for_hold', 'created_at': now(), 'engine_sha': args.sha,
            'session': args.session, 'cli_version': '2.6.0', 'cli_commit': cli_sha, 'command': command,
            'model': 'glm-5.3-flash', 'temperature': 1, 'top_p': .95, 'seed': 42, 'concurrency': 1,
            'trials': 1, 'thinking': True, 'retain': False, 'scenarios': 88, 'max_tokens_per_turn': 4096,
            'max_turns_default': 8, 'timeout_seconds': 120, 'candidate_hold': True}
manifest['protocol'] = args.protocol
if args.diagnostics_only:
    manifest['protocol'] = 'diagnostics_only'
    manifest['scenarios'] = 0
if args.protocol == 'reference_t0':
    # Match the user's reference command's sampling and template defaults.
    # Keep the original T=1 experiment separate and unmodified.
    command = [str(Path.home()/'.local/bin/uv'), 'tool', 'run', '--from',
               'git+https://github.com/SeraphimSerapis/tool-eval-bench.git@'+cli_sha,
               'tool-eval-bench', '--backend', 'vllm', '--base-url', url,
               '--model', 'glm-5.3-flash', '--seed', '42', '--hardmode',
               '--json-file', str(root/'result.json'), '--output-dir', str(root/'runs'),
               '--no-live', '--label', f'ST {args.sha[:8]} reference CLI 6be685f0 T0 C1']
    manifest.update(command=command, cli_version='2.6.1.dev65+g6be685f0e', temperature=0,
                    top_p='server default', thinking='server default', retain='server default')
elif args.protocol == 'reference_t1':
    # The reference's fixed CLI revision, while retaining the original T=1
    # sampling settings. Version changes are explicit in the manifest/report.
    command[-1] = f'ST {args.sha[:8]} tool template fix reference CLI 6be685f0 T1 C1'
    manifest.update(command=command, cli_version='2.6.1.dev65+g6be685f0e')

def save():
    (root/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')

save()
owned = False
try:
    for attempt in range(360):
        try:
            check = identity()
            c = check.get('container', {})
            owned = check['lease'].get('owner') == 'queue/' + args.session
            if owned and c.get('running') and c.get('mount') == '/home/choiceoh/st-releases/' + args.sha[:12]:
                with urllib.request.urlopen(url+'/v1/models', timeout=6) as response:
                    models = json.load(response)
                if any(m['id'] == 'glm-5.3-flash' for m in models['data']):
                    break
        except Exception as exc:
            if attempt % 12 == 0:
                print(now(), 'waiting:', str(exc)[:240], flush=True)
        time.sleep(15)
    else:
        raise TimeoutError('Candidate hold did not become ready within 90 minutes')
    if args.probes_first or args.diagnostics_only:
        manifest.update(status='diagnostics', diagnostics_started_at=now())
        save()
        probe = Path(__file__).with_name(args.probe_script)
        with (root/'diagnostics.log').open('ab', buffering=0) as log:
            p = subprocess.run(['python3', str(probe), '--base-url', url, '--out', str(root/'diagnostics')],
                               stdout=log, stderr=log, timeout=25*60)
        manifest['diagnostics_exit_code'] = p.returncode
        if p.returncode:
            raise RuntimeError('Diagnostic runner failed; inspect diagnostics.log')
    if args.diagnostics_only:
        manifest.update(status='complete', exit_code=0)
    else:
        snapshot('before')
        manifest.update(status='running', started_at=now())
        save()
        print(now(), 'RUN_STARTED', args.sha, flush=True)
        env = {k:v for k,v in os.environ.items() if not k.startswith('TOOL_EVAL_')}
        env['PYTHONUNBUFFERED'] = '1'
        with (root/'stdout.log').open('ab', buffering=0) as out, (root/'progress.jsonl').open('ab', buffering=0) as err:
            result = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    timeout=75*60)
        manifest.update(status='complete' if result.returncode == 0 else 'failed', exit_code=result.returncode)
except Exception as exc:
    manifest.update(status='failed', error=type(exc).__name__+': '+str(exc))
finally:
    snapshot('after')
    try:
        release = head('''
from pathlib import Path
import json
h=Path.home();session=''' + repr(args.session) + '''
lease=json.loads((h/'glm53-logs/st-fleet.lock').read_text())
ours=lease.get('owner')=='queue/'+session
if ours:
 p=h/'glm53-logs/st-bracket'/session/'stop';p.parent.mkdir(parents=True,exist_ok=True);p.touch()
print(json.dumps({'stop_requested':ours,'owner':lease.get('owner')}))
''')
        manifest['release'] = release
    except Exception as exc:
        manifest['release_error'] = str(exc)
    manifest['finished_at'] = now()
    save()
    (root/'status.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(now(), 'RUN_FINISHED', manifest['status'], flush=True)
