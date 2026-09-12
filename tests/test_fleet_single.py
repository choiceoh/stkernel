#!/usr/bin/env python3
"""The single-GPU lane: a one-GPU check goes to the 5050 on ost-97x, never to the four Sparks.

No GPU and no network. ssh, docker, flock and the GNU tools the queue expects are shims,
the helper scripts that would touch the experiment database answer canned, and the
queue's own admission functions run from this checkout's bench/fleet.sh.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import fleet_handoff as handoff
import fleet_onepass as policy
import fleet_single as single

BASH = shutil.which('bash')


class Done:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class EvidenceTests(unittest.TestCase):
    """That host's own GPU process list is the evidence; not knowing is a reason, never free."""

    def test_free_busy_unreachable_and_a_failed_query(self):
        calls = []
        def free(argv, **kwargs):
            calls.append(argv)
            return Done(0, '')
        self.assertEqual(single.evidence('ost-97x', run=free), [])
        self.assertEqual(calls[0][-2:], ['choiceoh@ost-97x', single.QUERY])   # the fleet's user unless named
        self.assertIn('BatchMode=yes', calls[0])
        busy = single.evidence('ost-97x', run=lambda *a, **k: Done(0, '4242, python3, 512\n4243, python3, 64\n'))
        self.assertEqual(busy, ['ost-97x: busy outside this queue -- pid 4242 (python3) and 1 more'])
        def unreachable(*args, **kwargs):
            raise subprocess.TimeoutExpired('ssh', 8)
        self.assertIn('unreachable', single.evidence('ost-97x', run=unreachable)[0])
        failed = single.evidence('ost-97x', run=lambda *a, **k: Done(255, '', 'ssh: connect to host ost-97x port 22: No route to host\n'))
        self.assertIn('cannot say it is free', failed[0])
        self.assertIn('No route to host', failed[0])
        self.assertIn('lane is off', single.evidence('')[0])

    def test_a_user_in_the_host_name_is_kept(self):
        self.assertEqual(single.target('ost@ost-97x'), 'ost@ost-97x')
        self.assertEqual(single.target('ost-97x'), 'choiceoh@ost-97x')

    def test_the_cache_remembers_one_answer_per_ttl_and_per_host(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = [1000.0]
            answers = iter([Done(0, '1, x, 1\n'), Done(0, ''), Done(0, '')])
            run = lambda *a, **k: next(answers)
            now = lambda: clock[0]
            first = single.cached_evidence('ost-97x', directory, now=now, run=run)
            self.assertTrue(first and 'busy' in first[0])
            clock[0] += 5
            self.assertEqual(single.cached_evidence('ost-97x', directory, now=now, run=run), first)   # remembered
            clock[0] += single.TTL_S
            self.assertEqual(single.cached_evidence('ost-97x', directory, now=now, run=run), [])      # asked again
            self.assertEqual(single.cached_evidence('other', directory, now=now, run=run), [])        # another host is asked
            with self.assertRaises(StopIteration):                                                     # ... exactly once
                single.cached_evidence('third', directory, now=now, run=run)

    def test_main_prints_reasons_and_exits_nonzero_when_not_free(self):
        out = io.StringIO()
        with patch.object(single, 'evidence', return_value=['ost-97x: busy outside this queue -- pid 1 (x)']), \
                contextlib.redirect_stdout(out):
            self.assertEqual(single.main(['evidence', '--host', 'ost-97x']), 1)
        self.assertIn('busy outside', out.getvalue())
        with patch.object(single, 'evidence', return_value=[]):
            self.assertEqual(single.main(['evidence', '--host', 'ost-97x']), 0)
        with patch.dict(os.environ, {'FLEET_SINGLE_GPU_HOST': ''}):
            self.assertEqual(single.host(), '')
            self.assertEqual(single.label(), 'off')
        with patch.dict(os.environ, {'FLEET_SINGLE_GPU_HOST': 'ost-97x', 'FLEET_SINGLE_GPU_NAME': '5050'}):
            self.assertEqual(single.label(), '5050 on ost-97x')


class ContractTests(unittest.TestCase):
    """The queue, its helpers, the runner and the launcher agree on what the lane is."""

    def setUp(self):
        self.fleet = (ROOT / 'bench/fleet.sh').read_text()
        self.runner = (ROOT / 'probes/run_engine_probe.sh').read_text()
        self.launcher = (ROOT / 'launchers/start-st-glm53.sh').read_text()

    def test_defaults_agree_between_the_shell_and_the_module(self):
        self.assertIn('FLEET_SINGLE_GPU_HOST=${FLEET_SINGLE_GPU_HOST-' + single.DEFAULT_HOST + '}', self.fleet)
        self.assertIn('FLEET_SINGLE_GPU_NAME=${FLEET_SINGLE_GPU_NAME:-' + single.DEFAULT_GPU + '}', self.fleet)
        self.assertIn('export FLEET_SINGLE_GPU_HOST FLEET_SINGLE_GPU_NAME', self.fleet)
        self.assertEqual((single.DEFAULT_HOST, single.DEFAULT_GPU), ('ost-97x', '5050'))

    def test_the_lane_has_its_own_holder_and_the_fleet_readers_keep_theirs(self):
        self.assertIn('HS=$FLEET_DIR/holder-single', self.fleet)
        self.assertEqual(handoff.holder_path('/f', 'single'), Path('/f/holder-single'))
        for kind in ('boot', 'probe'):
            self.assertEqual(handoff.holder_path('/f', kind), Path('/f/holder'))
        # the ST launcher reads the FLEET's holder and nothing else: a check on the 5050 must
        # never look like the fleet being held to it
        self.assertIn('FLEET_HOLDER=${FLEET_HOLDER:-/home/choiceoh/glm53-logs/fleet/holder}', self.launcher)
        self.assertNotIn('holder-single', self.launcher)
        # each lane takes its own head of the ranked order, not the queue's first row
        self.assertIn('[ "$(lane_front "$kind")" = "$s" ] || return 1', self.fleet)
        self.assertNotIn('''[ "$(head -1 "$Q" | cut -d'|' -f2)" = "$s" ] || return 1''', self.fleet)
        # the fleet's occupancy checks belong to the fleet lane; the single lane asks its host
        self.assertIn('if [ "$kind" != single ]; then', self.fleet)
        self.assertIn('single_refused "$s" "$why"; return 1', self.fleet)
        self.assertIn('FLEET_RULES=2', self.fleet)

    def test_the_runner_goes_where_the_lane_says_and_takes_no_lease_there(self):
        self.assertIn('probe_host=${ST_PROBE_HOST:-}', self.runner)
        remote = self.runner[self.runner.index('probe_host=${ST_PROBE_HOST:-}'):self.runner.index('mkdir -p "$cache"')]
        self.assertIn('rsync -a --delete --exclude __pycache__ -e "ssh $SSHOPT" "$repo/engine" "$repo/probes"', remote)
        self.assertIn('probe_host="choiceoh@$probe_host"', remote)
        self.assertIn('''trap 'ssh $SSHOPT "$probe_host" "docker rm -f $NAME"''', remote)
        self.assertIn('exit $rc', remote)
        self.assertNotIn('fleet_lease', remote)
        # the fleet path is untouched: the lease, the wait, the heartbeat
        local = self.runner[self.runner.index('mkdir -p "$cache"'):]
        for text in ('fleet_lease acquire', 'fleet_lease_beat', 'ST_PROBE_NO_LEASE'):
            self.assertIn(text, local)
        # the supervisor sets the host for the single lane, after the payload's own env prefix
        boot = (ROOT / 'bench/fleet_boot.py').read_text()
        self.assertIn("environment['ST_PROBE_HOST'] = host", boot)
        self.assertIn("if getattr(self, 'kind', 'boot') == handoff.SINGLE", boot)

    def test_the_policy_counts_gpus(self):
        self.assertEqual(policy.gpus_needed('probes/run_engine_check.sh', ['--layers', '0-4']), 1)
        self.assertEqual(policy.gpus_needed('probes/run_engine_probe.sh', ['probes/engine_kernel_check.py']), 1)
        self.assertEqual(policy.gpus_needed('probes/run_engine_probe.sh',
                                            ['engine/profiles/glm53/check.py', '--distributed']), 4)
        for entry in ('bench/pair.sh', 'bench/chain.sh', 'bench/ab-lever.sh', 'bench/onepass.py',
                      'bench/experiments.py', 'probes/run_ar_consumer_campaign.sh'):
            self.assertEqual(policy.gpus_needed(entry, []), 4, entry)


def shim(directory, name, text):
    path = directory / name
    path.write_text(text)
    path.chmod(0o700)
    return path


PYTHON_SHIM = '''#!{python}
import os, pathlib, sys
name = pathlib.Path(sys.argv[1]).name if len(sys.argv) > 1 else ''
if name in ('fleet_priority.py', 'fleet_idle.py'):      # the ranked order as written; the fleet's clock is not under test
    raise SystemExit(0)
if name == 'fleet_pause.py':                              # nothing parked, nothing paused, preparation fine
    raise SystemExit({{'reconcile': 1, 'admission': 0, 'is-paused': 1}}.get(sys.argv[2], 0))
os.execv({python!r}, [{python!r}, *sys.argv[1:]])
'''
SSH_SHIM = '''#!{python}
import os, pathlib, sys
lines = pathlib.Path(os.environ['SSH_ANSWER']).read_text().splitlines()
sys.stdout.write(''.join(line + '\\n' for line in lines[1:]))
raise SystemExit(int(lines[0]) if lines else 0)
'''
DATE_SHIM = '''#!{python}
import subprocess, sys, time
a = sys.argv[1:]
if len(a) >= 3 and a[0] == '-d' and a[1].startswith('@') and a[2].startswith('+'):
    print(time.strftime(a[2][1:], time.localtime(int(a[1][1:]))))
    raise SystemExit(0)
raise SystemExit(subprocess.call(['/bin/date', *a]))
'''


@unittest.skipUnless(BASH, 'bash is required')
class LaneAdmissionTests(unittest.TestCase):
    """The queue's admission, released from its dispatcher: two lanes, two holders, and
    neither waits for the other."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        for relative in ('bench/fleet.sh', 'bench/fleet_handoff.py', 'bench/fleet_single.py', 'bench/fleet_pause.py',
                         'bench/fleet_pending.py', 'bench/fleet_idle.py', 'bench/fleet_launch.py',
                         'launchers/lib/fleet-lease.sh', 'engine/base/fleet_lease.py'):
            (self.repo / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, self.repo / relative)
        self.logs = self.root / 'logs'
        self.fleet = self.logs / 'fleet'
        self.fleet.mkdir(parents=True)
        self.answer = self.root / 'ssh-answer'     # first line: exit code; the rest: nvidia-smi's compute apps
        self.answer.write_text('0\n')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        shim(self.bin, 'python3', PYTHON_SHIM.format(python=sys.executable))
        shim(self.bin, 'ssh', SSH_SHIM.format(python=sys.executable))
        shim(self.bin, 'date', DATE_SHIM.format(python=sys.executable))
        shim(self.bin, 'flock', '#!/bin/sh\nexit 0\n')
        shim(self.bin, 'timeout', '#!/bin/sh\nshift\nexec "$@"\n')
        shim(self.bin, 'docker', '#!/bin/sh\nexit 0\n')
        source = (ROOT / 'bench/fleet.sh').read_text()
        self.library = source[:source.index('\ncmd=${1:-status}')]
        self.environment = dict(PATH=str(self.bin) + os.pathsep + os.environ.get('PATH', ''), HOME=str(self.root),
                                REPO=str(self.repo), FLEET_RUNNER_REPO=str(self.repo), LOGD=str(self.logs),
                                FLEET_DIR=str(self.fleet), FLEET_NODES_IPS='', HEAD_URL='http://127.0.0.1:9',
                                FLEET_HEAD=socket.gethostname().split('.')[0], FLEET_LEASE_PATH=str(self.root / 'lease'),
                                SSH_ANSWER=str(self.answer), LANG='C', LC_ALL='C')

    def run_fleet(self, script, **environment):
        result = subprocess.run([BASH, '-c', self.library + '\n' + script], cwd=self.repo,
                                env=dict(self.environment, **environment), text=True, capture_output=True, timeout=120)
        self.assertNotIn('command not found', result.stderr, result.stderr)
        return result

    def queue(self, *rows):
        (self.fleet / 'queue').write_text(''.join('|'.join(map(str, row)) + '\n' for row in rows))

    def test_the_lanes_hold_separately_and_never_wait_for_each_other(self):
        now, pid = int(time.time()), os.getpid()
        self.queue((1, 'A', now, 30, 'a boot', 'boot', pid), (2, 'B', now, 5, 'a check', 'single', pid),
                   (3, 'C', now, 30, 'another boot', 'boot', pid), (4, 'D', now, 5, 'another check', 'single', pid))
        result = self.run_fleet('''
with_lock _try_hold A $$ 30 "a boot" boot; echo "A=$?"
with_lock _try_hold B $$ 5 "a check" single; echo "B=$?"
with_lock _try_hold C $$ 30 "another boot" boot; echo "C=$?"
with_lock _try_hold D $$ 5 "another check" single; echo "D=$?"
echo "held: fleet=$(cut -d'|' -f1 "$H") single=$(cut -d'|' -f1 "$HS")"
single_line
with_lock _release B; echo "releaseB=$?"
printf '0\\n4242, python3, 512\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR/.single-gpu-evidence"
with_lock _try_hold D $$ 5 "another check" single; echo "D2=$?"
with_lock _try_hold D $$ 5 "another check" single; echo "D3=$?"
single_line
printf '255\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR/.single-gpu-evidence"
with_lock _try_hold D $$ 5 "another check" single; echo "D4=$?"
printf '0\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR/.single-gpu-evidence"
with_lock _try_hold D $$ 5 "another check" single; echo "D5=$?"
echo "held: fleet=$(cut -d'|' -f1 "$H") single=$(cut -d'|' -f1 "$HS")"
with_lock _release A; echo "releaseA=$?"
with_lock _try_hold C $$ 30 "another boot" boot; echo "C2=$?"
with_lock _kick --force single; echo "kick=$?"
echo "held: fleet=$(cut -d'|' -f1 "$H") single=$(cat "$HS" 2>/dev/null | cut -d'|' -f1)"
echo "queue=$(grep -c . "$Q")"
''')
        # the queue's pid column is this python process and the holders carry the bash's own
        # $$; both are alive, which is all admission asks of them
        out = result.stdout
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('A=0', 'B=0', 'C=1', 'D=1', 'held: fleet=A single=B',
                         'single (5050 on ost-97x): HELD by B [single]', 'releaseB=0',
                         'D2=1', 'D3=1', 'single (5050 on ost-97x): ost-97x: busy outside this queue -- pid 4242 (python3)',
                         'D4=1', 'D5=0', 'held: fleet=A single=D', 'releaseA=0', 'C2=0', 'kicked', 'kick=0',
                         'held: fleet=C single=', 'queue=0'):
            self.assertIn(expected, out, out + result.stderr)
        log = (self.fleet / 'log').read_text()
        self.assertIn('GO B (pid', log)
        self.assertIn('[single: 5050 on ost-97x]', log)
        self.assertIn('release B [single]', log)
        self.assertEqual(log.count('hold refused (single)'), 2, log)     # busy once, unreachable once -- not once per poll
        self.assertIn('busy outside this queue -- pid 4242 (python3); D waits', log)
        self.assertIn('cannot read its GPU (rc 255)', log)
        ledger = [line.split('\t') for line in (self.fleet / 'ledger.tsv').read_text().splitlines()]
        self.assertEqual([(row[1], row[2], row[5].strip()) for row in ledger],   # BSD wc pads its count
                         [('B', 'single', '0'), ('A', 'boot', '0')])
        self.assertFalse((self.fleet / 'holder-single').exists())
        self.assertTrue((self.fleet / 'holder').read_text().startswith('C|'))

    def test_the_lane_off_refuses_rather_than_answering_free(self):
        now = int(time.time())
        self.queue((1, 'B', now, 5, 'a check', 'single', os.getpid()))
        result = self.run_fleet('''
with_lock _try_hold B $$ 5 "a check" single; echo "B=$?"
single_line
single_gpu_evidence
''', FLEET_SINGLE_GPU_HOST='')
        self.assertIn('B=1', result.stdout)
        self.assertIn('single: off (FLEET_SINGLE_GPU_HOST is empty', result.stdout)
        self.assertIn('lane is off', result.stdout)
        self.assertFalse((self.fleet / 'holder-single').exists())


@unittest.skipUnless(BASH, 'bash is required')
class RunLaneDecisionTests(unittest.TestCase):
    """`fleet.sh run --gpu` puts a one-GPU ST check into the single lane before preparation,
    keeps a boot in the fleet lane, and honours --fleet and an empty host. Preparation is
    a sentinel: nothing is queued and no payload runs."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        self.logs = self.root / 'logs'
        self.directory = self.logs / 'fleet'
        self.directory.mkdir(parents=True)
        for relative in ('bench/fleet.sh', 'bench/fleet_onepass.py', 'bench/fleet_prepare.py', 'bench/fleet_prepared.py',
                         'bench/fleet_classify.py', 'bench/fleet_single.py', 'bench/pair.sh', 'bench/chain.sh',
                         'bench/ab-lever.sh', 'bench/onepass.py', 'bench/onepass_deploy.py', 'bench/measurement_contract.py',
                         'bench/fleet_handoff.py', 'bench/fleet_pending.py', 'bench/fleet_idle.py',
                         'probes/run_ar_consumer_campaign.sh', 'probes/run_engine_probe.sh', 'probes/run_engine_check.sh',
                         *policy.ST_PROBES):
            (self.repo / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, self.repo / relative)
        self.prepared = self.root / 'preparation-started'
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        shim(self.bin, 'python3', '#!' + sys.executable + '\n'
             'import os, pathlib, sys\n'
             'if len(sys.argv)>1 and pathlib.Path(sys.argv[1]).name=="fleet_prepare.py":\n'
             '    pathlib.Path(os.environ["PREPARATION_SENTINEL"]).write_text("started")\n'
             '    raise SystemExit(79)\n'
             'os.execv(' + repr(sys.executable) + ', [' + repr(sys.executable) + ', *sys.argv[1:]])\n')
        self.environment = {'PATH': str(self.bin) + os.pathsep + os.environ.get('PATH', ''),
                            'HOME': str(self.root), 'REPO': str(self.repo), 'LOGD': str(self.logs),
                            'FLEET_DIR': str(self.directory), 'PREPARATION_SENTINEL': str(self.prepared)}

    def run_fleet(self, *args, **environment):
        return subprocess.run([BASH, str(self.repo / 'bench/fleet.sh'), *args], cwd=self.repo,
                              env=dict(self.environment, **environment), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)

    def assert_stopped_at_preparation(self, result):
        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertTrue(self.prepared.exists())
        self.prepared.unlink()
        self.assertFalse((self.directory / 'holder').exists())
        self.assertFalse((self.directory / 'holder-single').exists())
        self.assertFalse((self.directory / 'queue').read_text())

    def test_a_one_gpu_check_takes_the_single_lane_and_a_boot_or_fleet_keeps_the_sparks(self):
        check = ('bash', 'probes/run_engine_check.sh', '--layers', '0-4')
        lane = 'needs one GPU, not four: single-GPU lane (5050 on ost-97x)'
        result = self.run_fleet('run', '--gpu', 'st', '5', 'kernel check', '--', *check)
        self.assertIn(lane, result.stdout)
        self.assert_stopped_at_preparation(result)
        result = self.run_fleet('run', '--gpu', '--fleet', 'st', '5', 'kernel check', '--', *check)
        self.assertNotIn(lane, result.stdout)
        self.assert_stopped_at_preparation(result)
        result = self.run_fleet('run', '--gpu', 'st', '5', 'kernel check', '--', *check, FLEET_SINGLE_GPU_HOST='')
        self.assertNotIn(lane, result.stdout)
        self.assert_stopped_at_preparation(result)
        distributed = ('bash', 'probes/run_engine_probe.sh', 'engine/profiles/glm53/check.py', '--distributed')
        result = self.run_fleet('run', '--gpu', 'st', '30', 'four ranks', '--', *distributed)
        self.assertNotIn(lane, result.stdout)
        self.assert_stopped_at_preparation(result)
        result = self.run_fleet('run', '--gpu', 'pair', '25', 'a boot', '--', 'bash', 'bench/pair.sh', 'A', '')
        self.assertNotIn(lane, result.stdout)
        self.assert_stopped_at_preparation(result)

    def test_preflight_knows_the_lane_and_refuses_a_boot_in_it(self):
        result = self.run_fleet('preflight', '--single', 'st', '--', 'bash', 'probes/run_engine_check.sh', '--layers', '0-4')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('SKIP declared-knob check (single: no launcher in the path)', result.stdout)
        self.assertIn('-> PASS', result.stdout)
        result = self.run_fleet('preflight', '--single', 'st', '--', 'bash', 'bench/pair.sh', 'A')
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('needs the four Sparks', result.stdout)
        self.assertFalse(self.prepared.exists())


if __name__ == '__main__':
    unittest.main()
