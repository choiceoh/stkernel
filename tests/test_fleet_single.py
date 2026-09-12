#!/usr/bin/env python3
"""The single-GPU lane: a one-GPU check runs on one Spark beside production, never on the four.

No GPU and no network. ssh, docker, flock and the GNU tools the queue expects are shims,
the helper scripts that would touch the experiment database answer canned, and the
queue's own admission functions run from this checkout's bench/fleet.sh.
"""
import contextlib
import io
import os
from pathlib import Path
import re
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
FREE = 'MemTotal:       127535316 kB\nMemAvailable:   41943040 kB\n---\n'     # 40 GiB available
TIGHT = 'MemAvailable:   20971520 kB\n---\n'                                   # 20 GiB: 12 left, under the floor
BUSY = 'MemAvailable:   41943040 kB\n---\nst-probe-srv2-77\n'


class Done:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def answering(text, rc=0):
    return lambda *args, **kwargs: Done(rc, text)


class EvidenceTests(unittest.TestCase):
    """Beside production the GPU is never free: ROOM is the evidence, and not knowing is a reason."""

    def test_room_is_the_evidence_beside_production(self):
        self.assertEqual(single.evidence('srv4', 8, run=answering(FREE)), [])
        tight = single.evidence('srv4', 8, run=answering(TIGHT))
        self.assertEqual(len(tight), 1)
        self.assertIn('no room beside production', tight[0])
        self.assertIn('MemAvailable 20.0 GiB', tight[0])
        self.assertIn("this check's budget 8.0 GiB, floor 16.0: 12.0 GiB would be left", tight[0])
        busy = single.evidence('srv4', 8, run=answering(BUSY))
        self.assertEqual(busy, ['srv4: a probe is already running there (st-probe-srv2-77)'])
        # the budget is the check's, and a bigger one eats the same room
        self.assertIn('no room', single.evidence('srv4', 30, run=answering(FREE))[0])
        with patch.dict(os.environ, {'ST_PROBE_GIB': '30'}):
            self.assertEqual(single.budget_gib(), 30.0)
            self.assertIn('10.0 GiB would be left', single.evidence('srv4', run=answering(FREE))[0])
        with patch.dict(os.environ, {'ST_PROBE_GIB': 'lots'}):
            self.assertEqual(single.budget_gib(), single.DEFAULT_BUDGET_GIB)

    def test_unreachable_or_unreadable_is_not_room(self):
        def unreachable(*args, **kwargs):
            raise subprocess.TimeoutExpired('ssh', 8)
        self.assertIn('unreachable', single.evidence('srv4', 8, run=unreachable)[0])
        failed = single.evidence('srv4', 8, run=answering('', 255))
        self.assertIn('cannot say it has room', failed[0])
        self.assertIn('rc 255', failed[0])
        self.assertIn('cannot read MemAvailable', single.evidence('srv4', 8, run=answering('---\n'))[0])
        self.assertIn('lane is off', single.evidence('', 8)[0])

    def test_one_round_trip_to_the_alias_as_given(self):
        calls = []
        def free(argv, **kwargs):
            calls.append(argv)
            return Done(0, FREE)
        self.assertEqual(single.evidence('srv4', 8, run=free), [])
        self.assertEqual(calls[0][-2:], ['srv4', single.QUERY])
        self.assertIn('BatchMode=yes', calls[0])
        self.assertIn('/proc/meminfo', single.QUERY)
        self.assertIn('st-probe-', single.QUERY)
        # the controller's ~/.ssh/config owns the alias: nothing here forces a user or address
        self.assertEqual(single.target('ost@ost-97x'), 'ost@ost-97x')
        self.assertEqual(single.target('srv4'), 'srv4')
        self.assertFalse(hasattr(single, 'USER'))

    def test_which_hosts_are_the_fleets_own(self):
        for name in ('srv4', 'srv1', 'choiceoh@srv2', 'srv4.tail7fec17.ts.net', '10.10.10.4', '10.10.11.1',
                     '10.10.0.1', 'spark4tb', 'SRV3'):
            self.assertTrue(single.on_fleet(name, {}), name)
        for name in ('ost-97x', 'ost@ost-97x', '100.116.174.65', 'office-topsolar.tail7fec17.ts.net', 'srv5'):
            self.assertFalse(single.on_fleet(name, {}), name)
        self.assertFalse(single.on_fleet('srv4', {'FLEET_SINGLE_GPU_ON_FLEET': '0'}))
        self.assertTrue(single.on_fleet('ost-97x', {'FLEET_SINGLE_GPU_ON_FLEET': '1'}))

    def test_the_cache_remembers_one_answer_per_ttl_host_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = [1000.0]
            answers = iter([Done(0, TIGHT), Done(0, FREE), Done(0, FREE), Done(0, FREE)])
            run = lambda *a, **k: next(answers)
            now = lambda: clock[0]
            first = single.cached_evidence('srv4', directory, 8, now=now, run=run)
            self.assertTrue(first and 'no room' in first[0])
            clock[0] += 5
            self.assertEqual(single.cached_evidence('srv4', directory, 8, now=now, run=run), first)    # remembered
            self.assertEqual(single.cached_evidence('srv4', directory, 4, now=now, run=run), [])       # another budget is asked
            clock[0] += single.TTL_S + 1
            self.assertEqual(single.cached_evidence('srv4', directory, 4, now=now, run=run), [])       # asked again after the TTL
            self.assertEqual(single.cached_evidence('other', directory, 4, now=now, run=run), [])     # another host is asked
            with self.assertRaises(StopIteration):                                                     # ... exactly once each
                single.cached_evidence('third', directory, 4, now=now, run=run)

    def test_main_prints_reasons_and_exits_nonzero_when_there_is_no_room(self):
        out = io.StringIO()
        with patch.object(single, 'evidence', return_value=['srv4: no room beside production -- x']), \
                contextlib.redirect_stdout(out):
            self.assertEqual(single.main(['evidence', '--host', 'srv4', '--gib', '8']), 1)
        self.assertIn('no room', out.getvalue())
        with patch.object(single, 'evidence', return_value=[]):
            self.assertEqual(single.main(['evidence', '--host', 'srv4']), 0)
        with patch.dict(os.environ, {'FLEET_SINGLE_GPU_HOST': ''}):
            self.assertEqual(single.label(), 'off')
        with patch.dict(os.environ, {'FLEET_SINGLE_GPU_HOST': 'srv4', 'FLEET_SINGLE_GPU_NAME': 'GB10'}):
            self.assertEqual(single.label(), 'GB10 on srv4 beside production')
            self.assertEqual(single.main(['on-fleet']), 0)
        with patch.dict(os.environ, {'FLEET_SINGLE_GPU_HOST': 'ost-97x', 'FLEET_SINGLE_GPU_NAME': '5050'}):
            self.assertEqual(single.label(), '5050 on ost-97x')
            self.assertEqual(single.main(['on-fleet']), 1)


class ContractTests(unittest.TestCase):
    """The queue, its helpers, the runner, the launcher and the --test boot agree on what the lane is."""

    def setUp(self):
        self.fleet = (ROOT / 'bench/fleet.sh').read_text()
        self.runner = (ROOT / 'probes/run_engine_probe.sh').read_text()
        self.launcher = (ROOT / 'launchers/start-st-glm53.sh').read_text()

    def test_defaults_and_the_floor_agree_across_the_shell_the_module_and_the_test_boot(self):
        self.assertIn('FLEET_SINGLE_GPU_HOST=${FLEET_SINGLE_GPU_HOST-' + single.DEFAULT_HOST + '}', self.fleet)
        self.assertIn('FLEET_SINGLE_GPU_NAME=${FLEET_SINGLE_GPU_NAME:-' + single.DEFAULT_GPU + '}', self.fleet)
        self.assertIn('export FLEET_SINGLE_GPU_HOST FLEET_SINGLE_GPU_NAME', self.fleet)
        self.assertEqual((single.DEFAULT_HOST, single.DEFAULT_GPU), ('srv4', 'GB10'))
        boot = (ROOT / 'engine/profiles/glm53/boot.py').read_text()
        floor = re.search(r'^TEST_FLOOR_GIB = ([0-9.]+)', boot, re.M)
        self.assertIsNotNone(floor, 'the --test boot names the floor this lane keeps')
        self.assertEqual(float(floor.group(1)), single.FLOOR_GIB)
        self.assertGreater(single.FLOOR_GIB, 0)
        self.assertGreater(single.DEFAULT_BUDGET_GIB, 0)

    @unittest.skipUnless(BASH, 'bash is required')
    def test_the_shell_and_the_module_agree_on_which_hosts_are_the_fleets(self):
        line = next(l for l in self.fleet.splitlines() if l.startswith('single_on_fleet() {'))
        for name in ('srv4', 'choiceoh@srv1', 'srv4.tail7fec17.ts.net', '10.10.10.4', 'spark4tb',
                     'ost-97x', 'ost@ost-97x', '100.116.174.65', 'office-topsolar.tail7fec17.ts.net'):
            rc = subprocess.run([BASH, '-c', line + '\nsingle_on_fleet'], env=dict(os.environ, FLEET_SINGLE_GPU_HOST=name),
                                capture_output=True).returncode
            self.assertEqual(rc == 0, single.on_fleet(name, {}), name)
        rc = subprocess.run([BASH, '-c', line + '\nsingle_on_fleet'],
                            env=dict(os.environ, FLEET_SINGLE_GPU_HOST='srv4', FLEET_SINGLE_GPU_ON_FLEET='0'),
                            capture_output=True).returncode
        self.assertEqual(rc, 1)

    def test_the_lane_has_its_own_holder_and_the_lanes_share_no_fleet_box(self):
        self.assertIn('HS=$FLEET_DIR/holder-single', self.fleet)
        self.assertEqual(handoff.holder_path('/f', 'single'), Path('/f/holder-single'))
        for kind in ('boot', 'probe'):
            self.assertEqual(handoff.holder_path('/f', kind), Path('/f/holder'))
        # the ST launcher reads the fleet LEASE (the queue's one record, PR #770) and no holder
        # file at all: a check beside production must never look like the fleet being held to it
        self.assertIn('lease verify --owner "$LEASE_OWNER"', self.launcher)
        self.assertNotIn('FLEET_HOLDER', self.launcher)
        self.assertNotIn('holder-single', self.launcher)
        # each lane takes its own head of the ranked order
        self.assertIn('[ "$(lane_front "$kind")" = "$s" ] || return 1', self.fleet)
        # a fleet BOOT and a single check never share a fleet box; beside serving they run at once
        self.assertIn('if [ "$kind" = boot ] && single_on_fleet && [ -s "$HS" ] && holder_alive "$HS"; then return 1; fi', self.fleet)
        self.assertIn('holds this box too', self.fleet)
        self.assertIn('single_refused "$s" "$why"; return 1', self.fleet)
        self.assertIn('FLEET_RULES=4', self.fleet)

    def test_the_runner_waits_for_room_uses_the_production_image_and_takes_no_lease_there(self):
        self.assertIn('probe_host=${ST_PROBE_HOST:-}', self.runner)
        remote = self.runner[self.runner.index('probe_host=${ST_PROBE_HOST:-}'):self.runner.index('mkdir -p "$cache"')]
        self.assertIn('rsync -a --delete --exclude __pycache__ -e "ssh $SSHOPT" "$repo/engine" "$repo/probes"', remote)
        self.assertIn('fleet_single.py" evidence --host "$probe_host" --gib "${ST_PROBE_GIB:-8}"', remote)
        self.assertIn('waiting for room on $probe_host', remote)
        self.assertIn("docker inspect st-glm53 --format '{{.Config.Image}}'", remote)
        self.assertIn('''trap 'ssh $SSHOPT "$probe_host" "docker rm -f $NAME"''', remote)
        self.assertIn('exit $rc', remote)
        self.assertNotIn('fleet_lease', remote)
        self.assertNotIn('choiceoh@$probe_host', remote)          # the alias as given, never the fleet's user
        # the fleet path verifies the ticket's lease (the queue took it at GO, PR #770) and waits
        # for it; it takes none of its own
        local = self.runner[self.runner.index('mkdir -p "$cache"'):]
        for text in ('fleet_lease verify --owner "$ST_LEASE_OWNER"', 'ST_PROBE_WAIT_MINUTES', 'ST_PROBE_NO_LEASE'):
            self.assertIn(text, local)
        self.assertNotIn('fleet_lease acquire', local)
        boot = (ROOT / 'bench/fleet_boot.py').read_text()
        self.assertIn("environment['ST_PROBE_HOST'] = host", boot)

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
    """The queue's admission, released from its dispatcher: two lanes, two holders, and the one
    rule about sharing a box."""

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
        self.answer = self.root / 'ssh-answer'     # first line: exit code; the rest: what the box answers
        self.answer.write_text('0\n' + FREE)
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

    def test_on_a_box_of_its_own_the_lanes_never_wait_for_each_other(self):
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
printf '0\\nMemAvailable:   20971520 kB\\n---\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR/.single-gpu-evidence"
with_lock _try_hold D $$ 5 "another check" single; echo "D2=$?"
with_lock _try_hold D $$ 5 "another check" single; echo "D3=$?"
single_line
printf '255\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR/.single-gpu-evidence"
with_lock _try_hold D $$ 5 "another check" single; echo "D4=$?"
printf '0\\nMemAvailable:   41943040 kB\\n---\\nst-probe-elsewhere-1\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR/.single-gpu-evidence"
with_lock _try_hold D $$ 5 "another check" single; echo "D5=$?"
printf '0\\nMemAvailable:   41943040 kB\\n---\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR/.single-gpu-evidence"
with_lock _try_hold D $$ 5 "another check" single; echo "D6=$?"
echo "held: fleet=$(cut -d'|' -f1 "$H") single=$(cut -d'|' -f1 "$HS")"
with_lock _release A; echo "releaseA=$?"
with_lock _try_hold C $$ 30 "another boot" boot; echo "C2=$?"
with_lock _kick --force single; echo "kick=$?"
echo "held: fleet=$(cut -d'|' -f1 "$H") single=$(cat "$HS" 2>/dev/null | cut -d'|' -f1)"
echo "queue=$(grep -c . "$Q")"
''', FLEET_SINGLE_GPU_HOST='ost-97x', FLEET_SINGLE_GPU_NAME='5050')
        out = result.stdout
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('A=0', 'B=0', 'C=1', 'D=1', 'held: fleet=A single=B',
                         'single (5050 on ost-97x): HELD by B [single]', 'releaseB=0',
                         'D2=1', 'D3=1',
                         "single (5050 on ost-97x): ost-97x: no room beside production -- MemAvailable 20.0 GiB, "
                         "this check's budget 8.0 GiB, floor 16.0: 12.0 GiB would be left",
                         'D4=1', 'D5=1', 'D6=0', 'held: fleet=A single=D', 'releaseA=0', 'C2=0', 'kicked', 'kick=0',
                         'held: fleet=C single=', 'queue=0'):
            self.assertIn(expected, out, out + result.stderr)
        self.assertNotIn('holds this box too', out)
        log = (self.fleet / 'log').read_text()
        self.assertIn('GO B (pid', log)
        self.assertIn('[single: 5050 on ost-97x]', log)
        self.assertIn('release B [single]', log)
        self.assertEqual(log.count('hold refused (single)'), 3, log)   # no room, unreachable, a probe there: once each
        self.assertIn('no room beside production', log)
        self.assertIn('cannot read its memory (rc 255)', log)
        self.assertIn('a probe is already running there (st-probe-elsewhere-1)', log)
        ledger = [line.split('\t') for line in (self.fleet / 'ledger.tsv').read_text().splitlines()]
        self.assertEqual([(row[1], row[2], row[5].strip()) for row in ledger],   # BSD wc pads its count
                         [('B', 'single', '0'), ('A', 'boot', '0')])

    def test_on_a_fleet_box_a_boot_and_a_check_never_share_it(self):
        now, pid = int(time.time()), os.getpid()
        self.queue((1, 'A', now, 30, 'a boot', 'boot', pid), (2, 'B', now, 5, 'a check', 'single', pid))
        result = self.run_fleet('''
with_lock _try_hold A $$ 30 "a boot" boot; echo "A=$?"
with_lock _try_hold B $$ 5 "a check" single; echo "B=$?"
with_lock _try_hold B $$ 5 "a check" single; echo "B2=$?"
single_line
with_lock _release A; echo "releaseA=$?"
with_lock _try_hold B $$ 5 "a check" single; echo "B3=$?"
single_line
printf '%s|%s|%s|%s|%s|%s|%s\\n' 3 C "$(date +%s)" 30 "another boot" boot $$ >> "$Q"
with_lock _try_hold C $$ 30 "another boot" boot; echo "C=$?"
with_lock _release B; echo "releaseB=$?"
with_lock _try_hold C $$ 30 "another boot" boot; echo "C2=$?"
echo "held: fleet=$(cut -d'|' -f1 "$H") single=$(cat "$HS" 2>/dev/null | cut -d'|' -f1)"
''')
        out = result.stdout
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('A=0', 'B=1', 'B2=1',
                         'single (GB10 on srv4 beside production): FREE -- but the fleet boot A holds this box too',
                         'releaseA=0', 'B3=0', 'single (GB10 on srv4 beside production): HELD by B [single]',
                         'C=1', 'releaseB=0', 'C2=0', 'held: fleet=C single='):
            self.assertIn(expected, out, out + result.stderr)
        log = (self.fleet / 'log').read_text()
        self.assertEqual(log.count('hold refused (single)'), 1, log)
        self.assertIn('the fleet boot A holds this box too; B waits', log)

    def test_the_lane_off_refuses_rather_than_answering_room(self):
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
        lane = 'needs one GPU, not four: single-GPU lane (GB10 on srv4 beside production)'
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
