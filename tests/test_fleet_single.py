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

    def setUp(self):
        patcher = patch.dict(os.environ, {'FLEET_SINGLE_LOCAL_HOST': '-'})   # no host here is this machine (srv4 runs tests too)
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_the_floor_is_a_sparks_until_a_box_of_its_own_says_otherwise(self):
        """A discrete-card box does not owe production a GB10's 16 GiB of one shared pool."""
        # ost-97x on 2026-09-15: 13.6 GiB available, so the Spark's floor refuses even a zero budget --
        # which is what a box with no HOSTS entry still gets
        OST = 'MemAvailable:   14305396 kB\n---\n'
        self.assertIn('no room', single.evidence('another-box', 0, run=answering(OST))[0])
        self.assertEqual(single.floor_gib(), single.FLOOR_GIB)
        # ost-97x says what it owes itself (HOSTS), with or without a user in the alias
        self.assertEqual(single.floor_gib(name='ost-97x'), 4.0)
        self.assertEqual(single.floor_gib(name='choiceoh@ost-97x'), 4.0)
        self.assertEqual(single.evidence('ost-97x', 0, run=answering(OST)), [])
        with patch.dict(os.environ, {single.FLOOR_ENV: '4'}):
            self.assertEqual(single.floor_gib(), 4.0)
            self.assertEqual(single.evidence('another-box', 4, run=answering(OST)), [])
            # the floor is what is left over, not a licence: a budget that eats past it still refuses
            self.assertIn('floor 4.0', single.evidence('another-box', 12, run=answering(OST))[0])
        with patch.dict(os.environ, {single.FLOOR_ENV: '16'}):                 # the variable beats the box's entry
            self.assertEqual(single.floor_gib(name='ost-97x'), 16.0)
            self.assertIn('floor 16.0', single.evidence('ost-97x', 4, run=answering(OST))[0])
        # an explicit floor beats the environment, and nonsense falls back to the Spark's -- or the box's own
        self.assertEqual(single.evidence('another-box', 4, run=answering(OST), floor=4.0), [])
        for bad in ('', 'lots', '-1'):
            with patch.dict(os.environ, {single.FLOOR_ENV: bad}):
                self.assertEqual(single.floor_gib(), single.FLOOR_GIB)
                self.assertEqual(single.floor_gib(name='ost-97x'), 4.0)

    def test_a_changed_floor_does_not_read_the_old_answer_back(self):
        """The cache key carries the floor: the same host and budget can flip on the floor alone."""
        with tempfile.TemporaryDirectory() as directory:
            OST = 'MemAvailable:   14305396 kB\n---\n'
            with patch.dict(os.environ, {single.FLOOR_ENV: '16'}):
                self.assertIn('no room', single.cached_evidence('ost-97x', directory, 4, run=answering(OST))[0])
            with patch.dict(os.environ, {single.FLOOR_ENV: '4'}):
                self.assertEqual(single.cached_evidence('ost-97x', directory, 4, run=answering(OST)), [])

    def test_unreachable_or_unreadable_is_not_room(self):
        def unreachable(*args, **kwargs):
            raise subprocess.TimeoutExpired('ssh', 8)
        self.assertIn('unreachable', single.evidence('srv4', 8, run=unreachable)[0])
        failed = single.evidence('srv4', 8, run=answering('', 255))
        self.assertIn('cannot say it has room', failed[0])
        self.assertIn('rc 255', failed[0])
        self.assertIn('cannot read MemAvailable', single.evidence('srv4', 8, run=answering('---\n'))[0])
        self.assertIn('lane is off', single.evidence('', 8)[0])

    def test_reclaim_makes_the_budget_immediately_free_or_says_why_not(self):
        """MemAvailable is not what a device allocation can take on a UMA box: the lane's first real
        ticket OOMed on its first tensor with 26 GiB 'available'. The runner faults the budget on
        the box and gives it back (the arena's touch_pages), and waits if the box is still faulting."""
        calls = []
        def answer(text, rc=0):
            def run(argv, **kwargs):
                calls.append(argv)
                return Done(rc, text)
            return run
        self.assertEqual(single.reclaim('srv4', 8, run=answer('free 14.9\n')), [])
        self.assertEqual(calls[0][-2], 'srv4')
        self.assertIn('base64', calls[0][-1])
        self.assertTrue(calls[0][-1].endswith(' 8.0 16.0'))            # the budget and the floor, as arguments
        self.assertEqual(single.reclaim('srv4', 8, run=answer('reclaimed 9.3\n')), [])
        short = single.reclaim('srv4', 8, run=answer('short 2.1\n', 3))
        self.assertIn('only 2.1 GiB immediately free after reclaiming', short[0])
        self.assertIn('still faulting', short[0])
        none = single.reclaim('srv4', 8, run=answer('no-room 20.0\n', 2))
        self.assertIn('no room beside production', none[0])
        def unreachable(*args, **kwargs):
            raise subprocess.TimeoutExpired('ssh', 8)
        self.assertIn('unreachable', single.reclaim('srv4', 8, run=unreachable)[0])
        self.assertIn('could not reclaim room (rc 255', single.reclaim('srv4', 8, run=answer('', 255))[0])
        # the code that runs on the box is the arena's recipe: populate, then release
        for text in ('MAP_POPULATE', 'MemAvailable', 'MemFree', 'region.close()', "'no-room'", "'short'"):
            self.assertIn(text, single.RECLAIM)

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
        patcher = patch.dict(os.environ, {'FLEET_SINGLE_LOCAL_HOST': '-'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults_and_the_floor_agree_across_the_shell_the_module_and_the_test_boot(self):
        self.assertIn('FLEET_SINGLE_GPU_HOSTS=${FLEET_SINGLE_GPU_HOSTS-${FLEET_SINGLE_GPU_HOST-'
                      + ' '.join(single.DEFAULT_HOSTS) + '}}', self.fleet)
        self.assertIn('FLEET_SINGLE_GPU_HOST=${FLEET_SINGLE_GPU_HOSTS%% *}', self.fleet)
        self.assertIn('FLEET_SINGLE_GPU_NAME=${FLEET_SINGLE_GPU_NAME:-' + single.DEFAULT_GPU + '}', self.fleet)
        self.assertIn('export FLEET_SINGLE_GPU_HOSTS FLEET_SINGLE_GPU_HOST FLEET_SINGLE_GPU_NAME', self.fleet)
        self.assertEqual((single.DEFAULT_HOST, single.DEFAULT_GPU), ('srv4', 'GB10'))
        self.assertEqual(single.DEFAULT_HOSTS, ('srv4', 'srv3', 'srv1', 'srv2'))       # the lane's own host first
        self.assertEqual(single.DEFAULT_HOSTS[0], single.DEFAULT_HOST)
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
        # -- and with the lane a pool of the Sparks, a boot waits for every single check on one of them
        self.assertIn('if [ "$kind" = boot ] && single_on_fleet_held; then return 1; fi', self.fleet)
        self.assertIn('holds this box too', self.fleet)
        self.assertIn('single_refused "$s" "$why" "$kind"; return 1', self.fleet)
        self.assertIn('if [ "$rc" != 0 ]; then single_refused "$s" "$why" single; return 1; fi', self.fleet)
        self.assertIn('FLEET_RULES=8', self.fleet)
        self.assertIn("#   8  a card both one-GPU lanes name takes one check at a time", self.fleet)
        # a single check's results come back to the controller from the pool host it ran on -- off the queue lock
        self.assertIn('logit "release $1 [single] on $sh"; _collect_single "$1" "${t0:-0}" single "$sh"; return 0', self.fleet)
        self.assertIn('fi ) 9>&- >/dev/null 2>&1 &', self.fleet)
        # no estimate given: the ledger's history; no budget given: the probe's own
        self.assertIn('if [ -z "$est" ]; then est=$(expected_min "$s" 30)', self.fleet)
        self.assertIn('export ST_PROBE_GIB=$budget', self.fleet)

    def test_the_check_lane_is_a_box_of_its_own_with_its_own_holder_and_facts(self):
        """The second one-GPU lane (operator, 2026-09-19): the RTX 5050 on ost-97x, beside the single lane and the
        fleet, with what that box owes itself and runs as facts rather than knobs."""
        self.assertIn('FLEET_CHECK_GPU_HOST=${FLEET_CHECK_GPU_HOST-' + single.CHECK_DEFAULT_HOST + '}', self.fleet)
        self.assertIn('FLEET_CHECK_GPU_NAME=${FLEET_CHECK_GPU_NAME:-' + single.CHECK_DEFAULT_GPU + '}', self.fleet)
        self.assertIn('export FLEET_CHECK_GPU_HOST FLEET_CHECK_GPU_NAME', self.fleet)
        self.assertEqual((single.CHECK_DEFAULT_HOST, single.CHECK_DEFAULT_GPU), ('ost-97x', 'RTX5050'))
        self.assertEqual(single.host({}, lane='check'), 'ost-97x')
        self.assertEqual(single.host({'FLEET_CHECK_GPU_HOST': ''}, lane='check'), '')         # empty turns it off
        self.assertEqual(single.host({}), 'srv4')                                             # the single lane's, as before
        self.assertEqual(single.label({}, lane='check'), 'RTX5050 on ost-97x')
        self.assertFalse(single.on_fleet(single.CHECK_DEFAULT_HOST, {}))                     # a box of its own
        self.assertIn('HC=$FLEET_DIR/holder-check', self.fleet)
        self.assertEqual(handoff.holder_path('/f', 'check'), Path('/f/holder-check'))
        self.assertEqual((handoff.lane('check'), handoff.lane('single'), handoff.lane('probe')), ('check', 'single', 'fleet'))
        # what that box owes itself and runs (bench/OST_97X_LANE.md): facts, not exports
        box = single.HOSTS['ost-97x']
        self.assertEqual((box['floor_gib'], box['budget_gib'], box['image']), (4.0, 4.0, 'st-engine:glm53-sm120-x86'))
        self.assertEqual(single.image('ost-97x'), 'st-engine:glm53-sm120-x86')
        self.assertEqual(single.image('srv4'), '')                                           # a Spark runs production's
        self.assertEqual(single.budget_gib({}, name='ost-97x'), 4.0)
        self.assertEqual(single.budget_gib({'ST_PROBE_GIB': '6'}, name='ost-97x'), 6.0)       # the submitter's word wins
        self.assertEqual(single.budget_gib({}, name='srv4'), single.DEFAULT_BUDGET_GIB)
        # one evidence cache a host: two lanes asking about two boxes do not overwrite each other every poll
        self.assertEqual(single.cache_name('srv4'), single.CACHE)
        self.assertEqual(single.cache_name('ost-97x'), single.CACHE + '.ost-97x')
        # the supervisor gives each one-GPU lane its own host, never the command
        boot = (ROOT / 'bench/fleet_boot.py').read_text()
        self.assertIn('host = fleet_single.host(self.env, lane=kind)', boot)
        self.assertIn('FLEET_CHECK_GPU_HOST is empty', boot)
        # the runner: a box of its own runs its check image, with the Sparks' flashinfer over its site-packages
        remote = self.runner[self.runner.index('probe_host=${ST_PROBE_HOST:-}'):self.runner.index('mkdir -p "$cache"')]
        self.assertIn('own_image=$(python3 "$repo/bench/fleet_single.py" image --host "$probe_host")', remote)
        self.assertIn('mounts+=(--mount "type=bind,src=$home/$vendored/$entry,dst=$site/$entry,readonly")', remote)
        # the lane's own admission: one GPU, and no more than a kernel check's budget of a discrete 8 GiB card
        self.assertIn('check', policy.KINDS)
        self.assertEqual(policy.CHECK_MAX_BUDGET_GIB, 8.0)
        self.assertIn('--check) lane_force=check; shift;;', self.fleet)
        self.assertIn('logit "release $1 [check]"; _collect_single "$1" "${t0:-0}" check; return 0', self.fleet)
        self.assertIn('6  a second one-GPU lane', self.fleet)

    def test_the_module_answers_for_either_lane(self):
        out = io.StringIO()
        with patch.dict(os.environ, {'FLEET_CHECK_GPU_HOST': 'ost-97x'}), contextlib.redirect_stdout(out):
            for name in (single.BUDGET_ENV, single.FLOOR_ENV):
                os.environ.pop(name, None)
            self.assertEqual(single.main(['host', '--lane', 'check']), 0)
            self.assertEqual(single.main(['image', '--lane', 'check']), 0)
            self.assertEqual(single.main(['vendored', '--host', 'ost-97x']), 0)
            self.assertEqual(single.main(['vendored', '--host', 'srv4']), 0)
            self.assertEqual(single.main(['budget', '--host', 'ost-97x']), 0)
            self.assertEqual(single.main(['floor', '--host', 'ost-97x']), 0)
        self.assertEqual(out.getvalue().splitlines(),
                         ['ost-97x', 'st-engine:glm53-sm120-x86',
                          'st-x86-flashinfer/vendored /usr/local/lib/python3.12/site-packages', '', '4.0', '4.0'])

    def test_collect_copies_only_the_cache_files_find_listed(self):
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            if argv[0] == 'ssh':
                return Done(0, '.cache/st/decode-timeline-rows1.json\n.cache/st/prefill-chunk-profile.json\n'
                               'MemAvailable:   41943040 kB\n../../etc/passwd\n.cache/st/sub/dir.json\n')
            return Done(0 if 'decode-timeline' in argv[-2] else 1)
        with tempfile.TemporaryDirectory() as directory:
            copied = single.collect('srv4', 1789000000.7, directory, run=run)
        self.assertEqual(copied, ['decode-timeline-rows1.json'])            # the second scp failed: not claimed
        listing = calls[0]
        self.assertEqual(listing[:len(single.SSH)], list(single.SSH))
        self.assertIn('-newermt @1789000000', listing[-1])
        self.assertIn('-maxdepth 1', listing[-1])
        scps = [c for c in calls[1:]]
        self.assertEqual([c[-2] for c in scps], ['srv4:.cache/st/decode-timeline-rows1.json', 'srv4:.cache/st/prefill-chunk-profile.json'])
        # a host that cannot list says so; it never reads as "nothing to collect"
        with self.assertRaisesRegex(OSError, 'could not list'):
            single.collect('srv4', 0, '/tmp/unused', run=lambda argv, **kw: Done(255, '', 'ssh: connect refused'))

    def test_the_runner_waits_for_room_uses_the_production_image_and_takes_no_lease_there(self):
        self.assertIn('probe_host=${ST_PROBE_HOST:-}', self.runner)
        remote = self.runner[self.runner.index('probe_host=${ST_PROBE_HOST:-}'):self.runner.index('mkdir -p "$cache"')]
        self.assertIn('rsync -a --delete --exclude __pycache__ -e "ssh $SSHOPT" "$repo/engine" "$repo/probes" "$repo/tests"', remote)
        self.assertIn('budget=${ST_PROBE_GIB:-$(python3 "$repo/bench/fleet_single.py" budget --host "$probe_host")}', remote)
        self.assertIn('fleet_single.py" evidence --host "$probe_host" --gib "$budget"', remote)
        self.assertIn('fleet_single.py" reclaim --host "$probe_host" --gib "$budget"', remote)
        self.assertIn('waiting for room on $probe_host', remote)
        self.assertIn("docker inspect st-glm53 --format '{{.Config.Image}}'", remote)
        self.assertIn('''trap 'at "docker rm -f $NAME"''', remote)
        self.assertIn('rc=0; at "$remote" || rc=$?', remote)
        self.assertIn('exit $rc', remote)
        # a lane's run takes this path on the controller's own Spark too, with no ssh to itself
        self.assertIn('if [ -n "$probe_host" ] && { [ "$here" = 0 ] || [ -n "${ST_PROBE_LANE:-}" ]; }; then', self.runner)
        self.assertIn('at() { if [ "$here" = 1 ]; then bash -c "cd && $1"; else ssh $SSHOPT "$probe_host" "$1"; fi; }', remote)
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

    def test_a_box_of_its_own_runs_its_own_image_and_needs_its_flashinfer(self):
        """ost-97x's check image comes before any production tag found there (an ARM64 copy would not run), a box
        whose facts name a vendored flashinfer it does not keep refuses before any container, and the setup script's
        controller alias is the one the queue and HOSTS know, whatever the tailnet name (review of #1297)."""
        remote = self.runner[self.runner.index('probe_host=${ST_PROBE_HOST:-}'):self.runner.index('mkdir -p "$cache"')]
        self.assertLess(remote.index('image=$own_image'), remote.index("docker inspect st-glm53 --format '{{.Config.Image}}'"))
        self.assertIn('[ -n "$image" ] || image=$(at "docker inspect st-glm53', remote)
        self.assertIn('if [ -n "${vendored:-}" ] && [ "$image" = "$own_image" ]; then', remote)
        self.assertLess(remote.index('keeps no vendored flashinfer at ~/$vendored'), remote.index('docker run --rm --name'))
        setup = (ROOT / 'tools/ost-97x-lane-setup.sh').read_text()
        self.assertIn('ALIAS=ost-97x', setup)
        self.assertIn('    Host $ALIAS\n', setup)
        self.assertNotIn('Host $NODE', setup)
        self.assertEqual(single.CHECK_DEFAULT_HOST, 'ost-97x')
        self.assertIn('ost-97x', single.HOSTS)

    def test_the_policy_counts_gpus(self):
        self.assertEqual(policy.gpus_needed('probes/run_engine_check.sh', ['--layers', '0-4']), 1)
        self.assertEqual(policy.gpus_needed('probes/run_engine_probe.sh', ['probes/engine_kernel_check.py']), 1)
        self.assertEqual(policy.gpus_needed('probes/run_engine_probe.sh',
                                            ['engine/profiles/glm53/check.py', '--distributed']), 4)
        for entry in ('bench/onepass.py', 'bench/st_bracket.sh'):
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
answer = pathlib.Path(os.environ['SSH_ANSWER'])            # SSH_ANSWER.<host>, when there is one, answers for that host
own = answer.with_name(answer.name + '.' + sys.argv[-2]) if len(sys.argv) > 2 else answer
lines = (own if own.exists() else answer).read_text().splitlines()
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
                                SSH_ANSWER=str(self.answer), LANG='C', LC_ALL='C',
                                FLEET_SINGLE_LOCAL_HOST='-')      # no pool host is this machine: every answer is the shim's

    def run_fleet(self, script, **environment):
        result = subprocess.run([BASH, '-c', self.library + '\n' + script], cwd=self.repo,
                                env=dict(self.environment, **environment), text=True, capture_output=True, timeout=120)
        self.assertNotIn('command not found', result.stderr, result.stderr)
        return result

    def queue(self, *rows):
        (self.fleet / 'queue').write_text(''.join('|'.join(map(str, row)) + '\n' for row in rows))

    def test_the_expected_minutes_are_the_ledger_s_by_session_then_family_as_the_ranking_reads_them(self):
        rows = [('kern-timeline', 9), ('kern-timeline', 11), ('c4-rows-decode-profile', 30), ('c4-rows-decode-profile', 2),
                ('c4-rows-decode-profile', 3), ('c4-rows-decode-profile', 2), ('c4-rows-decode-profile', 2),
                ('c4-rows-decode-profile', 2), ('zero', 0)]
        (self.fleet / 'ledger.tsv').write_text(''.join(f'2026-09-13_12:00:00\t{s}\tsingle\tnote\t{m}\t0\t0\t0\n' for s, m in rows))
        result = self.run_fleet('for s in kern-timeline c4-rows2-decode-profile zero new; do echo "$s=$(expected_min "$s" 30)"; done')
        self.assertEqual(result.stdout.split(), ['kern-timeline=9', 'c4-rows2-decode-profile=2', 'zero=30', 'new=30'])
        sys.path.insert(0, str(ROOT / 'bench'))
        import fleet_priority
        python = fleet_priority.history_estimates(self.fleet / 'ledger.tsv', ['kern-timeline', 'c4-rows2-decode-profile'])
        self.assertEqual({s: int(v['minutes']) for s, v in python.items()}, {'kern-timeline': 9, 'c4-rows2-decode-profile': 2})

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
printf '0\\nMemAvailable:   6291456 kB\\n---\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR"/.single-gpu-evidence*
with_lock _try_hold D $$ 5 "another check" single; echo "D2=$?"
with_lock _try_hold D $$ 5 "another check" single; echo "D3=$?"
single_line
printf '255\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR"/.single-gpu-evidence*
with_lock _try_hold D $$ 5 "another check" single; echo "D4=$?"
printf '0\\nMemAvailable:   41943040 kB\\n---\\nst-probe-elsewhere-1\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR"/.single-gpu-evidence*
with_lock _try_hold D $$ 5 "another check" single; echo "D5=$?"
printf '0\\nMemAvailable:   41943040 kB\\n---\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR"/.single-gpu-evidence*
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
                         # that box's own floor and budget (fleet_single.HOSTS), not a Spark's
                         "single (5050 on ost-97x): ost-97x: no room beside production -- MemAvailable 6.0 GiB, "
                         "this check's budget 4.0 GiB, floor 4.0: 2.0 GiB would be left",
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
''', FLEET_SINGLE_GPU_HOST='srv4')                  # a pool of one fleet box: the rule on its own
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

    def test_the_check_lane_waits_for_neither_the_single_lane_nor_the_fleet(self):
        """Three lanes, three holders: a check on the 5050 is held beside a fleet boot, and beside a single check
        once the box rule lets that one in; its own next ticket waits only for it."""
        now, pid = int(time.time()), os.getpid()
        self.queue((1, 'A', now, 30, 'a boot', 'boot', pid), (2, 'B', now, 5, 'a check', 'single', pid),
                   (3, 'E', now, 5, 'a 5050 check', 'check', pid), (4, 'F', now, 5, 'another 5050 check', 'check', pid))
        result = self.run_fleet('''
with_lock _try_hold A $$ 30 "a boot" boot; echo "A=$?"
with_lock _try_hold E $$ 5 "a 5050 check" check; echo "E=$?"
with_lock _try_hold B $$ 5 "a check" single; echo "B=$?"
echo "held1: fleet=$(cut -d'|' -f1 "$H") single=$(cat "$HS" 2>/dev/null | cut -d'|' -f1) check=$(cut -d'|' -f1 "$HC")"
with_lock _release A; echo "releaseA=$?"
with_lock _try_hold B $$ 5 "a check" single; echo "B2=$?"
echo "held2: fleet=$(cat "$H" 2>/dev/null | cut -d'|' -f1) single=$(cut -d'|' -f1 "$HS") check=$(cut -d'|' -f1 "$HC")"
with_lock _try_hold F $$ 5 "another 5050 check" check; echo "F=$?"
check_line
echo "of E: $(basename "$(holder_file_of E)")"
with_lock _release E; echo "releaseE=$?"
printf '0\\nMemAvailable:   6291456 kB\\n---\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR"/.single-gpu-evidence*
with_lock _try_hold F $$ 5 "another 5050 check" check; echo "F2=$?"
check_line
printf '0\\nMemAvailable:   41943040 kB\\n---\\n' > "$SSH_ANSWER"; rm -f "$FLEET_DIR"/.single-gpu-evidence*
with_lock _try_hold F $$ 5 "another 5050 check" check; echo "F3=$?"
with_lock _kick --force check; echo "kick=$?"
echo "held3: single=$(cut -d'|' -f1 "$HS") check=$(cat "$HC" 2>/dev/null | cut -d'|' -f1)"
echo "queue=$(grep -c . "$Q")"
''')
        out = result.stdout
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('A=0', 'E=0', 'B=1', 'held1: fleet=A single= check=E',       # srv4 is a fleet box: the box rule
                         'releaseA=0', 'B2=0', 'held2: fleet= single=B check=E',
                         'F=1', 'check (RTX5050 on ost-97x, checks not numbers): HELD by E [check]',
                         'of E: holder-check', 'releaseE=0', 'F2=1',
                         "check (RTX5050 on ost-97x, checks not numbers): ost-97x: no room beside production -- "
                         "MemAvailable 6.0 GiB, this check's budget 4.0 GiB, floor 4.0: 2.0 GiB would be left",
                         'F3=0', 'kicked', 'kick=0', 'held3: single=B check=', 'queue=0'):
            self.assertIn(expected, out, out + result.stderr)
        log = (self.fleet / 'log').read_text()
        self.assertIn('GO E (pid', log)
        self.assertIn('[check: RTX5050 on ost-97x]', log)
        self.assertIn('release E [check]', log)
        self.assertEqual(log.count('hold refused (check)'), 1, log)
        self.assertIn('kick --force [check]', log)
        ledger = [line.split('\t') for line in (self.fleet / 'ledger.tsv').read_text().splitlines()]
        self.assertEqual([(row[1], row[2], row[5].strip()) for row in ledger], [('A', 'boot', '0'), ('E', 'check', '0')])

    def test_a_card_both_lanes_name_takes_one_check_at_a_time(self):
        """FLEET_SINGLE_GPU_HOST=ost-97x left from before the check lane: both lanes name the 5050 with a holder each,
        so each lane's live holder refuses the other -- one card, one check (review of #1297)."""
        now, pid = int(time.time()), os.getpid()
        self.queue((1, 'E', now, 5, 'a 5050 check', 'check', pid), (2, 'B', now, 5, 'a check', 'single', pid),
                   (3, 'F', now, 5, 'another 5050 check', 'check', pid))
        result = self.run_fleet('''
with_lock _try_hold E $$ 5 "a 5050 check" check; echo "E=$?"
with_lock _try_hold B $$ 5 "a check" single; echo "B=$?"
single_line
with_lock _release E; echo "releaseE=$?"
with_lock _try_hold B $$ 5 "a check" single; echo "B2=$?"
with_lock _try_hold F $$ 5 "another 5050 check" check; echo "F=$?"
check_line
with_lock _release B; echo "releaseB=$?"
with_lock _try_hold F $$ 5 "another 5050 check" check; echo "F2=$?"
''', FLEET_SINGLE_GPU_HOST='ost-97x')
        out = result.stdout
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('E=0', 'B=1', "single (GB10 on ost-97x): ost-97x: the check lane's E holds this card",
                         'releaseE=0', 'B2=0', 'F=1',
                         "check (RTX5050 on ost-97x, checks not numbers): ost-97x: the single lane's B holds this card",
                         'releaseB=0', 'F2=0'):
            self.assertIn(expected, out, out + result.stderr)
        log = (self.fleet / 'log').read_text()
        self.assertIn("hold refused (single): ost-97x: the check lane's E holds this card; B waits", log)
        self.assertIn("hold refused (check): ost-97x: the single lane's B holds this card; F waits", log)

    def test_four_checks_take_the_four_sparks_and_a_fifth_waits(self):
        """The single lane is a pool (operator, 2026-09-19): each check takes the first Spark with no live holder and
        room, up to four at once; a fifth waits; a Spark with no room is passed over; a boot waits for all of them;
        a check's host is the holder that names it, and it is released and kicked there."""
        now, pid = int(time.time()), os.getpid()
        self.queue(*[(i, f'S{i}', now, 5, f'check {i}', 'single', pid) for i in range(1, 6)],
                   (6, 'A', now, 30, 'a boot', 'boot', pid))
        (self.root / 'ssh-answer.srv3').write_text('0\n' + TIGHT)                 # srv3 has no room, for now
        result = self.run_fleet('''
for s in S1 S2 S3 S4; do with_lock _try_hold $s $$ 5 "check" single; echo "$s=$?"; done
rm -f "$SSH_ANSWER.srv3" "$FLEET_DIR"/.single-gpu-evidence*
with_lock _try_hold S4 $$ 5 "check" single; echo "S4b=$?"
with_lock _try_hold S5 $$ 5 "check" single; echo "S5=$?"
with_lock _try_hold A $$ 30 "a boot" boot; echo "A=$?"
for h in srv4 srv3 srv1 srv2; do echo "$h=$(cut -d'|' -f1 "$(single_holder $h)" 2>/dev/null)"; done
single_line
echo "S2 on $(python3 "$REPO/bench/fleet_single.py" assigned --session S2 --cache "$FLEET_DIR")"
with_lock _release S2; echo "releaseS2=$?"
with_lock _try_hold S5 $$ 5 "check" single; echo "S5b=$?"
with_lock _kick --force single srv3; echo "kick=$?"
for s in S1 S3 S5; do with_lock _release $s; done
with_lock _try_hold A $$ 30 "a boot" boot; echo "A2=$?"
''')
        out = result.stdout
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('S1=0', 'S2=0', 'S3=0', 'S4=1', 'S4b=0', 'S5=1', 'A=1',
                         'srv4=S1', 'srv3=S4', 'srv1=S2', 'srv2=S3',
                         'single (GB10 on srv4 srv3 srv1 srv2 beside production):', '  srv1: HELD by S2 [single]',
                         'S2 on srv1', 'releaseS2=0', 'S5b=0', 'kicked', 'kick=0', 'A2=0'):
            self.assertIn(expected, out, out + result.stderr)
        log = (self.fleet / 'log').read_text()
        for expected in ('[single: GB10 on srv4 beside production]', '[single: GB10 on srv1 beside production]',
                         '[single: GB10 on srv2 beside production]', '[single: GB10 on srv3 beside production]',
                         "hold refused (single): srv3: no room beside production -- MemAvailable 20.0 GiB",
                         'release S2 [single] on srv1', 'kick --force [single] of S4'):
            self.assertIn(expected, log, log)
        self.assertFalse(list(self.fleet.glob('holder-single*')))                  # every Spark handed back

    def test_a_check_lane_pointed_at_a_fleet_box_refuses(self):
        """The check lane exists so a check never meets the Sparks: a fleet box named as its host is a reason."""
        self.queue((1, 'E', int(time.time()), 5, 'a 5050 check', 'check', os.getpid()))
        result = self.run_fleet('''
with_lock _try_hold E $$ 5 "a 5050 check" check; echo "E=$?"
check_line
''', FLEET_CHECK_GPU_HOST='srv3')
        self.assertIn('E=1', result.stdout)
        self.assertIn("srv3 is one of the fleet's boxes -- the check lane is a box of its own", result.stdout)
        self.assertFalse((self.fleet / 'holder-check').exists())
        result = self.run_fleet('with_lock _try_hold E $$ 5 "a 5050 check" check; echo "E=$?"; check_line',
                                FLEET_CHECK_GPU_HOST='')
        self.assertIn('E=1', result.stdout)
        self.assertIn('check: off (FLEET_CHECK_GPU_HOST is empty)', result.stdout)

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
                         'bench/fleet_classify.py', 'bench/fleet_single.py', 'bench/onepass.py', 'bench/st_bracket.sh',
                         'bench/measurement_contract.py', 'bench/st_screen.py', 'bench/st_judge.py',
                         'bench/onepass_recording.py', 'bench/onepass_quality.py',
                         'bench/fleet_handoff.py', 'bench/fleet_pending.py', 'bench/fleet_idle.py',
                         'probes/run_engine_probe.sh', 'probes/run_engine_check.sh',
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
        lane = 'needs one GPU, not four: single-GPU lane (GB10 on srv4 srv3 srv1 srv2 beside production)'
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
        result = self.run_fleet('run', '--gpu', 'boot', '25', 'a boot', '--', 'python3', 'bench/onepass.py')
        self.assertNotIn(lane, result.stdout)
        self.assert_stopped_at_preparation(result)

    def test_check_sends_a_one_gpu_check_to_its_own_lane_and_nothing_else(self):
        check = ('bash', 'probes/run_engine_check.sh', '--layers', '0-4')
        result = self.run_fleet('run', '--gpu', '--check', 'st', '5', 'kernel check', '--', *check)
        self.assertIn('check lane (RTX5050 on ost-97x): a compile, correctness or shape verdict, never a number',
                      result.stdout)
        self.assertIn('budget: 4.0 GiB on ost-97x', result.stdout)                   # that box's, not a Spark's 8
        self.assertNotIn('single-GPU lane', result.stdout)
        self.assert_stopped_at_preparation(result)
        self.assertFalse((self.directory / 'holder-check').exists())
        for command in (('python3', 'bench/onepass.py'),
                        ('bash', 'probes/run_engine_probe.sh', 'engine/profiles/glm53/check.py', '--distributed')):
            result = self.run_fleet('run', '--gpu', '--check', 'st', '5', 'x', '--', *command)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn('REFUSED: --check takes a one-GPU ST check', result.stdout)
            self.assertFalse(self.prepared.exists())
        result = self.run_fleet('run', '--gpu', '--check', 'st', '5', 'x', '--', *check, FLEET_CHECK_GPU_HOST='')
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('the check lane is off', result.stdout)
        self.assertFalse(self.prepared.exists())

    def test_preflight_knows_the_check_lane_and_its_budget(self):
        result = self.run_fleet('preflight', '--check', 'st', '--', 'bash', 'probes/run_engine_check.sh', '--layers', '0-4')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('-> PASS', result.stdout)
        result = self.run_fleet('preflight', '--check', 'st', '--', 'python3', 'bench/onepass.py')
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('the check lane takes only an ST check', result.stdout)
        big = next(p for p in policy.ST_PROBES if policy.probe_budget_gib(p) > policy.CHECK_MAX_BUDGET_GIB)
        result = self.run_fleet('preflight', '--check', 'st', '--', 'bash', 'probes/run_engine_probe.sh', big)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('the check lane is a discrete 8 GiB card for kernel checks', result.stdout)
        result = self.run_fleet('preflight', '--single', 'st', '--', 'bash', 'probes/run_engine_probe.sh', big)
        self.assertEqual(result.returncode, 0, result.stdout)                           # the single lane still takes it
        self.assertFalse(self.prepared.exists())

    def test_preflight_knows_the_lane_and_refuses_a_boot_in_it(self):
        result = self.run_fleet('preflight', '--single', 'st', '--', 'bash', 'probes/run_engine_check.sh', '--layers', '0-4')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('-> PASS', result.stdout)
        result = self.run_fleet('preflight', '--single', 'st', '--', 'python3', 'bench/onepass.py')
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('needs the four Sparks', result.stdout)
        self.assertFalse(self.prepared.exists())


class PoolTests(unittest.TestCase):
    """The single lane's pool: which hosts, which holder each, where a check went, and the controller's own share."""

    def test_the_pool_its_holders_and_where_a_check_went(self):
        self.assertEqual(single.hosts({}), ['srv4', 'srv3', 'srv1', 'srv2'])
        self.assertEqual(single.hosts({'FLEET_SINGLE_GPU_HOST': 'ost-97x'}), ['ost-97x'])         # one host named: a pool of one
        self.assertEqual(single.hosts({'FLEET_SINGLE_GPU_HOST': ''}), [])                          # the lane off
        self.assertEqual(single.hosts({'FLEET_SINGLE_GPU_HOSTS': 'srv1 srv2', 'FLEET_SINGLE_GPU_HOST': 'srv4'}),
                         ['srv1', 'srv2'])
        self.assertEqual(single.hosts({}, lane='check'), ['ost-97x'])
        self.assertEqual(single.holder_name('srv4', {}), 'holder-single')                          # the lane's own file
        self.assertEqual(single.holder_name('srv2', {}), 'holder-single@srv2')
        self.assertEqual(single.holder_name('ost-97x', {'FLEET_SINGLE_GPU_HOST': 'ost-97x'}), 'holder-single')
        self.assertEqual(single.label({}), 'GB10 on srv4 srv3 srv1 srv2 beside production')
        with tempfile.TemporaryDirectory() as name, patch.dict(os.environ, {}, clear=False):
            for variable in (single.HOSTS_ENV, 'FLEET_SINGLE_GPU_HOST'):
                os.environ.pop(variable, None)
            directory = Path(name)
            (directory / 'queue').write_text('')
            self.assertTrue(handoff.admit(directory, 'S2', os.getpid(), 'single', '5', 'x', holder_file='holder-single@srv3'))
            self.assertEqual((directory / 'holder-single@srv3').read_text().split('|')[0], 'S2')
            self.assertEqual(single.assigned(directory, 'S2'), 'srv3')
            self.assertEqual(single.assigned(directory, 'S9'), '')
            self.assertEqual(handoff.holders(directory)['single@srv3'][0], 'S2')
            with self.assertRaises(ValueError):
                handoff.admit(directory, 'S3', os.getpid(), 'single', '5', 'x', holder_file='holder-check')
        boot = (ROOT / 'bench/fleet_boot.py').read_text()
        self.assertIn('host = fleet_single.assigned(self.directory, self.session, self.env) or host', boot)
        self.assertIn("environment['ST_PROBE_LANE'] = kind", boot)

    def test_the_controller_runs_its_own_share_here(self):
        """srv2 is one of the four and does not ssh to itself: its evidence, reclaim and results run here."""
        with patch.dict(os.environ, {'FLEET_SINGLE_LOCAL_HOST': 'srv2'}):
            self.assertEqual(single.on_host('srv2', 'x'), ['bash', '-c', 'cd && x'])
            self.assertEqual(single.on_host('choiceoh@srv2', 'x'), ['bash', '-c', 'cd && x'])
            self.assertEqual(single.on_host('srv4', 'x')[-2:], ['srv4', 'x'])
            calls = []

            def run(argv, **kwargs):
                calls.append(argv)
                return Done(0, '.cache/st/a.json\n') if argv[0] == 'bash' else Done(0)
            with tempfile.TemporaryDirectory() as into:
                self.assertEqual(single.collect('srv2', 0, into, run=run), ['a.json'])
            self.assertEqual(calls[1][:2], ['cp', str(Path.home() / '.cache/st/a.json')])


class PriorityLaneTests(unittest.TestCase):
    """Each one-GPU lane is its own line: the check lane's small tickets batch and rank among themselves."""

    def test_each_one_gpu_lane_ranks_its_own_head(self):
        import fleet_priority
        lines = ["1|boot|100|40|a boot|boot|", "2|s1|200|5|a GB10 check|single|", "3|c1|300|5|a 5050 check|check|",
                 "4|c2|400|30|a long 5050 check|check|", "5|s2|500|30|a long GB10 check|single|"]
        rows = fleet_priority.rank(lines, {}, 1000)
        by_lane = {}
        for row in rows:
            by_lane.setdefault(row["lane"], []).append(row["session"])
        self.assertEqual(by_lane, {'fleet': ['boot'], 'single': ['s1', 's2'], 'check': ['c1', 'c2']})
        self.assertEqual({r["session"]: r["batch"] for r in rows if r["lane"] == "check"}, dict(c1=True, c2=False))

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'the handoff receipt reads /proc')
    def test_the_lease_passes_to_the_fleet_lane_s_head_not_to_a_one_gpu_check(self):
        """A one-GPU check ranks ahead of a waiting boot in the one ranked order; the fleet's lease still goes to that
        boot, not away with production restarting under it (review of #1297)."""
        import fleet_priority
        with tempfile.TemporaryDirectory() as name:
            directory, pid = Path(name), os.getpid()
            lines = [f'1|c1|100|5|a 5050 check|check|{pid}', f'2|s1|150|5|a GB10 check|single|{pid}',
                     f'3|next|200|40|a boot|boot|{pid}']
            (directory / 'queue').write_text(''.join(line + '\n' for line in lines))
            self.assertEqual(fleet_priority.rank(lines, {}, time.time())[0]['session'], 'c1')     # the case at hand
            handoff.ready(directory, 'next', pid)
            self.assertEqual(handoff.successor(directory, 'donor')['session'], 'next')


if __name__ == '__main__':
    unittest.main()
