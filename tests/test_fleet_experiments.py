"""Behavioral tests of admission, duplicate callers and evidence, without GPUs.

The subprocess tests use committed temporary repos and a fake fleet boundary;
no SSH, Docker daemon, serving container or real fleet directory is touched.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import experiments as ex
import cpu_evidence
import fleet_priority
import probe_report
from measurement_contract import metadata

BASH = shutil.which("bash")
FAKE_FLEET = r'''#!/usr/bin/env bash
set -eu
action=$1; shift
case "$action" in
  classify) case "$*" in *GPU_MARKER*) echo gpu;; *) echo nogpu;; esac;;
  preflight) echo preflight >> "$LOGD/admissions";;
  restore-needed) exit 1;;
  yield) exit 0;;
  run)
    lane=$1; shift
    [ "${1:-}" != --probe ] || shift
    session=$1; shift 3
    [ "$1" = -- ]; shift
    echo "$session $lane" >> "$LOGD/admissions"
    sleep "${ADMISSION_DELAY:-0}"
    export FLEET_SESSION=$session
    if [ "$lane" = --gpu ]; then
      while ! mkdir "$FLEET_DIR/held" 2>/dev/null; do sleep .02; done
      echo "$session|$$|test|0|1|test|boot" > "$FLEET_DIR/holder"
      trap 'rm -f "$FLEET_DIR/holder"; rmdir "$FLEET_DIR/held"' EXIT
    fi
    "$@"
    ;;
  *) exit 20;;
esac
'''

FAKE_LEVER = r'''#!/usr/bin/env bash
set -eu
echo "$1 ${LEGS:-onepass}" >> "$LOGD/arms"
if [ "${FAIL_ARM:-}" = "$1" ]; then exit 7; fi
[ "${LEGS:-onepass}" != none ] || exit 0
python3 - "$1" "${2:-}" <<'PY'
import json, os, sys
from pathlib import Path
Path(os.environ['LOGD'],'served.json').write_text(json.dumps(dict(
 boot_id=sys.argv[1], knobs=dict(v.split('=',1) for v in sys.argv[2].split()))))
PY
python3 bench/onepass.py --name "$1"
'''

FAKE_ONEPASS = r'''import json, os, subprocess, sys
from pathlib import Path
from measurement_contract import metadata

def _served_build(repo):
 return json.loads(Path(os.environ['LOGD'],'served.json').read_text())

if __name__ == '__main__':
 name = sys.argv[sys.argv.index('--name')+1]
 served = _served_build('.')
 knobs = served['knobs']
 work = json.loads(os.environ.get('FLEET_WORKLOAD', '{}'))
 r = dict(name=name, git=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
          overlay=Path(os.environ['MK_OVERLAY_STAMP']).read_text().strip()[:12],
          **metadata(work), **served,
          runtime=json.loads(os.environ.get('FLEET_CONTEXT', '{}')),
          experiment_id=os.environ.get('FLEET_EXPERIMENT_ID'),
          quality={'ok':9,'total':9}, korean={'dirty':0,'n':5},
          decode={'windows_med':120 if knobs else 100},
          prefill=[dict(ctx=c,cold_s=.8 if knobs else 1) for c in work.get('ctx',[2000,32000,128000])],
          proof={k:True for k in knobs}, proof_ok=f'{len(knobs)}/{len(knobs)}')
 r['requests'] = [dict(fixed_decode=True, completion_tokens=work['fixed_decode_tokens'],
                       decode_s=8 if knobs else 10) for _ in range(work.get('fixed_decode_reps',0))]
 with open(os.environ['ONEPASS_JSONL'],'a') as f: f.write(json.dumps(r)+'\n')
'''


def record(name="base", speed=100, **changes):
    row = dict(name=name, overlay="a" * 12, git="b" * 40, **metadata(),
               knobs={}, quality={"ok": 9, "total": 9}, korean={"dirty": 0, "n": 5},
               decode={"windows_med": speed}, proof={}, proof_ok="0/0",
               boot_id=f"{name}-{speed}", runtime={})
    row.update(changes)
    return row




class PriorityTests(unittest.TestCase):
    def test_short_unlocking_job_wins_then_aging_protects_long_job(self):
        lines = ["1|long|100|30|long|boot|", "2|short|200|5|short|probe|"]
        self.assertEqual(fleet_priority.rank(lines, {"short": 4}, 300)[0]["session"], "short")
        # aged 30 minutes the long job no longer blocks the small batch; the batch's allowance ends at 45
        self.assertEqual(fleet_priority.rank(lines, {"short": 100}, 1900)[0]["session"], "short")
        self.assertEqual(fleet_priority.rank(lines, {"short": 100}, 100 + fleet_priority.STARVE_S)[0]["session"], "long")
        self.assertEqual(fleet_priority.rank(lines, {"short": 100}, 300, front="long")[0]["session"], "long")
        self.assertEqual(fleet_priority.rank(lines, {}, 1900, front="long", yielded="short")[0]["session"], "short")
        self.assertEqual(fleet_priority.rank(lines, {"short":100}, 300, probes_ready=False)[0]["session"], "long")

    def test_small_tickets_go_as_a_batch_oldest_first_up_to_the_cap_in_each_lane(self):
        """A CPU pipeline drains the short instructions queued behind a long one (operator, 2026-09-13): the
        lane's small tickets run ahead of anything larger, oldest first, as far as the cap lets them."""
        self.assertEqual((fleet_priority.SMALL_MAX_MIN, fleet_priority.BATCH_CAP_MIN), (5, 15))
        lines = ["1|big|100|30|a long check|single|",
                 "2|s1|200|5|small|single|", "3|s2|300|5|small|single|", "4|s3|400|5|small|single|",
                 "5|s4|500|2|small, past the cap|single|",
                 "6|boot|150|40|a boot|boot|", "7|f1|250|3|a small boot-lane ticket|probe|"]
        rows = fleet_priority.rank(lines, {}, 1000)
        single = [r["session"] for r in rows if r["lane"] == "single"]
        fleet = [r["session"] for r in rows if r["lane"] == "fleet"]
        self.assertEqual(single[:3], ["s1", "s2", "s3"])                      # 15 minutes of small work: the batch
        self.assertEqual({r["session"]: r["batch"] for r in rows if r["lane"] == "single"},
                         dict(big=False, s1=True, s2=True, s3=True, s4=False))  # a fourth would make 17
        self.assertEqual(single[3:], ["s4", "big"])                           # past the cap: the score (shorter first)
        self.assertEqual(fleet, ["f1", "boot"])                               # the other lane has its own batch
        # a ticket starved of its turn is passed by nothing but the explicit front and a yielded probe
        starved = fleet_priority.rank(lines, {}, 100 + fleet_priority.STARVE_S)
        self.assertEqual([r["session"] for r in starved if r["lane"] == "single"][0], "big")
        fronted = fleet_priority.rank(lines, {}, 100 + fleet_priority.STARVE_S, front="s4")
        self.assertEqual(fronted[0]["session"], "s4")

    def test_history_estimates_come_from_the_ledger_by_session_then_family(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.tsv"
            rows = [("kern-timeline", 9), ("kern-timeline", 11), ("c4-rows-decode-profile", 30),
                    ("c4-rows-decode-profile", 2), ("c4-rows-decode-profile", 3), ("c4-rows-decode-profile", 2),
                    ("c4-rows-decode-profile", 2), ("c4-rows-decode-profile", 2), ("zero", 0)]
            ledger.write_text("".join(f"2026-09-13_12:00:00\t{s}\tsingle\tnote\t{m}\t0\t0\t0\n" for s, m in rows)
                              + "a malformed line\n")
            got = fleet_priority.history_estimates(ledger, ["kern-timeline", "c4-rows2-decode-profile", "zero", "new"])
            self.assertEqual(got["kern-timeline"], dict(minutes=9, source="history"))       # the lower median
            self.assertEqual(got["c4-rows2-decode-profile"], dict(minutes=2, source="family"))  # last five of the family
            self.assertNotIn("zero", got)                                                    # a zero hold says nothing
            self.assertNotIn("new", got)
            self.assertEqual(fleet_priority.history_estimates(Path(directory) / "absent.tsv", ["new"]), {})
            # a ticket declared at 45 minutes that history knows takes 2 goes with the small batch
            lines = ["1|big|100|20|a long check|single|", "2|c4-rows2-decode-profile|200|45|declared long|single|"]
            ranked = fleet_priority.rank(lines, {}, 300, estimates=got)
            self.assertEqual(ranked[0]["session"], "c4-rows2-decode-profile")
            self.assertEqual((ranked[0]["estimate_min"], ranked[0]["estimate_source"], ranked[0]["batch"]), (2, "family", True))

    def test_downstream_counts_unique_pending_jobs_including_shared_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ex.Store(directory)
            def submit(name, deps):
                return store.submit(name, dict(spec=dict(depends_on=deps, revision="a"*40, hypothesis=name,
                                                       command=[name]), environment={}))["id"]
            base = submit("base", [])
            probe = submit("probe", [])
            left = submit("left", [probe])
            right = submit("right", [probe])
            final = submit("final", [left, right])
            with store.db:
                store.db.execute("INSERT INTO dependencies VALUES(?,?,?)", (probe, base, "baseline"))
            counts = fleet_priority.downstream(Path(directory) / "experiments.sqlite3")
            self.assertEqual(counts["exp-" + base], 4)
            self.assertEqual(counts["exp-" + probe], 3)
            store.state(final, "blocked")
            self.assertEqual(fleet_priority.downstream(Path(directory) / "experiments.sqlite3")["exp-" + probe], 2)

    def test_real_admission_function_preserves_live_holder_and_uses_priority_when_free(self):
        source = (ROOT / "bench/fleet.sh").read_text()
        function = source[source.index("_try_hold() {"):source.index("_ledger_row() {")]
        # the admission function's own one-line helpers: the kind, its lane, its holder, its head,
        # and whether the single-GPU lane's box is one of the fleet's
        helpers = [line for line in source.splitlines()
                   if line.startswith(("kind_of() {", "one_gpu() {", "lane_of() {", "holder_file() {", "lane_front() {",
                                       "single_on_fleet() {"))]
        self.assertEqual(len(helpers), 6)
        function = "\n".join(helpers) + "\n" + function
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue"
            holder = root / "holder"
            now = int(time.time())
            lines = f"1|long|{now}|30|long|boot|\n2|short|{now}|2|short|boot|\n"
            queue.write_text(lines)
            holder.write_text("active|123|host|0|30|pair|boot\n")
            # A free fleet whose lease the ticket takes at GO (PR #770): the occupancy check and
            # the lease helpers answer "free" and "taken"; nothing here touches a real lease.
            setup = '''
H="$FLEET_DIR/holder"; HS="$FLEET_DIR/holder-single"; Q="$FLEET_DIR/queue"
FLEET_SINGLE_GPU_HOST=srv4
holder_alive() { return 0; }
logit() { :; }
_event() { :; }
_dequeue() { :; }
serving_idle() { return 0; }
legacy_busy() { return 1; }
st_engine_up() { return 1; }
st_engine_ask() { :; }
lease() { return 0; }
lease_mine() { return 1; }
lease_state() { echo free; }
_lease_pass_on() { :; }
_kick_lease() { :; }
me() { echo host; }
now() { date +%s; }
'''
            # A restore can invoke this test from an older pinned controller.
            # The extracted admission function must use this checkout's
            # helpers and temporary fleet, not the caller's live runner.
            env = dict(os.environ, FLEET_DIR=directory, LOGD=directory,
                       REPO=str(ROOT), FLEET_RUNNER_REPO=str(ROOT))
            def admit():
                return subprocess.run([BASH, "-c", setup + function + "\n_try_hold short $$ 2 note boot"],
                                      env=env, capture_output=True, text=True)
            self.assertEqual(admit().returncode, 1)
            self.assertEqual(queue.read_text(), lines)
            self.assertTrue(holder.read_text().startswith("active|"))
            holder.unlink()
            result = admit()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('command not found', result.stderr)
            self.assertTrue(holder.read_text().startswith("short|"))


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "bench").mkdir()
        (self.repo / "profiles").mkdir()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.fleet = self.logs / "fleet"
        self.fleet.mkdir()
        self.jobs = self.root / "jobs"
        self.overlay = self.root / "overlay"
        self.overlay.mkdir()
        self.stamp = self.root / "stamp"
        self.stamp.write_text("a" * 64)
        self.image = "sha256:" + "d" * 64
        self.bin = self.root / "bin"
        self.bin.mkdir()
        docker = self.bin / "docker"
        docker.write_text("#!/bin/sh\necho " + self.image + "\n")
        docker.chmod(0o755)
        (self.repo / "profiles/glm53.env").write_text(
            f"PROFILE_OVERLAY_DIR={self.overlay}\nVLLM_TEST=0\n")
        (self.repo / "bench/fleet.sh").write_text(FAKE_FLEET)
        (self.repo / "bench/ab-lever.sh").write_text(FAKE_LEVER)
        (self.repo / "bench/onepass.py").write_text(FAKE_ONEPASS)
        for name in ("experiments.py", "cpu_checks.py",
                     "cpu_evidence.py", "fleet_source.py", "probe_report.py", "fleet_priority.py", "fleet_handoff.py", "measurement_contract.py",
                     "experiment_resources.py", "prepared_artifacts.py", "cpu_unittest.py",
                     "experiment_plan.py", "cpu_compile.py", "experiment_sharing.py", "experiment_retirement.py", "experiment_metrics.py", "experiment_submission.py", "experiment_explain.py"):
            shutil.copy(ROOT / "bench" / name, self.repo / "bench" / name)
        for script in (self.repo / "bench").glob("*.sh"):
            script.chmod(0o755)
        (self.repo / ".gitignore").write_text("__pycache__/\nbuild/\n")
        self.commit()
        (self.overlay / "manifest.tsv").write_text("# source_commit=" + self.sha + "\n")
        self.stamp.write_text(ex.digest(self.overlay / "manifest.tsv"))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("FLEET_", "ONEPASS_"))}
        self.env.update(REPO=str(self.repo), LOGD=str(self.logs), FLEET_DIR=str(self.fleet),
                        FLEET_EXPERIMENT_ROOT=str(self.jobs), MK_OVERLAY_STAMP=str(self.stamp),
                        PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        ONEPASS_JSONL=str(self.logs / "onepass.jsonl"),
                        ONEPASS_VERDICTS=str(self.logs / "verdicts.jsonl"))

    def commit(self):
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Fleet Test", "-c", "user.email=fleet@example.invalid",
                        "commit", "-qm", "fixture"], check=True)
        self.sha = ex.git(self.repo, "rev-parse", "HEAD")

    def cli(self, *args, ok=True):
        result = subprocess.run([sys.executable, str(ROOT / "bench/experiments.py"), *args], env=self.env,
                                text=True, capture_output=True, timeout=15)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return json.loads(result.stdout)
        return result

    def submit(self, session="agent-a", **changes):
        spec = dict(kind="cpu", revision=self.sha, hypothesis="check the contract",
                    command=[sys.executable, "-c", "print('CPU contract passed')"])
        spec.update(changes)
        path = self.root / (session + ".json")
        path.write_text(json.dumps(spec))
        return self.cli("submit", session, str(path))

    def wait(self, job):
        result = self.cli("wait", job, "--timeout", "10")
        self.assertIn(result["state"], ex.TERMINAL, result)
        return result

    def test_detached_cpu_work_and_shared_result(self):
        output = self.root / "count"
        release = self.root/'release'
        command = [sys.executable, "-c", "import time;from pathlib import Path\n"
            f"with open({str(output)!r},'a') as stream:stream.write('run\\n')\n"
            "deadline=time.monotonic()+10\n"
            f"while not Path({str(release)!r}).exists():\n assert time.monotonic()<deadline\n time.sleep(.01)\n"]
        a = self.submit(command=command)
        b = self.submit("agent-b", command=command, hypothesis="another consumer of the same contract")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(b['disposition'],'joined')
        release.touch()
        result = self.wait(a["id"])
        self.assertEqual(result["state"], "succeeded", result)
        self.assertEqual(result["result"]["evidence"], "cpu-only")
        self.assertEqual(output.read_text(), "run\n")
        c = self.submit("agent-c", command=command)
        self.assertEqual(c["disposition"], "reused")
        for session in ("agent-a", "agent-b", "agent-c"):
            inbox = self.cli("inbox", session)
            self.assertTrue(any(e["event"] == "succeeded" for e in inbox["events"]))
            self.assertEqual(self.cli("inbox", session, "--after", str(inbox["cursor"]))["events"], [])
        self.assertFalse((self.fleet / "holder").exists())
        stats = self.cli("stats")
        self.assertEqual(stats["by_kind"]["cpu"]["valid_results"], 1)
        self.assertEqual(stats["requests"]["submitted"], 1)


    def test_custom_gpu_probe_rejected_before_source_or_gpu_inspection(self):
        context = dict(image=self.image, model="fixture", hardware="fixture")
        raw = dict(kind="probe", revision=self.sha, hypothesis="custom GPU workload",
                   command=[sys.executable, "-c", "pass"], context=context)
        with patch.object(ex, 'snapshot', side_effect=AssertionError('no source inspection')):
            with self.assertRaisesRegex(ValueError, "onepass-only"):
                ex.normalize(raw, self.repo)
            raw['probe_contract'] = {"checks":{"mismatches":{"op":"eq","value":0}},
                                     "proof":["lane"],"min_samples":1}
            with self.assertRaisesRegex(ValueError, "onepass-only"):
                ex.normalize(raw, self.repo)
        self.assertFalse((self.logs / "admissions").exists())

    def test_legacy_gpu_probe_blocked_before_worker_checkout_or_execution(self):
        store = ex.Store(self.jobs)
        try:
            for entry in (ex.ensure_worker, ex.worker, ex.execute):
                with self.subTest(entry=entry.__name__):
                    spec = dict(kind="probe", revision=self.sha, hypothesis="legacy queued probe",
                                command=["GPU_MARKER"], depends_on=[])
                    payload = dict(spec=spec, repo=str(self.repo), environment={}, paths={}, snapshot={})
                    job = store.submit("legacy", payload, repeat=entry.__name__)["id"]
                    with patch.object(ex, 'verify', side_effect=AssertionError('no source inspection')), \
                         patch.object(ex.subprocess, 'Popen', side_effect=AssertionError('no child process')), \
                         patch.object(ex.subprocess, 'run', side_effect=AssertionError('no worktree')):
                        entry(store, job)
                    row = store.get(job)
                    self.assertEqual(row['state'], 'blocked')
                    self.assertEqual(row['result']['evidence'], 'onepass-policy')
                    self.assertIn('onepass-only', row['result']['reason'])
                    self.assertFalse((self.jobs / job / 'checkout').exists())
                    from experiment_retry import source
                    with self.assertRaisesRegex(ValueError, 'onepass-only'):
                        source(store, 'legacy', job)
        finally:
            store.db.close()
        self.assertFalse((self.logs / "admissions").exists())



    def test_named_cpu_cache_reuses_identical_merge_but_not_source_or_env_changes(self):
        (self.repo / "tests").mkdir()
        test = self.repo / "tests/test_fleet_experiments.py"
        test.write_text("import unittest\nclass Contract(unittest.TestCase):\n def test_ok(self): self.assertEqual(2+2,4)\n")
        self.commit()
        command = [sys.executable, "bench/cpu_checks.py", "--suite", "fleet"]
        first = self.submit("first", command=command)
        self.assertEqual(self.wait(first["id"])["state"], "succeeded")
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c", "user.email=t@example.invalid",
                        "commit", "--allow-empty", "-qm", "same tree merge"], check=True)
        self.sha = ex.git(self.repo, "rev-parse", "HEAD")
        second = self.submit("second", command=command)
        reused = self.wait(second["id"])
        self.assertEqual(reused["result"]["cache_source"], first["id"])
        self.assertEqual(reused["result"]["revision"], self.sha)
        self.assertNotEqual(reused["result"]["tested_revision"], self.sha)
        self.assertEqual((self.logs / "admissions").read_text().count("--cpu"), 1)
        changed_env = self.submit("env", command=command, env={"TEST_VARIANT":"new"})
        self.assertNotIn("cache_source", self.wait(changed_env["id"])["result"])
        test.write_text(test.read_text() + "# changed test source\n")
        self.commit()
        changed_source = self.submit("source", command=command)
        self.assertNotIn("cache_source", self.wait(changed_source["id"])["result"])


    def test_generic_cpu_command_is_never_content_cached(self):
        spec = ex.normalize(dict(kind="cpu", revision=self.sha, hypothesis="generic", command=["true"]), self.repo)
        self.assertIsNone(cpu_evidence.identity(self.repo, spec, {}))

    def test_concurrent_submit_is_one_job(self):
        spec = ex.normalize(dict(kind="cpu", revision=self.sha, hypothesis="race", command=["true"]), self.repo)
        payload = dict(spec=spec, repo=str(self.repo), environment={}, paths={}, snapshot={})
        ex.Store(self.jobs)
        def submit(index):
            store = ex.Store(self.jobs)
            try:
                return store.submit("agent-" + str(index), payload)["id"]
            finally:
                store.db.close()
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(submit, range(16)))
        self.assertEqual(len(set(jobs)), 1)


    def test_cpu_gpu_misclassification_is_refused(self):
        job = self.submit(command=["GPU_MARKER"])
        result = self.wait(job["id"])
        self.assertEqual(result["state"], "failed")
        self.assertIn("GPU use", result["result"]["reason"])
        self.assertFalse((self.logs / "admissions").exists())

    def test_cpu_timeout_blocks_dependents(self):
        job = self.submit(command=[sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=1)
        result = self.wait(job["id"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["result"]["returncode"], 124)

    def test_revision_changed_during_queue_never_executes_command(self):
        output = self.root / "ran"
        job = self.submit(command=[sys.executable, "-c", f"open({str(output)!r},'w').write('bad')"],
                          env={"ADMISSION_DELAY": "1"})
        deadline = time.monotonic() + 5
        while not (self.logs / "admissions").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        frozen = Path(ex.Store(self.jobs).get(job["id"])["payload"]["repo"])
        (frozen / "uncommitted").write_text("changed while waiting")
        result = self.wait(job["id"])
        self.assertEqual(result["state"], "failed", result)
        self.assertFalse(output.exists())

    def test_agent_can_keep_editing_after_submission(self):
        job = self.submit(env={"ADMISSION_DELAY": "1"})
        (self.repo / "next-change").write_text("agent continues implementation")
        result = self.wait(job["id"])
        self.assertEqual(result["state"], "succeeded", result)
        self.assertNotEqual(result["checkout"], str(self.repo))
        self.assertFalse((Path(result["checkout"]) / "next-change").exists())

    def test_explicit_repeat_is_new_sample_but_does_not_mutate_old_result(self):
        first = self.submit()
        old = self.wait(first["id"])
        repeat = self.cli("submit", "agent-a", str(self.root / "agent-a.json"), "--repeat", "confirm repeatability")
        self.assertNotEqual(first["id"], repeat["id"])
        self.wait(repeat["id"])
        self.assertEqual(self.cli("result", first["id"])["result"], old["result"])

    def test_stale_worker_does_not_blindly_repeat_gpu_work(self):
        store = ex.Store(self.jobs)
        payload = dict(spec=dict(depends_on=[], revision=self.sha, hypothesis="stale"), environment={})
        job = store.submit("agent", payload)["id"]
        store.state(job, "queued_fleet")
        ex.ensure_worker(store, job)
        self.assertEqual(store.get(job)["state"], "interrupted")

    def test_worker_finishing_during_lock_check_keeps_terminal_result(self):
        store = ex.Store(self.jobs)
        self.addCleanup(store.db.close)
        original_lock = ex.worker_lock
        for terminal in sorted(ex.TERMINAL):
            with self.subTest(terminal=terminal):
                payload = dict(spec=dict(depends_on=[], revision=self.sha,
                                         hypothesis=terminal, command=["true", terminal]), environment={})
                job = store.submit("agent", payload)["id"]
                def finish_then_lock(current, identifier):
                    current.state(identifier, terminal, {"completed": True})
                    return original_lock(current, identifier)
                with patch.object(ex, "worker_lock", side_effect=finish_then_lock), \
                     patch.object(ex.subprocess, "Popen") as launch:
                    ex.ensure_worker(store, job)
                    launch.assert_not_called()
                self.assertEqual(store.get(job)["state"], terminal)
                self.assertEqual(store.get(job)["result"], {"completed": True})





    def test_baseline_policy_rejects_unknown_values_and_non_pair_requests(self):
        for kind, policy in (('cpu','minimal'),('pair','skip'),('pair',None),('pair',[])):
            raw = dict(kind=kind, revision=self.sha, hypothesis='test', command=['true'] if kind=='cpu' else [],
                       knobs={} if kind=='cpu' else {'VLLM_TEST':'1'},context=self.pair_context(),baseline_policy=policy)
            with self.subTest(kind=kind,policy=policy), self.assertRaises(ValueError):
                ex.normalize(raw, self.repo)




    def test_invalid_manifest_and_old_prerequisite_are_rejected(self):
        for change in ({"revision":"main"}, {"env":{"FLEET_SESSION":"x"}}, {"env":{"FLEET_REHEARSE":"1"}},
                       {"env":{"PYTHONPATH":"/tmp"}}, {"command":"echo ok"}):
            spec = dict(kind="cpu", revision=self.sha, hypothesis="contract", command=["true"])
            spec.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ex.normalize(spec, self.repo)
        store = ex.Store(self.jobs)
        payload = dict(spec=dict(depends_on=[], revision=self.sha, hypothesis="first"), environment={})
        job = store.submit("a", payload)["id"]
        payload["spec"] = dict(depends_on=[job], revision="f"*40, hypothesis="new revision")
        with self.assertRaisesRegex(ValueError, "same committed"):
            store.submit("b", payload)



    def refresh_deployed_fixture(self):
        self.commit()
        (self.overlay / 'manifest.tsv').write_text('# source_commit=' + self.sha + '\n')
        self.stamp.write_text(ex.digest(self.overlay / 'manifest.tsv'))

    def pair_context(self):
        return dict(image=self.image, model='fixture', hardware='fixture')





    def test_individual_unittest_reports_counts_and_reuses_identical_tree(self):
        (self.repo/'tests').mkdir()
        (self.repo/'tests/test_individual.py').write_text(
            'import unittest\nclass Contract(unittest.TestCase):\n def test_ok(self): self.assertEqual(3*3,9)\n')
        self.commit()
        command = [sys.executable, 'tests/test_individual.py']
        first = self.submit(command=command)
        result = self.wait(first['id'])
        self.assertEqual(result['state'], 'succeeded', result)
        self.assertEqual(result['result']['checks']['tests_run'], 1)
        self.assertEqual(result['result']['command'][1:3], ['bench/cpu_checks.py','--test'])
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=T','-c','user.email=t@invalid',
                        'commit','--allow-empty','-qm','identical tree'], check=True)
        self.sha = ex.git(self.repo, 'rev-parse', 'HEAD')
        repeated = self.wait(self.submit('again', command=command)['id'])
        self.assertEqual(repeated['result']['cache_source'], first['id'])

    def test_empty_skipped_and_failing_unittests_never_report_complete_pass(self):
        (self.repo/'tests').mkdir()
        files = {
            'empty': 'import unittest\n',
            'skipped': "import unittest\nclass C(unittest.TestCase):\n @unittest.skip('fixture unavailable')\n def test_x(self): pass\n",
            'failure': 'import unittest\nclass C(unittest.TestCase):\n def test_x(self): self.fail("broken")\n'}
        for name, text in files.items():
            (self.repo/f'tests/test_{name}.py').write_text(text)
        self.commit()
        for name in files:
            result = self.wait(self.submit(name, command=[sys.executable,f'tests/test_{name}.py'])['id'])
            self.assertEqual(result['state'], 'failed', result)
            self.assertFalse(result['result']['checks']['passed'])
            self.assertEqual(result['result']['checks']['tests_run'], 0 if name=='empty' else 1)
        self.assertFalse((self.fleet/'holder').exists())

    def test_cpu_pool_capacity_and_stale_lease_recovery(self):
        import experiment_resources as resources
        (self.fleet/'cpu-policy.json').write_text(json.dumps(dict(slots=1,memory_mb=128,reserve_mb=0)))
        store = ex.Store(self.jobs)
        def job(name):
            return store.submit(name, dict(environment={}, paths={'FLEET_DIR':str(self.fleet)},
                spec=dict(revision=self.sha, depends_on=[], hypothesis=name, command=[name])))['id']
        first, second = job('first'), job('second')
        request = resources.normalize({'cpu_memory_mb':64})
        with patch.object(resources, 'memory', return_value=(1024,1024)):
            self.assertTrue(resources.acquire(store, first, request))
            self.assertFalse(resources.acquire(store, second, request))
            with store.db:
                store.db.execute('UPDATE cpu_leases SET pid=?', (1073741824,))
            self.assertTrue(resources.acquire(store, second, request))
            with self.assertRaisesRegex(ValueError, 'capacity'):
                resources.acquire(store, first, resources.normalize({'cpu_memory_mb':256}))

    def test_cpu_ram_budget_terminates_own_group_without_gpu_hold(self):
        command = [sys.executable, '-c', 'import time; data=bytearray(80*1024*1024); time.sleep(20)']
        job = self.submit(command=command, resources={'cpu_memory_mb':32})
        result = self.wait(job['id'])
        self.assertEqual(result['state'], 'failed', result)
        self.assertEqual(result['result']['returncode'], 137)
        self.assertFalse((self.fleet/'holder').exists())
        self.assertEqual(ex.Store(self.jobs).db.execute('SELECT count(*) FROM cpu_leases').fetchone()[0], 0)

    def test_cpu_resource_wait_is_counted_before_start(self):
        (self.fleet/'cpu-policy.json').write_text(json.dumps(dict(slots=1,memory_mb=256,reserve_mb=0)))
        release = self.root/'release'
        first = self.submit('first',command=[sys.executable,'-c','import time;from pathlib import Path\ndeadline=time.monotonic()+10\n'
                            f'while not Path({str(release)!r}).exists():\n assert time.monotonic()<deadline\n time.sleep(.01)\n'],
                            resources={'cpu_memory_mb':128})
        store = ex.Store(self.jobs)
        deadline = time.monotonic()+5
        while store.get(first['id'])['state'] != 'running' and time.monotonic()<deadline:
            time.sleep(.01)
        first_start = store.get(first['id'])['started']
        self.assertIsNotNone(first_start)
        second = self.submit('second',resources={'cpu_memory_mb':128})
        deadline=time.monotonic()+5
        while store.get(second['id'])['state']!='waiting_cpu' and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertEqual(store.get(second['id'])['state'],'waiting_cpu')
        self.assertIsNone(store.get(second['id'])['started'])
        released_at=time.time();release.touch()
        result = self.wait(second['id'])
        self.assertEqual(result['state'],'succeeded',result)
        self.assertGreaterEqual(result['started'],released_at)
        self.assertGreater(result['started'],result['created'])
        self.wait(first['id'])

    def test_cpu_cleanup_covers_child_after_leader_exits(self):
        import experiment_resources as resources
        marker = self.root/'child.pid'
        command = [sys.executable, '-c',
            "import subprocess; from pathlib import Path; p=subprocess.Popen(['sleep','30']); "
            f"Path({str(marker)!r}).write_text(str(p.pid))"]
        proc = subprocess.Popen(command, start_new_session=True)
        proc.wait(timeout=5)
        unrelated = subprocess.Popen(['sleep','30'], start_new_session=True)
        try:
            resources.stop(proc)
            status = subprocess.run(['ps','-p',marker.read_text(),'-o','stat='], capture_output=True,text=True)
            self.assertTrue(status.returncode or status.stdout.strip().startswith('Z'), status.stdout)
            self.assertIsNone(unrelated.poll())
        finally:
            resources.stop(unrelated)



    def test_compile_entrypoint_only_builds_an_object_and_is_cpu_classified(self):
        nvcc = self.bin/'nvcc'
        nvcc.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n'
                        'Path(sys.argv[-1]).write_text(json.dumps(sys.argv[1:]))\n')
        nvcc.chmod(0o755)
        (self.repo/'kernel.cu').write_text('__global__ void kernel() {}\n')
        self.commit()
        command = [sys.executable,'bench/cpu_compile.py','--source','kernel.cu','--arch','sm_90','--output','build/kernel.o']
        classified = subprocess.run([BASH,str(ROOT/'bench/fleet.sh'),'classify',*command],
                                    cwd=self.repo, env=self.env, text=True,capture_output=True,timeout=5)
        self.assertEqual(classified.stdout.strip(), 'nogpu', classified.stderr)
        job = self.submit(command=command,outputs=['build/kernel.o'])
        result = self.wait(job['id'])
        self.assertEqual(result['state'], 'succeeded', result)
        args = json.loads(Path(result['result']['artifacts'][0]['path']).read_text())
        self.assertEqual(args[:2], ['--compile','-arch=sm_90'])
        rejected = subprocess.run([*command,'--run'],cwd=self.repo,env=self.env,capture_output=True)
        self.assertNotEqual(rejected.returncode,0)


    def test_collect_delivery_and_ack_preserve_evidence_scope(self):
        path = self.root/'private.jsonl'
        path.write_text(json.dumps(record('private'))+'\n')
        archive = self.cli('collect','reader',str(path))
        result = self.cli('result',archive['id'])
        self.assertEqual(result['state'],'incomplete')
        self.assertEqual(result['result']['evidence'],'external-archive')
        self.assertEqual(Path(archive['artifact']).read_bytes(),path.read_bytes())
        job = self.submit('reader')
        self.wait(job['id'])
        rejected = self.cli('ack','reader','999999',ok=False)
        self.assertNotEqual(rejected.returncode,0)
        inbox = self.cli('inbox','reader','--wait','1')
        self.cli('ack','reader',str(inbox['cursor']))
        stats = self.cli('stats')['result_consumption']
        self.assertEqual(stats['delivery_s']['n'],1)
        self.assertEqual(stats['acknowledgment_s']['n'],1)
        self.assertEqual(stats['terminal_acknowledgment_s']['n'],2)
        self.assertGreaterEqual(stats['acknowledgment_s']['p50'],stats['delivery_s']['p50'])



if __name__ == "__main__":
    unittest.main()
