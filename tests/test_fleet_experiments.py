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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import experiments as ex
import judge
import cpu_evidence
import fleet_priority
import probe_report

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
import json, os, subprocess, sys
name, knobs = sys.argv[1:]
knobs = dict(v.split('=', 1) for v in knobs.split())
r = dict(name=name, git=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
         overlay=open(os.environ['MK_OVERLAY_STAMP']).read().strip()[:12], harness=39,
         doc_lang='ko', thinking=True, workload={'ctx':[2000,32000,128000], 'seed':7, 'max_tokens':400, 'combine_min_ctx':32000},
         boot_id=name,
         runtime=json.loads(os.environ.get('FLEET_CONTEXT', '{}')), knobs=knobs,
         experiment_id=os.environ.get('FLEET_EXPERIMENT_ID'),
         quality={'ok':9,'total':9}, korean={'dirty':0,'n':5},
         decode={'windows_med':120 if knobs else 100}, prefill=[],
         proof={k:True for k in knobs}, proof_ok=f'{len(knobs)}/{len(knobs)}')
with open(os.environ['ONEPASS_JSONL'], 'a') as f: f.write(json.dumps(r)+'\n')
PY
'''


def record(name="base", speed=100, **changes):
    row = dict(name=name, overlay="a" * 12, git="b" * 40, harness=39, doc_lang="ko", thinking=True,
               knobs={}, quality={"ok": 9, "total": 9}, korean={"dirty": 0, "n": 5},
               decode={"windows_med": speed}, proof={}, proof_ok="0/0",
               workload={"ctx": [2000, 32000, 128000], "seed": 7, "max_tokens":400, "combine_min_ctx":32000},
               boot_id=f"{name}-{speed}", runtime={})
    row.update(changes)
    return row


class EvidenceTests(unittest.TestCase):
    def test_replays_of_one_boot_do_not_establish_a_noise_floor(self):
        bases = [record(speed=n, boot_id="one-boot") for n in (99, 100, 101)]
        verdict = judge.judge(record("candidate", 120), bases[-1], bases)
        self.assertEqual(verdict["status"], "incomplete")
        self.assertEqual(verdict["floor_n"], 1)

    def test_probe_report_rejects_stale_partial_and_nonfinite_evidence(self):
        contract = {"checks": {"error": {"op": "le", "value": .01}},
                    "proof": ["candidate_lane"], "min_samples": 5}
        probe_report.contract(contract)
        good = dict(schema=1, nonce="fresh", experiment_id="job", binding={"revision": "rev"},
                    samples=5, device="fixture-device", metrics={"error": .001}, proof={"candidate_lane": True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            def judge_report(**changes):
                path.write_text(json.dumps(dict(good, **changes)))
                return probe_report.adjudicate(path, contract, "job", "fresh", {"revision": "rev"})[0]
            self.assertEqual(judge_report(), "succeeded")
            for change in ({"nonce": "old"}, {"binding": {}}, {"samples": 0}, {"samples": True}, {"device": ""}):
                self.assertEqual(judge_report(**change), "incomplete")
            for change in ({"metrics": {}}, {"metrics": {"error": float("nan")}},
                           {"metrics": {"error": True}}, {"proof": {}}, {"proof": {"candidate_lane": 1}}):
                self.assertEqual(judge_report(**change), "failed")

    def test_unknown_zero_of_zero_proof_cannot_be_valid(self):
        cand = record("cand", 120, knobs={"VLLM_TEST": "1"}, proof={"VLLM_TEST": None})
        self.assertEqual(judge.judge(cand, record(), [record(), record(speed=101)])["status"], "invalid")

    def test_missing_and_nonfinite_gates_are_rejected(self):
        for change in ({"quality": {}}, {"korean": {}}, {"decode": {"windows_med": float("nan")}},
                       {"decode": {"windows_med": float("inf")}}, {"quality": {"ok": 0, "total": 0}}):
            with self.subTest(change=change):
                self.assertEqual(judge.judge(record("cand", 120, **change), record(), [record()])["status"], "invalid")

    def test_wrong_context_and_failed_baseline_are_not_reused(self):
        for change in ({"overlay": "c" * 12}, {"workload": {"ctx": [2000]}}, {"runtime": {"image": "different"}},
                       {"quality": {"ok": 8, "total": 9}}, {"thinking": False}):
            with self.subTest(change=change):
                self.assertEqual(judge.judge(record("cand", 120), record(**change), [record()])["status"], "incomplete")

    def test_cross_build_floor_is_only_inconclusive(self):
        bases = [record(), record(speed=101, overlay="c" * 12)]
        verdict = judge.judge(record("cand", 120), bases[0], bases)
        self.assertEqual(verdict["status"], "inconclusive")

    def test_valid_negative_result_and_neutral_are_distinguished(self):
        bases = [record(speed=100), record(speed=101), record(speed=99)]
        verdict = judge.judge(record("cand", 80), bases[0], bases)
        self.assertEqual(verdict["status"], "valid")
        self.assertLess(verdict["delta"], 0)
        self.assertEqual(judge.judge(record("cand", 100), bases[0], bases)["status"], "inconclusive")


class PriorityTests(unittest.TestCase):
    def test_short_unlocking_job_wins_then_aging_protects_long_job(self):
        lines = ["1|long|100|30|long|boot|", "2|short|200|5|short|probe|"]
        self.assertEqual(fleet_priority.rank(lines, {"short": 4}, 300)[0]["session"], "short")
        self.assertEqual(fleet_priority.rank(lines, {"short": 100}, 1900)[0]["session"], "long")
        self.assertEqual(fleet_priority.rank(lines, {"short": 100}, 300, front="long")[0]["session"], "long")
        self.assertEqual(fleet_priority.rank(lines, {}, 1900, front="long", yielded="short")[0]["session"], "short")
        self.assertEqual(fleet_priority.rank(lines, {"short":100}, 300, probes_ready=False)[0]["session"], "long")

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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue"
            holder = root / "holder"
            now = int(time.time())
            lines = f"1|long|{now}|30|long|boot|\n2|short|{now}|2|short|boot|\n"
            queue.write_text(lines)
            holder.write_text("active|123|host|0|30|pair|boot\n")
            setup = '''
H="$FLEET_DIR/holder"; Q="$FLEET_DIR/queue"
holder_alive() { return 0; }
logit() { :; }
_event() { :; }
_dequeue() { :; }
serving_idle() { return 0; }
legacy_busy() { return 1; }
me() { echo host; }
now() { date +%s; }
'''
            env = dict(os.environ, FLEET_DIR=directory, LOGD=directory, REPO=str(ROOT))
            def admit():
                return subprocess.run([BASH, "-c", setup + function + "\n_try_hold short $$ 2 note boot"],
                                      env=env, capture_output=True, text=True)
            self.assertEqual(admit().returncode, 1)
            self.assertEqual(queue.read_text(), lines)
            self.assertTrue(holder.read_text().startswith("active|"))
            holder.unlink()
            result = admit()
            self.assertEqual(result.returncode, 0, result.stderr)
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
        for name in ("pair.sh", "chain.sh", "baseline.py", "judge.py", "experiments.py", "cpu_checks.py",
                     "cpu_evidence.py", "probe_report.py", "experiment_baselines.py", "fleet_priority.py"):
            shutil.copy(ROOT / "bench" / name, self.repo / "bench" / name)
        for script in (self.repo / "bench").glob("*.sh"):
            script.chmod(0o755)
        (self.repo / ".gitignore").write_text("__pycache__/\n")
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
        command = [sys.executable, "-c", f"import time; time.sleep(.4); open({str(output)!r},'a').write('run\\n')"]
        a = self.submit(command=command)
        b = self.submit("agent-b", command=command, hypothesis="another consumer of the same contract")
        self.assertEqual(a["id"], b["id"])
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

    def test_cpu_probe_pair_dependencies_complete_without_manual_promotion(self):
        context = dict(image=self.image, model="fixture", hardware="fixture")
        cpu = self.submit("cpu")
        command = [sys.executable, "-c", "import sys; sys.path.insert(0,'bench'); "
                   "from probe_report import write_report; "
                   "write_report({'mismatches':0},{'lane':True},12,'fake-gpu')"]
        probe = self.submit("probe", kind="probe", depends_on=[cpu["id"]], command=command, context=context,
                            probe_contract={"checks":{"mismatches":{"op":"eq","value":0}},
                                            "proof":["lane"],"min_samples":12})
        pair = self.submit("pair", kind="pair", command=[], knobs={"VLLM_TEST":"1"},
                           context=context, depends_on=[probe["id"]])
        self.assertEqual(self.wait(pair["id"])["state"], "succeeded")
        report = self.wait(probe["id"])["result"]
        self.assertEqual(report["evidence"], "gpu-probe")
        self.assertEqual(report["scope"], "declared numerical contract only")

    def test_probe_exit_zero_without_report_cannot_unlock_candidate(self):
        context = dict(image=self.image, model="fixture", hardware="fixture")
        probe = self.submit("probe", kind="probe", command=[sys.executable, "-c", "pass"], context=context,
                            probe_contract={"checks":{"mismatches":{"op":"eq","value":0}},
                                            "proof":["lane"],"min_samples":1})
        pair = self.submit("pair", kind="pair", command=[], knobs={"VLLM_TEST":"1"},
                           context=context, depends_on=[probe["id"]])
        self.assertEqual(self.wait(pair["id"])["state"], "blocked")
        self.assertFalse((self.logs / "arms").exists())

    def test_two_candidates_share_three_independent_baseline_boots(self):
        context = dict(image=self.image, model="fixture", hardware="fixture")
        first = self.submit("first", kind="pair", command=[], knobs={"VLLM_TEST":"1"}, context=context)
        second = self.submit("second", kind="pair", command=[], knobs={"VLLM_TEST":"2"}, context=context)
        self.assertEqual(self.wait(first["id"])["state"], "succeeded")
        self.assertEqual(self.wait(second["id"])["state"], "succeeded")
        arms = (self.logs / "arms").read_text()
        self.assertEqual(arms.count("-BASE-"), 3, arms)
        self.assertEqual(arms.count("onepass"), 5, arms)
        store = ex.Store(self.jobs)
        baselines = [json.loads(r[0]) for r in store.db.execute("SELECT payload FROM jobs")
                     if json.loads(r[0])["spec"]["kind"] == "baseline"]
        self.assertEqual(len(baselines), 1)

    def test_baseline_failure_blocks_candidates_without_measuring_them(self):
        lever = self.repo / "bench/ab-lever.sh"
        lever.write_text(FAKE_LEVER.replace('if [ "${FAIL_ARM:-}" = "$1" ]; then exit 7; fi',
                                           'case "$1" in *-BASE-*) exit 7;; esac'))
        self.commit()
        (self.overlay / "manifest.tsv").write_text("# source_commit=" + self.sha + "\n")
        self.stamp.write_text(ex.digest(self.overlay / "manifest.tsv"))
        context = dict(image=self.image, model="fixture", hardware="fixture")
        job = self.submit(kind="pair", command=[], knobs={"VLLM_TEST":"1"}, context=context)
        self.assertEqual(self.wait(job["id"])["state"], "blocked")
        self.assertEqual((self.logs / "arms").read_text().count("onepass"), 1)

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

    def test_startup_cache_ignores_unread_docs_and_invalidates_transitive_code(self):
        (self.repo / "tests").mkdir()
        (self.repo / "launchers").mkdir()
        for path in cpu_evidence.STARTUP_AUDIT:
            shutil.copy(ROOT / path, self.repo / path)
        launcher = self.repo / "launchers/start-glm53-nvfp4-tp4.sh"
        launcher.write_text("# launcher fixture\n")
        self.commit()
        spec = ex.normalize(dict(kind="cpu", revision=self.sha, hypothesis="startup", command=[
            sys.executable, "bench/cpu_checks.py", "--suite", "startup"]), self.repo)
        env = {k:v for k,v in self.env.items() if k in ex.BASE_ENV}
        def identity():
            return cpu_evidence.identity(self.repo, spec, env)
        first = identity()
        self.assertEqual(first["scope"], "audited-startup")
        (self.repo / "README.md").write_text("documentation only")
        self.commit()
        self.assertEqual(identity(), first)
        launcher.write_text("# actual launcher changed\n")
        self.commit()
        self.assertNotEqual(identity()["key"], first["key"])
        test = self.repo / "tests/test_glm53_startup.py"
        test.write_text(test.read_text() + "# dependency audit no longer matches\n")
        self.commit()
        self.assertEqual(identity()["scope"], "full-tree")

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

    def test_failed_prerequisite_blocks_gpu_before_admission(self):
        bad = self.submit(command=[sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(self.wait(bad["id"])["state"], "failed")
        gpu = self.submit("gpu-agent", kind="probe", command=["GPU_MARKER"], depends_on=[bad["id"]],
                          context={"image": self.image, "model": "fixture", "hardware": "fixture"})
        result = self.wait(gpu["id"])
        self.assertEqual(result["state"], "blocked", result)
        self.assertNotIn("exp-" + gpu["id"], (self.logs / "admissions").read_text())

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

    def test_pair_publishes_candidate_before_restore_and_has_valid_shared_evidence(self):
        context = dict(image=self.image, model="immutable-model-fixture", hardware="fake-nodes")
        bases = [record(speed=n, git=self.sha, overlay=self.stamp.read_text()[:12], runtime=context) for n in (99,100,101)]
        (self.logs / "onepass.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bases))
        job = self.submit(kind="pair", command=[], knobs={"VLLM_TEST": "1"}, context=context)
        result = self.wait(job["id"])
        self.assertEqual(result["state"], "succeeded", (result, Path(result["log"]).read_text()))
        self.assertEqual(result["result"]["verdict"]["status"], "valid")
        events = self.cli("inbox", "agent-a")["events"]
        kinds = [e["event"] for e in events]
        self.assertLess(kinds.index("arm_result"), kinds.index("succeeded"))
        self.assertEqual((self.logs / "arms").read_text().count("onepass"), 1)

    def test_shared_baseline_automatically_completes_first_pair(self):
        context = dict(image=self.image, model="fixture", hardware="fixture")
        job = self.submit(kind="pair", command=[], knobs={"VLLM_TEST": "1"}, context=context)
        result = self.wait(job["id"])
        self.assertEqual(result["state"], "succeeded", result)
        before = (self.logs / "arms").read_text()
        self.assertEqual(before.count("onepass"), 4)  # three independent defaults, one candidate
        self.assertEqual(result["result"]["verdict"]["floor_n"], 3)
        result = self.cli("result", job["id"])
        self.assertEqual(result["state"], "succeeded", result)
        self.assertEqual((self.logs / "arms").read_text(), before)

    def test_incompatible_old_floor_does_not_suppress_needed_baseline(self):
        context = dict(image=self.image, model="fixture", hardware="fixture")
        bases = [record(speed=n, git=self.sha, overlay=self.stamp.read_text()[:12], runtime={}) for n in (99,100,101)]
        (self.logs / "onepass.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bases))
        job = self.submit(kind="pair", command=[], knobs={"VLLM_TEST": "1"}, context=context)
        result = self.wait(job["id"])
        self.assertEqual(result["state"], "succeeded", result)
        self.assertEqual((self.logs / "arms").read_text().count("-BASE-"), 3)

    def test_invalid_manifest_and_old_prerequisite_are_rejected(self):
        for change in ({"revision":"main"}, {"env":{"SKIP_BOOT":"1"}}, {"env":{"FLEET_REHEARSE":"1"}},
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

    def test_skipped_cpu_math_is_partial_and_does_not_unlock_gpu(self):
        (self.repo / "tests").mkdir()
        (self.repo / "tests/test_logic.py").write_text("print('scale roundtrip: SKIP (no torch)')\n")
        self.commit()
        job = self.submit(command=[sys.executable, "bench/cpu_checks.py", "--suite", "logic"])
        result = self.wait(job["id"])
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["result"]["returncode"], 3)
        report = result["result"]["checks"]
        self.assertFalse(report["coverage_complete"])
        self.assertIn("no torch", report["checks"][0]["skipped"][0])

    def test_chain_stops_after_failed_arm_and_preserves_failure_status(self):
        env = dict(self.env, LEVER=str(self.repo / "bench/ab-lever.sh"), FLEET=str(self.repo / "bench/fleet.sh"),
                   FAIL_ARM="BROKEN", FLEET_SESSION="test")
        p = subprocess.run([BASH, str(self.repo / "bench/chain.sh"), "BROKEN=VLLM_TEST=1", "NEXT=VLLM_TEST=2"],
                           env=env, cwd=self.repo, capture_output=True, text=True, timeout=5)
        self.assertEqual(p.returncode, 7, p.stdout + p.stderr)
        self.assertNotIn("NEXT", (self.logs / "arms").read_text())


if __name__ == "__main__":
    unittest.main()
