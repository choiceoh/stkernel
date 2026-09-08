#!/usr/bin/env python3
"""Passive receipts for an independently scheduled canonical A/B onepass.

Only Docker reads, GET /health, SSH reads and local evidence writes are used.
No workload, reservation, serving lifecycle or fleet controller is changed.
Prepared means before the completed onepass record, not before its requests.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import decode_next_runtime_proof as proof

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import onepass_deploy

HOSTS = ("srv2", "srv1", "srv3", "srv4")
LOG = "/home/choiceoh/glm53-logs/glm53.log"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def atomic(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_bytes(data)
    temporary.replace(path)


def retain(path, data):
    """An earlier receipt is immutable, including after observer restart."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("refusing to overwrite retained evidence: " + path.name)
    else:
        atomic(path, data)


def frozen_manifest(repo):
    revision = onepass_deploy.source_revision(repo)
    modules = onepass_deploy.selected_modules(repo)
    rows = onepass_deploy.selected_manifest(repo, modules)
    build = repo / "build/glm53"
    actual = {}
    for line in (build / "manifest.tsv").read_text().splitlines():
        if line and not line.startswith("#"):
            name, target, base = line.split("\t")
            if name in actual:
                raise ValueError("duplicate composed manifest row")
            actual[name] = (target, base)
    if actual != rows:
        raise ValueError("composed manifest differs from selected committed modules")
    expected = {}
    for name, (target, _base) in rows.items():
        candidates = [repo / "overlay/modules" / module / name for module in modules
                      if (repo / "overlay/modules" / module / name).is_file()]
        if len(candidates) != 1 or candidates[0].resolve() != candidates[0]:
            raise ValueError("ambiguous or redirected source: " + name)
        data = candidates[0].read_bytes()
        if (build / name).read_bytes() != data:
            raise ValueError("stale composed source: " + name)
        expected[target] = digest(data)
    if onepass_deploy.source_revision(repo) != revision:
        raise ValueError("source changed during observer initialization")
    return revision, proof.validate_manifest(expected)


def mode_from_metadata(value, *, sf6_direct=False, sf6_unpack=False):
    if not value or value.get("running") is not True:
        return None
    for mode in ("candidate", "baseline"):
        wanted = proof.expected_knobs(mode, sf6_direct=sf6_direct, sf6_unpack=sf6_unpack)
        keys = proof.TARGET_KNOBS | {proof.SF6_UNPACK_KNOB} if sf6_unpack else proof.TARGET_KNOBS
        if all(value.get("knobs", {}).get(key) == wanted[key] for key in keys):
            return mode
    return None


def read_records(path, names):
    """Ignore an incomplete final write; hash complete original lines, sans CR/LF."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return {}
    records = {}
    for raw in data.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            continue
        raw = raw.rstrip(b"\r\n")
        if not raw.strip():
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError("onepass record must be an object")
        if record.get("name") not in names:
            continue
        name = record["name"]
        if name in records:
            raise ValueError("duplicate canonical onepass record: " + name)
        records[name] = dict(record=record, sha256=digest(raw))
    return records


def snapshot_errors(mode, expected, head_before, head_after, ranks, phase,
                    record_before=None, record_after=None, prepared=None, *, sf6_direct=False, sf6_unpack=False):
    """Pure validation; transition races are errors, never relabeled evidence."""
    errors = []
    if (not head_before or not head_after or not head_before.get("boot_id")
            or head_before != head_after
            or mode_from_metadata(head_before, sf6_direct=sf6_direct, sf6_unpack=sf6_unpack) != mode):
        errors.append("head identity/mode changed or unavailable during snapshot")
    if set(ranks) != set(HOSTS):
        errors.append("exactly four rank snapshots required")
    identities = set()
    for host in HOSTS:
        rank = ranks.get(host, {})
        report = rank.get("report", {})
        errors.extend(host + ": " + error for error in
                      proof.validate_report(report, expected, sf6_direct=sf6_direct, sf6_unpack=sf6_unpack))
        if report.get("mode") != mode:
            errors.append(host + ": wrong mode")
        if rank.get("error"):
            errors.append(host + ": " + rank["error"])
        if not isinstance(report.get("host"), str) or report.get("host") in identities:
            errors.append(host + ": duplicate/missing rank hostname")
        identities.add(str(report.get("host")))
        if host == "srv2" and report.get("boot_id") != (head_before or {}).get("boot_id"):
            errors.append("head proof does not match observed head boot")
        if prepared:
            errors.extend(host + ": " + error for error in
                          proof.compare_snapshots(prepared.get(host), report, sf6_direct=sf6_direct, sf6_unpack=sf6_unpack))
    if phase == "prepared":
        if record_before is not None or record_after is not None:
            errors.append("onepass record appeared before prepared snapshot completed")
    else:
        if not record_before or record_before != record_after:
            errors.append("completed onepass record missing or changed during runtime snapshot")
        elif record_before["record"].get("boot_id") != (head_before or {}).get("boot_id"):
            errors.append("onepass record belongs to a different head boot")
        if not prepared:
            errors.append("no valid prepared snapshot for this arm")
    return errors


def runtime_binding_errors(mode, expected, head, record, prepared, *, sf6_direct=False, sf6_unpack=False):
    """A completed record and valid preparation latch one existing boot.

    This permits passive reads after the hold is released. Current mounted
    sources still have to pass the subsequent four-rank snapshot comparison.
    """
    errors = []
    if not isinstance(prepared, dict) or set(prepared) != set(HOSTS):
        return ["runtime read requires four valid prepared rank reports"]
    for host, report in prepared.items():
        errors.extend(host + " prepared: " + error for error in
                      proof.validate_report(report, expected, sf6_direct=sf6_direct, sf6_unpack=sf6_unpack))
    reference = prepared.get("srv2", {})
    if (not isinstance(head, dict) or mode_from_metadata(head, sf6_direct=sf6_direct, sf6_unpack=sf6_unpack) != mode
            or head.get("image") != proof.IMAGE
            or any(head.get(key) != reference.get(key) for key in ("boot_id", "image", "knobs"))):
        errors.append("runtime head does not match the prepared boot/image/configuration")
    if (not record or record.get("record", {}).get("boot_id") != reference.get("boot_id")
            or not reference.get("boot_id")):
        errors.append("completed arm record does not match the prepared head boot")
    return errors


class Reader:
    """The only system interactions; all commands are reads of existing state."""

    def __init__(self, repo, port, session, fleet_dir, *, sf6_direct=False, sf6_unpack=False):
        proof.expected_knobs("candidate", sf6_direct=sf6_direct, sf6_unpack=sf6_unpack)
        if sf6_unpack and port != 18000:
            raise ValueError("SF6 unpack requires observer port 18000")
        self.repo, self.port = repo, port
        self.session, self.fleet_dir = session, Path(fleet_dir)
        self.sf6_direct = sf6_direct
        self.sf6_unpack = sf6_unpack

    def owned(self):
        try:
            fields = (self.fleet_dir / "holder").read_text().strip().split("|")
            return len(fields) == 7 and fields[0] == self.session and fields[-1] == "boot"
        except OSError:
            return False

    def head(self):
        try:
            raw = subprocess.check_output(["docker", "inspect", "glm53"], text=True,
                                          timeout=5, stderr=subprocess.DEVNULL)
            obj = json.loads(raw)[0]
            env = dict(item.split("=", 1) for item in obj["Config"]["Env"] if "=" in item)
            return dict(boot_id=obj["Id"] + "|" + obj["State"]["StartedAt"],
                        running=obj["State"]["Running"], image=obj["Image"],
                        knobs={key: value for key, value in env.items() if key.startswith("VLLM_")})
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return None

    def healthy(self):
        try:
            with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                return response.status == 200
        except (OSError, URLError):
            return False

    def rank(self, host, mode, expected):
        # Ship the exact local proof by stdin. The appended observer harness
        # retains the exact log it parses and rechecks this rank's container ID.
        source = (self.repo / "probes/decode_next_runtime_proof.py").read_text()
        wrapper = '''
import base64, hashlib, json, subprocess
ns = {"__name__": "passive_runtime_proof"}
exec(compile(SOURCE, "decode_next_runtime_proof.py", "exec"), ns)
report = ns["collect_report"](MODE, EXPECTED, sf6_direct=SF6_DIRECT, sf6_unpack=SF6_UNPACK)
raw = ns["Path"](LOG).read_bytes()
report["log_sha256"] = hashlib.sha256(raw).hexdigest()
report["markers"] = ns["parse_markers"](raw.decode(errors="replace"), sf6_unpack=SF6_UNPACK)
errors = ns["validate_report"](report, EXPECTED, sf6_direct=SF6_DIRECT, sf6_unpack=SF6_UNPACK)
again = json.loads(subprocess.check_output(["docker", "inspect", report["container"]], text=True, timeout=10))[0]
if again["Id"] + "|" + again["State"]["StartedAt"] != report["boot_id"] or not again["State"]["Running"]:
    errors.append("rank changed during snapshot")
report.update(valid=not errors, errors=errors)
print(json.dumps(dict(report=report, log_b64=base64.b64encode(raw).decode(), error="; ".join(errors))))
'''
        script = ("SOURCE=" + repr(source) + "\nMODE=" + repr(mode) + "\nEXPECTED="
                  + repr(expected) + "\nLOG=" + repr(LOG)
                  + "\nSF6_DIRECT=" + repr(self.sf6_direct)
                  + "\nSF6_UNPACK=" + repr(self.sf6_unpack) + "\n" + wrapper)
        command = ["python3", "-"]
        if host != "srv2":
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                       "choiceoh@10.10.10." + host[-1], shlex.join(command)]
        try:
            result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=55)
            if result.returncode:
                return dict(error=f"collector exit {result.returncode}: {result.stderr[-4000:]}",
                            stdout=result.stdout[-4000:])
            return json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return dict(error=str(exc))

    def ranks(self, mode, expected):
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = pool.map(lambda host: self.rank(host, mode, expected), HOSTS)
            return dict(zip(HOSTS, values))


class Observer:
    def __init__(self, out, candidate, baseline, revision, expected, reader,
                 *, clock=time.time, sf6_direct=False, sf6_unpack=False):
        proof.expected_knobs("candidate", sf6_direct=sf6_direct, sf6_unpack=sf6_unpack)
        self.out, self.reader, self.clock = Path(out), reader, clock
        self.sf6_direct = sf6_direct
        self.sf6_unpack = sf6_unpack
        if (getattr(reader, "sf6_direct", sf6_direct) is not sf6_direct
                or getattr(reader, "sf6_unpack", False) is not sf6_unpack):
            raise ValueError("reader and observer SF6 variant differ")
        self.expected = expected
        self.names = {candidate: "candidate", baseline: "baseline"}
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "attempts").mkdir(exist_ok=True)
        retain(self.out / "source.commit", (revision + "\n").encode())
        for name in self.names:
            retain(self.out / f"expected-{name}.json", json_bytes(expected))
        self.state = dict(schema=1, status="RUNNING", source_commit=revision,
                          candidate=candidate, baseline=baseline, sf6_direct=sf6_direct, sf6_unpack=sf6_unpack,
                          errors=[], arms={})
        old = self.out / "observer.json"
        if old.exists():
            value = json.loads(old.read_text())
            if any(value.get(key) != self.state[key] for key in
                   ("schema", "source_commit", "candidate", "baseline")):
                raise ValueError("existing observer belongs to a different source/campaign")
            if (value.get("sf6_direct", False) is not sf6_direct
                    or value.get("sf6_unpack", False) is not sf6_unpack):
                raise ValueError("existing observer belongs to a different SF6 variant")
            value["sf6_direct"] = sf6_direct
            value["sf6_unpack"] = sf6_unpack
            self.state = value
        for name, mode in self.names.items():
            arm = self.state["arms"].setdefault(name, dict(mode=mode, status="WAITING", errors=[]))
            for phase, key in (("prepared", "before_receipt"), ("runtime", "after_receipt")):
                path = self.out / f"observed-{phase}-{name}.json"
                if path.exists():
                    receipt = json.loads(path.read_text())
                    if receipt.get("status") != "PASS":
                        raise ValueError("non-PASS published snapshot")
                    if (receipt.get("sf6_direct", False) is not sf6_direct
                            or receipt.get("sf6_unpack", False) is not sf6_unpack):
                        raise ValueError("retained snapshot belongs to a different SF6 variant")
                    for filename, sha in receipt["artifacts_sha256"].items():
                        if Path(filename).name != filename or digest((self.out / filename).read_bytes()) != sha:
                            raise ValueError("retained snapshot artifact changed")
                    if arm.get("head_boot_id", receipt["head_before"]["boot_id"]) != receipt["head_before"]["boot_id"]:
                        raise ValueError("retained arm boot changed")
                    arm.update({key: path.name, "head_boot_id": receipt["head_before"]["boot_id"]})
                    if phase == "runtime":
                        arm.update(status="PASS", record_sha256=receipt["record_sha256"])
        self.last_attempt = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.save()

    def save(self):
        self.state["updated_at"] = self.clock()
        atomic(self.out / "observer.json", json_bytes(self.state))

    def records(self):
        return read_records(self.out / "records.raw.jsonl", self.names)

    def prepared(self, name):
        if not self.state["arms"][name].get("before_receipt"):
            return None
        return {host: json.loads((self.out / f"prepared-{name}-{host}.json").read_text()) for host in HOSTS}

    def snapshot(self, name, phase):
        mode = self.names[name]
        started = self.clock()
        before_record = self.records().get(name)
        head_before = self.reader.head()
        owned_before = self.reader.owned()
        prepared = self.prepared(name) if phase == "runtime" else None
        binding_errors = (runtime_binding_errors(mode, self.expected, head_before, before_record, prepared,
                                                sf6_direct=self.sf6_direct, sf6_unpack=self.sf6_unpack)
                          if phase == "runtime" else [])
        may_collect = not binding_errors if phase == "runtime" else owned_before
        ranks = self.reader.ranks(mode, self.expected) if may_collect else {}
        head_after = self.reader.head()
        owned_after = self.reader.owned()
        after_record = self.records().get(name)
        errors = snapshot_errors(mode, self.expected, head_before, head_after, ranks, phase,
                                 before_record, after_record, prepared,
                                 sf6_direct=self.sf6_direct, sf6_unpack=self.sf6_unpack)
        if phase == "prepared" and (not owned_before or not owned_after):
            errors.append("requested session did not own the boot hold throughout snapshot")
        if phase == "runtime":
            errors.extend(binding_errors)
            errors.extend(runtime_binding_errors(mode, self.expected, head_after, after_record, prepared,
                                                 sf6_direct=self.sf6_direct, sf6_unpack=self.sf6_unpack))
        arm = self.state["arms"][name]
        if arm.get("head_boot_id") and arm["head_boot_id"] != (head_before or {}).get("boot_id"):
            errors.append("a different boot already owns this arm's prepared proof")
        attempt = self.out / "attempts" / f"{phase}-{name}-{time.time_ns()}"
        attempt.mkdir()
        artifacts = {}
        for host, rank in ranks.items():
            report = rank.get("report", dict(valid=False, collection_error=rank.get("error", "missing report")))
            filename = f"{phase}-{name}-{host}.json"
            data = json_bytes(report)
            (attempt / filename).write_bytes(data)
            artifacts[filename] = digest(data)
            try:
                raw = base64.b64decode(rank["log_b64"], validate=True)
                if digest(raw) != report.get("log_sha256"):
                    raise ValueError("retained log/report hash mismatch")
                filename = f"{phase}-{name}-{host}.log"
                (attempt / filename).write_bytes(raw)
                artifacts[filename] = digest(raw)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(host + ": missing or invalid retained log: " + str(exc))
        receipt = dict(schema=1, status="FAIL" if errors else "PASS", phase=phase,
                       timing="before_record" if phase == "prepared" else "after_record",
                       arm=name, mode=mode, source_commit=self.state["source_commit"],
                       sf6_direct=self.sf6_direct, sf6_unpack=self.sf6_unpack,
                       started_at=started, completed_at=self.clock(), head_before=head_before,
                       head_after=head_after, record_sha256=(before_record or {}).get("sha256"),
                       owned_before=owned_before, owned_after=owned_after,
                       runtime_bound_before=not binding_errors if phase == "runtime" else None,
                       record_present_before=bool(before_record), record_present_after=bool(after_record),
                       artifacts_sha256=artifacts, errors=errors)
        (attempt / "receipt.json").write_bytes(json_bytes(receipt))
        self.last_attempt[name] = self.clock()
        if not errors:
            for filename in artifacts:
                retain(self.out / filename, (attempt / filename).read_bytes())
            receipt_name = f"observed-{phase}-{name}.json"
            retain(self.out / receipt_name, json_bytes(receipt))
            key = "before_receipt" if phase == "prepared" else "after_receipt"
            arm.update({key: receipt_name, "head_boot_id": head_before["boot_id"]})
            if phase == "runtime":
                arm.update(status="PASS", record_sha256=before_record["sha256"])
                for host in HOSTS:
                    retain(self.out / f"boot-{name}-{host}.log", (self.out / f"runtime-{name}-{host}.log").read_bytes())
        else:
            failure = dict(phase=phase, attempt=str(attempt.relative_to(self.out)), errors=errors)
            arm.setdefault("failed_attempts", []).append(failure)
            if phase == "runtime":
                # A record/boot transition is irreversible. Never retry by
                # using the next arm's boot to manufacture an after snapshot.
                arm["status"] = "FAIL"
                arm["errors"].append(failure)
        self.save()

    def copy_canonical_logs(self):
        for name in self.names:
            source = Path(LOG).with_name(f"boot-{name}.log")
            target = self.out / f"boot-{name}.log"
            # Ignore any stale named log left by an earlier invocation. The
            # canonical lever publishes this copy after the completed record.
            ledger = self.out / "records.raw.jsonl"
            if (source.is_file() and not target.exists() and name in self.records()
                    and source.stat().st_mtime_ns >= ledger.stat().st_mtime_ns):
                atomic(target, source.read_bytes())

    def step(self):
        records = self.records()
        self.copy_canonical_logs()
        # Continue polling ledger/metadata while SSH collection is in flight.
        # One all-rank snapshot at a time avoids redundant reads on the fleet.
        head = self.reader.head()
        if self.pending:
            if not self.pending.done():
                return False
            self.pending.result()
            self.pending = None
        for name, entry in records.items():
            arm = self.state["arms"][name]
            if arm.get("record_sha256") and arm["record_sha256"] != entry["sha256"]:
                raise ValueError("retained onepass record changed: " + name)
            if arm["status"] not in ("PASS", "FAIL"):
                self.pending = self.executor.submit(self.snapshot, name, "runtime")
                return False
        if all(arm["status"] in ("PASS", "FAIL") for arm in self.state["arms"].values()):
            errors = []
            if all(arm["status"] == "PASS" for arm in self.state["arms"].values()):
                errors = proof.compare_arms(self.prepared(self.state["baseline"]),
                                            self.prepared(self.state["candidate"]),
                                            sf6_direct=self.sf6_direct, sf6_unpack=self.sf6_unpack)
            self.state["errors"].extend(errors)
            self.state["status"] = "PASS" if not errors and all(
                arm["status"] == "PASS" for arm in self.state["arms"].values()) else "FAIL"
            self.save()
            return True
        mode = mode_from_metadata(head, sf6_direct=self.sf6_direct, sf6_unpack=self.sf6_unpack)
        for name, wanted in self.names.items():
            arm = self.state["arms"][name]
            if (self.reader.owned() and wanted == mode and name not in records and not arm.get("before_receipt")
                    and (wanted == "candidate" or self.state["candidate"] in records)
                    and self.clock() - self.last_attempt.get(name, 0) >= 10 and self.reader.healthy()):
                self.pending = self.executor.submit(self.snapshot, name, "prepared")
                break
        return False


def terminal_failure(directory, session, ticket):
    path = Path(directory) / "pending" / (digest(session.encode()) + ".json")
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if ticket and value.get("ticket") != ticket:
        return "reservation ticket changed while observing"
    if value.get("state") == "cancelled":
        return "reservation terminated: cancelled"
    if value.get("state") == "finished":
        rc, outcome = value.get("returncode"), value.get("outcome")
        # Current records retain returncode; older controllers may also retain
        # outcome. Successful release must not abort an in-flight passive read.
        if ((type(rc) is int and rc == 0 and outcome not in ("failed", "cancelled"))
                or (rc is None and outcome == "succeeded")):
            return None
        return "reservation terminated: " + str(outcome or f"returncode={rc}")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--session", required=True, help="prepare only this session's owned boot; finish reads on the same recorded boot")
    parser.add_argument("--ticket")
    parser.add_argument("--sf6-direct", action="store_true",
                        help="observe only direct SF6 versus raw scales; compact AR and inline RDMA remain off")
    parser.add_argument("--sf6-unpack", action="store_true",
                        help="both arms use SF6; compare actual u8x4=1 versus scalar=0 kernels")
    parser.add_argument("--fleet-dir", type=Path, default=Path("/home/choiceoh/glm53-logs/fleet"))
    args = parser.parse_args(argv)
    if args.sf6_direct and args.sf6_unpack:
        parser.error("SF6 variants are mutually exclusive")
    if args.sf6_unpack and args.port != 18000:
        parser.error("SF6 unpack requires --port 18000")
    if (args.candidate == args.baseline or not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            for name in (args.candidate, args.baseline)) or not 0 < args.timeout <= 7200
            or not 1 <= args.port <= 65535):
        parser.error("distinct literal arm names, timeout in (0,7200], and valid port required")
    session = args.session
    observer = None
    try:
        revision, expected = frozen_manifest(ROOT)
        observer = Observer(args.out, args.candidate, args.baseline, revision, expected,
                            Reader(ROOT, args.port, session, args.fleet_dir,
                                   sf6_direct=args.sf6_direct, sf6_unpack=args.sf6_unpack),
                            sf6_direct=args.sf6_direct, sf6_unpack=args.sf6_unpack)
        print("READY " + json.dumps(dict(out=str(args.out), source_commit=revision, targets=len(expected), session=session,
                                         candidate=args.candidate, baseline=args.baseline, port=args.port,
                                         sf6_direct=args.sf6_direct, sf6_unpack=args.sf6_unpack)), flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if observer.step():
                return int(observer.state["status"] != "PASS")
            if session:
                error = terminal_failure(args.fleet_dir, session, args.ticket)
                if error:
                    raise ValueError(error)
            time.sleep(0.5)
        observer.state["status"] = "TIMEOUT"
        observer.state["errors"].append("passive observer deadline expired")
        observer.save()
        return 1
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        if observer:
            observer.state["status"] = "FAIL"
            observer.state["errors"].append(str(exc) or type(exc).__name__)
            observer.save()
        print("PASSIVE OBSERVER FAIL: " + str(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        if observer:
            # Wait only for outstanding reads. No process other than this
            # observer is signalled, and no evidence can be written after exit.
            observer.executor.shutdown(wait=True, cancel_futures=True)
            observer.save()


if __name__ == "__main__":
    raise SystemExit(main())
