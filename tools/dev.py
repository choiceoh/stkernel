#!/usr/bin/env python3
"""Discover, inspect and invoke development tools through one JSON protocol."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile

from dev_catalog import BY_ID, DISCOVERY_DIRS, TOOLS

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
MAX_OUTPUT = 12000


class ToolError(Exception):
    def __init__(self, message, recovery=None):
        super().__init__(message)
        self.recovery = recovery or []


def emit(value):
    print(json.dumps({"schema_version": SCHEMA_VERSION, **value}, ensure_ascii=False, indent=2))


def pins():
    result = {}
    for line in (ROOT / "tools/devenv/versions.env").read_text().splitlines():
        if re.match(r"^[A-Z][A-Z0-9_]*=", line):
            key, value = line.split("=", 1)
            result[key] = " ".join(shlex.split(value, comments=True))
    return result


def python_runtime():
    """No shell startup files, package installations or GPU imports for discovery."""
    explicit = os.environ.get("ST_DEV_PYTHON")
    if explicit:
        candidate = shutil.which(explicit) or str(Path(explicit).expanduser())
        if not Path(candidate).is_file() or not os.access(candidate, os.X_OK):
            raise ToolError("ST_DEV_PYTHON is not executable", ["set ST_DEV_PYTHON to an existing interpreter"])
        return candidate, "ST_DEV_PYTHON"
    candidates = [(ROOT / ".venv/bin/python", "checkout .venv")]
    if platform.system() == "Darwin":
        candidates.append((Path.home() / ".venvs/stkernel/bin/python", "tools/devenv/mac.sh"))
    for path, source in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path), source
    return sys.executable, "current interpreter"


def child_environment():
    python, _ = python_runtime()
    env = os.environ.copy()
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", os.defpath)
    return env


def package_inventory():
    python, source = python_runtime()
    wanted = pins()
    expected = dict(spec.split("==", 1) for spec in shlex.split(wanted["PY_PACKAGES"]))
    expected["torch"] = wanted["TORCH_VERSION"]
    if platform.system() == "Linux":
        expected["torch"] += "+" + wanted["TORCH_INDEX"].rsplit("/", 1)[-1]
        expected["triton"] = wanted["TRITON_VERSION"]
    probe = """import importlib.metadata as m, json, platform, sys
versions = {}
for name in sys.argv[1:]:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': platform.python_version(), 'packages': versions}))
"""
    try:
        result = subprocess.run([python, "-c", probe, *expected], capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise ToolError("selected Python cannot read package metadata", [["./dev", "run", "env.setup"]])
        actual = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ToolError("cannot inspect selected Python: " + type(exc).__name__) from exc
    rows = [{"name": name, "expected": version, "installed": actual["packages"].get(name),
             "status": "ok" if actual["packages"].get(name) == version else
             "missing" if actual["packages"].get(name) is None else "version_mismatch"}
            for name, version in expected.items()]
    return {"executable": python, "selected_by": source, "version": actual["python"],
            "expected_version": wanted["PYTHON_VERSION"],
            "version_matches": actual["python"].startswith(wanted["PYTHON_VERSION"] + "."),
            "packages": rows}


def source_path(tool):
    return next((arg for arg in tool.command if "/" in arg and (ROOT / arg).is_file()), None)


def source_description(path):
    text = path.read_text(errors="replace")
    if path.suffix == ".py":
        try:
            return ast.get_docstring(ast.parse(text)) or ""
        except SyntaxError:
            return ""
    lines = []
    for line in text.splitlines():
        if line.startswith("#!"):
            continue
        if line.startswith("#"):
            lines.append(line.lstrip("# "))
        elif line.strip():
            break
    return "\n".join(lines)


def discover():
    """Read repository entrypoints; never import or execute an unregistered script."""
    managed = {source_path(tool) for tool in TOOLS}
    managed.update(("tools/dev.py", "tools/devenv/mac.sh", "tools/devenv/node.sh", "bench/fleet.sh"))
    rows = []
    for directory in DISCOVERY_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix not in (".py", ".sh"):
                continue
            if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(ROOT).parts):
                continue
            text = path.read_text(errors="replace")
            if path.suffix == ".py" and not re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', text):
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in managed:
                continue
            description = source_description(path)
            rows.append({"id": relative, "summary": next((line for line in description.splitlines() if line.strip()), relative),
                         "managed": False, "execution": "inspect_source",
                         "next": ["./dev", "describe", relative]})
    return rows


def compact(tool):
    return {"id": tool.id, "summary": tool.summary, "effects": tool.effects,
            "execution": tool.execution, "managed": True}


def describe(identifier):
    if identifier in BY_ID:
        tool = BY_ID[identifier]
        result = asdict(tool)
        result.update({"managed": True, "input_schema": {"type": "object", "properties": {
            "arguments": {"type": "array", "items": {"type": "string"},
                          "description": "Original tool arguments, passed literally; examples and source_help below."}},
            "additionalProperties": False},
            "invoke": ["./dev", "run", identifier, "--", "<arguments...>"],
            "examples": [["./dev", "run", identifier, "--", *args] for args in tool.examples]})
        path = source_path(tool)
        if path:
            result["source"] = str(ROOT / path)
            result["source_help"] = source_description(ROOT / path)[:6000]
        if tool.execution == "fleet":
            result["controller"] = os.environ.get("FLEET_CONTROLLER", "srv2")
            result["path_scope"] = "controller; local files are not copied"
            result["source_help"] = source_description(ROOT / "bench/fleet.sh")[:6000]
        return result
    match = next((row for row in discover() if row["id"] == identifier), None)
    if match:
        return {**match, "source": str(ROOT / identifier),
                "source_help": source_description(ROOT / identifier)[:6000],
                "notes": "Read source and its execution contract. Add a Tool adapter in tools/dev_catalog.py for managed execution."}
    raise ToolError("unknown tool: " + identifier, [["./dev", "search", identifier]])


def registry_errors():
    errors = []
    if len(BY_ID) != len(TOOLS):
        errors.append("duplicate tool IDs")
    for tool in TOOLS:
        if not (ROOT / tool.docs).is_file():
            errors.append(tool.id + ": missing documentation " + tool.docs)
        if tool.execution == "local":
            for arg in tool.command:
                if "/" in arg and not (ROOT / arg).is_file():
                    errors.append(tool.id + ": missing target " + arg)
    return errors


def status():
    runtime = package_inventory()
    env = child_environment()
    binaries = sorted({tool.command[0] for tool in TOOLS if tool.execution == "local" and tool.command
                       and tool.command[0] != "{python}"} | {"codex", "claude", "graphify", "mergiraf"})
    installed = []
    for name in binaries:
        path = shutil.which(name, path=env["PATH"])
        managed = name in ({"graphify", "ruff"} if platform.system() == "Darwin" else
                           {"uv", "gh", "node", "mergiraf", "wt", "codex", "claude", "graphify", "ruff"})
        installed.append({"name": name, "path": path, "status": "found" if path else "missing",
                          "management": "tools/devenv" if managed else "host package manager",
                          "recovery": [] if path else [["./dev", "describe", "env.setup"]] if managed else
                          [f"install {name} with the host package manager"]})
    failures = [row for row in runtime["packages"] if row["status"] != "ok"]
    errors = registry_errors()
    healthy = runtime["version_matches"] and not failures and not errors and all(row["path"] for row in installed)
    return {"status": "ok" if healthy else "needs_attention", "python": runtime,
            "tools": installed, "registry_errors": errors,
            "manifest": str(ROOT / "tools/devenv/versions.env"),
            "scope": "local metadata; no imports, GPU allocation, installation, SSH or credential reads",
            "limitations": "An executable path does not prove authentication, a running service or GPU readiness.",
            "recovery": ([["./dev", "run", "env.setup"]] if failures or not runtime["version_matches"] else [])
                        + ([["./dev", "audit"]] if errors else []),
            "inspect_fleet": ["./dev", "run", "fleet.status"]}


def command_for(tool, arguments):
    python, _ = python_runtime()
    env = child_environment()
    if tool.platforms and platform.system() not in tool.platforms:
        raise ToolError("requires platform: " + ", ".join(tool.platforms))
    if tool.id in ("check", "regress") and any(arg == "--gpu" or arg.startswith("--gpu=") for arg in arguments):
        raise ToolError("GPU work uses the existing fleet admission path", [["./dev", "describe", "fleet.run"]])
    if tool.execution == "setup":
        if platform.system() == "Darwin":
            command = ["zsh", str(ROOT / "tools/devenv/mac.sh")]
        elif platform.system() == "Linux":
            command = ["bash", str(ROOT / "tools/devenv/node.sh")]
        else:
            raise ToolError("env.setup supports macOS and Linux (including WSL2)")
    elif tool.execution == "fleet":
        controller = os.environ.get("FLEET_CONTROLLER", "srv2")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", controller):
            raise ToolError("FLEET_CONTROLLER must be an SSH host or user@host")
        configured_repo = os.environ.get("ST_DEV_FLEET_REPO")
        if configured_repo and not Path(configured_repo).is_absolute():
            raise ToolError("ST_DEV_FLEET_REPO must be an absolute path on the controller")
        payload = [*tool.command, *arguments]
        if socket.gethostname().split(".")[0] == controller:
            public = Path.home() / "glm53-logs/fleet.sh"
            if not public.is_file():
                raise ToolError("controller public fleet entrypoint is missing: " + str(public))
            env["REPO"] = configured_repo or str(Path.home() / "stkernel")
            if not (Path(env["REPO"]) / "bench/fleet.sh").is_file():
                raise ToolError("controller source checkout is missing: " + env["REPO"])
            return ["bash", str(public), *payload], env
        # SSH concatenates its remote arguments. Quote payload elements, never interpolate them as shell code.
        repo = shlex.quote(configured_repo) if configured_repo else '"$HOME/stkernel"'
        # The public entrypoint is a copied script, not a symlink. Its dirname cannot identify its helper tree.
        remote = f'cd {repo} && exec env REPO={repo} bash "$HOME/glm53-logs/fleet.sh" ' + shlex.join(payload)
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", controller, remote]
        if not shutil.which("ssh", path=env["PATH"]):
            raise ToolError("missing executable: ssh")
        return command, env
    else:
        command = [python if arg == "{python}" else str(ROOT / arg) if "/" in arg else arg
                   for arg in tool.command]
    if not command or not shutil.which(command[0], path=env["PATH"]):
        raise ToolError("missing executable: " + (command[0] if command else tool.id),
                        [["./dev", "status"], ["./dev", "describe", "env.setup"]])
    for part in command[1:]:
        if part.startswith(str(ROOT) + "/") and not Path(part).is_file():
            raise ToolError("missing tool source: " + part)
    return [*command, *arguments], env


def terminate_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def read_tail(stream, limit):
    size = stream.tell()
    stream.seek(max(0, size - limit))
    return stream.read().decode("utf-8", errors="replace"), size > limit


def execute(tool, arguments, timeout=None, limit=MAX_OUTPUT, dry_run=False):
    if tool.execution == "builtin":
        if arguments:
            raise ToolError("env.status takes no arguments")
        return {"status": "planned" if dry_run else "completed", "tool": tool.id,
                **({} if dry_run else {"result": status()})}, 0
    command, env = command_for(tool, arguments)
    if dry_run:
        return {"status": "planned", "tool": tool.id, "argv": command,
                "cwd": str(ROOT), "effects": tool.effects, "execution": tool.execution}, 0
    if tool.packages:
        inventory = package_inventory()
        missing = [row["name"] for row in inventory["packages"]
                   if row["name"] in tool.packages and row["installed"] is None]
        if missing:
            raise ToolError("missing Python packages: " + ", ".join(missing), [["./dev", "run", "env.setup"]])
    deadline = timeout if timeout is not None else tool.timeout
    state = "completed"
    cwd = (env["REPO"] if tool.execution == "fleet" and
           socket.gethostname().split(".")[0] == os.environ.get("FLEET_CONTROLLER", "srv2") else str(ROOT))
    # Spool instead of keeping an unbounded test/build log in memory. Nothing is persisted after the call.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            state = "timeout"
            terminate_group(process)
        except KeyboardInterrupt:
            state = "cancelled"
            terminate_group(process)
        out, out_cut = read_tail(stdout, limit)
        err, err_cut = read_tail(stderr, limit)
    code = process.returncode
    result = {"tool": tool.id, "status": state, "process_exit_code": code,
              "execution": tool.execution, "cwd": cwd, "stdout": out, "stderr": err,
              "truncated": {"stdout": out_cut, "stderr": err_cut}, "validation": "not_assessed"}
    if state in ("timeout", "cancelled"):
        result["recovery"] = [["./dev", "run", "fleet.show"]] if tool.execution == "fleet" else [
            ["./dev", "describe", tool.id]]
        result["note"] = ("Remote work may still exist; inspect it before retrying. This wrapper does not cancel queue tickets."
                          if tool.execution == "fleet" else "The local process group was terminated; inspect partial outputs before retrying.")
        return result, 124 if state == "timeout" else 130
    if code:
        result["status"] = "failed"
        result["recovery"] = [["./dev", "describe", tool.id], ["./dev", "status"]]
    elif tool.id == "fleet.status" and re.search(r"lease: unreadable|can't open file|Traceback \(most recent call last\)", out + err):
        result["status"] = "incomplete"
        result["observation"] = "Some queue evidence could not be read; do not infer FREE from this response."
        result["recovery"] = [["./dev", "describe", "fleet.status"]]
        return result, 3
    elif tool.id == "check":
        summary = re.search(r"(\d+) tests: \d+ ok, (\d+) failed, (\d+) cannot run(?:, (\d+) skipped)?", out)
        if summary:
            total, failed, cannot, skipped = (int(n or 0) for n in summary.groups())
            result["test_counts"] = {"tests": total, "failed_files": failed, "cannot_run_files": cannot, "skipped": skipped}
            result["validation"] = "passed" if total and not (failed or cannot or skipped) else "incomplete"
            if result["validation"] == "incomplete":
                result["status"] = "incomplete"
                result["recovery"] = [["./dev", "status"], ["./dev", "describe", "feedback"]]
                return result, 3
    if not out_cut:
        try:
            result["data"] = json.loads(out)
            result["stdout"] = None  # Do not charge an agent twice for the same structured output.
        except json.JSONDecodeError:
            pass
    return result, code if code >= 0 else 128 - code


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ToolError(message, [["./dev", "--help"]])


def main(argv=None):
    parser = Parser(description=__doc__)
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("status", help="local environment, installed tools and repair commands")
    sub.add_parser("audit", help="adapter drift and newly discovered scripts")
    search = sub.add_parser("search", help="search managed tools and repository entrypoints")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--offset", type=int, default=0)
    sub.add_parser("list", help="compact managed-tool catalog")
    show = sub.add_parser("describe", help="input contract, effects, source help and examples")
    show.add_argument("tool")
    run = sub.add_parser("run", help="invoke an adapter, with a bounded JSON result")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--timeout", type=float)
    run.add_argument("--max-output", type=int, default=MAX_OUTPUT)
    run.add_argument("tool")
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    try:
        args = parser.parse_args(argv)
        if args.action == "status":
            emit(status())
        elif args.action == "audit":
            errors = registry_errors()
            unmanaged = discover()
            emit({"status": "failed" if errors else "ok", "errors": errors,
                  "managed_tools": len(TOOLS), "unmanaged_entrypoints": len(unmanaged),
                  "next": ["./dev", "search", ""], "registry": str(ROOT / "tools/dev_catalog.py")})
            return 1 if errors else 0
        elif args.action == "list":
            emit({"tools": [compact(tool) for tool in TOOLS]})
        elif args.action == "search":
            if not 1 <= args.limit <= 100 or args.offset < 0:
                raise ToolError("search requires 1 <= limit <= 100 and offset >= 0")
            terms = args.query.casefold().split()
            rows = [compact(tool) for tool in TOOLS] + discover()
            matches = [row for row in rows if all(term in (row["id"] + " " + row["summary"]).casefold() for term in terms)]
            end = args.offset + args.limit
            emit({"query": args.query, "total": len(matches), "items": matches[args.offset:end],
                  "next_offset": end if end < len(matches) else None})
        elif args.action == "describe":
            emit(describe(args.tool))
        elif args.action == "run":
            if args.timeout is not None and not 0 < args.timeout <= 86400:
                raise ToolError("timeout must be between 0 and 86400 seconds")
            if not 256 <= args.max_output <= 1024 * 1024:
                raise ToolError("max-output must be between 256 and 1048576 bytes")
            if args.tool not in BY_ID:
                raise ToolError("tool has no execution adapter: " + args.tool, [["./dev", "describe", args.tool]])
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            result, code = execute(BY_ID[args.tool], arguments, args.timeout, args.max_output, args.dry_run)
            emit(result)
            return code
        else:
            python, selected_by = python_runtime()
            emit({"entrypoint": "./dev", "repo": str(ROOT),
                  "python": {"executable": python, "selected_by": selected_by},
                  "catalog_sha256": hashlib.sha256((ROOT / "tools/dev_catalog.py").read_bytes()).hexdigest(),
                  "managed_tools": len(TOOLS),
                  "commands": {"discover": "./dev search <intent>", "inventory": "./dev status",
                               "contract": "./dev describe <tool>", "execute": "./dev run <tool> -- <arguments>",
                               "preview": "./dev run --dry-run <tool> -- <arguments>", "maintenance": "./dev audit"},
                  "workflow": [
                      {"when": "session starts / dependency failure", "command": ["./dev", "status"]},
                      {"when": "choose relevant checks", "command": ["./dev", "run", "feedback", "--", "--base", "origin/main"]},
                      {"when": "CPU validation", "command": ["./dev", "describe", "check"]},
                      {"when": "GPU validation", "command": ["./dev", "describe", "fleet.submit"]},
                      {"when": "before pushing", "command": ["./dev", "run", "push.check"]}],
                  "evidence": "process completion is not GPU correctness, performance or adoption evidence",
                  "tools": [{"id": tool.id, "summary": tool.summary} for tool in TOOLS]})
        return 0
    except ToolError as exc:
        emit({"status": "blocked", "error": str(exc), "recovery": exc.recovery})
        return 2
    except (OSError, ValueError) as exc:
        emit({"status": "failed", "error": str(exc), "recovery": [["./dev", "audit"]]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
