"""tools/devenv: the development environment srv1..srv4, ost-97x and the Mac hold, with versions.env as its one source.

Held here: every version the scripts read is pinned in the manifest (and the Python ones exactly), the scripts parse,
and srv4's timer runs main's copy through the installed bootstrap -- never a working tree. What the scripts do on a node
is the node's to show (`bash tools/devenv/sync.sh --verify`); this runs where no node is. Named under the engine
pattern so the pull-request verdict (tools/check.py --pattern 'test_engine_*') runs it.

    python3 -m unittest tests.test_engine_devenv
"""
import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVENV = ROOT / "tools/devenv"
READ = re.compile(r'\$\{?=?([A-Z][A-Z0-9_]*(?:_VERSION|_INDEX|_PACKAGES|_NPM))\b')
MACHINES = ("AARCH64", "X86_64")


def manifest() -> dict:
    keys = {}
    for line in (DEVENV / "versions.env").read_text(encoding="utf-8").splitlines():
        m = re.match(r'^([A-Z][A-Z0-9_]*)=("?)([^"#]*)\2\s*(?:#.*)?$', line.strip())
        if m:
            keys[m.group(1)] = m.group(3).strip()
    return keys


def releases() -> list:
    return re.search(r'^RELEASES="([^"]+)"', (DEVENV / "node.sh").read_text(encoding="utf-8"), re.M).group(1).split()


def functions(*names) -> str:
    """node.sh's own definitions of these functions (one-line or block), to run apart from the script's side effects."""
    text = (DEVENV / "node.sh").read_text(encoding="utf-8")
    return "\n".join(re.search(rf'^{n}\(\) \{{(?:[^\n]*\}}$|.*?^\}}$)', text, re.M | re.S).group(0) for n in names)


def fake(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


class ManifestTests(unittest.TestCase):
    def test_every_version_a_script_reads_is_pinned(self):
        keys = manifest()
        for script in ("node.sh", "mac.sh"):
            read = set(READ.findall((DEVENV / script).read_text(encoding="utf-8")))
            self.assertTrue(read, script)
            for name in read:
                self.assertTrue(keys.get(name), f"{script} reads {name}, which versions.env does not set")

    def test_the_python_packages_are_exact_pins(self):
        for spec in manifest()["PY_PACKAGES"].split():
            self.assertRegex(spec, r'^[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.+-]*$')

    def test_graphify_is_the_committed_extractor(self):
        readme = (ROOT / "graphify-out/README.md").read_text(encoding="utf-8")
        self.assertIn(f"`graphifyy {manifest()['GRAPHIFY_VERSION']}`", readme)


class ScriptTests(unittest.TestCase):
    def test_the_scripts_parse(self):
        for script in ("node.sh", "sync.sh", "devenv-sync"):
            subprocess.run(["bash", "-n", str(DEVENV / script)], check=True)
        if shutil.which("zsh"):
            subprocess.run(["zsh", "-n", str(DEVENV / "mac.sh")], check=True)

    def test_the_timer_runs_mains_copy_through_the_bootstrap(self):
        service = (DEVENV / "devenv-sync.service").read_text(encoding="utf-8")
        self.assertIn("ExecStart=/home/choiceoh/.local/bin/devenv-sync", service)
        self.assertIn("OnCalendar=", (DEVENV / "devenv-sync.timer").read_text(encoding="utf-8"))
        boot = (DEVENV / "devenv-sync").read_text(encoding="utf-8")
        self.assertIn('git -C "$REPO" archive origin/main tools/devenv', boot)
        sync = (DEVENV / "sync.sh").read_text(encoding="utf-8")
        self.assertIn('install -m 0755 "$DIR/devenv-sync" "$HOME/.local/bin/devenv-sync"', sync)
        self.assertIn('cat "$DIR/versions.env" "$DIR/node.sh"', sync)          # the manifest travels with the script

    def test_each_machine_takes_its_own_releases(self):
        """The GB10 nodes are aarch64, the RTX 5050 PC x86_64: no release URL fixes the architecture, and a file a
        release replaces is set aside, never deleted."""
        node = (DEVENV / "node.sh").read_text(encoding="utf-8")
        self.assertIn("aarch64) TRIPLE=aarch64-unknown-linux GOARCH=arm64 NODEARCH=arm64", node)
        self.assertIn("x86_64) TRIPLE=x86_64-unknown-linux GOARCH=amd64 NODEARCH=x64", node)
        urls = re.findall(r'https://\S+', node)
        self.assertTrue(urls)
        self.assertFalse([u for u in urls if re.search(r'aarch64|x86_64|arm64|amd64|linux-x64', u)], urls)
        self.assertEqual(node.count("keep \"$BIN/"), 3)                    # the binaries, node's links, the agent CLIs
        self.assertIn("ost-97x", (DEVENV / "sync.sh").read_text(encoding="utf-8"))

    def test_torch_counts_as_installed_only_as_the_index_s_build(self):
        """2.12.1 from another index (+cpu, another CUDA) is not the pinned build: the local label is compared too."""
        node = (DEVENV / "node.sh").read_text(encoding="utf-8")
        self.assertIn("TORCH_BUILD=${TORCH_INDEX##*/}", node)
        self.assertIn("torch.__version__ != '$TORCH_VERSION+$TORCH_BUILD'", node)
        self.assertTrue(manifest()["TORCH_INDEX"].endswith("/cu130"))

    def test_a_linked_worktree_is_a_checkout(self):
        self.assertIn('if [ -e "$REPO/.git" ]; then', (DEVENV / "node.sh").read_text(encoding="utf-8"))

    def test_the_mac_venv_is_remade_at_the_pinned_python(self):
        mac = (DEVENV / "mac.sh").read_text(encoding="utf-8")
        self.assertIn('if [ "$have" != "$PYTHON_VERSION" ]; then', mac)
        self.assertIn('--python "$PYTHON_VERSION"', mac)
        # graphify follows its pin on the Mac too (reinstalled when GRAPHIFY_VERSION moves), from a venv of its own
        self.assertIn('if [ "$had" != "$GRAPHIFY_VERSION" ]; then', mac)
        self.assertIn('"graphifyy==$GRAPHIFY_VERSION"', mac)
        self.assertIn('ln -sfn "$GVENV/bin/graphify" "$VENV/bin/graphify"', mac)
        self.assertNotIn('command -v graphify >/dev/null; then', mac)          # "already on PATH" is not "at the pin"

    def test_a_node_s_own_git_settings_are_kept(self):
        """Every global git write goes through gset (set only where unset) or adds the gh credential helper."""
        node = (DEVENV / "node.sh").read_text(encoding="utf-8")
        writes = re.findall(r'git config --global (?!--get)(\S+)', node)
        self.assertTrue(writes)
        self.assertTrue(all(w in ('"$1"', "--add") for w in writes), writes)


class ReleaseTests(unittest.TestCase):
    """A release archive is unpacked only at the SHA-256 versions.env pins for the machine, and every release is pinned
    for both machines; node.sh's own functions run here with curl faked."""

    def test_every_release_is_pinned_for_both_machines(self):
        keys = manifest()
        for release in releases():
            for machine in MACHINES:
                self.assertRegex(keys.get(f"{release}_SHA256_{machine}", ""), r'^[0-9a-f]{64}$', (release, machine))

    def test_the_digests_are_read_off_each_machine_s_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_ = Path(tmp)
            log = bin_ / "urls"
            fake(bin_, "curl", f'url="${{@: -1}}"; echo "$url" >> "{log}"; echo "archive of $url"\n')
            env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}
            env.pop("DEVENV_MANIFEST", None)
            out = subprocess.run(["bash", str(DEVENV / "node.sh"), "--digests"], env=env, capture_output=True,
                                 text=True, check=True).stdout.split()
            urls = log.read_text(encoding="utf-8").split()
        self.assertEqual([line.split("=")[0] for line in out],
                         [f"{r}_SHA256_{m}" for m in MACHINES for r in releases()])
        self.assertEqual(len(urls), len(out))
        for line, url in zip(out, urls):
            name, digest = line.split("=")
            self.assertEqual(digest, hashlib.sha256(f"archive of {url}\n".encode()).hexdigest())
            self.assertRegex(url, r'aarch64|arm64' if name.endswith("_AARCH64") else r'x86_64|amd64|x64', name)

    def fetch(self, pin: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "release.tar.gz"
            data = b"#!/bin/sh\necho gh version 2.95.0\n"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("gh_2.95.0_linux_amd64/bin/gh")
                info.size, info.mode = len(data), 0o755
                tar.addfile(info, io.BytesIO(data))
            pinned = {"right": hashlib.sha256(archive.read_bytes()).hexdigest(), "wrong": "0" * 64, "none": ""}[pin]
            bin_ = tmp / "bin"
            bin_.mkdir()
            fake(bin_, "curl", f'while [ $# -gt 0 ]; do [ "$1" = -o ] && {{ cp "{archive}" "$2"; exit 0; }}; shift; done'
                               '; exit 22\n')
            script = "\n".join([functions("sha256", "url_of", "digest_of", "fetch"), "set -euo pipefail", f"TMP={tmp}",
                                "ARCH=X86_64 TRIPLE=x86_64-unknown-linux GOARCH=amd64 NODEARCH=x64", "GH_VERSION=2.95.0",
                                f"GH_SHA256_X86_64={pinned}", 'dir=$(fetch GH)', 'find "$dir" -type f -name gh'])
            return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                  env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"})

    def test_an_archive_is_unpacked_only_at_its_pinned_digest(self):
        right = self.fetch("right")
        self.assertEqual(right.returncode, 0, right.stderr)
        self.assertTrue(right.stdout.strip().endswith("gh_2.95.0_linux_amd64/bin/gh"), right.stdout)
        wrong = self.fetch("wrong")
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("versions.env pins 0000", wrong.stderr)
        self.assertIn("not installed", wrong.stderr)
        self.assertEqual(wrong.stdout, "")
        none = self.fetch("none")
        self.assertNotEqual(none.returncode, 0)
        self.assertIn("versions.env pins no GH_SHA256_X86_64", none.stderr)

    def test_a_replaced_file_is_set_aside_under_a_name_of_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = "\n".join([functions("version_of", "keep"), "set -euo pipefail", f"BIN={tmp}", "log() { :; }",
                                'for build in one two; do',
                                '  printf "#!/bin/sh\\necho tool 1.2.3 $build\\n" > "$BIN/tool"; chmod +x "$BIN/tool"',
                                '  keep "$BIN/tool"',
                                'done',
                                'ls "$BIN/.pre-devenv"; cat "$BIN"/.pre-devenv/*'])
            out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
        names = [line for line in out.splitlines() if line.startswith("tool-")]
        self.assertEqual(len(names), 2, out)
        self.assertIn("tool-1.2.3", names)
        self.assertIn("echo tool 1.2.3 one", out)
        self.assertIn("echo tool 1.2.3 two", out)

    def test_node_counts_as_installed_only_with_its_three_companions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            node_dir = tmp / "node-sdk/node-v24.18.0-linux-x64"
            (node_dir / "bin").mkdir(parents=True)
            for b in ("node", "npm", "npx", "corepack"):
                fake(node_dir / "bin", b, "echo v24.18.0\n")
            bin_ = tmp / "bin"
            bin_.mkdir()
            script = "\n".join([functions("version_of", "node_linked"), f"BIN={bin_}", f"NODE_DIR={node_dir}",
                                "NODE_VERSION=24.18.0",
                                'for b in node npm npx; do ln -s "$NODE_DIR/bin/$b" "$BIN/$b"; done',
                                "node_linked && echo three",
                                'ln -s "$NODE_DIR/bin/corepack" "$BIN/corepack"',
                                "node_linked && echo four",
                                "NODE_VERSION=24.19.0",
                                "node_linked && echo another-version"])
            out = subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout.split()
        self.assertEqual(out, ["four"])


class SyncTests(unittest.TestCase):
    """A server that does not answer fails the run; only an optional node (ost-97x by default) is skipped quietly. The
    node names here are never a real host's, so no node.sh runs."""

    def sync(self, nodes: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmp:
            bin_ = Path(tmp)
            fake(bin_, "ssh", "exit 255\n")
            env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "DEVENV_NODES": nodes,
                   "DEVENV_OPTIONAL": "sleepy-node", "DEVENV_LOGS": str(bin_ / "logs")}
            return subprocess.run(["bash", str(DEVENV / "sync.sh")], env=env, capture_output=True, text=True)

    def test_only_an_optional_node_may_be_unreachable(self):
        both = self.sync("absent-server sleepy-node")
        self.assertEqual(both.returncode, 1, both.stdout + both.stderr)
        self.assertIn("== absent-server: UNREACHABLE -- not synced", both.stdout)
        self.assertIn("== sleepy-node: unreachable -- skipped (optional)", both.stdout)
        optional = self.sync("sleepy-node")
        self.assertEqual(optional.returncode, 0, optional.stdout + optional.stderr)

    def test_ost_97x_is_the_optional_node(self):
        sync = (DEVENV / "sync.sh").read_text(encoding="utf-8")
        self.assertIn("OPTIONAL=${DEVENV_OPTIONAL-ost-97x}", sync)
        self.assertIn('mkdir -p "$HOME/.local/bin" "$HOME/.config/systemd/user"', sync)
