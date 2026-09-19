"""tools/devenv: the development environment srv1..srv4, ost-97x and the Mac hold, with versions.env as its one source.

Held here: every version the scripts read is pinned in the manifest (and the Python ones exactly), the scripts parse,
and srv4's timer runs main's copy through the installed bootstrap -- never a working tree. What the scripts do on a node
is the node's to show (`bash tools/devenv/sync.sh --verify`); this runs where no node is. Named under the engine
pattern so the pull-request verdict (tools/check.py --pattern 'test_engine_*') runs it.

    python3 -m unittest tests.test_engine_devenv
"""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVENV = ROOT / "tools/devenv"
READ = re.compile(r'\$\{?=?([A-Z][A-Z0-9_]*(?:_VERSION|_INDEX|_PACKAGES|_NPM))\b')


def manifest() -> dict:
    keys = {}
    for line in (DEVENV / "versions.env").read_text(encoding="utf-8").splitlines():
        m = re.match(r'^([A-Z][A-Z0-9_]*)=("?)([^"#]*)\2\s*(?:#.*)?$', line.strip())
        if m:
            keys[m.group(1)] = m.group(3).strip()
    return keys


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

    def test_a_node_s_own_git_settings_are_kept(self):
        """Every global git write goes through gset (set only where unset) or adds the gh credential helper."""
        node = (DEVENV / "node.sh").read_text(encoding="utf-8")
        writes = re.findall(r'git config --global (?!--get)(\S+)', node)
        self.assertTrue(writes)
        self.assertTrue(all(w in ('"$1"', "--add") for w in writes), writes)
