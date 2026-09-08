"""Exercise the deployment command against real files and real rsync."""
import os
from pathlib import Path
import stat
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'launchers/lib/glm53-overlay-sync.sh'


@unittest.skipUnless(shutil.which("rsync"), "rsync is needed for identical-source publication only")
class OverlaySyncTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.src, self.dst = self.root/'source files', self.root/'mounted overlays'
        self.src.mkdir(); self.dst.mkdir()
        self.cu = self.src/'kernel.cu'
        self.cu.write_text('kernel version A\n')
        self.manifest = self.src/'manifest.tsv'
        self.manifest.write_text('# revision one\nkernel.cu\t/pkg/kernel.cu\tabsent\n')
        self.copy()

    def copy(self, *extra, check=True):
        return subprocess.run(['bash', '-c', '. "$1"; shift; glm53_sync_overlays "$@"',
            'test', str(HELPER), str(self.cu), str(self.manifest), *map(str,extra), str(self.dst)+'/'],
            capture_output=True, text=True, check=check)

    def test_identical_bytes_keep_inode_and_mtime_despite_new_source_time(self):
        target = self.dst/self.cu.name
        os.utime(target, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
        before = target.stat()
        self.copy()
        self.assertEqual(target.read_bytes(), self.cu.read_bytes())
        self.assertEqual(target.stat().st_ino, before.st_ino)
        self.assertEqual(target.stat().st_mtime_ns, before.st_mtime_ns)

    def test_same_size_edit_with_older_source_timestamp_updates_content_and_time(self):
        target = self.dst/self.cu.name
        os.utime(target, (1_600_000_000,1_600_000_000))
        self.cu.write_text('kernel version B\n')
        os.utime(self.cu, (1_500_000_000,1_500_000_000))
        self.copy()
        self.assertEqual(target.read_bytes(), self.cu.read_bytes())
        self.assertGreater(target.stat().st_mtime_ns, 1_600_000_000_000_000_000)

    def test_manifest_revision_changes_without_touching_identical_kernel(self):
        target = self.dst/self.cu.name
        before = target.stat()
        self.manifest.write_text(self.manifest.read_text().replace('one','two'))
        self.copy()
        self.assertEqual((self.dst/self.manifest.name).read_bytes(), self.manifest.read_bytes())
        self.assertEqual(target.stat().st_ino, before.st_ino)
        self.assertEqual(target.stat().st_mtime_ns, before.st_mtime_ns)

    def test_permissions_are_normalized_and_missing_input_fails(self):
        target = self.dst/self.cu.name
        target.chmod(0o600)
        self.copy()
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)
        failed = self.copy(self.src/'missing.cu', check=False)
        self.assertNotEqual(failed.returncode, 0)

    def verify(self):
        return subprocess.run(['bash', '-c', '. "$1"; shift; glm53_verify_overlay_sources "$@"',
            'test', str(HELPER), str(self.dst), str(self.cu), str(self.manifest)], capture_output=True, text=True)

    def test_canonical_sha256_verification_accepts_identical_sources(self):
        self.assertEqual(self.verify().returncode, 0)

    def test_canonical_sha256_verification_rejects_stale_or_missing_destination(self):
        target = self.dst/self.cu.name
        target.write_text('kernel version B\n')
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('canonical source SHA256', result.stderr)
        target.unlink()
        self.assertNotEqual(self.verify().returncode, 0)


if __name__ == '__main__':
    unittest.main()
