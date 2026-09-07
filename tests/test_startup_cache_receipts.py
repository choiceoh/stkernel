"""Exercise the real boot receipt gate without restarting a fleet."""
import contextlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class BootReceiptTests(unittest.TestCase):
    def test_failed_snapshot_does_not_prevent_restore_or_nonzero_exit_receipt(self):
        source = (Path(__file__).resolve().parents[1] / "bench/startup_cache_boots.sh").read_text()
        handler = 'failed() {' + source.split('failed() {', 1)[1].split('\ntrap failed EXIT', 1)[0]
        with tempfile.TemporaryDirectory() as root:
            script = '\n'.join([
                'set -euo pipefail', 'EVIDENCE=$1', 'monitor_pid=', 'PREFIX=TEST',
                'restore_knobs=', 'FLEET_RESTORE_MANAGED=0',
                'snapshot() { return 9; }',
                'bash() { printf "restored\\n" > "$EVIDENCE/restored"; }',
                handler, 'trap failed EXIT', 'false'])
            result = subprocess.run(['bash', '-c', script, 'test', root], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(Path(root, 'exit-code').read_text().strip(), '1')
            self.assertEqual(Path(root, 'restored').read_text().strip(), 'restored')

    def check_receipts(self, stage, fast, fast_hits, legacy_hits, suffix="", mode="pack-io", key_fields="", packs="", campaign_knobs=None):
        script = (Path(__file__).resolve().parents[1] / "bench/startup_cache_boots.sh").read_text()
        gate = script.split('"$stage" "$MODE" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as root:
            if mode == 'campaign':
                Path(root, 'campaign.json').write_text(json.dumps(dict(arms=[dict(stage=stage, knobs=campaign_knobs if campaign_knobs is not None else {
                    'VLLM_GLM53_MK_PACK_FAST_IO':str(fast), 'VLLM_GLM53_MK_PACK_SHA256':'0'})])))
                Path(root, 'TEST-cache-env.json').write_text(json.dumps([
                    'VLLM_GLM53_MK_PACK_FAST_IO='+str(fast), 'VLLM_GLM53_MK_PACK_SHA256=0']))
            for node in (1, 2, 3, 4):
                Path(root, f"TEST-srv{node}.log").write_text(
                    "[rank-cache] hit rank=0\n" + 2 * (
                        "[fp8-cache] enabled=True hit=1 miss=0 errors=0\n"
                        f"[mk-pack-io] model fast={fast} fast_hits={fast_hits} legacy_hits={legacy_hits}{key_fields}\n{packs}"
                    ) + suffix)
            with patch.object(sys, "argv", ["gate", root, "TEST", stage, mode]), \
                    contextlib.redirect_stdout(io.StringIO()):
                exec(compile(gate, "startup_cache_boots receipt gate", "exec"), {})

    def test_prime_accepts_new_packs_but_timed_arms_require_hits(self):
        self.check_receipts("PRIME", 0, 0, 0)
        self.check_receipts("BASE1", 0, 0, 253)
        self.check_receipts("FAST1", 1, 253, 0)
        for stage, fast in (("BASE1", 0), ("FAST1", 1)):
            with self.subTest(stage=stage), self.assertRaisesRegex(AssertionError, "wrong pack IO path"):
                self.check_receipts(stage, fast, 0, 0)

    def test_pack_key_warm_arms_require_all_sha_hits_without_repacking(self):
        for stage, sha in (("BASE1", 0), ("FAST1", 1)):
            fields = f" sha_hits={253 if sha else 0} md5_fallback=0 aliases=0 alias_errors=0"
            args = dict(mode="pack-key", key_fields=fields,
                        packs="packs: rtn=0 gptq=0 gptq_failed=0 cached=253\n")
            self.check_receipts(stage, f"1 sha256={sha}", 253, 0, **args)
            if sha:
                for bad in (fields.replace("sha_hits=253", "sha_hits=252"),
                            fields.replace("md5_fallback=0", "md5_fallback=1"),
                            fields.replace("alias_errors=0", "alias_errors=1")):
                    with self.assertRaises(AssertionError):
                        self.check_receipts(stage, "1 sha256=1", 253, 0,
                                            mode="pack-key", key_fields=bad, packs=args["packs"])
            with self.assertRaisesRegex(AssertionError, "unexpected repack"):
                self.check_receipts(stage, f"1 sha256={sha}", 253, 0,
                                    mode="pack-key", key_fields=fields,
                                    packs=args["packs"].replace("rtn=0", "rtn=1"))

    def test_renderer_warmup_requires_completed_overlap_and_both_reuses(self):
        packs = "packs: rtn=0 gptq=0 gptq_failed=0 cached=254\n"
        suffix = ("[early-mm-warmup] submitted processors=2 before engine startup\n"
                  "[early-mm-warmup] completed processors=2/2 elapsed_s=10.0\n"
                  "[boot-stamp] load-model took 80.0s\n"
                  "[early-mm-warmup] reused Multi-modal join_s=0.000\n"
                  "[early-mm-warmup] reused Readonly multi-modal join_s=0.000\n")
        self.check_receipts("BASE1", 1, 254, 0, mode="renderer-warmup", packs=packs)
        self.check_receipts("FAST1", 1, 254, 0, mode="renderer-warmup", packs=packs, suffix=suffix)
        for bad in (suffix.replace("processors=2/2", "processors=1/2"),
                    suffix.replace("reused Readonly", "failed Readonly"),
                    "[boot-stamp] load-model took 80.0s\n" + suffix):
            with self.assertRaises(AssertionError):
                self.check_receipts("FAST1", 1, 254, 0, mode="renderer-warmup", packs=packs, suffix=bad)

    def test_graph_profile_requires_real_warmup_on_every_arm(self):
        packs = "packs: rtn=0 gptq=0 gptq_failed=0 cached=255\n"
        warmup = "".join(f"[boot-stamp] {p} took 1.0s\n" for p in
                         ("encoder-profile", "profile-run", "cudagraph-capture", "compile+warmup"))
        for stage, marker in (("BASE1", "[boot-stamp] cudagraph-memory-profile took 3.0s\n"),
                              ("FAST1", "[glm53-graph-profile] skipped unused estimate rank=0\n")):
            args = dict(mode="graph-profile", packs=packs)
            self.check_receipts(stage, 1, 255, 0, suffix=warmup + marker, **args)
            for bad in (warmup, marker, warmup.replace("encoder-profile", "skipped-encoder") + marker,
                        warmup + marker + "Traceback (most recent call last)"):
                with self.assertRaises(AssertionError):
                    self.check_receipts(stage, 1, 255, 0, suffix=bad, **args)

    def test_prime_still_rejects_wrong_path_and_restore_errors(self):
        for fast, hits, suffix in ((1, 0, ""), (0, 1, ""),
                                  (0, 0, "pack cache example unreadable\n"),
                                  (0, 0, "MK W4 pack build FAILED\n")):
            with self.subTest(fast=fast, hits=hits, suffix=suffix), self.assertRaises(AssertionError):
                self.check_receipts("PRIME", fast, hits, 0, suffix)

    def test_shared_campaign_retains_per_node_warm_pack_receipts(self):
        self.check_receipts('FASTIOR1', 1, 253, 0, mode='campaign')
        self.check_receipts('BASE1', 0, 0, 253, mode='campaign')
        with self.assertRaisesRegex(AssertionError, 'campaign pack IO mismatch'):
            self.check_receipts('FASTIOR1', 1, 0, 0, mode='campaign')

    def test_unrelated_campaign_does_not_infer_pack_gate_from_profile_defaults(self):
        self.check_receipts('OTHERR1', 1, 0, 0, mode='campaign', campaign_knobs={'VLLM_OTHER':'1'})


if __name__ == "__main__":
    unittest.main()
