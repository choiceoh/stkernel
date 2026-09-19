"""Qwen3.8's NVMe tiers (engine/profiles/qwen38/fleet.py, base/tiered_kv): the served layout is one the tier can move,
its parked bytes are named so that no other layout reads them, and the fleet hands the tiers to the runner and the door.

The model's half -- park, resume, the slot and snapshot bytes -- is base/composed's and adapter.py's, held by
tests/test_engine_composed.py and tests/test_engine_qwen38_draft_chain.py; the tier's is tests/test_engine_tier.py's."""
from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def facts():
    """The served facts of the checkpoint's own config (probes/qwen38_config.json, sha-pinned)."""
    from probes.engine_qwen38_cells import facts as served
    return served()


class QwenTierTests(unittest.TestCase):
    def test_the_served_layout_is_one_the_tier_can_move(self):
        """O_DIRECT moves whole sectors: a block must be a multiple of one (NvmeTier refuses any other) and a staging
        window must hold a block. A slot's and a snapshot's bytes need not be -- the tier pads their last sector on the
        way out and reads back only what the view holds (base/kv_tier)."""
        from engine.base.kv_tier import SECTOR
        from engine.base.tiered_kv import PREFIX_TIER_STAGE
        from engine.profiles.qwen38 import caches
        F = facts()
        p = caches.layout(F, range(F.layers), mtp=True)
        self.assertEqual(p.block_bytes % SECTOR, 0, p.block_bytes)
        self.assertGreaterEqual(PREFIX_TIER_STAGE, p.block_bytes)
        self.assertGreaterEqual(64 << 20, p.block_bytes)            # the conversation tier's staging (NvmeTier's default)

    def test_a_layout_change_makes_the_parked_bytes_foreign(self):
        """A string per layout (caches.state_format): a boot whose rings, blocks or state differ from the ones that
        wrote a conversation never resumes it -- the tier counts it, forgets it first, and replaces it under its key."""
        from engine.profiles.qwen38.caches import layout, snapshot_layout, state_format
        F = facts()
        layers = range(F.layers)
        p, n = layout(F, layers, mtp=True), snapshot_layout(F, layers)[0]
        served = state_format(F, p, n, mtp=True)
        k1 = dataclasses.replace(F, spec_k=1)
        others = {
            "K=1": state_format(k1, layout(k1, layers, mtp=True), n, mtp=True),
            "no MTP head": state_format(F, layout(F, layers, mtp=False), n, mtp=False),
            "GDN state dtype": state_format(dataclasses.replace(F, gdn_state_dtype="bf16"), p, n, mtp=True),
            "rank files": state_format(dataclasses.replace(F, weight_layout=F.weight_layout + "-next"), p, n, mtp=True),
            "snapshot bytes": state_format(F, p, n + 4096, mtp=True),
        }
        for why, other in others.items():
            self.assertNotEqual(other, served, why)
        self.assertTrue(served.startswith("qwen38-") and "glm53" not in served, served)

    def test_the_fleet_parks_through_its_runner_and_its_door(self):
        """What the fleet boot does with the tiers (it needs the GPU to run, so its text is what a CPU checks): the
        runner parks and restores through them, the door releases short turns rather than parking them, and the rank
        directory is claimed by the boot's lease owner."""
        text = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn("tiered=tiered, keep_idle=True, prefix=prefix)", text)
        self.assertIn("runner.prefix_tier = prefix_tier", text)
        self.assertIn("park_min_tokens=PARK_MIN_TOKENS)", text)
        self.assertIn('lease_owner=os.environ.get("ST_LEASE_OWNER") or None', text)
        self.assertIn("state_format=state_format(F, caches.layout, caches.snapshot_bytes_n, mtp=drafter)", text)


if __name__ == "__main__":
    unittest.main()
